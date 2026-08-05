"""
local_runtime: manage a local llama.cpp-family inference server as a first-class
backend for rexgraph-agent.

The agent talks to any OpenAI-compatible endpoint (``chat_model``). This module makes
a *local* one first-class: it launches ``llama-server`` (or a compatible server -
TurboQuant+, Cortex.cpp) as a managed SUBPROCESS, health-checks it, and registers its
URL so the whole stack - chat, the token perplexity/varentropy metrics, agentic_reading
- runs on the local model. Guarantees logprobs (llama-server returns them), so the
metrics light up.

DECOUPLED BY DESIGN: the engine runs as a subprocess behind the OpenAI-compatible seam,
never vendored/compiled into the wheel - so you can swap llama.cpp <-> TurboQuant+ <->
Cortex.cpp <-> vLLM by config, and the pure-Python-agent / compiled-Cython-core split
(CLAUDE.md) stays intact.
"""
from __future__ import annotations

import atexit
import contextlib
import json
import os
import shutil
import socket
import subprocess
import time

_PROC: subprocess.Popen | None = None
_STATE: dict = {}
_EMBED_PROC: subprocess.Popen | None = None    # dedicated embedding worker (the beehive embedder)
_EMBED_STATE: dict = {}

# Backend-agnostic launch defaults. RexGraph is a GENERAL platform: CUDA, ROCm,
# Vulkan, Metal, and CPU are all first-class; the backend is a build-time choice of the
# llama.cpp binary you point this at, and the launcher adapts (n_gpu_layers is derived
# from detected VRAM unless you set it). None of this is tied to any one machine.
DEFAULTS = {
    "n_gpu_layers": None,     # None -> auto: full offload if the model fits VRAM, else partial
    "ctx_size": 16384,        # room for RexGraph's structural system prompt
    "flash_attn": True,       # modern default; llama.cpp ignores it if the build lacks it
    "host": "127.0.0.1",
    "port_start": 8080,
}


def _fa_value(fa) -> str:
    """Map the flash_attn setting to llama.cpp's `--flash-attn` value. A bool picks on/off;
    the tri-state string 'on'/'off'/'auto' passes through (so callers/API bodies can request
    the build's `auto` mode, which the plain bool config cannot express)."""
    if isinstance(fa, str):
        v = fa.strip().lower()
        if v in ("on", "off", "auto"):
            return v
    return "on" if fa else "off"

# GENERAL, size-tiered catalog (not machine-specific). `recommend(budget_gb)` filters it
# to what fits the detected hardware. Repo/file names drift - pass them explicitly to
# ``pull``; these are guidance + sizing (~Q4). MoE entries note that speed tracks active
# params, so they punch above their memory footprint on any backend.
# The BEEHIVE stack (2026): a queen (main driver, MoE-first for unified memory) + focused worker
# bees + a tiny embedder that powers the swarm's alignment/hallucination signal. `recommend()`
# filters to what fits the detected hardware. Names drift, so pass explicit files to `pull`.
CATALOG = [
    {"name": "nomic-embed-text", "kind": "embed", "active": "137M", "approx_gb": 1,
     "tier": "embed", "role": "embedder",
     "why": "the swarm's alignment/hallucination signal (agent_complex.model_embed_fn) - always run one"},
    {"name": "Phi-4-Mini-3.8B", "kind": "dense", "active": "3.8B", "approx_gb": 3,
     "tier": "small", "role": "worker",
     "why": "fast triage/general worker bee; ~68% MMLU at ~2.5 GB"},
    {"name": "Qwen3.6-8B", "kind": "dense", "active": "8B", "approx_gb": 5,
     "tier": "small", "role": "worker",
     "why": "best quality/speed general worker bee"},
    {"name": "Qwen3-Coder-7B", "kind": "dense", "active": "7B", "approx_gb": 6,
     "tier": "small", "role": "worker",
     "why": "coding worker bee - autocomplete / fill-in-middle"},
    {"name": "Qwen3-Coder-14B", "kind": "dense", "active": "14B", "approx_gb": 9,
     "tier": "mid", "role": "worker",
     "why": "coding specialist - beats its size on HumanEval / SWE-bench"},
    {"name": "Qwen3.6-35B-A3B", "kind": "MoE", "active": "~3B", "approx_gb": 20,
     "tier": "mid", "role": "queen",
     "why": "fast queen - MoE, ~121 t/s; big-model quality at small-model speed"},
    {"name": "Llama-3.3-70B / Qwen2.5-72B", "kind": "dense", "active": "70B", "approx_gb": 42,
     "tier": "large", "role": "queen",
     "why": "max dense-quality queen; 48 GB+ VRAM or big unified memory"},
    {"name": "GPT-OSS-120B", "kind": "MoE", "active": "~5B", "approx_gb": 65,
     "tier": "xl", "role": "queen",
     "why": "frontier MoE queen that FITS ~96 GB unified - ~55 t/s on Strix Halo"},
    {"name": "GLM-4.6", "kind": "MoE", "active": "32B", "approx_gb": 180,
     "tier": "xl", "role": "queen",
     "why": "top math (93.9% AIME) but 355B - needs multi-GPU; won't fit 128 GB at Q4"},
]


def _ram_gb() -> float:
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / (1024 ** 2), 1)
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:  # macOS/BSD
            return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def _nvidia_vram_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5).decode()
        return round(max(float(x) for x in out.split()) / 1024, 1)
    except Exception:
        return 0.0


def detect_hardware() -> dict:
    """Detect available inference backends + memory so recommendations and launch args
    adapt to the ACTUAL machine - CUDA/ROCm/Vulkan/Metal/CPU all first-class. Returns
    {os, backends, gpu, ram_gb, model_budget_gb, recommended_backend}."""
    import platform
    osname = platform.system()
    backends: list[str] = []
    gpu = None
    if shutil.which("nvidia-smi"):
        backends.append("cuda")
        v = _nvidia_vram_gb()
        if v:
            gpu = {"vendor": "nvidia", "vram_gb": v, "unified": False}
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        backends.append("rocm")
        gpu = gpu or {"vendor": "amd", "vram_gb": None, "unified": None}
    if shutil.which("vulkaninfo"):
        backends.append("vulkan")
    if osname == "Darwin":
        backends.append("metal")
        gpu = gpu or {"vendor": "apple", "vram_gb": None, "unified": True}
    backends.append("cpu")
    ram = _ram_gb()
    # Model-fit budget: dedicated VRAM if known; unified GPUs (Apple / AMD iGPU) and the
    # CPU path draw on system RAM (leave headroom).
    vram = (gpu or {}).get("vram_gb")
    if vram:
        budget = vram
    elif (gpu or {}).get("unified") or not gpu:
        budget = round(ram * 0.75, 1)
    else:
        budget = round(ram * 0.75, 1)   # unknown dedicated VRAM -> fall back to RAM
    return {"os": osname, "backends": backends, "gpu": gpu, "ram_gb": ram,
            "model_budget_gb": budget,
            "recommended_backend": backends[0] if backends else "cpu"}


def recommend(budget_gb: float | None = None) -> list[dict]:
    """Catalog entries that fit ``budget_gb`` (VRAM or unified budget), biggest-that-fits
    first. Defaults to the detected machine's budget."""
    if budget_gb is None:
        budget_gb = detect_hardware()["model_budget_gb"] or 8.0
    fit = [c for c in CATALOG if c["approx_gb"] <= budget_gb]
    return sorted(fit or CATALOG[:1], key=lambda c: -c["approx_gb"])


def find_binary(bin_path: str | None = None) -> str | None:
    """Locate a llama.cpp-family OpenAI server binary (llama-server or compatible).
    Order: explicit arg -> LLAMA_SERVER_BIN env -> PATH -> common build locations."""
    if bin_path and os.path.exists(os.path.expanduser(bin_path)):
        return os.path.expanduser(bin_path)
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env and os.path.exists(os.path.expanduser(env)):
        return os.path.expanduser(env)
    for name in ("llama-server", "llama-server.exe", "cortex", "server"):
        p = shutil.which(name)
        if p:
            return p
    for cand in ("~/llama.cpp/build/bin/llama-server", "~/llama.cpp/llama-server",
                 "~/.local/bin/llama-server", "/usr/local/bin/llama-server",
                 "/opt/llama.cpp/llama-server"):
        p = os.path.expanduser(cand)
        if os.path.exists(p):
            return p
    return None


def _auto_ngl(model_path: str) -> int:
    """Pick n_gpu_layers from detected hardware: full offload if the model fits VRAM (or
    unified memory), CPU otherwise - the user can set --ngl for a manual GPU/CPU split.
    Backend-agnostic (CUDA/ROCm/Vulkan/Metal/CPU)."""
    hw = detect_hardware()
    gpu = hw.get("gpu")
    if not gpu:
        return 0                       # CPU-only build/host
    if gpu.get("unified"):
        return 999                     # unified memory (Apple / AMD iGPU) -> full offload
    vram = gpu.get("vram_gb")
    if vram is None:
        return 999                     # dedicated GPU, VRAM unknown (ROCm) -> attempt full
    try:
        model_gb = os.path.getsize(os.path.expanduser(model_path)) / (1024 ** 3)
    except Exception:
        return 999
    if model_gb <= vram * 0.92:
        return 999                     # fits dedicated VRAM -> all on GPU
    # Doesn't fit: offload the fraction of layers that DO fit, if we can read the layer
    # count from the file (partial offload beats dumping everything on the CPU).
    try:
        from agent import model_io
        n_layers = model_io.model_summary(model_path).get("n_layers")
        if n_layers:
            usable = max(0.0, vram * 0.9 - 1.0)     # reserve ~1 GB for KV cache + compute
            frac = max(0.0, min(1.0, usable / model_gb))
            return max(0, int(n_layers * frac))
    except Exception:
        pass
    return 0                            # unknown layer count -> CPU (user can set --ngl)


def _server_log_path(port: int) -> str:
    """Where a managed server's stdout+stderr is captured, so a failed launch is diagnosable
    instead of a black-box 'exit 1'."""
    d = os.path.join(os.environ.get("REXGRAPH_CONFIG_DIR",
                                    os.path.expanduser("~/.config/rexgraph")), "logs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"llama-server-{port}.log")


def _tail(path: str, n: int = 30) -> str:
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:]).rstrip() or "(server produced no output)"
    except Exception:
        return "(no log captured)"


def _free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def _wait_health(url: str, timeout: float) -> bool:
    try:
        import httpx
    except Exception:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ("/health", "/v1/models"):
            try:
                r = httpx.get(url + path, timeout=2.0)
                if r.status_code < 500:
                    return True
            except Exception:
                pass
        time.sleep(1.0)
    return False


def start(model_path: str, *, port: int | None = None, host: str | None = None,
          n_gpu_layers: int | None = None, ctx_size: int | None = None,
          flash_attn: bool | None = None, extra_args: list[str] | None = None,
          bin_path: str | None = None, wait: float = 90.0,
          register: bool = True) -> dict:
    """Launch a local llama.cpp server for ``model_path`` and register it as the chat
    backend so chat + metrics + agentic run on it. Stops any server this module
    previously started. Returns the runtime state; raises with actionable guidance if
    the binary or model is missing or the server won't come up."""
    global _PROC
    binary = find_binary(bin_path)
    if not binary:
        raise RuntimeError(
            "No llama.cpp server binary found. Build llama.cpp for your GPU backend and "
            "put `build/bin/llama-server` on PATH (or set LLAMA_SERVER_BIN): "
            "CUDA `-DGGML_CUDA=ON`, ROCm `-DGGML_HIP=ON`, Vulkan `-DGGML_VULKAN=ON`, "
            "Metal `-DGGML_METAL=ON`, or CPU (no flag). "
            "Repo: https://github.com/ggml-org/llama.cpp - TurboQuant+ KV compression: "
            "https://github.com/TheTom/llama-cpp-turboquant - or use Cortex.cpp / LM Studio.")
    mp = os.path.expanduser(model_path)
    if not os.path.exists(mp):
        raise RuntimeError(f"GGUF model file not found: {model_path}")
    stop()
    host = host or DEFAULTS["host"]
    port = port or _free_port(DEFAULTS["port_start"])
    ngl = _auto_ngl(mp) if n_gpu_layers is None else n_gpu_layers
    ctx = DEFAULTS["ctx_size"] if ctx_size is None else ctx_size
    fa = DEFAULTS["flash_attn"] if flash_attn is None else flash_attn
    args = [binary, "-m", mp, "--host", host, "--port", str(port),
            "-ngl", str(ngl), "-c", str(ctx), "--jinja"]
    # Current llama.cpp takes `--flash-attn on|off|auto`; a BARE flag is rejected
    # ("unknown value for --flash-attn"), which used to abort every spawn on new builds.
    args.extend(["--flash-attn", _fa_value(fa)])
    if extra_args:
        args.extend(extra_args)
    # A locally-built llama.cpp keeps its ggml shared libs next to the binary; make the server
    # find them without the caller having to set LD_LIBRARY_PATH.
    env = dict(os.environ)
    bindir = os.path.dirname(os.path.abspath(binary))
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = bindir + (os.pathsep + prev if prev else "")
    logpath = _server_log_path(port)
    _logf = open(logpath, "w")
    _PROC = subprocess.Popen(args, stdout=_logf, stderr=subprocess.STDOUT, env=env)
    _logf.close()                          # child keeps its own dup of the fd
    url = f"http://{host}:{port}"
    if not _wait_health(url, wait):
        code = _PROC.poll()
        stop()
        base = (f"llama.cpp server did not become ready within {wait:.0f}s"
                + (f" (exited with code {code})" if code is not None else
                   " (still starting - a large model can take longer; raise `wait`)"))
        raise RuntimeError(base + f"\n--- llama-server log tail ({logpath}) ---\n" + _tail(logpath)
                           + "\nTip: run the server by hand to see the full error:\n  "
                           + f"{binary} -m {mp} --port {port} -ngl {ngl}")
    _STATE.clear()
    _STATE.update({"url": url, "model": os.path.basename(mp), "model_path": mp,
                   "pid": _PROC.pid, "binary": binary, "ctx_size": ctx,
                   "n_gpu_layers": ngl, "flash_attn": fa})
    try:                               # real arch/params/dim/quant (no weight load)
        from agent import model_io
        _STATE["model_summary"] = model_io.model_summary(mp)
    except Exception:
        pass
    if register:
        try:
            from agent import chat_model
            chat_model.configure(url=url, model="")   # becomes the resolved chat backend
        except Exception:
            pass
    atexit.register(stop)
    return dict(_STATE)


def stop() -> None:
    """Stop the managed server (if any) and clear the chat-backend override."""
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        _PROC.terminate()
        try:
            _PROC.wait(timeout=10)
        except Exception:
            with contextlib.suppress(Exception):
                _PROC.kill()
    _PROC = None
    if _STATE.get("url"):
        try:
            from agent import chat_model
            chat_model.configure(url="")   # clear the override
        except Exception:
            pass
    _STATE.clear()


def _launch(args, wait: float, binary: str, port: int = 0):
    """Popen a llama-server with the ggml shared libs on LD_LIBRARY_PATH, capturing its output to a
    log so a failed launch is diagnosable. Returns (proc, logpath)."""
    env = dict(os.environ)
    bindir = os.path.dirname(os.path.abspath(binary))
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = bindir + (os.pathsep + prev if prev else "")
    logpath = _server_log_path(port)
    logf = open(logpath, "w")
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, env=env)
    logf.close()
    return proc, logpath


def spawn_server(model_path: str, *, port: int | None = None, host: str | None = None,
                 n_gpu_layers: int | None = None, ctx_size: int | None = None,
                 flash_attn: bool | None = None, embeddings: bool = False,
                 extra_args: list[str] | None = None, bin_path: str | None = None,
                 wait: float = 90.0):
    """Launch an INDEPENDENT llama-server and return ``(Popen, state)`` WITHOUT touching the
    module singletons or the global chat-backend registration. The primitive the hive uses for
    its worker bees - the CALLER owns the process lifecycle. Ports default into a worker range
    so bees don't collide with the managed chat (`start`) or embedder (`start_embedder`)."""
    binary = find_binary(bin_path)
    if not binary:
        raise RuntimeError("no llama.cpp binary found (see start() for build guidance).")
    mp = os.path.expanduser(model_path)
    if not os.path.exists(mp):
        raise RuntimeError(f"GGUF model file not found: {model_path}")
    host = host or DEFAULTS["host"]
    port = port or _free_port(DEFAULTS["port_start"] + 2000)
    ngl = _auto_ngl(mp) if n_gpu_layers is None else n_gpu_layers
    ctx = DEFAULTS["ctx_size"] if ctx_size is None else ctx_size
    args = [binary, "-m", mp, "--host", host, "--port", str(port), "-ngl", str(ngl),
            "-c", str(ctx), "--jinja"]
    if embeddings:
        args.append("--embeddings")
    else:
        fa = DEFAULTS["flash_attn"] if flash_attn is None else flash_attn
        args.extend(["--flash-attn", _fa_value(fa)])   # valued flag; bare form is rejected
    if extra_args:
        args.extend(extra_args)
    proc, logpath = _launch(args, wait, binary, port)
    url = f"http://{host}:{port}"
    if not _wait_health(url, wait):
        code = proc.poll()
        with contextlib.suppress(Exception):
            proc.terminate()
        raise RuntimeError("worker server did not become ready in {:.0f}s{}\n--- log tail ({}) ---\n{}".format(wait, f" (exit {code})" if code is not None else "", logpath, _tail(logpath)))
    state = {"url": url, "model": os.path.basename(mp), "model_path": mp, "pid": proc.pid,
             "binary": binary, "ctx_size": ctx, "n_gpu_layers": ngl, "embeddings": bool(embeddings)}
    try:
        from agent import model_io
        state["model_summary"] = model_io.model_summary(mp)
    except Exception:
        pass
    return proc, state


def start_embedder(model_path: str, *, port: int | None = None, host: str | None = None,
                   wait: float = 90.0, bin_path: str | None = None) -> dict:
    """Launch a DEDICATED embedding worker (`llama-server --embeddings`) - the beehive's
    nomic-embed-text bee. It runs ALONGSIDE the chat model so the swarm's semantic
    alignment/hallucination signal (agent_complex.model_embed_fn) is always live, independent of
    which queen/model is chatting. Registers its URL as the embedding endpoint (`embed_url`)."""
    global _EMBED_PROC
    binary = find_binary(bin_path)
    if not binary:
        raise RuntimeError("no llama.cpp binary found (see start() for build guidance).")
    mp = os.path.expanduser(model_path)
    if not os.path.exists(mp):
        raise RuntimeError(f"embedding model not found: {model_path}")
    stop_embedder()
    host = host or DEFAULTS["host"]
    port = port or _free_port(DEFAULTS["port_start"] + 1000)
    args = [binary, "-m", mp, "--host", host, "--port", str(port), "--embeddings",
            "-ngl", str(_auto_ngl(mp))]
    _EMBED_PROC, logpath = _launch(args, wait, binary, port)
    url = f"http://{host}:{port}"
    if not _wait_health(url, wait):
        code = _EMBED_PROC.poll(); stop_embedder()
        raise RuntimeError("embedding worker did not become ready in {:.0f}s{}\n--- log tail ({}) ---\n{}".format(wait, f" (exit {code})" if code is not None else "", logpath, _tail(logpath)))
    _EMBED_STATE.clear()
    _EMBED_STATE.update({"url": url, "model": os.path.basename(mp), "model_path": mp,
                         "pid": _EMBED_PROC.pid, "binary": binary})
    atexit.register(stop_embedder)
    return dict(_EMBED_STATE)


def stop_embedder() -> None:
    global _EMBED_PROC
    if _EMBED_PROC is not None and _EMBED_PROC.poll() is None:
        _EMBED_PROC.terminate()
        try:
            _EMBED_PROC.wait(timeout=10)
        except Exception:
            with contextlib.suppress(Exception):
                _EMBED_PROC.kill()
    _EMBED_PROC = None
    _EMBED_STATE.clear()


def embed_url() -> str | None:
    """The endpoint to use for embeddings: the dedicated embedding worker if running, else the
    main chat model (if it serves embeddings), else None."""
    if _EMBED_STATE.get("url"):
        return _EMBED_STATE["url"]
    return _STATE.get("url") or None


def embed_status() -> dict:
    running = _EMBED_PROC is not None and _EMBED_PROC.poll() is None
    return {"running": bool(running), "url": embed_url(), **_EMBED_STATE}


def status() -> dict:
    """Runtime status: whether a managed server is running, whether a llama.cpp binary is
    installed, the detected hardware (backends/VRAM/RAM), and the model recommendations
    that fit THIS machine - so the UI/CLI adapt to any host (CUDA/ROCm/Vulkan/Metal/CPU),
    not one laptop."""
    running = _PROC is not None and _PROC.poll() is None
    hw = detect_hardware()
    try:
        detected = discover_local_models()
    except Exception:
        detected = []
    return {"running": bool(running), "binary_found": bool(find_binary()),
            "hardware": hw, "recommended": recommend(hw.get("model_budget_gb")),
            "detected": detected, "embedder": embed_status(), **_STATE}


def pull(repo: str, filename: str, dest_dir: str | None = None) -> str:
    """Download a GGUF from Hugging Face (needs huggingface_hub). Returns the local
    path. Large files - check the catalog `approx_gb` first. Repo/file names drift, so
    they are explicit here rather than hardcoded."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "Downloading needs huggingface_hub (`pip install huggingface_hub`), or "
            "download the .gguf manually and pass its path to start().") from e
    dest = os.path.expanduser(dest_dir or os.environ.get(
        "REXGRAPH_MODELS_DIR", "~/.cache/rexgraph/models"))
    os.makedirs(dest, exist_ok=True)
    return hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)


def _default_scan_dirs() -> list[str]:
    """Where local models actually land, across the common toolchains. Extend with
    REXGRAPH_MODEL_DIRS (os.pathsep-separated) for non-standard locations."""
    home = os.path.expanduser("~")
    dirs = [
        os.path.join(home, ".cache", "huggingface", "hub"),      # HF (transformers, vLLM source, hf gguf)
        os.path.join(home, ".cache", "rexgraph", "models"),      # our own pull() dest
        os.environ.get("REXGRAPH_MODELS_DIR", ""),
        os.path.join(home, ".ollama", "models"),                 # ollama blobs
        os.path.join(home, ".lmstudio", "models"),               # LM Studio
        os.path.join(home, "models"),
        os.path.join(home, "gguf"),
        os.path.join(home, "llama.cpp", "models"),
        "/models", "/opt/models",
    ]
    dirs += [d for d in os.environ.get("REXGRAPH_MODEL_DIRS", "").split(os.pathsep) if d]
    seen, out = set(), []
    for d in dirs:
        d = os.path.expanduser(d.strip())
        if d and d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def discover_local_models(extra_dirs: list[str] | None = None, max_files: int = 400) -> list[dict]:
    """AUTO-DETECT models already on disk: no curated registry, no manual paths. Walks the
    common model locations (HF hub cache, ollama, LM Studio, ~/models, our pull() dir, plus
    REXGRAPH_MODEL_DIRS) and reports every GGUF file (llama.cpp-loadable, ready for start()),
    every ollama model (resolved via its manifest, since ollama's blobs are extension-less and
    content-addressed - see `_source`), and every HF transformers snapshot
    (vLLM/transformers-loadable). Each entry carries a `source` (hf-cache/ollama/lmstudio/
    rexgraph/dir), a `loadable` hint (gguf -> start() here; transformers -> serve via
    vLLM/transformers; anything else ollama can hold, e.g. an MLX model, llama.cpp cannot load -
    reported but not loadable), and a size. De-duped by real path."""
    roots = list(_default_scan_dirs())
    for d in (extra_dirs or []):
        d = os.path.expanduser(d)
        if os.path.isdir(d) and d not in roots:
            roots.append(d)

    def _source(path: str) -> str:
        p = path.lower()
        if ".ollama" in p:
            return "ollama"
        if ".lmstudio" in p:
            return "lmstudio"
        if os.sep + "hub" in p and "huggingface" in p:
            return "hf-cache"
        if "rexgraph" in p:
            return "rexgraph"
        return "dir"

    found: dict[str, dict] = {}
    n = 0
    # 1) GGUF files anywhere under the roots (llama.cpp / this runtime can load these directly).
    for root in roots:
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                if n >= max_files:
                    break
                low = fn.lower()
                if low.endswith(".gguf") and not low.startswith("ggml-vocab-"):  # skip llama.cpp test vocabs
                    fp = os.path.join(dp, fn)
                    try:
                        rp = os.path.realpath(fp)
                        if rp in found:
                            continue
                        sz = os.path.getsize(fp) / 1e9
                    except OSError:
                        continue
                    # skip mid-split shards past the first so one model = one entry
                    if ("-00002-of-" in low or "-00003-of-" in low or
                            (("of-" in low) and ("00001-of-" not in low) and low[low.find("of-") - 6:low.find("of-")].strip("-").isdigit())):
                        continue
                    found[rp] = {"name": os.path.splitext(fn)[0], "path": fp,
                                 "size_gb": round(sz, 2), "format": "gguf",
                                 "loadable": "llama.cpp", "source": _source(fp)}
                    n += 1
    # 2) Ollama models: stored as content-addressed, EXTENSION-LESS blobs under blobs/, named
    # only by sha256 digest - so the real name has to come from the manifest at
    # manifests/<registry>/<namespace>/<name>/<tag>, which we parse to find the model-weight
    # layer's digest and resolve it to a blob. Ollama can hold non-GGUF models too (e.g. MLX),
    # which llama.cpp cannot load - sniff the blob's magic bytes rather than trust the tag, so
    # `format`/`loadable` stay honest for plan_hive's `format == "gguf"` gate.
    for root in roots:
        manifests_dir = os.path.join(root, "manifests")
        blobs_dir = os.path.join(root, "blobs")
        if not (os.path.isdir(manifests_dir) and os.path.isdir(blobs_dir)):
            continue
        for dp, _dn, fns in os.walk(manifests_dir):
            for fn in fns:
                if n >= max_files:
                    break
                manifest_fp = os.path.join(dp, fn)
                try:
                    with open(manifest_fp, encoding="utf-8") as f:
                        manifest = json.load(f)
                except (OSError, ValueError):
                    continue
                layers = manifest.get("layers", [])
                parts = os.path.relpath(manifest_fp, manifests_dir).split(os.sep)
                tag = parts[-1]
                model = parts[-2] if len(parts) >= 2 else tag
                namespace = parts[-3] if len(parts) >= 3 else "library"
                name = f"{model}:{tag}" if namespace == "library" else f"{namespace}/{model}:{tag}"

                layer = next((ly for ly in layers
                              if str(ly.get("mediaType", "")).endswith(".model")), None)
                if layer is not None:
                    # Classic shape: one blob IS the whole model - sniff it for GGUF's magic
                    # bytes so `format`/`loadable` are honest rather than assumed from the tag.
                    digest = layer.get("digest", "")
                    if not digest.startswith("sha256:"):
                        continue
                    blob_fp = os.path.join(blobs_dir, digest.replace(":", "-", 1))
                    try:
                        rp = os.path.realpath(blob_fp)
                        if rp in found:
                            continue
                        sz = os.path.getsize(blob_fp) / 1e9
                        with open(blob_fp, "rb") as f:
                            is_gguf = f.read(4) == b"GGUF"
                    except OSError:
                        continue
                    found[rp] = {"name": name, "path": blob_fp, "size_gb": round(sz, 2),
                                 "format": "gguf" if is_gguf else "unknown",
                                 "loadable": "llama.cpp" if is_gguf else "unsupported",
                                 "source": "ollama"}
                    n += 1
                    continue

                # Newer shape (e.g. MLX-format models pulled through ollama): no single
                # "*.model" layer - the weights are split across many per-tensor blobs, so
                # there is no one file to hand llama-server. Never gguf/llama.cpp: nothing here
                # is a spawnable single blob regardless of what the tensors are encoded as.
                tensor_layers = [ly for ly in layers
                                  if str(ly.get("mediaType", "")).endswith(".tensor")]
                if not tensor_layers:
                    continue
                rp = os.path.realpath(manifest_fp)
                if rp in found:
                    continue
                sz = 0.0
                for ly in tensor_layers:
                    digest = ly.get("digest", "")
                    if not digest.startswith("sha256:"):
                        continue
                    blob_fp = os.path.join(blobs_dir, digest.replace(":", "-", 1))
                    with contextlib.suppress(OSError):
                        sz += os.path.getsize(blob_fp)
                found[rp] = {"name": name, "path": manifest_fp, "size_gb": round(sz / 1e9, 2),
                             "format": "unknown", "loadable": "unsupported", "source": "ollama"}
                n += 1
    # 3) HF transformers snapshots (models--org--name/snapshots/<hash>) - vLLM/transformers.
    for root in roots:
        if not (os.sep + "hub" in root and "huggingface" in root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for name in entries:
            if not name.startswith("models--"):
                continue
            snaps = os.path.join(root, name, "snapshots")
            if not os.path.isdir(snaps):
                continue
            try:
                revs = [os.path.join(snaps, r) for r in os.listdir(snaps)]
                revs = [r for r in revs if os.path.isdir(r)]
            except OSError:
                continue
            if not revs:
                continue
            rev = revs[0]
            # gguf-only repos already surfaced above; report a repo as transformers only if it
            # has config.json (i.e. a real HF model dir), so we don't double-count.
            if not os.path.exists(os.path.join(rev, "config.json")):
                continue
            model_id = name[len("models--"):].replace("--", "/")
            key = os.path.realpath(rev)
            if key in found:
                continue
            sz = 0.0
            try:
                for f in os.listdir(rev):
                    fp = os.path.join(rev, f)
                    if os.path.islink(fp) or os.path.isfile(fp):
                        with contextlib.suppress(OSError):
                            sz += os.path.getsize(os.path.realpath(fp))
            except OSError:
                pass
            found[key] = {"name": model_id, "path": rev, "size_gb": round(sz / 1e9, 2),
                          "format": "transformers", "loadable": "vllm/transformers",
                          "source": "hf-cache"}
    out = sorted(found.values(), key=lambda m: (m["format"] != "gguf", -m["size_gb"]))
    return out


def _default_probe_targets() -> list[dict]:
    """Well-known local inference servers. Extend with REXGRAPH_PROBE_URLS (os.pathsep or
    comma separated base URLs) for non-standard ports/hosts."""
    t = [
        {"url": "http://127.0.0.1:11434", "kind": "ollama"},     # Ollama
        {"url": "http://127.0.0.1:8080", "kind": "openai"},      # llama.cpp default
        {"url": "http://127.0.0.1:8000", "kind": "openai"},      # vLLM default
        {"url": "http://127.0.0.1:1234", "kind": "openai"},      # LM Studio
        {"url": "http://127.0.0.1:5000", "kind": "openai"},      # text-generation-webui
        {"url": "http://127.0.0.1:8081", "kind": "openai"},
    ]
    raw = os.environ.get("REXGRAPH_PROBE_URLS", "")
    for u in raw.replace(",", " ").split():        # comma/whitespace - NOT os.pathsep (URLs contain ':')
        u = u.strip().rstrip("/")
        if u:
            t.append({"url": u, "kind": "ollama" if "11434" in u else "openai"})
    # our own managed servers (chat + embedder), so the UI shows them as live too
    for st, label in ((_STATE, "rexgraph-chat"), (_EMBED_STATE, "rexgraph-embed")):
        u = (st or {}).get("url")
        if u:
            t.append({"url": u.rstrip("/"), "kind": "openai", "managed": label})
    seen, out = set(), []
    for e in t:
        if e["url"] not in seen:
            seen.add(e["url"]); out.append(e)
    return out


def probe_endpoints(timeout: float = 0.4) -> list[dict]:
    """PROBE live inference servers already running on this host, not files on disk, actual
    serving endpoints. Hits Ollama's /api/tags and the OpenAI-compatible /v1/models on the
    well-known ports (llama.cpp, vLLM, LM Studio, TGI) + REXGRAPH_PROBE_URLS. Returns only the
    reachable ones, each with the model ids it is serving, so the swarm can wire real backends."""
    try:
        import httpx
    except Exception:
        return []
    live = []
    for tgt in _default_probe_targets():
        url, kind = tgt["url"], tgt.get("kind", "openai")
        path = "/api/tags" if kind == "ollama" else "/v1/models"
        try:
            r = httpx.get(url + path, timeout=timeout)
            if r.status_code >= 500:
                continue
            data = r.json()
        except Exception:
            continue
        if kind == "ollama":
            models = [m.get("name") for m in (data.get("models") or []) if m.get("name")]
        else:
            models = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        live.append({"url": url, "kind": kind, "reachable": True,
                     "models": models, "n_models": len(models),
                     **({"managed": tgt["managed"]} if "managed" in tgt else {})})
    return live


def main(argv=None):
    """CLI: `python -m agent.local_runtime <start MODEL.gguf | status | stop | catalog | discover | endpoints>`."""
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="rexgraph-local", description=(
        "Manage a local llama.cpp model as the rexgraph-agent chat backend."))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="launch a GGUF and register it as the chat backend")
    s.add_argument("model_path"); s.add_argument("--ctx", type=int, default=None)
    s.add_argument("--ngl", type=int, default=None); s.add_argument("--port", type=int, default=None)
    s.add_argument("--no-flash", action="store_true"); s.add_argument("--foreground", action="store_true")
    sub.add_parser("status"); sub.add_parser("stop"); sub.add_parser("catalog")
    sub.add_parser("discover", help="auto-detect models already on disk (gguf + HF snapshots)")
    sub.add_parser("endpoints", help="probe live inference servers (ollama/vllm/llama.cpp/LM Studio)")
    a = ap.parse_args(argv)
    if a.cmd == "endpoints":
        live = probe_endpoints()
        if not live:
            print("no live inference servers found on the well-known ports.")
            print("start one (ollama serve / llama-server / vllm serve), or set REXGRAPH_PROBE_URLS.")
            return
        print(f"live endpoints ({len(live)}):")
        for e in live:
            tag = f" [{e['managed']}]" if e.get("managed") else ""
            print(f"  {e['url']:34s} {e['kind']:7s}{tag}  {e['n_models']} model(s): "
                  f"{', '.join(e['models'][:4])}{'…' if e['n_models']>4 else ''}")
        return
    if a.cmd == "discover":
        models = discover_local_models()
        if not models:
            print("no local models found. searched:", ", ".join(_default_scan_dirs()) or "(no existing dirs)")
            print("set REXGRAPH_MODEL_DIRS to point at your model folder(s).")
            return
        print(f"detected {len(models)} local model(s):")
        for m in models:
            print(f"  {m['name']:44s} {m['format']:12s} ~{m['size_gb']:>6.1f}GB  "
                  f"[{m['source']}] -> {m['loadable']}\n      {m['path']}")
        return
    if a.cmd == "catalog":
        hw = detect_hardware()
        print(f"detected: backends={hw['backends']} gpu={hw['gpu']} "
              f"ram={hw['ram_gb']}GB  model budget≈{hw['model_budget_gb']}GB")
        print("models that fit this machine (biggest first):")
        for c in recommend(hw["model_budget_gb"]):
            print(f"  {c['name']:34s} {c['kind']:5s} {c['active']:8s} ~{c['approx_gb']:>3}GB "
                  f"[{c['tier']}]\n      {c['why']}")
        return
    if a.cmd == "status":
        print(json.dumps(status(), indent=2)); return
    if a.cmd == "stop":
        stop(); print("stopped"); return
    st = start(a.model_path, ctx_size=a.ctx, n_gpu_layers=a.ngl, port=a.port,
               flash_attn=(not a.no_flash))
    print(json.dumps(st, indent=2))
    if a.foreground:
        print("serving - Ctrl-C to stop"); import time as _t
        try:
            while _PROC and _PROC.poll() is None:
                _t.sleep(1)
        except KeyboardInterrupt:
            stop(); print("\nstopped")


if __name__ == "__main__":
    main()
