"""rexgraph._env: best-effort, dependency-light host/environment detection.

This module answers three questions about *the machine RexGraph is running on right now*, so
nothing downstream is hardcoded to one laptop, one env manager, or one GPU:

    detect_python_env()       which Python env manager is active (conda / mamba / micromamba /
                              venv / virtualenv / uv / poetry / pdm / system), the interpreter,
                              the prefix, and whether the toolchain compiler is CONSISTENT with the
                              system compiler (a prior LTO link failure came from a conda gcc-14
                              linker meeting system gcc-16 objects - we detect and WARN on that).
    detect_compute_backends() the compute backends actually available on this host, with
                              capability info: CUDA, ROCm, Vulkan, Metal, CPU, and integrated/APU
                              GPUs (e.g. an AMD "Strix Halo" RDNA3.5 iGPU).
    recommend_backend()       the best backend for this host by a documented priority, overridable
                              by the REXGRAPH_BACKEND environment variable.
    summary()                 a human-readable diagnostic report of all of the above.

EVERYTHING here is best-effort and never raises: probes are wrapped in try/except, third-party
libraries are checked with importlib.util.find_spec (never imported as a hard dependency), and
external tools are run through subprocess with a short timeout. The library must import and run on
a bare, CPU-only, no-toolchain box.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

__all__ = [
    "detect_python_env",
    "detect_compute_backends",
    "recommend_backend",
    "summary",
    "BACKEND_PRIORITY",
    "REXGRAPH_BACKEND_ENV",
]

# The environment variable a user sets to force a backend, overriding auto-detection.
REXGRAPH_BACKEND_ENV = "REXGRAPH_BACKEND"

# Backend preference from most to least capable. recommend_backend() walks this over the
# backends that are actually available on the host. Documented policy, easily overridden.
BACKEND_PRIORITY = ["cuda", "rocm", "vulkan", "metal", "cpu"]


# ----------------------------------------------------------------------------------------------
# small, safe helpers
# ----------------------------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 4.0) -> str | None:
    """Run a command, returning stdout (best-effort) or None. Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return None


def _have(tool: str) -> str | None:
    """Path to an executable on PATH, or None."""
    try:
        return shutil.which(tool)
    except Exception:
        return None


def _has_module(name: str) -> bool:
    """True if an importable module `name` exists - WITHOUT importing it (no heavy side effects)."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _first_version_tuple(text: str | None) -> tuple | None:
    """Pull the first dotted version (e.g. '14.2.0') out of a --version blob -> (14, 2, 0)."""
    if not text:
        return None
    import re
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not m:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


# ----------------------------------------------------------------------------------------------
# Python environment / toolchain
# ----------------------------------------------------------------------------------------------

def _detect_manager() -> str:
    """Best-effort name of the environment manager governing the active interpreter."""
    prefix = sys.prefix
    # conda family: a conda-meta dir in the prefix, or CONDA_PREFIX pointing here.
    conda_meta = os.path.join(prefix, "conda-meta")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if os.path.isdir(conda_meta) or (conda_prefix and os.path.realpath(conda_prefix) == os.path.realpath(prefix)):
        mamba_exe = (os.environ.get("MAMBA_EXE") or "").lower()
        if "micromamba" in mamba_exe or _have("micromamba"):
            # micromamba is the frontend when it created/activated the env
            if "micromamba" in mamba_exe:
                return "micromamba"
        if "mamba" in mamba_exe:
            return "mamba"
        if os.environ.get("CONDA_EXE") or _have("conda"):
            return "conda"
        # a conda-style prefix with no obvious frontend on PATH
        return "conda"
    # PEP 405 virtualenv: prefix differs from the base interpreter prefix.
    in_venv = (getattr(sys, "base_prefix", sys.prefix) != sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))
    if in_venv:
        if os.environ.get("POETRY_ACTIVE") or os.environ.get("POETRY_VIRTUALENV"):
            return "poetry"
        if os.environ.get("PDM_PROJECT_ROOT") or os.environ.get("PDM_PYTHON"):
            return "pdm"
        # uv writes a `uv = <version>` line into pyvenv.cfg
        cfg = os.path.join(sys.prefix, "pyvenv.cfg")
        try:
            with open(cfg, encoding="utf-8", errors="ignore") as fh:
                body = fh.read().lower()
            if "uv =" in body or "uv=" in body:
                return "uv"
            if "virtualenv =" in body or "virtualenv=" in body:
                return "virtualenv"
        except Exception:
            pass
        return "venv"
    return "system"


def _cc_version(path_or_name: str | None) -> tuple | None:
    if not path_or_name:
        return None
    return _first_version_tuple(_run([path_or_name, "--version"]))


def _detect_compiler() -> dict[str, Any]:
    """Locate the env's C compiler and the system C compiler and check MAJOR-version consistency.

    A prior bug: conda-provided gcc-14 (the env's linker) tried to link objects built by the
    system's gcc-16, and LTO failed. When the env compiler and the bare system compiler differ in
    major version, we surface a warning so the build path can pin one toolchain.
    """
    info: dict[str, Any] = {
        "env_cc": None, "env_version": None,
        "system_cc": None, "system_version": None,
        "consistent": True, "warning": None,
    }
    # The env's compiler: honor CC, else a conda-style triplet cc in the prefix bin, else PATH cc.
    env_cc = os.environ.get("CC")
    if not env_cc:
        bindir = os.path.join(sys.prefix, "bin")
        for cand in ("cc", "gcc", "clang", "x86_64-conda-linux-gnu-cc", "x86_64-conda-linux-gnu-gcc"):
            p = os.path.join(bindir, cand)
            if os.path.exists(p):
                env_cc = p
                break
    if not env_cc:
        env_cc = _have("cc") or _have("gcc") or _have("clang")
    info["env_cc"] = env_cc
    info["env_version"] = _cc_version(env_cc)

    # The bare system compiler (outside the env), typically /usr/bin.
    sys_cc = None
    for p in ("/usr/bin/cc", "/usr/bin/gcc", "/usr/bin/clang"):
        if os.path.exists(p):
            sys_cc = p
            break
    info["system_cc"] = sys_cc
    info["system_version"] = _cc_version(sys_cc)

    ev, sv = info["env_version"], info["system_version"]
    if ev and sv and env_cc and sys_cc and os.path.realpath(env_cc) != os.path.realpath(sys_cc):
        if ev[0] != sv[0]:
            info["consistent"] = False
            info["warning"] = (
                f"env compiler {env_cc} (v{ev[0]}) and system compiler {sys_cc} (v{sv[0]}) differ "
                f"in major version; mixing their objects/linker can break LTO. Pin one toolchain "
                f"(set CC/CXX to the env compiler) for native builds."
            )
    return info


def detect_python_env() -> dict[str, Any]:
    """Describe the active Python environment manager and toolchain (best-effort, never raises)."""
    env: dict[str, Any] = {
        "manager": "system",
        "python": sys.executable,
        "version": platform.python_version(),
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "in_venv": False,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "frontends": [],
        "compiler": {},
        "has_toolchain": False,
        "warnings": [],
    }
    try:
        env["manager"] = _detect_manager()
    except Exception:
        env["manager"] = "system"
    with contextlib.suppress(Exception):
        env["in_venv"] = (getattr(sys, "base_prefix", sys.prefix) != sys.prefix) or bool(env["virtual_env"])
    with contextlib.suppress(Exception):
        env["frontends"] = [t for t in ("micromamba", "mamba", "conda", "uv", "poetry", "pdm")
                            if _have(t)]
    try:
        comp = _detect_compiler()
        env["compiler"] = comp
        env["has_toolchain"] = bool(comp.get("env_cc") or comp.get("system_cc"))
        if comp.get("warning"):
            env["warnings"].append(comp["warning"])
    except Exception:
        env["compiler"] = {}
    return env


# ----------------------------------------------------------------------------------------------
# Compute backend detection
# ----------------------------------------------------------------------------------------------

def _cpu_backend() -> dict[str, Any]:
    cores = os.cpu_count() or 1
    simd: list[str] = []
    try:
        if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", errors="ignore") as fh:
                blob = fh.read().lower()
            for flag in ("avx512f", "avx2", "avx", "sse4_2", "neon", "asimd"):
                if flag in blob:
                    simd.append(flag)
    except Exception:
        pass
    if not simd and platform.machine().lower() in ("arm64", "aarch64"):
        simd.append("neon")
    return {
        "name": "cpu", "kind": "cpu", "available": True, "integrated": False,
        "vendor": platform.machine() or "unknown", "via": "builtin",
        "devices": 1, "cores": cores, "simd": simd,
        "detail": f"{cores} logical cores"
        + (f"; SIMD {','.join(simd)}" if simd else ""),
    }


def _detect_amd_igpus() -> list[dict[str, Any]]:
    """Integrated/APU AMD GPUs via /sys/class/drm vendor id 0x1002, or lspci VGA/Display lines.

    Covers APUs like the AMD "Strix Halo" Ryzen AI Max whose RDNA3.5 iGPU shares system memory.
    """
    found: list[dict[str, Any]] = []
    # sysfs: /sys/class/drm/card*/device/vendor == 0x1002 (AMD), no discrete VRAM sysfs => integrated
    try:
        drm = "/sys/class/drm"
        if os.path.isdir(drm):
            seen = set()
            for entry in sorted(os.listdir(drm)):
                if not entry.startswith("card") or "-" in entry:
                    continue
                vpath = os.path.join(drm, entry, "device", "vendor")
                try:
                    with open(vpath) as fh:
                        vendor = fh.read().strip().lower()
                except Exception:
                    continue
                if vendor != "0x1002":
                    continue
                # integrated heuristic: no dedicated vram info file, or tiny/absent mem_info_vram_total
                dev = os.path.join(drm, entry, "device")
                integrated = True
                try:
                    vram_file = os.path.join(dev, "mem_info_vram_total")
                    if os.path.exists(vram_file):
                        with open(vram_file) as fh:
                            total = int(fh.read().strip() or "0")
                        # >= 4 GiB dedicated strongly suggests a discrete card, not an APU
                        integrated = total < (4 * 1024 * 1024 * 1024)
                except Exception:
                    integrated = True
                key = ("amd", entry)
                if key in seen:
                    continue
                seen.add(key)
                found.append({"vendor": "amd", "integrated": integrated, "via": "sysfs",
                              "name": f"AMD GPU ({entry})"})
    except Exception:
        pass
    # lspci fallback (only if sysfs found nothing)
    if not found and _have("lspci"):
        out = _run(["lspci"]) or ""
        for line in out.splitlines():
            low = line.lower()
            if ("vga" in low or "display" in low or "3d controller" in low) and \
               ("amd" in low or "radeon" in low or "ati" in low):
                found.append({"vendor": "amd", "integrated": True, "via": "lspci",
                              "name": line.split(":", 2)[-1].strip() or "AMD GPU"})
    return found


def detect_compute_backends() -> list[dict[str, Any]]:
    """Ordered list of compute backends available on this host, with capability info.

    Every entry has at least: name, kind ('cpu'|'gpu'), available (True), integrated (bool),
    vendor, via (how it was detected), devices (count), detail (str). CPU is ALWAYS present and
    always last. GPU entries are ordered by BACKEND_PRIORITY. Nothing here raises.
    """
    backends: list[dict[str, Any]] = []

    # --- CUDA (NVIDIA) ---
    try:
        cuda_devs = 0
        via = None
        smi = _have("nvidia-smi")
        if smi:
            out = _run([smi, "-L"])
            if out:
                n = sum(1 for ln in out.splitlines() if ln.strip().lower().startswith("gpu "))
                if n:
                    cuda_devs, via = n, "nvidia-smi"
        if cuda_devs == 0 and (_has_module("cupy") or _has_module("torch")):
            # library present but no device enumerated - report as a *capable* path, unproven device
            via = via or ("cupy" if _has_module("cupy") else "torch")
        if cuda_devs > 0:
            backends.append({"name": "cuda", "kind": "gpu", "available": True, "integrated": False,
                             "vendor": "nvidia", "via": via, "devices": cuda_devs,
                             "detail": f"{cuda_devs} NVIDIA GPU(s) via {via}"})
    except Exception:
        pass

    # --- ROCm (AMD, discrete or APU) ---
    try:
        rocm_devs = 0
        integrated = False
        via = None
        rinfo = _have("rocminfo")
        if rinfo:
            out = _run([rinfo])
            if out:
                # count agents whose Device Type is GPU
                gpu_agents = 0
                block_is_gpu = False
                for ln in out.splitlines():
                    s = ln.strip()
                    if s.startswith("Device Type:"):
                        block_is_gpu = "GPU" in s
                        if block_is_gpu:
                            gpu_agents += 1
                if gpu_agents:
                    rocm_devs, via = gpu_agents, "rocminfo"
                    # APU hint: rocminfo mentions APU / integrated memory pools
                    low = out.lower()
                    integrated = ("apu" in low) or ("integrated" in low)
        if rocm_devs == 0 and (os.environ.get("ROCR_VISIBLE_DEVICES") is not None
                               or _has_module("torch")):
            via = via or "env/torch"
        if rocm_devs > 0:
            backends.append({"name": "rocm", "kind": "gpu", "available": True,
                             "integrated": integrated, "vendor": "amd", "via": via,
                             "devices": rocm_devs,
                             "detail": f"{rocm_devs} AMD GPU(s) via {via}"
                             + (" (integrated/APU)" if integrated else "")})
    except Exception:
        pass

    # --- Integrated/APU AMD GPUs not surfaced by rocminfo (no ROCm stack installed) ---
    try:
        have_rocm = any(b["name"] == "rocm" for b in backends)
        if not have_rocm:
            igpus = _detect_amd_igpus()
            if igpus:
                integrated = any(g.get("integrated") for g in igpus)
                names = "; ".join(g.get("name", "AMD GPU") for g in igpus[:2])
                backends.append({"name": "rocm", "kind": "gpu", "available": True,
                                 "integrated": integrated, "vendor": "amd",
                                 "via": igpus[0].get("via", "sysfs"), "devices": len(igpus),
                                 "detail": f"AMD GPU present without ROCm runtime: {names}"
                                 + (" (integrated/APU)" if integrated else "")
                                 + " - needs a ROCm/Vulkan stack to compute"})
    except Exception:
        pass

    # --- Vulkan ---
    try:
        vk = None
        vinfo = _have("vulkaninfo")
        if vinfo:
            out = _run([vinfo, "--summary"]) or _run([vinfo])
            if out and ("GPU" in out or "deviceName" in out or "apiVersion" in out):
                vk = "vulkaninfo"
        if vk is None:
            try:
                import ctypes.util
                if ctypes.util.find_library("vulkan"):
                    vk = "libvulkan"
            except Exception:
                pass
        if vk:
            backends.append({"name": "vulkan", "kind": "gpu", "available": True,
                             "integrated": False, "vendor": "any", "via": vk, "devices": 0,
                             "detail": f"Vulkan runtime present via {vk}"})
    except Exception:
        pass

    # --- Metal (Apple) ---
    try:
        if platform.system() == "Darwin":
            backends.append({"name": "metal", "kind": "gpu", "available": True,
                             "integrated": True, "vendor": "apple", "via": "platform",
                             "devices": 1, "detail": "Apple Metal (macOS)"})
    except Exception:
        pass

    # --- CPU (always) ---
    try:
        backends.append(_cpu_backend())
    except Exception:
        backends.append({"name": "cpu", "kind": "cpu", "available": True, "integrated": False,
                         "vendor": "unknown", "via": "builtin", "devices": 1, "cores": 1,
                         "simd": [], "detail": "CPU"})

    # order by BACKEND_PRIORITY (unknown names, if any, before cpu)
    def _rank(b: dict[str, Any]) -> int:
        try:
            return BACKEND_PRIORITY.index(b["name"])
        except ValueError:
            return len(BACKEND_PRIORITY) - 1  # just before cpu
    backends.sort(key=_rank)
    return backends


def _names(available: Any) -> list[str]:
    if available is None:
        available = detect_compute_backends()
    out: list[str] = []
    for item in available:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("name"):
            out.append(item["name"])
    return out


def recommend_backend(available: Any = None) -> str:
    """The best backend name for this host.

    Priority:
      1. The REXGRAPH_BACKEND environment variable, if set and non-empty - it WINS (explicit
         user/operator override, returned verbatim, lower-cased).
      2. Otherwise the first backend in BACKEND_PRIORITY (cuda > rocm > vulkan > metal > cpu)
         that is actually available on this host.
      3. Otherwise 'cpu' (always a valid fallback).

    `available` may be omitted (auto-detected), a list of backend names, or the list of dicts
    returned by detect_compute_backends().
    """
    override = os.environ.get(REXGRAPH_BACKEND_ENV)
    if override and override.strip():
        return override.strip().lower()
    names = _names(available)
    for cand in BACKEND_PRIORITY:
        if cand in names:
            return cand
    return "cpu"


# ----------------------------------------------------------------------------------------------
# human-readable report
# ----------------------------------------------------------------------------------------------

def summary() -> str:
    """A human-readable diagnostic of the Python env, toolchain, and compute backends."""
    lines: list[str] = []
    try:
        env = detect_python_env()
    except Exception as e:  # pragma: no cover - defensive
        env = {"manager": "unknown", "warnings": [f"env detection failed: {e}"]}
    lines.append("RexGraph environment")
    lines.append("=" * 60)
    lines.append(f"  manager    : {env.get('manager')}")
    lines.append(f"  python     : {env.get('version')}  ({env.get('python')})")
    lines.append(f"  prefix     : {env.get('prefix')}")
    if env.get("in_venv"):
        lines.append(f"  virtualenv : {env.get('virtual_env') or env.get('prefix')}")
    if env.get("frontends"):
        lines.append(f"  frontends  : {', '.join(env['frontends'])}")
    comp = env.get("compiler") or {}
    if comp.get("env_cc"):
        ev = comp.get("env_version")
        lines.append(f"  compiler   : {comp['env_cc']}"
                     + (f" (v{'.'.join(map(str, ev))})" if ev else ""))
    if comp.get("system_cc") and comp.get("system_cc") != comp.get("env_cc"):
        sv = comp.get("system_version")
        lines.append(f"  system cc  : {comp['system_cc']}"
                     + (f" (v{'.'.join(map(str, sv))})" if sv else ""))
    for w in env.get("warnings", []):
        lines.append(f"  WARNING    : {w}")

    lines.append("")
    lines.append("Compute backends (best first)")
    lines.append("=" * 60)
    try:
        backends = detect_compute_backends()
    except Exception as e:  # pragma: no cover - defensive
        backends = []
        lines.append(f"  backend detection failed: {e}")
    for b in backends:
        tag = "GPU" if b.get("kind") == "gpu" else "CPU"
        igpu = " [integrated]" if b.get("integrated") else ""
        lines.append(f"  {b['name']:<8} {tag}{igpu}: {b.get('detail', '')}")
    try:
        rec = recommend_backend(backends)
    except Exception:
        rec = "cpu"
    override = os.environ.get(REXGRAPH_BACKEND_ENV)
    lines.append("")
    lines.append(f"  recommended: {rec}"
                 + (f"  (forced via {REXGRAPH_BACKEND_ENV})" if override else ""))
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(summary())
