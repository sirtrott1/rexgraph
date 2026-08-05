"""
attn_introspect: Tier-2 attention bridge (Python side).

Runs the `rex_attn_capture` host (agent/agent/native/, built against the local llama.cpp via
its stock cb_eval callback, no ggml patch) to pull a model's internal per-layer attention
weights, then feeds each layer's map to the RCF analyzer so the relational-complex math reads
the model's own attention: Hodge grad/curl/harmonic, the four channels, ∂²=0 compliance, per-
layer structural character. This reaches what the OpenAI API never surfaces.

Degrades cleanly: if the capture host isn't built (or no model given), `available()` is False
and callers fall back to Tier-1 (`model_introspect.embed`). Public code paths never require it.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np


def _capture_binary() -> str | None:
    """Locate the built rex_attn_capture host (env override, then native/ dir)."""
    env = os.environ.get("REX_ATTN_CAPTURE_BIN")
    if env and os.path.exists(os.path.expanduser(env)):
        return os.path.expanduser(env)
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "rex_attn_capture")
    return here if os.path.exists(here) else None


def available() -> bool:
    return _capture_binary() is not None


def _resolve_model(model_path: str | None) -> str | None:
    if model_path:
        return os.path.expanduser(model_path)
    try:                                             # the model local_runtime currently serves
        from agent import local_runtime
        mp = (local_runtime.status() or {}).get("model_path")
        if mp and os.path.exists(mp):
            return mp
    except Exception:
        pass
    return None


def _lib_env() -> dict:
    """Ensure the capture host finds llama.cpp's shared libs (they sit in build/bin)."""
    env = dict(os.environ)
    from agent import local_runtime
    b = local_runtime.find_binary()
    if b:
        bindir = os.path.dirname(os.path.abspath(b))
        env["LD_LIBRARY_PATH"] = bindir + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def capture_attention(prompt: str, model_path: str | None = None, *, n_gpu_layers: int = 999,
                      timeout: float = 120.0) -> dict:
    """Run one forward pass and capture per-layer attention (averaged over heads). Returns
    {n_tokens, n_layers, layers:[{layer, n_kv, n_q, n_head, attn:[[...]]}]}. Raises with
    actionable guidance if the host or model is missing."""
    binary = _capture_binary()
    if binary is None:
        raise RuntimeError(
            "Tier-2 attention capture host not built. Build llama.cpp, then "
            "`LLAMA_DIR=~/llama.cpp bash agent/agent/native/build.sh`. "
            "Until then use Tier-1 embeddings (agent.model_introspect).")
    model = _resolve_model(model_path)
    if model is None:
        raise RuntimeError("No GGUF model. Pass model_path or start one via local_runtime.")
    out = subprocess.run([binary, model, prompt, str(n_gpu_layers)],
                         capture_output=True, text=True, timeout=timeout, env=_lib_env())
    if out.returncode != 0:
        raise RuntimeError("attention capture failed: %s" % (out.stderr.strip()[:300] or "unknown"))
    return json.loads(out.stdout)


def attention_complex(prompt: str, model_path: str | None = None, *, layers: list[int] | None = None,
                      threshold: float = 0.05) -> dict:
    """Capture the model's attention and run the RCF analysis per layer: each layer's [n_q×n_kv]
    map -> relational complex -> Hodge grad/curl/harmonic, four channels, coherence, Betti. This is
    the model reading its OWN attention through the relational-complex math (Tier-2)."""
    from agent.integrations.huggingface_analyzer import quick_attention_analysis
    cap = capture_attention(prompt, model_path)
    reports = []
    for L in cap.get("layers", []):
        if layers is not None and L["layer"] not in layers:
            continue
        A = np.array(L["attn"], dtype=np.float64)
        if A.ndim != 2 or A.shape[0] < 3:
            continue
        try:
            r = quick_attention_analysis(A, threshold=threshold)
        except Exception as e:
            r = {"note": f"analysis failed: {type(e).__name__}"}
        r["layer"] = L["layer"]; r["n_tokens"] = int(A.shape[0])
        reports.append(r)
    return {"n_tokens": cap.get("n_tokens"), "n_layers": cap.get("n_layers"),
            "prompt": prompt, "per_layer": reports}
