"""
agent.diagnostics: verify the compiled-kernel surface.

The pipeline is meant to dispatch heavy linear algebra (boundary maps,
Hodge decomposition, spectral bundles, RCFE strain) into the compiled
Cython extensions in ``rexgraph.core``.  If those extensions are not
built, ``rexgraph.core`` silently falls back and the pipeline quietly
runs pure-Python paths that are far slower.

This module reports, at runtime, exactly which core modules loaded as
compiled extensions versus which are missing, and whether the key
``RexGraph`` methods the pipeline relies on actually execute.  Use it to
confirm a build before trusting benchmark numbers.
"""

from __future__ import annotations

import importlib

# The core modules the pipeline touches directly or transitively.
PIPELINE_CRITICAL = [
    "_rex", "_boundary", "_faces", "_laplacians", "_hodge", "_harmonic",
    "_spectral", "_relational", "_character", "_rcfe", "_interfacing",
    "_persistence", "_linalg", "_sparse",
]


def _is_compiled(mod) -> bool:
    """A compiled extension has a .so/.pyd file; a .py fallback does not."""
    f = getattr(mod, "__file__", "") or ""
    return f.endswith((".so", ".pyd")) or ".cpython-" in f


def core_module_report() -> dict[str, dict]:
    """Return {module_name: {loaded, compiled, file}} for core modules."""
    report: dict[str, dict] = {}
    try:
        core = importlib.import_module("rexgraph.core")
    except Exception as e:
        return {"__error__": {"loaded": False, "compiled": False, "file": str(e)}}

    # rexgraph.core loads submodules into its namespace; probe each.
    from rexgraph.core import __init__ as _  # noqa: F401
    # Re-read the declared module list if present, else use critical set.
    names = getattr(core, "_MODULES", None) or PIPELINE_CRITICAL
    for name in names:
        entry = {"loaded": False, "compiled": False, "file": ""}
        try:
            sub = importlib.import_module(f"rexgraph.core.{name}")
            entry["loaded"] = True
            entry["compiled"] = _is_compiled(sub)
            entry["file"] = getattr(sub, "__file__", "") or ""
        except Exception as e:
            entry["file"] = f"import error: {e}"
        report[name] = entry
    return report


def method_dispatch_report() -> dict[str, bool]:
    """Run the key RexGraph methods on a tiny complex and record success.

    A ``True`` means the method executed (dispatching into whatever
    backend is present); a ``False`` means it raised, typically because
    its Cython kernel is not compiled.
    """
    import numpy as np

    from rexgraph.graph import RexGraph

    out: dict[str, bool] = {}
    try:
        s = np.array([0, 1, 2, 0, 1, 3], dtype=np.int32)
        t = np.array([1, 2, 0, 2, 3, 0], dtype=np.int32)
        rex = RexGraph(sources=s, targets=t).promote()
    except Exception as e:
        return {"__construct__": False, "__error__": str(e)}  # type: ignore

    checks = {
        "betti": lambda: list(rex.betti),
        "hodge_full": lambda: rex.hodge_full(np.ones(rex.nE)),
        "spectral_bundle": lambda: rex.spectral_bundle,
        "structural_character": lambda: rex.structural_character,
        "coherence": lambda: rex.coherence,
        "to_dict": lambda: rex.to_dict(),
        # demand-driven agentic-reading kernels - the higher-level health/context
        # layers depend on these; smoke-test them so a missing kernel is reported
        # rather than silently degrading the reading.
        "coherence_response": lambda: rex.coherence_response([0]),
        "effective_resistance": lambda: rex.effective_resistance(0),
        "local_context": lambda: rex.local_context([0]),
        "explain_context": lambda: rex.explain_context([0], [0]),
        "agentic_reading": lambda: rex.agentic_reading(vertices=[0]),
    }
    for name, fn in checks.items():
        try:
            fn()
            out[name] = True
        except Exception:
            out[name] = False
    return out


def summary() -> dict:
    """Full diagnostic payload: compiled coverage + method dispatch."""
    mods = core_module_report()
    n_compiled = sum(1 for v in mods.values() if v.get("compiled"))
    n_loaded = sum(1 for v in mods.values() if v.get("loaded"))
    critical_missing = [
        m for m in PIPELINE_CRITICAL
        if m in mods and not mods[m].get("compiled")
    ]
    return {
        "n_modules": len(mods),
        "n_loaded": n_loaded,
        "n_compiled": n_compiled,
        "all_critical_compiled": len(critical_missing) == 0,
        "critical_missing": critical_missing,
        "modules": mods,
        "method_dispatch": method_dispatch_report(),
    }


def format_report() -> str:
    """Human-readable one-screen summary."""
    s = summary()
    lines = [
        "RexGraph kernel diagnostics",
        "=" * 34,
        f"core modules loaded : {s['n_loaded']}/{s['n_modules']}",
        f"compiled extensions : {s['n_compiled']}/{s['n_modules']}",
        f"critical compiled   : {'YES' if s['all_critical_compiled'] else 'NO'}",
    ]
    if s["critical_missing"]:
        lines.append("  MISSING (pure-Python fallback, slow):")
        for m in s["critical_missing"]:
            lines.append(f"    - rexgraph.core.{m}")
        lines.append("  Rebuild with: make build   (or: pip install -e . )")
    lines.append("")
    lines.append("Method dispatch (executes without error?):")
    for name, ok in s["method_dispatch"].items():
        if not isinstance(ok, bool):
            lines.append(f"  --   {name}: {ok}")
            continue
        lines.append(f"  {'ok ' if ok else 'ERR'}  {name}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(format_report())
