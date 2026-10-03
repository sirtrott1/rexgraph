"""Compiled RexGraph kernels with explicit capability ownership.

Importing :mod:`rexgraph.core` remains possible from a source tree with a partial native
build. Missing required kernels produce a capability diagnostic. Every shipped
extension belongs to one of three sets:

``required``
    Production RexGraph mathematics.  ``require_core()`` fails with one diagnostic when
    any requested required kernel is unavailable.
``optional``
    Integration accelerators whose owning higher layer may be absent.
``oracle``
    Independent dense/reference implementations retained for parity and benchmarks.

This keeps oracle code installed without allowing it to become an automatic production
fallback.
"""
from __future__ import annotations

import importlib as _importlib
import warnings as _warnings

_REQUIRED_MODULES = (
    # shared ABI / sparse substrate
    "_common", "_sparse", "_linalg",
    # construction
    "_rex", "_boundary", "_cycles", "_faces",
    # Laplacian / relational structure
    "_overlap", "_frustration", "_laplacians", "_relational", "_character", "_hodge",
    # topology / curvature
    "_void", "_rcfe", "_curvature",
    # dynamics / state
    "_state", "_wave", "_transition", "_field", "_dirac",
    # analysis
    "_standard", "_spectral", "_temporal", "_temporal_entity", "_persistence", "_hypermanifold",
    # RCF / sparse operators
    "_quotient", "_joins", "_query", "_fiber", "_signal", "_interfacing", "_channels",
    "_cross_complex", "_l_gb", "_holomorphic", "_exact_ratio", "_ternary", "_channel_tower",
)

# RCDB owns the record log codec.  RexGraph itself does not need it to construct or
# analyze a complex, so absence is reported but does not make the mathematical core
# incomplete.
_OPTIONAL_MODULES = ("_recordlog",)

# Dense harmonic eigensolvers are retained only as independent parity/reference code.
_ORACLE_MODULES = ("_harmonic",)

_MODULES = _REQUIRED_MODULES + _OPTIONAL_MODULES + _ORACLE_MODULES

_loaded: list[str] = []
_failed: list[str] = []
_failures: dict[str, str] = {}

for _mod_name in _MODULES:
    try:
        _mod = _importlib.import_module(f".{_mod_name}", __name__)
        _names = getattr(_mod, "__all__", None)
        if _names is None:
            _names = [n for n in dir(_mod) if not n.startswith("_")]
        globals()[_mod_name] = _mod
        globals().update({n: getattr(_mod, n) for n in _names})
        _loaded.append(_mod_name)
    except ImportError as _e:
        _failed.append(_mod_name)
        _failures[_mod_name] = str(_e)
        _warnings.warn(
            f"rexgraph.core.{_mod_name} not available: {_e}",
            ImportWarning,
            stacklevel=1,
        )
    except Exception as _e:  # pragma: no cover: import time native failure
        _failed.append(_mod_name)
        _failures[_mod_name] = f"{type(_e).__name__}: {_e}"
        _warnings.warn(
            f"rexgraph.core.{_mod_name} failed to import: {_e}",
            ImportWarning,
            stacklevel=1,
        )


def core_status() -> dict[str, object]:
    """Return an introspectable capability table for the native runtime."""
    loaded = frozenset(_loaded)

    def group(names):
        return {
            "modules": tuple(names),
            "loaded": tuple(name for name in names if name in loaded),
            "missing": tuple(name for name in names if name not in loaded),
        }

    return {
        "required": group(_REQUIRED_MODULES),
        "optional": group(_OPTIONAL_MODULES),
        "oracle": group(_ORACLE_MODULES),
        "failures": dict(_failures),
    }


def require_core(*modules: str) -> None:
    """Fail fast when required native kernels for an operation are unavailable.

    With no arguments, require the complete production mathematical core.  A caller may
    pass a smaller feature set when construction itself needs only a subset.
    """
    requested = tuple(modules) if modules else _REQUIRED_MODULES
    known = _REQUIRED_MODULES + _OPTIONAL_MODULES + _ORACLE_MODULES
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(f"unknown RexGraph core module(s): {', '.join(unknown)}")
    missing = [name for name in requested if name not in _loaded]
    if not missing:
        return
    details = "; ".join(
        f"{name}: {_failures.get(name, 'not loaded')}" for name in missing
    )
    raise ImportError(
        "RexGraph native runtime is incomplete; missing "
        + ", ".join(missing)
        + (f" ({details})" if details else "")
    )


__all__ = ["core_status", "require_core"]

del _importlib, _warnings, _mod_name, _MODULES
try:
    del _mod, _names, _e
except NameError:
    pass
