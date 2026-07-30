"""
Core Cython-accelerated algorithms for rexgraph.

Modules are imported with graceful fallback: uncompiled or
unavailable extensions are silently skipped so the package
remains importable during incremental development.
"""
import importlib as _importlib
import warnings as _warnings

_MODULES = [
    # Shared infrastructure
    '_common',
    '_sparse',
    '_linalg',

    # Chain complex construction
    '_rex',
    '_boundary',
    '_cycles',
    '_faces',

    # Laplacians and spectral
    '_overlap',
    '_frustration',
    '_laplacians',
    '_relational',
    '_character',
    '_hodge',
    '_harmonic',

    # Topology and curvature
    '_void',
    '_rcfe',
    '_curvature',

    # Dynamics and state
    '_state',
    '_wave',
    '_transition',
    '_field',
    '_dirac',

    # Analysis
    '_standard',
    '_spectral',
    '_temporal',
    '_temporal_entity',
    '_persistence',
    '_hypermanifold',

    # RCF operations
    '_quotient',
    '_joins',
    '_query',
    '_fiber',
    '_signal',
    '_interfacing',
    '_channels',
    '_cross_complex',
    '_l_gb',
    '_holomorphic',

    # Domain-specific
    '_color',
]

_loaded = []
_failed = []

for _mod_name in _MODULES:
    try:
        _mod = _importlib.import_module(f'.{_mod_name}', __name__)
        _names = getattr(_mod, '__all__', None)
        if _names is None:
            _names = [n for n in dir(_mod) if not n.startswith('_')]
        globals()[_mod_name] = _mod
        globals().update({n: getattr(_mod, n) for n in _names})
        _loaded.append(_mod_name)
    except ImportError as _e:
        # Extension not compiled or its shared library dependencies are
        # missing (e.g. libopenblas, libgomp). Emit a warning so a stale
        # or partial build does not silently reduce functionality;
        # downstream code that tries to use the module will surface a
        # clearer AttributeError or None-method error.
        _failed.append(_mod_name)
        _warnings.warn(
            f"rexgraph.core.{_mod_name} not available: {_e}",
            ImportWarning,
            stacklevel=1,
        )
    except Exception as _e:
        _failed.append(_mod_name)
        _warnings.warn(
            f"rexgraph.core.{_mod_name} failed to import: {_e}",
            ImportWarning,
            stacklevel=1,
        )

__all__ = []
del _importlib, _warnings, _mod_name, _MODULES
try:
    del _mod, _names, _e
except NameError:
    pass
