"""Compatibility alias for :mod:`rexgraph.span_codec`.

The semantic contract moved out of :mod:`rexgraph.io`; this legacy module path
resolves to the same module object so imports and monkeypatch based integrations
continue to observe one source of truth.
"""
from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module

_impl = _import_module("rexgraph.span_codec")
_sys.modules[__name__] = _impl
