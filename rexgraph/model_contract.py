"""Model/runtime contracts that are independent of training implementations and IO."""
from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

_checkpoint_capture: Callable | None = None


def register_checkpoint_capture(callback: Callable) -> None:
    """Register the concrete training layer checkpoint capture implementation.

    The flow/model layers depend on this contract rather than importing ``nn.lifecycle``.
    Importing again the same implementation is harmless; a different provider is rejected so
    one process cannot silently change checkpoint semantics.
    """
    global _checkpoint_capture
    if not callable(callback):
        raise TypeError("checkpoint capture implementation must be callable")
    if _checkpoint_capture is not None and _checkpoint_capture is not callback:
        previous = (_checkpoint_capture.__module__, _checkpoint_capture.__qualname__)
        incoming = (callback.__module__, callback.__qualname__)
        if previous != incoming:
            raise RuntimeError("a checkpoint capture implementation is already registered")
    _checkpoint_capture = callback


def capture_model_checkpoint(*args, **kwargs):
    """Capture a model checkpoint through the registered lifecycle implementation.

    ``nn.lifecycle`` is imported only as the default provider bootstrap when an application
    has not imported the training layer yet.  That keeps the dependency direction at the
    contract boundary while preserving the historical ``model.checkpoint()`` convenience.
    """
    if _checkpoint_capture is None:
        import_module("rexgraph.nn.lifecycle")
    if _checkpoint_capture is None:  # pragma: no cover: defensive against broken providers
        raise RuntimeError("no model checkpoint capture implementation is registered")
    return _checkpoint_capture(*args, **kwargs)


def model_coordinates(source, grade=1):
    """Return the declared native coordinate space for a model grade."""
    from rexgraph.chain_map import CoordinateComplex

    complex_ = CoordinateComplex.from_rex(source)
    if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade < len(complex_.spaces):
        raise ValueError("model grade is outside the native tower")
    return complex_.spaces[grade]


__all__ = ["model_coordinates", "register_checkpoint_capture", "capture_model_checkpoint"]
