"""Requested output contracts; policies never silently change arithmetic."""
from __future__ import annotations

from collections.abc import Mapping

from .types import Exactness


def check_policy(policy):
    if type(policy) is not str or policy not in {"declared", "exact"}:
        raise ValueError("exactness must be 'declared' or 'exact'")
    return policy


def require_exact(value, *, seen=None):
    """Reject approximate numeric outputs, including mixed result containers.

    Structural sources/descriptors retain their structural contracts; this does
    not reinterpret their annotations as numerical query output.
    """
    from rexgraph.value import Approx
    from .executor import value_exactness
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Approx) or value_exactness(value) in {Exactness.APPROXIMATE, Exactness.ROUNDED}:
        raise ValueError("exact output requested but the result contains approximate or rounded values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_exact(key, seen=seen); require_exact(item, seen=seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_exact(item, seen=seen)
    elif getattr(getattr(value, "dtype", None), "kind", None) == "O":
        for item in value.flat:
            require_exact(item, seen=seen)
    elif type(value).__module__.startswith("rcql."):
        from dataclasses import fields, is_dataclass
        if is_dataclass(value):
            for field in fields(value):
                if field.name in {"values", "outputs", "result"}:
                    require_exact(getattr(value, field.name), seen=seen)


__all__ = ["check_policy", "require_exact"]
