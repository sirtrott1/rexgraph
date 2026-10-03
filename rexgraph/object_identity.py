"""Semantic identities for native RexGraph values.

This module binds graph/state meaning, not a storage container.  It therefore sits
above the raw digest primitives in :mod:`rexgraph.identity` and below HDF5/Zarr/RCBD,
RCDB, and other persistence adapters.
"""
from __future__ import annotations

from typing import Any

from rexgraph.identity import manifest_digest

OBJECT_IDENTITY_VERSION = 2

__all__ = ["OBJECT_IDENTITY_VERSION", "object_digest", "state_object_digest"]


def object_digest(value: Any) -> str:
    """Return the semantic identity of a native static or temporal complex."""
    from rexgraph.graph import TemporalRex

    if isinstance(value, TemporalRex):
        from rexgraph.temporal_state import to_temporal_state, verify_temporal_state

        state = to_temporal_state(value)
        if not verify_temporal_state(state):  # pragma: no cover: writer invariant
            raise ValueError("could not produce a verified TemporalState identity")
        return str(state.header["digest"])

    from rexgraph.state import to_state

    return state_object_digest(to_state(value))


def state_object_digest(state: Any) -> str:
    """Return semantic identity directly from a verified canonical ``RexState``."""
    from rexgraph.state import RexState, verify_state

    if not isinstance(state, RexState) or not verify_state(state):
        raise ValueError("a verified canonical RexState is required")
    if state.header.get("format_version") == 10:
        # Native tensors seal the complete semantic record already. Container
        # magic, wire framing and optional derived cache metadata are not identity.
        return state.header["digest"]
    return manifest_digest(
        {
            "header": state.header,
            "object_identity_version": OBJECT_IDENTITY_VERSION,
            "object_type": "RexGraphState",
        }
    )
