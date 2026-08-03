"""rexgraph.flow: a lazy field navigator over a changing relational complex.

The flow subsystem watches a stream of rex snapshots and reacts to structural
surprise rather than to raw size of change. Task 1 is the MalaughGate, a
condensed scalar gate on the Malaugh harmonic log entropy H_T. Later tasks in
this slice (the field navigator itself, the per-step flow update, and changed
edge detection) attach to this package as they land.
"""
from __future__ import annotations

from rexgraph.flow.attention import (
    CoParticipationAttention,
    coparticipation_attention,
    coparticipation_neighbors,
)
from rexgraph.flow.gate import MalaughGate, malaugh_entropy
from rexgraph.flow.hyperflow import FlowComplex, build_flow_complex

__all__ = [
    "FlowComplex",
    "build_flow_complex",
    "MalaughGate",
    "malaugh_entropy",
    "coparticipation_neighbors",
    "coparticipation_attention",
    "CoParticipationAttention",
]

# the cochain classifier needs torch (optional dependency); export it when available
from rexgraph.flow.cochain import coparticipation_adjacency  # noqa: E402

__all__ += ["coparticipation_adjacency"]
try:
    from rexgraph.flow.cochain import CoParticipationCochain  # noqa: E402

    __all__ += ["CoParticipationCochain"]
except ImportError:
    pass

try:
    from rexgraph.flow.navigator import FieldNavigator, changed_edges, flow_step

    __all__ += ["FieldNavigator", "flow_step", "changed_edges"]
except ImportError:
    pass
