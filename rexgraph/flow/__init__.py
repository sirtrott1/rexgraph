"""rexgraph.flow: a lazy field navigator over a changing relational complex.

The flow subsystem watches a stream of rex snapshots and reacts to structural
SURPRISE rather than to the raw size of a change. A large edit that leaves the
topology alone is not news; a small one that opens a cycle is.

    MalaughGate       a condensed scalar gate on the Malaugh harmonic log entropy
                      H_T, which decides whether a step is worth reacting to
    FlowComplex       the complex carried across steps, built by build_flow_complex
    attention         co-participation attention over the relation neighbourhood
    cochain           the torch-side classifier, exported only when torch is present
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
from rexgraph.flow.hyperflow import flow_adjacency  # noqa: E402

__all__ += ["coparticipation_adjacency", "flow_adjacency"]
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

try:                                    # needs the compiled ternary core, like its neighbours
    from rexgraph.flow.ternary_cochain import TernaryCochain  # noqa: E402

    __all__ += ["TernaryCochain"]
except ImportError:
    pass
