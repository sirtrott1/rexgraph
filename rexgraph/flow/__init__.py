"""Flow navigation and learning over a changing relational complex.

The native navigator follows changes in Malaugh entropy. Attention, flow
complexes and classifiers load when requested, keeping their compatibility
and learning dependencies out of the navigator import path.
"""
from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

from rexgraph.flow.gate import MalaughGate, malaugh_entropy
from rexgraph.flow.navigator import FieldNavigator, changed_edges, flow_step

_LAZY_EXPORTS = {
    "TurnField": "turn_field",
    "FlowComplex": "hyperflow",
    "build_flow_complex": "hyperflow",
    "flow_adjacency": "hyperflow",
    "coparticipation_neighbors": "attention",
    "coparticipation_attention": "attention",
    "CoParticipationAttention": "attention",
    "coparticipation_adjacency": "cochain",
    "CoParticipationCochain": "cochain",
    "TernaryCochain": "ternary_cochain",
}

__all__ = ["MalaughGate", "malaugh_entropy", "FieldNavigator", "flow_step", "changed_edges"]
__all__ += [name for name in _LAZY_EXPORTS if name != "CoParticipationCochain"]
if find_spec("torch") is not None:
    __all__.append("CoParticipationCochain")


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
