"""rexgraph.nn - components to build and train models with the RCF math.

Provides the parts to build and train models, not the models themselves. torch is an optional
dependency (guarded at use); without it the numpy `rexgraph` core and the hodge_* primitives
still work.

  optim                - HodgeAdam / HodgeSGD (vector-Hodge / functional-ANOVA preconditioners),
                         build_optimizer, hodge_groups (architecture-aware), training backends
  relational_attention - PropagatorAttention / CausalPropagatorAttention (attention on f(L_W))
  rcf_torch            - differentiable RCF propagators (heat / wave / green_resolvent)
  factory              - component registry and builders (make_optimizer, build_attention,
                         build_model); native pieces are the defaults
"""
from . import rcf_torch  # noqa: F401

from .optim import (  # noqa: F401
    HodgeAdam, HodgeSGD, build_optimizer, hodge_groups,
    training_backends, pick_device, save_hodge_trajectory,
    hodge_matrix_decompose, hodge_matrix_precondition,
    hodge_flow_decompose, hodge_flow_precondition,
)
from .relational_attention import PropagatorAttention, CausalPropagatorAttention  # noqa: F401
from .layers import GreenResolvent, PropagatorMix, RelationalBlock  # noqa: F401
from .factory import (  # noqa: F401
    register, list_components, inventory, default_name, available,
    build_attention, make_optimizer, build_model,
)

__all__ = [
    "rcf_torch",
    "HodgeAdam", "HodgeSGD", "build_optimizer", "hodge_groups",
    "training_backends", "pick_device", "save_hodge_trajectory",
    "hodge_matrix_decompose", "hodge_matrix_precondition",
    "hodge_flow_decompose", "hodge_flow_precondition",
    "PropagatorAttention", "CausalPropagatorAttention",
    "GreenResolvent", "PropagatorMix", "RelationalBlock",
    "register", "list_components", "inventory", "default_name", "available",
    "build_attention", "make_optimizer", "build_model",
]
