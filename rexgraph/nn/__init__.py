"""rexgraph.nn: components to build and train models with the RCF math.

Provides the parts to build and train models, not the models themselves. torch is an optional
dependency (guarded at use); without it the numpy `rexgraph` core and the hodge_* primitives
still work.

  optim                - the optimizers. Build one with `factory.make_optimizer("auto", ...)`,
                         which routes to GreensCochain (Green's-preconditioned Adam, the native
                         optimizer for relational-native cochain models) or plain Adam for a
                         standard feature-space model. HodgeAdam / HodgeSGD are back-compat only
                         (they tie plain Adam on standard models) and live in `rexgraph.nn.optim`,
                         not on this top-level surface, so nothing reaches for them by name.
                         Also: build_optimizer, hodge_groups (architecture-aware), training backends
  relational_attention: PropagatorAttention / CausalPropagatorAttention (attention on f(L_W))
  rcf_torch            - differentiable RCF propagators (heat / wave / green_resolvent)
  factory              - component registry and builders (make_optimizer, build_attention,
                         build_model); native pieces are the defaults
"""
from . import rcf_torch  # noqa: F401
from .factory import (  # noqa: F401
    available,
    build_attention,
    build_model,
    default_name,
    inventory,
    list_components,
    make_optimizer,
    register,
)
from .layers import GreenResolvent, PropagatorMix, RelationalBlock  # noqa: F401
from .optim import (  # noqa: F401
    GreensCochain,
    GreensFlow,
    build_optimizer,
    generate_khop_channel,
    hodge_flow_decompose,
    hodge_flow_precondition,
    hodge_groups,
    hodge_matrix_decompose,
    hodge_matrix_precondition,
    pick_device,
    save_hodge_trajectory,
    training_backends,
)

# HodgeAdam / HodgeSGD are back-compat only (tie plain Adam on standard models); reach them at
# rexgraph.nn.optim if a legacy caller needs them. Prefer factory.make_optimizer("auto", ...).
from .relational_attention import CausalPropagatorAttention, PropagatorAttention  # noqa: F401

__all__ = [
    "rcf_torch",
    "GreensCochain",
    "GreensFlow", "generate_khop_channel", "build_optimizer", "hodge_groups",
    "training_backends", "pick_device", "save_hodge_trajectory",
    "hodge_matrix_decompose", "hodge_matrix_precondition",
    "hodge_flow_decompose", "hodge_flow_precondition",
    "PropagatorAttention", "CausalPropagatorAttention",
    "GreenResolvent", "PropagatorMix", "RelationalBlock",
    "register", "list_components", "inventory", "default_name", "available",
    "build_attention", "make_optimizer", "build_model",
]
