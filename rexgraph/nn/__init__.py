"""Neural network operators, layers, optimizer builders and checkpoint lifecycle.

PropagatorAttention and CausalPropagatorAttention use functions of relation
operators to mix signals. The composable layers expose heat and Green
operations with gradients. make_optimizer selects GreensCochain for models
that supply complex parameter groups and Adam for ordinary parameter groups.
The registry also provides AdamW and SGD by name.

PyTorch is optional. Matrix and flow decomposition helpers remain available
without it. The lifecycle interface captures model declarations, weights and
optimizer state for restoration and continued training.
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
    hodge_matrix_decompose,
    hodge_matrix_precondition,
    pick_device,
    save_hodge_trajectory,
    training_backends,
)

from .relational_attention import CausalPropagatorAttention, PropagatorAttention  # noqa: F401
from .packed import PackedTernaryLinear  # noqa: F401

__all__ = [
    "rcf_torch",
    "GreensCochain",
    "GreensFlow", "generate_khop_channel", "build_optimizer",
    "training_backends", "pick_device", "save_hodge_trajectory",
    "hodge_matrix_decompose", "hodge_matrix_precondition",
    "hodge_flow_decompose", "hodge_flow_precondition",
    "PropagatorAttention", "CausalPropagatorAttention",
    "GreenResolvent", "PropagatorMix", "RelationalBlock", "PackedTernaryLinear",
    "register", "list_components", "inventory", "default_name", "available",
    "build_attention", "make_optimizer", "build_model",
]

from .lifecycle import (
    ModelAdapter, register_model_adapter, create_checkpoint, capture_checkpoint,
    restore_checkpoint, infer_checkpoint, train_checkpoint, transport_checkpoint,
)
__all__ += ["ModelAdapter", "register_model_adapter", "create_checkpoint", "capture_checkpoint",
            "restore_checkpoint", "infer_checkpoint", "train_checkpoint", "transport_checkpoint"]
