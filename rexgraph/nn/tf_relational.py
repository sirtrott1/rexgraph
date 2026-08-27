"""The banded causal propagator, in TensorFlow.

Same operator as `relational_attention.CausalPropagatorAttention` with sparse=True,
written against Keras so the relational path is not torch-only. The point of having
both is that they can be checked against each other: the arithmetic is fixed by the
complex, not by the framework, so if the two disagree one of them is wrong.

The sparsity is the causal window, a structural fact about which tokens are
reachable, so the band drops exactly the terms a masked dense softmax multiplies by
zero. Nothing here thresholds a score, takes a spectrum or solves an eigenproblem:
the propagator is the finite series sum_k c_k A^k V and each hop is a banded matvec.

TensorFlow is optional and guarded at use, the same way torch is.
"""
from __future__ import annotations

import math
import os as _os
import sys as _sys

# TensorFlow and a ROCm/CUDA torch each bundle their own LLVM, and whichever
# registers its CommandLine options SECOND aborts the process:
#
#     : CommandLine Error: Option 'print-inst-addrs' registered more than once!
#     LLVM ERROR: inconsistency in registered CommandLine options
#
# That is an abort, not an exception: no traceback, no chance to catch it, the
# interpreter is simply gone. `import tensorflow` then `import torch` is fine; the
# other order is not. Since this module is the place someone first pulls TF into a
# torch process, it checks and RAISES instead, so the failure is legible and names
# the fix. Set REXGRAPH_ALLOW_TF_AFTER_TORCH=1 to take the risk deliberately.
_TF_AFTER_TORCH = (
    "torch" in _sys.modules and "tensorflow" not in _sys.modules
    and _os.environ.get("REXGRAPH_ALLOW_TF_AFTER_TORCH", "") not in ("1", "true", "yes")
)
if _TF_AFTER_TORCH:
    _HAS_TF = False
    _Layer = object
    _TF_ERROR = (
        "torch is already imported and tensorflow is not. Importing TensorFlow now "
        "aborts the process (both bundle LLVM and the second to register its "
        "CommandLine options calls LLVM ERROR). Import tensorflow BEFORE torch, or "
        "run the two in separate processes. Set REXGRAPH_ALLOW_TF_AFTER_TORCH=1 to "
        "override.")
else:
    _TF_ERROR = "rexgraph.nn.tf_relational requires TensorFlow (optional dependency)."
    try:
        import tensorflow as _tf
        _HAS_TF = True
        _Layer = _tf.keras.layers.Layer
    except Exception:                                # pragma: no cover
        _HAS_TF = False
        _Layer = object


def _require():
    if not _HAS_TF:
        raise ImportError(_TF_ERROR)


def causal_windows(z, w):
    """`z` [B,H,T,d] -> [B,H,T,w,d] with out[...,i,m,:] = z[..., i-w+1+m, :].

    Left-pads the token axis and slices the band. tf has no `unfold`, so this uses
    `extract_patches` semantics via gather on a padded index grid, which is a
    materialised band rather than a view: the same O(T*w*d) the torch einsum path
    ends up paying, stated plainly rather than implied.
    """
    _require()
    shape = _tf.shape(z)
    _B, _H, T, _d = shape[0], shape[1], shape[2], shape[3]
    zp = _tf.pad(z, [[0, 0], [0, 0], [w - 1, 0], [0, 0]])          # [B,H,T+w-1,d]
    idx = _tf.range(T)[:, None] + _tf.range(w)[None, :]            # [T,w]
    return _tf.gather(zp, idx, axis=2)                             # [B,H,T,w,d]


def band_valid(T, w):
    """[T,w] bool: entry m of row i is a real token iff i - w + 1 + m >= 0."""
    _require()
    i = _tf.range(T)[:, None]
    m = _tf.range(w)[None, :]
    return (i - (w - 1) + m) >= 0


class CausalPropagatorAttention(_Layer):
    """Y = sum_k c_k A^k V on a banded causal token graph, in TensorFlow.

    `A` is the row-stochastic softmax over each token's window of prior tokens, so
    the graph is a DAG and A^k routes information k hops back. `hops` is how far to
    reach and the hop weights are learnable. The [B,H,T,T] object is never formed.

    Cross-checked against the torch implementation to 1e-5 in
    rexgraph/tests/test_tf_integration.py; that agreement is the reason to trust
    either of them, since the operator is a property of the complex and not of the
    framework it is written in.
    """

    def __init__(self, d: int, n_head: int, hops: int = 4, window: int = 64,
                 learn_hops: bool = True, **kw):
        _require()
        super().__init__(**kw)
        if d % n_head:
            raise ValueError(f"d={d} must divide by n_head={n_head}")
        if window is None or window < 1:
            raise ValueError("a banded propagator needs a window of at least 1")
        self.d, self.h, self.dk = d, n_head, d // n_head
        self.hops, self.window, self.learn_hops = hops, window, learn_hops

    def build(self, input_shape):
        self.qkv = _tf.keras.layers.Dense(3 * self.d, name="qkv")
        self.out = _tf.keras.layers.Dense(self.d, name="proj")
        self.log_c = (self.add_weight(name="log_c", shape=(self.hops + 1,),
                                      initializer="zeros", trainable=True)
                      if self.learn_hops else None)
        super().build(input_shape)

    def call(self, x):
        shape = _tf.shape(x)
        B, T = shape[0], shape[1]
        w = min(int(self.window), int(x.shape[1])) if x.shape[1] is not None \
            else int(self.window)
        q, k, v = _tf.split(self.qkv(x), 3, axis=-1)

        def heads(z):
            z = _tf.reshape(z, [B, T, self.h, self.dk])
            return _tf.transpose(z, [0, 2, 1, 3])                  # [B,H,T,dk]

        q, k, v = heads(q), heads(k), heads(v)
        kw = causal_windows(k, w)                                  # [B,H,T,w,dk]
        scores = _tf.einsum('bhtd,bhtwd->bhtw', q, kw) / math.sqrt(self.dk)
        valid = band_valid(T, w)                                   # [T,w]
        scores = _tf.where(valid, scores, _tf.fill(_tf.shape(scores), scores.dtype.min))
        A = _tf.nn.softmax(scores, axis=-1)                        # [B,H,T,w]

        c = (_tf.nn.softmax(self.log_c) if self.log_c is not None
             else _tf.fill([self.hops + 1], 1.0 / (self.hops + 1)))
        P = v
        acc = c[0] * P
        for kk in range(1, self.hops + 1):
            P = _tf.einsum('bhtw,bhtwd->bhtd', A, causal_windows(P, w))
            acc = acc + c[kk] * P
        acc = _tf.reshape(_tf.transpose(acc, [0, 2, 1, 3]), [B, T, self.d])
        return self.out(acc)


def from_torch(torch_module):
    """A TF layer carrying a torch `CausalPropagatorAttention`'s weights.

    Exists so the two can be compared on the SAME parameters rather than on two
    random initialisations, which would only ever show that both produce numbers.
    """
    _require()

    src = torch_module
    layer = CausalPropagatorAttention(src.h * src.dk, src.h, hops=src.hops,
                                      window=src.window,
                                      learn_hops=src.log_c is not None)
    d = src.h * src.dk
    # Keras 3 does not build sublayers from an outer build(), so run one forward on
    # a throwaway input to materialise the Dense kernels before assigning to them.
    layer(_tf.zeros([1, max(int(src.window or 1), 2), d]))
    # torch Linear stores [out, in]; keras Dense stores [in, out]
    layer.qkv.set_weights([src.qkv.weight.detach().numpy().T,
                           src.qkv.bias.detach().numpy()])
    layer.out.set_weights([src.proj.weight.detach().numpy().T,
                           src.proj.bias.detach().numpy()])
    if layer.log_c is not None:
        layer.log_c.assign(src.log_c.detach().numpy())
    return layer
