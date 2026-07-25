"""rexgraph.nn.layers - composable relational NN building blocks (nn.Module).

The pieces you compose into models, the way message-passing primitives compose in a graph library.
Each layer takes a fixed Laplacian operator L at forward time and learns the transforms around it,
built on the eigen-free rcf_torch propagators and the self-adjoint Green's solve:

    PropagatorMix   f(L)*(X W) via Chebyshev matvec (heat / wave), learnable scale t - matrix-free
    GreenResolvent  (I + a L)^-1 (X W), learnable a - the implicit self-adjoint layer (one operator,
                    O(1)-memory backward: forward and gradient are the same CG solve)
    RelationalBlock LayerNorm -> a mix -> nonlinearity, residual - stack these into a relational net

L is a torch tensor (heat/wave), or a tensor or a matvec callable (green). torch is optional
(rexgraph[nn]); the classes raise a clear error if it is missing.
"""
from __future__ import annotations

import math

try:
    import torch as _t
    import torch.nn as _nn
    _Module = _nn.Module
except Exception:                                            # torch is the optional [nn] extra
    _t = None

    class _Module:
        def __init__(self, *a, **k):
            raise ImportError("rexgraph.nn.layers needs PyTorch: pip install 'rexgraph[nn]'")

from . import rcf_torch as _R


def _matvec(L):
    """Adapt a Laplacian given as a tensor or a matvec callable to a matvec Y -> L*Y."""
    return L if callable(L) else (lambda v: L @ v)


class GreenResolvent(_Module):
    """Implicit self-adjoint layer: X' = (I + a*L)^-1 (X W), with a learnable.

    The resolvent is the converged (infinite-step) propagator, solved by CG; the backward pass is
    the same solve applied to the incoming gradient (one operator, O(1) memory, no unrolling). The
    relational analog of a residual / equilibrium block. L may be a tensor or a matvec callable."""

    def __init__(self, d_in, d_out, *, alpha_init: float = 1.0, tol: float = 1e-5, max_iter: int = 50):
        super().__init__()
        self.lin = _nn.Linear(d_in, d_out)
        self.log_alpha = _nn.Parameter(_t.tensor(math.log(alpha_init)))
        self.tol, self.max_iter = tol, max_iter

    def forward(self, X, L):
        return _R.green_resolvent(self.lin(X), self.log_alpha.exp(), _matvec(L),
                                  tol=self.tol, max_iter=self.max_iter)


class PropagatorMix(_Module):
    """Mix features by a learnable-scale propagator f(L)*(X W), matrix-free via Chebyshev matvec.

    channel 'heat' is diffusive (e^{-tL}); 'wave' is oscillatory (the real part of e^{-itL}). The
    n x n operator is never formed - this is the building block behind PropagatorAttention, usable
    on any complex. L is a torch tensor."""

    def __init__(self, d_in, d_out, *, channel: str = "heat", cheb_order: int = 16, t_init: float = 1.0):
        super().__init__()
        self.lin = _nn.Linear(d_in, d_out)
        self.log_t = _nn.Parameter(_t.tensor(math.log(t_init)))
        self.channel, self.K = channel, cheb_order

    def forward(self, X, L):
        Y = self.lin(X)
        t = self.log_t.exp()
        if self.channel == "wave":
            return _R.wave_apply(L, Y, t, K=self.K)[0]       # real part of e^{-itL}
        return _R.heat_apply(L, Y, t, K=self.K)


class RelationalBlock(_Module):
    """One composable relational layer: LayerNorm -> a propagator/resolvent mix -> GELU, residual.
    `op` is 'heat' | 'wave' | 'green'. Stack these to build a relational network on a fixed complex."""

    def __init__(self, d, *, op: str = "heat", cheb_order: int = 16):
        super().__init__()
        self.norm = _nn.LayerNorm(d)
        self.mix = GreenResolvent(d, d) if op == "green" else \
            PropagatorMix(d, d, channel=op, cheb_order=cheb_order)
        self.act = _nn.GELU()

    def forward(self, X, L):
        return X + self.act(self.mix(self.norm(X), L))
