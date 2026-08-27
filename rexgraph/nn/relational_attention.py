"""
relational_attention: attention as a propagator on a content-weighted token complex,
built on the eigen-free rcf_torch primitives.

Standard attention forms softmax(QKᵀ), a dense metric object, and uses it directly as the
one-hop mixing matrix. Propagator attention treats the tokens as a weighted graph (edge
weights = content affinity) and mixes values by a propagator f(L_W)·V computed with a
Chebyshev sparse-matvec recurrence; the n×n mixing operator f(L_W) is never formed
(O(nnz·K·d)). Two routing channels come from the light propagator's exact split:

  * heat  e^{-tL}·V  - diffusive / gradient routing (multi-hop reachability)
  * curl  Im(e^{-itL})·V = -sin(tL)·V - rotational / directional routing, which softmax
    cannot express in one hop ("complex rotation is curl")

The token graph is content-weighted, so the weighting's curvature (: weighted
degree / participation ratio) and the per-head varentropy self-diagnostic (collision-vs-diffusion gap) are computed readouts. `t` (propagation scale) is learnable; the
topology is multi-hop, gradient⊕curl.

v1 stores the T×T content-affinity like standard attention does, but the mixing operator is
matrix-free; the fully sparse (scatter, no T×T) path uses rcf_torch.cheb_apply_op, see the
cost study in agent.benchmarks. torch is optional; import guarded at use.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

try:
    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F
    _HAS_TORCH = True
    _Base = _nn.Module
except Exception:                                    # pragma: no cover
    _HAS_TORCH = False
    _Base = object


class PropagatorAttention(_Base):
    """Attention = f(L_W)·V on a content-weighted token graph, mixing channels ∈
    {heat, gradient, curl}. Reduces to a diffusive graph-attention at channels=('heat',),
    order 1; multi-hop/rotational at higher order and with the curl channel."""

    def __init__(self, d: int, n_head: int, cheb_order: int = 16,
                 channels: Sequence[str] = ("heat", "curl"), learn_time: bool = True,
                 init_time: float = 0.5, importance: bool = False):
        if not _HAS_TORCH:
            raise ImportError("PropagatorAttention requires PyTorch (optional dependency).")
        super().__init__()
        assert d % n_head == 0
        self.h = n_head; self.dk = d // n_head
        self.channels = tuple(channels)
        self.order = cheb_order
        self.importance = importance
        self.qkv = _nn.Linear(d, 3 * d)
        self.proj = _nn.Linear(d * len(self.channels), d)
        self.init_time = init_time
        self.log_t = _nn.Parameter(_t.full((), math.log(init_time))) if learn_time else None
        # learnable strength of the importance gate (0 = off); starts mild
        self.imp_gain = _nn.Parameter(_t.tensor(0.5)) if importance else None

    def _laplacian(self, q, k):
        """Weighted token Laplacian L_W = D - W from content affinity W = softplus(QKᵀ/√dk),
        symmetrized (PSD), zero diagonal. Returns (L, W, deg)."""
        S = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)      # [B,H,T,T] affinity scores
        W = _F.softplus(S)
        W = 0.5 * (W + W.transpose(-2, -1))                     # symmetric -> PSD Laplacian
        W = W - _t.diag_embed(_t.diagonal(W, dim1=-2, dim2=-1))  # no self-edge
        deg = W.sum(-1)                                          # [B,H,T]
        L = _t.diag_embed(deg) - W
        return L, W, deg

    def forward(self, x, return_diag: bool = False):
        from rexgraph.nn import rcf_torch as R
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)       # [B,H,T,dk]
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        L, W, deg = self._laplacian(q, k)
        lam_max = R.spectral_bound(L)
        t = self.log_t.exp() if self.log_t is not None else self.init_time

        # importance gate (opt-in, default off): per-token structural centrality (2-hop vs
        # 1-hop reach). Finding (associative-recall bench): this gate is inert (Δ≈0), because
        # the importance the task needs is already intrinsic to the propagator's multi-hop
        # routing (propagator solves recall 1.0 vs standard 0.55). Kept opt-in in case a task
        # rewards explicit centrality; not the default.
        imp = None
        if self.importance:
            r1 = deg                                          # 1-hop weighted reach [B,H,T]
            r2 = (W @ r1.unsqueeze(-1)).squeeze(-1)           # 2-hop reach
            imp = r2 / (r1 + 1e-6)                             # reach divergence (centrality)
            imp = imp / imp.mean(dim=-1, keepdim=True).clamp_min(1e-6)
            gate = 1.0 + _F.softplus(self.imp_gain) * (imp - 1.0)   # gate≈1 at init, learns up
            v = v * gate.unsqueeze(-1)                         # important source tokens broadcast more

        chans = []
        need_wave = ("gradient" in self.channels) or ("curl" in self.channels)
        wave_re = wave_im = None
        if need_wave:
            wave_re, wave_im = R.wave_apply(L, v, t, K=self.order, lam_max=lam_max)
        for ch in self.channels:
            if ch == "heat":
                y = R.heat_apply(L, v, t, K=self.order, lam_max=lam_max)
            elif ch == "gradient":
                y = wave_re
            elif ch == "curl":
                y = wave_im
            else:
                raise ValueError(f"unknown channel {ch!r}")
            chans.append(y.transpose(1, 2).reshape(B, T, d))    # [B,T,d] per channel
        out = self.proj(_t.cat(chans, dim=-1))

        if not return_diag:
            return out, None
        with _t.no_grad():
            #: per-head varentropy gap (collision vs diffusion) - routing structure
            vg = R.varentropy_gap(L)
            #: weight concentration = participation ratio N_eff = (Σw)²/Σw² per head
            wsum = W.sum(dim=(-2, -1)); w2sum = (W * W).sum(dim=(-2, -1))
            n_eff = (wsum * wsum) / w2sum.clamp_min(1e-12)
            diag = {
                "renyi2": vg["renyi2"].mean().item(),
                "varentropy_gap": vg["gap"].mean().item(),   # small->flat/diffuse, large->structured
                "weight_participation": n_eff.mean().item(),  # curvature/concentration of routing
                "heat_time": float(t),
            }
        return out, diag


def _causal_windows(z, w):
    """`z` [B,H,T,d] -> [B,H,T,w,d] with out[...,i,m,:] = z[..., i-w+1+m, :].

    A strided view of the left-padded tensor, so the band is addressed rather than
    built. Rows with i < w-1 read padding, which the score mask removes.
    """
    zp = _F.pad(z, (0, 0, w - 1, 0))                 # pad the T axis on the left
    return zp.unfold(2, w, 1).permute(0, 1, 2, 4, 3)


def _band_valid(T, w, device):
    """[T,w] bool: entry m of row i is a real token iff i - w + 1 + m >= 0."""
    m = _t.arange(w, device=device)
    i = _t.arange(T, device=device)
    return (i[:, None] - (w - 1) + m[None, :]) >= 0


class CausalPropagatorAttention(_Base):
    """Causal relational attention: the decoder-LM form, where causality, sparsity, and
    multi-hop propagation are one structural choice.

    A causal, windowed prior-token neighborhood is simultaneously the causal mask and the
    O(n·w) sparse graph. Because a causal token graph is a DAG, its (row-stochastic) adjacency
    A is nilpotent-under-truncation, so the propagator is a finite matvec series

        Y = Σ_{k=0}^{K} c_k · Aᵏ · V            (A lower-triangular => Y[i] depends only on j≤i)

    no complex spectrum and no convergence tower. Aᵏ routes information k hops back along
    the causal graph; the hop weights c = softmax(·) are learnable (how far to reach),
    and k=0 keeps the token's own value. This is the causal counterpart of the heat
    propagator (real/gradient); a directional (curl) channel can be added later via a
    signed hop combination.

    TWO PATHS, and `sparse` chooses. The dense one forms [B,H,T,T] and masks it, so
    `window` changes which entries are masked rather than how many are computed and
    a full forward is strictly more work than standard attention. The banded one
    addresses the window directly and never builds the T×T; it is selected by
    default whenever a window is given.

    The band is not an approximation. The window is a structural fact about which
    tokens are reachable, not a threshold on scores, so the band drops exactly the
    terms the dense path multiplies by zero. Outputs agree to 0.0e+00 and gradients
    to 1e-5 across every shape tested, including T < w; see
    rexgraph/tests/test_banded_attention.py, which is the reason to believe it.

    COST, measured (agent.benchmarks.bench_attention_speed), 8060S, B=4 d=256 h=4,
    hops=4, window=64, prefill, against torch's fused causal SDPA:

        T              512    1024    2048    4096    8192
        banded/fused  2.51x   1.50x   1.13x   1.00x   0.52x
        banded/dense  1.75x   0.74x   0.63x   0.63x   0.33x
        banded MB       185     335     635    1235    2435
        dense MB         90     243     838    3186   12514

    so the band overtakes the dense propagator around T = 1024 and torch's own
    kernel around T = 4096, and by T = 8192 it is roughly twice as fast as fused
    SDPA on a fifth of the memory. A wider window costs proportionally: w = 128 is
    slower than fused at every length tested, since the einsum materialises the
    [B,H,T,w,dk] window rather than streaming it, which is O(T·w·dk) and not the
    O(T·dk) a fully streamed kernel would use.

    Where the shape does pay is DECODE, one token against a KV cache, which is what
    token/s measures: the window bounds the read and the hops buy back the reach it
    gave up, at a cost proportional to K. Same benchmark, B=8 d=512 h=8, window 64,
    against full-history decode:

        cache T    256    1024    4096    8192
        K=4       0.18x   0.53x   2.13x   3.91x   (>1 is faster)

    so it is a loss below roughly T = 2000 and a win above it. Prefill with this class
    is a loss at every length tested."""

    def __init__(self, d: int, n_head: int, hops: int = 4, window: int = None,
                 learn_hops: bool = True, sparse: bool | None = None):
        if not _HAS_TORCH:
            raise ImportError("CausalPropagatorAttention requires PyTorch (optional dependency).")
        super().__init__()
        assert d % n_head == 0
        self.h = n_head; self.dk = d // n_head
        self.hops = hops
        self.window = window                                  # None = full causal history
        # The band IS the causal window, so the sparse path is the same operator
        # addressed differently and not an approximation: no threshold, no dropped
        # term, no spectrum. It needs a window to be a band at all, so with
        # window=None it is off and the dense path runs.
        self.sparse = (window is not None) if sparse is None else bool(sparse)
        if self.sparse and window is None:
            raise ValueError("sparse=True needs a window: without one the band is "
                             "the full history and there is nothing to save.")
        self.qkv = _nn.Linear(d, 3 * d)
        self.proj = _nn.Linear(d, d)
        self.log_c = _nn.Parameter(_t.zeros(hops + 1)) if learn_hops else None

    def _forward_banded(self, q, k, v, B, T, d, return_diag):
        """The O(T*w*K*dk) path: the [B,H,T,T] is never built.

        Identical arithmetic to the dense branch, not an approximation of it. The
        dense branch masks every entry outside the window to -inf before the
        softmax, so those terms contribute exactly zero; addressing the band
        directly drops the same zeros instead of computing them. Verified against
        the dense branch to float tolerance in the tests, which is the only reason
        to believe this rather than the shape of the code.
        """
        w = min(int(self.window), T)
        valid = _band_valid(T, w, q.device)                       # [T,w]
        kw = _causal_windows(k, w)                                # [B,H,T,w,dk] view
        scores = _t.einsum('bhtd,bhtwd->bhtw', q, kw) / math.sqrt(self.dk)
        A = scores.masked_fill(~valid, float("-inf")).softmax(dim=-1)   # [B,H,T,w]
        c = (self.log_c.softmax(0) if self.log_c is not None
             else _t.full((self.hops + 1,), 1.0 / (self.hops + 1), device=q.device))
        P = v
        acc = c[0] * P
        for kk in range(1, self.hops + 1):
            P = _t.einsum('bhtw,bhtwd->bhtd', A, _causal_windows(P, w))
            acc = acc + c[kk] * P
        out = self.proj(acc.transpose(1, 2).reshape(B, T, d))
        if not return_diag:
            return out, None
        with _t.no_grad():
            diag = {"hop_weights": (self.log_c.softmax(0).tolist()
                                    if self.log_c is not None else None),
                    "band": w, "path": "banded"}
        return out, diag

    def forward_dense(self, x, return_diag: bool = False):
        """The dense branch, kept addressable so the sparse one can be checked
        against it rather than trusted."""
        was, self.sparse = self.sparse, False
        try:
            return self.forward(x, return_diag)
        finally:
            self.sparse = was

    def forward(self, x, return_diag: bool = False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)     # [B,H,T,dk]
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        if self.sparse:
            return self._forward_banded(q, k, v, B, T, d, return_diag)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)     # [B,H,T,T]
        i = _t.arange(T, device=x.device)
        causal = i[:, None] >= i[None, :]                     # j ≤ i (lower-triangular)
        if self.window is not None:
            causal = causal & (i[:, None] - i[None, :] < self.window)
        A = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)   # row-stochastic DAG
        c = (self.log_c.softmax(0) if self.log_c is not None
             else _t.full((self.hops + 1,), 1.0 / (self.hops + 1), device=x.device))
        # finite causal propagator series Y = Σ_k c_k Aᵏ V (K matvecs, causality preserved)
        P = v
        acc = c[0] * P
        for kk in range(1, self.hops + 1):
            P = A @ P
            acc = acc + c[kk] * P
        out = self.proj(acc.transpose(1, 2).reshape(B, T, d))
        if not return_diag:
            return out, None
        with _t.no_grad():
            diag = {"hop_weights": self.log_c.softmax(0).tolist() if self.log_c is not None else None}
        return out, diag
