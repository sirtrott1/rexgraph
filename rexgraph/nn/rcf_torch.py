"""
rcf_torch: differentiable, eigen free RCF primitives in torch.

Shared by relational attention and the Hodge optimizer. Every primitive here is the sparse /
integer / matrix free form, not a dense eigensolve. The governing rule: dense matrices and full
eigendecompositions are an implementation choice, not the math; keep operators as matvecs and
scalars as traces, and never form an n×n object whose zeros are the topology.

What lives here (script -> primitive):
  * [13] propagator f(L)·X via Chebyshev sparse matvec recurrence: heat e^{-tL}
    (gradient/diffusive), wave e^{-itL} split into real=gradient(cos) / imag=curl(sin).
    O(nnz·K·d), spectrum never formed. Differentiable (matvecs), incl. w.r.t. the time t.
  * [14] energy character diag(L²) = row norms ‖L[e,:]‖², the short time propagator
    moment; O(nnz), no inversion.
  * [15] scale moments (L^k)_vv (closed k-walks): local<->global, sparse matvec.
  * [09,10] combinatorial harmonic basis (spanning tree fundamental cycles, integer ±1,
    B₁H=0) + exact low-rank projector H(HᵀH)⁻¹Hᵀ, no eigensolve.
  * [18,19] harmonic log = Rényi-2 (collision) entropy via traces; varentropy gap
    (H₁-H₂) = curvature self-diagnostic. Eigen-free.
  * [20] weighted curvature = chain residual R = B₁(W-I)B₂; additive edge decomposition.

`spectral_bound` is a Gershgorin upper bound so the Chebyshev rescale stays eigen free. torch
is an optional dep, import guarded.
"""
from __future__ import annotations

import math
from collections.abc import Callable

try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:                                    # pragma: no cover
    _HAS_TORCH = False


def _require():
    if not _HAS_TORCH:
        raise ImportError("rcf_torch requires PyTorch (optional dependency).")


# propagator: f(L)·X, eigen free [13]

def spectral_bound(L) -> float:
    """Cheap eigen free upper bound on λ_max via Gershgorin (max absolute row sum). Keeps
    the Chebyshev rescale from needing an eigensolve."""
    _require()
    if L.numel() == 0:
        return 0.0
    if L.layout != _torch.strided:
        # Reduce only stored entries. Sparse max is unsupported, and converting
        # the full operator to dense would defeat the sparse propagator contract.
        coo = L.to_sparse_coo().coalesce()
        rows = _torch.zeros(L.shape[-2], device=L.device, dtype=L.dtype)
        rows.scatter_add_(0, coo.indices()[0], coo.values().abs())
        return float(rows.max().item())
    return float(L.abs().sum(dim=-1).max().item())


def cheb_coeffs(func: Callable, K: int, lam_max: float, *, device=None, dtype=None):
    """Chebyshev coefficients of a scalar spectral function ``func`` on [0, lam_max], via
    sampling at Chebyshev nodes + DCT. ``func`` maps a torch
    tensor of eigen samples -> values; it may depend on a learnable parameter (e.g. heat
    time t): the coefficients stay differentiable through it."""
    _require()
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise ValueError("Chebyshev coefficient count must be a positive integer")
    t = _torch
    dtype = dtype or t.get_default_dtype()
    j = t.arange(K, device=device, dtype=dtype)
    xs = t.cos(math.pi * (j + 0.5) / K)              # Chebyshev nodes in [-1,1]
    lam = (xs + 1.0) * (lam_max / 2.0)               # mapped to [0, lam_max]
    fvals = func(lam)                                 # differentiable in any params of func
    k = t.arange(K, device=device, dtype=dtype)
    cos_mat = t.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / K)   # [K,K]
    c = (2.0 / K) * (cos_mat @ fvals)
    c = c.clone(); c[0] = c[0] / 2.0
    return c


def cheb_apply(L, X, coeffs, lam_max: float | None = None):
    """Apply Σ_k c_k T_k(L̃) to X, where L̃ = 2L/λ_max - I rescales the spectrum to [-1,1].
    Pure matvec recurrence T_{k+1} = 2L̃T_k - T_{k-1} - O(nnz·K·d), never forms f(L). L may
    be a dense or sparse torch tensor; X is [..., n, d] (or [n, d]). Differentiable."""
    _require()
    if coeffs.ndim != 1 or coeffs.numel() == 0:
        raise ValueError("Chebyshev coefficients must be a nonempty vector")
    if coeffs.numel() == 1:
        return coeffs[0] * X
    if lam_max is None:
        lam_max = spectral_bound(L)
    return cheb_apply_op(lambda Y: L @ Y, X, coeffs, lam_max)


def cheb_apply_op(matvec: Callable, X, coeffs, lam_max: float):
    """Matrix free Chebyshev: apply Σ_k c_k T_k(L̃) to X where L is given only as a linear
    operator ``matvec: Y -> L·Y`` (e.g. a weighted graph Laplacian applied by edge scatter,
    never materialized). L̃ = 2L/λ_max - I. Keeps attention at O(nnz·K·d) with no n×n object.
    Differentiable through ``matvec`` and ``coeffs``."""
    _require()
    if coeffs.ndim != 1 or coeffs.numel() == 0:
        raise ValueError("Chebyshev coefficients must be a nonempty vector")
    if coeffs.numel() == 1:
        return coeffs[0] * X
    if not math.isfinite(lam_max) or lam_max <= 0:
        raise ValueError("Chebyshev recurrence requires a finite positive spectral bound")
    two_over = 2.0 / lam_max

    def Lt(Y):
        return two_over * matvec(Y) - Y

    T_prev = X
    T_cur = Lt(X)
    acc = coeffs[0] * T_prev + coeffs[1] * T_cur
    for kk in range(2, coeffs.shape[0]):
        T_next = 2.0 * Lt(T_cur) - T_prev
        acc = acc + coeffs[kk] * T_next
        T_prev, T_cur = T_cur, T_next
    return acc


def _cg_solve(A, b, tol: float, max_iter: int):
    """Matrix free CG solve of A y = b (A symmetric PD as a matvec). Returns (y, n_iters). The
    iteration count is dynamic: it stops at the tolerance, not a fixed number of steps."""
    x = _torch.zeros_like(b)
    r = b - A(x); p = r.clone(); rs = (r * r).sum()
    bn = (b * b).sum().clamp_min(1e-30)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = A(p); alpha = rs / ((p * Ap).sum() + 1e-30)
        x = x + alpha * p; r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new / bn < tol * tol:
            break
        p = r + (rs_new / (rs + 1e-30)) * p; rs = rs_new
    return x, it


if _HAS_TORCH:

    class _GreenResolvent(_torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha, matvec_L, tol, max_iter):
            A = lambda v: v + alpha * matvec_L(v)                  # (I + α·L), symmetric PD
            y, iters = _cg_solve(A, x, tol, max_iter)
            ctx.save_for_backward(y, alpha)
            ctx.matvec_L = matvec_L; ctx.tol = tol; ctx.max_iter = max_iter; ctx.iters = iters
            return y

        @staticmethod
        def backward(ctx, grad_y):
            y, alpha = ctx.saved_tensors
            A = lambda v: v + alpha * ctx.matvec_L(v)
            # self adjoint: A symmetric => (A⁻¹)ᵀ = A⁻¹, so the gradient flows through the same
            # solve as the forward; forward and backward are one operator.
            grad_x, _ = _cg_solve(A, grad_y, ctx.tol, ctx.max_iter)
            grad_alpha = None
            if ctx.needs_input_grad[1]:                            # dL/dα = -(grad_x · L y)
                grad_alpha = -(grad_x * ctx.matvec_L(y)).sum().reshape(alpha.shape)
            return grad_x, grad_alpha, None, None, None


def green_resolvent(x, alpha, matvec_L, tol: float = 1e-5, max_iter: int = 50):
    """Green's-function (implicit resolvent) layer: y = (I + α·L)⁻¹ x, solved matrix free by CG.
    The equilibrium of the diffusion; one solve captures all propagation depth, so there are no
    hardcoded hops. Because (I + α·L) is symmetric PD, the adjoint (backward) is the same solve
    (self adjoint): forward and gradient flow through one operator. ``matvec_L`` is the Hodge/graph
    operator applied matrix free; ``alpha`` is a differentiable scalar."""
    _require()
    return _GreenResolvent.apply(x, alpha, matvec_L, tol, max_iter)


def propagator_apply(L, X, t: float, channels=("heat", "gradient", "curl"),
                     K: int = 32, lam_max: float | None = None) -> tuple:
    """Apply selected heat/cos/-sin filters using ONE shared Chebyshev basis.

    Each channel has the same polynomial and accuracy as its independent call.
    Sharing T_k(L)X avoids repeated operator products and their backward graphs;
    it never forms a mixing matrix or caches a content dependent operator.
    Returns a tuple in the requested channel order, including repetitions.
    """
    _require()
    channels = tuple(channels)
    functions = {"heat": lambda l: _torch.exp(-t * l),
                 "gradient": lambda l: _torch.cos(t * l),
                 "curl": lambda l: -_torch.sin(t * l)}
    if not channels or any(ch not in functions for ch in channels):
        raise ValueError("propagator channels must be a nonempty selection of heat, gradient, curl")
    lam_max = lam_max if lam_max is not None else spectral_bound(L)
    if lam_max == 0:
        if not L.requires_grad:
            return tuple(X * 0 + 0 * t if ch == "curl" else X + 0 * t for ch in channels)
        # A learnable zero operator still has a nonzero derivative with respect
        # to L. A positive upper bound preserves the differentiable recurrence.
        lam_max = 1.0
    coeffs = [cheb_coeffs(functions[ch], K, lam_max, device=X.device, dtype=X.dtype)
              for ch in channels]
    previous = X
    out = [c[0] * previous for c in coeffs]
    if K == 1:
        return tuple(out)
    if not math.isfinite(lam_max) or lam_max <= 0:
        raise ValueError("Chebyshev recurrence requires a finite positive spectral bound")
    two_over = 2.0 / lam_max
    current = two_over * (L @ X) - X
    out = [y + c[1] * current for y, c in zip(out, coeffs, strict=True)]
    for k in range(2, K):
        following = 2.0 * (two_over * (L @ current) - current) - previous
        out = [y + c[k] * following for y, c in zip(out, coeffs, strict=True)]
        previous, current = current, following
    return tuple(out)


def heat_apply(L, X, t: float, K: int = 32, lam_max: float | None = None):
    """Heat propagator e^{-tL} applied to X (diffusive / gradient routing), eigen free."""
    return propagator_apply(L, X, t, ("heat",), K, lam_max)[0]


def wave_apply(L, X, t: float, K: int = 32, lam_max: float | None = None) -> tuple:
    """Light/wave propagator e^{-itL} applied to X, returned as (real, imag) =
    (gradient/cos, curl/sin) channels [13]. The imag channel is the directional/rotational
    routing component."""
    return propagator_apply(L, X, t, ("gradient", "curl"), K, lam_max)


# energy character & scale moments [14,15]

def energy_character(L):
    """diag(L²) as row norms ‖L[e,:]‖²: O(nnz), no inversion. The short time (t²) moment of
    the heat propagator: the local character [14]."""
    _require()
    return (L * L).sum(dim=-1)


def scale_moments(L, k_max: int):
    """Closed walk scale moments (L^k)_vv for k=0..k_max, via sparse matvec on the identity
    columns: structure at scale k (girth) [15]. Returns [k_max+1, n]."""
    _require()
    n = L.shape[-1]
    E = _torch.eye(n, device=L.device, dtype=L.dtype)
    V = E
    out = [(_torch.ones(n, device=L.device, dtype=L.dtype))]   # (L^0)_vv = 1
    for _ in range(1, k_max + 1):
        V = L @ V
        out.append((E * V).sum(dim=0))
    return _torch.stack(out, dim=0)


# combinatorial harmonic basis + exact projector [09,10]

def spanning_tree_cycles(sources, targets, nV: int):
    """Fundamental cycles of a spanning tree = the combinatorial harmonic basis: integer ±1
    vectors H ∈ {-1,0,+1}^{nE×β₁} with B₁H = 0 by construction, no eigensolve [10]. Inputs
    are edge endpoint arrays (sparse native). Returns H as a torch tensor (nE × β₁)."""
    _require()
    src = [int(x) for x in sources]; tgt = [int(x) for x in targets]
    nE = len(src)
    parent = list(range(nV))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    adj = {v: [] for v in range(nV)}
    tree, nontree = [], []
    for e in range(nE):
        i, j = src[e], tgt[e]
        if find(i) != find(j):
            parent[find(i)] = find(j); tree.append(e)
            adj[i].append((j, e)); adj[j].append((i, e))
        else:
            nontree.append(e)

    def tree_path(a, b):                              # path a->b through tree edges
        prev = {a: None}; stack = [a]
        while stack:
            u = stack.pop()
            for v, e in adj[u]:
                if v not in prev:
                    prev[v] = (u, e); stack.append(v)
        out = []; cur = b
        while prev[cur] is not None:
            u, e = prev[cur]; out.append((u, cur, e)); cur = u
        return out

    H = _torch.zeros((nE, len(nontree)), dtype=_torch.get_default_dtype())
    for c, e in enumerate(nontree):
        i, j = src[e], tgt[e]
        H[e, c] = 1.0
        for (u, v, te) in tree_path(j, i):
            ti, tj = src[te], tgt[te]
            H[te, c] += 1.0 if (ti, tj) == (u, v) else -1.0
    return H


def harmonic_projector_apply(H, z):
    """Apply the exact low rank harmonic projector P_H = H(HᵀH)⁻¹Hᵀ to z without forming an
    nE×nE matrix: three small products, invert the β₁×β₁ Gram [10]. Lands z in the cycle
    space (B₁ P_H z = 0)."""
    _require()
    Gram = H.transpose(-2, -1) @ H                    # β₁×β₁, small (integer if H integer)
    rhs = H.transpose(-2, -1) @ z
    sol = _torch.linalg.solve(Gram, rhs)
    return H @ sol


# harmonic log (Rényi-2) & varentropy [18,19]

def renyi2(L, eps: float = 1e-12) -> float:
    """Harmonic log = Rényi-2 (collision) entropy of the normalized spectrum, eigen free:
    H₂ = -log( tr(L²)/tr(L)² ) [18]. tr(L²) via the row norm sum (no matrix square formed)."""
    _require()
    tr = _torch.diagonal(L, dim1=-2, dim2=-1).sum(-1)
    safe_trace = _torch.where(tr > 0, tr, _torch.ones_like(tr))
    normalized = L / safe_trace[..., None, None]
    tr2 = energy_character(normalized).sum(-1)
    return _torch.where(tr > 0, -_torch.log(tr2.clamp_min(eps)), _torch.zeros_like(tr))


def renyi_order(L, a: int, eps: float = 1e-12):
    """Integer order Rényi entropy H_a = (1-a)⁻¹ log( tr(Lᵃ)/tr(L)ᵃ ), eigen free via a-1
    matvecs [19]."""
    _require()
    if isinstance(a, bool) or not isinstance(a, int) or a < 2:
        raise ValueError("integer Renyi order must be at least two")
    tr = _torch.diagonal(L, dim1=-2, dim2=-1).sum(-1)
    safe_trace = _torch.where(tr > 0, tr, _torch.ones_like(tr))
    normalized = L / safe_trace[..., None, None]
    P = normalized
    for _ in range(a - 1):
        P = P @ normalized
    trA = _torch.diagonal(P, dim1=-2, dim2=-1).sum(-1)
    entropy = (1.0 / (1 - a)) * _torch.log(trA.clamp_min(eps))
    return _torch.where(tr > 0, entropy, _torch.zeros_like(tr))


def varentropy_gap(L):
    """Curvature self diagnostic: the collision->diffusion gap approximated from integer order
    Rényi moments H₂,H₃ (extrapolating toward Shannon). Small => near flat spectrum, Rényi-2
    trustworthy; large => weight structure the 2nd moment misses [19]. Returns H₂ and the
    (H₃-based) gap estimate."""
    _require()
    h2 = renyi2(L); h3 = renyi_order(L, 3)
    return {"renyi2": h2, "gap": (h2 - h3).abs()}      # |H2-H3|: 0 on flat, grows with weight


# weighted curvature [20]

def chain_residual(B1, B2, w):
    """Weighted tower curvature = the chain residual R = B₁(W-I)B₂ [20]. R=0 iff w uniform;
    nonzero R is the curvature (deviation from the unweighted ∂²=0 ideal). Sparse."""
    _require()
    Wm1 = _torch.diag(w - 1.0)
    return B1 @ Wm1 @ B2
