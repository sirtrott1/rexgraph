"""rexgraph.field_propagator: matrix-free evolution of the coupled (edge, face)
field on the graded vector space C_1 (+) C_2.

The field operator

    M = [[ RL1,     -g B2 ],
         [ -g B2ᵀ,   L2   ]]

acts on a GRADED VECTOR SPACE: the edge block C_1 stacked with the face block C_2.
A field here is not merely a vector: a block ``F`` of shape ``(nE+nF, m)`` carries a
TENSOR SHAPE (m components), and the boundary weighting W supplies a TENSOR METRIC
(the inner product on the graded space). So the field IS a dynamic tensor object -
static structure = the graded operator M itself, dynamic evolution = a matrix
function of M applied to the field.

Evolution is matrix-free, via a Chebyshev polynomial of M (assembled SPARSE, never
the dense (nE+nF)^2 matrix):

    heat   e^{-tM} F        diffusion on the graded field (M is PSD)
    wave   cos(t sqrt(M)) F oscillation at omega_k = sqrt(lambda_k)

O(nnz.order) spmv/spmm, ANY t, NO eigendecomposition. A block field propagates as an
spmm - the exact shape the multi-core / GPU backend batches over - so a whole tensor
field evolves at once. The dense ``core._field`` spectral evolvers remain as the
oracle these are parity-checked against.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from rexgraph import scale_propagator as _spg

_f64 = np.float64

__all__ = [
    "assemble_field_operator",
    "field_coupling",
    "field_heat",
    "field_heat_trajectory",
    "field_wave",
    "field_wave_trajectory",
]


def field_coupling(rex):
    """The coupling g = 1/max(||B2||_F, 1) (matches core._field / graph.field_coupling_psd)."""
    from rexgraph.core._sparse import to_scipy_csr
    if int(rex.nF_hodge) == 0 or rex._B2_hodge_dual is None:
        return 1.0
    B2 = to_scipy_csr(rex._B2_hodge_dual).tocsr()
    b2f = float(np.sqrt(B2.multiply(B2).sum()))
    return 1.0 / (b2f if b2f > 1.0 else 1.0)


def _field_blocks(rex, g=None):
    """(RL1, L2, B2, g, nE, nF) as scipy CSR - the sparse pieces of M. RL1 is the
    relational Laplacian if built, else L1 (same fallback as graph.field_coupling_psd)."""
    from rexgraph.core._sparse import to_scipy_csr
    nE = int(rex.nE)
    nF = int(rex.nF_hodge)
    if g is None:
        g = field_coupling(rex)
    RL1 = rex.relational_laplacian
    if RL1 is not None:
        RL1 = sp.csr_matrix(np.asarray(RL1)) if not sp.issparse(RL1) else RL1.tocsr()
    else:
        RL1 = rex.L1_sparse.tocsr()
    if nF > 0 and rex._B2_hodge_dual is not None:
        B2 = to_scipy_csr(rex._B2_hodge_dual).tocsr()          # nE x nF
        L2 = rex.L2_sparse.tocsr()
    else:
        B2 = sp.csr_matrix((nE, 0), dtype=_f64)
        L2 = sp.csr_matrix((0, 0), dtype=_f64)
    return RL1, L2, B2, float(g), nE, nF


def assemble_field_operator(rex, g=None):
    """The field operator M as a SPARSE (nE+nF) x (nE+nF) CSR block matrix - O(nnz),
    never the dense form. Symmetric PSD (the graded Hodge-coupled operator)."""
    RL1, L2, B2, g, nE, nF = _field_blocks(rex, g)
    if nF == 0:
        return RL1.tocsr()
    return sp.bmat([[RL1, (-g) * B2], [(-g) * B2.T, L2]], format="csr")


def _as_field_block(F, nE, nF):
    """Accept an edge-only signal (length nE) or a full graded state (length nE+nF),
    as a vector or a block; pad edge-only inputs with zero face components."""
    F = np.asarray(F, dtype=_f64)
    N = nE + nF
    if F.shape[0] == N:
        return F
    if F.shape[0] == nE and nF > 0:                            # edge signal -> graded
        pad = (nF,) + F.shape[1:]
        return np.concatenate([F, np.zeros(pad, dtype=_f64)], axis=0)
    raise ValueError(f"field state has length {F.shape[0]}, expected {nE} or {N}")


def field_metric(rex, W=None):
    """The TENSOR METRIC W on the graded space C1(+)C2 - the weighted inner product
    that makes the field a graded vector space with a metric. Returns one of:
      ('identity', None)  - no weighting (an unweighted complex): field evolution is
                            e^{-tM} unchanged;
      ('diag', d)         - a diagonal metric W = diag(d) (the default sqrt-w boundary
                            weights: edge weights on C1, uniform on C2, reducing to
                            identity when unweighted; or a caller-supplied 1D W);
      ('full', Wdense)    - a general SPD metric tensor (caller-supplied 2D/sparse).
    The metric enters via W-symmetric conjugation (see :func:`_apply_with_metric`)."""
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    N = nE + nF
    if W is None:
        w_E = getattr(rex, "w_E", None)
        if w_E is None:
            return ("identity", None)
        w_E = np.abs(np.asarray(w_E, dtype=_f64).ravel())
        if w_E.size != nE or np.allclose(w_E, 1.0):
            return ("identity", None)
        d = np.ones(N, dtype=_f64)
        d[:nE] = w_E                                    # face weights default to 1
        return ("diag", d)
    if sp.issparse(W):
        if W.shape == (1, N) or W.shape == (N, 1):
            return ("diag", np.asarray(W.todense(), dtype=_f64).ravel())
        return ("full", W)
    W = np.asarray(W, dtype=_f64)
    if W.ndim == 1:
        return ("diag", W)
    return ("full", W)


def _diag_conjugate(M, d):
    """Symmetric conjugate S = D^{-1/2} M D^{-1/2} (sparse) plus the scale vectors
    ds = sqrt(d), dsi = 1/sqrt(d) for a diagonal metric W = diag(d)."""
    ds = np.sqrt(d)
    dsi = 1.0 / ds
    Di = sp.diags(dsi)
    S = (Di @ _spg._csr(M) @ Di).tocsr()
    return S, ds, dsi


def _apply_with_metric(M, Fb, metric, apply_fn, out_naxis=0):
    """Apply a matrix function of the W-metric generator W^{-1}M to Fb via the
    symmetric conjugate S, so a symmetric-operator primitive `apply_fn(S, X)` (heat /
    wave Chebyshev) can be used:
        f(W^{-1}M) Fb = W^{-1/2} apply_fn(S, W^{1/2} Fb)   [diagonal W]
                      = L^{-T}   apply_fn(S, L^T   Fb)     [full W = L L^T].
    `out_naxis` is the axis carrying the N (graded-dim) index in apply_fn's output
    (0 for a single apply, 1 for a trajectory of shape (T, N, ...))."""
    kind, data = metric
    if kind == "identity":
        return apply_fn(M, Fb)
    if kind == "diag":
        S, ds, dsi = _diag_conjugate(M, data)
        in_shp = (-1,) + (1,) * (Fb.ndim - 1)           # N on axis 0 of the input
        Y = apply_fn(S, ds.reshape(in_shp) * Fb)
        out_shp = tuple(-1 if ax == out_naxis else 1 for ax in range(Y.ndim))
        return dsi.reshape(out_shp) * Y
    # full SPD metric: Cholesky W = L L^T, S = L^{-1} M L^{-T} (dense; advanced override)
    Wd = np.asarray(data.todense()) if sp.issparse(data) else np.asarray(data, dtype=_f64)
    L = np.linalg.cholesky(Wd)
    Md = np.asarray(_spg._csr(M).todense())
    S = np.linalg.solve(L, np.linalg.solve(L, Md).T).T   # L^{-1} M L^{-T}
    Y = apply_fn(sp.csr_matrix(S), L.T @ Fb)             # apply_fn(S, L^T Fb)
    if out_naxis == 0:
        return np.linalg.solve(L.T, Y)                   # L^{-T} Y
    raise NotImplementedError("full-metric trajectories: use per-t field_heat/field_wave")


def _wave_order(t, lam_max, given):
    # cos(t*sqrt(lambda)) oscillates in sqrt(lambda); resolve it against t*sqrt(lam_max)
    if given is not None:
        return int(given)
    return int(max(24, min(800, 2.0 * float(t) * np.sqrt(lam_max) + 24)))


def field_heat(rex, F, t, g=None, order=None, M=None, W=None):
    """Heat evolution ``e^{-t W^{-1}M} F`` of the graded field under the tensor metric
    W, matrix-free Chebyshev on the SPARSE M (via the W-symmetric conjugate), any t,
    no eigendecomposition. W defaults to the sqrt-w boundary weights (identity when the
    complex is unweighted, so this reduces to e^{-tM}); override with a 1D diagonal or
    a full SPD metric. F may be an edge signal (nE), a graded state (nE+nF), or a
    block/tensor field (..., m). Returns the propagated field, graded-input shape."""
    if M is None:
        M = assemble_field_operator(rex, g)
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    Fb = _as_field_block(F, nE, nF)
    return _apply_with_metric(
        M, Fb, field_metric(rex, W),
        lambda S, X: _spg.heat_apply(S, X, float(t), order=order), out_naxis=0)


def field_heat_trajectory(rex, F, times, g=None, order=None, M=None, W=None):
    """[e^{-t W^{-1}M} F for t in times] under the tensor metric W, sharing one set of
    Chebyshev vectors. Returns (len(times),) + graded-F shape."""
    if M is None:
        M = assemble_field_operator(rex, g)
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    Fb = _as_field_block(F, nE, nF)
    return _apply_with_metric(
        M, Fb, field_metric(rex, W),
        lambda S, X: _spg.heat_trajectory(S, X, times, order=order), out_naxis=1)


def field_wave(rex, F, t, g=None, order=None, M=None, W=None):
    """Wave evolution ``cos(t sqrt(W^{-1}M)) F`` of the graded field under the tensor
    metric W - matrix-free Chebyshev, oscillation at omega_k = sqrt(lambda_k) of the
    metric generator, any t, no eigendecomposition. Same shape/metric contract as
    :func:`field_heat`."""
    if M is None:
        M = assemble_field_operator(rex, g)
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    Fb = _as_field_block(F, nE, nF)

    def _apply(S, X):
        lm = _spg._gershgorin_bound(S) * 1.0001 + 1e-30
        return _spg.matfunc_apply(
            S, X, lambda l: np.cos(float(t) * np.sqrt(np.maximum(l, 0.0))),
            _wave_order(t, lm, order), lam_max=lm)

    return _apply_with_metric(M, Fb, field_metric(rex, W), _apply, out_naxis=0)


def field_wave_trajectory(rex, F, times, g=None, order=None, M=None, W=None):
    """[cos(t sqrt(W^{-1}M)) F for t in times] under the tensor metric W, sharing one
    set of Chebyshev vectors."""
    if M is None:
        M = assemble_field_operator(rex, g)
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    Fb = _as_field_block(F, nE, nF)
    tvec = np.asarray(times, dtype=_f64).ravel()
    tmax = float(tvec.max()) if tvec.size else 1.0

    def _apply(S, X):
        lm = _spg._gershgorin_bound(S) * 1.0001 + 1e-30
        funcs = [(lambda l, tt=tt: np.cos(tt * np.sqrt(np.maximum(l, 0.0)))) for tt in tvec]
        return _spg.matfunc_trajectory(S, X, funcs, _wave_order(tmax, lm, order), lam_max=lm)

    return _apply_with_metric(M, Fb, field_metric(rex, W), _apply, out_naxis=1)


def field_wave_full(rex, F, times, g=None, order=None, M=None, W=None):
    """Positions AND velocities of the wave equation on the graded field under the
    tensor metric W, both matrix-free (no eigendecomposition):
        position(t)  =  cos(t sqrt(K)) F
        velocity(t)  = -sqrt(K) sin(t sqrt(K)) F,     K = W^{-1} M
    Returns (positions, velocities), each (len(times),) + graded-F shape. The velocity
    feeds the wave kinetic energy; K is the metric generator (K = M when W = identity)."""
    if M is None:
        M = assemble_field_operator(rex, g)
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    Fb = _as_field_block(F, nE, nF)
    metric = field_metric(rex, W)
    tvec = np.asarray(times, dtype=_f64).ravel()
    tmax = float(tvec.max()) if tvec.size else 1.0

    def _traj(func_of):                                    # func_of(tt) -> lambda l: ...
        def apply_fn(S, X):
            lm = _spg._gershgorin_bound(S) * 1.0001 + 1e-30
            funcs = [func_of(tt) for tt in tvec]
            return _spg.matfunc_trajectory(S, X, funcs, _wave_order(tmax, lm, order), lam_max=lm)
        return _apply_with_metric(M, Fb, metric, apply_fn, out_naxis=1)

    pos = _traj(lambda tt: (lambda l: np.cos(tt * np.sqrt(np.maximum(l, 0.0)))))
    vel = _traj(lambda tt: (lambda l: (-np.sqrt(np.maximum(l, 0.0))
                                       * np.sin(tt * np.sqrt(np.maximum(l, 0.0))))))
    return pos, vel
