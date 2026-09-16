"""rexgraph._experimental: alternative kernels and compatibility imports.

Some implementations are retained alternatives; live primitives and the promoted
factored channel compatibility import are identified individually below.

- ``build_factored_operator``: compatibility import of the now native
  ``channel_operator`` implementation, used by RCQL channel actions. The
  edge primacy MATRIX FREE RL/channel operator
  (every channel applied through B1/|B1|, never materializing the hub clique
  blocks). Numerically agrees with the assembled channels within floating
  tolerance; this is not a bit identity guarantee. It was measured to be
  *overhead*-bound versus a single assembled sparse matmul at moderate nE (many
  small scipy calls per matvec), so ``sparse_character.compute_sparse_character``
  uses the assembled ``RL @ P`` matvec by default. This factored form only wins for
  *extreme*-degree hubs where nnz(RL) ≫ nnz(factored). RCQL actions use its
  incidence factorization without a size gate; existing assembled callers remain.

- ``_hutchinson_phi``: a **uniform** stochastic estimator of the vertex character
  (Rademacher trace estimation, ~1/√n_probe error at every scale). It is a distinct
  Monte Carlo *method*, NOT a size gated fallback - the default character path
  (``compute_sparse_character``) is now exact block CG at all scales. Kept for the
  rare case where an approximate but cheap character is explicitly wanted.

- ``_cheb_apply`` / ``_spectral_bounds``: apply a general analytic f(L) to a dense
  block via a Chebyshev polynomial of sparse mat vecs (no eigendecomposition), plus
  cheap Lanczos spectral bounds. Still LIVE building blocks, referenced by graph.py's
  Chebyshev heat responses, kept here as the shared primitive.

- ``chebyshev_diag`` / ``heat_propagator_diag``: **DEPRECATED / retired.** These chased
  diag(e^{-tL}) on the EDGE space - the diagonal of a general matrix function, which
  has no exact O(nnz) form and is blind to inter grade transport. They are SUPERSEDED
  by Dirac STATE propagation, which carries amplitude ACROSS grades through the
  off diagonal boundary blocks:
    - grade-crossing "heat" / wave transport ->
      ``rexgraph.dirac_propagator.SparseDirac.light`` / ``dirac_light`` (the bounded
      grade-crossing operator is sin(tD), the imaginary part of e^{-itD}).
    - exact O(nnz) LOCAL heat moments -> ``scale_propagator.energy_character``
      (short-time diagonal) + ``scale_propagator.harmonic_entropy`` (trace / global role).
  The resolvent diagonal diag(L⁻¹) has an exact scale free form on the live path
  (``scale_propagator.greens_diagonal``, block CG). The two functions remain importable
  (they emit a ``DeprecationWarning`` and still return correct numbers) so existing
  callers do not break, but they are no longer a recommended path.
"""
from __future__ import annotations

import warnings

import numpy as np

from rexgraph.sparse_character import _block_cg, _f64


def _hutchinson_phi(apply_rl, apply_hat, active_names, Bs, dinv, nhats,
                    uniform, n_probe=400, seed=0):
    """Vertex character via Hutchinson diagonal estimation:
    diag(B1 RL^-1 ĥ_k RL^-1 B1^T) and diag(B1 RL^-1 B1^T) from O(n_probe)
    matrix free solves for ALL vertices at once, independent of nV. Stochastic
    (~1/sqrt(n_probe) relative error), UNIFORM at every scale (not a size gated
    fallback). The default character path is exact block CG; this is a preserved
    alternative for when an approximate but cheap character is explicitly wanted."""
    nV = Bs.shape[0]
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, size=(nV, n_probe)).astype(_f64) * 2.0 - 1.0   # Rademacher
    from rexgraph.native_sparse import NativeSparse
    forward = Bs.apply if isinstance(Bs, NativeSparse) else lambda x: Bs @ x
    transpose = Bs.transpose_apply if isinstance(Bs, NativeSparse) else lambda x: Bs.T @ x
    Y = transpose(Z)                                 # nE x n_probe = B1^T Z
    A = _block_cg(apply_rl, Y, dinv)                 # RL^-1 Y
    den = (Z * forward(A)).mean(axis=1)               # diag(B1 RL^-1 B1^T)
    num = np.zeros((nV, nhats), dtype=_f64)
    for k, name in enumerate(active_names):
        C = _block_cg(apply_rl, apply_hat(name, A), dinv)   # RL^-1 ĥ_k RL^-1 Y
        num[:, k] = (Z * forward(C)).mean(axis=1)
    phi = np.full((nV, nhats), uniform, dtype=_f64)
    ok = np.abs(den) > 1e-15
    phi[ok] = num[ok] / den[ok, None]
    return phi


def _cheb_apply(matvec, func, lam_max, lam_min, order, P):
    """Apply func(L) to a dense block P via a Chebyshev polynomial of L (sparse
    mat vecs only) - returns func(L) @ P, no eigendecomposition. Coefficients from
    the kernel polynomial (discrete cosine) sampling of func on the spectrum."""
    j = np.arange(order)
    xs = np.cos(np.pi * (j + 0.5) / order)
    lam = (xs + 1.0) * (lam_max - lam_min) / 2.0 + lam_min
    fvals = func(lam)
    c = np.array([(2.0 / order) * np.sum(fvals * np.cos(np.pi * k * (j + 0.5) / order))
                  for k in range(order)])
    c[0] /= 2.0
    scale = 2.0 / (lam_max - lam_min)
    shift = (lam_max + lam_min) / (lam_max - lam_min)

    def Ls(X):                                   # rescaled operator on [-1,1]
        return scale * matvec(X) - shift * X

    Tkm1 = P
    Tk = Ls(P)
    acc = c[0] * Tkm1 + c[1] * Tk
    for k in range(2, order):
        Tkp1 = 2.0 * Ls(Tk) - Tkm1
        acc = acc + c[k] * Tkp1
        Tkm1, Tk = Tk, Tkp1
    return acc


_DEPRECATION_MSG = (
    "{name} is deprecated: the edge-space diagonal diag(f(L)) is superseded by Dirac "
    "STATE propagation. Use rexgraph.dirac_propagator.SparseDirac.light / dirac_light "
    "for grade-crossing heat/wave transport, and scale_propagator.energy_character + "
    "scale_propagator.harmonic_entropy for the exact O(nnz) local heat moments. This "
    "function still returns correct numbers but is no longer a recommended path."
)


def chebyshev_diag(matvec, n, func, lam_max, lam_min=0.0, order=48,
                   mode='exact', n_probe=256, seed=0):
    """DEPRECATED. diag(func(L)) for a GENERAL analytic f via a Chebyshev polynomial of
    L applied by sparse mat vecs: no eigendecomposition.

    Superseded by Dirac state propagation (see the module docstring and
    :data:`_DEPRECATION_MSG`). Still returns correct numbers so existing imports keep
    working; emits a ``DeprecationWarning``. mode='exact' applies the polynomial to
    identity columns (O(n²·order), exact); mode='stochastic' is a Hutchinson estimate
    (O(n_probe·nnz·order), ~1/√n_probe err). Returns f64[n]."""
    warnings.warn(_DEPRECATION_MSG.format(name="chebyshev_diag"),
                  DeprecationWarning, stacklevel=2)
    return _chebyshev_diag_impl(matvec, n, func, lam_max, lam_min=lam_min, order=order,
                                mode=mode, n_probe=n_probe, seed=seed)


def _chebyshev_diag_impl(matvec, n, func, lam_max, lam_min=0.0, order=48,
                         mode='exact', n_probe=256, seed=0):
    """Warning free compute core of the retired :func:`chebyshev_diag` (so the
    deprecated public wrappers warn exactly once)."""
    if n == 0:
        return np.zeros(0, dtype=_f64)
    if mode == 'exact':
        diag = np.zeros(n, dtype=_f64)
        block = min(n, 512)
        for start in range(0, n, block):
            stop = min(start + block, n)
            E = np.zeros((n, stop - start), dtype=_f64)
            for i in range(start, stop):
                E[i, i - start] = 1.0
            acc = _cheb_apply(matvec, func, lam_max, lam_min, order, E)
            for i in range(start, stop):
                diag[i] = acc[i, i - start]
        return diag
    if mode == 'stochastic':
        rng = np.random.default_rng(seed)
        Z = rng.integers(0, 2, size=(n, int(n_probe))).astype(_f64) * 2.0 - 1.0
        FZ = _cheb_apply(matvec, func, lam_max, lam_min, order, Z)
        return (Z * FZ).mean(axis=1)
    raise ValueError("mode must be 'exact' or 'stochastic'")


def _spectral_bounds(R, n):
    """Cheap (Lanczos) estimates of (λ_min, λ_max) of symmetric R: a few mat vecs,
    NOT a full eigendecomposition. Falls back to Gershgorin if Lanczos is unavailable
    or the operator is tiny."""
    gersh = float(np.asarray(np.abs(R).sum(axis=1)).ravel().max()) if n else 1.0
    if n <= 3:
        ev = np.linalg.eigvalsh(R.toarray())
        return float(max(ev.min(), 1e-12)), float(ev.max())
    try:
        import scipy.sparse.linalg as sla
        hi = float(sla.eigsh(R, k=1, which='LA', return_eigenvectors=False,
                             maxiter=n * 5, tol=1e-3)[0])
        lo = float(sla.eigsh(R, k=1, which='SA', return_eigenvectors=False,
                             maxiter=n * 10, tol=1e-3)[0])
        return max(lo, 1e-12), max(hi, gersh * 0.5)
    except Exception:
        return 1e-6, gersh


def heat_propagator_diag(RL4, t, lam_max=None, order=48, mode='exact'):
    """DEPRECATED. diag(e^{-t·RL4}) via the Chebyshev heat propagator (general-f, no
    exact O(nnz) form).

    Superseded by Dirac state propagation. Use
    ``rexgraph.dirac_propagator.SparseDirac.light`` / ``dirac_light`` for grade crossing
    heat transport, and ``scale_propagator.energy_character`` (short time) +
    ``scale_propagator.harmonic_entropy`` for the exact O(nnz) local heat moments.
    Still returns correct numbers so existing imports keep working; emits a
    ``DeprecationWarning``."""
    warnings.warn(_DEPRECATION_MSG.format(name="heat_propagator_diag"),
                  DeprecationWarning, stacklevel=2)
    import scipy.sparse as sp
    R = RL4.tocsr() if sp.issparse(RL4) else sp.csr_matrix(np.asarray(RL4, dtype=_f64))
    n = R.shape[0]
    if n == 0:
        return np.zeros(0, dtype=_f64)
    if lam_max is None:
        lam_max = float(np.asarray(np.abs(R).sum(axis=1)).ravel().max())
    return _chebyshev_diag_impl(lambda P: R @ P, n, lambda l: np.exp(-t * l),
                                lam_max, lam_min=0.0, order=order, mode=mode)


# Compatibility import: the tested factored action is now native RCQL machinery.
from rexgraph.channel_operator import build_factored_operator  # noqa: E402,F401
