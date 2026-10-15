"""Polynomial application of scalar functions to symmetric operators.

The helpers apply Chebyshev polynomials through a supplied matrix product.
Diagonal evaluation uses identity blocks or an explicitly selected stochastic
estimator, without constructing the full matrix function.
"""
from __future__ import annotations

import numpy as np

_f64 = np.float64

def _cheb_apply(matvec, func, lam_max, lam_min, order, P):
    """Apply func(L) to a dense block P via a Chebyshev polynomial of L (sparse
    mat vecs only): returns func(L) @ P, no eigendecomposition. Coefficients from
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


def _chebyshev_diag_impl(matvec, n, func, lam_max, lam_min=0.0, order=48,
                         mode='exact', n_probe=256, seed=0):
    """Evaluate the diagonal of func(L) using its Chebyshev polynomial.

    mode="exact" applies the polynomial to blocks of identity columns and
    returns their diagonal entries. mode="stochastic" estimates the diagonal
    from n_probe Rademacher vectors using the specified seed. Both modes use
    the supplied spectral interval and polynomial order.
    """
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
