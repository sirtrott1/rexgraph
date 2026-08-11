# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._curvature: Lagrangian curvature and the five curvature
localizations, on the sparse/integer path.

The Lagrangians and the five curvatures. This is the math that previously lived in
the agent layer (`schema_complex._lagrangian_curvature` / `_star_curvature`) as
dense `np.trace(L @ L)` - an O(nE^3)/O(nE^2) crash source. Here every quantity is
a SPARSE reduction or a pure-integer degree sum; no dense nE x nE product and no
eigendecomposition.

GLOBAL Lagrangian curvature - NORMALIZED inverse-participation-ratio Lagrangians
The Lagrangians are normalized concentrations, NOT bare traces:
    L_T = tr(T^2) / tr(T)^2      T  = B1^w^T B1^w   (topological / down)
    L_S = tr(L1^2) / tr(L1)^2    L1 = B2^w B2^w^T   (geometric / up)
    c2  = L_T / L_S ;  curvature = |log c2| = |H_S - H_T|   (direction-free)
Each L_X = tr(X^2)/tr(X)^2 = Sum p_i^2 = e^{-H_X} is the inverse participation ratio
of the normalized spectrum, so the Lagrangians ARE the harmonic-log machinery. On
K_k, c2 = (k-2)/2 (1, 3/2, 2, 5/2). The trace identity tr((B^T B)^2) = ||B B^T||_F^2
keeps the numerators sparse (L0 nnz ~ 2*nE; L2 is nF x nF). The bare integer tensors
tr(T^2) = Sum deg^2 + 2*nE and tr(L1^2) are returned as L_T_trace/L_S_trace (still
valid, reframed as the IPR numerators; exact via `lagrangian_L_T_integer`). Small/
unweighted -> exact Fraction (c2_exact); large/weighted -> the normalized ratio stays
O(1) so raw int64 tr(T^2) (~4e12 at weighted K20) never has to be formed. The bare
L_S/L_T ratio (the pre-correction form) is available via normalized=False for diffing;
it coincides with the canonical L_T/L_S only on regular graphs.

FIVE curvatures (localizations of R = B1 diag(w) B2, all sparse):
  1 scalar      ||R||_F
  2 per-face    ||R[:,f]||
  3 per-edge    |w_e| * ||B1[:,e]|| * ||B2[e,:]||
  4 per-vertex star  Sum_{e in star(v)} |w_e - mean_star(v)|   (grade-0; fires on spans)
  5 weighted degree  Sum_{e in star(v)} w_e
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

from rexgraph.core._common cimport i64, f64

np.import_array()


# Helpers (sparse, no densification)

cdef object _as_scipy(obj):
    """Return obj as a scipy CSR *without densifying*. Accepts DualCSR/CSRMatrix
    (from _sparse), any scipy sparse, a dense ndarray, or None."""
    from scipy import sparse as _sp
    if obj is None:
        return None
    if _sp.issparse(obj):
        return obj.tocsr()
    try:
        from rexgraph.core._sparse import to_scipy_csr
        return to_scipy_csr(obj)
    except (TypeError, ImportError):
        return _sp.csr_matrix(np.ascontiguousarray(obj, dtype=np.float64))


cdef object _wdiag(w, Py_ssize_t nE):
    from scipy import sparse as _sp
    if w is None:
        return _sp.identity(nE, format='csr', dtype=np.float64)
    return _sp.diags(np.ascontiguousarray(w, dtype=np.float64).ravel(), format='csr')


# Global Lagrangian curvature (sparse trace identities)

def lagrangian_curvature(B1_in, B2_in, w=None, bint normalized=True):
    """Global Lagrangian curvature {L_T, L_S, c2, curvature, L_T_trace, L_S_trace}.

    NORMALIZED inverse-participation-ratio Lagrangians:
        L_T = tr(T^2)/tr(T)^2, L_S = tr(L1^2)/tr(L1)^2, c2 = L_T/L_S,
        curvature = |log c2| = |H_S - H_T|  (direction-free; None when L_T == 0).
    On K_k, c2 = (k-2)/2. The exact integer numerators tr(T^2), tr(L1^2) (and their
    denominators tr(T), tr(L1)) are returned too; for unweighted inputs c2_exact is
    the exact Fraction. `normalized=False` restores the legacy bare L_S/L_T ratio
    (matches the canonical value only on regular graphs) for diffing.
    """
    B1s = _as_scipy(B1_in)
    cdef Py_ssize_t nE = B1s.shape[1]
    W = _wdiag(w, nE)

    L0w = (B1s @ W) @ B1s.T                       # nnz ~ 2*nE, never densified
    cdef double trT  = float(L0w.diagonal().sum())          # tr(T)
    cdef double trT2 = float(L0w.multiply(L0w).sum())        # tr(T^2) = ||L0^w||_F^2

    cdef double trL = 0.0, trL2 = 0.0
    B2s = _as_scipy(B2_in)
    cdef bint has_faces = B2s is not None and B2s.shape[1] > 0
    if has_faces:
        L2w = (B2s.T @ W) @ B2s                   # nF x nF (small)
        trL  = float(L2w.diagonal().sum())                   # tr(L1)
        trL2 = float(L2w.multiply(L2w).sum())                # tr(L1^2)

    # normalized IPR Lagrangians: bounded in (0,1], no int64 overflow at scale
    cdef double L_T = (trT2 / (trT * trT)) if trT > 0.0 else 0.0
    cdef double L_S = (trL2 / (trL * trL)) if trL > 0.0 else 0.0

    cdef double eps = 1e-12
    c2 = None
    curv = None
    if normalized:
        if L_S > 0.0:
            c2 = L_T / L_S                        # canonical: topological / geometric
        if L_T > 0.0:
            curv = abs(float(np.log((L_T + eps) / (L_S + eps))))   # |log c2|, direction-free
        out = {'L_T': L_T, 'L_S': L_S, 'c2': c2, 'curvature': curv,
               'tr_T': trT, 'tr_L1': trL}
        if w is None:
            # exact integer tensors + exact rational c2 (integer path, no overflow)
            trT_i = int(round(trT)); trT2_i = int(round(trT2))
            trL_i = int(round(trL)); trL2_i = int(round(trL2))
            out['L_T_trace'] = trT2_i
            out['L_S_trace'] = trL2_i
            if trT_i > 0 and trL_i > 0 and trL2_i > 0:
                from fractions import Fraction as _Fr
                out['c2_exact'] = str(_Fr(trT2_i, trT_i * trT_i)
                                      / _Fr(trL2_i, trL_i * trL_i))
        else:
            out['L_T_trace'] = trT2
            out['L_S_trace'] = trL2
        return out

    # legacy bare ratio (pre-correction), kept for diffing
    if trT2 > 0.0:
        c2 = trL2 / trT2
        curv = abs(float(np.log((trL2 + eps) / (trT2 + eps))))
    return {'L_T': trT2, 'L_S': trL2, 'c2': c2, 'curvature': curv,
            'L_T_trace': trT2, 'L_S_trace': trL2, 'tr_T': trT, 'tr_L1': trL}


def lagrangian_L_T_integer(sources, targets, Py_ssize_t nV):
    """Exact integer L_T = Sum_v deg(v)^2 + 2*nE from the degree sequence
    (unweighted). Pure integer, no matrix formed."""
    cdef i64[::1] s = np.ascontiguousarray(sources, dtype=np.int64)
    cdef i64[::1] t = np.ascontiguousarray(targets, dtype=np.int64)
    cdef Py_ssize_t nE = s.shape[0]
    cdef np.ndarray[i64, ndim=1] deg = np.zeros(nV, dtype=np.int64)
    cdef i64[::1] dv = deg
    cdef Py_ssize_t e, v
    for e in range(nE):
        dv[s[e]] += 1
        dv[t[e]] += 1
    cdef i64 acc = 0
    for v in range(nV):
        acc += dv[v] * dv[v]
    return int(acc + 2 * nE)


# Grade-0 localizations (per-vertex; tight integer/rational loops)

def weighted_degree(sources, targets, w, Py_ssize_t nV):
    """Total incident weight per vertex (grade-0). Returns f64[nV]."""
    cdef i64[::1] s = np.ascontiguousarray(sources, dtype=np.int64)
    cdef i64[::1] t = np.ascontiguousarray(targets, dtype=np.int64)
    cdef Py_ssize_t nE = s.shape[0]
    cdef f64[::1] wv = (np.ones(nE, dtype=np.float64) if w is None
                        else np.ascontiguousarray(w, dtype=np.float64))
    cdef np.ndarray[f64, ndim=1] out = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] ov = out
    cdef Py_ssize_t e
    for e in range(nE):
        ov[s[e]] += wv[e]
        ov[t[e]] += wv[e]
    return out


def star_curvature(sources, targets, w, Py_ssize_t nV):
    """Per-vertex star curvature: Sum_{e in star(v)} |w_e - mean_star(v)|, the
    grade-0 localization that fires on spans. 0 for vertices of degree <= 1
    (matches the agent semantics: needs > 1 incident edge). f64[nV]."""
    cdef i64[::1] s = np.ascontiguousarray(sources, dtype=np.int64)
    cdef i64[::1] t = np.ascontiguousarray(targets, dtype=np.int64)
    cdef Py_ssize_t nE = s.shape[0]
    cdef f64[::1] wv = (np.ones(nE, dtype=np.float64) if w is None
                        else np.ascontiguousarray(w, dtype=np.float64))
    cdef np.ndarray[i64, ndim=1] deg = np.zeros(nV, dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] wsum = np.zeros(nV, dtype=np.float64)
    cdef i64[::1] dv = deg
    cdef f64[::1] wsv = wsum
    cdef Py_ssize_t e, v
    for e in range(nE):
        dv[s[e]] += 1; wsv[s[e]] += wv[e]
        dv[t[e]] += 1; wsv[t[e]] += wv[e]

    cdef np.ndarray[f64, ndim=1] mean = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] mv = mean
    for v in range(nV):
        if dv[v] > 0:
            mv[v] = wsv[v] / dv[v]

    cdef np.ndarray[f64, ndim=1] out = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] ov = out
    cdef f64 d
    for e in range(nE):
        v = s[e]; d = wv[e] - mv[v]; ov[v] += (d if d >= 0 else -d)
        v = t[e]; d = wv[e] - mv[v]; ov[v] += (d if d >= 0 else -d)
    for v in range(nV):
        if dv[v] <= 1:
            ov[v] = 0.0
    return out


# Face-bound curvatures from R = B1 diag(w) B2 (sparse)

def curvature_operator(B1_in, B2_in, w=None):
    """The face-bound curvatures from R = B1 diag(w) B2 (grades 1-2, sparse):
    {'scalar': ||R||_F, 'per_face': ||R[:,f]|| (f64[nF]),
     'per_edge': |w_e|*||B1[:,e]||*||B2[e,:]|| (f64[nE])}.
    These read 0 on a span (no face) by construction."""
    B1s = _as_scipy(B1_in)
    B2s = _as_scipy(B2_in)
    cdef Py_ssize_t nE = B1s.shape[1]
    if B2s is None or B2s.shape[1] == 0:
        return {'scalar': 0.0,
                'per_face': np.zeros(0, dtype=np.float64),
                'per_edge': np.zeros(nE, dtype=np.float64)}
    W = _wdiag(w, nE)
    R = ((B1s @ W) @ B2s).tocsc()                 # nV x nF, sparse
    cdef double scalar = float(np.sqrt(R.multiply(R).sum()))
    per_face = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())
    b1col = np.sqrt(np.asarray(B1s.multiply(B1s).sum(axis=0)).ravel())   # ||B1[:,e]||
    b2row = np.sqrt(np.asarray(B2s.multiply(B2s).sum(axis=1)).ravel())   # ||B2[e,:]||
    wv = (np.ones(nE, dtype=np.float64) if w is None
          else np.abs(np.ascontiguousarray(w, dtype=np.float64).ravel()))
    per_edge = wv * b1col * b2row
    return {'scalar': scalar,
            'per_face': np.ascontiguousarray(per_face, dtype=np.float64),
            'per_edge': np.ascontiguousarray(per_edge, dtype=np.float64)}
