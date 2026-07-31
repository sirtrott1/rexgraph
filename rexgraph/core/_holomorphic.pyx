# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._holomorphic: Holomorphic Lagrangian structure on RL_4.

Implements the complex analytic structure of the time-space Lagrangian
decomposition on the RL_4 channel hats rather than on the graded Laplacian.
Both are built on the same relational complex: the graded Laplacian comes
straight from its boundary maps, the RL_4 hats from its four typed channels.

Key mathematical facts:
    - L_t (time, channel T = hat_0) and L_s (space, channels G+F+C)
      satisfy Cauchy-Riemann conditions within a single analytic domain.
    - f(e) = L_t(e) + i*L_s(e) is holomorphic per edge.
    - On the graded Laplacian, [L_t, L_s] = 0 identically (algebraic
      consequence of B_1 B_2 = 0). This is a tautology.
    - On the RL_4 channel hats, [hat_T, hat_S] != 0. The relational
      operators interact through overlap, frustration, and coupling
      channels.
    - The per-edge CR violation |dTdS(e) - dSdT(e)| on the relational
      complex is a category-specific invariant: each topological regime
      has its own characteristic CR value.
    - At the boundary between two regimes, the CR violation of each
      subcomplex converges to its category's characteristic value.
      The boundary is the saddle point where CR_L = CR_R.

Functions:
    lagrangian_fields      Per-edge L_t(e), L_s(e), c^2(e), f(e).
    relational_cr          Per-edge CR violation from RL_4 hat operators.
    cr_saddle_score        Mean CR violation scalar (for boundary scans).
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport fabs, sqrt

ctypedef double f64
ctypedef int i32

np.import_array()


def lagrangian_fields(list hats):
    """Per-edge Lagrangian fields from RL_4 channel hat operators.

    The time Lagrangian L_t is the diagonal of hat_T (channel 0).
    The space Lagrangian L_s is the diagonal of hat_S = hat_G + hat_F + hat_C
    (channels 1, 2, 3).  The complex Lagrangian is f(e) = L_t(e) + i*L_s(e).

    Parameters
    ----------
    hats : list of ndarray, each shape (nE, nE)
        The four RL_4 channel hat operators [hat_T, hat_G, hat_F, hat_C].

    Returns
    -------
    dict with keys:
        Lt : f64[nE]       per-edge time Lagrangian
        Ls : f64[nE]       per-edge space Lagrangian (action)
        c2 : f64[nE]       per-edge speed of light squared, Ls/Lt
        f_mag : f64[nE]    |f(e)| = sqrt(Lt^2 + Ls^2)
        f_arg : f64[nE]    arg(f(e)) = arctan(Ls/Lt)
    """
    cdef int nE, e
    cdef np.ndarray[f64, ndim=2] hat_T, hat_G, hat_F, hat_C
    cdef np.ndarray[f64, ndim=1] Lt, Ls, c2, f_mag, f_arg
    cdef f64 lt_val, ls_val

    if len(hats) < 4:
        raise ValueError("lagrangian_fields requires 4 hat operators (RL_4)")

    hat_T = np.ascontiguousarray(hats[0], dtype=np.float64)
    hat_G = np.ascontiguousarray(hats[1], dtype=np.float64)
    hat_F = np.ascontiguousarray(hats[2], dtype=np.float64)
    hat_C = np.ascontiguousarray(hats[3], dtype=np.float64)

    nE = hat_T.shape[0]
    Lt = np.empty(nE, dtype=np.float64)
    Ls = np.empty(nE, dtype=np.float64)
    c2 = np.empty(nE, dtype=np.float64)
    f_mag = np.empty(nE, dtype=np.float64)
    f_arg = np.empty(nE, dtype=np.float64)

    for e in range(nE):
        lt_val = hat_T[e, e]
        ls_val = hat_G[e, e] + hat_F[e, e] + hat_C[e, e]
        Lt[e] = lt_val
        Ls[e] = ls_val
        c2[e] = ls_val / lt_val if fabs(lt_val) > 1e-15 else 0.0
        f_mag[e] = sqrt(lt_val * lt_val + ls_val * ls_val)
        f_arg[e] = np.arctan2(ls_val, lt_val)

    return {
        'Lt': Lt,
        'Ls': Ls,
        'c2': c2,
        'f_mag': f_mag,
        'f_arg': f_arg,
    }


def relational_cr(list hats):
    """Per-edge Cauchy-Riemann violation in the relational complex.

    Computes the partial derivatives dTdS(e) = (hat_T @ hat_S)[e,e] / hat_S[e,e]
    and dSdT(e) = (hat_S @ hat_T)[e,e] / hat_T[e,e] from the RL_4 hat operators.
    The CR violation at each edge is |dTdS(e) - dSdT(e)|.

    On the graded Laplacian (L_1 = L_t + L_s), these are identically equal
    because B_1 B_2 = 0 forces [L_t, L_s] = 0.  On the RL_4 channel hats
    (RL_4 = hat_T + hat_G + hat_F + hat_C), the channels interact and the
    CR violation is nonzero and category-specific.

    Parameters
    ----------
    hats : list of ndarray, each shape (nE, nE)
        The four RL_4 channel hat operators [hat_T, hat_G, hat_F, hat_C].

    Returns
    -------
    dict with keys:
        dTdS : f64[nE]         per-edge dL_t/dL_s
        dSdT : f64[nE]         per-edge dL_s/dL_t
        cr   : f64[nE]         per-edge |dTdS - dSdT|
        cr_mean : float        mean CR violation
        cr_std  : float        std of CR violation
    """
    cdef int nE, e
    cdef np.ndarray[f64, ndim=2] hat_T, hat_S, TS, ST
    cdef np.ndarray[f64, ndim=1] dTdS_arr, dSdT_arr, cr_arr
    cdef f64 ts_val, st_val, s_diag, t_diag

    if len(hats) < 4:
        raise ValueError("relational_cr requires 4 hat operators (RL_4)")

    hat_T = np.ascontiguousarray(hats[0], dtype=np.float64)
    hat_S = np.ascontiguousarray(
        hats[1] + hats[2] + hats[3], dtype=np.float64)

    nE = hat_T.shape[0]
    # Only the DIAGONALS of the products are used: diag(TS)_e = Σ_k T[e,k]S[k,e]
    # (row·col), O(nE²) per edge total - never form the nE×nE products (O(nE³)).
    cdef np.ndarray[f64, ndim=1] TS_diag = np.einsum('ek,ke->e', hat_T, hat_S)
    cdef np.ndarray[f64, ndim=1] ST_diag = np.einsum('ek,ke->e', hat_S, hat_T)

    dTdS_arr = np.zeros(nE, dtype=np.float64)
    dSdT_arr = np.zeros(nE, dtype=np.float64)
    cr_arr = np.zeros(nE, dtype=np.float64)

    for e in range(nE):
        s_diag = hat_S[e, e]
        t_diag = hat_T[e, e]

        if fabs(s_diag) > 1e-15:
            dTdS_arr[e] = TS_diag[e] / s_diag
        if fabs(t_diag) > 1e-15:
            dSdT_arr[e] = ST_diag[e] / t_diag

        cr_arr[e] = fabs(dTdS_arr[e] - dSdT_arr[e])

    cdef f64 cr_mean = 0.0
    cdef f64 cr_var = 0.0
    for e in range(nE):
        cr_mean += cr_arr[e]
    cr_mean /= nE

    for e in range(nE):
        cr_var += (cr_arr[e] - cr_mean) * (cr_arr[e] - cr_mean)
    cr_var /= nE

    return {
        'dTdS': dTdS_arr,
        'dSdT': dSdT_arr,
        'cr': cr_arr,
        'cr_mean': float(cr_mean),
        'cr_std': float(sqrt(cr_var)),
    }


def cr_saddle_score(list hats):
    """Mean CR violation scalar for boundary-scan use.

    A thin wrapper around relational_cr that returns only the scalar
    mean, avoiding dict overhead in tight scan loops.

    Parameters
    ----------
    hats : list of ndarray, each shape (nE, nE)

    Returns
    -------
    float
        Mean per-edge CR violation in the relational complex.
    """
    if len(hats) < 4:
        return 0.0
    # Reuse the (de-densified) per-edge CR - diag(TS)/diag(ST) are O(nE²) row·col
    # dots, no nE×nE products.
    cr = relational_cr(hats)['cr']
    return float(np.mean(cr)) if cr.size > 0 else 0.0
