# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._holomorphic: diagonal and product readings of RL_4 channel hats.

L_t is the T diagonal and L_s is the G+F+C diagonal. Their complex pair and
ratio describe each relation. relational_cr is the retained name for the
absolute difference of two normalized product diagonals. It does not certify
holomorphicity or operator commutation.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport fabs, sqrt

ctypedef double f64
ctypedef int i32

np.import_array()


def _validated_channel_hats(list hats):
    """Return the first four hats as finite square arrays on one carrier."""
    if len(hats) < 4:
        raise ValueError("requires 4 hat operators (RL_4)")
    arrays = [np.ascontiguousarray(h, dtype=np.float64) for h in hats[:4]]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("channel hats must be square matrices")
    if any(a.shape != shape or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("channel hats require one shape and finite coefficients")
    return arrays


def lagrangian_fields(list hats):
    """Per edge Lagrangian fields from RL_4 channel hat operators.

    The time Lagrangian L_t is the diagonal of hat_T (channel 0).
    The space Lagrangian L_s is the diagonal of hat_S = hat_G + hat_F + hat_C
    (channels 1, 2, 3).  The complex Lagrangian is f(e) = L_t(e) + i*L_s(e).

    Parameters

    hats : list of ndarray, each shape (nE, nE)
        The four RL_4 channel hat operators [hat_T, hat_G, hat_F, hat_C].

    Returns

    dict with keys:
        Lt : f64[nE]       per-edge time Lagrangian
        Ls : f64[nE]       per-edge space Lagrangian (action)
        c2 : f64[nE]       per-edge ratio Ls/Lt, zero when Lt is negligible
        f_mag : f64[nE]    |f(e)| = sqrt(Lt^2 + Ls^2)
        f_arg : f64[nE]    arg(f(e)) = arctan(Ls/Lt)
    """
    cdef int nE, e
    cdef np.ndarray[f64, ndim=2] hat_T, hat_G, hat_F, hat_C
    cdef np.ndarray[f64, ndim=1] Lt, Ls, c2, f_mag, f_arg
    cdef f64 lt_val, ls_val

    hat_T, hat_G, hat_F, hat_C = _validated_channel_hats(hats)

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
    """Per relation imbalance of normalized channel product diagonals.

    dTdS(e) = (hat_T @ hat_S)[e,e] / hat_S[e,e] and
    dSdT(e) = (hat_S @ hat_T)[e,e] / hat_T[e,e], with hat_S = hat_G+hat_F+hat_C.
    A negligible denominator returns zero. cr is `abs(dTdS - dSdT)`.
    For symmetric hats the two product diagonals agree, but their denominators
    may differ. A nonzero cr can therefore occur for commuting operators.

    Parameters

    hats : list of ndarray, each shape (nE, nE)
        The four RL_4 channel hat operators [hat_T, hat_G, hat_F, hat_C].

    Returns

    dict with keys:
        dTdS : f64[nE]         diag(T S) / diag(S)
        dSdT : f64[nE]         diag(S T) / diag(T)
        cr   : f64[nE]         per-edge |dTdS - dSdT|
        cr_mean : float        mean imbalance, zero on an empty carrier
        cr_std  : float        standard deviation of the imbalance
    """
    cdef int nE, e
    cdef np.ndarray[f64, ndim=2] hat_T, hat_S, TS, ST
    cdef np.ndarray[f64, ndim=1] dTdS_arr, dSdT_arr, cr_arr
    cdef f64 ts_val, st_val, s_diag, t_diag

    checked = _validated_channel_hats(hats)
    hat_T = checked[0]
    hat_S = np.ascontiguousarray(
        checked[1] + checked[2] + checked[3], dtype=np.float64)

    nE = hat_T.shape[0]
    # Product diagonals cost O(nE^2) for the full relation carrier.
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
    if nE:
        cr_mean /= nE

    for e in range(nE):
        cr_var += (cr_arr[e] - cr_mean) * (cr_arr[e] - cr_mean)
    if nE:
        cr_var /= nE

    return {
        'dTdS': dTdS_arr,
        'dSdT': dSdT_arr,
        'cr': cr_arr,
        'cr_mean': float(cr_mean),
        'cr_std': float(sqrt(cr_var)),
    }


def cr_saddle_score(list hats):
    """Mean normalized product imbalance for boundary scans.

    Parameters

    hats : list of ndarray, each shape (nE, nE)

    Returns

    float
        Mean per relation imbalance in the relational complex.
    """
    if len(hats) < 4:
        return 0.0
    cr = relational_cr(hats)['cr']
    return float(np.mean(cr)) if cr.size > 0 else 0.0
