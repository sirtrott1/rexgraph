# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._character - Structural character decomposition.

chi, phi, chi_star, kappa, per-channel mixing times, face-void dipole.
All LAPACK/BLAS, zero Python in hot paths.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, log, sqrt
from libc.string cimport memset

cimport cython

from rexgraph.core._common cimport i32, i64, f64, idx_t
from rexgraph.core._linalg cimport (
    bl_gemm_nn, bl_gemm_nt, bl_dot, bl_nrm2,
    spectral_pinv, spectral_pinv_matvec, lp_lstsq, lp_eigh,
)

np.import_array()


# Chi: structural character per edge (C-level, no allocation beyond output)

cdef void _compute_chi(const f64* rl_data, const f64* const* hat_data,
                        f64* chi_out, int nE, int nhats) noexcept nogil:
    """chi(e, k) = hat_k[e,e] / RL[e,e]. O(nE * nhats)."""
    cdef int e, k
    cdef f64 rl_ee, uniform
    uniform = 1.0 / nhats if nhats > 0 else 0.0
    for e in range(nE):
        rl_ee = rl_data[e * nE + e]
        if rl_ee > 1e-15:
            for k in range(nhats):
                chi_out[e * nhats + k] = hat_data[k][e * nE + e] / rl_ee
        else:
            for k in range(nhats):
                chi_out[e * nhats + k] = uniform


def compute_chi(np.ndarray[f64, ndim=2] RL, list hats, int nhats, int nE):
    """Structural character per edge."""
    cdef np.ndarray[f64, ndim=2] chi = np.zeros((nE, nhats), dtype=np.float64)
    cdef f64* hat_ptrs[8]  # max 8 hats
    cdef int k
    cdef np.ndarray[f64, ndim=2] hat_arr
    for k in range(nhats):
        hat_arr = hats[k]
        hat_ptrs[k] = &hat_arr[0, 0]
    _compute_chi(&RL[0, 0], <const f64* const*>hat_ptrs, &chi[0, 0], nE, nhats)
    return chi


# Phi: vertex character (dense path — all BLAS)

cdef void _compute_phi_dense(const f64* B1, const f64* RLp,
                              const f64* const* hat_data,
                              f64* phi_out, f64* B1_RLp_buf, f64* tmp_buf,
                              int nV, int nE, int nhats) noexcept nogil:
    """phi(v,k) = diag(B1_RLp @ hat_k @ B1_RLp^T)[v] / diag(B1_RLp @ B1^T)[v].

    B1_RLp_buf: pre-allocated nV x nE.
    tmp_buf: pre-allocated nV x nE.
    """
    cdef int v, e, k
    cdef f64 s0_vv, phi_vk

    # B1_RLp = B1 @ RLp
    bl_gemm_nn(B1, RLp, B1_RLp_buf, nV, nE, nE)

    # S0 diagonal = einsum('ve,ve->v', B1_RLp, B1)
    cdef f64 uniform = 1.0 / nhats if nhats > 0 else 0.0

    for k in range(nhats):
        # tmp = B1_RLp @ hat_k
        bl_gemm_nn(B1_RLp_buf, hat_data[k], tmp_buf, nV, nE, nE)
        # phi[:, k] = diag(tmp @ B1_RLp^T) / S0_diag
        for v in range(nV):
            # S0[v,v] = B1_RLp[v,:] . B1[v,:]
            s0_vv = 0
            for e in range(nE):
                s0_vv += B1_RLp_buf[v * nE + e] * B1[v * nE + e]
            if fabs(s0_vv) > 1e-15:
                phi_vk = 0
                for e in range(nE):
                    phi_vk += tmp_buf[v * nE + e] * B1_RLp_buf[v * nE + e]
                phi_out[v * nhats + k] = phi_vk / s0_vv
            else:
                phi_out[v * nhats + k] = uniform


def compute_phi_dense(np.ndarray[f64, ndim=2] B1,
                       np.ndarray[f64, ndim=2] RLp,
                       list hats, int nhats, int nV, int nE):
    """Vertex character (dense path). All BLAS, zero Python."""
    cdef np.ndarray[f64, ndim=2] phi = np.zeros((nV, nhats), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] B1_RLp_buf = np.empty((nV, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] tmp_buf = np.empty((nV, nE), dtype=np.float64)
    cdef f64* hat_ptrs[8]
    cdef int k
    cdef np.ndarray[f64, ndim=2] hat_arr_p
    for k in range(nhats):
        hat_arr_p = hats[k]
        hat_ptrs[k] = &hat_arr_p[0, 0]
    _compute_phi_dense(&B1[0, 0], &RLp[0, 0],
                        <const f64* const*>hat_ptrs,
                        &phi[0, 0], &B1_RLp_buf[0, 0], &tmp_buf[0, 0],
                        nV, nE, nhats)
    return phi


# Phi: sparse path (per-vertex LAPACK lstsq)

def compute_phi_sparse_single(np.ndarray[f64, ndim=2] RL,
                                list hats, int nhats,
                                np.ndarray[f64, ndim=2] B1,
                                int vertex_idx, int nV, int nE):
    """phi(v) for a single vertex via lstsq solve."""
    # rhs = B1[v, :] (row of B1 = B1^T @ e_v for dense B1)
    cdef np.ndarray[f64, ndim=1] rhs = B1[vertex_idx, :].copy()
    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(RL.copy())
    cdef np.ndarray[f64, ndim=1] S = np.empty(nE, dtype=np.float64)
    cdef int rank = 0
    lp_lstsq(&A_F[0, 0], &rhs[0], nE, nE, 1, &S[0], &rank)
    # rhs is now x = RL^+ @ B1^T e_v

    cdef np.ndarray[f64, ndim=1] x = rhs
    cdef f64[::1] xv = x
    cdef f64[:, ::1] b1v = B1

    # S0[v,v] = B1[v,:] . x
    cdef f64 svv = 0
    cdef int e, k
    for e in range(nE):
        svv += b1v[vertex_idx, e] * xv[e]

    cdef np.ndarray[f64, ndim=1] phi_v = np.zeros(nhats, dtype=np.float64)
    cdef f64 uniform = 1.0 / nhats if nhats > 0 else 0.0

    if fabs(svv) < 1e-15:
        for k in range(nhats): phi_v[k] = uniform
        return phi_v

    cdef np.ndarray[f64, ndim=2] hat_k
    cdef f64[:, ::1] hkv
    cdef f64 val
    cdef int i_c, j_c

    for k in range(nhats):
        # phi(v,k) = x^T hat_k x / svv
        hat_k = hats[k]
        hkv = hat_k
        val = 0
        for i_c in range(nE):
            for j_c in range(nE):
                val += xv[i_c] * hkv[i_c, j_c] * xv[j_c]
        phi_v[k] = val / svv

    return phi_v


def compute_phi(np.ndarray[f64, ndim=2] B1,
                np.ndarray[f64, ndim=2] RL,
                list hats, int nhats, int nV, int nE,
                green_cache=None):
    """Vertex character. Dispatches dense (BLAS) or sparse (lstsq)."""
    if green_cache is not None and green_cache.get('dense', False):
        return compute_phi_dense(B1, green_cache['RL_pinv'], hats, nhats, nV, nE)
    else:
        phi = np.zeros((nV, nhats), dtype=np.float64)
        for v in range(nV):
            phi[v, :] = compute_phi_sparse_single(RL, hats, nhats, B1, v, nV, nE)
        return phi


# Chi-star (C-level)

cdef void _compute_chi_star(const f64* chi, const i32* v2e_ptr, const i32* v2e_idx,
                             f64* chi_star, int nV, int nhats) noexcept nogil:
    """chi*(v) = mean of chi(e) over incident edges. O(sum deg)."""
    cdef int v, j, lo, hi, cnt, k, e
    cdef f64 uniform = 1.0 / nhats if nhats > 0 else 0.0
    for v in range(nV):
        lo = v2e_ptr[v]
        hi = v2e_ptr[v + 1]
        cnt = hi - lo
        if cnt == 0:
            for k in range(nhats): chi_star[v * nhats + k] = uniform
            continue
        for k in range(nhats): chi_star[v * nhats + k] = 0
        for j in range(lo, hi):
            e = v2e_idx[j]
            for k in range(nhats):
                chi_star[v * nhats + k] += chi[e * nhats + k]
        for k in range(nhats):
            chi_star[v * nhats + k] /= <f64>cnt


def compute_chi_star(np.ndarray[f64, ndim=2] chi,
                      np.ndarray[i32, ndim=1] v2e_ptr,
                      np.ndarray[i32, ndim=1] v2e_idx,
                      int nV, int nhats):
    cdef np.ndarray[f64, ndim=2] cs = np.zeros((nV, nhats), dtype=np.float64)
    _compute_chi_star(&chi[0, 0], &v2e_ptr[0], &v2e_idx[0],
                       &cs[0, 0], nV, nhats)
    return cs


# Kappa (C-level)

cdef void _compute_kappa(const f64* phi, const f64* chi_star,
                          f64* kappa, int nV, int nhats) noexcept nogil:
    cdef int v, k
    cdef f64 l1_norm
    for v in range(nV):
        l1_norm = 0
        for k in range(nhats):
            l1_norm += fabs(phi[v * nhats + k] - chi_star[v * nhats + k])
        kappa[v] = 1.0 - 0.5 * l1_norm


def compute_kappa(np.ndarray[f64, ndim=2] phi,
                   np.ndarray[f64, ndim=2] chi_star,
                   int nV, int nhats):
    cdef np.ndarray[f64, ndim=1] kappa = np.empty(nV, dtype=np.float64)
    _compute_kappa(&phi[0, 0], &chi_star[0, 0], &kappa[0], nV, nhats)
    return kappa


# Combined builder

def build_character_bundle(np.ndarray[f64, ndim=2] B1,
                            np.ndarray[f64, ndim=2] RL,
                            list hats, int nhats, int nV, int nE,
                            np.ndarray[i32, ndim=1] v2e_ptr,
                            np.ndarray[i32, ndim=1] v2e_idx,
                            green_cache=None):
    chi = compute_chi(RL, hats, nhats, nE)
    phi = compute_phi(B1, RL, hats, nhats, nV, nE, green_cache=green_cache)
    chi_star = compute_chi_star(chi, v2e_ptr, v2e_idx, nV, nhats)
    kappa = compute_kappa(phi, chi_star, nV, nhats)
    return {'chi': chi, 'phi': phi, 'chi_star': chi_star, 'kappa': kappa}


# Per-vertex curvature

cdef void _per_vertex_curvature(const f64* phi, const f64* chi_star,
                                 const f64* kappa, f64* curv,
                                 int nV, int nhats) noexcept nogil:
    cdef int v, k
    cdef f64 diff, sq_sum
    for v in range(nV):
        sq_sum = 0
        for k in range(nhats):
            diff = phi[v * nhats + k] - chi_star[v * nhats + k]
            sq_sum += diff * diff
        curv[v] = sq_sum / kappa[v] if kappa[v] > 1e-15 else sq_sum


def per_vertex_curvature(np.ndarray[f64, ndim=2] phi,
                          np.ndarray[f64, ndim=2] chi_star,
                          np.ndarray[f64, ndim=1] kappa,
                          int nV, int nhats):
    cdef np.ndarray[f64, ndim=1] curv = np.zeros(nV, dtype=np.float64)
    _per_vertex_curvature(&phi[0, 0], &chi_star[0, 0], &kappa[0],
                           &curv[0], nV, nhats)
    return curv


# Structural summary

def structural_summary(np.ndarray[f64, ndim=2] chi,
                        np.ndarray[f64, ndim=2] phi,
                        np.ndarray[f64, ndim=1] kappa,
                        int nE, int nV, int nhats):
    result = {}
    for k in range(nhats):
        result[f'mean_chi_{k}'] = float(np.mean(chi[:, k]))
        result[f'std_chi_{k}'] = float(np.std(chi[:, k]))
    result['mean_kappa'] = float(np.mean(kappa))
    result['std_kappa'] = float(np.std(kappa))
    result['low_kappa_count'] = int(np.sum(kappa < 0.5))
    dominant = np.argmax(chi, axis=1)
    for k in range(nhats):
        result[f'dominant_{k}_count'] = int(np.sum(dominant == k))
    return result


# ═══ Derived coefficients and operators (from DRCT/I-SPY2 case studies) ═══

def topological_integrity(np.ndarray[f64, ndim=2] chi,
                           np.ndarray[f64, ndim=1] edge_weights,
                           int nE, int nhats):
    """Topological integrity: weighted character sums per channel.

    IT = sum_e w(e) * chi(e, 0)   (topological channel weight)
    IF = sum_e w(e) * chi(e, 2)   (frustration channel weight)
    regime = 'NORMAL' if IT > IF else 'INVERTED'

    Returns dict with IT, IF, channel_weights[nhats], regime.
    """
    cdef np.ndarray[f64, ndim=1] cw = np.zeros(nhats, dtype=np.float64)
    cdef f64[:, ::1] cv = chi
    cdef f64[::1] wv = edge_weights, cwv = cw
    cdef int e, k
    for e in range(nE):
        for k in range(nhats):
            cwv[k] += wv[e] * cv[e, k]
    IT = float(cw[0]) if nhats > 0 else 0.0
    IF_val = float(cw[2]) if nhats > 2 else 0.0
    return {
        'IT': IT, 'IF': IF_val,
        'channel_weights': cw,
        'regime': 'NORMAL' if IT > IF_val else 'INVERTED',
    }


def structural_entropy(np.ndarray[f64, ndim=2] chi, int nE, int nhats):
    """Shannon entropy of the mean structural character across edges.

    H = -sum_k chi_bar_k * ln(chi_bar_k)
    where chi_bar_k = (1/nE) * sum_e chi(e, k).
    Maximum entropy = ln(nhats) when uniform.
    """
    cdef np.ndarray[f64, ndim=1] mean_chi = np.zeros(nhats, dtype=np.float64)
    cdef f64[::1] mc = mean_chi
    cdef f64[:, ::1] cv = chi
    cdef int e, k
    cdef f64 ent = 0.0

    for e in range(nE):
        for k in range(nhats):
            mc[k] += cv[e, k]
    for k in range(nhats):
        mc[k] /= nE

    for k in range(nhats):
        if mc[k] > 1e-15:
            ent -= mc[k] * log(mc[k])

    return float(ent)


def per_vertex_weighted_curvature(np.ndarray[f64, ndim=2] chi,
                                    np.ndarray[f64, ndim=1] edge_weights,
                                    np.ndarray[i32, ndim=1] sources,
                                    np.ndarray[i32, ndim=1] targets,
                                    int nV, int nE, int nhats):
    """Per-vertex weighted curvature from character deviation.

    C(v) = (1/deg(v)) * sum_{e in star(v)} w(e) * ||chi(e) - 1/nhats||_1
    Measures how far incident edges deviate from uniform character.
    """
    cdef np.ndarray[f64, ndim=1] curv = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] deg = np.zeros(nV, dtype=np.int32)
    cdef f64[::1] cv_out = curv, wv = edge_weights
    cdef i32[::1] dv = deg, sv = sources, tv = targets
    cdef f64[:, ::1] chiv = chi
    cdef int e, k, v
    cdef f64 dev, uniform = 1.0 / nhats if nhats > 0 else 0.0

    for e in range(nE):
        dev = 0
        for k in range(nhats):
            dev += fabs(chiv[e, k] - uniform)
        cv_out[sv[e]] += wv[e] * dev
        cv_out[tv[e]] += wv[e] * dev
        dv[sv[e]] += 1
        dv[tv[e]] += 1

    for v in range(nV):
        if dv[v] > 0:
            cv_out[v] /= dv[v]

    return curv


def self_response(np.ndarray[f64, ndim=2] RLp, int nE):
    """Self-response (effective resistance) per edge: R_self(e) = RL^+[e,e].

    Higher self-response = edge is more structurally isolated.
    """
    cdef np.ndarray[f64, ndim=1] rs = np.empty(nE, dtype=np.float64)
    cdef f64[::1] rv = rs
    cdef f64[:, ::1] rp = RLp
    cdef int e
    for e in range(nE):
        rv[e] = rp[e, e]
    return rs


def signed_cosine_matrix(np.ndarray[f64, ndim=2] RL, int nE):
    """Signed cosine similarity from RL: cos_s(i,j) = RL[i,j] / sqrt(RL[i,i]*RL[j,j]).

    Measures structural coupling between edges through the relational Laplacian.
    """
    cdef np.ndarray[f64, ndim=2] sc = np.zeros((nE, nE), dtype=np.float64)
    cdef f64[:, ::1] scv = sc, rv = RL
    cdef int i, j
    cdef f64 dii, djj, denom

    for i in range(nE):
        for j in range(i, nE):
            dii = rv[i, i]
            djj = rv[j, j]
            denom = sqrt(fabs(dii * djj))
            if denom > 1e-15:
                scv[i, j] = rv[i, j] / denom
                scv[j, i] = scv[i, j]
            elif i == j:
                scv[i, j] = 1.0

    return sc


def mixing_time(np.ndarray[f64, ndim=1] rl_evals, int nE):
    """Mixing time from RL spectral gap: tau_mix = ln(nE) / lambda_2.

    lambda_2 = smallest positive eigenvalue of RL (algebraic connectivity).
    Smaller mixing time = faster information propagation.
    """
    cdef f64[::1] ev = rl_evals
    cdef int k
    cdef f64 lambda2 = 0

    for k in range(nE):
        if ev[k] > 1e-10:
            lambda2 = ev[k]
            break

    if lambda2 < 1e-15:
        return float('inf')
    return float(log(<f64>nE) / lambda2)


def derived_constants(int nV):
    """Derived constants from vertex space dimension.

    H0 = 1.0 (multiplicative ground state identity)
    amp_coeff = (nV-1)/nV (amplitude coupling coefficient)
    scaffold_floor = 1/nV^2 (minimum scaffold edge weight)
    probe_floor = 1/nV^3 (minimum probe edge weight)
    """
    return {
        'H0': 1.0,
        'amp_coeff': float(nV - 1) / nV if nV > 0 else 0.0,
        'scaffold_floor': 1.0 / (nV * nV) if nV > 0 else 0.0,
        'probe_floor': 1.0 / (nV * nV * nV) if nV > 0 else 0.0,
    }


# Per-channel mixing time


cdef f64 _lambda2_from_evals(const f64* evals, int n) noexcept nogil:
    """Extract smallest positive eigenvalue (spectral gap) from sorted evals."""
    cdef int k
    for k in range(n):
        if evals[k] > 1e-10:
            return evals[k]
    return 0.0


def hat_eigen(np.ndarray[f64, ndim=2] hat, int nE):
    """Eigendecompose a single hat operator via LAPACK dsyev_.

    Parameters
    ----------
    hat : f64[nE, nE]
        Trace-normalized typed Laplacian.
    nE : int

    Returns
    -------
    evals : f64[nE], ascending
    evecs : f64[nE, nE], columns are eigenvectors
    """
    if nE == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(hat.copy())
    cdef np.ndarray[f64, ndim=1] evals = np.empty(nE, dtype=np.float64)

    lp_eigh(&A_F[0, 0], &evals[0], nE)

    cdef int i
    cdef f64[::1] ev = evals
    for i in range(nE):
        if fabs(ev[i]) < 1e-12:
            ev[i] = 0.0
        elif ev[i] < 0.0 and fabs(ev[i]) < 1e-9:
            ev[i] = 0.0

    cdef np.ndarray[f64, ndim=2] evecs = np.ascontiguousarray(A_F)
    return evals, evecs


def hat_eigen_all(list hats, int nhats, int nE):
    """Eigendecompose all hat operators. Returns list of (evals, evecs).

    Parameters
    ----------
    hats : list of f64[nE, nE]
    nhats : int
    nE : int

    Returns
    -------
    list of (evals f64[nE], evecs f64[nE, nE]) per hat.
    """
    result = []
    for k in range(nhats):
        result.append(hat_eigen(hats[k], nE))
    return result


def per_channel_mixing_time(np.ndarray[f64, ndim=1] hat_evals, int nE):
    """Per-channel mixing time from pre-computed hat eigenvalues.

    mu_X = ln(nE) / lambda_2(hat_L_X).

    Parameters
    ----------
    hat_evals : f64[nE]
        Eigenvalues of a single hat operator (ascending).
    nE : int

    Returns
    -------
    float
        Mixing time for this channel. inf if no spectral gap.
    """
    if nE <= 1:
        return float('inf')

    cdef f64[::1] ev = hat_evals
    cdef int k
    cdef f64 lambda2 = 0

    for k in range(nE):
        if ev[k] > 1e-10:
            lambda2 = ev[k]
            break

    if lambda2 < 1e-15:
        return float('inf')
    return float(log(<f64>nE) / lambda2)


def per_channel_mixing_times_from_evals(list hat_evals_list, int nhats, int nE):
    """Per-channel mixing times from pre-computed hat eigenvalues.

    Parameters
    ----------
    hat_evals_list : list of f64[nE]
        Eigenvalues per hat from hat_eigen_all.
    nhats : int
    nE : int

    Returns
    -------
    f64[nhats]
    """
    cdef np.ndarray[f64, ndim=1] times = np.empty(nhats, dtype=np.float64)
    cdef int k
    for k in range(nhats):
        times[k] = per_channel_mixing_time(hat_evals_list[k], nE)
    return times


def per_channel_mixing_times(list hats, int nhats, int nE):
    """Per-channel mixing times, eigendecomposing each hat internally.

    Convenience wrapper when hat eigendata is not already cached.

    Parameters
    ----------
    hats : list of f64[nE, nE]
    nhats : int
    nE : int

    Returns
    -------
    f64[nhats]
    """
    cdef np.ndarray[f64, ndim=1] times = np.empty(nhats, dtype=np.float64)
    cdef int k
    for k in range(nhats):
        evals, _ = hat_eigen(hats[k], nE)
        times[k] = per_channel_mixing_time(evals, nE)
    return times


def mixing_time_anisotropy(np.ndarray[f64, ndim=1] channel_times,
                            int nhats):
    """Ratios between per-channel mixing times.

    Computes pairwise ratios tau_i / tau_j, finds the fastest-mixing
    and slowest channels, and returns the anisotropy ratio.

    Parameters
    ----------
    channel_times : f64[nhats]
        Per-channel mixing times from per_channel_mixing_times.
    nhats : int
        Number of channels.

    Returns
    -------
    dict
        ratios : f64[nhats, nhats] pairwise tau_i / tau_j
        dominant_channel : int  (fastest, smallest finite tau)
        slowest_channel : int   (slowest, largest finite tau)
        anisotropy : float      max(finite tau) / min(finite tau)
    """
    cdef np.ndarray[f64, ndim=2] ratios = np.ones((nhats, nhats), dtype=np.float64)
    cdef f64[::1] tv = channel_times
    cdef f64[:, ::1] rv = ratios
    cdef int i, j
    cdef f64 ti, tj

    for i in range(nhats):
        for j in range(nhats):
            ti = tv[i]
            tj = tv[j]
            if tj > 1e-15 and tj != float('inf'):
                if ti == float('inf'):
                    rv[i, j] = float('inf')
                else:
                    rv[i, j] = ti / tj
            elif ti == float('inf') and tj == float('inf'):
                rv[i, j] = 1.0
            else:
                rv[i, j] = float('inf')

    # Find dominant (smallest finite) and slowest (largest finite)
    cdef f64 best = float('inf')
    cdef f64 worst = 0.0
    cdef int dom = 0, slow = 0

    for i in range(nhats):
        ti = tv[i]
        if ti < best and ti > 0:
            best = ti
            dom = i
        if ti > worst and ti != float('inf'):
            worst = ti
            slow = i

    cdef f64 aniso = worst / best if best > 1e-15 else float('inf')

    return {
        'ratios': ratios,
        'dominant_channel': dom,
        'slowest_channel': slow,
        'anisotropy': float(aniso),
    }


# Face-void dipole


cdef void _face_void_dipole(const f64* psi, const f64* B2,
                             const f64* Bvoid,
                             f64* face_aff, f64* void_aff,
                             int nE, int nF, int n_voids,
                             f64 psi_norm_sq) noexcept nogil:
    """Face and void affinity of an edge signal.

    face_affinity = sum_f |psi^T B2[:,f]|^2 / ||psi||^2
    void_affinity = sum_v |psi^T Bvoid[:,v]|^2 / ||psi||^2
    """
    cdef int f, v, e
    cdef f64 dot, inv_norm

    face_aff[0] = 0.0
    void_aff[0] = 0.0

    if psi_norm_sq < 1e-30:
        return

    inv_norm = 1.0 / psi_norm_sq

    for f in range(nF):
        dot = 0.0
        for e in range(nE):
            dot += psi[e] * B2[e * nF + f]
        face_aff[0] += dot * dot * inv_norm

    if Bvoid != NULL and n_voids > 0:
        for v in range(n_voids):
            dot = 0.0
            for e in range(nE):
                dot += psi[e] * Bvoid[e * n_voids + v]
            void_aff[0] += dot * dot * inv_norm


def face_void_dipole(np.ndarray[f64, ndim=1] psi,
                      np.ndarray[f64, ndim=2] B2,
                      Bvoid_in,
                      int nE, int nF):
    """Face-void dipole of an edge signal.

    Projects an edge signal onto the realized face basis (B2 columns)
    and the void basis (Bvoid columns), measuring how much signal
    energy flows through each. The dipole ratio separates signals
    that operate through existing higher-order structure (face-mediated)
    from those that operate through structural gaps (void-mediated).

    face_affinity = sum_f |psi^T B2[:,f]|^2 / ||psi||^2
    void_affinity = sum_v |psi^T Bvoid[:,v]|^2 / ||psi||^2
    dipole_ratio  = (face - void) / (face + void)

    Parameters
    ----------
    psi : f64[nE]
        Edge signal.
    B2 : f64[nE, nF]
        Edge-face boundary operator (realized faces).
    Bvoid_in : f64[nE, n_voids] or None
        Void boundary operator. None if no voids.
    nE : int
        Number of edges.
    nF : int
        Number of realized faces.

    Returns
    -------
    dict
        face_affinity : float >= 0
        void_affinity : float >= 0
        dipole_ratio : float in [-1, 1]
            +1 = entirely face-mediated, -1 = entirely void-mediated,
            0 = balanced.
        total_projection : float
            face_affinity + void_affinity.
    """
    cdef f64[::1] pv = psi
    cdef f64 psi_norm_sq = 0.0
    cdef int e

    for e in range(nE):
        psi_norm_sq += pv[e] * pv[e]

    cdef f64 fa = 0.0, va = 0.0
    cdef int n_voids = 0
    cdef np.ndarray[f64, ndim=2] Bvoid

    if nF > 0:
        if Bvoid_in is not None:
            Bvoid = np.ascontiguousarray(Bvoid_in, dtype=np.float64)
            n_voids = Bvoid.shape[1]
            _face_void_dipole(&pv[0], &B2[0, 0], &Bvoid[0, 0],
                               &fa, &va, nE, nF, n_voids, psi_norm_sq)
        else:
            _face_void_dipole(&pv[0], &B2[0, 0], NULL,
                               &fa, &va, nE, nF, 0, psi_norm_sq)
    elif Bvoid_in is not None:
        Bvoid = np.ascontiguousarray(Bvoid_in, dtype=np.float64)
        n_voids = Bvoid.shape[1]
        if psi_norm_sq > 1e-30:
            _face_void_dipole(&pv[0], NULL, &Bvoid[0, 0],
                               &fa, &va, nE, 0, n_voids, psi_norm_sq)

    cdef f64 total = fa + va
    cdef f64 dipole = 0.0
    if total > 1e-30:
        dipole = (fa - va) / total

    return {
        'face_affinity': float(fa),
        'void_affinity': float(va),
        'dipole_ratio': float(dipole),
        'total_projection': float(total),
    }
