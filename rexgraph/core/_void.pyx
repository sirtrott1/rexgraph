# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._void - Void spectral theory.

The void complex records potential faces that could exist but don't.
Each void v has a boundary cycle bv in ker(B1) with harmonic content
eta(v) in [0,1]. If eta > 0, filling v decreases beta_1 by 1.

Key identities:
    B1 @ Bvoid = 0 (void boundary cycles are in ker(B1))
    L_up + Lvoid = [B2|Bvoid][B2|Bvoid]^T
    S^void = tr(Lvoid) = sum ||bv||^2
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from scipy.sparse import csc_matrix

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    binary_search_i32,
    binary_search_contains_i32,
    should_use_dense_eigen,
    get_EPSILON_NORM,
)

np.import_array()


# Orientation helpers (direct cycle-sign computation)

cdef inline i32 _shared_vertex(i32 pa, i32 ma, i32 pb, i32 mb) noexcept nogil:
    """Return the vertex common to edges a={pa,ma} and b={pb,mb}."""
    if pa == pb or pa == mb:
        return pa
    return ma


# Triangle enumeration

@cython.boundscheck(False)
@cython.wraparound(False)
def find_potential_triangles_i32(np.ndarray[i32, ndim=1] adj_ptr,
                                  np.ndarray[i32, ndim=1] adj_idx,
                                  np.ndarray[i32, ndim=1] adj_edge,
                                  Py_ssize_t nV, Py_ssize_t nE):
    """Find all triangles in the 1-skeleton via adjacency CSR.

    For each vertex v, for each pair of neighbors (u, w) with u < w < v (to avoid duplicates):
        if edge (u,w) exists: triangle (u,w,v) found.

    Returns (tri_edges[nT, 3], nT) where tri_edges[k] = edge indices of triangle k.
    """
    cdef i32[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge
    cdef Py_ssize_t v, ni, nj, lo_v, hi_v, lo_u, hi_u
    cdef i32 u, w
    cdef idx_t pos
    cdef Py_ssize_t nT = 0, k = 0

    # Pass 1: count triangles (matches the docstring's "count first, then
    # fill" - the previous version built a Python list of tuples instead).
    for v in range(nV):
        lo_v = ap[v]
        hi_v = ap[v + 1]
        for ni in range(lo_v, hi_v):
            u = ai[ni]
            if u >= v:
                continue
            lo_u = ap[u]
            hi_u = ap[u + 1]
            for nj in range(ni + 1, hi_v):
                w = ai[nj]
                if w >= v or w <= u:
                    continue
                if binary_search_i32(&ai[lo_u], hi_u - lo_u, w) >= 0:
                    nT += 1

    if nT == 0:
        return np.zeros((0, 3), dtype=np.int32), 0

    cdef np.ndarray[i32, ndim=2] tri_edges = np.empty((nT, 3), dtype=np.int32)
    cdef i32[:, ::1] te = tri_edges

    # Pass 2: fill the preallocated array directly.
    for v in range(nV):
        lo_v = ap[v]
        hi_v = ap[v + 1]
        for ni in range(lo_v, hi_v):
            u = ai[ni]
            if u >= v:
                continue
            lo_u = ap[u]
            hi_u = ap[u + 1]
            for nj in range(ni + 1, hi_v):
                w = ai[nj]
                if w >= v or w <= u:
                    continue
                pos = binary_search_i32(&ai[lo_u], hi_u - lo_u, w)
                if pos >= 0:
                    te[k, 0] = ae[ni]
                    te[k, 1] = ae[nj]
                    te[k, 2] = ae[lo_u + pos]
                    k += 1

    return tri_edges, nT


def find_potential_triangles(adj_ptr, adj_idx, adj_edge,
                              Py_ssize_t nV, Py_ssize_t nE):
    """Dispatcher."""
    return find_potential_triangles_i32(
        np.asarray(adj_ptr, dtype=np.int32),
        np.asarray(adj_idx, dtype=np.int32),
        np.asarray(adj_edge, dtype=np.int32),
        nV, nE)


# Classify triangles as realized or void

def classify_triangles(B2, tri_edges, Py_ssize_t nT, Py_ssize_t nE):
    """For each potential triangle, check if it matches a column of B2.

    Realized faces are encoded as sorted-edge integer keys, sorted once,
    then each potential triangle is matched by binary search - no Python
    set/tuple objects in the hot loop.

    Returns (realized[nT], void_indices[n_voids], n_voids).
    """
    if nT == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), 0

    cdef np.ndarray[f64, ndim=2] B2_d = np.ascontiguousarray(
        np.asarray(B2, dtype=np.float64)) if (B2 is not None and
        np.asarray(B2).ndim == 2) else np.zeros((nE, 0), dtype=np.float64)
    cdef Py_ssize_t nF = B2_d.shape[1]
    cdef f64[:, ::1] b2 = B2_d
    cdef i64 nE64 = <i64>nE

    cdef np.ndarray[i32, ndim=2] te = np.ascontiguousarray(
        np.asarray(tri_edges, dtype=np.int32))
    cdef i32[:, ::1] tem = te

    # Encode realized faces (columns of B2 with exactly 3 nonzero edges).
    cdef np.ndarray[i64, ndim=1] rk = np.empty(nF, dtype=np.int64)
    cdef i64[::1] rkv = rk
    cdef Py_ssize_t f, e, cnt, nR = 0
    cdef i32 a0, a1, a2, lo, hi, mid
    for f in range(nF):
        cnt = 0
        a0 = a1 = a2 = -1
        for e in range(nE):
            if b2[e, f] > 0.5 or b2[e, f] < -0.5:
                if cnt == 0:
                    a0 = <i32>e
                elif cnt == 1:
                    a1 = <i32>e
                elif cnt == 2:
                    a2 = <i32>e
                cnt += 1
                if cnt > 3:
                    break
        if cnt == 3:
            rkv[nR] = _sorted_key(a0, a1, a2, nE64)
            nR += 1

    cdef np.ndarray[i64, ndim=1] realized_keys = np.sort(rk[:nR])
    cdef i64[::1] rks = realized_keys

    cdef np.ndarray[i32, ndim=1] realized = np.zeros(nT, dtype=np.int32)
    cdef i32[::1] rz = realized
    cdef np.ndarray[i32, ndim=1] void_idx = np.empty(nT, dtype=np.int32)
    cdef i32[::1] vz = void_idx
    cdef Py_ssize_t k, nv = 0
    cdef i64 key

    for k in range(nT):
        key = _sorted_key(tem[k, 0], tem[k, 1], tem[k, 2], nE64)
        if nR > 0 and _i64_contains(&rks[0], nR, key):
            rz[k] = 1
        else:
            vz[nv] = <i32>k
            nv += 1

    return realized, void_idx[:nv].copy(), nv


cdef inline i64 _sorted_key(i32 e0, i32 e1, i32 e2, i64 nE64) noexcept nogil:
    """Encode a sorted edge-triple as a single int64 key."""
    cdef i32 lo = e0, hi = e0, mid
    if e1 < lo: lo = e1
    if e2 < lo: lo = e2
    if e1 > hi: hi = e1
    if e2 > hi: hi = e2
    mid = (e0 + e1 + e2) - lo - hi
    return (<i64>lo) * nE64 * nE64 + (<i64>mid) * nE64 + (<i64>hi)


cdef inline bint _i64_contains(i64* arr, Py_ssize_t n, i64 key) noexcept nogil:
    """Binary search for key in a sorted i64 array."""
    cdef Py_ssize_t lo = 0, hi = n - 1, mid
    while lo <= hi:
        mid = (lo + hi) >> 1
        if arr[mid] == key:
            return True
        elif arr[mid] < key:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


# Void boundary operator

def build_void_boundary(B1, B2, tri_edges, Py_ssize_t nT,
                         Py_ssize_t nV, Py_ssize_t nE):
    """Build Bvoid: nE x n_voids CSC of void boundary cycles.

    Each void triangle's boundary cycle is oriented directly from the
    edge endpoints (one shared-vertex traversal), so B1 @ Bvoid = 0 by
    construction. The result is sparse (exactly 3 nonzeros per column) -
    the previous version brute-forced 8 sign patterns with a full dense
    B1 matvec per void and stored a dense nE x n_voids matrix.
    """
    _, void_indices, n_voids = classify_triangles(B2, tri_edges, nT, nE)

    if n_voids == 0:
        return None, void_indices, 0

    # +1 / -1 endpoint row of each edge, extracted once (vectorized).
    cdef np.ndarray[f64, ndim=2] B1_d = np.ascontiguousarray(
        np.asarray(B1, dtype=np.float64))
    cdef np.ndarray[i32, ndim=1] pe = np.ascontiguousarray(
        np.argmax(B1_d, axis=0).astype(np.int32))
    cdef np.ndarray[i32, ndim=1] me = np.ascontiguousarray(
        np.argmin(B1_d, axis=0).astype(np.int32))
    cdef i32[::1] pv = pe, mv = me

    cdef np.ndarray[i32, ndim=2] te = np.ascontiguousarray(
        np.asarray(tri_edges, dtype=np.int32))
    cdef i32[:, ::1] tem = te
    cdef np.ndarray[i32, ndim=1] vi = np.ascontiguousarray(
        np.asarray(void_indices, dtype=np.int32))
    cdef i32[::1] viv = vi

    cdef np.ndarray[f64, ndim=1] data = np.empty(3 * n_voids, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] indices = np.empty(3 * n_voids, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] indptr = np.arange(
        0, 3 * (n_voids + 1), 3, dtype=np.int32)
    cdef f64[::1] dv = data
    cdef i32[::1] iv = indices

    cdef Py_ssize_t c, k
    cdef i32 e0, e1, e2, a, cc
    cdef f64 alpha0, beta1, delta2

    for c in range(n_voids):
        k = viv[c]
        e0 = tem[k, 0]; e1 = tem[k, 1]; e2 = tem[k, 2]
        # Triangle vertices: a shared by (e0,e1), cc shared by (e0,e2).
        a = _shared_vertex(pv[e0], mv[e0], pv[e1], mv[e1])
        cc = _shared_vertex(pv[e0], mv[e0], pv[e2], mv[e2])
        alpha0 = 1.0 if pv[e0] == a else -1.0
        beta1 = 1.0 if pv[e1] == a else -1.0
        delta2 = 1.0 if pv[e2] == cc else -1.0
        dv[3 * c] = 1.0;              iv[3 * c] = e0
        dv[3 * c + 1] = -alpha0 * beta1; iv[3 * c + 1] = e1
        dv[3 * c + 2] = alpha0 * delta2; iv[3 * c + 2] = e2

    Bvoid = csc_matrix((data, indices, indptr), shape=(nE, n_voids))
    return Bvoid, void_indices, n_voids


# Harmonic content

def harmonic_content_single(bv, evals_L1, evecs_L1, Py_ssize_t nE):
    """eta = ||proj_harm(bv)||^2 / ||bv||^2.

    Projects bv onto ker(L1) (harmonic space).
    """
    cdef f64 bv_norm_sq = float(np.dot(bv, bv))
    if bv_norm_sq < 1e-15:
        return 0.0

    # Project onto harmonic eigenvectors (eigenvalue near zero)
    harm_mask = np.abs(evals_L1) < 1e-10
    if not np.any(harm_mask):
        return 0.0

    harm_vecs = evecs_L1[:, harm_mask]
    coeffs = harm_vecs.T @ bv
    proj = harm_vecs @ coeffs
    proj_norm_sq = float(np.dot(proj, proj))

    return proj_norm_sq / bv_norm_sq


cdef _void_compact(Bvoid, Py_ssize_t n_voids):
    """Extract (edges[n_voids,3], signs[n_voids,3]) from a CSC Bvoid.

    Each void column has exactly 3 nonzeros, so the CSC data/indices lay
    out as consecutive triples. Falls back to a dense scan if the layout
    is unexpected.
    """
    if hasattr(Bvoid, 'indptr'):
        idx = np.asarray(Bvoid.indices, dtype=np.int32)
        dat = np.asarray(Bvoid.data, dtype=np.float64)
        if idx.shape[0] == 3 * n_voids:
            return idx.reshape(n_voids, 3), dat.reshape(n_voids, 3)
    # Fallback: densify and scan (rare / unexpected layout).
    Bd = np.asarray(Bvoid.toarray() if hasattr(Bvoid, 'toarray') else Bvoid,
                    dtype=np.float64)
    ee = np.zeros((n_voids, 3), dtype=np.int32)
    ss = np.zeros((n_voids, 3), dtype=np.float64)
    for c in range(n_voids):
        nz = np.nonzero(np.abs(Bd[:, c]) > 1e-12)[0][:3]
        ee[c, :len(nz)] = nz
        ss[c, :len(nz)] = Bd[nz, c]
    return ee, ss


def harmonic_content_all(Bvoid, evals_L1, evecs_L1,
                          Py_ssize_t n_voids, Py_ssize_t nE):
    """Harmonic content eta_k = ||proj_ker(L1) bv_k||^2 / ||bv_k||^2.

    Each bv has 3 nonzeros, so the projection touches only 3 rows of the
    harmonic eigenbasis; this gathers those rows and batches all voids
    with no per-void Python loop and no dense Bvoid.
    """
    if n_voids == 0:
        return np.zeros(0, dtype=np.float64)

    evals = np.asarray(evals_L1, dtype=np.float64)
    evecs = np.asarray(evecs_L1, dtype=np.float64)
    harm_mask = np.abs(evals) < 1e-10
    if not np.any(harm_mask):
        return np.zeros(n_voids, dtype=np.float64)
    harm = np.ascontiguousarray(evecs[:, harm_mask])   # nE x h (orthonormal)

    ee, ss = _void_compact(Bvoid, n_voids)
    # C[k,:] = sum_a ss[k,a] * harm[ee[k,a], :]   (n_voids x h)
    C = (ss[:, 0:1] * harm[ee[:, 0]]
         + ss[:, 1:2] * harm[ee[:, 1]]
         + ss[:, 2:3] * harm[ee[:, 2]])
    proj_norm_sq = np.einsum('ij,ij->i', C, C)
    bv_norm_sq = np.einsum('ij,ij->i', ss, ss)
    eta = np.zeros(n_voids, dtype=np.float64)
    nz = bv_norm_sq > 1e-15
    eta[nz] = proj_norm_sq[nz] / bv_norm_sq[nz]
    return eta


def harmonic_content_all_sparse(B1, B2, Bvoid, Py_ssize_t n_voids, Py_ssize_t nE):
    """Eigen-free harmonic content eta_k = ||P_ker(L1) bv_k||^2 / ||bv_k||^2, via the
    combinatorial LOW-RANK harmonic projector P_H = H(HᵀH)⁻¹Hᵀ (H =
    harmonic_basis_from_boundaries(B1, B2)) instead of a dense eigendecomposition of
    L1. H spans the same ker(L1) as the dense harmonic eigenbasis, so eta is IDENTICAL
    (to ~1e-9) - but it is scale-free: the void harmonic content is now available even
    when no dense L1 spectrum was computed (previously NaN on large graphs).

    eta_k = bv_kᵀ P_H bv_k / ||bv_k||^2 = (Hᵀbv_k)ᵀ (HᵀH)⁻¹ (Hᵀbv_k) / ||bv_k||^2,
    batched over all voids: G = Hᵀ Bvoid (dim_H x n_voids), one shared sparse
    factorization of HᵀH applied to the whole block, no per-void loop, no dense nE x nE.
    """
    if n_voids == 0:
        return np.zeros(0, dtype=np.float64)
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from rexgraph.harmonic_sparse import harmonic_basis_from_boundaries

    B1s = B1.tocsc() if sp.issparse(B1) else sp.csc_matrix(np.asarray(B1, dtype=np.float64))
    B2s = None
    if B2 is not None:
        B2s = B2.tocsr() if sp.issparse(B2) else sp.csr_matrix(np.asarray(B2, dtype=np.float64))
    H = harmonic_basis_from_boundaries(B1s, B2s)

    Bv = Bvoid.tocsc() if sp.issparse(Bvoid) else sp.csc_matrix(np.asarray(Bvoid, dtype=np.float64))
    bv_norm_sq = np.asarray(Bv.multiply(Bv).sum(axis=0)).ravel()
    eta = np.zeros(n_voids, dtype=np.float64)
    cdef Py_ssize_t k = H.shape[1]
    if k == 0:
        return eta

    Hs = H.tocsr()
    G = np.asarray((Hs.T @ Bv).todense())              # dim_H x n_voids
    HtH = (Hs.T @ Hs).tocsc()                           # dim_H x dim_H SPD
    try:
        Y = sla.splu(HtH).solve(G)
    except Exception:
        Y = np.linalg.solve(np.asarray(HtH.todense()), G)
    proj_norm_sq = np.einsum('ij,ij->j', G, np.asarray(Y).reshape(k, n_voids))
    nz = bv_norm_sq > 1e-15
    eta[nz] = proj_norm_sq[nz] / bv_norm_sq[nz]
    return eta


# Void character

def void_character_single(bv, RL, hats, Py_ssize_t nhats, Py_ssize_t nE):
    """chi^void(k) = bv^T hat_k bv / (bv^T RL bv).

    bv has 3 nonzeros, so this touches at most 9 entries of each matrix.
    """

    # bv^T RL bv
    if False:
        rl_bv = RL.dot(bv)
    else:
        rl_bv = np.asarray(RL, dtype=np.float64) @ bv
    erl = float(np.dot(bv, rl_bv))

    chi_v = np.zeros(nhats, dtype=np.float64)
    if erl < 1e-15:
        chi_v[:] = 1.0 / nhats if nhats > 0 else 0.0
        return chi_v

    for k in range(nhats):
        hat_k = hats[k]
        if False:
            hat_bv = hat_k.dot(bv)
        else:
            hat_bv = np.asarray(hat_k, dtype=np.float64) @ bv
        chi_v[k] = float(np.dot(bv, hat_bv)) / erl

    return chi_v


def void_character_all(Bvoid, RL, hats, Py_ssize_t nhats,
                        Py_ssize_t n_voids, Py_ssize_t nE):
    """chi^void[k,j] = bv^T hat_j bv / bv^T RL bv for all voids.

    bv has 3 nonzeros, so each quadratic form is the sum over a 3x3
    submatrix (9 entries) - computed directly instead of a full dense
    matvec per void. No per-void Python loop, no dense Bvoid, and hats
    are consumed one at a time (no nhats*nE*nE stack).
    """
    chi_void = np.zeros((n_voids, nhats), dtype=np.float64)
    if n_voids == 0 or nhats == 0:
        return chi_void

    ee, ss = _void_compact(Bvoid, n_voids)
    cdef i32[:, ::1] eem = np.ascontiguousarray(ee, dtype=np.int32)
    cdef f64[:, ::1] ssm = np.ascontiguousarray(ss, dtype=np.float64)
    cdef f64[:, ::1] rl = np.ascontiguousarray(np.asarray(RL, dtype=np.float64))
    cdef f64[:, ::1] chi = chi_void

    cdef np.ndarray[f64, ndim=1] den = np.empty(n_voids, dtype=np.float64)
    cdef f64[::1] dv = den
    cdef Py_ssize_t k, a, b, j
    cdef i32 ea, eb
    cdef f64 sa, acc

    # Pass 1: denominators bv^T RL bv.
    for k in range(n_voids):
        acc = 0.0
        for a in range(3):
            ea = eem[k, a]; sa = ssm[k, a]
            for b in range(3):
                eb = eem[k, b]
                acc += sa * ssm[k, b] * rl[ea, eb]
        dv[k] = acc

    # Pass 2: one hat at a time.
    cdef f64[:, ::1] ht
    cdef f64 num, d
    for j in range(nhats):
        ht = np.ascontiguousarray(np.asarray(hats[j], dtype=np.float64))
        for k in range(n_voids):
            d = dv[k]
            if d < 1e-15:
                chi[k, j] = 1.0 / nhats
                continue
            num = 0.0
            for a in range(3):
                ea = eem[k, a]; sa = ssm[k, a]
                for b in range(3):
                    eb = eem[k, b]
                    num += sa * ssm[k, b] * ht[ea, eb]
            chi[k, j] = num / d

    return chi_void


# Void strain

def void_strain(Bvoid, Py_ssize_t n_voids, Py_ssize_t nE):
    """S^void = sum ||bv||^2 = tr(Lvoid) = ||Bvoid||_F^2."""
    if n_voids == 0 or Bvoid is None:
        return 0.0
    if hasattr(Bvoid, 'data'):          # sparse: sum of squared nonzeros
        d = np.asarray(Bvoid.data, dtype=np.float64)
        return float(np.dot(d, d))
    Bv = np.asarray(Bvoid, dtype=np.float64)
    return float(np.sum(Bv * Bv))


# Filling prediction

def fills_beta(np.ndarray[f64, ndim=1] eta, Py_ssize_t n_voids):
    """fills_beta[k] = 1 if eta[k] > epsilon (filling changes beta_1)."""
    cdef np.ndarray[i32, ndim=1] fb = np.zeros(n_voids, dtype=np.int32)
    cdef i32[::1] fv = fb
    cdef f64[::1] ev = eta
    cdef Py_ssize_t k
    for k in range(n_voids):
        fv[k] = 1 if ev[k] > 1e-10 else 0
    return fb


# Void type decomposition

def void_type_decomposition(void_indices, tri_edges, edge_types,
                              Py_ssize_t n_voids, Py_ssize_t n_types):
    """Count voids by bitmask of edge types present."""
    n_combos = 1 << n_types
    counts = np.zeros(n_combos, dtype=np.int32)

    et = np.asarray(edge_types, dtype=np.int32)
    te = np.asarray(tri_edges, dtype=np.int32)
    vi = np.asarray(void_indices, dtype=np.int32)

    for k in range(n_voids):
        tri_k = int(vi[k])
        combo = 0
        combo |= (1 << et[te[tri_k, 0]])
        combo |= (1 << et[te[tri_k, 1]])
        combo |= (1 << et[te[tri_k, 2]])
        counts[combo] += 1

    return counts


# L_full identity check

def verify_void_identity(B2, Bvoid, Py_ssize_t nE, f64 tol=1e-10):
    """Check L_up + Lvoid = [B2|Bvoid][B2|Bvoid]^T, computed SPARSELY: the residual
    is formed as a sparse matrix difference and only its stored nonzeros are
    inspected (mirrors graded_boundary.verify_chain), so no nE x nE array is ever
    materialized. Returns identical (valid, max_abs_residual) to the dense version."""
    import scipy.sparse as sp

    def _csc(M):
        if M is None:
            return None
        return M.tocsc() if sp.issparse(M) else sp.csc_matrix(np.asarray(M, dtype=np.float64))

    B2s = _csc(B2)
    if B2s is None:
        B2s = sp.csc_matrix((nE, 0), dtype=np.float64)
    Bvs = _csc(Bvoid)

    L_up = B2s @ B2s.T
    if Bvs is None:
        Lvoid = sp.csc_matrix((nE, nE), dtype=np.float64)
        Bfull = B2s
    else:
        Lvoid = Bvs @ Bvs.T
        Bfull = sp.hstack([B2s, Bvs], format='csc')

    resid_mat = (L_up + Lvoid - Bfull @ Bfull.T).tocoo()
    residual = float(np.abs(resid_mat.data).max()) if resid_mat.nnz else 0.0

    return residual < tol, residual


# Combined builder

def build_void_complex(B1, B2, adj_ptr, adj_idx, adj_edge,
                        Py_ssize_t nV, Py_ssize_t nE,
                        RL=None, hats=None, Py_ssize_t nhats=0,
                        evals_L1=None, evecs_L1=None):
    """Build the complete void complex.

    Returns dict with Bvoid, Lvoid, n_voids, n_potential,
    eta, chi_void, fills_beta_arr, void_strain_val.
    """

    # Step 1: enumerate triangles
    tri_edges, nT = find_potential_triangles(adj_ptr, adj_idx, adj_edge, nV, nE)

    # Step 2: build void boundary
    Bvoid, void_indices, n_voids = build_void_boundary(B1, B2, tri_edges, nT, nV, nE)

    result = {
        'Bvoid': Bvoid,
        'n_voids': n_voids,
        'n_potential': int(nT),
        'tri_edges': tri_edges,
        'void_indices': void_indices,
    }

    if n_voids == 0:
        result['Lvoid'] = None
        result['eta'] = np.zeros(0, dtype=np.float64)
        result['chi_void'] = np.zeros((0, max(nhats, 1)), dtype=np.float64)
        result['fills_beta'] = np.zeros(0, dtype=np.int32)
        result['void_strain'] = 0.0
        return result

    # Step 3: Lvoid = Bvoid Bvoid^T. Naturally sparse (~3 nnz/col) and NO consumer
    # needs it dense: void nullity is read from the sparse Bvoid (agent pipeline),
    # and void_strain = tr(Lvoid) is computed directly from Bvoid below. Keep it
    # sparse to avoid the nE x nE materialization on the void path. VoidComplex.Lvoid
    # is typed `object`; `.toarray()` reproduces the old dense array bit-for-bit.
    result['Lvoid'] = (Bvoid @ Bvoid.T).tocsr()

    # Step 4: harmonic content eta_k = ||P_ker(L1) bv_k||^2 / ||bv_k||^2. Prefer the
    # dense eigenbasis when the spectral bundle already provided it (small graphs,
    # exact oracle); otherwise fall back to the EIGEN-FREE combinatorial low-rank
    # projector built from B1/B2 (== the dense value to ~1e-9), so void harmonic
    # content is available at scale instead of NaN.
    if evals_L1 is not None and evecs_L1 is not None:
        eta_arr = harmonic_content_all(Bvoid, evals_L1, evecs_L1, n_voids, nE)
    else:
        eta_arr = harmonic_content_all_sparse(B1, B2, Bvoid, n_voids, nE)
    result['eta'] = eta_arr

    # Step 5: void character
    if RL is not None and hats is not None and nhats > 0:
        chi_void = void_character_all(Bvoid, RL, hats, nhats, n_voids, nE)
    else:
        chi_void = np.zeros((n_voids, max(nhats, 1)), dtype=np.float64)
    result['chi_void'] = chi_void

    # Step 6: fills_beta
    if not np.any(np.isnan(eta_arr)):
        result['fills_beta'] = fills_beta(eta_arr, n_voids)
    else:
        result['fills_beta'] = np.zeros(n_voids, dtype=np.int32)

    # Step 7: void strain
    result['void_strain'] = void_strain(Bvoid, n_voids, nE)

    return result
