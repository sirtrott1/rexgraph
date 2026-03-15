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

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    binary_search_i32,
    binary_search_contains_i32,
    should_use_dense_eigen,
    get_EPSILON_NORM,
)

np.import_array()


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

    # Count first, then fill
    tri_list = []

    for v in range(nV):
        lo_v = ap[v]
        hi_v = ap[v + 1]
        for ni in range(lo_v, hi_v):
            u = ai[ni]
            if u >= v:
                continue
            # Edge v-u exists (edge index ae[ni])
            e_vu = ae[ni]
            lo_u = ap[u]
            hi_u = ap[u + 1]
            for nj in range(ni + 1, hi_v):
                w = ai[nj]
                if w >= v:
                    continue
                if w <= u:
                    continue
                # Check if edge u-w exists
                pos = binary_search_i32(&ai[lo_u], hi_u - lo_u, w)
                if pos >= 0:
                    e_vw = ae[nj]
                    e_uw = ae[lo_u + pos]
                    tri_list.append((e_vu, e_vw, e_uw))

    nT = len(tri_list)
    if nT == 0:
        return np.zeros((0, 3), dtype=np.int32), 0

    cdef np.ndarray[i32, ndim=2] tri_edges = np.empty((nT, 3), dtype=np.int32)
    for k in range(nT):
        tri_edges[k, 0] = tri_list[k][0]
        tri_edges[k, 1] = tri_list[k][1]
        tri_edges[k, 2] = tri_list[k][2]

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

    Returns (realized[nT], void_indices[n_voids], n_voids).
    """

    if nT == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), 0

    # Get B2 as dense for simplicity (nF is usually small)
    B2_d = np.asarray(B2, dtype=np.float64)

    nF = B2_d.shape[1] if B2_d.ndim == 2 else 0

    # Build set of realized edge triples
    realized_set = set()
    for f in range(nF):
        col = B2_d[:, f]
        nonzero_edges = tuple(sorted(np.where(np.abs(col) > 0.5)[0]))
        if len(nonzero_edges) == 3:
            realized_set.add(nonzero_edges)

    realized = np.zeros(nT, dtype=np.int32)
    void_list = []

    for k in range(nT):
        key = tuple(sorted([int(tri_edges[k, 0]), int(tri_edges[k, 1]), int(tri_edges[k, 2])]))
        if key in realized_set:
            realized[k] = 1
        else:
            void_list.append(k)

    void_indices = np.array(void_list, dtype=np.int32)
    return realized, void_indices, len(void_list)


# Void boundary operator

def build_void_boundary(B1, B2, tri_edges, Py_ssize_t nT,
                         Py_ssize_t nV, Py_ssize_t nE):
    """Build Bvoid: nE x n_voids matrix of void boundary cycles.

    For each void triangle, find the +/-1 kernel of B1 restricted to 3 edges.
    Guarantees B1 @ Bvoid = 0 by construction.
    """

    _, void_indices, n_voids = classify_triangles(B2, tri_edges, nT, nE)

    if n_voids == 0:
        return None, void_indices, 0

    # Get dense B1 for sign computation
    B1_d = np.asarray(B1, dtype=np.float64)

    # COO construction
    rows = []
    cols = []
    vals = []

    for col_idx in range(n_voids):
        k = int(void_indices[col_idx])
        e0 = int(tri_edges[k, 0])
        e1 = int(tri_edges[k, 1])
        e2 = int(tri_edges[k, 2])

        # Try all 8 sign patterns
        found = False
        for signs in range(8):
            s = [1 if signs & (1 << b) else -1 for b in range(3)]
            col = np.zeros(nE, dtype=np.float64)
            col[e0] = s[0]
            col[e1] = s[1]
            col[e2] = s[2]
            if np.max(np.abs(B1_d @ col)) < 1e-10:
                rows.extend([e0, e1, e2])
                cols.extend([col_idx, col_idx, col_idx])
                vals.extend([s[0], s[1], s[2]])
                found = True
                break

        if not found:
            # Fallback (shouldn't happen for valid triangles)
            rows.extend([e0, e1, e2])
            cols.extend([col_idx, col_idx, col_idx])
            vals.extend([1.0, 1.0, -1.0])

    Bvoid = np.zeros((nE, n_voids), dtype=np.float64)
    for _i in range(len(rows)):
        Bvoid[rows[_i], cols[_i]] = vals[_i]

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


def harmonic_content_all(Bvoid, evals_L1, evecs_L1,
                          Py_ssize_t n_voids, Py_ssize_t nE):
    """Harmonic content for all voids."""

    eta = np.zeros(n_voids, dtype=np.float64)

    if hasattr(Bvoid, 'toarray'):
        Bvoid_d = np.asarray(Bvoid.toarray(), dtype=np.float64)
    else:
        Bvoid_d = np.asarray(Bvoid, dtype=np.float64)

    evals = np.asarray(evals_L1, dtype=np.float64)
    evecs = np.asarray(evecs_L1, dtype=np.float64)

    for k in range(n_voids):
        bv = Bvoid_d[:, k]
        eta[k] = harmonic_content_single(bv, evals, evecs, nE)

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
    """Void character for all voids."""

    chi_void = np.zeros((n_voids, nhats), dtype=np.float64)

    if hasattr(Bvoid, 'toarray'):
        Bvoid_d = np.asarray(Bvoid.toarray(), dtype=np.float64)
    else:
        Bvoid_d = np.asarray(Bvoid, dtype=np.float64)

    for k in range(n_voids):
        bv = Bvoid_d[:, k]
        chi_void[k, :] = void_character_single(bv, RL, hats, nhats, nE)

    return chi_void


# Void strain

def void_strain(Bvoid, Py_ssize_t n_voids, Py_ssize_t nE):
    """S^void = sum ||bv||^2 = tr(Lvoid)."""

    if n_voids == 0:
        return 0.0

    if hasattr(Bvoid, 'toarray'):
        Bvoid_d = np.asarray(Bvoid.toarray(), dtype=np.float64)
    else:
        Bvoid_d = np.asarray(Bvoid, dtype=np.float64)

    total = 0.0
    for k in range(n_voids):
        total += float(np.dot(Bvoid_d[:, k], Bvoid_d[:, k]))
    return total


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
    """Check L_up + Lvoid = [B2|Bvoid][B2|Bvoid]^T."""

    B2_d = np.asarray(B2, dtype=np.float64)

    L_up = B2_d @ B2_d.T

    if Bvoid is None:
        Lvoid = np.zeros((nE, nE), dtype=np.float64)
        Bfull = B2_d
    else:
        if hasattr(Bvoid, 'toarray'):
            Bv_d = np.asarray(Bvoid.toarray(), dtype=np.float64)
        else:
            Bv_d = np.asarray(Bvoid, dtype=np.float64)
        Lvoid = Bv_d @ Bv_d.T
        Bfull = np.hstack([B2_d, Bv_d])

    L_full = Bfull @ Bfull.T
    residual = float(np.max(np.abs(L_up + Lvoid - L_full)))

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

    # Step 3: Lvoid
    if False:
        Lvoid = Bvoid @ Bvoid.T
    else:
        Bv_d = np.asarray(Bvoid, dtype=np.float64)
        Lvoid = Bv_d @ Bv_d.T
    result['Lvoid'] = Lvoid

    # Step 4: harmonic content
    if evals_L1 is not None and evecs_L1 is not None:
        eta_arr = harmonic_content_all(Bvoid, evals_L1, evecs_L1, n_voids, nE)
    else:
        eta_arr = np.full(n_voids, float('nan'), dtype=np.float64)
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
