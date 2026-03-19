# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._joins - Chain complex join operations.

Two complexes share structure through a vertex identification map.
All joins produce valid chain complexes (B1j @ B2j = 0 guaranteed
because restriction/extension of chain complexes preserves the chain condition).
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    get_EPSILON_NORM,
)

np.import_array()


# Shared vertex/edge map construction

def build_shared_vertex_map(labels_R, labels_S):
    """Map R-vertices to S-vertices by matching labels.

    Returns i32[nV_R] where shared[v_R] = v_S or -1.
    """
    label_to_s = {}
    for v_s, lab in enumerate(labels_S):
        label_to_s[lab] = v_s

    nV_R = len(labels_R)
    shared = np.full(nV_R, -1, dtype=np.int32)
    for v_r, lab in enumerate(labels_R):
        if lab in label_to_s:
            shared[v_r] = label_to_s[lab]

    return shared


def build_shared_edge_map(src_R, tgt_R, src_S, tgt_S, shared_vertices):
    """Map R-edges to S-edges by matching vertex pairs.

    Returns i32[nE_R] where shared_edges[e_R] = e_S or -1.
    """
    sv = np.asarray(shared_vertices, dtype=np.int32)
    nE_R = len(src_R)
    nE_S = len(src_S)

    # Build S edge lookup: (min_v, max_v) -> edge_idx
    s_pairs = {}
    for e in range(nE_S):
        s = int(src_S[e])
        t = int(tgt_S[e])
        key = (min(s, t), max(s, t))
        s_pairs[key] = e

    shared_edges = np.full(nE_R, -1, dtype=np.int32)
    for e in range(nE_R):
        s_r = int(src_R[e])
        t_r = int(tgt_R[e])
        s_s = int(sv[s_r]) if sv[s_r] >= 0 else -1
        t_s = int(sv[t_r]) if sv[t_r] >= 0 else -1
        if s_s >= 0 and t_s >= 0:
            key = (min(s_s, t_s), max(s_s, t_s))
            if key in s_pairs:
                shared_edges[e] = s_pairs[key]

    return shared_edges


# Inner join (intersection)

def inner_join(B1_R_dense, B2_R_dense, Py_ssize_t nV_R, Py_ssize_t nE_R, Py_ssize_t nF_R,
               B1_S_dense, B2_S_dense, Py_ssize_t nV_S, Py_ssize_t nE_S, Py_ssize_t nF_S,
               np.ndarray[i32, ndim=1] shared_vertices):
    """Inner join: cells present in both complexes.

    An edge qualifies if both endpoints are shared AND the same
    vertex pair is connected in both R and S.
    """
    B1R = np.asarray(B1_R_dense, dtype=np.float64)
    B1S = np.asarray(B1_S_dense, dtype=np.float64)
    B2R = np.asarray(B2_R_dense, dtype=np.float64)
    B2S = np.asarray(B2_S_dense, dtype=np.float64)
    cdef i32[::1] sv = shared_vertices

    # Vertex maps
    r_to_j = np.full(nV_R, -1, dtype=np.int32)
    s_to_j = np.full(nV_S, -1, dtype=np.int32)
    nVj = 0
    for v in range(nV_R):
        if sv[v] >= 0:
            r_to_j[v] = nVj
            s_to_j[sv[v]] = nVj
            nVj += 1

    # Find matching edges
    # Build S adjacency for lookup
    s_edge_lookup = {}
    for e in range(nE_S):
        endpoints_s = []
        for v in range(nV_S):
            if abs(B1S[v, e]) > 0.5:
                if s_to_j[v] >= 0:
                    endpoints_s.append(s_to_j[v])
        if len(endpoints_s) == 2:
            key = (min(endpoints_s[0], endpoints_s[1]),
                   max(endpoints_s[0], endpoints_s[1]))
            s_edge_lookup[key] = e

    matched = []
    for e in range(nE_R):
        endpoints_r = []
        for v in range(nV_R):
            if abs(B1R[v, e]) > 0.5:
                if r_to_j[v] >= 0:
                    endpoints_r.append(r_to_j[v])
        if len(endpoints_r) == 2:
            key = (min(endpoints_r[0], endpoints_r[1]),
                   max(endpoints_r[0], endpoints_r[1]))
            if key in s_edge_lookup:
                matched.append(e)

    nEj = len(matched)

    # Build B1j
    B1j = np.zeros((nVj, nEj), dtype=np.float64)
    for j, e in enumerate(matched):
        for v in range(nV_R):
            if abs(B1R[v, e]) > 0.5 and r_to_j[v] >= 0:
                B1j[r_to_j[v], j] = B1R[v, e]

    # Find faces with all edges in matched set
    matched_set = set(matched)
    face_list = []
    for f in range(nF_R):
        all_in = True
        for e in range(nE_R):
            if abs(B2R[e, f]) > 0.5 and e not in matched_set:
                all_in = False
                break
        if all_in:
            face_list.append(f)

    nFj = len(face_list)
    e_map = {old: new for new, old in enumerate(matched)}
    B2j = np.zeros((nEj, nFj), dtype=np.float64)
    for j, f in enumerate(face_list):
        for e in range(nE_R):
            if abs(B2R[e, f]) > 0.5 and e in e_map:
                B2j[e_map[e], j] = B2R[e, f]

    # Betti
    from numpy.linalg import matrix_rank
    r1 = matrix_rank(B1j) if min(nVj, nEj) > 0 else 0
    r2 = matrix_rank(B2j) if min(nEj, nFj) > 0 else 0
    beta = (nVj - r1, nEj - r1 - r2, nFj - r2)

    chain_res = float(np.max(np.abs(B1j @ B2j))) if nVj > 0 and nEj > 0 and nFj > 0 else 0.0

    return {
        'B1j': B1j, 'B2j': B2j,
        'nVj': nVj, 'nEj': nEj, 'nFj': nFj,
        'beta': beta, 'chain_residual': chain_res,
        'matched_edges_R': np.array(matched, dtype=np.int32),
    }


# Outer join (pushout)

def outer_join(B1_R_dense, B2_R_dense, Py_ssize_t nV_R, Py_ssize_t nE_R, Py_ssize_t nF_R,
               B1_S_dense, B2_S_dense, Py_ssize_t nV_S, Py_ssize_t nE_S, Py_ssize_t nF_S,
               np.ndarray[i32, ndim=1] shared_vertices):
    """Outer join: V_R union V_S, E_R union E_S, F_R union F_S."""
    B1R = np.asarray(B1_R_dense, dtype=np.float64)
    B1S = np.asarray(B1_S_dense, dtype=np.float64)
    B2R = np.asarray(B2_R_dense, dtype=np.float64)
    B2S = np.asarray(B2_S_dense, dtype=np.float64)
    cdef i32[::1] sv = shared_vertices

    # Vertex map: R vertices keep indices, unshared S vertices get new ones
    s_to_j = np.full(nV_S, -1, dtype=np.int32)
    nVj = nV_R
    for v in range(nV_R):
        if sv[v] >= 0:
            s_to_j[sv[v]] = v
    for v in range(nV_S):
        if s_to_j[v] < 0:
            s_to_j[v] = nVj
            nVj += 1

    nEj = nE_R + nE_S
    nFj = nF_R + nF_S

    B1j = np.zeros((nVj, nEj), dtype=np.float64)
    # R part
    B1j[:nV_R, :nE_R] = B1R
    # S part
    for e in range(nE_S):
        for v in range(nV_S):
            if abs(B1S[v, e]) > 0.5:
                B1j[s_to_j[v], nE_R + e] = B1S[v, e]

    B2j = np.zeros((nEj, nFj), dtype=np.float64)
    B2j[:nE_R, :nF_R] = B2R
    B2j[nE_R:nE_R+nE_S, nF_R:nF_R+nF_S] = B2S

    from numpy.linalg import matrix_rank
    r1 = matrix_rank(B1j) if min(nVj, nEj) > 0 else 0
    r2 = matrix_rank(B2j) if min(nEj, nFj) > 0 else 0
    beta = (nVj - r1, nEj - r1 - r2, nFj - r2)

    chain_res = float(np.max(np.abs(B1j @ B2j))) if nVj > 0 and nEj > 0 and nFj > 0 else 0.0

    return {
        'B1j': B1j, 'B2j': B2j,
        'nVj': nVj, 'nEj': nEj, 'nFj': nFj,
        'beta': beta, 'chain_residual': chain_res,
    }


# Left join

def left_join(B1_R_dense, B2_R_dense, Py_ssize_t nV_R, Py_ssize_t nE_R, Py_ssize_t nF_R,
              B1_S_dense, B2_S_dense, Py_ssize_t nV_S, Py_ssize_t nE_S, Py_ssize_t nF_S,
              np.ndarray[i32, ndim=1] shared_vertices):
    """Left join: keep all of R, add S-edges between shared vertices."""
    B1R = np.asarray(B1_R_dense, dtype=np.float64)
    B1S = np.asarray(B1_S_dense, dtype=np.float64)
    B2R = np.asarray(B2_R_dense, dtype=np.float64)
    cdef i32[::1] sv = shared_vertices

    s_to_r = np.full(nV_S, -1, dtype=np.int32)
    for v in range(nV_R):
        if sv[v] >= 0:
            s_to_r[sv[v]] = v

    # Find R edge pairs for dedup
    r_pairs = set()
    for e in range(nE_R):
        endpoints = []
        for v in range(nV_R):
            if abs(B1R[v, e]) > 0.5:
                endpoints.append(v)
        if len(endpoints) == 2:
            r_pairs.add((min(endpoints[0], endpoints[1]),
                         max(endpoints[0], endpoints[1])))

    # S edges between shared vertices not already in R
    new_edges = []
    for e in range(nE_S):
        endpoints_r = []
        for v in range(nV_S):
            if abs(B1S[v, e]) > 0.5 and s_to_r[v] >= 0:
                endpoints_r.append(s_to_r[v])
        if len(endpoints_r) == 2:
            key = (min(endpoints_r[0], endpoints_r[1]),
                   max(endpoints_r[0], endpoints_r[1]))
            if key not in r_pairs:
                new_edges.append((endpoints_r[0], endpoints_r[1], B1S[:, e]))
                r_pairs.add(key)

    nEj = nE_R + len(new_edges)
    B1j = np.zeros((nV_R, nEj), dtype=np.float64)
    B1j[:, :nE_R] = B1R

    for j, (s, t, _) in enumerate(new_edges):
        B1j[s, nE_R + j] = -1.0
        B1j[t, nE_R + j] = 1.0

    # Only R faces (S faces would need cross-edge B2 construction)
    B2j = np.zeros((nEj, nF_R), dtype=np.float64)
    B2j[:nE_R, :] = B2R

    from numpy.linalg import matrix_rank
    r1 = matrix_rank(B1j) if min(nV_R, nEj) > 0 else 0
    r2 = matrix_rank(B2j) if min(nEj, nF_R) > 0 else 0
    beta = (nV_R - r1, nEj - r1 - r2, nF_R - r2)

    chain_res = float(np.max(np.abs(B1j @ B2j))) if nV_R > 0 and nEj > 0 and nF_R > 0 else 0.0

    return {
        'B1j': B1j, 'B2j': B2j,
        'nVj': nV_R, 'nEj': nEj, 'nFj': nF_R,
        'beta': beta, 'chain_residual': chain_res,
        'n_new_edges': len(new_edges),
    }


# Attribute merge

def attribute_merge(Py_ssize_t nV_R, Py_ssize_t nE_R,
                     np.ndarray[f64, ndim=1] ew_R,
                     np.ndarray[f64, ndim=1] amps_R,
                     np.ndarray[f64, ndim=1] ew_S,
                     np.ndarray[f64, ndim=1] amps_S,
                     np.ndarray[i32, ndim=1] shared_vertices,
                     f64 alpha=0.5):
    """Blend attributes at shared vertices.

    merged = (1-alpha)*R + alpha*S at shared cells.
    """
    cdef i32[::1] sv = shared_vertices
    merged_amps = amps_R.copy()
    merged_ew = ew_R.copy()
    n_enriched = 0

    for v in range(nV_R):
        if sv[v] >= 0 and sv[v] < len(amps_S):
            merged_amps[v] = (1.0 - alpha) * amps_R[v] + alpha * amps_S[sv[v]]
            n_enriched += 1

    return {
        'merged_ew': merged_ew,
        'merged_amps': merged_amps,
        'n_enriched': n_enriched,
    }
