# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._fiber - Fiber character and similarity complex.

Fiber character uses full spectral structure (not just diagonal).
Similarity complex thresholds pairwise chi/phi cosine into a new graph.
Sphere projection maps simplex coordinates to 3D for visualization.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, acos, cos, sin

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    get_EPSILON_NORM,
)

np.import_array()


# Cosine similarity

@cython.boundscheck(False)
@cython.wraparound(False)
def chi_cosine(np.ndarray[f64, ndim=2] chi, Py_ssize_t nE, Py_ssize_t nhats):
    """Pairwise cosine similarity of structural character vectors."""
    cdef np.ndarray[f64, ndim=2] sim = np.zeros((nE, nE), dtype=np.float64)
    cdef f64[:, ::1] sv = sim, cv = chi
    cdef Py_ssize_t i, j, k
    cdef f64 dot, ni, nj

    for i in range(nE):
        for j in range(i, nE):
            dot = 0.0
            ni = 0.0
            nj = 0.0
            for k in range(nhats):
                dot += cv[i, k] * cv[j, k]
                ni += cv[i, k] * cv[i, k]
                nj += cv[j, k] * cv[j, k]
            ni = sqrt(ni)
            nj = sqrt(nj)
            if ni > 1e-15 and nj > 1e-15:
                sv[i, j] = dot / (ni * nj)
                sv[j, i] = sv[i, j]
            elif i == j:
                sv[i, j] = 1.0

    return sim


@cython.boundscheck(False)
@cython.wraparound(False)
def phi_cosine(np.ndarray[f64, ndim=2] phi, Py_ssize_t nV, Py_ssize_t nhats):
    """Pairwise cosine similarity of vertex character vectors."""
    cdef np.ndarray[f64, ndim=2] sim = np.zeros((nV, nV), dtype=np.float64)
    cdef f64[:, ::1] sv = sim, pv = phi
    cdef Py_ssize_t i, j, k
    cdef f64 dot, ni, nj

    for i in range(nV):
        for j in range(i, nV):
            dot = 0.0
            ni = 0.0
            nj = 0.0
            for k in range(nhats):
                dot += pv[i, k] * pv[j, k]
                ni += pv[i, k] * pv[i, k]
                nj += pv[j, k] * pv[j, k]
            ni = sqrt(ni)
            nj = sqrt(nj)
            if ni > 1e-15 and nj > 1e-15:
                sv[i, j] = dot / (ni * nj)
                sv[j, i] = sv[i, j]
            elif i == j:
                sv[i, j] = 1.0

    return sim


# Threshold graph

def threshold_graph(np.ndarray[f64, ndim=2] similarity,
                     Py_ssize_t n, f64 threshold):
    """Threshold a similarity matrix into edge arrays.

    Returns (src, tgt, weights, n_edges).
    """
    src_list = []
    tgt_list = []
    wt_list = []

    cdef Py_ssize_t i, j
    cdef f64 val

    for i in range(n):
        for j in range(i + 1, n):
            val = similarity[i, j]
            if val > threshold:
                src_list.append(i)
                tgt_list.append(j)
                wt_list.append(val)

    n_edges = len(src_list)
    return (
        np.array(src_list, dtype=np.int32),
        np.array(tgt_list, dtype=np.int32),
        np.array(wt_list, dtype=np.float64),
        n_edges,
    )


def similarity_complex(np.ndarray[f64, ndim=2] similarity,
                        Py_ssize_t n, f64 threshold):
    """Build a chain complex from thresholded similarity.

    Uses fundamental cycle basis for automatic face generation.
    """
    src, tgt, weights, n_edges = threshold_graph(similarity, n, threshold)

    if n_edges == 0:
        return {
            'src': src, 'tgt': tgt, 'weights': weights,
            'n_edges': 0, 'nV': n, 'nF': 0,
            'beta': (n, 0, 0),
        }

    from rexgraph.core._cycles import find_fundamental_cycles
    from rexgraph.core._boundary import build_B1, build_B2_from_cycles

    nV = n
    B1 = build_B1(nV, n_edges, src, tgt)
    cycle_edges, cycle_signs, cycle_lengths, nF, n_comp = \
        find_fundamental_cycles(nV, n_edges, src, tgt)

    B2 = None
    if nF > 0:
        B2 = build_B2_from_cycles(n_edges, cycle_edges, cycle_signs, cycle_lengths)

    from numpy.linalg import matrix_rank
    # Betti from Euler relation
    beta_0 = n_comp
    beta_1 = n_edges - nV + n_comp - nF  # from cycle basis
    beta_2 = 0

    return {
        'src': src, 'tgt': tgt, 'weights': weights,
        'n_edges': n_edges, 'nV': nV, 'nF': nF,
        'B1': B1, 'B2': B2,
        'beta': (beta_0, beta_1, beta_2),
    }


# Barycentric to 3D projection

@cython.boundscheck(False)
@cython.wraparound(False)
def signal_sphere_proj(np.ndarray[f64, ndim=2] chi,
                        Py_ssize_t nE, Py_ssize_t nhats):
    """Project chi vectors from simplex to 3D Cartesian.

    For nhats=3: standard barycentric coordinates.
    (x, y, z) where x = chi_0, y = chi_1, z = chi_2 mapped to
    equilateral triangle vertices in 3D.
    """
    cdef np.ndarray[f64, ndim=2] pts = np.zeros((nE, 3), dtype=np.float64)
    cdef f64[:, ::1] pv = pts
    cdef f64[:, ::1] cv = chi
    cdef Py_ssize_t e
    cdef f64 sq3_2 = sqrt(3.0) / 2.0

    if nhats == 3:
        # Barycentric to Cartesian on equilateral triangle
        # Vertices at (0, 0), (1, 0), (0.5, sqrt(3)/2)
        for e in range(nE):
            pv[e, 0] = cv[e, 1] + 0.5 * cv[e, 2]  # x
            pv[e, 1] = sq3_2 * cv[e, 2]             # y
            pv[e, 2] = 0.0                            # z (flat)
    elif nhats == 4:
        # Tetrahedron vertices
        for e in range(nE):
            pv[e, 0] = cv[e, 0] * 0.0 + cv[e, 1] * 1.0 + cv[e, 2] * 0.5 + cv[e, 3] * 0.5
            pv[e, 1] = cv[e, 0] * 0.0 + cv[e, 1] * 0.0 + cv[e, 2] * sq3_2 + cv[e, 3] * (sqrt(3.0) / 6.0)
            pv[e, 2] = cv[e, 3] * sqrt(2.0 / 3.0)
    else:
        # Generic: use first 3 components
        for e in range(nE):
            for k in range(min(nhats, 3)):
                pv[e, k] = cv[e, k]

    return pts


# ═══ φ-similarity (vertex character distance) ═══

@cython.boundscheck(False)
@cython.wraparound(False)
def phi_similarity_score(np.ndarray[f64, ndim=1] phi_a,
                          np.ndarray[f64, ndim=1] phi_b,
                          int nhats):
    """φ-similarity: 1 - ½||φ_a - φ_b||₁.

    Same metric as cross-dimensional coherence but between two vertices.
    Returns scalar in [0, 1]. 1 = identical character, 0 = maximally different.
    """
    cdef f64[::1] a = phi_a, b = phi_b
    cdef f64 l1 = 0
    cdef int k
    for k in range(nhats):
        l1 += fabs(a[k] - b[k])
    return 1.0 - 0.5 * l1


@cython.boundscheck(False)
@cython.wraparound(False)
def phi_similarity_matrix(np.ndarray[f64, ndim=2] phi, int nV, int nhats):
    """Full φ-similarity matrix: S_φ[i,j] = 1 - ½||φ_i - φ_j||₁."""
    cdef np.ndarray[f64, ndim=2] sim = np.zeros((nV, nV), dtype=np.float64)
    cdef f64[:, ::1] sv = sim, pv = phi
    cdef int i, j, k
    cdef f64 l1

    for i in range(nV):
        sv[i, i] = 1.0
        for j in range(i + 1, nV):
            l1 = 0
            for k in range(nhats):
                l1 += fabs(pv[i, k] - pv[j, k])
            sv[i, j] = 1.0 - 0.5 * l1
            sv[j, i] = sv[i, j]

    return sim


@cython.boundscheck(False)
@cython.wraparound(False)
def sfb_similarity_matrix(np.ndarray[f64, ndim=2] fchi,
                           np.ndarray[f64, ndim=2] phi,
                           int n, int nhats):
    """S_fb fiber bundle similarity matrix.

    S_fb[i,j] = max(cos(fchi_i, fchi_j), 0) * phi_similarity(phi_i, phi_j).

    Combines structural character cosine (fiber alignment) with
    vertex character agreement (cross-dimensional coherence between vertices).
    """
    cdef np.ndarray[f64, ndim=2] sfb = np.zeros((n, n), dtype=np.float64)
    cdef f64[:, ::1] sv = sfb, fv = fchi, pv = phi
    cdef int i, j, k
    cdef f64 dot, na, nb, cos_val, l1, phi_sim

    for i in range(n):
        for j in range(i + 1, n):
            # Cosine of fiber character
            dot = 0; na = 0; nb = 0
            for k in range(nhats):
                dot += fv[i, k] * fv[j, k]
                na += fv[i, k] * fv[i, k]
                nb += fv[j, k] * fv[j, k]
            na = sqrt(na); nb = sqrt(nb)
            cos_val = dot / (na * nb) if na > 1e-15 and nb > 1e-15 else 0.0
            if cos_val < 0: cos_val = 0

            # φ-similarity
            l1 = 0
            for k in range(nhats):
                l1 += fabs(pv[i, k] - pv[j, k])
            phi_sim = 1.0 - 0.5 * l1

            sv[i, j] = cos_val * phi_sim
            sv[j, i] = sv[i, j]

    return sfb


# Linkage complex from S_fb


@cython.boundscheck(False)
@cython.wraparound(False)
def linkage_complex(np.ndarray[f64, ndim=2] sfb_matrix,
                          f64 threshold,
                          Py_ssize_t n_entities):
    """Build a relational complex from pairwise fiber bundle similarity.

    1. Threshold S_fb to produce edges (1-skeleton).
    2. Enumerate all triangles via adjacency intersection.
    3. Build B1 and B2 from the triangle set.
    4. Compute Betti numbers.

    Edges connect entities with S_fb above threshold. Faces represent
    three-way coherence: triples where all pairwise similarities
    exceed the threshold simultaneously.

    Parameters
    ----------
    sfb_matrix : f64[n_entities, n_entities]
        Fiber bundle similarity matrix. S_fb[i,j] in [0, 1].
    threshold : float
        Minimum S_fb value for an edge. Typical range 0.7-0.95.
    n_entities : int
        Number of entities (vertices in the linkage complex).

    Returns
    -------
    dict
        src, tgt : i32 arrays, edge endpoints
        weights : f64 array, S_fb values for each edge
        n_edges : int
        nV : int (= n_entities)
        nF : int
        B1 : f64[nV, nE] or None
        B2 : f64[nE, nF] or None
        beta : (beta_0, beta_1, beta_2)
        triangles : i32[nF, 3], vertex triples per face
    """
    src, tgt, weights, n_edges = threshold_graph(sfb_matrix, n_entities, threshold)

    if n_edges == 0:
        return {
            'src': src, 'tgt': tgt, 'weights': weights,
            'n_edges': 0, 'nV': int(n_entities), 'nF': 0,
            'B1': None, 'B2': None,
            'beta': (int(n_entities), 0, 0),
            'triangles': np.zeros((0, 3), dtype=np.int32),
        }

    from rexgraph.core._boundary import build_B1, build_B2_from_cycles
    from rexgraph.core._cycles import build_symmetric_adjacency

    cdef Py_ssize_t nV = n_entities
    cdef Py_ssize_t nE = n_edges

    B1_dual = build_B1(nV, nE, src, tgt)

    from rexgraph.core._sparse import to_dense_f64
    B1 = to_dense_f64(B1_dual)

    # Build adjacency for triangle enumeration
    adj_ptr, adj_idx, adj_edge = build_symmetric_adjacency(nV, nE, src, tgt)
    cdef i32[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge

    # Enumerate all triangles via sorted adjacency intersection.
    # For each u, for each neighbor v > u, intersect N(u) and N(v)
    # for w > v. Each triangle is found exactly once.
    tri_list = []        # (e_uv, e_uw, e_vw) per triangle
    tri_verts = []       # (u, v, w) per triangle

    cdef Py_ssize_t u, v, w
    cdef Py_ssize_t j_v, lo_v, hi_v, lo_w, hi_w
    cdef Py_ssize_t p_u, p_w
    cdef i32 e_uv, e_uw, e_vw

    for u in range(nV):
        lo_v = ap[u]
        hi_v = ap[u + 1]
        for j_v in range(lo_v, hi_v):
            v = ai[j_v]
            if v <= u:
                continue
            e_uv = ae[j_v]

            lo_w = ap[v]
            hi_w = ap[v + 1]
            p_u = lo_v
            p_w = lo_w

            while p_u < hi_v and ai[p_u] <= v:
                p_u += 1
            while p_w < hi_w and ai[p_w] <= v:
                p_w += 1

            while p_u < hi_v and p_w < hi_w:
                if ai[p_u] < ai[p_w]:
                    p_u += 1
                elif ai[p_u] > ai[p_w]:
                    p_w += 1
                else:
                    w = ai[p_u]
                    e_uw = ae[p_u]
                    e_vw = ae[p_w]
                    tri_list.append((int(e_uv), int(e_uw), int(e_vw)))
                    tri_verts.append((int(u), int(v), int(w)))
                    p_u += 1
                    p_w += 1

    cdef Py_ssize_t nF = len(tri_list)

    if nF == 0:
        # 1-skeleton only, no faces
        from rexgraph.core._cycles import cycle_space_dimension
        beta_1_nf = cycle_space_dimension(nV, nE, src, tgt)
        beta_0 = beta_1_nf - nE + nV
        return {
            'src': src, 'tgt': tgt, 'weights': weights,
            'n_edges': int(nE), 'nV': int(nV), 'nF': 0,
            'B1': B1, 'B2': None,
            'beta': (int(beta_0), int(beta_1_nf), 0),
            'triangles': np.zeros((0, 3), dtype=np.int32),
        }

    # Build B2 from triangles: each triangle is a 3-cycle
    cdef np.ndarray[i32, ndim=1] cycle_edges = np.empty(nF * 3, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] cycle_signs = np.empty(nF * 3, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] cycle_lengths = np.full(nF, 3, dtype=np.int32)
    cdef np.ndarray[i32, ndim=2] triangles = np.empty((nF, 3), dtype=np.int32)

    cdef Py_ssize_t fi
    for fi in range(nF):
        cycle_edges[fi * 3] = tri_list[fi][0]
        cycle_edges[fi * 3 + 1] = tri_list[fi][1]
        cycle_edges[fi * 3 + 2] = tri_list[fi][2]
        # Standard orientation: d(u,v,w) = (u,v) - (u,w) + (v,w)
        cycle_signs[fi * 3] = 1.0
        cycle_signs[fi * 3 + 1] = -1.0
        cycle_signs[fi * 3 + 2] = 1.0
        triangles[fi, 0] = tri_verts[fi][0]
        triangles[fi, 1] = tri_verts[fi][1]
        triangles[fi, 2] = tri_verts[fi][2]

    B2_dual = build_B2_from_cycles(nE, cycle_edges, cycle_signs, cycle_lengths)

    # Convert DualCSR to dense for return
    B2 = to_dense_f64(B2_dual)

    # Betti numbers via Euler relation and rank computation.
    # beta_0 from connected components via union-find.
    from rexgraph.core._cycles import cycle_space_dimension
    beta_1_no_faces = cycle_space_dimension(nV, nE, src, tgt)
    beta_0 = beta_1_no_faces - nE + nV

    # beta_1 = beta_1_no_faces - rank(B2)
    # beta_2 = nF - rank(B2)
    cdef int rank_B2 = 0
    if B2 is not None:
        sv = np.linalg.svd(B2, compute_uv=False)
        rank_B2 = int(np.sum(sv > 1e-10))

    cdef int beta_1 = beta_1_no_faces - rank_B2
    cdef int beta_2 = nF - rank_B2
    if beta_1 < 0:
        beta_1 = 0

    return {
        'src': src, 'tgt': tgt, 'weights': weights,
        'n_edges': int(nE), 'nV': int(nV), 'nF': int(nF),
        'B1': B1, 'B2': B2,
        'beta': (int(beta_0), int(beta_1), int(beta_2)),
        'triangles': triangles,
    }
