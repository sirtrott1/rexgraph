# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._fiber: Fiber character and similarity complex.

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
    cdef f64[:, ::1] sm = np.ascontiguousarray(similarity)
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = 0

    # Pass 1: count edges above threshold (no Python list accumulation).
    for i in range(n):
        for j in range(i + 1, n):
            if sm[i, j] > threshold:
                m += 1

    cdef np.ndarray[i32, ndim=1] src = np.empty(m, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] tgt = np.empty(m, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] wt = np.empty(m, dtype=np.float64)
    cdef i32[::1] sv = src, tv = tgt
    cdef f64[::1] wv = wt
    cdef Py_ssize_t k = 0

    # Pass 2: fill preallocated arrays.
    for i in range(n):
        for j in range(i + 1, n):
            if sm[i, j] > threshold:
                sv[k] = <i32>i
                tv[k] = <i32>j
                wv[k] = sm[i, j]
                k += 1

    return (src, tgt, wt, int(m))


def similarity_complex(np.ndarray[f64, ndim=2] similarity,
                        Py_ssize_t n, f64 threshold):
    """Build a relational complex from thresholded similarity.

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
    """Project chi vectors from the channel simplex to 3D Cartesian.

    Barycentric throughout: equal shares land at the simplex centre and a cell carrying
    one channel lands on that channel's corner, so a coordinate reads back as "this cell
    is mostly frustration" without a legend.

    nhats is 4 (the channels are exactly L1_down, L_O, L_SG and L_C) or fewer, since a
    channel carrying nothing is dropped as inactive: two disjoint relations have no
    co-participation and no frustration and read nhats=2. So

        4   the regular tetrahedron
        3   the equilateral triangle, flat in z
        <=3 the identity, which IS the barycentric embedding of the lower simplex: at
            nhats=2 the image is the segment c_0 + c_1 = 1

    Above 4 there is no simplex to embed into three dimensions without dropping a
    channel, and dropping one silently would return the same point for cells that differ
    only in what was dropped. There is no fifth channel, so this is an error rather than
    a branch.
    """
    if nhats > 4:
        raise ValueError(
            "signal_sphere_proj is barycentric over the four channels; nhats=%d has no "
            "faithful 3D embedding and dropping a channel would collapse distinct cells "
            "onto one point" % nhats)
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
        # nhats <= 2: the identity is the barycentric embedding of the lower simplex
        for e in range(nE):
            for k in range(nhats):
                pv[e, k] = cv[e, k]

    return pts


# φ-similarity (vertex character distance)

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
                          Py_ssize_t n_entities,
                          str face_fill='clique'):
    """Build a relational complex from pairwise fiber bundle similarity.

    1. Threshold S_fb to produce edges (1-skeleton).
    2. Fill faces (see ``face_fill``).
    3. Build B1 and B2 from the face set.
    4. Compute Betti numbers (EIGEN-FREE: exact integer rank / Euler, never SVD).

    Edges connect entities with S_fb above threshold.

    Parameters
    ----------
    sfb_matrix : f64[n_entities, n_entities]
        Fiber bundle similarity matrix. S_fb[i,j] in [0, 1].
    threshold : float
        Minimum S_fb value for an edge. Typical range 0.7-0.95.
    n_entities : int
        Number of entities (vertices in the linkage complex).
    face_fill : {'clique', 'cycle'}, default 'clique'
        'clique' - faces are ALL 3-cliques (triangles): "three-way coherence",
            triples whose three pairwise similarities all exceed the threshold.
            Overlapping triangles can share edges, so rank(B2) may be < nF and beta
            uses the exact integer rank. This is the historical behavior; the
            ``triangles`` output is populated (i32[nF, 3]).
        'cycle' - faces are the fundamental cycle basis (arbitrary-arity n-gon faces,
            matching ``similarity_complex`` and the soccer-ball 5-6-gon principle).
            Every fundamental cycle is independent, so rank(B2)=nF and beta follows
            from Euler with NO rank computation. Faces are not triangles, so
            ``triangles`` is empty (0, 3) and ``face_lengths`` (i32[nF]) is added.

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

    if face_fill == 'cycle':
        # Arbitrary-arity faces = fundamental cycle basis (n-gon faces, as
        # similarity_complex). Every fundamental cycle is independent, so rank(B2)=nF
        # and beta follows from Euler with NO rank computation (beta_1 = 0, beta_2 = 0).
        from rexgraph.core._cycles import find_fundamental_cycles
        c_edges, c_signs, c_lengths, nF_cyc, n_comp = find_fundamental_cycles(
            nV, nE, src, tgt)
        if nF_cyc == 0:
            return {
                'src': src, 'tgt': tgt, 'weights': weights,
                'n_edges': int(nE), 'nV': int(nV), 'nF': 0,
                'B1': B1, 'B2': None,
                'beta': (int(n_comp), 0, 0),
                'triangles': np.zeros((0, 3), dtype=np.int32),
                'face_lengths': np.zeros(0, dtype=np.int32),
            }
        B2_cyc = to_dense_f64(build_B2_from_cycles(nE, c_edges, c_signs, c_lengths))
        return {
            'src': src, 'tgt': tgt, 'weights': weights,
            'n_edges': int(nE), 'nV': int(nV), 'nF': int(nF_cyc),
            'B1': B1, 'B2': B2_cyc,
            'beta': (int(n_comp), 0, 0),
            'triangles': np.zeros((0, 3), dtype=np.int32),
            'face_lengths': np.asarray(c_lengths, dtype=np.int32),
        }
    elif face_fill != 'clique':
        raise ValueError("face_fill must be 'clique' or 'cycle'")

    # Build adjacency for triangle enumeration
    adj_ptr, adj_idx, adj_edge = build_symmetric_adjacency(nV, nE, src, tgt)
    cdef i32[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge

    # Enumerate all triangles via sorted adjacency intersection.
    # For each u, for each neighbor v > u, intersect N(u) and N(v)
    # for w > v. Each triangle is found exactly once. Pass 1 counts,
    # pass 2 fills the final arrays directly - the previous version
    # built Python lists of int-tuples and then copied them over.
    cdef Py_ssize_t u, v, w
    cdef Py_ssize_t j_v, lo_v, hi_v, lo_w, hi_w
    cdef Py_ssize_t p_u, p_w
    cdef i32 e_uv, e_uw, e_vw
    cdef Py_ssize_t nF = 0

    for u in range(nV):
        lo_v = ap[u]
        hi_v = ap[u + 1]
        for j_v in range(lo_v, hi_v):
            v = ai[j_v]
            if v <= u:
                continue
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
                    nF += 1
                    p_u += 1
                    p_w += 1

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

    # Build B2 from triangles: each triangle is a 3-cycle. Pass 2 fills
    # these arrays directly during a second enumeration.
    cdef np.ndarray[i32, ndim=1] cycle_edges = np.empty(nF * 3, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] cycle_signs = np.empty(nF * 3, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] cycle_lengths = np.full(nF, 3, dtype=np.int32)
    cdef np.ndarray[i32, ndim=2] triangles = np.empty((nF, 3), dtype=np.int32)
    cdef i32[::1] ce = cycle_edges
    cdef f64[::1] cs = cycle_signs
    cdef i32[:, ::1] tv = triangles
    cdef Py_ssize_t fi = 0

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
                    # Standard orientation: d(u,v,w) = (u,v) - (u,w) + (v,w)
                    ce[fi * 3] = e_uv;     cs[fi * 3] = 1.0
                    ce[fi * 3 + 1] = e_uw; cs[fi * 3 + 1] = -1.0
                    ce[fi * 3 + 2] = e_vw; cs[fi * 3 + 2] = 1.0
                    tv[fi, 0] = <i32>u; tv[fi, 1] = <i32>v; tv[fi, 2] = <i32>w
                    fi += 1
                    p_u += 1
                    p_w += 1

    B2_dual = build_B2_from_cycles(nE, cycle_edges, cycle_signs, cycle_lengths)

    # Convert DualCSR to dense for return
    B2 = to_dense_f64(B2_dual)

    # Betti numbers via Euler relation and rank computation.
    # beta_0 from connected components via union-find.
    from rexgraph.core._cycles import cycle_space_dimension
    beta_1_no_faces = cycle_space_dimension(nV, nE, src, tgt)
    beta_0 = beta_1_no_faces - nE + nV

    # beta_1 = beta_1_no_faces - rank(B2), beta_2 = nF - rank(B2). Exact, eigen-free
    # rank of the INTEGER B2 via rational column reduction (no SVD, no spectrum-Betti);
    # overlapping triangles share edges, so rank(B2) may be < nF.
    cdef int rank_B2 = 0
    if B2 is not None:
        import scipy.sparse as _sp_local
        from rexgraph.graded_boundary import _sparse_rank
        rank_B2 = _sparse_rank(_sp_local.csc_matrix(np.asarray(B2, dtype=np.float64)))

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
