# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._quotient - quotient complexes and relative homology on rex.

Given a 2-rex R = (E, F, d_1, d_2) with d_1 d_2 = 0 and derived vertex
set V = union_e supp(d_1(e)), and a subcomplex I specified by masks on
edges and faces with induced vertex set V_I, this module builds the
quotient complex R/I and computes relative homological invariants.

Mathematics
-----------
A subcomplex I of a chain complex R is a sub-chain-complex: collections
of cells at each dimension that are closed under the boundary maps. The
quotient R/I identifies all cells in I with zero, which yields boundary
operators B1_quot and B2_quot that still satisfy B1_quot B2_quot = 0.
The homology of R/I is the relative homology H_k(R, I), related to
H_k(R) and H_k(I) by the long exact sequence of the pair.

Relational complex features
---------------------------
- Edge-primary subcomplex selection: edges are the primitive cells,
  vertices are derived via boundary closure. Subcomplexes are specified
  by edge masks; vertex masks are computed, not independent inputs.
- Quotient boundary operators built from B1 (signed incidence) and B2
  (face-edge incidence) in the edge-primary basis.
- Relational Laplacian on the quotient: RL_1^quot = L_1^quot + alpha_G * L_O^quot,
  preserving the topological-geometric energy decomposition.
- Energy-based subcomplex selection by E_kin/E_pot ratio regime.
- Congruence testing for edge and face chains modulo a chosen subcomplex.
- Signal restriction (R to R/I) and lifting (R/I to R) for real signals,
  complex amplitudes, and (E, F) field states.
- Integration with hyperslices, edge types, and temporal bundles.

All functions are stateless: arrays in, arrays out.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, sqrt, log
from libc.string cimport memcpy

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
    get_EPSILON_NORM,
)

np.import_array()


# Section 1: Subcomplex selection


def validate_subcomplex(np.ndarray[np.uint8_t, ndim=1] v_mask,
                        np.ndarray[np.uint8_t, ndim=1] e_mask,
                        np.ndarray[np.uint8_t, ndim=1] f_mask,
                        np.ndarray[i32, ndim=1] boundary_ptr,
                        np.ndarray[i32, ndim=1] boundary_idx,
                        np.ndarray[i32, ndim=1] B2_col_ptr,
                        np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Check that (v_mask, e_mask, f_mask) defines a subcomplex of a 2-rex.

    Closure conditions in the derived vertex set:
      1. For every edge selected by e_mask, all boundary vertices given by
         the general boundary representation must be selected by v_mask.
      2. For every face selected by f_mask, all boundary edges must be
         selected by e_mask.

    Parameters
    ----------
    v_mask, e_mask, f_mask : uint8[n], uint8[m], uint8[f]
        Masks with 1 for cells in the subcomplex and 0 otherwise.
    boundary_ptr : i32[nE+1]
        CSR row pointer for the general edge boundary representation.
        boundary_idx[boundary_ptr[e]:boundary_ptr[e+1]] are the boundary
        vertices of edge e.
    boundary_idx : i32[nnz]
        Boundary vertex indices for all edges.
    B2_col_ptr, B2_row_idx : i32[f+1], i32[nnz]
        CSC representation of B2, with faces as columns and edges as rows.

    Returns
    -------
    valid : bool
        True when the closure conditions hold.
    violations : list
        Tuples describing closure violations.
    """
    cdef Py_ssize_t nE = e_mask.shape[0], nF = f_mask.shape[0]
    cdef Py_ssize_t e, f, j
    cdef np.uint8_t[::1] vm = v_mask, em = e_mask, fm = f_mask
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx

    violations = []

    # Edge closure: boundary vertices of selected edges must be in v_mask
    for e in range(nE):
        if em[e]:
            for j in range(bp[e], bp[e + 1]):
                if not vm[bi[j]]:
                    violations.append(("edge_vertex", int(e), int(bi[j])))

    # Face closure: boundary edges of selected faces must be in e_mask
    for f in range(nF):
        if fm[f]:
            for j in range(cp[f], cp[f + 1]):
                if not em[ri[j]]:
                    violations.append(("face_edge", int(f), int(ri[j])))

    return (len(violations) == 0), violations


def closure_of_edges(np.ndarray[np.uint8_t, ndim=1] e_mask,
                     Py_ssize_t nV,
                     np.ndarray[i32, ndim=1] boundary_ptr,
                     np.ndarray[i32, ndim=1] boundary_idx):
    """
    Compute the closure of an edge set by adding boundary vertices.

    Uses the general boundary representation so standard, self-loop,
    branching, and witness edges are handled in a uniform way.

    Returns
    -------
    v_mask : uint8[nV]
        Mask for boundary vertices of selected edges.
    e_mask_out : uint8[nE]
        Copy of the input edge mask.
    f_mask : uint8[0]
        Empty face mask.
    """
    cdef Py_ssize_t nE = e_mask.shape[0], e, j
    cdef np.uint8_t[::1] em = e_mask
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx

    cdef np.ndarray[np.uint8_t, ndim=1] v_mask = np.zeros(nV, dtype=np.uint8)
    cdef np.uint8_t[::1] vm = v_mask

    for e in range(nE):
        if em[e]:
            for j in range(bp[e], bp[e + 1]):
                vm[bi[j]] = 1

    return v_mask, e_mask.copy(), np.zeros(0, dtype=np.uint8)


def closure_of_faces(np.ndarray[np.uint8_t, ndim=1] f_mask,
                     Py_ssize_t nV, Py_ssize_t nE,
                     np.ndarray[i32, ndim=1] boundary_ptr,
                     np.ndarray[i32, ndim=1] boundary_idx,
                     np.ndarray[i32, ndim=1] B2_col_ptr,
                     np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Compute the closure of a face set by adding boundary edges and vertices.

    Faces add their boundary edges through B2, and edges add their
    boundary vertices through the general boundary representation.

    Returns
    -------
    v_mask : uint8[nV]
        Mask for vertices in the closure.
    e_mask : uint8[nE]
        Mask for edges in the closure.
    f_mask_out : uint8[nF]
        Copy of the input face mask.
    """
    cdef Py_ssize_t nF = f_mask.shape[0], f, j, e
    cdef np.uint8_t[::1] fm = f_mask
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx

    cdef np.ndarray[np.uint8_t, ndim=1] e_mask = np.zeros(nE, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] v_mask = np.zeros(nV, dtype=np.uint8)
    cdef np.uint8_t[::1] em = e_mask, vm = v_mask

    for f in range(nF):
        if fm[f]:
            for j in range(cp[f], cp[f + 1]):
                em[ri[j]] = 1

    for e in range(nE):
        if em[e]:
            for j in range(bp[e], bp[e + 1]):
                vm[bi[j]] = 1

    return v_mask, e_mask, f_mask.copy()


def closure_of_faces_and_edges(np.ndarray[np.uint8_t, ndim=1] v_mask,
                               np.ndarray[np.uint8_t, ndim=1] e_mask,
                               np.ndarray[np.uint8_t, ndim=1] f_mask,
                               Py_ssize_t nV, Py_ssize_t nE,
                               np.ndarray[i32, ndim=1] boundary_ptr,
                               np.ndarray[i32, ndim=1] boundary_idx,
                               np.ndarray[i32, ndim=1] B2_col_ptr,
                               np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Close masks downward to form a subcomplex in a 2-rex.

    Faces add their boundary edges, and edges add their boundary vertices
    through the general boundary representation.

    Returns
    -------
    v_mask_out, e_mask_out, f_mask_out : uint8 arrays
        Closed masks for vertices, edges, and faces.
    """
    cdef Py_ssize_t nF = f_mask.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] vm = v_mask.copy()
    cdef np.ndarray[np.uint8_t, ndim=1] em = e_mask.copy()
    cdef np.ndarray[np.uint8_t, ndim=1] fm = f_mask.copy()

    cdef Py_ssize_t f, j, e

    if B2_col_ptr.shape[0] > 0 and nF > 0:
        for f in range(nF):
            if fm[f]:
                for j in range(B2_col_ptr[f], B2_col_ptr[f + 1]):
                    em[B2_row_idx[j]] = 1

    if boundary_ptr.shape[0] > 0:
        for e in range(nE):
            if em[e]:
                for j in range(boundary_ptr[e], boundary_ptr[e + 1]):
                    vm[boundary_idx[j]] = 1

    return vm, em, fm


def subcomplex_by_edge_type(np.ndarray[np.uint8_t, ndim=1] edge_types,
                            np.uint8_t select_type,
                            Py_ssize_t nV,
                            np.ndarray[i32, ndim=1] boundary_ptr,
                            np.ndarray[i32, ndim=1] boundary_idx):
    """
    Build a subcomplex from edges with a single type code.

    Type codes follow classifyedgesgeneral in the rex core:
    0 for standard, 1 for self-loop, 2 for branching, and 3 for witness.
    The closure adds all boundary vertices of selected edges.

    Returns
    -------
    v_mask, e_mask, f_mask : uint8 arrays
        Masks for the closed subcomplex.
    """
    cdef Py_ssize_t nE = edge_types.shape[0], e
    cdef np.uint8_t[::1] et = edge_types

    cdef np.ndarray[np.uint8_t, ndim=1] e_mask = np.zeros(nE, dtype=np.uint8)
    cdef np.uint8_t[::1] em = e_mask

    for e in range(nE):
        if et[e] == select_type:
            em[e] = 1

    return closure_of_edges(e_mask, nV, boundary_ptr, boundary_idx)


def subcomplex_by_threshold(np.ndarray[f64, ndim=1] signal,
                            f64 threshold,
                            bint select_below,
                            Py_ssize_t nV,
                            np.ndarray[i32, ndim=1] boundary_ptr,
                            np.ndarray[i32, ndim=1] boundary_idx):
    """
    Build a subcomplex by thresholding an edge signal.

    If select_below is True, edges with absolute value below threshold
    are selected. Otherwise edges with absolute value at least threshold
    are selected. The closure adds all boundary vertices.

    Returns
    -------
    v_mask, e_mask, f_mask : uint8 arrays
        Masks for the closed subcomplex.
    """
    cdef Py_ssize_t nE = signal.shape[0], e
    cdef f64[::1] sig = signal

    cdef np.ndarray[np.uint8_t, ndim=1] e_mask = np.zeros(nE, dtype=np.uint8)
    cdef np.uint8_t[::1] em = e_mask

    for e in range(nE):
        if select_below:
            if fabs(sig[e]) < threshold:
                em[e] = 1
        else:
            if fabs(sig[e]) >= threshold:
                em[e] = 1

    return closure_of_edges(e_mask, nV, boundary_ptr, boundary_idx)


def star_of_vertex(i32 v, Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF,
                   np.ndarray[i32, ndim=1] boundary_ptr,
                   np.ndarray[i32, ndim=1] boundary_idx,
                   np.ndarray[i32, ndim=1] v2e_ptr,
                   np.ndarray[i32, ndim=1] v2e_idx,
                   np.ndarray[i32, ndim=1] e2f_ptr,
                   np.ndarray[i32, ndim=1] e2f_idx,
                   np.ndarray[i32, ndim=1] B2_col_ptr,
                   np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Build the star of a vertex as a subcomplex of a 2-rex.

    The star contains the vertex, edges incident to it, and faces incident
    to those edges, and is then closed downward by adding boundary edges
    of faces and boundary vertices of edges.

    Returns
    -------
    v_mask, e_mask, f_mask : uint8 arrays
        Masks for the star subcomplex.
    """
    cdef np.ndarray[np.uint8_t, ndim=1] v_mask = np.zeros(nV, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] e_mask = np.zeros(nE, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] f_mask = np.zeros(nF, dtype=np.uint8)
    cdef np.uint8_t[::1] vm = v_mask, em = e_mask, fm = f_mask
    cdef i32[::1] vep = v2e_ptr, vei = v2e_idx, efp = e2f_ptr, efi = e2f_idx
    cdef Py_ssize_t j, k, e, f

    vm[v] = 1

    for j in range(vep[v], vep[v + 1]):
        e = vei[j]
        em[e] = 1

    for e in range(nE):
        if em[e]:
            for k in range(efp[e], efp[e + 1]):
                fm[efi[k]] = 1

    return closure_of_faces_and_edges(v_mask, e_mask, f_mask,
                                      nV, nE, boundary_ptr, boundary_idx,
                                      B2_col_ptr, B2_row_idx)


def star_of_edge(i32 edge_idx, Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF,
                 np.ndarray[i32, ndim=1] boundary_ptr,
                 np.ndarray[i32, ndim=1] boundary_idx,
                 np.ndarray[i32, ndim=1] v2e_ptr,
                 np.ndarray[i32, ndim=1] v2e_idx,
                 np.ndarray[i32, ndim=1] e2f_ptr,
                 np.ndarray[i32, ndim=1] e2f_idx,
                 np.ndarray[i32, ndim=1] B2_col_ptr,
                 np.ndarray[i32, ndim=1] B2_row_idx):
    """Build the star of an edge as a subcomplex of a 2-rex.

    The edge star contains the edge itself, all edges sharing a boundary
    vertex with it, all faces incident to those edges, and is then closed
    downward. This is the natural local neighborhood in the edge-primary
    framework: edges are the primitive cells, so their stars define the
    local geometry.

    Returns
    -------
    v_mask, e_mask, f_mask : uint8 arrays
        Masks for the star subcomplex.
    """
    cdef np.ndarray[np.uint8_t, ndim=1] v_mask = np.zeros(nV, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] e_mask = np.zeros(nE, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] f_mask = np.zeros(nF, dtype=np.uint8)
    cdef np.uint8_t[::1] vm = v_mask, em = e_mask, fm = f_mask
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx
    cdef i32[::1] vep = v2e_ptr, vei = v2e_idx
    cdef i32[::1] efp = e2f_ptr, efi = e2f_idx
    cdef Py_ssize_t j, k, v, e2

    # The edge itself
    em[edge_idx] = 1

    # Its boundary vertices (derived, not independent)
    for j in range(bp[edge_idx], bp[edge_idx + 1]):
        vm[bi[j]] = 1

    # All edges sharing a boundary vertex (overlap neighborhood)
    for j in range(bp[edge_idx], bp[edge_idx + 1]):
        v = bi[j]
        for k in range(vep[v], vep[v + 1]):
            em[vei[k]] = 1

    # Faces incident to any edge in the star
    for e2 in range(nE):
        if em[e2]:
            for k in range(efp[e2], efp[e2 + 1]):
                fm[efi[k]] = 1

    return closure_of_faces_and_edges(v_mask, e_mask, f_mask,
                                      nV, nE, boundary_ptr, boundary_idx,
                                      B2_col_ptr, B2_row_idx)


def subcomplex_by_energy_regime(np.ndarray[f64, ndim=1] E_kin_per_edge,
                                np.ndarray[f64, ndim=1] E_pot_per_edge,
                                int regime,
                                double ratio_tol=0.2,
                                Py_ssize_t nV=0,
                                np.ndarray[i32, ndim=1] boundary_ptr=None,
                                np.ndarray[i32, ndim=1] boundary_idx=None,
                                double floor=1e-12):
    """Build a subcomplex from edges in a specific energy regime.

    Each edge contributes to the total E_kin = <f|L_1|f> and E_pot = <f|L_O|f>
    via its per-edge energy components. The log ratio log(E_kin_e / E_pot_e)
    classifies each edge:
        regime 0: kinetic-dominated (log ratio > ratio_tol)
        regime 1: crossover (|log ratio| <= ratio_tol)
        regime 2: potential-dominated (log ratio < -ratio_tol)

    Parameters
    ----------
    E_kin_per_edge : f64[nE] - per-edge topological energy
    E_pot_per_edge : f64[nE] - per-edge geometric energy
    regime : int (0=kinetic, 1=crossover, 2=potential)
    ratio_tol : float
    nV : int - vertex count (for closure)
    boundary_ptr, boundary_idx : boundary representation (for closure)

    Returns
    -------
    v_mask, e_mask, f_mask : uint8 arrays
    """
    cdef Py_ssize_t nE = E_kin_per_edge.shape[0], e
    cdef f64[::1] ek = E_kin_per_edge, ep = E_pot_per_edge
    cdef np.ndarray[np.uint8_t, ndim=1] e_mask = np.zeros(nE, dtype=np.uint8)
    cdef np.uint8_t[::1] em = e_mask
    cdef f64 ek_e, ep_e, lr

    for e in range(nE):
        ek_e = ek[e] if ek[e] > floor else floor
        ep_e = ep[e] if ep[e] > floor else floor
        lr = log(ek_e / ep_e)

        if regime == 0 and lr > ratio_tol:
            em[e] = 1
        elif regime == 1 and fabs(lr) <= ratio_tol:
            em[e] = 1
        elif regime == 2 and lr < -ratio_tol:
            em[e] = 1

    if boundary_ptr is not None and boundary_idx is not None and nV > 0:
        return closure_of_edges(e_mask, nV, boundary_ptr, boundary_idx)

    return np.zeros(max(nV, 0), dtype=np.uint8), e_mask, np.zeros(0, dtype=np.uint8)


# Section 2: Quotient construction


def quotient_maps(np.ndarray[np.uint8_t, ndim=1] v_mask,
                  np.ndarray[np.uint8_t, ndim=1] e_mask,
                  np.ndarray[np.uint8_t, ndim=1] f_mask):
    """
    Build reindexing maps for the quotient complex R/I.

    Vertices selecting the subcomplex (v_mask == 1) collapse to a single
    basepoint, and vertices outside that set keep their order and get
    consecutive indices. Edges and faces in the subcomplex are removed,
    and survivors are reindexed.

    Returns
    -------
    v_reindex : i32[nV]
        Old vertex index to new index, or -1 when collapsed to the basepoint.
    v_star : int
        Index of the basepoint in the quotient, or -1 when v_mask is all zero.
    e_reindex : i32[nE]
        Old edge index to new edge index, or -1 when removed.
    f_reindex : i32[nF]
        Old face index to new face index, or -1 when removed.
    nV_quot, nE_quot, nF_quot : int
        Dimensions of the quotient complex.
    """
    cdef Py_ssize_t nV = v_mask.shape[0], nE = e_mask.shape[0]
    cdef Py_ssize_t nF = f_mask.shape[0] if f_mask.shape[0] > 0 else 0
    cdef np.uint8_t[::1] vm = v_mask, em = e_mask
    cdef Py_ssize_t i

    cdef np.ndarray[i32, ndim=1] v_reindex = np.full(nV, -1, dtype=np.int32)
    cdef i32[::1] vr = v_reindex
    cdef i32 v_count = 0
    cdef bint has_collapsed = False

    for i in range(nV):
        if vm[i]:
            has_collapsed = True
        else:
            vr[i] = v_count
            v_count += 1

    cdef i32 v_star = v_count if has_collapsed else -1
    cdef i32 nV_quot = v_count + (1 if has_collapsed else 0)

    cdef np.ndarray[i32, ndim=1] e_reindex = np.full(nE, -1, dtype=np.int32)
    cdef i32[::1] er = e_reindex
    cdef i32 e_count = 0

    for i in range(nE):
        if not em[i]:
            er[i] = e_count
            e_count += 1

    cdef np.ndarray[i32, ndim=1] f_reindex
    cdef i32 f_count = 0
    if nF > 0:
        f_reindex = np.full(nF, -1, dtype=np.int32)
        for i in range(nF):
            if not f_mask[i]:
                f_reindex[i] = f_count
                f_count += 1
    else:
        f_reindex = np.zeros(0, dtype=np.int32)

    return v_reindex, int(v_star), e_reindex, f_reindex, int(nV_quot), int(e_count), int(f_count)


def quotient_B1(np.ndarray[f64, ndim=2] B1,
                np.ndarray[np.uint8_t, ndim=1] v_mask,
                np.ndarray[np.uint8_t, ndim=1] e_mask,
                np.ndarray[i32, ndim=1] v_reindex,
                i32 v_star,
                np.ndarray[i32, ndim=1] e_reindex,
                Py_ssize_t nV_quot, Py_ssize_t nE_quot):
    """
    Build the quotient boundary operator B1_quot from B1.

    Columns for edges selected by e_mask are removed. For surviving
    edges, their signed incidence columns are remapped so vertices
    in the subcomplex go to the basepoint and other vertices go
    through v_reindex.

    Parameters
    ----------
    B1 : f64[nV, nE]
        Signed incidence matrix on edges.
    v_mask, e_mask : uint8 arrays
        Subcomplex masks for vertices and edges.
    v_reindex : i32[nV]
        Vertex reindexing array.
    v_star : i32
        Basepoint index in the quotient.
    e_reindex : i32[nE]
        Edge reindexing array.
    nV_quot, nE_quot : int
        Dimensions of the quotient.

    Returns
    -------
    B1_quot : f64[nV_quot, nE_quot]
        Dense quotient vertex-edge boundary operator.
    """
    cdef Py_ssize_t nV = B1.shape[0], nE = B1.shape[1], e, v
    cdef np.uint8_t[::1] em = e_mask, vm = v_mask
    cdef i32[::1] vr = v_reindex, er = e_reindex
    cdef f64[:, ::1] Bsrc = B1

    cdef np.ndarray[f64, ndim=2] B1q = np.zeros((nV_quot, nE_quot), dtype=np.float64)
    cdef f64[:, ::1] Bv = B1q
    cdef i32 v_new, e_new
    cdef f64 val

    for e in range(nE):
        if em[e]:
            continue

        e_new = er[e]
        if e_new < 0:
            continue

        for v in range(nV):
            val = Bsrc[v, e]
            if fabs(val) < get_EPSILON_NORM():
                continue

            if vm[v]:
                v_new = v_star
            else:
                v_new = vr[v]

            if v_new >= 0 and v_new < nV_quot:
                Bv[v_new, e_new] = Bv[v_new, e_new] + val

    return B1q


def quotient_B2(np.ndarray[i32, ndim=1] B2_col_ptr,
                np.ndarray[i32, ndim=1] B2_row_idx,
                np.ndarray[f64, ndim=1] B2_vals,
                np.ndarray[np.uint8_t, ndim=1] e_mask,
                np.ndarray[np.uint8_t, ndim=1] f_mask,
                np.ndarray[i32, ndim=1] e_reindex,
                np.ndarray[i32, ndim=1] f_reindex,
                Py_ssize_t nE_quot, Py_ssize_t nF_quot):
    """
    Build the quotient boundary operator B2_quot from B2 in CSC form.

    Columns for faces selected by f_mask are removed. Entries involving
    edges selected by e_mask are dropped, and surviving entries are
    remapped by e_reindex and f_reindex.

    Parameters
    ----------
    B2_col_ptr, B2_row_idx, B2_vals : arrays
        CSC representation of B2 with edges as rows and faces as columns.
    e_mask, f_mask : uint8 arrays
        Subcomplex masks for edges and faces.
    e_reindex, f_reindex : i32 arrays
        Edge and face reindexing arrays.
    nE_quot, nF_quot : int
        Dimensions of the quotient.

    Returns
    -------
    B2_quot : f64[nE_quot, nF_quot]
        Dense quotient edge-face boundary operator.
    """
    cdef Py_ssize_t nF = f_mask.shape[0], f, j
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef f64[::1] vl = B2_vals
    cdef np.uint8_t[::1] em = e_mask, fm = f_mask
    cdef i32[::1] er = e_reindex, fr = f_reindex

    cdef np.ndarray[f64, ndim=2] B2q = np.zeros((nE_quot, nF_quot), dtype=np.float64)
    cdef f64[:, ::1] Bv = B2q
    cdef i32 e_new, f_new, edge_idx

    for f in range(nF):
        if fm[f]:
            continue

        f_new = fr[f]
        if f_new < 0:
            continue

        for j in range(cp[f], cp[f + 1]):
            edge_idx = ri[j]
            if em[edge_idx]:
                continue

            e_new = er[edge_idx]
            if e_new >= 0 and e_new < nE_quot:
                Bv[e_new, f_new] = vl[j]

    return B2q


def quotient_verify_chain(np.ndarray[f64, ndim=2] B1_quot,
                          np.ndarray[f64, ndim=2] B2_quot,
                          f64 tol=1e-10):
    """
    Check the chain complex condition B1_quot B2_quot = 0.

    Returns
    -------
    valid : bool
        True when the maximum absolute entry is below tol.
    max_abs_entry : float
        Maximum absolute value in B1_quot B2_quot.
    """
    if B1_quot.shape[1] == 0 or B2_quot.shape[1] == 0:
        return True, 0.0

    cdef np.ndarray[f64, ndim=2] product = B1_quot @ B2_quot
    cdef f64 max_abs = np.max(np.abs(product))
    return max_abs < tol, float(max_abs)


# Section 3: Relative homology


def relative_betti(np.ndarray[f64, ndim=2] B1_quot,
                   np.ndarray[f64, ndim=2] B2_quot,
                   f64 tol=1e-10):
    """
    Compute relative Betti numbers from quotient boundary operators.

    The ranks of B1_quot and B2_quot are computed from their singular
    values, and beta_k(R, I) are derived from these ranks and the
    quotient dimensions. Here B1_quot and B2_quot play the roles of
    d_1 and d_2 in the quotient chain complex C_2(R/I) to C_1(R/I) to
    C_0(R/I), with C_0(R/I) indexed by derived vertices.

    Returns
    -------
    beta0_rel, beta1_rel, beta2_rel : int
        Relative Betti numbers for k = 0, 1, 2.
    """
    cdef Py_ssize_t nV_quot = B1_quot.shape[0]
    cdef Py_ssize_t nE_quot = B1_quot.shape[1]
    cdef Py_ssize_t nF_quot = B2_quot.shape[1] if B2_quot.shape[0] > 0 else 0

    cdef Py_ssize_t rank_B1 = 0, rank_B2 = 0

    if nV_quot > 0 and nE_quot > 0:
        from rexgraph.core._linalg import svd as _lp_svd
        _, s1, _ = _lp_svd(np.asarray(B1_quot, dtype=np.float64))
        rank_B1 = int(np.sum(s1 > tol))

    if nE_quot > 0 and nF_quot > 0:
        from rexgraph.core._linalg import svd as _lp_svd
        _, s2, _ = _lp_svd(np.asarray(B2_quot, dtype=np.float64))
        rank_B2 = int(np.sum(s2 > tol))

    cdef Py_ssize_t b0 = nV_quot - rank_B1
    cdef Py_ssize_t b1 = nE_quot - rank_B1 - rank_B2
    cdef Py_ssize_t b2 = nF_quot - rank_B2

    return int(b0), int(b1), int(b2)


def relative_cycle_basis(np.ndarray[f64, ndim=2] B1_quot,
                         np.ndarray[f64, ndim=2] B2_quot,
                         f64 tol=1e-10):
    """
    Compute a basis for relative 1-cycles in the edge-primary setting.

    The basis spans the harmonic subspace of the quotient edge Laplacian
    L1_quot = B1_quot.T B1_quot + B2_quot B2_quot.T, and its columns are
    edge signals representing generators of H_1(R, I) with vertices
    derived from d_1.

    Returns
    -------
    basis : f64[nE_quot, beta1_rel]
        Orthonormal harmonic edge signals for the quotient.
    """
    cdef Py_ssize_t nE = B1_quot.shape[1]

    if nE == 0:
        return np.zeros((0, 0), dtype=np.float64)

    L1q = B1_quot.T @ B1_quot
    if B2_quot.shape[0] > 0 and B2_quot.shape[1] > 0:
        L1q = L1q + B2_quot @ B2_quot.T

    from rexgraph.core._linalg import eigh as _lp_eigh
    evals, evecs = _lp_eigh(np.asarray(L1q, dtype=np.float64))

    harmonic_idx = np.where(np.abs(evals) < tol)[0]
    return evecs[:, harmonic_idx].copy()


def connecting_homomorphism(np.ndarray[f64, ndim=2] B1_full,
                            np.ndarray[np.uint8_t, ndim=1] v_mask,
                            np.ndarray[np.uint8_t, ndim=1] e_mask,
                            np.ndarray[f64, ndim=1] relative_cycle,
                            np.ndarray[i32, ndim=1] e_reindex):
    """
    Apply the connecting homomorphism from H_1(R, I) to H_0(I).

    A relative 1-cycle on quotient edges is lifted to the full edge
    space, B1_full is applied, and the result is restricted to vertices
    in the subcomplex.

    Parameters
    ----------
    B1_full : f64[nV, nE]
        Signed incidence matrix for the full complex.
    v_mask, e_mask : uint8 arrays
        Subcomplex masks for vertices and edges.
    relative_cycle : f64[nE_quot]
        Coefficients of a cycle on quotient edges.
    e_reindex : i32[nE]
        Edge reindexing array.

    Returns
    -------
    boundary_in_I : f64[nV_I]
        Boundary values on vertices in the subcomplex.
    """
    cdef Py_ssize_t nV = B1_full.shape[0], nE = B1_full.shape[1]
    cdef Py_ssize_t e, i
    cdef i32[::1] er = e_reindex
    cdef np.uint8_t[::1] vm = v_mask

    cdef np.ndarray[f64, ndim=1] lifted = np.zeros(nE, dtype=np.float64)
    cdef f64[::1] lv = lifted
    for e in range(nE):
        if er[e] >= 0:
            lv[e] = relative_cycle[er[e]]

    cdef np.ndarray[f64, ndim=1] boundary = B1_full @ lifted

    # Count vertices in I and extract
    cdef Py_ssize_t nV_I = 0, v
    for v in range(nV):
        if vm[v]:
            nV_I += 1

    if nV_I == 0:
        return np.zeros(0, dtype=np.float64)

    cdef np.ndarray[f64, ndim=1] result = np.empty(nV_I, dtype=np.float64)
    cdef f64[::1] rv = result, bv = boundary
    cdef Py_ssize_t idx = 0
    for v in range(nV):
        if vm[v]:
            rv[idx] = bv[v]
            idx += 1

    return result


# Section 4: Congruence


def congruent_edges(Py_ssize_t a, Py_ssize_t b,
                    np.ndarray[f64, ndim=2] B1,
                    np.ndarray[np.uint8_t, ndim=1] e_mask,
                    f64 tol=1e-10):
    """
    Test whether edges a and b are congruent modulo the subcomplex.

    The signed incidence difference B1[:, a] - B1[:, b] is compared to
    the column span of B1 restricted to edges in the subcomplex. A small
    residual indicates congruence modulo those edges.

    Parameters
    ----------
    a, b : int
        Indices of edges in the full complex.
    B1 : f64[nV, nE]
        Signed incidence matrix.
    e_mask : uint8[nE]
        Edge mask for the subcomplex.
    tol : float
        Tolerance on the residual norm.

    Returns
    -------
    is_congruent : bool
        True when the residual norm is below tol.
    residual : float
        L2 norm of the residual after projection.
    """
    cdef Py_ssize_t nV = B1.shape[0], nE = B1.shape[1]

    cdef np.ndarray[f64, ndim=1] d = B1[:, a] - B1[:, b]

    cdef np.ndarray idx_I = np.where(np.asarray(e_mask))[0]
    if idx_I.shape[0] == 0:
        r = float(np.linalg.norm(d))
        return r < tol, r

    cdef np.ndarray[f64, ndim=2] basis = B1[:, idx_I]

    from rexgraph.core._linalg import lstsq as _lp_lstsq
    sol, _rank = _lp_lstsq(np.asarray(basis, dtype=np.float64), np.asarray(d, dtype=np.float64))
    residual_vec = d - basis @ sol
    r = float(np.linalg.norm(residual_vec))

    return r < tol, r


def congruent_faces(Py_ssize_t a, Py_ssize_t b,
                    np.ndarray[f64, ndim=2] B2,
                    np.ndarray[np.uint8_t, ndim=1] f_mask,
                    f64 tol=1e-10):
    """
    Test whether faces a and b are congruent modulo the subcomplex.

    The signed boundary difference B2[:, a] - B2[:, b] is compared to
    the column span of B2 restricted to faces in the subcomplex.

    Returns
    -------
    is_congruent : bool
        True when the residual norm is below tol.
    residual : float
        L2 norm of the residual after projection.
    """
    cdef Py_ssize_t nE = B2.shape[0], nF = B2.shape[1]

    cdef np.ndarray[f64, ndim=1] d = B2[:, a] - B2[:, b]

    cdef np.ndarray idx_I = np.where(np.asarray(f_mask))[0]
    if idx_I.shape[0] == 0:
        r = float(np.linalg.norm(d))
        return r < tol, r

    cdef np.ndarray[f64, ndim=2] basis = B2[:, idx_I]

    from rexgraph.core._linalg import lstsq as _lp_lstsq
    sol, _rank = _lp_lstsq(np.asarray(basis, dtype=np.float64), np.asarray(d, dtype=np.float64))
    residual_vec = d - basis @ sol
    r = float(np.linalg.norm(residual_vec))

    return r < tol, r


def congruence_classes_edges(np.ndarray[f64, ndim=2] B1,
                             np.ndarray[np.uint8_t, ndim=1] e_mask,
                             f64 tol=1e-10):
    """
    Partition edges outside the subcomplex into congruence classes.

    Two surviving edges share a class when their signed incidence
    difference lies in the column span of edges in the subcomplex
    up to the given tolerance.

    Returns
    -------
    labels : i32[nE]
        Class label for each edge, or -1 for edges in the subcomplex.
    n_classes : int
        Number of equivalence classes among surviving edges.
    """
    cdef Py_ssize_t nE = B1.shape[1], nV = B1.shape[0]
    cdef Py_ssize_t i, j
    cdef np.uint8_t[::1] em = e_mask

    cdef np.ndarray[i32, ndim=1] labels = np.full(nE, -1, dtype=np.int32)
    cdef i32[::1] lv = labels
    cdef i32 next_label = 0

    survivors = np.where(~np.asarray(e_mask).astype(bool))[0]
    if survivors.shape[0] == 0:
        return labels, 0

    cdef np.ndarray idx_I = np.where(np.asarray(e_mask))[0]
    cdef bint has_basis = idx_I.shape[0] > 0

    cdef np.ndarray[f64, ndim=2] basis
    if has_basis:
        basis = B1[:, idx_I]

    for i in range(len(survivors)):
        if lv[survivors[i]] >= 0:
            continue

        lv[survivors[i]] = next_label

        for j in range(i + 1, len(survivors)):
            if lv[survivors[j]] >= 0:
                continue

            d = B1[:, survivors[i]] - B1[:, survivors[j]]

            if not has_basis:
                if float(np.linalg.norm(d)) < tol:
                    lv[survivors[j]] = next_label
            else:
                from rexgraph.core._linalg import lstsq as _lp_lstsq
                sol = _lp_lstsq(np.asarray(basis, dtype=np.float64), np.asarray(d, dtype=np.float64))[0]
                residual = d - basis @ sol
                if float(np.linalg.norm(residual)) < tol:
                    lv[survivors[j]] = next_label

        next_label += 1

    return labels, int(next_label)


def congruence_classes_faces(np.ndarray[f64, ndim=2] B2,
                             np.ndarray[np.uint8_t, ndim=1] f_mask,
                             f64 tol=1e-10):
    """
    Partition faces outside the subcomplex into congruence classes.

    Two surviving faces share a class when their signed boundary
    difference lies in the column span of faces in the subcomplex
    up to the given tolerance.

    Returns
    -------
    labels : i32[nF]
        Class label for each face, or -1 for faces in the subcomplex.
    n_classes : int
        Number of equivalence classes among surviving faces.
    """
    cdef Py_ssize_t nF = B2.shape[1]
    cdef Py_ssize_t i, j
    cdef np.uint8_t[::1] fm = f_mask

    cdef np.ndarray[i32, ndim=1] labels = np.full(nF, -1, dtype=np.int32)
    cdef i32[::1] lv = labels
    cdef i32 next_label = 0

    survivors = np.where(~np.asarray(f_mask).astype(bool))[0]
    if survivors.shape[0] == 0:
        return labels, 0

    cdef np.ndarray idx_I = np.where(np.asarray(f_mask))[0]
    cdef bint has_basis = idx_I.shape[0] > 0
    cdef np.ndarray[f64, ndim=2] basis

    if has_basis:
        basis = B2[:, idx_I]

    for i in range(len(survivors)):
        if lv[survivors[i]] >= 0:
            continue

        lv[survivors[i]] = next_label

        for j in range(i + 1, len(survivors)):
            if lv[survivors[j]] >= 0:
                continue

            d = B2[:, survivors[i]] - B2[:, survivors[j]]

            if not has_basis:
                if float(np.linalg.norm(d)) < tol:
                    lv[survivors[j]] = next_label
            else:
                from rexgraph.core._linalg import lstsq as _lp_lstsq
                sol = _lp_lstsq(np.asarray(basis, dtype=np.float64), np.asarray(d, dtype=np.float64))[0]
                residual = d - basis @ sol
                if float(np.linalg.norm(residual)) < tol:
                    lv[survivors[j]] = next_label

        next_label += 1

    return labels, int(next_label)


# Section 5: Signal operations


def restrict_signal(np.ndarray[f64, ndim=1] signal,
                    np.ndarray[np.uint8_t, ndim=1] mask):
    """
    Restrict a real signal from the full complex to the quotient.

    Cells in the subcomplex are dropped, and surviving entries are
    compacted into a new array.

    Parameters
    ----------
    signal : f64[n]
        Signal on k-cells of the full complex.
    mask : uint8[n]
        Subcomplex mask with 1 for cells in I.

    Returns
    -------
    signal_quot : f64[n_quot]
        Signal restricted to surviving cells.
    """
    cdef Py_ssize_t n = signal.shape[0], i, count = 0
    cdef np.uint8_t[::1] m = mask
    cdef f64[::1] sv = signal

    for i in range(n):
        if not m[i]:
            count += 1

    cdef np.ndarray[f64, ndim=1] out = np.empty(count, dtype=np.float64)
    cdef f64[::1] ov = out
    cdef Py_ssize_t j = 0

    for i in range(n):
        if not m[i]:
            ov[j] = sv[i]
            j += 1

    return out


def restrict_signal_complex(np.ndarray[np.complex128_t, ndim=1] signal,
                            np.ndarray[np.uint8_t, ndim=1] mask):
    """
    Restrict a complex signal from the full complex to the quotient.

    Cells in the subcomplex are dropped and surviving entries are
    compacted into a new array.
    """
    cdef Py_ssize_t n = signal.shape[0], i, count = 0
    cdef np.uint8_t[::1] m = mask

    for i in range(n):
        if not m[i]:
            count += 1

    cdef np.ndarray[np.complex128_t, ndim=1] out = np.empty(count, dtype=np.complex128)
    cdef Py_ssize_t j = 0

    for i in range(n):
        if not m[i]:
            out[j] = signal[i]
            j += 1

    return out


def lift_signal(np.ndarray[f64, ndim=1] signal_quot,
                np.ndarray[np.uint8_t, ndim=1] mask,
                f64 fill_value=0.0):
    """
    Lift a real signal from the quotient back to the full complex.

    Entries at cells in the subcomplex are filled with fill_value,
    and surviving entries are copied from signal_quot.

    Parameters
    ----------
    signal_quot : f64[n_quot]
        Signal on surviving cells.
    mask : uint8[n]
        Subcomplex mask with 1 for cells in I.
    fill_value : float
        Value to assign at cells in I.

    Returns
    -------
    signal_full : f64[n]
        Signal on the full complex.
    """
    cdef Py_ssize_t n = mask.shape[0], i, j = 0
    cdef np.uint8_t[::1] m = mask
    cdef f64[::1] sv = signal_quot

    cdef np.ndarray[f64, ndim=1] out = np.full(n, fill_value, dtype=np.float64)
    cdef f64[::1] ov = out

    for i in range(n):
        if not m[i]:
            ov[i] = sv[j]
            j += 1

    return out


def lift_signal_complex(np.ndarray[np.complex128_t, ndim=1] signal_quot,
                        np.ndarray[np.uint8_t, ndim=1] mask):
    """
    Lift a complex signal from the quotient back to the full complex.

    Cells in the subcomplex are set to zero and surviving entries are
    copied from signal_quot.
    """
    cdef Py_ssize_t n = mask.shape[0], i, j = 0
    cdef np.uint8_t[::1] m = mask

    cdef np.ndarray[np.complex128_t, ndim=1] out = np.zeros(n, dtype=np.complex128)

    for i in range(n):
        if not m[i]:
            out[i] = signal_quot[j]
            j += 1

    return out


def quotient_energy(np.ndarray[f64, ndim=1] signal_quot,
                    np.ndarray[f64, ndim=2] L_quot):
    """Compute the Rayleigh quotient on the quotient complex.

    The value is signal_quot.T L_quot signal_quot divided by
    signal_quot.T signal_quot, with a zero result when the denominator
    is below the global norm tolerance.

    Returns
    -------
    energy : float
        Rayleigh quotient for the given signal and operator.
    """
    cdef f64 numer = float(signal_quot @ L_quot @ signal_quot)
    cdef f64 denom = float(signal_quot @ signal_quot)
    if denom < get_EPSILON_NORM():
        return 0.0
    return numer / denom


def quotient_RL1(np.ndarray[f64, ndim=2] B1_quot,
                 np.ndarray[f64, ndim=2] B2_quot,
                 np.ndarray[f64, ndim=2] LO_quot,
                 double alpha_G):
    """Build the Relational Laplacian on the quotient edge space.

    RL_1^quot = L_1^quot + alpha_G * L_O^quot

    where L_1^quot = B1_quot^T B1_quot + B2_quot B2_quot^T is the
    Hodge Laplacian of the quotient, preserving the topological-geometric
    decomposition from the full complex.

    Parameters
    ----------
    B1_quot : f64[nV_q, nE_q]
    B2_quot : f64[nE_q, nF_q]
    LO_quot : f64[nE_q, nE_q] - overlap Laplacian on quotient edges
    alpha_G : float - coupling constant

    Returns
    -------
    RL1_quot : f64[nE_q, nE_q]
    L1_quot : f64[nE_q, nE_q]
    """
    cdef Py_ssize_t nE_q = B1_quot.shape[1]

    cdef np.ndarray[f64, ndim=2] L1q = B1_quot.T @ B1_quot
    if B2_quot.shape[0] > 0 and B2_quot.shape[1] > 0:
        L1q = L1q + B2_quot @ B2_quot.T

    cdef np.ndarray[f64, ndim=2] RL1q = L1q + alpha_G * LO_quot

    return RL1q, L1q


def quotient_energy_kin_pot(np.ndarray[f64, ndim=1] signal_quot,
                            np.ndarray[f64, ndim=2] L1_quot,
                            np.ndarray[f64, ndim=2] LO_quot):
    """Compute E_kin and E_pot on the quotient edge space.

    E_kin = <f | L_1^quot | f>  (topological energy)
    E_pot = <f | L_O^quot | f>  (geometric energy)

    This preserves the energy decomposition from the full complex
    in the quotient setting.

    Returns
    -------
    E_kin : float
    E_pot : float
    ratio : float (E_kin / E_pot)
    """
    cdef np.ndarray[f64, ndim=1] L1f = L1_quot @ signal_quot
    cdef np.ndarray[f64, ndim=1] LOf = LO_quot @ signal_quot

    cdef f64[::1] sv = signal_quot, l1v = L1f, lov = LOf
    cdef Py_ssize_t n = signal_quot.shape[0], j
    cdef f64 ek = 0.0, ep = 0.0

    for j in range(n):
        ek += sv[j] * l1v[j]
        ep += sv[j] * lov[j]

    cdef f64 ratio
    if ep > 1e-15:
        ratio = ek / ep
    elif ek > 1e-15:
        ratio = 1e15
    else:
        ratio = 1.0

    return ek, ep, ratio


def restrict_field_state(np.ndarray[f64, ndim=1] f_E,
                         np.ndarray[f64, ndim=1] f_F,
                         np.ndarray[np.uint8_t, ndim=1] e_mask,
                         np.ndarray[np.uint8_t, ndim=1] f_mask):
    """Restrict an (E, F) field state to the quotient.

    In the rex framework, the field state lives on edges and faces.
    Vertex signals are derived from edge signals via B_1, so they
    don't need independent restriction.

    Parameters
    ----------
    f_E : f64[nE] - edge signal
    f_F : f64[nF] - face signal
    e_mask, f_mask : uint8 - subcomplex masks (1 = in subcomplex)

    Returns
    -------
    f_E_quot : f64[nE_quot]
    f_F_quot : f64[nF_quot]
    """
    cdef Py_ssize_t nE = f_E.shape[0], nF = f_F.shape[0]
    cdef Py_ssize_t e, f, je = 0, jf = 0
    cdef np.uint8_t[::1] em = e_mask, fm = f_mask
    cdef f64[::1] ev = f_E, fv = f_F

    # Count survivors
    cdef Py_ssize_t nE_q = 0, nF_q = 0
    for e in range(nE):
        if not em[e]: nE_q += 1
    for f in range(nF):
        if not fm[f]: nF_q += 1

    cdef np.ndarray[f64, ndim=1] fE_q = np.empty(nE_q, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] fF_q = np.empty(nF_q, dtype=np.float64)
    cdef f64[::1] eqv = fE_q, fqv = fF_q

    for e in range(nE):
        if not em[e]:
            eqv[je] = ev[e]
            je += 1

    for f in range(nF):
        if not fm[f]:
            fqv[jf] = fv[f]
            jf += 1

    return fE_q, fF_q


def lift_field_state(np.ndarray[f64, ndim=1] f_E_quot,
                     np.ndarray[f64, ndim=1] f_F_quot,
                     np.ndarray[np.uint8_t, ndim=1] e_mask,
                     np.ndarray[np.uint8_t, ndim=1] f_mask,
                     f64 fill_value=0.0):
    """Lift an (E, F) field state from the quotient to the full complex.

    Entries at subcomplex cells are filled with fill_value.
    Surviving entries are copied from the quotient signals.

    Returns
    -------
    f_E : f64[nE]
    f_F : f64[nF]
    """
    cdef Py_ssize_t nE = e_mask.shape[0], nF = f_mask.shape[0]
    cdef Py_ssize_t e, f, je = 0, jf = 0
    cdef np.uint8_t[::1] em = e_mask, fm = f_mask
    cdef f64[::1] eqv = f_E_quot, fqv = f_F_quot

    cdef np.ndarray[f64, ndim=1] f_E = np.full(nE, fill_value, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_F = np.full(nF, fill_value, dtype=np.float64)
    cdef f64[::1] ev = f_E, fv = f_F

    for e in range(nE):
        if not em[e]:
            ev[e] = eqv[je]
            je += 1

    for f in range(nF):
        if not fm[f]:
            fv[f] = fqv[jf]
            jf += 1

    return f_E, f_F


def per_edge_energy(np.ndarray[f64, ndim=1] f_E, object L1, object LO):
    """Compute per-edge contribution to E_kin and E_pot.

    For each edge e, the energy contribution is:
        E_kin_e = f_E[e] * (L_1 f_E)[e]
        E_pot_e = f_E[e] * (L_O f_E)[e]

    These sum to the total: sum(E_kin_e) = <f|L_1|f>.

    Parameters
    ----------
    f_E : f64[nE] - edge signal
    L1, LO : (nE, nE) - Hodge and overlap Laplacians

    Returns
    -------
    E_kin_per_edge : f64[nE]
    E_pot_per_edge : f64[nE]
    """
    cdef Py_ssize_t nE = f_E.shape[0], e
    cdef np.ndarray[f64, ndim=1] L1f = np.asarray(L1.dot(f_E), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] LOf = np.asarray(LO.dot(f_E), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] ek = np.empty(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] ep = np.empty(nE, dtype=np.float64)
    cdef f64[::1] fv = f_E, l1v = L1f, lov = LOf, ekv = ek, epv = ep

    for e in range(nE):
        ekv[e] = fv[e] * l1v[e]
        epv[e] = fv[e] * lov[e]

    return ek, ep


# Section 6: Hyperslice, edge-type, and temporal integration


def hyperslice_quotient(Py_ssize_t dim, Py_ssize_t cell_idx,
                        Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF,
                        np.ndarray[i32, ndim=1] boundary_ptr,
                        np.ndarray[i32, ndim=1] boundary_idx,
                        np.ndarray[i32, ndim=1] v2e_ptr,
                        np.ndarray[i32, ndim=1] v2e_idx,
                        np.ndarray[i32, ndim=1] e2f_ptr,
                        np.ndarray[i32, ndim=1] e2f_idx,
                        np.ndarray[i32, ndim=1] B2_col_ptr,
                        np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Form a subcomplex from the hyperslice around a single cell.

    For a vertex, the hyperslice contains the vertex, edges incident to
    it, and faces incident to those edges. For an edge, it contains the
    edge, all of its boundary vertices, incident faces, and edges sharing
    a boundary vertex. For a face, it contains the face and its boundary
    edges. The result is closed downward.

    Parameters
    ----------
    dim : int
        Cell dimension (0 for vertex, 1 for edge, 2 for face).
    cell_idx : int
        Index of the cell.

    Returns
    -------
    v_mask, e_mask, f_mask : uint8 arrays
        Masks for the hyperslice subcomplex.
    """
    cdef np.ndarray[np.uint8_t, ndim=1] v_mask = np.zeros(nV, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] e_mask = np.zeros(nE, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] f_mask = np.zeros(nF, dtype=np.uint8)
    cdef np.uint8_t[::1] vm = v_mask, em = e_mask, fm = f_mask
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx
    cdef i32[::1] vep = v2e_ptr, vei = v2e_idx
    cdef i32[::1] efp = e2f_ptr, efi = e2f_idx
    cdef Py_ssize_t j, k, e, f, v

    if dim == 0:
        v = cell_idx
        vm[v] = 1
        for j in range(vep[v], vep[v + 1]):
            e = vei[j]
            em[e] = 1
            for k in range(bp[e], bp[e + 1]):
                if bi[k] != v:
                    vm[bi[k]] = 1
            for k in range(efp[e], efp[e + 1]):
                fm[efi[k]] = 1

    elif dim == 1:
        e = cell_idx
        em[e] = 1
        for j in range(bp[e], bp[e + 1]):
            vm[bi[j]] = 1
        for k in range(efp[e], efp[e + 1]):
            fm[efi[k]] = 1
        for j in range(bp[e], bp[e + 1]):
            v = bi[j]
            for k in range(vep[v], vep[v + 1]):
                em[vei[k]] = 1

    elif dim == 2:
        f = cell_idx
        fm[f] = 1
        if B2_col_ptr.shape[0] > 0:
            for j in range(B2_col_ptr[f], B2_col_ptr[f + 1]):
                em[B2_row_idx[j]] = 1

    return closure_of_faces_and_edges(v_mask, e_mask, f_mask,
                                      nV, nE, boundary_ptr, boundary_idx,
                                      B2_col_ptr, B2_row_idx)


def edge_type_quotient(np.ndarray[np.uint8_t, ndim=1] edge_types,
                       list type_codes_to_quotient,
                       Py_ssize_t nV,
                       np.ndarray[i32, ndim=1] boundary_ptr,
                       np.ndarray[i32, ndim=1] boundary_idx):
    """
    Build a subcomplex from edges with selected type codes.

    The input is a list of edge type codes and a full edge type array.
    All edges with codes in that list are selected, and the closure adds
    boundary vertices.

    Returns
    -------
    v_mask, e_mask, f_mask : uint8 arrays
        Masks for the closed subcomplex.
    """
    cdef Py_ssize_t nE = edge_types.shape[0], e
    cdef np.uint8_t[::1] et = edge_types
    cdef set codes = set(type_codes_to_quotient)

    cdef np.ndarray[np.uint8_t, ndim=1] e_mask = np.zeros(nE, dtype=np.uint8)
    cdef np.uint8_t[::1] em = e_mask

    for e in range(nE):
        if int(et[e]) in codes:
            em[e] = 1

    return closure_of_edges(e_mask, nV, boundary_ptr, boundary_idx)


def temporal_quotient(Py_ssize_t n_snapshots,
                      np.ndarray[np.uint8_t, ndim=1] time_mask,
                      list snapshot_sources,
                      list snapshot_targets,
                      Py_ssize_t nV):
    """
    Build a subcomplex from edges present at selected time steps.

    Edges across all snapshots with time_mask equal to 1 are collected
    into a union graph. The closure then adds boundary vertices for that
    union.

    Parameters
    ----------
    n_snapshots : int
        Number of temporal snapshots.
    time_mask : uint8[n_snapshots]
        Mask over snapshots, with 1 for snapshots in the subcomplex.
    snapshot_sources, snapshot_targets : list of arrays
        Source and target arrays for each snapshot.
    nV : int
        Vertex count across all snapshots.

    Returns
    -------
    v_mask : uint8[nV]
        Mask for vertices in the union subcomplex.
    e_mask_union : uint8[nE_union]
        Edge mask for the union graph.
    """
    cdef np.uint8_t[::1] tm = time_mask
    cdef Py_ssize_t t

    all_edges = set()
    for t in range(n_snapshots):
        if tm[t]:
            src_t = snapshot_sources[t]
            tgt_t = snapshot_targets[t]
            for e in range(len(src_t)):
                all_edges.add((int(src_t[e]), int(tgt_t[e])))

    if not all_edges:
        return np.zeros(nV, dtype=np.uint8), np.zeros(0, dtype=np.uint8)

    edges_list = sorted(all_edges)
    nE_union = len(edges_list)
    src_union = np.array([e[0] for e in edges_list], dtype=np.int32)
    tgt_union = np.array([e[1] for e in edges_list], dtype=np.int32)
    e_mask_union = np.ones(nE_union, dtype=np.uint8)

    return closure_of_edges(e_mask_union, nV, src_union, tgt_union)


# Section 7: Convenience - full quotient pipeline


def build_quotient(np.ndarray[f64, ndim=2] B1,
                   np.ndarray[np.uint8_t, ndim=1] v_mask,
                   np.ndarray[np.uint8_t, ndim=1] e_mask,
                   np.ndarray[np.uint8_t, ndim=1] f_mask,
                   np.ndarray[i32, ndim=1] B2_col_ptr,
                   np.ndarray[i32, ndim=1] B2_row_idx,
                   np.ndarray[f64, ndim=1] B2_vals,
                   object LO=None,
                   double alpha_G=0.0):
    """Run the full quotient pipeline from masks to operators and Betti numbers.

    The pipeline builds reindexing arrays, constructs B1_quot and B2_quot,
    checks the chain condition, computes relative Betti numbers, and
    optionally builds L1_quot and RL_1^quot.

    Parameters
    ----------
    B1 : f64[nV, nE]
        Signed incidence matrix for edges.
    v_mask, e_mask, f_mask : uint8 arrays
        Masks for the subcomplex.
    B2_col_ptr, B2_row_idx, B2_vals : arrays
        CSC representation of B2.
    LO : f64[nE, nE] or None
        Overlap Laplacian. If provided, the quotient RL_1 is computed.
    alpha_G : float
        Coupling constant for RL_1 = L_1 + alpha_G * L_O.

    Returns
    -------
    info : dict
        Dictionary with keys:
        - 'B1_quot', 'B2_quot' : quotient boundary operators
        - 'L1_quot' : Hodge Laplacian on quotient edges
        - 'betti_rel' : relative Betti numbers
        - 'chain_valid', 'chain_error' : chain condition check
        - 'v_reindex', 'e_reindex', 'f_reindex' : reindexing arrays
        - 'v_star' : basepoint index in the quotient
        - 'dims' : tuple (nV_quot, nE_quot, nF_quot)
        If LO is provided, also includes:
        - 'LO_quot' : overlap Laplacian on quotient edges
        - 'RL1_quot' : Relational Laplacian on quotient edges
    """
    v_re, v_star, e_re, f_re, nVq, nEq, nFq = quotient_maps(v_mask, e_mask, f_mask)

    B1q = quotient_B1(B1, v_mask, e_mask,
                      v_re, <i32>v_star, e_re, nVq, nEq)

    if nEq > 0 and nFq > 0 and B2_col_ptr.shape[0] > 0:
        B2q = quotient_B2(B2_col_ptr, B2_row_idx, B2_vals,
                          e_mask, f_mask, e_re, f_re, nEq, nFq)
    else:
        B2q = np.zeros((max(nEq, 1), 0), dtype=np.float64)

    chain_ok, chain_err = quotient_verify_chain(B1q, B2q)

    betti = relative_betti(B1q, B2q)

    # Build L1_quot = B1_q^T B1_q + B2_q B2_q^T
    L1q = B1q.T @ B1q
    if B2q.shape[0] > 0 and B2q.shape[1] > 0:
        L1q = L1q + B2q @ B2q.T

    result = {
        "B1_quot": B1q,
        "B2_quot": B2q,
        "L1_quot": L1q,
        "betti_rel": betti,
        "chain_valid": chain_ok,
        "chain_error": float(chain_err),
        "v_reindex": v_re,
        "e_reindex": e_re,
        "f_reindex": f_re,
        "v_star": int(v_star),
        "dims": (nVq, nEq, nFq),
    }

    # Optionally build RL_1^quot if overlap Laplacian is provided
    if LO is not None and nEq > 0:
        LO_full = np.asarray(LO, dtype=np.float64)
        surv = np.where(~np.asarray(e_mask).astype(bool))[0]
        LO_q = LO_full[np.ix_(surv, surv)]
        RL1q = L1q + alpha_G * LO_q
        result["LO_quot"] = LO_q
        result["RL1_quot"] = RL1q

    return result
