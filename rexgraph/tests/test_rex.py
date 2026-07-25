"""
Tests for rexgraph.core._rex - structural operations for the relational complex.

Verifies:
    - Edge classification (standard, self-loop, branching, witness)
    - Vertex derivation from edge boundaries
    - CSR incidence structures (vertex-to-edge, edge-to-face)
    - Branching edge clique expansion
    - Hyperslice queries at all dimensions
    - Edge insertion and deletion with vertex lifecycle
    - Dimensional projection and Betti deltas
    - Subsumption embeddings (graph, hypergraph, simplicial 2-complex)
    - Chain condition verification
    - Coboundary queries
"""
import numpy as np
import pytest

from rexgraph.core import _rex


# Fixtures

@pytest.fixture
def triangle_arrays():
    """Triangle graph: 3 vertices, 3 edges."""
    src = np.array([0, 1, 0], dtype=np.int32)
    tgt = np.array([1, 2, 2], dtype=np.int32)
    return src, tgt, 3, 3


@pytest.fixture
def k4_arrays():
    """K4: 4 vertices, 6 edges."""
    src = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
    return src, tgt, 4, 6


@pytest.fixture
def self_loop_arrays():
    """Graph with a self-loop."""
    src = np.array([0, 1, 2], dtype=np.int32)
    tgt = np.array([1, 2, 2], dtype=np.int32)
    return src, tgt, 3, 3


# Edge Classification

class TestEdgeClassification:

    def test_standard_edges(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        types = _rex.classify_edges_standard(nE, src, tgt)
        assert types.shape == (nE,)
        assert np.all(types == _rex.EDGE_STANDARD)

    def test_self_loop_detected(self):
        src = np.array([0, 1, 2], dtype=np.int32)
        tgt = np.array([1, 2, 2], dtype=np.int32)
        types = _rex.classify_edges_standard(3, src, tgt)
        assert types[0] == _rex.EDGE_STANDARD
        assert types[1] == _rex.EDGE_STANDARD
        assert types[2] == _rex.EDGE_SELF_LOOP

    def test_general_classification_standard(self, triangle_arrays):
        """General classifier on a simple graph matches standard classifier."""
        src, tgt, nV, nE = triangle_arrays
        bp = np.zeros(nE + 1, dtype=np.int32)
        bi_list = []
        for j in range(nE):
            bi_list.extend([src[j], tgt[j]])
            bp[j + 1] = bp[j] + 2
        bi = np.array(bi_list, dtype=np.int32)
        types, sizes = _rex.classify_edges_general(nE, bp, bi)
        assert np.all(types == _rex.EDGE_STANDARD)
        assert np.all(sizes == 2)

    def test_branching_edge(self):
        """A hyperedge connecting 3 vertices is classified as BRANCHING."""
        bp = np.array([0, 3], dtype=np.int32)
        bi = np.array([0, 1, 2], dtype=np.int32)
        types, sizes = _rex.classify_edges_general(1, bp, bi)
        assert types[0] == _rex.EDGE_BRANCHING
        assert sizes[0] == 3

    def test_witness_edge(self):
        """An edge with a single boundary vertex (multiplicity 1) is WITNESS."""
        bp = np.array([0, 1], dtype=np.int32)
        bi = np.array([5], dtype=np.int32)
        types, sizes = _rex.classify_edges_general(1, bp, bi)
        assert types[0] == _rex.EDGE_WITNESS
        assert sizes[0] == 1


# Vertex Derivation

class TestVertexDerivation:

    def test_vertex_count(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        nV_out, degree, in_deg, out_deg = _rex.derive_vertex_set(nE, src, tgt)
        assert nV_out == 3

    def test_degree_sum(self, triangle_arrays):
        """Sum of degrees = 2 * nE (handshaking lemma)."""
        src, tgt, nV, nE = triangle_arrays
        nV_out, degree, in_deg, out_deg = _rex.derive_vertex_set(nE, src, tgt)
        assert degree.sum() == 2 * nE

    def test_degree_shape(self, k4_arrays):
        src, tgt, nV, nE = k4_arrays
        nV_out, degree, in_deg, out_deg = _rex.derive_vertex_set(nE, src, tgt)
        assert degree.shape == (nV_out,)
        assert in_deg.shape == (nV_out,)
        assert out_deg.shape == (nV_out,)

    def test_k4_uniform_degree(self, k4_arrays):
        """Every vertex in K4 has degree 3."""
        src, tgt, nV, nE = k4_arrays
        _, degree, _, _ = _rex.derive_vertex_set(nE, src, tgt)
        assert np.all(degree == 3)


# CSR Incidence

class TestCSRIncidence:

    def test_v2e_shape(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        vptr, vidx = _rex.build_vertex_to_edge_csr(nV, nE, src, tgt)
        assert vptr.shape == (nV + 1,)
        assert vptr[0] == 0
        assert vptr[-1] == 2 * nE  # each edge contributes to 2 vertices

    def test_v2e_contains_correct_edges(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        vptr, vidx = _rex.build_vertex_to_edge_csr(nV, nE, src, tgt)
        # Vertex 0 should be incident to edges 0 (0-1) and 2 (0-2)
        v0_edges = set(vidx[vptr[0]:vptr[1]])
        assert 0 in v0_edges
        assert 2 in v0_edges

    def test_e2f_shape(self):
        """Edge-to-face CSR from B2 in CSC format."""
        # Single triangle: 3 edges, 1 face. B2 has 3 nonzeros.
        nE, nF = 3, 1
        B2_cp = np.array([0, 3], dtype=np.int32)
        B2_ri = np.array([0, 1, 2], dtype=np.int32)
        eptr, eidx = _rex.build_edge_to_face_csr(nE, nF, B2_cp, B2_ri)
        assert eptr.shape == (nE + 1,)
        # Each of the 3 edges is in 1 face
        for e in range(nE):
            assert eptr[e + 1] - eptr[e] == 1


# Branching Edge Expansion

class TestBranchingExpansion:

    def test_standard_edges_pass_through(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        types = _rex.classify_edges_standard(nE, src, tgt)
        bp = np.zeros(nE + 1, dtype=np.int32)
        bi_list = []
        for j in range(nE):
            bi_list.extend(sorted([src[j], tgt[j]]))
            bp[j + 1] = bp[j] + 2
        bi = np.array(bi_list, dtype=np.int32)
        ns, nt, nw, pe = _rex.clique_expand_branching(nE, bp, bi, types)
        assert len(ns) == nE  # no expansion needed
        assert np.allclose(nw, 1.0)

    def test_branching_expansion_count(self):
        """A hyperedge with 4 vertices produces C(4,2) = 6 edges."""
        bp = np.array([0, 4], dtype=np.int32)
        bi = np.array([0, 1, 2, 3], dtype=np.int32)
        types = np.array([_rex.EDGE_BRANCHING], dtype=np.int32)
        ns, nt, nw, pe = _rex.clique_expand_branching(1, bp, bi, types)
        assert len(ns) == 6

    def test_branching_weight(self):
        """Expanded edges from k=3 vertices get weight 1/(k-1) = 0.5."""
        bp = np.array([0, 3], dtype=np.int32)
        bi = np.array([0, 1, 2], dtype=np.int32)
        types = np.array([_rex.EDGE_BRANCHING], dtype=np.int32)
        ns, nt, nw, pe = _rex.clique_expand_branching(1, bp, bi, types)
        assert len(ns) == 3  # C(3,2) = 3
        assert np.allclose(nw, 0.5)


# Hyperslice Queries

class TestHyperslice:

    def test_vertex_hyperslice(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        vptr, vidx = _rex.build_vertex_to_edge_csr(nV, nE, src, tgt)
        above, lateral = _rex.hyperslice_vertex_i32(0, vptr, vidx, src, tgt)
        assert len(above) == 2  # vertex 0 has degree 2
        assert len(lateral) >= 1  # at least 1 neighbor

    def test_edge_hyperslice(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        vptr, vidx = _rex.build_vertex_to_edge_csr(nV, nE, src, tgt)
        # No faces, so make empty e2f
        eptr = np.zeros(nE + 1, dtype=np.int32)
        eidx = np.array([], dtype=np.int32)
        below, above, lateral = _rex.hyperslice_edge_i32(
            0, src, tgt, eptr, eidx, vptr, vidx)
        assert len(below) == 2  # edge 0 has 2 boundary vertices
        assert len(above) == 0  # no faces
        assert len(lateral) >= 1  # at least 1 lateral edge

    def test_face_hyperslice(self):
        """Face hyperslice returns boundary edges and lateral faces."""
        nE, nF = 6, 4  # K4: 6 edges, 4 faces
        # Simplified B2 CSC: each face has 3 boundary edges
        B2_cp = np.array([0, 3, 6, 9, 12], dtype=np.int32)
        B2_ri = np.array([0, 1, 3, 0, 2, 4, 1, 2, 5, 3, 4, 5], dtype=np.int32)
        eptr, eidx = _rex.build_edge_to_face_csr(nE, nF, B2_cp, B2_ri)
        below, lateral = _rex.hyperslice_face_i32(0, nF, B2_cp, B2_ri, eptr, eidx)
        assert len(below) == 3  # triangle has 3 boundary edges
        assert len(lateral) >= 1  # shares edges with other faces


# Edge Insertion and Deletion

class TestEdgeMutation:

    def test_insert_extends_arrays(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        new_s = np.array([1, 0], dtype=np.int32)
        new_t = np.array([3, 3], dtype=np.int32)
        out_s, out_t, nV_new = _rex.insert_edges(nV, nE, src, tgt, new_s, new_t)
        assert len(out_s) == nE + 2
        assert nV_new == 4  # vertex 3 is new

    def test_insert_preserves_existing(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        new_s = np.array([3], dtype=np.int32)
        new_t = np.array([4], dtype=np.int32)
        out_s, out_t, nV_new = _rex.insert_edges(nV, nE, src, tgt, new_s, new_t)
        assert np.array_equal(out_s[:nE], src)
        assert np.array_equal(out_t[:nE], tgt)

    def test_delete_removes_edges(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        mask = np.array([1, 0, 0], dtype=np.int32)  # delete edge 0
        ns, nt, nV_new, vm, em = _rex.delete_edges(nV, nE, src, tgt, mask)
        assert len(ns) == 2

    def test_delete_removes_isolated_vertices(self):
        """Deleting all edges incident to a vertex removes that vertex."""
        # Path: 0-1-2. Delete edge 0-1 -> vertex 0 becomes isolated.
        src = np.array([0, 1], dtype=np.int32)
        tgt = np.array([1, 2], dtype=np.int32)
        mask = np.array([1, 0], dtype=np.int32)
        ns, nt, nV_new, vm, em = _rex.delete_edges(3, 2, src, tgt, mask)
        assert nV_new == 2  # vertex 0 removed
        assert vm[0] == -1  # vertex 0 is gone

    def test_delete_vertex_map_consistent(self, k4_arrays):
        src, tgt, nV, nE = k4_arrays
        mask = np.zeros(nE, dtype=np.int32)
        mask[0] = 1  # delete one edge
        ns, nt, nV_new, vm, em = _rex.delete_edges(nV, nE, src, tgt, mask)
        # All vertices should survive (K4 - 1 edge still has all vertices)
        assert nV_new == 4
        for v in range(4):
            assert vm[v] >= 0

    def test_edge_map_consistent(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        mask = np.array([0, 1, 0], dtype=np.int32)  # delete edge 1
        ns, nt, nV_new, vm, em = _rex.delete_edges(nV, nE, src, tgt, mask)
        assert em[0] >= 0  # edge 0 survives
        assert em[1] == -1  # edge 1 deleted
        assert em[2] >= 0  # edge 2 survives


# Dimensional Projection

class TestDimensionalProjection:

    def test_full_dimension(self):
        """At dimension 2, Betti numbers follow the standard formula."""
        betti, chi = _rex.project_to_dimension(2, 4, 6, 4, 3, 3)
        # K4: beta_0=1, beta_1=0, beta_2=1
        assert betti[0] == 1
        assert chi == 4 - 6 + 4  # = 2

    def test_dimension_1(self):
        """At dimension 1, faces are ignored."""
        betti, chi = _rex.project_to_dimension(1, 4, 6, 4, 3, 3)
        # Without faces: beta_0 = nV - rank_B1 = 4 - 3 = 1
        # beta_1 = nE - rank_B1 = 6 - 3 = 3
        assert betti[0] == 1
        assert betti[1] == 3
        assert betti[2] == 0
        assert chi == 4 - 6  # = -2

    def test_dimension_0(self):
        """At dimension 0, only vertices remain."""
        betti, chi = _rex.project_to_dimension(0, 4, 6, 4, 3, 3)
        assert betti[0] == 4
        assert betti[1] == 0
        assert betti[2] == 0
        assert chi == 4

    def test_euler_relation(self):
        """chi = beta_0 - beta_1 + beta_2 at every dimension."""
        for dim in [0, 1, 2]:
            betti, chi = _rex.project_to_dimension(dim, 4, 6, 4, 3, 3)
            assert chi == betti[0] - betti[1] + betti[2]


# Betti Deltas

class TestBettiDeltas:

    def test_deltas_shape(self):
        d = _rex.betti_deltas(4, 6, 4, 3, 3)
        assert "2to1" in d
        assert "1to0" in d
        assert len(d["2to1"]) == 3
        assert len(d["1to0"]) == 3


# Subsumption Embeddings

class TestSubsumption:

    def test_from_graph(self):
        src = np.array([0, 1, 0], dtype=np.int32)
        tgt = np.array([1, 2, 2], dtype=np.int32)
        s, t, types, nV = _rex.from_graph(3, src, tgt)
        assert nV == 3
        assert len(s) == 3
        assert np.all(types == _rex.EDGE_STANDARD)

    def test_from_graph_self_loop(self):
        src = np.array([0, 1], dtype=np.int32)
        tgt = np.array([1, 1], dtype=np.int32)
        s, t, types, nV = _rex.from_graph(2, src, tgt)
        assert types[0] == _rex.EDGE_STANDARD
        assert types[1] == _rex.EDGE_SELF_LOOP

    def test_from_hypergraph(self):
        hp = np.array([0, 2, 5], dtype=np.int32)
        hi = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        bp, bi, types = _rex.from_hypergraph(5, hp, hi)
        assert types[0] == _rex.EDGE_STANDARD  # 2 vertices
        assert types[1] == _rex.EDGE_BRANCHING  # 3 vertices

    def test_from_simplicial_2complex(self):
        """Simplicial 2-complex produces B2 in CSC with correct structure."""
        src = np.array([0, 1, 0], dtype=np.int32)
        tgt = np.array([1, 2, 2], dtype=np.int32)
        # 1 triangle using edges 0, 1, 2
        e0 = np.array([0], dtype=np.int32)
        e1 = np.array([1], dtype=np.int32)
        e2 = np.array([2], dtype=np.int32)
        s0 = np.array([1.0], dtype=np.float64)
        s1 = np.array([1.0], dtype=np.float64)
        s2 = np.array([-1.0], dtype=np.float64)
        cp, ri, vv = _rex.from_simplicial_2complex(3, src, tgt, e0, e1, e2, s0, s1, s2)
        assert cp[0] == 0
        assert cp[1] == 3  # 3 nonzeros for 1 triangle
        assert len(ri) == 3
        assert len(vv) == 3


# Chain Condition

class TestChainCondition:

    def test_verify_chain_standard(self):
        """B1 * B2 = 0 for a triangle with correct signs."""
        # B1 in CSR: 3x3 matrix
        # Edge 0: 0->1, Edge 1: 1->2, Edge 2: 0->2
        rp = np.array([0, 2, 2, 4, 4, 6, 6], dtype=np.int32)
        # Actually, let's use a simpler encoding.
        # B1 as CSR (nV=3 rows, nE=3 cols):
        # row 0: col 0 = -1, col 2 = -1
        # row 1: col 0 = +1, col 1 = -1
        # row 2: col 1 = +1, col 2 = +1
        B1_rp = np.array([0, 2, 4, 6], dtype=np.int32)
        B1_ci = np.array([0, 2, 0, 1, 1, 2], dtype=np.int32)
        B1_v = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float64)

        # B2 in CSC (nE=3 rows, nF=1 cols):
        # col 0: row 0 = +1, row 1 = +1, row 2 = -1
        B2_cp = np.array([0, 3], dtype=np.int32)
        B2_ri = np.array([0, 1, 2], dtype=np.int32)
        B2_v = np.array([1.0, 1.0, -1.0], dtype=np.float64)

        ok, max_err = _rex.verify_chain_condition_Bk(
            3, B1_rp, B1_ci, B1_v, B2_cp, B2_ri, B2_v)
        assert ok
        assert max_err < 1e-10


# Coboundary Queries

class TestCoboundary:

    def test_coboundary_vertex(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        vptr, vidx = _rex.build_vertex_to_edge_csr(nV, nE, src, tgt)
        edges = _rex.coboundary_vertex_i32(0, vptr, vidx)
        assert len(edges) == 2  # vertex 0 is in edges 0 and 2

    def test_coboundary_edge(self):
        nE, nF = 3, 1
        B2_cp = np.array([0, 3], dtype=np.int32)
        B2_ri = np.array([0, 1, 2], dtype=np.int32)
        eptr, eidx = _rex.build_edge_to_face_csr(nE, nF, B2_cp, B2_ri)
        faces = _rex.coboundary_edge_i32(0, eptr, eidx)
        assert len(faces) == 1
        assert faces[0] == 0


# Convenience

class TestConvenience:

    def test_build_1rex(self, triangle_arrays):
        src, tgt, nV, nE = triangle_arrays
        d = _rex.build_1rex(nV, nE, src, tgt)
        assert d["nV"] == 3
        assert d["nE"] == 3
        assert "degree" in d
        assert "edge_types" in d
        assert "v2e_ptr" in d
