"""
Tests for typed face selection, context face selection, and
void type composition.
"""
import numpy as np
import pytest

from rexgraph.core import _faces
from rexgraph.graph import RexGraph
from rexgraph import Absent, Relations, VertexTable
from rexgraph.native_rank import primary_columns
from rexgraph.state import to_state
from fractions import Fraction as Q


@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


# typed_face_selection

class TestTypedFaceSelection:

    def _adj(self, rex):
        return rex._adjacency_bundle

    def test_all_same_type(self, k4):
        """All edges same type -> all triangles are faces."""
        types = np.zeros(k4.nE, dtype=np.int32)
        adj_ptr, adj_idx, adj_edge = self._adj(k4)
        result = _faces.typed_face_selection(
            types, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE, 1)
        assert result['nF_realized'] == result['n_triangles']
        assert result['nF_void'] == 0

    def test_all_different_type(self, k4):
        """Each edge unique type -> zero faces, all voids."""
        types = np.arange(k4.nE, dtype=np.int32)
        adj_ptr, adj_idx, adj_edge = self._adj(k4)
        result = _faces.typed_face_selection(
            types, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE, k4.nE)
        assert result['nF_realized'] == 0
        assert result['nF_void'] == result['n_triangles']

    def test_partition_sums(self, k4):
        """nF_realized + nF_void = n_triangles."""
        types = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        adj_ptr, adj_idx, adj_edge = self._adj(k4)
        result = _faces.typed_face_selection(
            types, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE, 2)
        assert result['nF_realized'] + result['nF_void'] == result['n_triangles']

    def test_face_types_match(self, k4):
        """face_types[f] matches the shared type of that face's edges."""
        types = np.zeros(k4.nE, dtype=np.int32)
        types[3] = 1  # edge (1,2) is type 1
        adj_ptr, adj_idx, adj_edge = self._adj(k4)
        result = _faces.typed_face_selection(
            types, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE, 2)
        ft = result['face_types']
        re = result['realized_edges']
        for f in range(result['nF_realized']):
            e0, e1, e2 = re[f*3], re[f*3+1], re[f*3+2]
            assert types[e0] == types[e1] == types[e2] == ft[f]

    def test_no_triangles(self, triangle):
        """Graph with no triangles -> zero faces and zero voids."""
        # Path graph: 0-1, 1-2, 0-2 is a triangle, but a tree is not
        tree = RexGraph.from_graph([0, 1, 2], [1, 2, 3])
        types = np.zeros(tree.nE, dtype=np.int32)
        adj_ptr, adj_idx, adj_edge = tree._adjacency_bundle
        result = _faces.typed_face_selection(
            types, adj_ptr, adj_idx, adj_edge, tree.nV, tree.nE, 1)
        assert result['n_triangles'] == 0

    def test_graph_method(self, k4):
        """RexGraph.typed_face_selection returns a RexGraph."""
        types = np.zeros(k4.nE, dtype=np.int32)
        rex = k4.typed_face_selection(types)
        assert isinstance(rex, RexGraph)
        assert rex.nF > 0


# context_face_selection

class TestContextFaceSelection:

    def _adj(self, rex):
        return rex._adjacency_bundle

    def test_full_context_all_realized(self, k4):
        """Context covering all vertices -> all triangles realized."""
        ctx = np.ones((1, k4.nV), dtype=np.uint8)
        adj_ptr, adj_idx, adj_edge = self._adj(k4)
        result = _faces.context_face_selection(
            k4.B1, ctx, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        assert result['nF'] == result['n_triangles']

    def test_empty_context_none_realized(self, k4):
        """All zero context -> zero faces."""
        ctx = np.zeros((1, k4.nV), dtype=np.uint8)
        adj_ptr, adj_idx, adj_edge = self._adj(k4)
        result = _faces.context_face_selection(
            k4.B1, ctx, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        assert result['nF'] == 0

    def test_chain_condition(self, k4):
        """B1 @ B2_selected = 0."""
        ctx = np.ones((1, k4.nV), dtype=np.uint8)
        rex = k4.context_face_selection(ctx)
        if rex.nF > 0:
            product = rex.B1 @ rex.B2
            assert np.max(np.abs(product)) < 1e-10

    def test_per_context_counts_shape(self, k4):
        n_ctx = 3
        ctx = np.ones((n_ctx, k4.nV), dtype=np.uint8)
        adj_ptr, adj_idx, adj_edge = self._adj(k4)
        result = _faces.context_face_selection(
            k4.B1, ctx, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        assert result['per_context_face_count'].shape == (n_ctx,)
        assert result['per_context_void_fraction'].shape == (n_ctx,)

    def test_void_fraction_range(self, k4):
        ctx = np.ones((2, k4.nV), dtype=np.uint8)
        ctx[1, 0] = 0  # second context misses vertex 0
        adj_ptr, adj_idx, adj_edge = self._adj(k4)
        result = _faces.context_face_selection(
            k4.B1, ctx, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        vf = result['per_context_void_fraction']
        assert np.all(vf >= 0.0)
        assert np.all(vf <= 1.0)

    def test_graph_method_returns_rex(self, k4):
        ctx = np.ones((1, k4.nV), dtype=np.uint8)
        rex = k4.context_face_selection(ctx)
        assert isinstance(rex, RexGraph)

    def test_graph_method_stores_result(self, k4):
        ctx = np.ones((2, k4.nV), dtype=np.uint8)
        rex = k4.context_face_selection(ctx)
        assert hasattr(rex, '_context_face_result')
        assert 'per_context_face_count' in rex._context_face_result


# void_type_composition


@pytest.mark.parametrize("selection", ["context", "typed"])
@pytest.mark.parametrize("heads", [[0, 0, 0], [1, 0, 1], [1, 1, 1]])
def test_selection_preserves_declared_state_and_orients_selected_faces(selection, heads):
    rex = RexGraph.from_relations(Relations.from_supports(
        [[1, 0], [0, 2], [2, 1]], heads=heads,
        vertices=VertexTable(("a", "b", "c", "isolated")),
        weights=[Absent, Q(2, 7), Absent], relation_ids=[17, 18, 19],
        attributes={1: {0: {"note": [Q(1, 7), np.array([1, 2])]}}}),
        g_channel="normalized", c_channel="count")
    rex.set_provenance({"source": "selection", "exact": Q(1, 7)})
    before = to_state(rex).header["digest"]
    selected = (rex.context_face_selection(np.ones((1, rex.nV), np.uint8)) if selection == "context"
                else rex.typed_face_selection(np.zeros(rex.nE, np.int32)))
    assert (selected.nV, selected.nE, selected.nF, selected.nF_hodge) == (4, 3, 1, 1)
    assert selected.chain_valid
    assert np.array_equal(selected.B1 @ selected.B2, np.zeros((4, 1)))
    assert primary_columns(selected) == primary_columns(rex)
    assert selected.relation_keys == (17, 18, 19)
    assert selected.relations.weight.values().tolist() == [Absent, Q(2, 7), Absent]
    assert selected.g_channel == "normalized" and selected.c_channel == "count"
    assert selected.relations.vertices.ids == ("a", "b", "c", "isolated")
    assert selected.provenance == rex.provenance
    assert selected.get_metadata(1, 0, "note")[0] == Q(1, 7)
    selected.get_metadata(1, 0, "note")[1][0] = 99
    assert rex.get_metadata(1, 0, "note")[1][0] == 1
    assert to_state(rex).header["digest"] == before


@pytest.mark.parametrize("selection", ["context", "typed"])
def test_void_diagnostics_use_declared_orientation(selection):
    rex = RexGraph.from_relations(Relations.from_supports([[1, 0], [0, 2], [2, 1]], heads=[1, 0, 1]))
    if selection == "context":
        selected = rex.context_face_selection(np.zeros((1, rex.nV), np.uint8))
        result = selected._context_face_result
    else:
        selected = rex.typed_face_selection(np.arange(rex.nE, dtype=np.int32))
        result = selected._typed_face_result
    edges = result["void_edges"].reshape(-1, 3)
    signs = result["void_signs"].reshape(-1, 3)
    assert selected.nF == 0 and len(edges) == 1
    assert np.array_equal(rex.B1[:, edges[0]] @ signs[0], np.zeros(rex.nV))


@pytest.mark.parametrize("bad", [np.ones(3), np.ones((1, 2)), [[0, 256, 0]],
                                  [[0, -1, 0]], [[0, .5, 0]], [[0, np.nan, 0]],
                                  [["0", "1", "0"]]])
def test_invalid_context_is_rejected_before_native_selection(triangle, bad):
    with pytest.raises(ValueError, match="context matrix must be binary"):
        triangle.context_face_selection(bad)


def test_replacing_faces_preserves_isolates_and_removes_face_metadata(k4):
    k4._ensure_vertex_count(5)
    k4.attach_metadata(2, 0, "old", "face")
    selected = k4.context_face_selection(np.ones((1, 5), np.uint8))
    assert selected.nV == 5 and selected.nF == 4 and selected.chain_valid
    assert selected.get_metadata(2, 0, "old") is None


def test_empty_context_family_is_valid(triangle):
    selected = triangle.context_face_selection(np.empty((0, triangle.nV), np.uint8))
    assert selected.nF == 0 and selected._context_face_result["nF_void"] == 1

class TestVoidTypeComposition:

    def test_counts_sum(self, k4):
        """pair_counts values sum to nF_void."""
        types = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        adj_ptr, adj_idx, adj_edge = k4._adjacency_bundle
        sel = _faces.typed_face_selection(
            types, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE, 2)
        if sel['nF_void'] == 0:
            pytest.skip("no voids in this configuration")
        comp = _faces.void_type_composition(
            sel['void_edges'], types, sel['nF_void'], 2)
        total = sum(comp['pair_counts'].values())
        assert total == sel['nF_void']

    def test_fractions_sum_to_one(self, k4):
        types = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        adj_ptr, adj_idx, adj_edge = k4._adjacency_bundle
        sel = _faces.typed_face_selection(
            types, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE, 2)
        if sel['nF_void'] == 0:
            pytest.skip("no voids in this configuration")
        comp = _faces.void_type_composition(
            sel['void_edges'], types, sel['nF_void'], 2)
        total_frac = sum(comp['pair_fractions'].values())
        assert abs(total_frac - 1.0) < 1e-10

    def test_cross_type_only(self, k4):
        """Every void has 2+ distinct edge types."""
        types = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        adj_ptr, adj_idx, adj_edge = k4._adjacency_bundle
        sel = _faces.typed_face_selection(
            types, adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE, 2)
        if sel['nF_void'] == 0:
            pytest.skip("no voids in this configuration")
        comp = _faces.void_type_composition(
            sel['void_edges'], types, sel['nF_void'], 2)
        for tp in comp['type_pairs']:
            assert len(tp) >= 2
