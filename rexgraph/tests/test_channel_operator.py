"""Native channel handles and the shared local/global sparse moment walk."""
import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.channel_operator import channel_operator
from rexgraph.graph import RexGraph
from rexgraph.scale_propagator import trace_moments


def test_experimental_import_is_only_a_compatibility_alias():
    from rexgraph._experimental import build_factored_operator as old
    from rexgraph.channel_operator import build_factored_operator as native
    assert old is native


@pytest.mark.parametrize("name", list("TGFC"))
def test_large_hub_action_never_forms_the_relation_square(name, monkeypatch):
    n = 4096
    rex = RexGraph.from_graph(np.zeros(n, dtype=np.int32), np.arange(1, n+1, dtype=np.int32))
    def forbidden(*args, **kwargs):
        pytest.fail("channel action formed a relation-square matrix")
    monkeypatch.setattr(RexGraph, "overlap_gramian_sparse", property(forbidden))
    import rexgraph.channel_operator as module
    monkeypatch.setattr(module, "build_sparse_channels", forbidden)
    op = channel_operator(rex, name)
    result = op.apply(np.ones(n))
    # All relations share the negative head. T=G=I+11^T; F=0;
    # the unweighted share/count C Laplacian annihilates constants.
    np.testing.assert_array_equal(result, np.full(n, n+1 if name in "TG" else 0))
    assert op.matrix is None


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
def test_local_moments_sum_to_existing_traces_and_match_reference(order):
    dense = np.array([[2., -1, 0], [-1, 3, -1], [0, -1, 2]])
    matrix = sp.csr_matrix(dense)
    local, total = trace_moments(matrix, order, local=True), trace_moments(matrix, order)
    for k in range(1, order+1):
        np.testing.assert_array_equal(local[k-1], np.diag(np.linalg.matrix_power(dense, k)))
        assert local[k-1].sum() == total[k-1]


def test_local_moment_walk_keeps_the_halved_multiply_count(monkeypatch):
    from rexgraph.native_sparse import NativeSparse
    matrix = sp.csr_matrix([[2., -1], [-1, 2]])
    original = NativeSparse.product
    calls = []
    def counted(left, right):
        calls.append((left.shape, right.shape))
        return original(left, right)
    monkeypatch.setattr(NativeSparse, "product", counted)
    trace_moments(matrix, 5, local=True)
    assert len(calls) == 2
    calls.clear()
    trace_moments(matrix, 2, local=True)
    assert calls == []


def test_diagonal_detects_changed_population():
    rex = RexGraph.from_graph([0], [1])
    action = channel_operator(rex, "T")
    rex.add_edges([1], [2])
    with pytest.raises(ValueError, match="changed"):
        action.diagonal(exact=True)
