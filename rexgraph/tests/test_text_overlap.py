"""Native primary overlap against small independent tensor oracles."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.text_overlap import TextOverlapView


def fixture(**kwargs):
    return RexGraph.from_cells([4, [[0, 1, 2], [1, 3], [2], [0, 0]]],
                               w_E=[Q(2), Q(3), Q(5), Q(7)], **kwargs)


def oracle():
    x = np.array([[2, 0, 0, 0], [1, 3, 0, 0], [1, 0, 5, 0], [0, 3, 0, 0]], object)
    a = x.T@x
    np.fill_diagonal(a, 0)
    return a


@pytest.mark.parametrize("shape", [(4,), (4, 0), (4, 3)])
def test_exact_numeric_transpose_and_complex_blocks(shape):
    view = TextOverlapView(fixture())
    x = np.arange(np.prod(shape)).reshape(shape)
    expected = oracle()@x
    for action in (view.apply, view.transpose_apply):
        exact = action(x.astype(object), exact=True)
        np.testing.assert_array_equal(exact, expected)
        assert all(isinstance(v, Q) for v in exact.flat)
        np.testing.assert_allclose(action(x), np.asarray(expected, float))
        np.testing.assert_allclose(action(x*(1+2j)), np.asarray(expected, float)*(1+2j))
    assert view.symmetric and not view.psd
    assert view.source.nE == view.shape[0] == 4


def test_paper_overlap_path_without_invented_sentence_vertices():
    source = RexGraph.from_cells([4, [[0, 1], [1, 2], [2, 3]]])
    view = TextOverlapView(source)
    a = view.apply(np.eye(3, dtype=object), exact=True)
    np.testing.assert_array_equal(a, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    z = np.array([1, -1, 1], object)
    assert z@view.apply(z, exact=True) == -4  # Overlap is not a PSD Gram.


def test_exact_core_gram_routine_is_reused(monkeypatch):
    import rexgraph.text_overlap as overlap
    from rexgraph.channel_operator import _exact_gram_action
    calls = []
    def measured(columns, values):
        calls.append(len(columns))
        return _exact_gram_action(columns, values)
    monkeypatch.setattr(overlap, "_exact_gram_action", measured)
    view = TextOverlapView(fixture())
    view.apply(np.ones(4, object), exact=True)
    assert calls == [4]


def test_no_gram_materialization_adjacency_or_scipy(monkeypatch):
    from rexgraph.native_sparse import NativeSparse
    def refuse(*a, **kw):
        pytest.fail("pair expansion or optional oracle reached")
    r = RexGraph.from_cells([513, [list(range(512)), [511, 512]]])
    for name in ("_ensure_src_tgt", "_require_pairwise_c1"):
        monkeypatch.setattr(RexGraph, name, refuse)
    for name in ("product", "as_scipy"):
        monkeypatch.setattr(NativeSparse, name, refuse)
    view = TextOverlapView(r)
    assert view.matrix is None and view.matrix_factory is None
    expected = np.array([Q(1, 511), Q(1, 511)], object)
    np.testing.assert_array_equal(view.apply(np.ones(2, object), exact=True), expected)
    np.testing.assert_allclose(view.apply(np.ones(2)), np.asarray(expected, float))


@pytest.mark.parametrize("n", [0, 1, 4])
def test_absent_relation_axis(n):
    view = TextOverlapView(RexGraph.from_cells([n, []]))
    for exact in (False, True):
        assert view.apply(np.empty(0, object) if exact else np.empty(0), exact=exact).shape == (0,)


@pytest.mark.parametrize("g", ["raw", "normalized"])
@pytest.mark.parametrize("c", ["share", "count"])
def test_overlap_is_not_selected_by_channel_settings(g, c):
    r = fixture(g_channel=g)
    r._c_channel = c
    np.testing.assert_array_equal(TextOverlapView(r).apply(np.ones(4, object), exact=True), oracle()@np.ones(4, object))


def test_relation_axes_survive_section_coarsening_and_repeated_terms():
    from rexgraph.sectioning import add_sectioning, add_coarsening
    r = fixture()
    add_sectioning(r, "span", {"s0": [0, 1], "s1": [2, 3]})
    add_coarsening(r, "sentence", "span", [0, 0], ["sentence0"])
    view = TextOverlapView(r)
    assert view.shape == (4, 4) and view.domain_grade == 1
    np.testing.assert_array_equal(view.apply(np.ones(4, object), exact=True), oracle()@np.ones(4, object))


def test_source_mutation_invalidates_captured_overlap():
    r = fixture()
    view = TextOverlapView(r)
    r.set_cell_attrs([0, 1, 2, 3], w_E=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="state changed"):
        view.apply(np.ones(4))


@pytest.mark.parametrize("weight", [Q(1, 10**400), Q(1, 10**200), Q(10**400)])
def test_out_of_range_metrics_keep_exact_overlap(weight):
    r = RexGraph.from_cells([3, [[0, 1], [1, 2]]], w_E=[weight, weight])
    view = TextOverlapView(r)
    expected = np.full(2, weight**2, object)
    np.testing.assert_array_equal(view.apply(np.ones(2, object), exact=True), expected)
    with pytest.raises(FloatingPointError):
        view.apply(np.ones(2))


def test_zero_weight_and_negative_metric_contracts():
    r = RexGraph.from_cells([3, [[0, 1], [1, 2]]], w_E=[0, 1])
    np.testing.assert_array_equal(TextOverlapView(r).apply(np.ones(2, object), exact=True), [0, 0])
    with pytest.raises(ValueError, match="nonnegative"):
        TextOverlapView(RexGraph.from_cells([2, [[0, 1]]], w_E=[-1]))
