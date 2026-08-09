"""The float predicates and float sources that a second read turned up.

Each is the same shape: an exact answer computed from an inexact input, or a tolerance
standing in for arithmetic. They agree with the truth on the fixtures that covered them
and diverge off those fixtures, which is why the suite was green throughout.

    chain_valid        a float check at 1e-10 over the RAW B2, so it reported on the
                       faces as DECLARED while nF_hodge operates on the ones that bound
    grade_spread       gram(exact=True) over a densified float64 B1, which returns the
                       exact value of a double instead of the value
    flow residual      a float max, returning 0.0 for k = 3..12 because (k-1)*fl(1/(k-1))
                       happens to round to 1, which it does not at 483 arities below 4000
    edge_metric        float64 only, so a rational weight could not reach the exact tower
    left join          from_s = [] unconditionally, making it a synonym for inner while
                       the docstring said it adds S's relations on identified vertices
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.joins import join
from rexgraph.rational_trig import exact_channel_diagonals


#### chain_valid is the same predicate that decides nF_hodge


def _mixed_faces():
    """One face that bounds and one that does not."""
    rex = RexGraph(sources=np.array([0, 1, 2, 3, 0], dtype=np.int32),
                   targets=np.array([1, 2, 3, 0, 2], dtype=np.int32))
    rex.add_faces([np.array([0, 1, 4], dtype=np.int32),
                   np.array([2, 3, 4], dtype=np.int32)],
                  [np.array([1.0, 1.0, -1.0]), np.array([1.0, 1.0, -1.0])])
    rex._ensure_clean()
    return rex


def test_chain_valid_and_the_filter_agree():
    """They could not before: this ran on the raw B2 at a 1e-10 tolerance while nF_hodge
    ran the exact per-face predicate."""
    rex = _mixed_faces()
    assert rex.chain_valid is False
    assert rex.nF_hodge == 1, "the complex still uses the face that bounds"


def test_the_report_says_which_face_and_how_many_survived():
    """'invalid' and 'face 1 does not bound' are different amounts of help."""
    report = _mixed_faces().chain_report()
    assert report["unbounded"] == [1]
    assert report["n_faces"] == 2
    assert report["n_bounding"] == 1
    assert report["n_discarded"] == 1


def test_a_complex_whose_faces_all_bound_is_valid():
    rex = RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                   targets=np.array([1, 2, 0], dtype=np.int32))
    rex.add_faces([np.array([0, 1, 2], dtype=np.int32)], signs=None)
    rex._ensure_clean()
    assert rex.chain_valid is True
    assert rex.chain_report()["n_discarded"] == 0


#### grade_spread reads an exact source


@pytest.mark.parametrize("k,expected", [(2, "2"), (3, "3/2"), (4, "4/3"),
                                        (5, "5/4"), (7, "7/6")])
def test_the_shared_denominator_is_the_boundary_norm(k, expected):
    """A lone relation's Gram is 1x1, so the shared denominator IS T = 1 + 1/(k-1).
    Densifying first returned the exact value of the stored double: at k=4 that was
    432691404877902290367942354447019/324518553658426726783156020576256."""
    rex = RexGraph.from_hypergraph(np.array([0, k], dtype=np.int32),
                                   np.array(list(range(k)), dtype=np.int32))
    assert rex.grade_spread(1)["shared_denominator"] == expected


def test_the_denominator_stays_small():
    """The symptom that made it visible: an exact reading of a float has a denominator
    the complex never produced."""
    rex = RexGraph.from_hypergraph(np.array([0, 4], dtype=np.int32),
                                   np.array([0, 1, 2, 3], dtype=np.int32))
    assert Fraction(rex.grade_spread(1)["shared_denominator"]).denominator < 100


#### the metric can be exact


def test_a_rational_weight_reaches_the_exact_tower():
    """`edge_metric` is float64 by construction, so it could not carry one. T at weight
    1/3 is 2 * (1/3)^2 = 2/9, not the exact value of the double nearest 1/3."""
    rex = RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                   targets=np.array([1, 2, 0], dtype=np.int32),
                   w_E=[Fraction(1, 3), Fraction(1), Fraction(1)])
    assert rex.edge_metric_exact[0] == Fraction(1, 3)
    diagonals, _names = exact_channel_diagonals(rex)
    assert diagonals["L1_down"][0] == Fraction(2, 9)


def test_the_float_view_still_works_alongside_it():
    rex = RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                   targets=np.array([1, 2, 0], dtype=np.int32),
                   w_E=[Fraction(1, 3), Fraction(1), Fraction(1)])
    assert rex.edge_metric[0] == pytest.approx(1 / 3)


def test_a_float_weight_is_carried_as_the_double_it_is():
    """Not a failure, but not the intended rational either: exactness needs an exact
    source, and this reports what it was actually given."""
    rex = RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                   targets=np.array([1, 2, 0], dtype=np.int32),
                   w_E=np.array([1 / 3, 1.0, 1.0]))
    assert rex.edge_metric_exact[0] == Fraction(1 / 3)
    assert rex.edge_metric_exact[0] != Fraction(1, 3)


def test_an_unweighted_complex_reports_no_metric():
    rex = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                   targets=np.array([1, 2], dtype=np.int32))
    assert rex.edge_metric_exact is None


#### the three joins are three different joins


def _pair():
    """R over labels a,b,c with a-b and a-c; S over b,c,d with b-c and c-d."""
    R = RexGraph(sources=np.array([0, 0], dtype=np.int32),
                 targets=np.array([1, 2], dtype=np.int32))
    S = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                 targets=np.array([1, 2], dtype=np.int32))
    return R, S, ["a", "b", "c"], ["b", "c", "d"]


def test_left_adds_what_lies_on_identified_vertices():
    """It added nothing, which made it a synonym for inner. b-c is on two identified
    vertices and belongs; c-d touches d, which is not identified, and does not."""
    R, S, lr, ls = _pair()
    _rex, report = join(R, S, how="left", labels_r=lr, labels_s=ls)
    assert report["kept_from_r"] == 2
    assert report["kept_from_s"] == 1


def test_left_does_not_invent_a_vertex():
    """That is what separates it from outer."""
    R, S, lr, ls = _pair()
    left, _ = join(R, S, how="left", labels_r=lr, labels_s=ls)
    outer, _ = join(R, S, how="outer", labels_r=lr, labels_s=ls)
    assert left.nV == 3
    assert outer.nV == 4


def test_the_three_kinds_differ():
    R, S, lr, ls = _pair()
    sizes = {how: join(R, S, how=how, labels_r=lr, labels_s=ls)[0].nE
             for how in ("inner", "left", "outer")}
    assert len(set(sizes.values())) == 3, f"two joins are the same operation: {sizes}"


#### co-participation has one definition


def test_the_flow_operators_delegate_to_the_library_gramian():
    """Three implementations of |B1|^T |B1| had to agree by coincidence. They are one
    now, so a change to the Gramian reaches the learners."""
    from rexgraph.flow import coparticipation_neighbors

    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 4], dtype=np.int32))
    rex._ensure_clean()
    canonical = rex.overlap_counts_sparse.tocsr().copy()
    canonical.setdiag(0)
    canonical.eliminate_zeros()
    ptr, idx = coparticipation_neighbors(rex)
    assert ptr.tolist() == canonical.indptr.tolist()
    assert idx.tolist() == canonical.indices.tolist()


def test_the_face_aware_operator_is_reachable():
    """`flow_adjacency` is the only propagation operator that reads B2, and it had one
    grep hit: its own definition."""
    import rexgraph.flow as flow

    assert "flow_adjacency" in flow.__all__
    assert callable(flow.flow_adjacency)
