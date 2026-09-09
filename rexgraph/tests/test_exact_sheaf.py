"""Exact incidence-level gluing over the primary relational boundaries."""
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.sheaf import ExactSheaf, Sheaf, UndeclaredRestrictionError


def two_branching_relations():
    """Two C1 relations meeting at two vertices, with a non-binary share."""
    return RexGraph.from_hypergraph(
        np.array([0, 4, 8], dtype=np.int64),
        np.array([0, 1, 2, 3, 0, 1, 4, 5], dtype=np.int64),
    )


def test_exact_stalks_keep_a_branching_share_exact():
    sheaf = ExactSheaf(two_branching_relations(), stalk_dim=1)
    sheaf.assign(0, [Fraction(1, 3)])
    sheaf.assign(1, [Fraction(1, 3)])

    result = sheaf.glue()

    assert result.ratio == Fraction(1)
    assert result.h0 == 1
    assert result.obstruction_count == 0
    assert sheaf._stalks == [(Fraction(1, 3),), (Fraction(1, 3),)]


def test_exact_gluing_retains_each_failed_mediator_and_its_residual():
    sheaf = ExactSheaf(two_branching_relations(), stalk_dim=1)
    sheaf.assign(0, [Fraction(1, 3)])
    sheaf.assign(1, [Fraction(1, 2)])

    result = sheaf.glue()

    # The relations meet at 0 and 1.  A one-pair failure must not erase either
    # incidence's evidence or relabel a count as a cohomology class.
    assert result.gluable == 1
    assert result.glued == 0
    assert result.ratio == Fraction(0)
    assert result.h0 == 2
    assert result.obstruction_count == 2
    assert result.failed_pairs == ((0, 1),)
    assert [(item.mediator, item.residual) for item in result.obstructions] == [
        (0, (Fraction(-1, 6),)),
        (1, (Fraction(-1, 6),)),
    ]


def test_exact_boundary_restrictions_read_declared_head_and_share_not_float_b1():
    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6], dtype=np.int64),
        np.array([0, 1, 2, 3, 0, 1], dtype=np.int64),
    )
    sheaf = ExactSheaf(rex, stalk_dim=1)
    sheaf.bind_boundary()

    assert sheaf._R[(0, 0)] == ((Fraction(-1),),)
    assert sheaf._R[(0, 1)] == ((Fraction(1, 3),),)
    assert sheaf._R[(0, 2)] == ((Fraction(1, 3),),)
    assert sheaf._R[(0, 3)] == ((Fraction(1, 3),),)
    assert sheaf._R[(1, 0)] == ((Fraction(-1),),)
    assert sheaf._R[(1, 1)] == ((Fraction(1),),)


def test_exact_path_refuses_approximate_section_and_restriction_input():
    sheaf = ExactSheaf(two_branching_relations(), stalk_dim=1)

    with pytest.raises(TypeError, match="approximate"):
        sheaf.assign(0, [1.0 / 3.0])
    with pytest.raises(TypeError, match="approximate"):
        sheaf.restrict(0, [[1.0]])


def test_empty_complex_has_zero_gluing_components_on_both_readers():
    rex = RexGraph.from_hypergraph(
        np.array([0], dtype=np.int64), np.array([], dtype=np.int64),
    )

    approximate = Sheaf(rex).glue()
    exact = ExactSheaf(rex).glue()

    assert approximate["H0"] == 0
    assert exact.h0 == 0
    assert exact.components == ()


def test_strict_exact_sections_refuse_an_undeclared_cross_state_incidence():
    rex = RexGraph.from_hypergraph(
        np.array([0, 2], dtype=np.int64), np.array([0, 1], dtype=np.int64),
    )
    sheaf = ExactSheaf(rex, grade=0, require_declared_restrictions=True)
    sheaf.assign(0, [Fraction(7)])
    sheaf.assign(1, [Fraction(7)])

    with pytest.raises(UndeclaredRestrictionError) as error:
        sheaf.glue()

    assert error.value.missing == ((0, 0), (1, 0))
    sheaf.restrict(0, [[1]], mediator=0)
    sheaf.restrict(1, [[1]], mediator=0)
    assert sheaf.glue().ratio == Fraction(1)
