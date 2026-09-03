"""Metric curvature remains on actual C1 relation boundaries at every arity."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.cochain import Cochain
from rexgraph.graph import RexGraph
from rexgraph.metric_field import MetricCurvature, relation_metric_curvature


def test_branching_metric_curvature_retains_exact_share_incidences():
    # e0 = (0; 1,2), e1 = (0; 2,3,4): their common boundary participant
    # sees weights 1 and 1, while participant 2 sees declared shares 1/2 and
    # 1/3.  A pairwise source/target projection cannot produce this answer.
    rex = RexGraph.from_hypergraph(
        np.asarray([0, 3, 7], dtype=np.int32),
        np.asarray([0, 1, 2, 0, 2, 3, 4], dtype=np.int32),
    )
    metric = Cochain(1, np.asarray([2, 5], dtype=np.int64), source=rex)
    result = relation_metric_curvature(rex, metric)

    assert isinstance(result, MetricCurvature)
    assert result.local_mean.values.tolist() == [Fraction(7, 2), Fraction(2), Fraction(16, 5), Fraction(5), Fraction(5)]
    assert result.curvature.values.tolist() == [Fraction(3), Fraction(0), Fraction(6, 5), Fraction(0), Fraction(0)]
    assert result.relation_contribution.values.tolist() == [Fraction(21, 10), Fraction(21, 10)]
    assert result.total == Fraction(21, 5)
    assert sum(result.relation_contribution.values.tolist(), Fraction(0)) == result.total


def test_metric_curvature_keeps_repeated_primary_incidence_when_signed_b1_cancels():
    # The first relation has a repeated C0 participant.  Its signed B1 column
    # cancels there, but metric curvature must still see both relation legs.
    rex = RexGraph.from_hypergraph(
        np.asarray([0, 2, 4], dtype=np.int32),
        np.asarray([0, 0, 0, 1], dtype=np.int32),
    )
    result = relation_metric_curvature(
        rex, Cochain(1, np.asarray([2, 4], dtype=np.int64), source=rex)
    )

    assert result.local_mean.values.tolist() == [Fraction(8, 3), Fraction(4)]
    assert result.curvature.values.tolist() == [Fraction(8, 3), Fraction(0)]
    assert result.relation_contribution.values.tolist() == [Fraction(4, 3), Fraction(4, 3)]
    assert result.total == Fraction(8, 3)


def test_metric_curvature_rejects_unbound_or_non_c1_field():
    rex = RexGraph.from_graph(np.asarray([0]), np.asarray([1]))
    with pytest.raises(ValueError, match="grade 1"):
        relation_metric_curvature(rex, Cochain(0, np.ones(2), source=rex))
    with pytest.raises(ValueError, match="bound to its source"):
        relation_metric_curvature(rex, Cochain(1, np.ones(1), source=object()))
