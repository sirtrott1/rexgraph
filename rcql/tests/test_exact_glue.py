"""RCQL application of exact local-section gluing."""
from fractions import Fraction

import numpy as np
import pytest

from rcql import Exactness, Executor, parse


def branching_rex():
    from rexgraph.graph import RexGraph

    return RexGraph.from_hypergraph(
        np.array([0, 4, 8], dtype=np.int64),
        np.array([0, 1, 2, 3, 0, 1, 4, 5], dtype=np.int64),
    )


def test_one_phrase_glues_exact_relational_stalks_and_returns_residuals():
    from rexgraph.sheaf import ExactGlueResult, ExactSheaf

    rex = branching_rex()
    section = ExactSheaf(rex, stalk_dim=1)
    section.assign(0, [Fraction(1, 3)])
    section.assign(1, [Fraction(1, 2)])

    result = Executor(sources={"rex": rex}, params={"section": section}).execute(
        parse("FROM $rex RETURN GLUE($section)")
    )

    reading = result.values[0]
    assert isinstance(reading, ExactGlueResult)
    assert reading.ratio == Fraction(0)
    assert reading.obstruction_count == 2
    assert [item.residual for item in reading.obstructions] == [
        (Fraction(-1, 6),), (Fraction(-1, 6),),
    ]
    assert result.exactness == (Exactness.STRUCTURAL,)


def test_explain_types_phrase_gluing_without_executing_it():
    from rexgraph.sheaf import ExactSheaf

    rex = branching_rex()
    section = ExactSheaf(rex, stalk_dim=1)
    explanation = Executor(sources={"rex": rex}, params={"section": section}).execute(
        parse("EXPLAIN FROM $rex RETURN GLUE($section)")
    ).values[0]

    call = explanation["returns"][0]
    assert call["operator"] == "GLUE"
    assert call["result"]["kind"] == "ExactGlueResult"
    assert call["result"]["grade"] == 1
    assert "cross-state gluing requires an explicit chain-preserving correspondence map" in (
        call["preconditions"]
    )


def test_phrase_glue_refuses_a_stalk_from_a_different_relational_complex_before_execution():
    from rexgraph.sheaf import ExactSheaf

    rex = branching_rex()
    foreign = branching_rex()
    section = ExactSheaf(foreign, stalk_dim=1)

    with pytest.raises(TypeError, match="source-bound"):
        Executor(sources={"rex": rex}, params={"section": section}).execute(
            parse("FROM $rex RETURN GLUE($section)")
        )
