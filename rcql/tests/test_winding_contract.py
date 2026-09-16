"""Winding retains the input arithmetic contract and the cycle frame scale."""
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph import RexGraph
from rexgraph.cochain import Cochain
from rcql import Executor, parse


@pytest.mark.parametrize("values,expected,arithmetic", [
    ([1, 1, 1], 3, "integer"), ([1., 1., 1.], 3., "approximate"),
    ([Q(1, 3)] * 3, Q(1), "rational"), ([2**90 + 1] * 3, 3 * (2**90 + 1), "integer")])
@pytest.mark.parametrize("filled", [False, True])
def test_winding_explain_and_execution_agree(values, expected, arithmetic, filled):
    rex = RexGraph.from_simplicial([0, 1, 2], [1, 2, 0], [[0, 1, 2]] if filled else [])
    field = Cochain(1, np.asarray(values), source=rex)
    executor = Executor(sources={"r": rex}, params={"f": field})
    result = executor.execute(parse('FROM $r RETURN WINDING($f)'))
    explain = executor.execute(parse('EXPLAIN FROM $r RETURN WINDING($f)'))
    assert result.values[0].tolist() == ([] if filled else [expected])
    assert result.exactness[0].value == arithmetic
    assert explain.values[0]["returns"][0]["result"]["exactness"] == arithmetic
