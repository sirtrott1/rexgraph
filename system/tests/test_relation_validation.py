"""Candidate readings retain exact rational residuals through System JSON."""
import json

from rexgraph.graph import RexGraph
from rcql import Executor, parse
from system.serialize import json_value


def test_candidate_result_json():
    rex = RexGraph.from_cells([4, [[0, 1, 2, 3]]])
    out = Executor(sources={"r": rex}, params={"p": [[(0, 1)], []]}).execute(parse(
        'FROM $r RETURN VALIDATE_RELATIONS($p)'))
    value = json.loads(json.dumps(json_value(out.values[0]), allow_nan=False))
    assert value["valid"] == [False, False]
    assert value["closed"] == [False, True]
    assert value["residuals"][0] == [[0, {"numerator": -1, "denominator": 1}],
        *[[i, {"numerator": 1, "denominator": 3}] for i in range(1, 4)]]
    assert "python_type" not in str(value)
