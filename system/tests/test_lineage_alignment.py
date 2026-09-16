"""Sparse aligned values and presence retain their meaning through System JSON."""
from fractions import Fraction
import json

from rexgraph.graph import RexGraph, TemporalRex
from rcql import Executor, parse
from system.serialize import json_value


def test_alignment_json_distinguishes_zero_and_missing():
    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_cells([2, [[0], [1]]], relation_ids=[3, 4]))
    timeline.append_snapshot(RexGraph.from_cells([2, [[1]]], relation_ids=[4]))
    out = Executor(sources={"t": timeline}, params={"v": [[0, Fraction(1, 3)], [0]]}).execute(parse(
        'FROM $t RETURN ALIGN_BY_LINEAGE($v)'))
    result = json.loads(json.dumps(json_value(out.values[0]), allow_nan=False))
    assert result["keys"] == [3, 4]
    assert result["entries"] == [[0, 1, {"numerator": 1, "denominator": 3}]]
    assert result["presence"] == [[0, 0], [0, 1], [1, 1]]
    assert result["missing"] == "outside-presence"
