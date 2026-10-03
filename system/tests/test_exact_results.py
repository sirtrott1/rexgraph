"""Exact result previews remain exact at the browser JSON boundary."""
from decimal import Decimal
from fractions import Fraction as Q
import json

import numpy as np
import pytest
from rexgraph.exact_array import ExactArray
from rexgraph.value import Absent, Approx, ExactTime, TimeRange
from system.serialize import json_value


@pytest.mark.parametrize("value", [2**53, -(2**100), 2**16000], ids=["js-boundary", "negative", "bigint-escape"])
def test_large_integers_use_exact_decimal_text(value):
    result = json_value(value)
    assert result == {"kind": "Integer", "decimal": str(Decimal(value))}
    assert int(Decimal(result["decimal"])) == value
    json.dumps(result, allow_nan=False)


def test_rationals_approximation_and_absence_are_distinct():
    result = json_value([Q(2**100, 3), Approx(.1, "sensor"), Absent, None, 0])
    assert result[0] == {"numerator": json_value(2**100), "denominator": 3}
    assert result[1] == {"kind": "Approx", "value": .1, "source": "sensor"}
    assert result[2:] == [{"kind": "Absent"}, None, 0]


def test_exact_tensor_preview_does_not_materialize_the_full_tensor(monkeypatch):
    value = ExactArray.from_values(np.arange(100_000))
    original = ExactArray.values
    def bounded(self):
        assert self.size <= 3
        return original(self)
    monkeypatch.setattr(ExactArray, "values", bounded)
    assert json_value(value, max_values=3) == {
        "kind": "ExactArray", "shape": [100_000], "sample": [0, 1, 2], "truncated": True}


def test_exact_tensor_presence_kind_and_shape_are_visible():
    value = ExactArray.from_values([[Q(1), 1], [Absent, Q(1, 3)]])
    result = json_value(value)
    assert result["values"] == [[{"numerator": 1, "denominator": 1}, 1],
                                [{"kind": "Absent"}, {"numerator": 1, "denominator": 3}]]
    assert result["shape"] == [2, 2] and not result["truncated"]
    assert json_value(value, max_values=0)["sample"] == []


def test_exact_time_range_uses_rational_seconds():
    value = TimeRange(ExactTime(Q(1, 3)))
    assert json_value(value) == {"kind": "TimeRange",
        "start": {"kind": "ExactTime", "seconds": {"numerator": 1, "denominator": 3}},
        "end": {"kind": "Absent"}}


def test_extended_binary_scalar_does_not_recurse_or_round():
    value = np.longdouble(1) + np.longdouble(2)**-60
    if not isinstance(value.item(), np.generic):
        pytest.skip("this platform's long double uses a Python float carrier")
    result = json_value(value)
    assert result == {"kind": "BinaryFloat", "dtype": str(value.dtype),
                      "exact": json_value(Q(*value.as_integer_ratio()))}
    json.dumps(result, allow_nan=False)


def test_selection_lineage_and_partition_previews_bound_original_cell_addresses():
    from rexgraph import RexGraph, Selection, restrict
    source = RexGraph.from_graph(np.arange(1000), np.arange(1, 1001))
    selected = Selection(source, {1: range(1000)})
    part = restrict(source, selected)
    preview = json_value(selected, max_values=3)
    assert preview["indices"][1] == {"grade": 1, "count": 1000,
        "indices": [0, 1, 2], "truncated": True}
    lineage = json_value(part.lineage, max_values=3)
    assert lineage["cell_maps"][0]["count"] == 1001
    assert lineage["cell_maps"][0]["indices"] == [0, 1, 2]
    assert lineage["requested"][1] == preview["indices"][1]
    output = json_value(part, max_values=0)
    assert output["lineage"]["digest"] == part.lineage.digest
    assert output["cell_maps"][1]["indices"] == [] and output["cell_maps"][1]["truncated"]
    json.dumps(output, allow_nan=False)
    part.rex.attach_metadata(1, 0, "changed", True)
    with pytest.raises(ValueError, match="changed"):
        json_value(part)
