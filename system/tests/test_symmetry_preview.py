"""A group preview does not materialize products or traverse its source graph."""
from rexgraph import RexGraph
from rexgraph.chain_map import CoordinateComplex, GradedMap, SymmetryGroup
from system.serialize import json_value


def test_bounded_lazy_preview(monkeypatch):
    r = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    c = CoordinateComplex.from_rex(r)
    p = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))
    g = SymmetryGroup([p], [1, 1, 1])
    monkeypatch.setattr(GradedMap, "then", lambda *a: (_ for _ in ()).throw(AssertionError("product built")))
    out = json_value(g, max_values=1)
    assert out["word"] == [1] and out["sizes"] == [3]
    assert out["truncated"] and out["word_length"] == 3 and out["map"] == "lazy"
    assert "generators" not in out and "source" not in out
