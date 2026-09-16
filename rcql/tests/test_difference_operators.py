"""Core ownership, static refusals and stored endpoints for difference names."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.cochain import Chain, Cochain
from rexgraph.field_delta import field_delta
from rexgraph.graph import RexGraph
from rexgraph.tensor_diff import difference_tensor
from rcql import BoundSource, Executor, SourcePolicy, call, parse, query, source


def fixture():
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [[(0, 1), (1, 1), (2, -1)]]])
    c = CoordinateComplex.from_rex(rex)
    mapping = GradedMap(c, c, tuple(tuple((i, i, Q(k+1, 3)) for i in range(n)) for k, n in enumerate(c.sizes)))
    return rex, Chain(1, np.array([1, 2, 3]), source=rex), mapping


def test_delta_fields_and_moments_match_core_and_keep_target_coordinates():
    rex, x, j = fixture()
    engine = Executor(sources={"r": rex}, params={"x": x, "j": j})
    out = engine.execute(parse('FROM $r LET d=FIELD_DELTA(field=$x,correspondence=$j) '
        'RETURN d,d.down,d.up,d.moment,d.oriented_moment,FIELD_DELTA_MOMENT($x,$j),ORIENTED_FIELD_DELTA_MOMENT($x,$j)'))
    expected = field_delta(x, j)
    assert out.values[0] == expected and out.values[1:3] == (expected["down"], expected["up"])
    assert out.values[3:] == (expected["moment"], expected["oriented_moment"])*2
    assert out.exactness[-1].value == out.exactness[-2].value == "rational"


def test_diff_members_literal_capture_and_no_source_basis_invention():
    rex, _, _ = fixture()
    other = RexGraph.from_cells([4, [[0], [3, 0]]])
    d, expected = difference_tensor(rex, other)
    engine = Executor(sources={"r": rex}, params={"other": other, "d": d})
    out = engine.execute(parse('FROM $r LET d=DIFF(other=$other) '
        'RETURN d,d.entries,d.vertex_keys,d.relation_pairs,d.readings,$d.readings')).values
    assert out[1:4] == (d.entries, d.vertex_keys, d.relation_pairs)
    assert out[4] == out[5] == expected
    for expr in ('DIFF($other).reference', 'DIFF($other).other', 'DIFF($other).apply', 'APPLY(DIFF($other),$d)'):
        with pytest.raises((TypeError, ValueError)):
            engine.execute(parse(f'FROM $r RETURN {expr}'))


def test_delta_coordinate_members_compose_without_relabelling_the_basis():
    rex, x, j = fixture()
    engine = Executor(sources={"r": rex}, params={"x": x, "j": j})
    out = engine.execute(parse('FROM $r LET d=FIELD_DELTA($x,$j) '
        'RETURN d.grade,d.metrics,d.down.grade,d.down.name,COUNT(d.down.values),d.up.shape'))
    assert out.values == (1, "identity", 0, "C0", 3, (1,))
    for member in ("unknown", "down.unknown"):
        with pytest.raises(TypeError, match="no declared member"):
            engine.execute(parse(f'EXPLAIN FROM $r RETURN FIELD_DELTA($x,$j).{member}'))


@pytest.mark.parametrize("name", ["FIELD_DELTA", "FIELD_DELTA_MOMENT", "ORIENTED_FIELD_DELTA_MOMENT"])
@pytest.mark.parametrize("kind", ["float", "cochain", "foreign", "foreign-map", "stale-map", "raw-array"])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_field_and_correspondence_refused_before_adapter(name, kind, explain, monkeypatch):
    import rcql.executor
    rex, x, j = fixture()
    if kind == "float":
        x = Chain(1, np.ones(3), source=rex)
    elif kind == "cochain":
        x = Cochain(1, x.values, source=rex)
    elif kind == "foreign":
        x = fixture()[1]
    elif kind == "foreign-map":
        j = fixture()[2]
    elif kind == "stale-map":
        rex.add_edges([0], [2])
    else:
        x = x.values
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": rex}, params={"x": x, "j": j}).execute(replace(
            parse(f'FROM $r RETURN {name}($x,$j)'), explain=explain))


def test_explain_does_not_evaluate_difference_or_defect_actions(monkeypatch):
    import rexgraph.boundary_difference as core
    import rexgraph.field_delta as delta
    rex, x, j = fixture()
    monkeypatch.setattr(core, "boundary_difference", lambda *a, **kw: pytest.fail("difference ran"))
    monkeypatch.setattr(delta, "field_delta", lambda *a, **kw: pytest.fail("defect ran"))
    result = Executor(sources={"r": rex}, params={"other": rex, "x": x, "j": j}).execute(parse(
        'EXPLAIN FROM $r RETURN DIFF($other),FIELD_DELTA($x,$j),FIELD_DELTA_MOMENT($x,$j)'))
    assert result.execution == ()


def test_diff_defers_rank_until_readings_are_requested(monkeypatch):
    import rexgraph.boundary_difference as core
    rex, _, _ = fixture()
    other = RexGraph.from_cells([3, [[0]]])
    calls = []
    rank = core._rank_integer_columns

    def counted(*args, **kwargs):
        calls.append(True)
        return rank(*args, **kwargs)

    monkeypatch.setattr(core, "_rank_integer_columns", counted)
    engine = Executor(sources={"r": rex}, params={"other": other})
    result = engine.execute(parse('FROM $r LET d=DIFF($other) RETURN d,d.entries')).values
    assert result[1] == result[0].entries and calls == []
    assert result[0].readings["rank"] > 0 and calls == [True]
    assert result[0].readings["rank"] > 0 and calls == [True]


@pytest.mark.parametrize("explain", [False, True])
def test_diff_identity_permission_and_bad_labels(explain):
    rex, _, _ = fixture()
    with pytest.raises(PermissionError):
        Executor(sources={"r": BoundSource(rex, SourcePolicy.allow("read"))}, params={"r": rex}).execute(
            replace(parse('FROM $r RETURN DIFF($r)'), explain=explain))
    with pytest.raises(ValueError, match="unique"):
        Executor(sources={"r": rex}).execute(replace(query(source("r"), call("DIFF", rex, ["x"]*3, ["a", "b", "c"])), explain=explain))


def test_stale_difference_is_refused():
    rex, _, _ = fixture()
    other = RexGraph.from_cells([3, [[0]]])
    d, _ = difference_tensor(rex, other)
    other.add_edges([0], [1])
    with pytest.raises(ValueError, match="changed"):
        Executor(sources={"r": rex}, params={"d": d}).execute(parse('FROM $r RETURN $d.entries'))


def test_reopened_rcdb_states_feed_native_diff_and_correspondence(tmp_path):
    import rcdb
    path = f"rex://{tmp_path / 'db'}"
    before = RexGraph.from_cells([2, [[0, 1]]], relation_ids=[17])
    after = RexGraph.from_cells([2, [[1, 0], [0]]], relation_ids=[17, 19])
    store = rcdb.open_store(path)
    for name, value in (("before", before), ("after", after)):
        Executor(sources={"db": store}, params={"x": value}).execute(parse(
            f'FROM $db MUTATE "{name}" SET state=$x, actor="Art" COMMIT'))
    store.close()
    store = rcdb.open_store(path)
    try:
        e = Executor(sources={"db": store}, params={"after": store.get("after")})
        d = e.execute(parse('FROM RCDB_GET($db,"before") RETURN DIFF($after)')).values[0]
        assert d.matching == "identity" and d.readings["max_column_sum"] == 1
        a, b = d.reference, d.other
        j = GradedMap(CoordinateComplex.from_rex(a), CoordinateComplex.from_rex(b),
                      (((0, 0, 1), (1, 1, 1)), ((0, 0, 1),)))
        out = Executor(sources={"r": a}, params={"x": Chain(1, np.array([3]), source=a), "j": j}).execute(
            parse('FROM $r RETURN FIELD_DELTA_MOMENT($x,$j)'))
        assert out.values == (Q(72),)
    finally:
        store.close()
