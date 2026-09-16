"""Structural phrases preserve exact primary coordinates and all carried grades."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.cells import Cell
from rexgraph.cochain import Chain, Cochain
from rexgraph.column_expansion import ColumnExpansion
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import boundary_operator

from rcql import Executor, call, parse, query, source


def run(rex, name, *args, explain=False):
    return Executor(sources={"r": rex}).execute(replace(query(source("r"), call(name, *args)), explain=explain)).values[0]


def fixture():
    return RexGraph.from_cells([4, [[0, 1, 2, 3], [0, 0], [3]]])


def test_text_expansion_members_and_reconstruction_apply_exactly():
    rex = fixture()
    x = Chain(1, np.array([Q(3), 2**100, Q(1, 7)], object), source=rex)
    q = parse("FROM $r LET e = COLUMN_EXPANSION(BOUNDARY(1)) "
              "LET b = PRIMARY_LIFT(e.legs, e.lift) "
              "RETURN APPLY(b, $x, exact=TRUE), e.groups, e.references, e.legs.shape, e.lift.entries")
    result = Executor(sources={"r": rex}, params={"x": x}).execute(q)
    assert result.values[0].values.tolist() == [-3, 1, 1, Q(8, 7)]
    assert result.values[0].grade == 0 and result.values[0].source is rex
    assert result.values[1:4] == (((0, 3), (3, 3), (3, 4)), (0, 0, 3), (4, 4))
    assert result.values[4][-1] == (3, 2, Q(1))
    assert result.exactness[0].value == "rational"
    facts = Executor(sources={"r": rex}, params={"x": x}).execute(replace(q, explain=True)).values[0]
    assert facts["returns"][0]["result"]["grade"] == 0
    assert facts["returns"][0]["result"]["exactness"] == "rational"


def test_literal_actions_keep_descriptor_and_match_nested_actions():
    rex = fixture()
    e = run(rex, "COLUMN_EXPANSION", call("BOUNDARY", 1))
    b = run(rex, "PRIMARY_LIFT", e.legs, e.lift)
    x = Chain(1, np.array([Q(1, 3), Q(1), Q(1, 7)], object), source=rex)
    assert run(rex, "APPLY", b, x, True).values.tolist() == [Q(-1, 3), Q(1, 9), Q(1, 9), Q(16, 63)]
    np.testing.assert_allclose(run(rex, "APPLY", b, x).values, np.asarray(b.apply(x.values, exact=True), float))
    plan = run(rex, "PRIMARY_LIFT", e.legs, e.lift, explain=True)["returns"][0]
    assert any(p["name"] == "identical_expansion" and p["status"] == "verified" for p in plan["predicates"])


@pytest.mark.parametrize("explain", [False, True])
def test_source_variance_and_factor_mismatch_are_refused(explain, monkeypatch):
    import rcql.executor
    rex = fixture()
    a, b = (ColumnExpansion(boundary_operator(rex, 1)) for _ in range(2))
    foreign = ColumnExpansion(boundary_operator(fixture(), 1))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("invalid input reached adapter"))
    for parts in ((a.legs, b.lift), (a.legs, foreign.lift), (a.lift, a.legs)):
        with pytest.raises((TypeError, ValueError)):
            run(rex, "PRIMARY_LIFT", *parts, explain=explain)
    with pytest.raises(TypeError):
        run(rex, "COLUMN_EXPANSION", call("HODGE_OPERATOR", 1), explain=explain)
    with pytest.raises(TypeError):
        run(rex, "APPLY", call("PRIMARY_LIFT", a.legs, a.lift),
            Cochain(1, np.ones(3), source=rex), explain=explain)


def test_explain_does_not_construct_factors_or_evaluate_operators(monkeypatch):
    import rcql.executor
    from rexgraph.linear_operator import RexOperator
    rex = fixture()
    b = boundary_operator(rex, 1)
    monkeypatch.setattr(ColumnExpansion, "__post_init__", lambda *a: pytest.fail("constructed factors"))
    monkeypatch.setattr(RexOperator, "apply", lambda *a, **k: pytest.fail("applied boundary"))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("EXPLAIN ran adapter"))
    for value in (b, call("BOUNDARY", 1)):
        assert run(rex, "COLUMN_EXPANSION", value, explain=True)["returns"][0]["result"]["kind"] == "ColumnExpansion"


@pytest.mark.parametrize("member", ["below", "above", "lateral", "cell"])
def test_hyperslice_text_members_remain_cells_in_their_own_grade(member):
    rex = fixture()
    result = Executor(sources={"r": rex}).execute(parse(
        f"FROM $r LET h = HYPERSLICE(CELL(1, 0)) RETURN h.{member}"))
    value = result.values[0]
    assert value.source is rex
    assert value.grade == {"below": 0, "above": 2, "lateral": 1, "cell": 1}[member]
    q = parse(f"FROM $r LET h = HYPERSLICE(CELL(1, 0)) RETURN COUNT(h.{member})")
    if member != "cell":
        assert Executor(sources={"r": rex}).execute(q).values[0] == {"below": 4, "above": 0, "lateral": 2}[member]


def test_bottom_hyperslice_has_none_below_and_forbids_unlisted_members():
    rex = fixture()
    assert Executor(sources={"r": rex}).execute(parse(
        "FROM $r LET h = HYPERSLICE(CELL(0, 0)) RETURN h.below")).values == (None,)
    assert run(rex, "HYPERSLICE", Cell(rex, 1, 1)).below.indices == (0,)
    for expression in ("HYPERSLICE(CELL(0, 0)).source", "COLUMN_EXPANSION(BOUNDARY(1)).boundary"):
        with pytest.raises(TypeError):
            Executor(sources={"r": rex}).execute(parse(f"FROM $r RETURN {expression}"))


def test_structural_reading_from_rcdb_changes_no_version(tmp_path):
    import rcdb
    store = rcdb.open_store(f"rex://{tmp_path / 'db'}")
    try:
        store.put("r", fixture())
        before = store.read_record("r").record.version
        q = parse('FROM RCDB_GET($db, "r") LET e = COLUMN_EXPANSION(BOUNDARY(1)) '
                  'RETURN PRIMARY_LIFT(e.legs, e.lift), HYPERSLICE(CELL(1, 0)).below')
        result = Executor(sources={"db": store}).execute(q)
        assert result.values[0].shape == (4, 3)
        assert result.values[1].indices == (0, 1, 2, 3)
        assert store.read_record("r").record.version == before
    finally:
        store.close()
