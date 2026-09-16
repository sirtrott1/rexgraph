"""Scalar homology readings certify the declared quotient before execution."""
from dataclasses import replace

import pytest
from rexgraph.graph import RexGraph

from rcql import Executor, call, parse, query, source
from rcql.operators import get_operator


NAMES = ("SIMPLE_HOMOLOGY", "MULTIPLICITY_HOMOLOGY")


@pytest.mark.parametrize("filled", [False, True])
def test_text_direct_and_stored_readings(filled, tmp_path):
    import rcdb
    rex = RexGraph.from_cells([2, [[0, 1], [1, 0]], [[(0, 1), (1, 1)]] if filled else []])
    expected = (0, int(not filled))
    assert tuple(get_operator(name).fn(rex, 1) for name in NAMES) == expected
    q = parse("FROM $r RETURN SIMPLE_HOMOLOGY(grade=1), MULTIPLICITY_HOMOLOGY(grade=1)")
    result = Executor(sources={"r": rex}).execute(q)
    assert result.values == expected
    assert all(e.value == "integer" for e in result.exactness)
    store = rcdb.open_store(f"rex://{tmp_path / 'store'}")
    try:
        store.put("r", rex, analytics=False)
        before = store.read_record("r").record.version
        result = Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_GET($db,"r") RETURN SIMPLE_HOMOLOGY(1), MULTIPLICITY_HOMOLOGY(1)'))
        assert result.values == expected
        assert store.read_record("r").record.version == before
    finally:
        store.close()


@pytest.mark.parametrize("name", NAMES)
def test_explain_is_exact_and_does_not_run_rank(name, monkeypatch):
    import rcql.executor
    import rexgraph.native_homology
    rex = RexGraph.from_cells([2, [[0, 1], [0, 1]]])
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("EXPLAIN ran adapter"))
    monkeypatch.setattr(rexgraph.native_homology, "_rank_integer_columns", lambda *a: pytest.fail("EXPLAIN ran rank"))
    q = replace(query(source("r"), call(name, 1)), explain=True)
    plan = Executor(sources={"r": rex}).execute(q).values[0]["returns"][0]
    assert plan["result"]["exactness"] == "integer"
    assert any(p["name"] == "chain_condition" and p["status"] == "verified" for p in plan["predicates"])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("grade", [-1, 7, True, 1.0, "1"])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_grade_is_rejected_before_any_adapter(name, grade, explain, monkeypatch):
    import rcql.executor
    rex = RexGraph.from_graph([0], [1])
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("invalid input ran adapter"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": rex}).execute(replace(query(source("r"), call(name, grade)), explain=explain))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("explain", [False, True])
def test_invalid_higher_chain_rejected_before_execution(name, explain, monkeypatch):
    import rcql.executor
    from rexgraph.native_sparse import native_coo
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [[(0, 1), (1, 1), (2, -1)]]])
    rex._graded_duals = [native_coo([0], [0], [1], (1, 1)).dual]
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("invalid chain ran adapter"))
    with pytest.raises(ValueError, match="chain condition"):
        Executor(sources={"r": rex}).execute(replace(query(source("r"), call(name, 1)), explain=explain))


def test_full_graded_names_read_the_requested_quotient():
    face = [(0, 1), (1, 1), (2, -1)]
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [face, face]])
    result = Executor(sources={"r": rex}).execute(parse(
        "FROM $r RETURN SIMPLE_HOMOLOGY(0), MULTIPLICITY_HOMOLOGY(0), "
        "SIMPLE_HOMOLOGY(2), MULTIPLICITY_HOMOLOGY(2)"))
    assert result.values == (1, 0, 0, 1)
