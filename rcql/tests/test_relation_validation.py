"""Declared candidate columns use Core validation and the bound source."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.relation_validation import validate_relations
from rcql import BoundSource, Executor, SourcePolicy, parse


def fixture():
    return RexGraph.from_cells([4, [[0, 1, 2, 3], [0, 1], [0, 2], [0, 3]]])


def proposals():
    return [[(0, 3), (1, -1), (2, -1), (3, -1)], [(0, Q(1, 2))], []]


def test_core_parity_members_and_trace():
    rex = fixture()
    result = Executor(sources={"r": rex}, params={"p": proposals()}).execute(parse(
        'FROM $r LET v = VALIDATE_RELATIONS(candidates=$p, grade=2) RETURN v, v.valid, v.accepted, v.residuals, v.grade'))
    expected = validate_relations(rex, proposals())
    assert result.values == (expected, expected["valid"], (0,), expected["residuals"], 2)
    assert "core-exact-candidate-boundaries" in str(result.execution)


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("args", ["$p, grade=1", "$p, grade=3", "$p, grade=TRUE", "$p, grade=2.5",
                                 "[[(0, 1.0)]]", "$bad"])
def test_bad_contract_refused_before_adapter(args, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError, SyntaxError)):
        Executor(sources={"r": fixture()}, params={"p": proposals(), "bad": [[(99, 1)]]}).execute(
            replace(parse(f'FROM $r RETURN VALIDATE_RELATIONS({args})'), explain=explain))


def test_explain_defers_candidate_product(monkeypatch):
    import rexgraph.relation_validation as module
    monkeypatch.setattr(module, "_exact_compose_columns", lambda *a: pytest.fail("candidate product ran"))
    result = Executor(sources={"r": fixture()}, params={"p": proposals()}).execute(parse(
        'EXPLAIN FROM $r RETURN VALIDATE_RELATIONS($p)'))
    assert result.execution == ()


def test_unknown_member_and_read_policy():
    rex = fixture()
    engine = Executor(sources={"r": BoundSource(rex, SourcePolicy.allow("read"))}, params={"p": proposals()})
    assert engine.execute(parse('FROM $r RETURN VALIDATE_RELATIONS($p).accepted')).values == ((0,),)
    with pytest.raises(TypeError, match="declared member"):
        engine.execute(parse('FROM $r RETURN VALIDATE_RELATIONS($p).attach'))
    denied = Executor(sources={"r": BoundSource(rex, SourcePolicy.allow())}, params={"p": proposals()})
    with pytest.raises(PermissionError):
        denied.execute(parse('FROM $r RETURN VALIDATE_RELATIONS($p)'))


def test_rcdb_reopen_validation_never_publishes(tmp_path):
    import rcdb
    path = f"rex://{tmp_path / 'db'}"
    rex = fixture()
    store = rcdb.open_store(path)
    store.put("r", rex)
    store.close()
    store = rcdb.open_store(path)
    try:
        engine = Executor(sources={"db": store}, params={"p": proposals()})
        query = parse('FROM RCDB_GET($db,"r") RETURN VALIDATE_RELATIONS($p), STATE_HASH()')
        result = engine.execute(query)
        assert result.values == (validate_relations(rex, proposals()), object_digest(rex))
        assert engine.execute(query).values == result.values
        assert store.get("r").nF == 0
    finally:
        store.close()
