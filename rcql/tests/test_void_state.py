"""VOID is a declared native region reading, not an implicit face attachment."""
from dataclasses import replace

import pytest

from rexgraph.cells import CellSet
from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rcql import BoundSource, Executor, SourcePolicy, parse


def fixture():
    return RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]]])


def test_native_members_and_no_attachment():
    rex = fixture()
    before = object_digest(rex)
    result = Executor(sources={"r": rex}).execute(parse(
        'FROM $r LET v=VOID(CELLS(1)) RETURN v.n_voids, v.strain, v.homology.independent_fillings, v.columns'))
    assert result.values[:3] == (1, 3, 1)
    assert "core-exact-triangle-void" in str(result.execution)
    assert len(result.values[3]) == 1 and object_digest(rex) == before


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("mode", ["foreign", "wrong-grade", "list"])
def test_bad_basis_refused_before_adapter(mode, explain, monkeypatch):
    import rcql.executor
    rex = fixture()
    value = {"foreign": CellSet(fixture(), 1, [0]), "wrong-grade": CellSet(rex, 0, [0]), "list": [0, 1]}[mode]
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": rex}, params={"s": value}).execute(replace(parse('FROM $r RETURN VOID($s)'), explain=explain))


def test_explain_never_enumerates_or_reduces(monkeypatch):
    import rexgraph.void_state as module
    monkeypatch.setattr(module, "void_state", lambda *a: pytest.fail("void executed"))
    assert Executor(sources={"r": fixture()}).execute(parse(
        'EXPLAIN FROM $r RETURN VOID(CELLS(1)).homology')).execution == ()


def test_read_policy_and_unknown_members():
    rex = fixture()
    engine = Executor(sources={"r": BoundSource(rex, SourcePolicy.allow("read"))})
    assert engine.execute(parse('FROM $r RETURN VOID(CELLS(1)).n_voids')).values == (1,)
    with pytest.raises(TypeError, match="declared member"):
        engine.execute(parse('FROM $r RETURN VOID(CELLS(1)).source'))
    denied = Executor(sources={"r": BoundSource(rex, SourcePolicy.allow())})
    with pytest.raises(PermissionError):
        denied.execute(parse('FROM $r RETURN VOID(CELLS(1))'))


def test_reopened_rcdb_void_reading(tmp_path):
    import rcdb
    path = f"rex://{tmp_path / 'db'}"
    store = rcdb.open_store(path)
    rex = fixture()
    store.put("r", rex)
    store.close()
    store = rcdb.open_store(path)
    try:
        result = Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_GET($db,"r") RETURN VOID(CELLS(1)).n_voids, STATE_HASH()'))
        assert result.values == (1, object_digest(rex))
        assert store.get("r").nF == 0
    finally:
        store.close()
