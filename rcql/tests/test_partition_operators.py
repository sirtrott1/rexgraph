"""Typed partitions preserve source contracts, exact closure and stored state."""
from dataclasses import replace

import pytest

from rexgraph.cells import Cell, CellSet
from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.io.partition_state import partition_tower
from rcql import BoundSource, Executor, SourcePolicy, call, parse, query, source


def fixture():
    face = [(0, 1), (1, 1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    return RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]],
        [face, face], [difference, difference], [difference]], relation_ids=[91, 17, 53])


def test_text_names_members_bound_policy_and_rebinding():
    rex = fixture()
    policy = SourcePolicy.allow("read", "identity")
    engine = Executor(sources={"r": BoundSource(rex, policy)})
    q = parse('FROM $r LET p=PARTITION(selection=CELL(4,0),closure="subcomplex") '
              'RETURN p, p.rex, p.manifest, p.digest, p.cell_maps, COUNT(FACES(support=CELLS(1)))')
    result = engine.execute(q)
    p, child, manifest, digest, maps, count = result.values
    assert child is p.rex and child is not rex
    assert manifest == p.state.manifest() and digest == p.state.digest
    assert manifest["source_state"] == object_digest(rex)
    assert manifest["result_state"] == object_digest(child)
    assert manifest["policy_digest"] == policy.digest
    assert maps == ((0, 1, 2), (0, 1, 2), (0, 1), (0, 1), (0,)) and count == 2
    assert child.relation_ids.tolist() == [91, 17, 53]
    rebound = Executor(sources={"c": child}).execute(parse('FROM $c RETURN GRADE(), BETTI(4), STATE_HASH()'))
    assert rebound.values == (4, 0, manifest["result_state"])
    assert all(e.value in {"structural", "integer"} for e in result.exactness)
    explained = engine.execute(parse('EXPLAIN FROM $r RETURN PARTITION(CELL(4,0))')).values[0]
    assert explained["returns"][0]["result"]["kind"] == "RexPartition"
    assert engine.execute(parse('FROM $r RETURN RESTRICT(CELL(4,0))')).values[0].nF == 2


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("name", ["RESTRICT", "PARTITION", "QUOTIENT"])
def test_identity_required_before_reading_or_constructing(name, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    rex = fixture()
    executor = Executor(sources={"r": BoundSource(rex, SourcePolicy.allow("read"))})
    with pytest.raises(PermissionError):
        executor.execute(replace(query(source("r"), call(name, call("CELL", 1, 0))), explain=explain))


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("name", ["FACES", "RESTRICT", "PARTITION", "QUOTIENT"])
def test_foreign_selections_rejected_before_adapters(name, explain, monkeypatch):
    import rcql.executor
    a, b = fixture(), fixture()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((ValueError, TypeError)):
        Executor(sources={"r": a}).execute(replace(
            query(source("r"), call(name, Cell(b, 1, 0))), explain=explain))


@pytest.mark.parametrize("text", ['RESTRICT(CELL(1,0),"projection")',
    'PARTITION(CELLS(1),"cluster")', 'FACES(CELLS(0))', 'PARTITION([0,1])',
    'PARTITION(CELL(4,0)).state', 'PARTITION(CELL(4,0)).source'])
@pytest.mark.parametrize("explain", [False, True])
def test_unsupported_policies_inputs_and_members_are_refused(text, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((ValueError, TypeError)):
        Executor(sources={"r": fixture()}).execute(replace(parse(f'FROM $r RETURN {text}'), explain=explain))


def test_explain_does_not_construct_a_partition(monkeypatch):
    import rexgraph.io.partition_state as core
    import rcql.executor
    monkeypatch.setattr(core, "build_rex_partition", lambda *a, **kw: pytest.fail("partition built"))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    for name in ("FACES", "RESTRICT", "PARTITION"):
        result = Executor(sources={"r": fixture()}).execute(parse(f'EXPLAIN FROM $r RETURN {name}(CELLS(1))'))
        assert result.execution == ()


def test_saved_partition_members_refuse_changed_result():
    rex = fixture()
    p = Executor(sources={"r": rex}).execute(parse('FROM $r RETURN PARTITION(CELL(1,0))')).values[0]
    executor = Executor(sources={"r": rex}, params={"p": p})
    assert executor.execute(parse('FROM $r RETURN $p.cell_maps')).values[0] == ((0, 1), (0,), (), (), ())
    p.rex.add_edges([0], [1], relation_ids=[101])
    for prefix in ("", "EXPLAIN "):
        with pytest.raises(ValueError, match="changed"):
            executor.execute(parse(prefix+'FROM $r RETURN $p.manifest'))


def test_isolated_vertex_and_empty_selection_are_owned_results():
    rex = fixture()
    executor = Executor(sources={"r": rex}, params={"empty": CellSet(rex, 1, ())})
    a, b = executor.execute(parse('FROM $r RETURN RESTRICT(CELL(0,3)), RESTRICT($empty)')).values
    assert (a.nV, a.nE, b.nV, b.nE) == (1, 0, 0, 0)
    assert a.relation_ids.tolist() == b.relation_ids.tolist() == []


def test_rcdb_partition_roundtrip_and_rcql_commit_leave_source_unchanged(tmp_path):
    import rcdb
    path = f"rex://{tmp_path / 'db'}"
    store = rcdb.open_store(path).configure_security(require_commits=True)
    try:
        rex = fixture()
        engine = Executor(sources={"db": store}, params={"r": rex})
        engine.execute(parse('FROM $db MUTATE "r" SET state=$r, actor="Art" COMMIT'))
        before = store.read_record("r")
        p = engine.execute(parse('FROM RCDB_GET($db,"r") RETURN PARTITION(CELL(4,0))')).values[0]
        assert p.state.source_state == before.state_digest
        engine.params["child"] = p.rex
        engine.execute(parse('FROM $db MUTATE "child" SET state=$child, actor="Art" COMMIT'))
        assert store.read_record("r").record.version == 1
        assert store.read_record("r").state_digest == before.state_digest
        assert store.verify_commits("child")
        digest = p.state.result_state
    finally:
        store.close()
    store = rcdb.open_store(path)
    try:
        engine = Executor(sources={"db": store})
        out = engine.execute(parse('FROM RCDB_GET($db,"child") RETURN STATE_HASH(), BETTI(4), '
                                   'COUNT(FACES(CELLS(1))), PARTITION(CELL(3,0))'))
        assert out.values[:3] == (digest, 0, 2)
        child = store.get("child")
        assert child.relation_ids.tolist() == [91, 17, 53]
        assert len(partition_tower(child)[0]) == 4
    finally:
        store.close()
