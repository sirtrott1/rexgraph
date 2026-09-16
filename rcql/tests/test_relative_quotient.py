"""RCQL relative coordinates, native Core dispatch and persistent sources."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from rcql import Executor, parse
from rexgraph.cells import Cell, CellSet, GradedCellPattern
from rexgraph.graph import RexGraph
from rexgraph.relative_quotient import relative_quotient


def fixture():
    return RexGraph.from_cells([4, [[0, 1, 2, 3], [0], [0, 1, 2, 3]]], relation_ids=[10, 11, 12])


def test_query_members_and_materialized_projection_use_core():
    source = fixture()
    engine = Executor(sources={"r": source})
    result = engine.execute(parse('FROM $r LET q=QUOTIENT(CELL(0,1)) '
        'RETURN q,q.sizes,q.boundaries,q.cell_maps,q.removed_cells,q.residuals,'
        'q.readings.betti,q.readings.ranks,q.readings.methods,q.coefficient_digest,q.source_boundary_digest,q.projection'))
    expected = relative_quotient(Cell(source, 0, 1))
    q = result.values[0]
    assert q.boundaries == expected.boundaries
    assert q.readings["betti"] == expected.readings["betti"]
    assert q.readings["ranks"] == expected.readings["ranks"]
    assert result.values[1:6] == (q.sizes, q.boundaries, q.cell_maps, q.removed_cells, q.residuals)
    assert result.values[6:11] == (q.readings["betti"], q.readings["ranks"], q.readings["methods"],
                                  q.coefficient_digest, q.source_boundary_digest)
    assert all(value == Q(0) for value in q.residuals)
    engine.params["p"] = result.values[-1]
    proof = engine.execute(parse('FROM $r RETURN CHAIN_MAP($p)')).values[0]
    assert proof.commutation_residuals == q.residuals


def test_explain_never_builds_quotient_or_computes_rank(monkeypatch):
    import rexgraph.relative_quotient as core
    import rexgraph.graded_boundary as graded
    def fail(*args, **kwargs):
        pytest.fail("quotient execution during EXPLAIN")
    monkeypatch.setattr(core, "relative_quotient", fail)
    monkeypatch.setattr(graded, "_rank_integer_columns", fail)
    result = Executor(sources={"r": fixture()}).execute(parse(
        'EXPLAIN FROM $r LET q=QUOTIENT(CELL(0,1)) RETURN q.readings.betti,q.projection'))
    assert result.execution == ()


def test_let_and_repeated_calls_reuse_one_core_construction(monkeypatch):
    import rexgraph.relative_quotient as core
    calls = []
    original = core.relative_quotient
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(core, "relative_quotient", counted)
    result = Executor(sources={"r": fixture()}).execute(parse(
        'FROM $r RETURN QUOTIENT(CELL(0,1)),QUOTIENT(CELL(0,1)).readings'))
    assert calls == [True] and result.values[0].readings == result.values[1]


@pytest.mark.parametrize("text", ['QUOTIENT(CELL(0,1),"projection")', 'QUOTIENT(CELL(8,0))',
    'QUOTIENT(CELL(0,1)).complex', 'QUOTIENT(CELL(0,1)).rex', 'QUOTIENT(CELL(0,1)).readings.nope',
    'CHAIN_MAP(QUOTIENT(CELL(0,1)).projection)'])
@pytest.mark.parametrize("explain", [False, True])
def test_undeclared_members_and_unresolved_target_axes_refuse(text, explain):
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": fixture()}).execute(replace(parse('FROM $r RETURN '+text), explain=explain))


@pytest.mark.parametrize("explain", [False, True])
def test_foreign_selection_is_not_reinterpreted(explain):
    engine = Executor(sources={"r": fixture()}, params={"s": Cell(fixture(), 0, 0)})
    with pytest.raises((TypeError, ValueError)):
        engine.execute(replace(parse('FROM $r RETURN QUOTIENT($s)'), explain=explain))


def test_rcdb_source_roundtrip_keeps_relative_shares(tmp_path):
    from rcdb import RexStore
    store = RexStore(str(tmp_path / "db"))
    try:
        source = fixture()
        store.put("r", source)
        restored = Executor(sources={"db": store}).execute(parse('FROM $db RETURN RCDB_GET("r")')).values[0]
        q = Executor(sources={"r": restored}).execute(parse('FROM $r RETURN QUOTIENT(CELL(0,1))')).values[0]
        assert q.boundaries == relative_quotient(Cell(source, 0, 1)).boundaries
        assert q.source_boundary_digest == relative_quotient(Cell(source, 0, 1)).source_boundary_digest
    finally:
        store.close()


def test_graded_patterns_compose_and_literal_patterns_retain_their_source():
    rex = RexGraph.from_cells([4, [[0, 1], [1, 2]]])
    pattern = GradedCellPattern(rex, (CellSet(rex, 0, (3,)), CellSet(rex, 1, (0,))))
    engine = Executor(sources={"r": rex}, params={"p": pattern})
    result = engine.execute(parse('FROM $r RETURN QUOTIENT($p).sizes,QUOTIENT(STAR(CELL(0,0))).sizes'))
    assert result.values == ((1, 1), (2, 1))
    foreign = RexGraph.from_cells([4, [[0, 1], [1, 2]]])
    engine.params["p"] = GradedCellPattern(foreign, (CellSet(foreign, 1, (0,)),))
    with pytest.raises((TypeError, ValueError)):
        engine.execute(parse('EXPLAIN FROM $r RETURN QUOTIENT($p)'))


def test_coordinate_quotient_is_not_a_canonical_rex_to_publish():
    from rcdb import MemoryStore
    store = MemoryStore()
    try:
        rex = fixture()
        candidate = relative_quotient(Cell(rex, 0, 1))
        executor = Executor(sources={"db": store}, params={"q": candidate})
        with pytest.raises((TypeError, ValueError)):
            executor.execute(parse('FROM $db MUTATE "q" SET state=$q COMMIT'))
        assert store.list() == []
    finally:
        store.close()
