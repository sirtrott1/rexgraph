"""RCQL exposes Core C1 alignment on the bound timeline without new transport math."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.lineage_alignment import align_by_lineage
from rcql import BoundSource, Executor, SourcePolicy, parse


def fixture():
    source = TemporalRex([])
    source.append_snapshot(RexGraph.from_cells([3, [[0, 1, 2], [2]]], relation_ids=[11, 4]), at=2)
    source.append_snapshot(RexGraph.from_cells([3, [[2, 0, 1]]], relation_ids=[11]), at=4)
    return source


def test_parity_members_and_trace():
    source = fixture()
    values = [[Q(2, 3), 0], [7]]
    result = Executor(sources={"t": source}, params={"v": values}).execute(parse(
        'FROM $t LET a = ALIGN_BY_LINEAGE(values=$v) RETURN a, a.presence, a.cell_maps, a.coefficient_domain'))
    expected = align_by_lineage(source, values)
    assert result.values == (expected, expected["presence"], expected["cell_maps"], "Q")
    assert "core-sparse-c1-lineage-alignment" in str(result.execution)


def test_interval_and_unknown_member():
    engine = Executor(sources={"t": fixture()}, params={"v": [[0]]})
    assert engine.execute(parse('FROM $t RETURN ALIGN_BY_LINEAGE($v, start=1, stop=2).steps')).values == ((1,),)
    with pytest.raises(TypeError, match="declared member"):
        engine.execute(parse('FROM $t RETURN ALIGN_BY_LINEAGE($v,1,2).transport'))


@pytest.mark.parametrize("explain", [True, False])
@pytest.mark.parametrize("values,start,stop", [([], 0, None), ([[True], [1]], 0, None),
    ([[1], [1]], True, None), ([[1], [1]], 0, 3), ([[[1]], [1]], 0, None)])
def test_bad_inputs_refused_before_adapter(values, start, stop, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    engine = Executor(sources={"t": fixture()}, params={"v": values, "s": start, "e": stop})
    with pytest.raises((TypeError, ValueError)):
        engine.execute(replace(parse('FROM $t RETURN ALIGN_BY_LINEAGE($v,$s,$e)'), explain=explain))


def test_explain_defers_snapshots_and_axis_ambiguity(monkeypatch):
    source = fixture()
    monkeypatch.setattr(TemporalRex, "reconstruct_at", lambda *a: pytest.fail("reconstructed"))
    result = Executor(sources={"t": source}, params={"v": [[1, 2], [3]]}).execute(parse(
        'EXPLAIN FROM $t RETURN ALIGN_BY_LINEAGE($v)'))
    assert result.execution == ()


def test_read_policy_and_source_kind():
    source = fixture()
    values = [[1, 2], [3]]
    engine = Executor(sources={"t": BoundSource(source, SourcePolicy.allow("read"))}, params={"v": values})
    assert engine.execute(parse('FROM $t RETURN ALIGN_BY_LINEAGE($v).shape')).values == ((2, 2),)
    denied = Executor(sources={"t": BoundSource(source, SourcePolicy.allow())}, params={"v": values})
    with pytest.raises(PermissionError):
        denied.execute(parse('FROM $t RETURN ALIGN_BY_LINEAGE($v)'))
    with pytest.raises(TypeError):
        Executor(sources={"t": source.reconstruct_at(0)}, params={"v": values}).execute(parse(
            'FROM $t RETURN ALIGN_BY_LINEAGE($v)'))


def test_reopened_rcdb_timeline_alignment(tmp_path):
    import rcdb
    from rexgraph.io.catalog import object_digest
    source = fixture()
    path = f"rex://{tmp_path / 'db'}"
    store = rcdb.open_store(path)
    store.put("t", source)
    store.close()
    store = rcdb.open_store(path)
    try:
        before = object_digest(store.get("t"))
        values = [[Q(3, 7), 0], [2**80+1]]
        out = Executor(sources={"db": store}, params={"v": values}).execute(parse(
            'FROM RCDB_GET($db,"t") RETURN ALIGN_BY_LINEAGE($v)'))
        assert out.values == (align_by_lineage(source, values),)
        assert object_digest(store.get("t")) == before
    finally:
        store.close()
