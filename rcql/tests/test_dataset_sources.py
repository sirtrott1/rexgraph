"""Declared datasets carry exact Core state, policy and declaration identities."""
from contextlib import closing
from fractions import Fraction as Q
from io import StringIO

import pytest

from rcql import BoundSource, DatasetSource, Executor, SourcePolicy, classify, dataset, parse, query, call, ValueKind
from rcql.query_cache import QueryCache
from rexgraph import NumberRule, RelationSpec, ValueRules, VertexTable
from rexgraph.io.declaration import DatasetDeclaration
from rexgraph.io.records import RecordField, RecordSchema


def declaration(**options):
    schema = RecordSchema((RecordField("source"), RecordField("target"),
        RecordField("weight", "number", ValueRules(NumberRule.DECIMAL_EXACT))), **options)
    return DatasetDeclaration("csv", schema, RelationSpec("pair", ("source", "target"), weight="weight"),
                              vertices=VertexTable(("a", "b", "isolated")))


def test_declared_dataset_exact_values_isolates_and_native_result():
    source = DatasetSource(declaration(), StringIO("source,target,weight\na,b,0.1\n"))
    assert classify(source) is ValueKind.DATASET
    graph = source.materialize()
    assert graph.nV == 3 and graph.relations.weight.values().tolist() == [Q(1,10)]
    executor = Executor(sources={"data": source})
    result = executor.execute(parse('FROM DATASET("data") RETURN COUNT(CELLS(1)), BETTI(0), STATE_HASH()'))
    assert result.values[:2] == (1, 2)
    assert result.values[2] == source.as_record()["state_digest"]
    assert source.digest in result.provenance[0]["source_state"]["name"]
    assert type(result).from_bytes(result.to_bytes()).values == result.values
    assert executor.execute(query(dataset("data"), call("BETTI", 0))).values == (2,)


def test_binding_is_pinned_and_results_are_detached(tmp_path):
    path = tmp_path / "source.csv"; path.write_text("source,target,weight\na,b,0.1\n")
    source = DatasetSource(declaration(), path)
    path.write_text("source,target,weight\na,b,2\n")
    graph = source.materialize(); graph.attach_metadata(1, 0, "changed", True)
    assert source.materialize().get_metadata(1, 0, "changed") is None
    assert source.materialize().relations.weight.values().tolist() == [Q(1,10)]
    assert DatasetSource(declaration(), path).digest != source.digest


def test_read_denial_precedes_materialization(monkeypatch):
    source = DatasetSource(declaration(), StringIO("source,target,weight\na,b,1\n"))
    monkeypatch.setattr(DatasetSource, "materialize", lambda self: pytest.fail("denied source materialized"))
    with pytest.raises(PermissionError):
        Executor(sources={"data": BoundSource(source, SourcePolicy.allow("identity"))}).execute(
            parse('FROM DATASET("data") RETURN BETTI(0)'))


def test_declared_policy_survives_dataset_transform():
    source = DatasetSource(declaration(), StringIO("source,target,weight\na,b,1\n"))
    executor = Executor(sources={"data": BoundSource(source, SourcePolicy.allow("read"))})
    assert executor.execute(parse('FROM DATASET("data") RETURN BETTI(0)')).values == (2,)
    selected = executor._eval_source(dataset("data"))
    assert selected.policy == SourcePolicy.allow("read")
    with pytest.raises(PermissionError):
        selected.require("mutate")


def test_cache_distinguishes_declarations_for_identical_graphs():
    rcdb = pytest.importorskip("rcdb")
    content = "source,target,weight\na,b,1\n"
    one = DatasetSource(declaration(), StringIO(content))
    two = DatasetSource(declaration(unknown_fields="ignore"), StringIO(content))
    assert one.as_record()["state_digest"] == two.as_record()["state_digest"]
    assert one.digest != two.digest
    with closing(rcdb.MemoryStore()) as store:
        cache = QueryCache(store); request = parse('FROM DATASET("data") RETURN BETTI(0)')
        a = Executor(sources={"data": one}).execute_cached(request, cache)
        b = Executor(sources={"data": two}).execute_cached(request, cache)
        hit = Executor(sources={"data": one}).execute_cached(request, cache)
        assert a.values == b.values == hit.values == (2,)
        assert a.native_plan["cache"]["key"] != b.native_plan["cache"]["key"]
        assert hit.native_plan["cache"]["hit"]


@pytest.mark.parametrize("text", ['FROM DATASET("absent") RETURN BETTI(0)',
    'FROM DATASET(1) RETURN BETTI(0)', 'FROM DATASET("wrong") RETURN BETTI(0)'])
def test_query_text_cannot_open_or_infer_dataset_sources(text):
    with pytest.raises((KeyError, TypeError)):
        Executor(sources={"wrong": declaration()}).execute(parse(text))


def test_source_registry_refuses_ambiguity_and_preserves_live_classification():
    from rcql import SourceRegistry, SourceSignature
    registry = SourceRegistry()
    registry.register(SourceSignature("one", ValueKind.REX, lambda v: True))
    registry.register(SourceSignature("two", ValueKind.TEMPORAL_REX, lambda v: True))
    with pytest.raises(ValueError, match="ambiguous.*one.*two"):
        classify(object(), registry=registry)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(SourceSignature("one", ValueKind.REX, lambda v: True))
    registry.register(SourceSignature("explicit", ValueKind.RCDB_STORE, lambda v: True, 1))
    assert classify(object(), registry=registry) is ValueKind.RCDB_STORE


def test_exact_evidence_clock_does_not_narrow_before_cutoff():
    from types import SimpleNamespace
    from threading import RLock
    from rcql.source_context import SnapshotContext, SourceSelection
    from rexgraph import RexGraph
    from rexgraph.object_identity import object_digest
    graph = RexGraph.from_graph([0], [1]); clock = Q(2**54+1, 2)
    snapshot = SimpleNamespace(value=graph, state_digest=object_digest(graph),
        record=SimpleNamespace(id="record", version=1, tx_from=clock))
    class Source:
        _transaction_lock = RLock()
        def read_record(self, *a, **k):
            return snapshot
    with pytest.raises(ValueError, match="after"):
        SnapshotContext.select(Source(), [SourceSelection("graph", "record", version=1)], cutoff=clock-Q(1,4))
    context = SnapshotContext.select(Source(), [SourceSelection("graph", "record", version=1)], cutoff=clock)
    assert context.as_record()["closure"][0]["recorded_at"] == str(clock)
