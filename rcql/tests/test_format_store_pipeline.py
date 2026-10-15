"""Whole graph state and portable RCQL results survive checked store transfers."""
from contextlib import closing
from fractions import Fraction

import pytest

from rexgraph import Absent, Relations, RexGraph, VertexTable
from rexgraph.object_identity import object_digest
from rcql import Executor, Result, parse, register_result_storage_codec


def graph():
    return RexGraph.from_relations(Relations.from_supports(
        [[0, 1, 2], [1, 2], [2, 2]],
        vertices=VertexTable(("a", "b", "c", "isolated")),
        weights=[Fraction(2, 7), Absent, 0], relation_ids=["branch", "pair", "loop"],
        attributes={1: {0: {"exact": Fraction(3, 11), "large": 2**500}}}), c_channel="count")


def target(kind, path):
    rcdb = pytest.importorskip("rcdb")
    if kind == "local":
        return rcdb.LocalStore(path)
    if kind == "sqlite":
        pytest.importorskip("sqlalchemy")
        return rcdb.SQLStore(f"sqlite:///{path}.sqlite")
    pytest.importorskip("fsspec")
    return rcdb.NativeObjectStore("file://" + str(path))


@pytest.mark.parametrize("kind", ["local", "sqlite", "object-local"])
@pytest.mark.parametrize("format", ["rcbd", "safetensors", "h5", "zarr", "json", "arrow", "sql"])
def test_format_history_query_result_and_checked_copy(kind, format, tmp_path):
    rcdb = pytest.importorskip("rcdb")
    dependency = {"safetensors": "safetensors", "h5": "h5py", "zarr": "zarr",
                  "arrow": "pyarrow", "sql": "sqlalchemy"}.get(format)
    if dependency:
        pytest.importorskip(dependency)
    original = graph()
    if format == "arrow":
        from rexgraph.io.arrow_bridge import arrow_to_rex, rex_to_arrow
        converted = arrow_to_rex(rex_to_arrow(original))
    elif format == "sql":
        from rexgraph.io.sql_bridge import write_boundary_sql, reconstruct_rex_sql
        uri = f"sqlite:///{tmp_path / 'format.sqlite'}"
        write_boundary_sql(original, uri)
        converted = reconstruct_rex_sql(uri, boundary="boundary")
    else:
        from rexgraph.io import save, load
        path = tmp_path / ("graph." + format)
        save(path, original)
        converted = load(path)
    expected = object_digest(original)
    assert object_digest(converted) == expected
    codec = register_result_storage_codec()
    root = tmp_path / "database"
    with closing(rcdb.MemoryStore()) as source, closing(target(kind, root)) as store:
        source.put("input", converted, analytics=False, _tx_time=10.0)
        source.put("input", RexGraph.from_graph([0], [1]), analytics=False, _tx_time=20.0)
        receipt = rcdb.copy_record(source, store, source.get_record("input", as_of=15.0),
                                   return_receipt=True, tx_time=10.0)
        assert receipt.source_digest == receipt.destination_digest == expected
        store.put("input", RexGraph.from_graph([0], [1]), analytics=False, _tx_time=20.0)
        result = Executor(sources={"db": store}).execute(
            parse('FROM $db RETURN RCDB_GET("input@1"), RCDB_HASH("input@1")'))
        assert object_digest(result.values[0]) == result.values[1] == expected
        portable = Result.from_bytes(result.to_bytes())
        assert portable.to_bytes() == result.to_bytes()
        source.put_record("query", portable, codec=codec)
        rcdb.copy_record(source, store, source.get_record("query"))
        packet = portable.to_bytes()
    with closing(target(kind, root)) as restored:
        assert object_digest(restored.get("input", as_of=15.0)) == expected
        assert restored.get("input", as_of=25.0).nE == 1
        query = restored.read_record("query").value
        assert query.to_bytes() == packet
        assert object_digest(query.values[0]) == query.values[1] == expected
