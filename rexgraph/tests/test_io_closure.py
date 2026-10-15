"""Result containers and SQL streams preserve data across their public adapters."""
from collections import namedtuple
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.io._serialization import (
    HDF5Adapter, NpyAdapter, ZarrAdapter, read_namedtuple, read_result_dict,
    write_namedtuple, write_result_dict,
)


@pytest.fixture(params=["npy", "hdf5", "zarr"])
def adapter(request, tmp_path):
    if request.param == "npy":
        yield NpyAdapter(tmp_path / "arrays")
    elif request.param == "hdf5":
        h5py = pytest.importorskip("h5py")
        with h5py.File(tmp_path / "arrays.h5", "w") as group:
            yield HDF5Adapter(group)
    else:
        zarr = pytest.importorskip("zarr")
        yield ZarrAdapter(zarr.open_group(str(tmp_path / "arrays.zarr"), mode="w"))


def test_nested_result_arrays_are_retained(adapter):
    write_result_dict(adapter, "result", {"nested": {"values": np.arange(4), "count": 4}})
    restored = read_result_dict(adapter, "result")
    np.testing.assert_array_equal(restored["nested"]["values"], np.arange(4))
    assert restored["nested"]["count"] == 4


def test_empty_result_replacement_does_not_restore_old_scalars(adapter):
    write_result_dict(adapter, "result", {"old": 4})
    write_result_dict(adapter, "result", {})
    assert read_result_dict(adapter, "result") == {}


def test_unknown_namedtuple_retains_fields_in_its_dictionary_fallback(adapter):
    result_type = namedtuple("ExternalSyntheticResult", "values count optional")
    write_namedtuple(adapter, "result", result_type(np.arange(3), 3, None))
    restored = read_namedtuple(adapter, "result")
    np.testing.assert_array_equal(restored["values"], np.arange(3))
    assert restored["count"] == 3 and restored["optional"] is None


def test_namedtuple_replacement_does_not_keep_old_none_fields(adapter):
    result_type = namedtuple("ExternalSyntheticResult", "values")
    write_namedtuple(adapter, "result", result_type(None))
    write_namedtuple(adapter, "result", result_type(np.arange(3)))
    np.testing.assert_array_equal(read_namedtuple(adapter, "result", result_type).values, np.arange(3))


def test_namedtuple_replacement_uses_the_current_field_kind(adapter):
    result_type = namedtuple("ExternalSyntheticResult", "values")
    write_namedtuple(adapter, "result", result_type(np.arange(3)))
    write_namedtuple(adapter, "result", result_type(7))
    assert read_namedtuple(adapter, "result", result_type).values == 7


def test_string_lookup_retains_a_none_default(adapter):
    assert adapter.get_string("absent", None) is None


def test_legacy_namedtuple_dictionary_layout_remains_readable(adapter):
    result_type = namedtuple("LegacySyntheticResult", "nested")
    root = adapter.subgroup("old")
    root.put_string("_type_name", result_type.__name__)
    nested = root.subgroup("nested")
    nested.put_scalar("_is_dict", True)
    nested.put_array("values", np.arange(3))
    nested.put_json("_array_keys", ["values"])
    nested.put_json("_scalars", {"count": 3})
    restored = read_namedtuple(adapter, "old", result_type)
    np.testing.assert_array_equal(restored.nested["values"], np.arange(3))
    assert restored.nested["count"] == 3


def test_storage_member_names_are_literal_and_keep_their_namespace(adapter):
    adapter.put_scalar("root", 7)
    for name in [".", "..", "nested/name", "%2F", "../other"]:
        child = adapter.subgroup(name)
        child.put_scalar("root", 9)
        child.put_array(name, np.arange(3))
        np.testing.assert_array_equal(adapter.subgroup(name).get_array(name), np.arange(3))
    assert adapter.get_scalar("root") == 7


def test_nested_result_dictionaries_retain_arrays_at_each_depth(adapter):
    write_result_dict(adapter, "result", {"a": {"b": {"values": np.arange(3)}}})
    restored = read_result_dict(adapter, "result")
    np.testing.assert_array_equal(restored["a"]["b"]["values"], np.arange(3))


@pytest.mark.parametrize("array", [
    np.array(9, dtype=">i8"),
    np.array(complex(-0.0, -0.0), dtype=">c16"),
    np.array([complex(-0.0, -0.0), complex(1.0, -0.0)], dtype=np.complex128),
])
def test_arrays_keep_scalar_shape_dtype_signed_zeros_and_support_replacement(adapter, array):
    result_type = namedtuple("ScalarAndComplexSyntheticResult", "values")
    write_namedtuple(adapter, "result", result_type(array))
    for _ in range(2):
        restored = read_namedtuple(adapter, "result", result_type).values
        assert restored.shape == array.shape and restored.dtype == array.dtype
        assert restored.tobytes() == array.tobytes()
        write_namedtuple(adapter, "result", result_type(array))


def test_numpy_adapter_names_do_not_escape_the_container(tmp_path):
    adapter = NpyAdapter(tmp_path / "arrays")
    adapter.put_array("../outside", np.arange(3))
    assert not (tmp_path / "outside.npy").exists()
    np.testing.assert_array_equal(adapter.get_array("../outside"), np.arange(3))
    adapter.subgroup("../escaped").put_scalar("value", 7)
    assert not (tmp_path / "escaped").exists()
    assert adapter.subgroup("../escaped").get_scalar("value") == 7


@pytest.mark.parametrize("name", ["with spaces", "select\nmeasurements"])
def test_sql_stream_accepts_the_same_literal_table_names_as_its_writer(name, tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    from rexgraph.io.sql_bridge import write_metrics_sql, read_sql_batches
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'tables.db'}")
    try:
        values = [Fraction(1, 7), Fraction(2, 7), 2**200]
        write_metrics_sql({"value": values}, engine, name)
        batches = list(read_sql_batches(engine, name, chunksize=2))
        assert np.concatenate([batch["value"] for batch in batches]).tolist() == values
    finally:
        engine.dispose()


@pytest.mark.parametrize("query", ["SELECT\n7", "SELECT\t7"])
def test_sql_stream_accepts_multiline_and_tabbed_queries(query, tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    from rexgraph.io.sql_bridge import read_sql_batches
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'queries.db'}")
    try:
        assert list(next(read_sql_batches(engine, query)).values())[0].tolist() == [7]
    finally:
        engine.dispose()


def test_sql_stream_uses_one_cursor_and_closes_it_on_cancellation(tmp_path, monkeypatch):
    sa = pytest.importorskip("sqlalchemy")
    from rexgraph.io.sql_bridge import write_metrics_sql, read_sql_batches
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'cursor.db'}")
    write_metrics_sql({"value": np.arange(9)}, engine)
    results = []
    execute = sa.engine.Connection.execute
    def tracked(connection, statement, *args, **kwargs):
        result = execute(connection, statement, *args, **kwargs)
        if isinstance(statement, sa.sql.Select) and any(t.name == "metrics" for t in statement.get_final_froms()):
            results.append(result)
        return result
    monkeypatch.setattr(sa.engine.Connection, "execute", tracked)
    try:
        with engine.connect() as connection:
            batches = list(read_sql_batches(connection, "metrics", chunksize=2))
            assert np.concatenate([batch["value"] for batch in batches]).tolist() == list(range(9))
            assert len(results) == 1 and results[0].closed
            stream = read_sql_batches(connection, "metrics", chunksize=2)
            assert next(stream)["value"].tolist() == [0, 1]
            stream.close()
            assert results[-1].closed
            assert not connection.closed
            assert connection.execute(sa.select(sa.literal(7))).scalar_one() == 7
    finally:
        engine.dispose()
