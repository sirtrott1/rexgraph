"""Exact columns across full, projected and streaming Parquet/Arrow reads."""
from fractions import Fraction as Q
import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from rexgraph.exact_array import ExactArray
from rexgraph.io.columnar import exact_from_arrow, exact_to_arrow
from rexgraph.io.export import export_parquet, parquet_bytes, verify_export
from rexgraph.io.parquet_bridge import read_parquet, read_parquet_batches, write_parquet
from rexgraph.io.readers import read_batches
from rexgraph.io.records import RecordField, RecordSchema
from rexgraph.value import Absent, NumberRule, ValueRules


@pytest.fixture
def exact():
    return ExactArray.from_values([0, Q(1), Q(-2, 3), Absent, 2**16000, Q(1, 2**100)])


def test_exact_arrow_preserves_coefficient_kind_presence_and_bigints(exact):
    physical = exact_to_arrow(exact)
    assert pa.types.is_struct(physical.type)
    assert physical.field("numerator").type == pa.int64()
    assert exact_from_arrow(physical) == exact
    assert exact_from_arrow(pa.chunked_array([physical.slice(0, 2), physical.slice(2)])) == exact


@pytest.mark.parametrize("escape", [(b"\x01\x01", None), (b"\x02\x01\x00", b"\x01\x01")])
def test_incomplete_or_noncanonical_bigint_escapes_are_refused(escape):
    physical = exact_to_arrow(ExactArray.from_values([1]))
    children = [physical.field(i) for i in range(4)] + [pa.array([v], type=pa.binary()) for v in escape]
    malformed = pa.StructArray.from_arrays(children, names=physical.type.names)
    with pytest.raises(ValueError):
        exact_from_arrow(malformed)


def test_parquet_export_reimports_exact_values_and_verifies_lineage(tmp_path, exact):
    payload, manifest = export_parquet({"measure": exact}, partition_digest="declared-lineage")
    assert verify_export(payload, manifest)
    path = tmp_path / "exact.parquet"
    path.write_bytes(payload)
    loaded = read_parquet(path)
    assert ExactArray.from_values(loaded["measure"]) == exact
    assert parquet_bytes({"measure": exact}) == parquet_bytes({"measure": exact.values()})


def test_exact_two_dimensional_projection_and_streaming(tmp_path):
    value = ExactArray.from_values([[Q(2, 3), Absent], [2**100, Q(1, 7)], [0, Q(1)]])
    path = tmp_path / "matrix.parquet"
    write_parquet({"matrix": value, "other": np.arange(3)}, path)
    assert ExactArray.from_values(read_parquet(path, columns=["matrix"])["matrix"]) == value
    assert read_parquet(path, columns=["matrix_1"])["matrix_1"][0] is Absent
    batches = list(read_parquet_batches(path, batch_rows=1, columns=["matrix"]))
    assert all(set(batch) == {"matrix"} for batch in batches)
    assert ExactArray.from_values(np.concatenate([b["matrix"] for b in batches])) == value


@pytest.mark.parametrize("reader", ["parquet", "arrow"])
def test_declared_record_readers_consume_exact_physical_columns(tmp_path, exact, reader):
    payload = parquet_bytes({"measure": exact})
    table = pq.read_table(pa.BufferReader(payload))
    path = tmp_path / ("records." + reader)
    if reader == "parquet":
        path.write_bytes(payload)
    else:
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    schema = RecordSchema((RecordField("measure", "number", nullable=True,
                           rules=ValueRules(NumberRule.TYPED_COLUMNAR)),))
    batches = list(read_batches(path, reader=reader, schema=schema, batch_size=2))
    assert ExactArray.from_values([r.fields["measure"] for b in batches for r in b.records()]) == exact


def test_logical_column_collisions_and_unknown_codecs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="collides"):
        parquet_bytes({"x": np.ones((2, 2)), "x_0": np.ones(2)})
    path = tmp_path / "unknown.parquet"
    table = pa.table({"x": [1]}).replace_schema_metadata({b"rex_col_meta": json.dumps(
        {"x": {"shape": [1], "split": False, "codec": "future"}}).encode()})
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="unsupported"):
        read_parquet(path)


def test_one_oversized_record_cannot_hide_in_a_small_average_batch(tmp_path):
    path = tmp_path / "oversized.parquet"
    pq.write_table(pa.table({"text": ["x"*500] + ["x"]*19}), path)
    schema = RecordSchema((RecordField("text", "string"),))
    with pytest.raises(ValueError, match="record exceeds"):
        list(read_batches(path, reader="parquet", schema=schema, max_record_bytes=100))
