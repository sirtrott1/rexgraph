"""Reader independent declarations preserve values, IDs, absence and branching."""
from fractions import Fraction as Q
from io import StringIO
from contextlib import closing
import csv
import json
import sqlite3

import pytest

from rexgraph import Absent, NumberRule, RelationSpec, VertexTable, ValueRules
from rexgraph.construction import construct
from rexgraph.io.declaration import DatasetDeclaration
from rexgraph.io.records import RecordBatch, RecordField, RecordSchema, SourcePointer, TypedRecord
from rexgraph.io.readers import ReaderRegistry, ReaderSpec, read_batches
from rexgraph.native_rank import primary_columns
from rexgraph.state import from_state, to_state


def pair_schema(rule=NumberRule.DECIMAL_EXACT):
    return RecordSchema((RecordField("source", nullable=False), RecordField("target", nullable=False),
                         RecordField("id", nullable=False), RecordField("weight", "number", ValueRules(rule, ("", "NA")))))


def pair_spec():
    return RelationSpec("pair", ("source", "target"), weight="weight", identity="id")


def test_csv_batches_preserve_text_decimal_absence_and_source_positions():
    data = StringIO("source,target,id,weight\n00123,b,pair/1,0.1\nb,c,pair/2,NA\nc,d,pair/3,1e-3\n")
    read = list(read_batches(data, reader="csv", schema=pair_schema(), batch_size=2))
    assert [len(b) for b in read] == [2, 1]
    assert read[0].columns["source"].values[0] == "00123"
    assert read[0].columns["weight"].exact().values().tolist() == [Q(1, 10), Absent]
    assert read[1].pointers[0].record == 2
    assert isinstance(read[0], RecordBatch)
    relations = construct(read, pair_spec())
    assert relations.vertices.ids == ("00123", "b", "c", "d")
    assert relations.weight.values().tolist() == [Q(1, 10), Absent, Q(1, 1000)]
    assert relations.relation_id == ("pair/1", "pair/2", "pair/3")


@pytest.mark.parametrize("reader,content", [
    ("json", '[{"source":"00123","target":"b","id":"r","weight":0.1}]'),
    ("jsonl", '{"source":"00123","target":"b","id":"r","weight":0.1}\n'),
])
def test_json_decimal_tokens_do_not_pass_through_binary_float(reader, content):
    declaration = DatasetDeclaration(reader, pair_schema(), pair_spec())
    r = declaration.to_rex(StringIO(content))
    assert r.relations.weight.values().tolist() == [Q(1, 10)]
    assert r.relations.vertices.ids[0] == "00123"
    assert r.relation_keys == ("r",)
    assert from_state(to_state(r)).relation_keys == ("r",)


def test_row_relation_rational_declaration_has_one_cell_and_independent_weight():
    rule = ValueRules(NumberRule.JSON_RATIONAL)
    schema = RecordSchema((RecordField("participants", "sequence"), RecordField("shares", "sequence", rule),
                           RecordField("weight", "number", rule), RecordField("id"), RecordField("sign", "number")))
    rows = [{"participants": ["a", "b", "c", "d"], "shares": [{"numerator": n, "denominator": d} for n,d in [(0,1),(1,4),(1,2),(1,4)]],
             "weight": {"numerator": 2, "denominator": 3}, "id": "group/1", "sign": -1}]
    declaration = DatasetDeclaration("json", schema, RelationSpec("row", ("participants",), share="shares", weight="weight", sign="sign", identity="id"),
                                     vertices=VertexTable(("a", "b", "c", "d", "isolate")))
    r = declaration.to_rex(StringIO(json.dumps(rows)))
    assert r.nE == 1 and r.nV == 5
    assert primary_columns(r) == [{0: Q(-1), 1: Q(1,4), 2: Q(1,2), 3: Q(1,4)}]
    assert r._exact_column_norms_B1() == [Q(11,8)]
    assert r.relations.weight.values().tolist() == [Q(2,3)]
    assert r.relations.sign.tolist() == [-1]
    assert DatasetDeclaration.from_bytes(declaration.to_bytes()).to_bytes() == declaration.to_bytes()


def test_long_form_grouping_spans_batches_and_preserves_identity_order():
    rules = ValueRules(NumberRule.DECIMAL_EXACT)
    schema = RecordSchema((RecordField("id"), RecordField("participant"), RecordField("share", "number", rules)))
    text = "id,participant,share\ngroup,a,0\ngroup,b,.25\ngroup,c,.5\ngroup,d,.25\n"
    declaration = DatasetDeclaration("csv", schema, RelationSpec("long", ("participant",), identity="id", share="share"), {"batch_size": 2})
    r = declaration.to_rex(StringIO(text))
    assert r.nE == 1 and r.relation_keys == ("group",)
    assert r._exact_column_norms_B1() == [Q(11,8)]
    assert r.relations.weight.values()[0] is Absent
    assert len(r.provenance["source_pointers"][0]) == 4


def test_long_form_conflicting_relation_values_are_refused():
    schema = RecordSchema((RecordField("id"), RecordField("participant"), RecordField("weight", "number")))
    content = '[{"id":"r","participant":"a","weight":1},{"id":"r","participant":"b","weight":2}]'
    declaration = DatasetDeclaration("json", schema, RelationSpec("long", ("participant",), identity="id", weight="weight"))
    with pytest.raises(ValueError, match="conflicting"):
        declaration.to_rex(StringIO(content))


def test_record_registry_requires_explicit_resolution_of_equal_priority_matches():
    registry = ReaderRegistry()
    registry.register(ReaderSpec("one", (".csv",), lambda *a, **k: iter(())))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ReaderSpec("one", (".csv",), lambda *a, **k: iter(())))
    registry.register(ReaderSpec("two", (".csv",), lambda *a, **k: iter(())))
    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve("data.csv")
    assert registry.resolve("data.csv", reader="one").name == "one"
    registry.register(ReaderSpec("three", (".csv",), lambda *a, **k: iter(()), priority=1))
    assert registry.resolve("data.csv").name == "three"


@pytest.mark.parametrize("reader,raw", [("csv", "a,a\n1,2\n"), ("jsonl", '{"a":"1","a":"2"}\n'),
                                        ("jsonl", '{"a":NaN}\n'), ("csv", 'a\n"unterminated\n')])
def test_malformed_source_records_are_refused(reader, raw):
    with pytest.raises((ValueError, TypeError, csv.Error)):
        list(read_batches(StringIO(raw), reader=reader, schema=RecordSchema((RecordField("a"),))))


@pytest.mark.parametrize("reader,raw", [("csv", 'a\n"' + "x"*100 + '"\n'), ("jsonl", '{"a":"' + "x"*100 + '"}\n')])
def test_record_limits_bound_source_reads(reader, raw):
    with pytest.raises(ValueError, match="byte limit"):
        list(read_batches(StringIO(raw), reader=reader, schema=RecordSchema((RecordField("a"),)), max_record_bytes=32))


def test_required_and_unknown_fields_are_not_silently_lost():
    schema = RecordSchema((RecordField("a", nullable=False),))
    for raw in ('[{"b":"missing"}]', '[{"a":null}]', '[{"a":"present","extra":"lost"}]'):
        with pytest.raises(ValueError):
            list(read_batches(StringIO(raw), reader="json", schema=schema))


@pytest.mark.parametrize("format", ["arrow", "parquet", "sqlite", "xlsx"])
def test_tabular_readers_preserve_integer_above_binary_float_precision(format, tmp_path):
    number = 2**53+1
    schema = RecordSchema((RecordField("key", "string"), RecordField("number", "number", ValueRules(NumberRule.TYPED_COLUMNAR))))
    if format in {"arrow", "parquet"}:
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"key": ["00123"], "number": pa.array([number], type=pa.int64())})
        path = tmp_path / ("input."+format)
        if format == "parquet":
            import pyarrow.parquet as pq
            pq.write_table(table, path)
        else:
            with pa.OSFile(str(path), "wb") as stream, pa.ipc.new_file(stream, table.schema) as writer:
                writer.write_table(table)
        options = {}
    elif format == "sqlite":
        path = tmp_path / "input.sqlite"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE data (key TEXT, number INTEGER)")
            connection.execute("INSERT INTO data VALUES (?, ?)", ("00123", number))
        options = {"query": "SELECT key, number FROM data"}
    else:
        openpyxl = pytest.importorskip("openpyxl")
        # XLSX stores numbers through decimal/binary spreadsheet carriers. A wide
        # identifier must be text; it is deliberately not promoted to a number.
        path = tmp_path / "input.xlsx"
        book = openpyxl.Workbook(); sheet = book.active
        sheet.append(["key", "number"]); sheet.append(["00123", 123]); book.save(path); book.close()
        number = 123
        options = {}
    batch = next(read_batches(path, reader=format, schema=schema, batch_size=1, **options))
    assert batch.columns["number"].values == (number,)
    assert type(batch.columns["number"].values[0]) is int
    assert batch.columns["key"].values == ("00123",)


def test_sql_reader_cannot_mutate_the_source(tmp_path):
    path = tmp_path / "read.sqlite"
    with closing(sqlite3.connect(path)) as c, c:
        c.execute("CREATE TABLE records (a TEXT)"); c.execute("INSERT INTO records VALUES ('retained')")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        list(read_batches(path, reader="sqlite", schema=RecordSchema((RecordField("a"),)), query="DELETE FROM records RETURNING a"))
    with closing(sqlite3.connect(path)) as c:
        assert c.execute("SELECT a FROM records").fetchall() == [("retained",)]


def test_sql_reader_cannot_create_an_attached_database(tmp_path):
    source = tmp_path / "source.sqlite"
    external = tmp_path / "external.sqlite"
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("CREATE TABLE records (a TEXT)")
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        list(read_batches(source, reader="sqlite", schema=RecordSchema((RecordField("a"),)),
                          query="ATTACH DATABASE ? AS external", parameters=(str(external),)))
    assert not external.exists()


def test_xlsx_shortest_decimal_is_a_declared_source_rule(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "decimal.xlsx"
    book = openpyxl.Workbook(); sheet = book.active
    sheet.append(["identity", "number"]); sheet.append(["00123", .1]); book.save(p); book.close()
    schema = RecordSchema((RecordField("identity"), RecordField("number", "number", ValueRules(NumberRule.XLSX_SHORTEST_DECIMAL))))
    batch = next(read_batches(p, schema=schema))
    assert batch.columns["identity"].values == ("00123",)
    assert batch.columns["number"].values == (Q(1,10),)


def test_json_record_limit_is_independent_of_the_document_limit(tmp_path):
    path = tmp_path / "records.json"
    path.write_text('[{"text":"short"},{"text":"' + "x"*100 + '"}]')
    schema = RecordSchema((RecordField("text"),))
    stream = read_batches(path, schema=schema, batch_size=1,
                          max_record_bytes=30, max_document_bytes=1000)
    assert next(stream).columns["text"].values == ("short",)
    with pytest.raises(ValueError, match="JSON record exceeds"):
        next(stream)


@pytest.mark.parametrize("document", [
    '[{"text":"x"},]', '[{"text":"x"}', '[{"text":"x"} {}]',
    '[{"text":"x"}] trailing', '[{"text":"x","text":"y"}]',
])
def test_incremental_json_array_reader_keeps_strict_framing(tmp_path, document):
    path = tmp_path / "invalid.json"
    path.write_text(document)
    with pytest.raises(ValueError):
        list(read_batches(path, schema=RecordSchema((RecordField("text"),)), batch_size=1))


def test_json_array_reader_counts_utf8_bytes_and_accepts_empty_arrays(tmp_path):
    path = tmp_path / "unicode.json"
    path.write_text('[{"text":"€€€"}]')
    schema = RecordSchema((RecordField("text"),))
    with pytest.raises(ValueError, match="record exceeds"):
        list(read_batches(path, schema=schema, max_record_bytes=19))
    path.write_text(' \n[ ]\t')
    assert list(read_batches(path, schema=schema)) == []


def test_construction_does_not_import_dense_oracle_paths(monkeypatch):
    import rexgraph.evaluator as evaluator
    monkeypatch.setattr(evaluator, "require_explicit_dense_oracle", lambda *a, **k: pytest.fail("dense oracle reached"))
    records = [TypedRecord({"source": "a", "target": "b", "weight": Q(1,3), "id": "r"}, SourcePointer("memory:fixture", record=0))]
    assert construct(records, pair_spec()).n_relations == 1
