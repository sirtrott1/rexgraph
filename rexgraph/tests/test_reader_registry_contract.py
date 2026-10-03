"""Reader resolution is shared by declarations and direct batch reads."""
from io import StringIO

import pytest

from rexgraph.io.readers import ReaderRegistry, ReaderSpec, read_batches
from rexgraph.io.records import RecordBatch, RecordField, RecordSchema, SourcePointer


def _reader(source, *, schema, batch_size, max_record_bytes):
    yield RecordBatch.from_records(schema, [schema.convert({"key": source.read()}, SourcePointer("memory:test"))])


def test_media_hint_reaches_batch_resolution():
    registry = ReaderRegistry()
    registry.register(ReaderSpec("plain", (), _reader, media_types=("text/plain",)))
    batches = list(read_batches(StringIO("00123"), registry=registry,
        schema=RecordSchema((RecordField("key"),)), media_type="text/plain; charset=utf-8"))
    assert batches[0].columns["key"].values == ("00123",)


def test_sniff_hint_reaches_batch_resolution():
    registry = ReaderRegistry()
    registry.register(ReaderSpec("magic", (), _reader, sniffer=lambda prefix: prefix == b"MAGIC"))
    assert next(read_batches(StringIO("retained"), registry=registry,
        schema=RecordSchema((RecordField("key"),)), prefix=b"MAGIC")).columns["key"].values == ("retained",)


def test_conflicting_media_and_sniff_claims_refuse():
    registry = ReaderRegistry()
    registry.register(ReaderSpec("media", (), _reader, media_types=("text/plain",)))
    registry.register(ReaderSpec("magic", (), _reader, sniffer=lambda prefix: prefix == b"MAGIC"))
    with pytest.raises(ValueError, match="ambiguous.*magic.*media"):
        next(read_batches(StringIO("data"), registry=registry,
            schema=RecordSchema((RecordField("key"),)), media_type="text/plain", prefix=b"MAGIC"))


def test_wrong_plugin_output_is_refused():
    registry = ReaderRegistry()
    registry.register(ReaderSpec("wrong", (), lambda *a, **k: iter([{"key": "untyped"}])))
    with pytest.raises(TypeError, match="RecordBatch"):
        next(read_batches(StringIO("data"), registry=registry, reader="wrong",
            schema=RecordSchema((RecordField("key"),))))


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "100"])
def test_declared_builtin_limits_refuse_before_opening(value):
    with pytest.raises((TypeError, ValueError)):
        next(read_batches("missing.json", reader="json", schema=RecordSchema((RecordField("key"),)),
                          max_document_bytes=value))


def test_unknown_options_and_missing_required_options_refuse_before_opening():
    schema = RecordSchema((RecordField("key"),))
    with pytest.raises(ValueError, match="undeclared.*guess"):
        next(read_batches("missing.csv", reader="csv", schema=schema, guess=True))
    with pytest.raises(ValueError, match="required.*query"):
        next(read_batches("missing.sqlite", reader="sqlite", schema=schema))


def test_profile_is_explicit_bounded_and_uses_csv_framing():
    from rexgraph.io.readers import profile
    source = StringIO('identifier,weight\n00123,0.1\nnext,"multiline\nrecord"\nmalformed\n')
    observed = profile(source, reader="csv", max_records=2)
    assert observed.fields == ("identifier", "weight") and observed.sampled_records == 2
    assert source.readline() == "malformed\n"
    with pytest.raises(ValueError, match="row"):
        profile(StringIO("a,b\n1\n"), reader="csv")
    with pytest.raises(ValueError, match="byte limit"):
        profile(StringIO("a\n"+"x"*100+"\n"), reader="csv", max_record_bytes=32)


def test_parameter_defaults_are_detached_and_have_stable_identity():
    from rexgraph.io.readers import ReaderParameter, ParameterSchema
    original = {"key": [2**53+1]}
    parameter = ReaderParameter("mapping", default=original)
    spec = ReaderSpec("custom", (), _reader, parameters=ParameterSchema((parameter,)))
    identity = spec.digest
    original["key"].append(0); parameter.default["key"].append(1)
    bound = spec.parameters.bind({}); bound["mapping"]["key"].append(2)
    assert spec.parameters.bind({}) == {"mapping": {"key": [2**53+1]}}
    assert spec.digest == identity


def test_reader_plugins_load_only_on_explicit_request_and_publish_atomically(monkeypatch):
    from types import SimpleNamespace
    import importlib.metadata
    registry = ReaderRegistry(); calls = []
    one = ReaderSpec("one", (".csv",), _reader)
    def entry(name, value):
        def load():
            calls.append(name); return value
        return SimpleNamespace(name=name, load=load)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **k: [entry("one", one), entry("bad", object())])
    assert registry.available() == [] and calls == []
    with pytest.raises(TypeError, match="ReaderSpec"):
        registry.load_plugins()
    assert registry.available() == []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **k: [entry("one", one)])
    assert registry.load_plugins() == ("one",)
    with pytest.raises(ValueError, match="already registered"):
        registry.load_plugins()


@pytest.mark.parametrize("prefix", ["bytes", b"x"*(64*1024+1)])
def test_prefix_limits_apply_even_with_an_explicit_reader(prefix):
    with pytest.raises(ValueError, match="bounded bytes"):
        next(read_batches(StringIO("a\n1\n"), reader="csv", prefix=prefix,
                          schema=RecordSchema((RecordField("a"),))))


def test_plugin_batch_schema_size_and_close_contract():
    schema = RecordSchema((RecordField("key"),)); closed = []
    def oversized(*args, **kwargs):
        try:
            yield RecordBatch.from_records(schema, [schema.convert({"key": "a"}, SourcePointer("memory:test"))]*2)
        finally:
            closed.append(True)
    registry = ReaderRegistry(); registry.register(ReaderSpec("oversized", (), oversized))
    with pytest.raises(ValueError, match="schema or size"):
        next(read_batches(StringIO(""), reader="oversized", registry=registry, schema=schema, batch_size=1))
    assert closed == [True]


def test_dataset_declaration_refuses_changed_nested_options():
    from rexgraph import RelationSpec
    from rexgraph.io.declaration import DatasetDeclaration
    declaration = DatasetDeclaration("sqlite", RecordSchema((RecordField("a"), RecordField("b"))),
        RelationSpec("pair", ("a", "b")), {"query": "SELECT a,b FROM data WHERE id=?", "parameters": [1]})
    declaration.options["parameters"].append(2)
    with pytest.raises(ValueError, match="declaration changed"):
        declaration.read("missing.sqlite")
    with pytest.raises(ValueError, match="declaration changed"):
        declaration.to_bytes()


def test_tab_separated_media_declares_tab_delimiter_for_an_unnamed_stream():
    from rexgraph.io.readers import profile
    schema = RecordSchema((RecordField("a"), RecordField("b")))
    batch = next(read_batches(StringIO("a\tb\nx\ty\n"), schema=schema,
                              media_type="text/tab-separated-values"))
    assert batch.columns["a"].values == ("x",)
    assert profile(StringIO("a\tb\nx\ty\n"), media_type="text/tab-separated-values").reader == "tsv"
