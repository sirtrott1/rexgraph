"""Typed native records share every provider's one ownership/version engine."""
from contextlib import closing
from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from rexgraph import Absent, RexGraph
from rexgraph.value_codec import pack_value, unpack_value
from rcdb import (BlobCodecSpec, CodecRef, COPY_RECEIPT_CODEC, DECLARATION_CODEC, FileStore,
                  LocalStore, MemoryStore, NativeObjectStore, MIGRATION_PLAN_CODEC, MIGRATION_PROGRESS_CODEC,
                  MIGRATION_STEP_CODEC, PROVENANCE_CODEC, RecordCodec, SQLStore, StoredRecord,
                  VALUE_CODEC, VersionConflictError, copy_record, migrate_batch, plan_migration,
                  register_record_codec, unregister_record_codec)
from rcdb.header import DEFAULT_RECORD_CODECS


def opened(kind, path, **options):
    if kind == "memory": return MemoryStore(**options)
    if kind == "local": return LocalStore(path, **options)
    if kind.startswith("object-"):
        prefix = "memory://" if kind == "object-memory" else "file://"
        return NativeObjectStore(prefix+str(path), **options)
    pytest.importorskip("sqlalchemy")
    return SQLStore(f"sqlite:///{path}.sqlite", **options)


def value(n=1):
    return {"q": Fraction(n, 7), "large": 2**90, "absent": Absent,
            "array": np.array([[1, 2]], np.int16), "tuple": (1, [])}


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_native_values_share_version_closure_tombstones_and_owned_payloads(kind, tmp_path):
    with closing(opened(kind, tmp_path / "store", compression=BlobCodecSpec("zlib"))) as store:
        original = value()
        expected = pack_value(original)
        first = store.put_record("literal/id@1", original, meta={"exact": Fraction(2, 7)},
                                 tags=["record"], tx_time=10.0, valid_from=1.0, valid_to=9.0, expected_version=0)
        assert isinstance(first, StoredRecord) and first.envelope.codec == VALUE_CODEC.name
        original["array"][0, 0] = 999
        snapshot = store.read_record(first.id, version=1)
        assert pack_value(snapshot.value) == expected
        assert snapshot.state_digest == first.envelope.object_digest
        snapshot.value["array"] = snapshot.value["array"].copy()
        snapshot.value["array"][0, 0] = 123
        assert pack_value(store.get_version(first.id, 1)) == expected
        second = store.put_record(first.id, value(2), tx_time=20.0, expected_version=1)
        assert second.version == 2 and store.history(first.id)[0].tx_to == 20.0
        assert pack_value(store.get(first.id, as_of=15.0)) == expected
        assert store.delete(first.id, tx_time=30.0)
        assert store.get(first.id) is None and store.get_version(first.id, 1) is not None
        assert store.put_record(first.id, value(3), tx_time=40.0, expected_version=0).version == 3
        assert [f.operation for f in store.changes()] == ["put", "put", "delete", "put"]
        digest = store.state_digest()
        assert not store.verify_commits(first.id)
        if kind != "memory":
            identity, cursor, header = store.store_id, store.change_cursor, store.header
            store.close()
            with closing(opened(kind, tmp_path / "store")) as reopened:
                assert reopened.store_id == identity and reopened.change_cursor == cursor and reopened.header == header
                assert reopened.state_digest() == digest


@pytest.mark.parametrize("source_kind,target_kind", list(product(("memory", "local", "sql", "object-file", "object-memory"), repeat=2)))
def test_mixed_native_history_replays_through_single_copy_seam(source_kind, target_kind, tmp_path):
    with closing(opened(source_kind, tmp_path / "source")) as source:
        with closing(opened(target_kind, tmp_path / "target", compression=BlobCodecSpec("zlib"))) as target:
            source.put("graph", RexGraph.from_graph([0], [1]), analytics=False, _tx_time=10.0)
            source.put_record("data", value(), tx_time=10.0, meta={"q": Fraction(1, 7)}, tags=["typed"])
            source.put_record("proof", {"digest": source.state_digest(), "method": "exact"},
                              codec=PROVENANCE_CODEC, tx_time=15.0)
            source.delete("data", tx_time=20.0)
            source.put_record("data", value(2), tx_time=30.0)
            plan = plan_migration(source, target)
            first = migrate_batch(source, target, plan, limit=2)
            result = migrate_batch(source, target, plan, progress=first.progress)
            assert result.complete and source.state_digest() == target.state_digest()
            assert all(r.copy is None or r.copy.source_digest == r.copy.destination_digest
                       for r in (*first.receipts, *result.receipts))
            assert target.get("graph").nE == 1
            assert pack_value(target.get("data")) == pack_value(value(2))


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_migration_declarations_and_copy_receipts_persist_as_typed_audit_records(kind, tmp_path):
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put_record("r", value(), tx_time=10.0)
        plan = plan_migration(source, target)
        batch = migrate_batch(source, target, plan)
        declarations = ((MIGRATION_PLAN_CODEC, plan), (MIGRATION_PROGRESS_CODEC, batch.progress),
                        (MIGRATION_STEP_CODEC, batch.receipts[0]), (COPY_RECEIPT_CODEC, batch.receipts[0].copy))
        with closing(opened(kind, tmp_path / "audit")) as audit:
            for i, (codec, declaration) in enumerate(declarations):
                record = audit.put_record(f"audit/{i}", declaration, codec=codec, tx_time=10.0)
                loaded = audit.read_record(record.id)
                assert type(loaded.value) is type(declaration) and loaded.value.to_bytes() == declaration.to_bytes()
            # Audit records live under a separate identity, leaving replay resumable.
            assert migrate_batch(source, target, plan).complete
            digest = audit.state_digest()
            if kind != "memory":
                audit.close()
                with closing(opened(kind, tmp_path / "audit")) as reopened:
                    assert reopened.state_digest() == digest
                    assert reopened.get("audit/0") == plan


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_declared_dataset_is_reconstructed_without_executing_a_reader(kind, tmp_path, monkeypatch):
    from rexgraph.io.declaration import DatasetDeclaration
    from rexgraph.io.records import RecordField, RecordSchema
    from rexgraph.relations import RelationSpec
    declaration = DatasetDeclaration("uninstalled.reader", RecordSchema((RecordField("source"), RecordField("target"))),
                                     RelationSpec("pair", ("source", "target")))
    def executed(*args, **kwargs): raise AssertionError("a persisted declaration executed a reader")
    monkeypatch.setattr(DatasetDeclaration, "read", executed)
    with closing(opened(kind, tmp_path / "store")) as store:
        store.put_record("dataset", declaration, codec=DECLARATION_CODEC)
        restored = store.get("dataset")
        assert isinstance(restored, DatasetDeclaration) and restored.to_bytes() == declaration.to_bytes()


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_explicit_result_codec_reuses_rcql_and_system_portable_results(kind, tmp_path):
    rcql = pytest.importorskip("rcql")
    from rcql import Result
    from rcql.types import Exactness
    result = Result(values=(Fraction(1, 7), Absent, np.array([Fraction(2, 7)], object)),
                         exactness=(Exactness.RATIONAL, None, Exactness.RATIONAL),
                         provenance=({"method": "native", "exact": Fraction(1, 7)},))
    reference = rcql.register_result_storage_codec()
    assert rcql.register_result_storage_codec() == reference
    with closing(opened(kind, tmp_path / "store")) as store:
        store.put_record("query", result, codec=reference)
        restored = store.get("query")
        assert isinstance(restored, Result) and restored.to_bytes() == result.to_bytes()
        system = pytest.importorskip("system.server.app")
        assert system._result_payload(restored) == system._result_payload(result)
        assert system._result_payload(restored)["exactness"] == ["rational", None, "rational"]


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_old_graph_only_headers_are_not_implicitly_expanded(kind, tmp_path):
    graph_only = (DEFAULT_RECORD_CODECS[0],)
    with closing(opened(kind, tmp_path / "store", record_codecs=graph_only)) as store:
        store.put("graph", RexGraph.from_graph([0], [1]), analytics=False)
        before = store.change_cursor
        with pytest.raises(ValueError, match="not declared"):
            store.put_record("generic", value())
        assert store.change_cursor == before and store.header.record_codecs == graph_only
        if kind != "memory":
            store.close()
            with closing(opened(kind, tmp_path / "store")) as reopened:
                assert reopened.header.record_codecs == graph_only
            with pytest.raises(ValueError, match="record codecs.*differ"):
                opened(kind, tmp_path / "store", record_codecs=DEFAULT_RECORD_CODECS)


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_governance_and_stale_versions_refuse_generic_publications(kind, tmp_path):
    with closing(opened(kind, tmp_path / "store")) as store:
        store.put_record("r", value(), expected_version=0)
        before = store.change_cursor
        with pytest.raises(VersionConflictError): store.put_record("r", value(2), expected_version=0)
        store.configure_security(require_commits=True)
        with pytest.raises(PermissionError, match="cannot govern"):
            store.put_record("r", value(2))
        assert store.change_cursor == before


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_encryption_projection_and_structural_queries_keep_generic_types_distinct(kind, tmp_path):
    pytest.importorskip("cryptography")
    from rexgraph.io.security import StaticKeyProvider
    with closing(opened(kind, tmp_path / "store", compression=BlobCodecSpec("zlib"))) as store:
        store.configure_security(key_id="key", keys=StaticKeyProvider({"key": b"k"*32}), metadata_fields=["kept"])
        store.put_record("r", value(), meta={"kept": Fraction(1, 7), "drop": "private"}, tags=["value"])
        assert pack_value(store.get("r")) == pack_value(value())
        assert store.get_record("r").meta == {"kept": Fraction(1, 7)}
        assert not store.get_record("r").is_complex
        assert [r.id for r in store.query(record_type="NativeValue")] == ["r"]
        assert store.query(record_type="RexGraph") == []
        assert [r.id for r in store.query(tags_any=["value"])] == ["r"]
        for filter in ({"min_nV": 0}, {"max_betti1": 0}, {"has_voids": False}):
            assert store.query(**filter) == []


def test_unknown_persisted_provider_never_imports_a_module_and_refuses_before_publication(monkeypatch):
    import builtins
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("untrusted"):
            raise AssertionError("persisted provider name triggered an import")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    reference = CodecRef("untrusted.module.provider", 1)
    with closing(MemoryStore(record_codecs=(*DEFAULT_RECORD_CODECS, reference))) as store:
        with pytest.raises(ValueError, match="installed capability"):
            store.put_record("r", value(), codec=reference)
        assert store.change_cursor.sequence == 0
        codec = RecordCodec(reference, "TestValue", pack_value, unpack_value)
        register_record_codec(codec)
        try:
            store.put_record("published", value(), codec=reference)
        finally:
            unregister_record_codec(reference)
        with pytest.raises(ValueError, match="installed capability"):
            store.get("published")


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_custom_installed_codecs_are_bounded_canonical_and_not_transplanted(kind, tmp_path):
    reference = CodecRef("test.small-value", 1)
    codec = RecordCodec(reference, "SmallValue", pack_value, unpack_value, max_bytes=100)
    register_record_codec(codec)
    try:
        with closing(opened(kind, tmp_path / "store", record_codecs=(*DEFAULT_RECORD_CODECS, reference))) as store:
            store.put_record("small", Fraction(1, 7), codec=reference)
            before = store.change_cursor
            with pytest.raises(ValueError, match="byte limit"):
                store.put_record("large", "x"*101, codec=reference)
            assert store.change_cursor == before
            unregister_record_codec(reference)
            with pytest.raises(ValueError, match="installed capability"): store.get("small")
            assert store.change_cursor == before
    finally:
        unregister_record_codec(reference)


def test_payload_substitution_is_refused_for_generic_values_even_without_graph_verification():
    with closing(MemoryStore()) as store:
        store.put_record("a", value())
        store.put_record("b", value(2))
        store._blobs[("a", 1)] = store._blobs[("b", 1)]
        with pytest.raises(ValueError, match="published content digest"):
            store.get("a", verify=False)


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
@pytest.mark.parametrize("bad", [object(), float("nan"), float("inf")])
def test_unsupported_native_values_leave_the_published_prefix_unchanged(kind, bad, tmp_path):
    with closing(opened(kind, tmp_path / "store")) as store:
        store.put_record("r", value())
        before = store.change_cursor
        with pytest.raises((TypeError, ValueError)):
            store.put_record("r", {"invalid": bad})
        assert store.change_cursor == before and pack_value(store.get("r")) == pack_value(value())


def test_legacy_storage_refuses_generic_records_before_writing(tmp_path):
    with closing(FileStore(str(tmp_path / "legacy"), read_only=False)) as store:
        with pytest.raises(ValueError, match="native record-state"):
            store.put_record("generic", value())
        assert store.list() == []


@pytest.mark.parametrize("kind", ["local", "sql", "object-file", "object-memory"])
def test_independent_handles_refuse_stale_generic_version_preconditions(kind, tmp_path):
    with closing(opened(kind, tmp_path / "store")) as owner:
        with closing(opened(kind, tmp_path / "store")) as observer:
            owner.put_record("r", value(), expected_version=0)
            with pytest.raises(VersionConflictError): observer.put_record("r", value(2), expected_version=0)
            assert observer.get_record("r").version == 1
            observer.put_record("r", value(2), expected_version=1)
            assert owner.get_record("r").version == 2


def test_noncanonical_registered_payload_refuses_before_publication():
    reference = CodecRef("test.noncanonical", 1)
    codec = RecordCodec(reference, "Noncanonical", lambda value: b"always", lambda raw: raw, max_bytes=20)
    # The installed encoder changes representation when given its decoded value.
    def encode(value): return b"initial" if isinstance(value, int) else b"decoded"
    codec = replace(codec, encode=encode)
    register_record_codec(codec)
    try:
        with closing(MemoryStore(record_codecs=(*DEFAULT_RECORD_CODECS, reference))) as store:
            with pytest.raises(ValueError, match="not canonical"):
                store.put_record("r", 1, codec=reference)
            assert store.change_cursor.sequence == 0
    finally:
        unregister_record_codec(reference)


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_graph_family_analysis_skips_native_receipts_and_values(kind, tmp_path, monkeypatch):
    from rcdb import cluster_complexes, find_similar
    import rcdb.core as core
    graph = RexGraph.from_graph([0], [1])
    with closing(opened(kind, tmp_path / "store")) as store:
        store.put("graph", graph, analytics=False, meta={"vertex_labels": ["a", "b"]})
        store.put_record("value", value(), meta={"vertex_labels": ["a", "b"]})
        def score(candidate, *args):
            assert isinstance(candidate, RexGraph)
            return {"n_shared": 2, "score": 1.0, "kappa_mean": .5, "context_size": 2}
        monkeypatch.setattr(core, "_SIMILARITY_HOOK", score)
        assert [r["id"] for r in find_similar(store, graph, ["a", "b"])] == ["graph"]
        assert cluster_complexes(store)["n"] == 1


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
@pytest.mark.parametrize("scalar", [None, Absent, False, 0, (), [], {}])
def test_published_empty_and_absent_values_are_not_missing_record_payloads(kind, scalar, tmp_path):
    with closing(opened(kind, tmp_path / "store")) as store, closing(MemoryStore()) as target:
        record = store.put_record("value", scalar, tx_time=10.0)
        selected = store.read_record("value")
        assert selected is not None and pack_value(selected.value) == pack_value(scalar)
        assert selected.state_digest == record.envelope.object_digest
        assert store.read_record("missing") is None
        copied = copy_record(store, target, record, return_receipt=True,
                             tx_time=record.tx_from, preserve_signature=True)
        assert copied.source_digest == copied.destination_digest
        assert target.state_digest() == store.state_digest()
