"""Valid native payloads cannot be substituted across record addresses."""
from pathlib import Path
from dataclasses import replace

import pytest

from rcdb import FileStore, MemoryStore, ObjectStore, RexStore, SQLStore
from rexgraph import RexGraph


def opened(kind, root):
    if kind == "memory":
        return MemoryStore()
    root.mkdir(parents=True, exist_ok=True)
    if kind == "sql":
        pytest.importorskip("sqlalchemy")
        return SQLStore(f"sqlite:///{root / 'store.sqlite'}")
    if kind == "file":
        return FileStore(str(root), read_only=False)
    if kind == "rex":
        return RexStore(str(root), read_only=False)
    pytest.importorskip("fsspec")
    return ObjectStore(f"file://{root}", read_only=False)


def blob(store, rid, version):
    if store.backend == "memory":
        return store._blobs[(rid, version)]
    if store.backend in {"file", "rex"}:
        return store._read_blob(rid, version)
    if store.backend == "sql":
        from sqlalchemy import select
        with store.engine.connect() as connection:
            return connection.execute(select(store.table.c.blob).where(
                store.table.c.id == rid, store.table.c.version == version)).scalar_one()
    with store.fs.open(store._blob_key(rid, version), "rb") as stream:
        return stream.read()


def replace_blob(store, rid, version, payload):
    if store.backend == "memory":
        store._blobs[(rid, version)] = payload
    elif store.backend == "file":
        Path(store._blob_path(rid, version)).write_bytes(payload)
    elif store.backend == "rex":
        with open(store._blobs_path, "ab") as stream:
            offset = stream.tell()
            stream.write(payload)
        store._blob_at[(rid, version)] = (offset, len(payload))
    elif store.backend == "sql":
        with store.engine.begin() as connection:
            connection.execute(store.table.update().where(
                store.table.c.id == rid, store.table.c.version == version).values(blob=payload))
    else:
        with store.fs.open(store._blob_key(rid, version), "wb") as stream:
            stream.write(payload)


@pytest.mark.parametrize("kind", ["memory", "file", "rex", "sql", "object"])
@pytest.mark.parametrize("substitution", ["record", "version", "store"])
def test_published_payload_is_bound_to_store_record_and_version(kind, substitution, tmp_path):
    store = opened(kind, tmp_path / "first")
    other = None
    try:
        # Identical graph content still has different storage addresses. Comparing
        # graph digests alone cannot certify store + record + version ownership.
        graph = RexGraph.from_graph([0], [1])
        store.put("first", graph, analytics=False)
        store.put("first", graph, analytics=False)
        store.put("second", graph, analytics=False)
        if substitution == "record":
            replacement = blob(store, "second", 1)
        elif substitution == "version":
            replacement = blob(store, "first", 2)
        else:
            other = opened(kind, tmp_path / "other")
            other.put("first", graph, analytics=False)
            replacement = blob(other, "first", 1)
        replace_blob(store, "first", 1, replacement)
        with pytest.raises(ValueError, match="record|envelope|binding|payload|digest"):
            store.read_record("first", version=1)
        with pytest.raises(ValueError, match="record|envelope|binding|payload|digest"):
            store.get_version("first", 1)
    finally:
        store.close()
        if other is not None:
            other.close()


@pytest.mark.parametrize("kind", ["memory", "file", "rex", "sql", "object"])
def test_new_claim_refuses_legacy_payload_replacement_even_without_graph_verify(kind, tmp_path):
    from rcdb.core import serialize_complex
    store = opened(kind, tmp_path / "store")
    try:
        graph = RexGraph.from_graph([0], [1])
        record = store.put("r", graph, analytics=False)
        assert record.envelope.store_id == store.store_id
        replace_blob(store, "r", 1, serialize_complex(graph))
        with pytest.raises(ValueError, match="legacy payload|published content digest"):
            store.get("r", verify=False)
    finally:
        store.close()


@pytest.mark.parametrize("kind", ["file", "rex", "sql", "object"])
def test_bound_history_survives_reopen_and_compaction(kind, tmp_path):
    root = tmp_path / "store"
    store = opened(kind, root)
    graph = RexGraph.from_graph([0], [1])
    try:
        first = store.put("literal/id@1", graph, analytics=False, meta={"empty": (), "large": 2**90})
        second = store.put("literal/id@1", graph, analytics=False)
        identity = store.store_id
        if hasattr(store, "compact"):
            store.compact()
    finally:
        store.close()
    store = opened(kind, root)
    try:
        assert store.store_id == identity
        for expected in (first, second):
            snapshot = store.read_record(expected.id, version=expected.version)
            assert snapshot.record.envelope == expected.envelope
            assert snapshot.record.meta == expected.meta
            assert snapshot.value.nE == 1
        # Closing a predecessor is derived history, not a rewrite of publication.
        assert store.history(first.id)[0].tx_to == second.tx_from
    finally:
        store.close()


def test_metadata_edit_is_refused_and_content_substitution_cannot_be_reframed():
    from rcdb.envelope import RecordEnvelope
    from rcdb.core import serialize_complex
    import hashlib
    store = MemoryStore()
    graph = RexGraph.from_graph([0], [1])
    store.put("r", graph, analytics=False, meta={"source": "original"})
    stored = store._recs["r"][0]
    stored.meta["source"] = "edited"
    with pytest.raises(ValueError, match="metadata digest"):
        store.get("r")
    stored.meta["source"] = "original"
    envelope, _ = RecordEnvelope.from_bytes(blob(store, "r", 1))
    payload = serialize_complex(RexGraph.from_graph([0, 1], [1, 2]))
    replacement = replace(envelope, payload_digest=hashlib.sha256(payload).hexdigest(), payload_size=len(payload))
    # Even replacing the expected physical claim cannot override semantic identity.
    stored.envelope = replacement
    replace_blob(store, "r", 1, replacement.to_bytes(payload))
    with pytest.raises(ValueError, match="object identity|published content digest"):
        store.get("r", verify=False)
    # The engine's immutable physical claim refuses the replacement first.
    # Even a caller presenting rewritten physical metadata directly to the
    # envelope reader cannot replace its retained semantic object identity.
    from rcdb.engine import open_record
    with pytest.raises(ValueError, match="object identity"):
        open_record(stored, replacement.to_bytes(payload), store_id=store.store_id)
    store.close()


@pytest.mark.parametrize("kind", ["memory", "file", "rex", "sql", "object"])
def test_prepared_ingest_checks_state_without_reconstructing_a_graph(kind, tmp_path, monkeypatch):
    from rcdb.core import serialize_complex, structural_signature
    from rexgraph import state
    graph = RexGraph.from_graph([0], [1])
    payload = serialize_complex(graph)
    signature = structural_signature(graph, analytics=False)
    def forbidden(*args, **kwargs):
        pytest.fail("prepared ingest reconstructed the graph")
    store = opened(kind, tmp_path / "store")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(state, "from_state", forbidden)
            record = store.put_prepared("r", payload, signature)
            assert record.envelope.object_type == "RexGraph"
        assert store.read_record("r").value.nE == 1
        with pytest.raises(ValueError):
            store.put_prepared("r", b"invalid native state", signature)
        assert len(store.history("r")) == 1
    finally:
        store.close()


def test_recompression_preserves_the_published_envelope(tmp_path):
    store = FileStore(str(tmp_path), read_only=False)
    try:
        record = store.put("r", RexGraph.from_graph([0], [1]), analytics=False)
        before = blob(store, "r", 1)
        outcome = store.recompress(force=True)
        assert outcome["skipped"] == 1 and outcome["failed"] == 0
        assert blob(store, "r", 1) == before
        assert store.read_record("r").record.envelope == record.envelope
    finally:
        store.close()


def test_a_new_log_tail_cannot_replace_an_unreadable_base_snapshot(tmp_path):
    store = FileStore(str(tmp_path), read_only=False)
    store.put("base", RexGraph.from_graph([0], [1]), analytics=False)
    store.compact()
    store.put("tail", RexGraph.from_graph([0], [1]), analytics=False)
    store.close()
    Path(store._index_path).write_bytes(b"invalid authoritative snapshot")
    from safetensors import SafetensorError
    with pytest.raises((ValueError, SafetensorError)):
        FileStore(str(tmp_path), read_only=False)


def test_a_malformed_legacy_index_is_refused_instead_of_an_empty_store(tmp_path):
    (tmp_path / "index.json").write_text('{"r":')
    import json
    with pytest.raises(json.JSONDecodeError):
        FileStore(str(tmp_path), read_only=False)
