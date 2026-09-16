"""Corpus snapshots select versions before constructing accession incidence."""
from fractions import Fraction as Q

import pytest
from rcdb import open_store
from rexgraph import RexGraph


@pytest.fixture(params=["memory", "file", "rex", "sqlite", "object"])
def store(request, tmp_path):
    name = request.param
    if name == "sqlite":
        pytest.importorskip("sqlalchemy")
    if name == "object":
        pytest.importorskip("fsspec")
    uri = ("memory://" if name == "memory" else f"sqlite:///{tmp_path / 'db.sqlite'}"
           if name == "sqlite" else f"object+file://{tmp_path / 'objects'}"
           if name == "object" else f"{name}://{tmp_path / name}")
    if name == "object":
        from rcdb import ObjectStore
        value = ObjectStore(f"file://{tmp_path / 'objects'}")
    else:
        value = open_store(uri)
    yield value
    value.close()


def put(store, rid, tags, **kwargs):
    return store.put(rid, RexGraph.from_hypergraph([0, 2], [0, 1]),
                     tags=tags, analytics=False, **kwargs)


def test_visible_versions_and_retained_snapshot(store):
    put(store, "b", ["old"], valid_from=1, valid_to=2)
    first = store.get_record("b")
    old = store.corpus_snapshot(signature_fields=["tags"])
    put(store, "b", ["new"], valid_from=2)
    put(store, "a", ["new", "other"], valid_from=2)
    current = store.corpus_snapshot(signature_fields=["tags"])
    assert current.ids == ("a", "b") and current.versions == (1, 2)
    assert current.response(["new"])["scores"] == (Q(1, 2), Q(1, 2))
    assert current.response(["new"], reading="share")["scores"] == (Q(1, 4), Q(1, 2))
    assert old.response(["old"])["scores"] == (Q(1),)
    assert old.digest != current.digest
    at = store.corpus_snapshot(as_of=first.tx_from, signature_fields=["tags"])
    valid = store.corpus_snapshot(valid_at=1.5, signature_fields=["tags"])
    assert at.ids == valid.ids == ("b",)
    assert at.versions == valid.versions == (1,)
    assert store.corpus_snapshot(as_of=first.tx_from, valid_at=2.5).ids == ()
    store.delete("b")
    assert store.corpus_snapshot().ids == ("a",)
    assert current.ids == ("a", "b")


def test_projection_precedes_degree_and_response(store):
    put(store, "a", ["x"], meta={"vertex_labels": ["secret", "x"]})
    put(store, "b", ["x"], meta={"vertex_labels": ["secret"]})
    projected = store.corpus_snapshot(signature_fields=["tags"])
    assert projected.fields == ("tags",)
    assert projected.response(["secret"])["scores"] == (Q(0), Q(0))
    assert projected.response(["x"])["scores"] == (Q(1, 2), Q(1, 2))
    all_fields = store.corpus_snapshot()
    # The stored labels_sample adds another incidence of x at record a.
    assert all_fields.response(["x"])["scores"] == (Q(3, 4), Q(1, 4))
    assert all_fields.digest != projected.digest


def test_cache_reuses_only_owned_local_index(store, monkeypatch):
    put(store, "a", ["x"])
    first = store.corpus_snapshot()
    second = store.corpus_snapshot()
    assert (first is second) == store._cache_corpus
    if store._cache_corpus:
        monkeypatch.setattr(store, "list", lambda **kw: pytest.fail("cached snapshot rescanned"))
        assert store.corpus_snapshot() is first


def test_rex_compaction_append_log_and_reopening(tmp_path):
    uri = f"rex://{tmp_path / 'corpus'}"
    store = open_store(uri)
    put(store, "old", ["x"])
    store.compact()
    put(store, "tail", ["x"])
    put(store, "old", ["y"])
    captured = store.corpus_snapshot(signature_fields=["tags"])
    assert captured.ids == ("old", "tail")
    assert captured.versions == (2, 1)
    assert captured.response(["x"])["scores"] == (Q(0), Q(1))
    store.close()
    reopened = open_store(uri)
    try:
        assert reopened.corpus_snapshot(signature_fields=["tags"]).digest == captured.digest
    finally:
        reopened.close()


def test_sql_snapshot_observes_writes_from_another_handle(tmp_path):
    pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'corpus.sqlite'}"
    left, right = open_store(uri), open_store(uri)
    try:
        assert left.corpus_snapshot().ids == ()
        put(right, "fresh", ["x"])
        assert left.corpus_snapshot().ids == ("fresh",)
    finally:
        left.close()
        right.close()


def test_failed_write_drops_cache_without_changing_retained_snapshot(monkeypatch):
    store = open_store("memory://")
    put(store, "a", ["x"])
    before = store.corpus_snapshot()
    def fail(*args, **kwargs):
        raise RuntimeError("write failed")
    monkeypatch.setattr(store, "_put_impl", fail)
    with pytest.raises(RuntimeError):
        put(store, "b", ["y"])
    assert store._corpus_cache is None
    assert before.ids == store.corpus_snapshot().ids == ("a",)


def test_compaction_does_not_reuse_old_snapshot_offsets(tmp_path):
    store = open_store(f"rex://{tmp_path / 'offsets'}")
    put(store, "a", ["x"])
    put(store, "b", ["y"])
    store.write_index()
    store.delete("a")
    before = store.read_record("b")
    store.compact()
    assert store.corpus_snapshot().ids == ("b",)
    assert store.read_record("b").state_digest == before.state_digest


def test_legacy_log_append_and_compaction_keep_one_format(tmp_path):
    import json
    import struct
    from rcdb.index import LOG_MAGIC
    root = tmp_path / "legacy"
    uri = f"rex://{root}"
    store = open_store(uri)
    put(store, "old", ["x"])
    row = store.get_record("old")
    offset, length = store._blob_at[("old", 1)]
    entry = dict(op="put", id="old", version=1, signature=row.signature,
                 meta=row.meta, created=row.created, tx_from=row.tx_from,
                 blob_off=offset, blob_len=length)
    store.close()
    payload = json.dumps(entry).encode()
    path = root / "records.log"
    path.write_bytes(struct.pack("<I", len(payload)) + payload)
    store = open_store(uri)
    put(store, "new", ["y"])
    store.delete("old")
    store.close()
    store = open_store(uri)
    assert store.corpus_snapshot().ids == ("new",)
    store.compact()
    assert path.read_bytes().startswith(LOG_MAGIC)
    put(store, "tail", ["z"])
    store.close()
    store = open_store(uri)
    assert store.corpus_snapshot().ids == ("new", "tail")
    assert store.get("new").nE == store.get("tail").nE == 1
    store.close()
