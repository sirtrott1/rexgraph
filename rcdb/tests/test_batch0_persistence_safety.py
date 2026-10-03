"""Check record ownership, damaged persistence and explicit migration recovery."""
import json
from pathlib import Path

import pytest

from rcdb import FileStore, MemoryStore, ObjectStore, RexStore, SQLStore
from rexgraph.graph import RexGraph


def graph(n=2):
    return RexGraph.from_graph(list(range(n)), list(range(1, n + 1)))


def test_commit_names_and_delete_do_not_collide(tmp_path):
    store = FileStore(str(tmp_path), read_only=False)
    for rid in ("core/alpha", "core_alpha"):
        store.commit_mutation(rid, graph(), analytics=False)
    assert store._commit_path("core/alpha", 1) != store._commit_path("core_alpha", 1)
    assert store.verify_commits("core/alpha")
    assert store.verify_commits("core_alpha")
    store.delete("core/alpha")
    assert store.verify_commits("core_alpha")
    assert store.get("core_alpha") is not None


def test_unambiguous_legacy_commit_migrates_on_read(tmp_path):
    store = FileStore(str(tmp_path), read_only=False)
    store.put("core/alpha", graph(), analytics=False)
    legacy = tmp_path / "commits" / "core_alpha@1.rexpkg"
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_bytes(b"legacy-artifact")
    assert store._load_commit_bytes("core/alpha", 1) == b"legacy-artifact"
    assert Path(store._commit_path("core/alpha", 1)).read_bytes() == b"legacy-artifact"
    assert not legacy.exists()


def test_ambiguous_legacy_commit_is_refused_and_preserved(tmp_path):
    store = FileStore(str(tmp_path), read_only=False)
    for rid in ("core/alpha", "core_alpha"):
        store.put(rid, graph(), analytics=False)
    legacy = tmp_path / "commits" / "core_alpha@1.rexpkg"
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_bytes(b"ambiguous")
    with pytest.raises(ValueError, match="ambiguous"):
        store._load_commit_bytes("core/alpha", 1)
    assert legacy.read_bytes() == b"ambiguous"
    store._delete_commit_bytes("core/alpha", 1)
    assert legacy.read_bytes() == b"ambiguous"


def test_legacy_and_framed_logs_merge_and_deduplicate(tmp_path):
    store = FileStore(str(tmp_path), read_only=False)
    old = store.put("legacy", graph(), analytics=False)
    duplicate = store.put("shared", graph(), meta={"source": "framed"}, analytics=False)
    # Simulate an upgrade that left an older line log prefix beside new frames.
    legacy_copy = duplicate.to_dict()
    legacy_copy["meta"] = {"source": "legacy"}
    (tmp_path / "index.log").write_text("\n".join(json.dumps({
        "op": "put", "id": r["id"], "record": r,
    }) for r in (old.to_dict(), legacy_copy)) + "\n")
    # Keep 'legacy' only in the old log, and 'shared' in both logs.
    from rcdb import index
    Path(store._log_path).unlink()
    index.log_append(store._log_path, "put", "shared", duplicate)
    reopened = FileStore(str(tmp_path), read_only=False)
    assert {r.id for r in reopened.list()} == {"legacy", "shared"}
    assert len(reopened.history("shared")) == 1
    assert reopened.get_record("shared").meta["source"] == "framed"
    reopened.compact()
    assert (tmp_path / "index.log.migrated").exists()
    assert {r.id for r in FileStore(str(tmp_path), read_only=False).list()} == {"legacy", "shared"}


@pytest.mark.parametrize("backend", ["memory", "file", "rex", "sql", "object"])
def test_literal_version_shaped_id_wins(tmp_path, backend):
    if backend == "memory":
        store = MemoryStore()
    elif backend == "sql":
        pytest.importorskip("sqlalchemy")
        store = SQLStore(f"sqlite:///{tmp_path / 'store.sqlite'}")
    elif backend == "file":
        store = FileStore(str(tmp_path), read_only=False)
    elif backend == "rex":
        store = RexStore(str(tmp_path), read_only=False)
    else:
        pytest.importorskip("fsspec")
        store = ObjectStore(f"file://{tmp_path}", read_only=False)
    try:
        store.put("base", graph(2), analytics=False)
        store.put("base@1", graph(3), analytics=False)
        assert store.get("base@1").nE == 3
        assert store.get_version("base", 1).nE == 2
        # A literal id outside its time window must not become another id's version.
        assert store.get("base@1", as_of=0) is None
    finally:
        store.close()


@pytest.mark.parametrize("backend", ["file", "rex"])
def test_tombstone_survives_compaction_and_stale_payload(tmp_path, backend):
    cls = FileStore if backend == "file" else RexStore
    store = cls(str(tmp_path), read_only=False)
    store.put("deleted", graph(), analytics=False)
    if backend == "file":
        blob = Path(store._blob_path("deleted", 1))
        saved = blob.read_bytes()
    assert store.delete("deleted")
    if backend == "file":
        store.compact()
        blob.write_bytes(saved)
    else:
        store.write_index()
    reopened = cls(str(tmp_path))
    assert reopened.get("deleted") is None
    assert reopened.get_version("deleted", 1) is None
    assert reopened.history("deleted") == []
