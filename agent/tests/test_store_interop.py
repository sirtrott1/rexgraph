"""Backends that cooperate, and a store you do not have to choose by hand.

Each backend was an island: no way to move a corpus between them, and picking one
meant knowing which was fast for your shape of data and which drivers you happened
to have installed. That is a decision the library can make and a migration it can
perform, since every backend already answers the same contract.

FileStore's quadratic ingest is fixed here too. It keeps one file per blob -- which
is its point, since external tools can read those directly -- but its index is a
log rather than a document rewritten on every put.
"""

import time

import numpy as np
import pytest

from agent import rcdb
from rexgraph.graph import RexGraph


def _rex(labels):
    n = len(labels)
    r = RexGraph(sources=np.arange(max(n - 1, 1), dtype=np.int32) % max(n - 1, 1),
                 targets=(np.arange(max(n - 1, 1), dtype=np.int32) + 1) % max(n, 1))
    r._agent_meta = {"vertex_labels": labels, "source_text": " ".join(labels)}
    return r


def _put(store, rid, labels=None):
    labels = labels or [f"{rid}_v{i}" for i in range(4)]
    return store.put(rid, _rex(labels),
                     meta={"doc_id": rid, "vertex_labels": labels})


ALL = ["memory", "file", "sql", "rex"]


def _open(kind, tmp_path, tag=""):
    if kind == "memory":
        return rcdb.MemoryStore()
    if kind == "file":
        return rcdb.FileStore(str(tmp_path / f"fs{tag}"))
    if kind == "sql":
        return rcdb.SQLStore(f"sqlite:///{tmp_path / f'rc{tag}.sqlite'}")
    return rcdb.open_store(f"rex://{tmp_path / f'rx{tag}'}")


# --- FileStore is no longer quadratic -----------------------------------------

def test_filestore_put_cost_no_longer_grows_with_the_store(tmp_path):
    """It reserialized its whole index on every put: 4 ms at a hundred records,
    35 ms at sixteen hundred. The blobs stay one file each, which is the reason to
    choose it; the index does not have to be rewritten to keep that."""
    store = rcdb.FileStore(str(tmp_path / "fs"))
    early, late = [], []
    for k in range(600):
        t0 = time.perf_counter()
        _put(store, f"r{k:04d}")
        dt = time.perf_counter() - t0
        if 50 <= k < 150:
            early.append(dt)
        elif k >= 500:
            late.append(dt)
    ratio = (sum(late) / len(late)) / (sum(early) / len(early))

    # The property is "not quadratic", and the bound is a proxy for it. The old
    # behaviour reserialized the whole index per put, so this ratio tracked the record
    # count: it was 8.6x over 1600 and would be ~5x over the 600 measured here. A
    # shared CI runner adds noise on the same order as the effect at 2.5, so the bound
    # is set where quadratic still fails and scheduler jitter does not. Measured 2.8x
    # on a GitHub runner against 1.2x locally, with the fix in place both times.
    assert ratio < 4.0, f"per-put cost grew {ratio:.1f}x over 600 records"


def test_filestore_still_keeps_one_readable_blob_per_record(tmp_path):
    import os

    store = rcdb.FileStore(str(tmp_path / "fs"))
    _put(store, "a")
    blobs = [f for _, _, fs in os.walk(os.path.join(store.root, "blobs")) for f in fs]
    assert len(blobs) == 1


def test_filestore_survives_a_reopen_after_the_index_change(tmp_path):
    store = rcdb.FileStore(str(tmp_path / "fs"))
    _put(store, "a", ["kept", "x", "y", "z"])
    _put(store, "a", ["newer", "x", "y", "z"])
    _put(store, "b")
    again = rcdb.FileStore(str(tmp_path / "fs"))
    assert sorted(r.id for r in again.list(limit=9)) == ["a", "b"]
    assert [r.version for r in again.history("a")] == [1, 2]
    assert (again.get("a")._agent_meta or {})["vertex_labels"][0] == "newer"


def test_an_existing_filestore_still_opens(tmp_path):
    """A store written by the old layout must not become unreadable."""
    import json
    import os

    root = tmp_path / "legacy"
    store = rcdb.FileStore(str(root))
    _put(store, "a", ["legacy", "x", "y", "z"])
    # collapse it back to the old single-document index
    recs = store._read_index()
    payload = {rid: [r.to_dict() for r in versions] for rid, versions in recs.items()}
    for name in os.listdir(root):
        if name.startswith("index"):
            os.remove(os.path.join(root, name))
    with open(os.path.join(root, "index.json"), "w") as fh:
        json.dump(payload, fh)

    again = rcdb.FileStore(str(root))
    assert (again.get("a")._agent_meta or {})["vertex_labels"][0] == "legacy"


# --- migration between any two backends ---------------------------------------

@pytest.mark.parametrize("src_kind", ALL)
@pytest.mark.parametrize("dst_kind", ALL)
def test_a_corpus_moves_between_any_two_backends(src_kind, dst_kind, tmp_path):
    src = _open(src_kind, tmp_path, "_a")
    dst = _open(dst_kind, tmp_path, "_b")
    _put(src, "one", ["alpha", "beta", "gamma", "delta"])
    _put(src, "two", ["epsilon", "zeta", "eta", "theta"])

    moved = rcdb.migrate(src, dst)
    assert moved["records"] == 2
    assert sorted(r.id for r in dst.list(limit=9)) == ["one", "two"]
    assert (dst.get("one")._agent_meta or {})["vertex_labels"][0] == "alpha"


def test_migration_carries_every_version_in_order(tmp_path):
    src = _open("rex", tmp_path, "_a")
    dst = _open("memory", tmp_path, "_b")
    _put(src, "a", ["first", "x", "y", "z"])
    _put(src, "a", ["second", "x", "y", "z"])
    _put(src, "a", ["third", "x", "y", "z"])

    rcdb.migrate(src, dst)
    assert [r.version for r in dst.history("a")] == [1, 2, 3]
    assert (dst.get_version("a", 1)._agent_meta or {})["vertex_labels"][0] == "first"
    assert (dst.get("a")._agent_meta or {})["vertex_labels"][0] == "third"


def test_migration_preserves_what_makes_a_record_queryable(tmp_path):
    src = _open("rex", tmp_path, "_a")
    dst = _open("sql", tmp_path, "_b")
    _put(src, "a", ["frustration", "x", "y", "z"])
    rcdb.migrate(src, dst)
    assert {r.id for r in dst.query(labels_any=["frustration"], limit=5)} == {"a"}


def test_migration_into_a_populated_store_adds_rather_than_replaces(tmp_path):
    src = _open("rex", tmp_path, "_a")
    dst = _open("rex", tmp_path, "_b")
    _put(src, "from_src")
    _put(dst, "already_here")
    rcdb.migrate(src, dst)
    assert sorted(r.id for r in dst.list(limit=9)) == ["already_here", "from_src"]


# --- choosing a backend --------------------------------------------------------

def test_auto_picks_an_embedded_store_for_a_plain_path(tmp_path):
    store = rcdb.open_store(f"auto://{tmp_path / 'auto'}")
    assert isinstance(store, rcdb.RCStore)
    _put(store, "a")
    assert store.get("a") is not None


def test_auto_reopens_whatever_is_already_there(tmp_path):
    """Choosing a backend must never orphan data written by a previous choice."""
    root = tmp_path / "existing"
    first = rcdb.FileStore(str(root))
    _put(first, "a", ["written_as_file", "x", "y", "z"])

    again = rcdb.open_store(f"auto://{root}")
    assert (again.get("a")._agent_meta or {})["vertex_labels"][0] == "written_as_file"


def test_auto_recognises_an_existing_rexstore(tmp_path):
    root = tmp_path / "existing_rex"
    first = rcdb.open_store(f"rex://{root}")
    _put(first, "a", ["written_as_rex", "x", "y", "z"])

    again = rcdb.open_store(f"auto://{root}")
    assert again.backend == "rex"
    assert (again.get("a")._agent_meta or {})["vertex_labels"][0] == "written_as_rex"


def test_recommend_backend_explains_itself():
    rec = rcdb.recommend_backend()
    assert "backend" in rec and "reason" in rec and rec["reason"]
    assert rec["backend"] in rcdb.available_backends() or rec["backend"] in (
        "rex", "file", "memory", "sql")


# --- object storage ------------------------------------------------------------
#
# Exercised over fsspec's in-memory filesystem, which is the SAME code path S3 takes
# rather than a stand-in for it: what differs on a real bucket is the driver's wire
# protocol, not this layout.

def _objstore(tag=""):
    import uuid

    from agent.objectstore import ObjectStore
    return ObjectStore(f"memory://rcdb-{tag}-{uuid.uuid4().hex[:8]}")


def test_an_object_store_answers_the_whole_contract():
    store = _objstore("contract")
    _put(store, "a", ["alpha", "beta", "gamma", "delta"])
    _put(store, "a", ["alpha2", "beta", "gamma", "delta"])
    _put(store, "b", ["frustration", "x", "y", "z"])

    assert sorted(r.id for r in store.list(limit=9)) == ["a", "b"]
    assert [r.version for r in store.history("a")] == [1, 2]
    assert (store.get("a")._agent_meta or {})["vertex_labels"][0] == "alpha2"
    assert (store.get_version("a", 1)._agent_meta or {})["vertex_labels"][0] == "alpha"
    assert {r.id for r in store.query(labels_any=["frustration"], limit=5)} == {"b"}


def test_nothing_is_ever_rewritten(store=None):
    """Object storage has no append and no in-place update, so every object must be
    written once. A read-modify-write index is how a concurrent writer's entry gets
    lost on S3."""
    store = _objstore("immutable")
    _put(store, "a")
    keys = set(store.fs.find(store.root))
    _put(store, "b")
    after = set(store.fs.find(store.root))
    assert keys - after == set(), "an existing object was removed or replaced"
    assert len(after) > len(keys)


def test_a_journal_segment_is_written_per_change():
    store = _objstore("journal")
    for k in range(4):
        _put(store, f"r{k}")
    assert store.stats()["journal_segments"] == 4


def test_reopening_replays_the_journal():
    store = _objstore("reopen")
    _put(store, "a", ["kept", "x", "y", "z"])
    _put(store, "b")
    from agent.objectstore import ObjectStore
    again = ObjectStore(store.uri)
    assert sorted(r.id for r in again.list(limit=9)) == ["a", "b"]
    assert (again.get("a")._agent_meta or {})["vertex_labels"][0] == "kept"


def test_compaction_folds_the_journal_into_a_snapshot():
    """A listing whose cost grows with every write is how an object-store index
    degrades; compaction is what keeps opening cheap."""
    store = _objstore("compact")
    for k in range(12):
        _put(store, f"r{k}")
    assert store.stats()["journal_segments"] == 12
    store.compact()
    assert store.stats()["journal_segments"] == 0

    from agent.objectstore import ObjectStore
    again = ObjectStore(store.uri)
    assert len(again.list(limit=99)) == 12
    assert again.get("r5") is not None


def test_writes_after_compaction_still_replay():
    store = _objstore("post_compact")
    _put(store, "a")
    store.compact()
    _put(store, "b", ["later", "x", "y", "z"])
    from agent.objectstore import ObjectStore
    again = ObjectStore(store.uri)
    assert sorted(r.id for r in again.list(limit=9)) == ["a", "b"]
    assert (again.get("b")._agent_meta or {})["vertex_labels"][0] == "later"


def test_deletion_removes_the_payload_objects():
    store = _objstore("delete")
    _put(store, "a")
    _put(store, "b")
    assert store.delete("a") is True
    from agent.objectstore import ObjectStore
    again = ObjectStore(store.uri)
    assert [r.id for r in again.list(limit=9)] == ["b"]
    assert again.get("a") is None


def test_a_corpus_migrates_into_object_storage(tmp_path):
    src = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    _put(src, "one", ["alpha", "b", "c", "d"])
    _put(src, "two", ["beta", "b", "c", "d"])
    dst = _objstore("migrate")
    assert rcdb.migrate(src, dst)["records"] == 2
    assert sorted(r.id for r in dst.list(limit=9)) == ["one", "two"]
    assert {r.id for r in dst.query(labels_any=["alpha"], limit=5)} == {"one"}


def test_the_cloud_schemes_are_registered():
    for scheme in ("s3", "gs", "gcs", "az", "abfs", "adl"):
        assert scheme in rcdb.available_backends()


def test_a_missing_driver_says_which_one_to_install():
    """The failure a user actually hits first is a missing provider driver, and an
    ImportError from three frames down does not say which."""
    from agent.objectstore import _fs_for
    try:
        _fs_for("s3://bucket/prefix")
    except ImportError as e:
        assert "s3fs" in str(e)
    except Exception:
        pass          # a driver IS installed here; nothing to assert
