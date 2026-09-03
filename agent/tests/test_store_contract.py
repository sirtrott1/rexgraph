"""The RCStore contract, run against every backend.

Each backend had its own tests, so what a store OWES a caller was implicit and the
backends drifted: only SQLStore indexed labels, only some honoured as_of on the
collection accessors, and FileStore's ingest was quadratic without anything saying
it should not be. This is the contract itself: a new backend is finished when it
passes.
"""

import json
import os
import shutil
import time

import numpy as np
import pytest

from agent import rcdb
from rexgraph.graph import RexGraph


def _rex(n=6, labels=None):
    r = RexGraph(sources=np.arange(n - 1, dtype=np.int32),
                 targets=np.arange(1, n, dtype=np.int32))
    r._agent_meta = {"vertex_labels": labels or [f"v{i}" for i in range(n)],
                     "source_text": "hello"}
    return r


def _put(store, rid, labels=None, **kw):
    labels = labels or [f"v{i}" for i in range(6)]
    return store.put(rid, _rex(labels=labels),
                     meta={"doc_id": rid, "vertex_labels": labels}, **kw)


@pytest.fixture(params=["memory", "file", "sql", "rex"])
def store(request, tmp_path):
    kind = request.param
    if kind == "memory":
        st = rcdb.MemoryStore()
    elif kind == "file":
        st = rcdb.FileStore(str(tmp_path / "fs"))
    elif kind == "sql":
        st = rcdb.SQLStore(f"sqlite:///{tmp_path / 'rc.sqlite'}")
    else:
        st = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    # yield, not return: a SQL store holds a connection pool that stays open until the
    # collector reaches it unless the owner closes it. Every branch goes through one
    # yield so the teardown cannot be skipped by the backend that happens to be chosen.
    yield st
    st.close()


# identity and round-trip

def test_a_stored_complex_comes_back(store):
    _put(store, "a")
    rex = store.get("a")
    assert rex is not None and int(rex.nE) == 5


def test_the_agent_metadata_rides_with_it(store):
    _put(store, "a", labels=["x", "y", "z", "p", "q", "r"])
    meta = store.get("a")._agent_meta or {}
    assert meta["vertex_labels"] == ["x", "y", "z", "p", "q", "r"]
    assert meta["source_text"] == "hello"


def test_a_missing_id_is_none_not_an_exception(store):
    assert store.get("nope") is None
    assert store.get_record("nope") is None


def test_the_record_carries_a_queryable_signature(store):
    _put(store, "a")
    sig = store.get_record("a").signature
    assert sig["nE"] == 5 and sig["nV"] == 6
    assert sig["n_labels"] == 6


# versions

def test_versions_append_rather_than_replace(store):
    _put(store, "a")
    _put(store, "a", labels=["q"] * 6)
    hist = store.history("a")
    assert [r.version for r in hist] == [1, 2]


def test_get_returns_the_latest_version(store):
    _put(store, "a")
    _put(store, "a", labels=["later"] * 6)
    assert (store.get("a")._agent_meta or {})["vertex_labels"] == ["later"] * 6


def test_an_older_version_is_still_reachable(store):
    _put(store, "a", labels=["first"] * 6)
    _put(store, "a", labels=["second"] * 6)
    old = store.get_version("a", 1)
    assert (old._agent_meta or {})["vertex_labels"] == ["first"] * 6


# bitemporal

def test_as_of_reads_the_store_as_it_stood(store):
    _put(store, "a", labels=["first"] * 6)
    time.sleep(0.02)
    mid = time.time()
    time.sleep(0.02)
    _put(store, "a", labels=["second"] * 6)
    assert (store.get("a", as_of=mid)._agent_meta or {})["vertex_labels"] == ["first"] * 6


def test_list_as_of_predates_everything_is_empty(store):
    _put(store, "a")
    assert store.list(limit=10, as_of=1.0) == []


def test_query_as_of_applies_to_the_version_that_was_current(store):
    _put(store, "a", labels=["alpha"] * 6)
    time.sleep(0.02)
    mid = time.time()
    time.sleep(0.02)
    _put(store, "a", labels=["omega"] * 6)
    assert {r.id for r in store.query(labels_any=["alpha"], limit=5, as_of=mid)} == {"a"}
    assert store.query(labels_any=["alpha"], limit=5) == []


# structural and vocabulary query

def test_labels_any_selects_by_vocabulary(store):
    _put(store, "a", labels=["frustration"] * 6)
    _put(store, "b", labels=["persistence"] * 6)
    assert {r.id for r in store.query(labels_any=["frustration"], limit=9)} == {"a"}


def test_labels_any_is_case_insensitive(store):
    _put(store, "a", labels=["Frustration"] * 6)
    assert {r.id for r in store.query(labels_any=["frustration"], limit=9)} == {"a"}


def test_structural_predicates_compose_with_vocabulary(store):
    _put(store, "a", labels=["frustration"] * 6)
    assert store.query(labels_any=["frustration"], min_nE=10 ** 9, limit=9) == []
    assert store.query(labels_any=["frustration"], min_nE=1, limit=9)


def test_limit_is_honoured(store):
    for i in range(5):
        _put(store, f"r{i}", labels=["shared"] * 6)
    assert len(store.query(labels_any=["shared"], limit=2)) == 2


def test_list_returns_current_versions_only(store):
    _put(store, "a")
    _put(store, "a")
    _put(store, "b")
    assert sorted(r.id for r in store.list(limit=10)) == ["a", "b"]


# deletion

def test_delete_removes_every_version(store):
    _put(store, "a")
    _put(store, "a")
    assert store.delete("a") is True
    assert store.get("a") is None
    assert store.history("a") == []
    assert [r.id for r in store.list(limit=10)] == []


def test_deleting_what_is_not_there_is_false(store):
    assert store.delete("nope") is False


# durability

def test_a_persistent_store_survives_being_reopened(store, tmp_path):
    if isinstance(store, rcdb.MemoryStore):
        pytest.skip("memory is not persistent by construction")
    _put(store, "a", labels=["kept"] * 6)
    reopened = type(store)(store.root) if hasattr(store, "root") else \
        rcdb.open_store(store.uri) if hasattr(store, "uri") else None
    if reopened is None:
        reopened = rcdb.SQLStore(store.conn_str)
    try:
        assert (reopened.get("a")._agent_meta or {})["vertex_labels"] == ["kept"] * 6
    finally:
        reopened.close()          # the second store is owned here, not by the fixture


def test_stats_report_something_sane(store):
    _put(store, "a")
    s = store.stats()
    assert isinstance(s, dict)


# what the embedded backend exists for

def _rexstore(tmp_path):
    return rcdb.open_store(f"rex://{tmp_path / 'rx'}")


def test_put_cost_does_not_grow_with_the_store(tmp_path):
    """FileStore reserialized its whole index on every put, so per-put cost rose
    with the record count: 4 ms at a hundred, 35 ms at sixteen hundred. Quadratic
    ingest is the difference between minutes and days at consortium scale."""
    store = _rexstore(tmp_path)
    early, late = [], []
    for k in range(600):
        t0 = time.perf_counter()
        _put(store, f"r{k:04d}", labels=[f"l{k}", "shared"])
        dt = time.perf_counter() - t0
        if 50 <= k < 150:
            early.append(dt)
        elif k >= 500:
            late.append(dt)
    ratio = (sum(late) / len(late)) / (sum(early) / len(early))

    # Same bound and same reasoning as test_store_interop's copy of this measurement.
    # The property is "not quadratic": the old behaviour reserialized the index per put
    # and grew 8.6x over 1600 records. A shared CI runner's jitter is the same size as
    # the effect at 2.0, and it measured 2.5x there against well under 2 locally, with
    # the fix in place both times. 4.0 is where quadratic still fails and noise does not.
    assert ratio < 4.0, f"per-put cost grew {ratio:.1f}x over 600 records"


def test_the_store_is_a_fixed_number_of_files(tmp_path):
    """One file per record is what makes a network filesystem (which is what a
    cloud VM has) the bottleneck."""
    import os

    store = _rexstore(tmp_path)
    for k in range(50):
        _put(store, f"r{k}")
    files = sum(len(f) for _, _, f in os.walk(store.root))
    assert files == 3, f"{files} files for 50 records"


def test_everything_survives_a_reopen(tmp_path):
    store = _rexstore(tmp_path)
    _put(store, "a", labels=["kept"] * 6)
    _put(store, "a", labels=["newer"] * 6)
    _put(store, "b")
    reopened = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    assert sorted(r.id for r in reopened.list(limit=9)) == ["a", "b"]
    assert [r.version for r in reopened.history("a")] == [1, 2]
    assert (reopened.get("a")._agent_meta or {})["vertex_labels"] == ["newer"] * 6


def test_a_torn_tail_costs_only_the_entry_being_written(tmp_path):
    """A crash mid-append can only damage the last entry. The length prefix makes
    that detectable, so recovery is truncation rather than repair."""
    store = _rexstore(tmp_path)
    _put(store, "a")
    _put(store, "b")
    log = store._records_path
    with open(log, "ab") as fh:                 # a half-written third entry
        fh.write(b"\x99\x00\x00\x00partial-json-that-never-finished")
    reopened = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    assert sorted(r.id for r in reopened.list(limit=9)) == ["a", "b"]


def test_compaction_reclaims_what_deletion_left_behind(tmp_path):
    store = _rexstore(tmp_path)
    for k in range(20):
        _put(store, f"r{k}")
    for k in range(15):
        store.delete(f"r{k}")
    before = store.stats()["blob_bytes"]
    store.compact()
    after = store.stats()
    assert after["blob_bytes"] < before
    assert after["n_records"] == 5
    assert store.get("r19") is not None


def test_compaction_preserves_history(tmp_path):
    store = _rexstore(tmp_path)
    _put(store, "a", labels=["first"] * 6)
    _put(store, "a", labels=["second"] * 6)
    store.compact()
    assert [r.version for r in store.history("a")] == [1, 2]
    assert (store.get_version("a", 1)._agent_meta or {})["vertex_labels"] == ["first"] * 6


def test_a_vocabulary_query_does_not_scan_every_record(tmp_path):
    """The inverted index is why this backend can answer a retrieval prefilter at
    all: it touches the ids carrying a term, not the store."""
    store = _rexstore(tmp_path)
    for k in range(200):
        _put(store, f"r{k}", labels=[f"only{k}"])
    seen = []
    real = store._select_version
    store._select_version = lambda v, a, b: (seen.append(1), real(v, a, b))[1]
    store.query(labels_any=["only7"], limit=5)
    store._select_version = real
    assert len(seen) <= 5, f"examined {len(seen)} records for a one-record answer"


# the index, as tensors

def test_an_indexed_store_reads_exactly_what_replay_would(tmp_path):
    """A faster open that answers differently is not an open. Every read is compared
    against the same store loaded by replaying its log."""
    import os as _os

    store = _rexstore(tmp_path)
    for k in range(40):
        _put(store, f"r{k:03d}", labels=[f"l{k}", "shared", f"x{k % 5}"])
    _put(store, "r000", labels=["revised", "shared"])
    store.write_index()

    indexed = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    assert indexed._index is not None, "the index was not used"

    _os.remove(indexed._index_path)
    replayed = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    assert replayed._index is None

    assert sorted(r.id for r in indexed.list(limit=99)) == \
           sorted(r.id for r in replayed.list(limit=99))
    assert [r.version for r in indexed.history("r000")] == \
           [r.version for r in replayed.history("r000")]
    assert {r.id for r in indexed.query(labels_any=["shared"], limit=99)} == \
           {r.id for r in replayed.query(labels_any=["shared"], limit=99)}
    assert (indexed.get("r000")._agent_meta or {})["vertex_labels"][0] == "revised"
    assert int(indexed.get("r007").nE) == int(replayed.get("r007").nE)


def test_writes_after_an_index_still_appear(tmp_path):
    """The index covers a prefix of the log; whatever came after it must replay."""
    store = _rexstore(tmp_path)
    _put(store, "before")
    store.write_index()
    _put(store, "after", labels=["later", "shared"])

    again = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    assert sorted(r.id for r in again.list(limit=9)) == ["after", "before"]
    assert {r.id for r in again.query(labels_any=["later"], limit=9)} == {"after"}


def test_a_record_is_parsed_only_when_it_is_asked_for(tmp_path):
    """The point of the offset table: opening must not parse every document."""
    store = _rexstore(tmp_path)
    for k in range(30):
        _put(store, f"r{k:03d}")
    store.write_index()

    again = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    holder = again._recs["r005"]
    assert holder.__class__.__name__ == "_LazyVersions"
    assert holder._cache is None, "documents were parsed at open"
    assert again.get_record("r005").version == 1
    assert holder._cache is not None


def test_a_corrupt_index_falls_back_to_the_log(tmp_path):
    """An index is derived. Losing it costs speed, never data."""
    store = _rexstore(tmp_path)
    _put(store, "a", labels=["kept", "shared"])
    store.write_index()
    with open(store._index_path, "wb") as fh:
        fh.write(b"not a safetensors file")

    again = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    assert again._index is None
    assert (again.get("a")._agent_meta or {})["vertex_labels"][0] == "kept"


def test_compaction_leaves_an_index_behind(tmp_path):
    import os as _os

    store = _rexstore(tmp_path)
    for k in range(10):
        _put(store, f"r{k}")
    store.compact()
    assert _os.path.exists(store._index_path)
    again = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    assert again._index is not None
    assert len(again.list(limit=99)) == 10


def test_a_deleted_record_does_not_come_back_from_the_index(tmp_path):
    store = _rexstore(tmp_path)
    _put(store, "a")
    _put(store, "b")
    store.write_index()
    store.delete("a")

    again = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    assert [r.id for r in again.list(limit=9)] == ["b"]
    assert again.get("a") is None


def test_a_corrupt_snapshot_with_a_log_still_reads(tmp_path):
    """While the log holds the same changes the snapshot is derived, so losing it
    costs speed and not data."""
    store = rcdb.FileStore(str(tmp_path / "fs"))
    _put(store, "a", labels=["kept"])
    store._write_index(store._read_index())          # snapshot, and the log is folded in
    store.put("b", _rex(), meta={"vertex_labels": ["later"]})   # a log after it
    assert os.path.exists(store._log_path)

    with open(store._index_path, "r+b") as fh:
        fh.seek(os.path.getsize(store._index_path) // 2)
        b = fh.read(1)
        fh.seek(os.path.getsize(store._index_path) // 2)
        fh.write(bytes([b[0] ^ 0xFF]))

    again = rcdb.FileStore(str(tmp_path / "fs"))
    assert [r.id for r in again.list(limit=9)] == ["b"], "the log's own records survive"


def test_a_corrupt_snapshot_without_a_log_raises(tmp_path):
    """`_write_index` removes the log once it has folded it in, so past a compaction
    the snapshot IS the index. It used to be caught and dropped, and the store then
    reported ZERO records over five intact blobs with no error raised."""
    store = rcdb.FileStore(str(tmp_path / "fs"))
    for k in range(5):
        _put(store, f"r{k}")
    store.compact()
    assert not os.path.exists(store._log_path)
    assert len(rcdb.FileStore(str(tmp_path / "fs")).list(limit=9)) == 5

    with open(store._index_path, "rb") as fh:
        header = int.from_bytes(fh.read(8), "little")
    size = os.path.getsize(store._index_path)
    off = 8 + header + (size - 8 - header) // 2       # inside the payload, so the DIGEST fires
    with open(store._index_path, "r+b") as fh:
        fh.seek(off)
        b = fh.read(1)
        fh.seek(off)
        fh.write(bytes([b[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="digest mismatch"):
        rcdb.FileStore(str(tmp_path / "fs")).list(limit=9)


def _legacy_log_line(rec):
    """One line of the json log a 1.0.x FileStore appended, op/id/record per line."""
    return json.dumps({"op": "put", "id": rec.id, "record": rec.to_dict()})


def test_a_1_0_x_store_reads_from_its_json_log(tmp_path):
    """1.0.x wrote index.log as json lines and index.json as the snapshot. Verified
    against stores built by v1.0.9 itself, both shapes, including a two-version id."""
    store = rcdb.FileStore(str(tmp_path / "fs"))
    _put(store, "a", labels=["first"])
    _put(store, "a", labels=["second"])
    _put(store, "b", labels=["other"])
    idx = store._read_index()
    lines = [_legacy_log_line(r) for versions in idx.values() for r in versions]

    legacy = tmp_path / "legacy"
    (legacy / "blobs").mkdir(parents=True)
    for f in os.listdir(os.path.join(str(tmp_path / "fs"), "blobs")):
        shutil.copy(os.path.join(str(tmp_path / "fs"), "blobs", f), legacy / "blobs" / f)
    (legacy / "index.log").write_text("\n".join(lines) + "\n")

    again = rcdb.FileStore(str(legacy))
    assert sorted(r.id for r in again.list(limit=9)) == ["a", "b"]
    assert len(again.history("a")) == 2, "both versions survive the legacy log"
    assert (again.get_record("a").meta or {})["vertex_labels"][0] == "second"


def test_migrating_a_1_0_x_store_renames_its_json_rather_than_dropping_it(tmp_path):
    """Compaction writes index.rexidx and renames the json aside. Losing the snapshot
    with a stale index.json still in place would reopen on the stale one."""
    store = rcdb.FileStore(str(tmp_path / "fs"))
    _put(store, "a", labels=["kept"])
    _put(store, "b", labels=["also"])
    idx = store._read_index()
    lines = [_legacy_log_line(r) for versions in idx.values() for r in versions]

    legacy = tmp_path / "legacy"
    (legacy / "blobs").mkdir(parents=True)
    for f in os.listdir(os.path.join(str(tmp_path / "fs"), "blobs")):
        shutil.copy(os.path.join(str(tmp_path / "fs"), "blobs", f), legacy / "blobs" / f)
    (legacy / "index.log").write_text("\n".join(lines) + "\n")

    rcdb.FileStore(str(legacy)).compact()
    assert (legacy / "index.rexidx").exists()
    assert (legacy / "index.log.migrated").exists(), "renamed, not deleted"
    assert not (legacy / "index.log").exists()
    assert sorted(r.id for r in rcdb.FileStore(str(legacy)).list(limit=9)) == ["a", "b"]


def test_replaying_a_legacy_log_over_a_snapshot_does_not_duplicate_versions(tmp_path):
    """If migration is interrupted after the snapshot is written and before the json is
    renamed, the legacy log is read again on top of a snapshot that already holds it.
    `_LazyIndex._build` drops a snapshot record whose version a log record repeats, so
    the replay is idempotent rather than additive."""
    store = rcdb.FileStore(str(tmp_path / "fs"))
    _put(store, "a", labels=["one"])
    _put(store, "a", labels=["two"])
    idx = store._read_index()
    lines = [_legacy_log_line(r) for versions in idx.values() for r in versions]

    legacy = tmp_path / "legacy"
    (legacy / "blobs").mkdir(parents=True)
    for f in os.listdir(os.path.join(str(tmp_path / "fs"), "blobs")):
        shutil.copy(os.path.join(str(tmp_path / "fs"), "blobs", f), legacy / "blobs" / f)
    (legacy / "index.log").write_text("\n".join(lines) + "\n")

    rcdb.FileStore(str(legacy)).compact()
    shutil.move(str(legacy / "index.log.migrated"), str(legacy / "index.log"))

    again = rcdb.FileStore(str(legacy))
    assert len(again.history("a")) == 2, "the replay must not append a second copy"
    assert (again.get_record("a").meta or {})["vertex_labels"][0] == "two"
