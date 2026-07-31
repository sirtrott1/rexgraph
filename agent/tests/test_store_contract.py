"""The RCStore contract, run against every backend.

Each backend had its own tests, so what a store OWES a caller was implicit and the
backends drifted: only SQLStore indexed labels, only some honoured as_of on the
collection accessors, and FileStore's ingest was quadratic without anything saying
it should not be. This is the contract itself -- a new backend is finished when it
passes.
"""

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
        return rcdb.MemoryStore()
    if kind == "file":
        return rcdb.FileStore(str(tmp_path / "fs"))
    if kind == "sql":
        return rcdb.SQLStore(f"sqlite:///{tmp_path / 'rc.sqlite'}")
    return rcdb.open_store(f"rex://{tmp_path / 'rx'}")


# --- identity and round-trip ---------------------------------------------------

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


# --- versions ------------------------------------------------------------------

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


# --- bitemporal ----------------------------------------------------------------

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


# --- structural and vocabulary query ------------------------------------------

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


# --- deletion ------------------------------------------------------------------

def test_delete_removes_every_version(store):
    _put(store, "a")
    _put(store, "a")
    assert store.delete("a") is True
    assert store.get("a") is None
    assert store.history("a") == []
    assert [r.id for r in store.list(limit=10)] == []


def test_deleting_what_is_not_there_is_false(store):
    assert store.delete("nope") is False


# --- durability ----------------------------------------------------------------

def test_a_persistent_store_survives_being_reopened(store, tmp_path):
    if isinstance(store, rcdb.MemoryStore):
        pytest.skip("memory is not persistent by construction")
    _put(store, "a", labels=["kept"] * 6)
    reopened = type(store)(store.root) if hasattr(store, "root") else \
        rcdb.open_store(store.uri) if hasattr(store, "uri") else None
    if reopened is None:
        reopened = rcdb.SQLStore(store.conn_str)
    assert (reopened.get("a")._agent_meta or {})["vertex_labels"] == ["kept"] * 6


def test_stats_report_something_sane(store):
    _put(store, "a")
    s = store.stats()
    assert isinstance(s, dict)


# --- what the embedded backend exists for -------------------------------------

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
    assert ratio < 2.0, f"per-put cost grew {ratio:.1f}x over 600 records"


def test_the_store_is_a_fixed_number_of_files(tmp_path):
    """One file per record is what makes a network filesystem -- which is what a
    cloud VM has -- the bottleneck."""
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
