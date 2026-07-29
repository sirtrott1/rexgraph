import os
import time
import numpy as np
import pytest
from rexgraph.graph import RexGraph
from agent.agent.rcdb import ComplexRecord


def _rex(nedges=3):
    return RexGraph(sources=np.arange(nedges, dtype=np.int32),
                    targets=np.arange(1, nedges + 1, dtype=np.int32))


def test_complex_record_has_bitemporal_fields():
    r = ComplexRecord(id="a", signature={"nV": 4}, version=2, tx_from=100.0, tx_to=None,
                      valid_from=90.0, valid_to=None)
    d = r.to_dict()
    assert d["version"] == 2 and d["tx_from"] == 100.0 and d["tx_to"] is None
    assert d["valid_from"] == 90.0 and d["valid_to"] is None
    assert ComplexRecord.from_dict(d).version == 2          # round-trips
    # legacy dict (no temporal fields) backfills to version 1
    legacy = {"id": "b", "signature": {}, "created": 50.0, "meta": {}}
    lr = ComplexRecord.from_dict(legacy)
    assert lr.version == 1 and lr.tx_from == 50.0 and lr.tx_to is None


def test_select_version_bitemporal():
    from agent.agent.rcdb import MemoryStore
    st = MemoryStore()
    recs = [
        ComplexRecord(id="a", signature={}, version=1, tx_from=10.0, tx_to=20.0, valid_from=10.0, valid_to=20.0),
        ComplexRecord(id="a", signature={}, version=2, tx_from=20.0, tx_to=None, valid_from=20.0, valid_to=None),
    ]
    assert st._select_version(recs, None, None).version == 2          # current
    assert st._select_version(recs, 15.0, None).version == 1          # tx as_of in v1 window
    assert st._select_version(recs, 25.0, None).version == 2
    assert st._select_version(recs, 5.0, None) is None                # before any version
    assert st._select_version(recs, None, 12.0).version == 1          # valid_at in v1


def test_memorystore_append_only_and_time_travel():
    from agent.agent.rcdb import MemoryStore
    st = MemoryStore()
    st.put("g", _rex(3))                          # version 1
    st.put("g", _rex(5))                          # version 2 (more edges)
    hist = st.history("g")
    assert [r.version for r in hist] == [1, 2]
    assert hist[0].tx_to == hist[1].tx_from        # v1 closed at v2's tx_from
    assert hist[1].tx_to is None                    # v2 is live
    assert st.get("g").nE == 5                       # current
    assert st.get("g", as_of=hist[0].tx_from + 1e-9).nE == 3   # as of v1 window
    assert st.next_version("g") == 3
    assert st.get("missing") is None


def test_filestore_versions_and_legacy_read(tmp_path):
    from agent.agent.rcdb import FileStore
    st = FileStore(str(tmp_path / "db"))
    st.put("g", _rex(3)); st.put("g", _rex(4))
    assert [r.version for r in st.history("g")] == [1, 2]
    # reopen from disk (index.json persisted) and time-travel
    st2 = FileStore(str(tmp_path / "db"))
    assert st2.get("g").nE == 4
    v1 = st2.history("g")[0]
    assert st2.get("g", as_of=v1.tx_from + 1e-9).nE == 3
    # a legacy index.json {id -> record} reads as version 1 (write one by hand)
    import json
    legacy_dir = tmp_path / "legacy"; (legacy_dir / "blobs").mkdir(parents=True)
    # reuse a real blob by copying g@1
    import shutil
    shutil.copy(tmp_path / "db" / "blobs" / "g@1.safetensors", legacy_dir / "blobs" / "old.safetensors")
    rec = st2.history("g")[0].to_dict(); rec["id"] = "old"
    rec.pop("version"); rec.pop("tx_from"); rec.pop("tx_to"); rec.pop("valid_from"); rec.pop("valid_to")
    (legacy_dir / "index.json").write_text(json.dumps({"old": rec}))
    # but the blob path for a legacy record is blobs/old.safetensors (no @version)
    st3 = FileStore(str(legacy_dir))
    assert st3.history("old")[0].version == 1
    assert st3.get("old") is not None
    assert st3.get("old").nE == 3


def test_sqlstore_append_only_and_migration(tmp_path):
    from agent.agent.rcdb import open_store
    uri = "sqlite:///%s/rc.db" % tmp_path
    st = open_store(uri)
    st.put("g", _rex(3)); st.put("g", _rex(6))
    assert [r.version for r in st.history("g")] == [1, 2]
    assert st.get("g").nE == 6
    v1 = st.history("g")[0]
    assert st.get("g", as_of=v1.tx_from + 1e-9).nE == 3
    st.close()
    # reopen: migration is idempotent, data intact
    st2 = open_store(uri)
    assert st2.get("g").nE == 6
    st2.close()


@pytest.mark.parametrize("uri_factory", ["memory", "file", "sql"])
def test_change_feed_emitted(tmp_path, uri_factory):
    """put/delete must emit rcdb.put/rcdb.delete events on the shared activity journal,
    for every backend. Each parametrization uses its own id so events from other tests
    (and other parametrizations, sharing the same process-wide singleton log) cannot be
    mistaken for this one's."""
    from agent.agent.rcdb import open_store
    from agent import activity
    id_ = "cf_" + uri_factory
    uri = {"memory": "memory://", "file": "file://%s/db" % tmp_path,
           "sql": "sqlite:///%s/rc.db" % tmp_path}[uri_factory]
    log = activity.get_log()
    st = open_store(uri)
    st.put(id_, _rex(3))
    evs = [e for e in log.events(limit=50) if e.get("action") == "rcdb.put"
           and e.get("detail", {}).get("id") == id_]
    assert evs and evs[-1]["detail"]["version"] == 1 and evs[-1]["detail"]["nE"] == 3
    st.delete(id_)
    devs = [e for e in log.events(limit=50) if e.get("action") == "rcdb.delete"
            and e.get("detail", {}).get("id") == id_]
    assert devs
    if hasattr(st, "close"):
        st.close()


def test_lineage_incremental_no_full_scan():
    from agent.agent.rcdb import MemoryStore, put_version, lineage
    st = MemoryStore()
    put_version(st, "L", _rex(3)); put_version(st, "L", _rex(4)); put_version(st, "L", _rex(5))
    lin = lineage(st, "L")
    assert [x["version"] for x in lin] == [1, 2, 3]
    # unrelated ids do not affect L's lineage or version numbering
    put_version(st, "OTHER", _rex(2))
    assert [x["version"] for x in lineage(st, "L")] == [1, 2, 3]


def test_put_temporal_rex_payload():
    from agent.agent.rcdb import MemoryStore
    from rexgraph.graph import TemporalRex
    st = MemoryStore()
    trex = TemporalRex([])
    trex.append_snapshot(_rex(3)); trex.append_snapshot(_rex(5))
    st.put("seq", trex)
    got = st.get("seq")
    from rexgraph.graph import TemporalRex as TR
    assert isinstance(got, TR)
    assert got.reconstruct_at(1).nE == 5
    rec = st.get_record("seq")
    assert rec.signature.get("object_type") == "TemporalRex" and rec.signature.get("T") == 2


def test_sqlstore_legacy_idonly_pk_upgrades_and_versions(tmp_path):
    import sqlalchemy as sa
    from agent.agent.rcdb import open_store, serialize_complex
    dbfile = tmp_path / "legacy.db"
    uri = "sqlite:///%s" % dbfile
    # hand build a pre Slice C table: PRIMARY KEY (id) only, no temporal columns, one row
    eng = sa.create_engine(uri)
    with eng.begin() as c:
        c.exec_driver_sql("CREATE TABLE rc_complexes (id TEXT PRIMARY KEY, signature TEXT, "
                          "meta TEXT, created FLOAT, blob BLOB, nV INTEGER, nE INTEGER, "
                          "betti1 INTEGER, kappa_mean FLOAT, chain_valid BOOLEAN, source VARCHAR(256))")
        blob = serialize_complex(_rex(3))
        c.exec_driver_sql("INSERT INTO rc_complexes (id, signature, meta, created, blob, nV, nE) "
                          "VALUES ('leg', '{}', '{}', 100.0, :b, 4, 3)", {"b": blob})
    eng.dispose()
    # open through SQLStore: migration must repair the PK to composite (id, version)
    st = open_store(uri)
    assert st.get("leg").nE == 3                          # legacy row reads as version 1
    st.put("leg", _rex(6))                                # UPDATE the existing id, version 2 (was crashing)
    assert [r.version for r in st.history("leg")] == [1, 2]
    assert st.get("leg").nE == 6
    st.close()
    # reopen: PK already composite, migration is a no op, data intact
    st2 = open_store(uri)
    assert st2.get("leg").nE == 6
    assert [r.version for r in st2.history("leg")] == [1, 2]
    st2.close()


def test_trajectory_reports_signed_movement():
    from agent.agent.rcdb import MemoryStore, trajectory, drift
    st = MemoryStore()
    st.put("g", _rex(3)); st.put("g", _rex(5)); st.put("g", _rex(4))
    traj = trajectory(st, "g")
    assert [s["version"] for s in traj["versions"]] == [1, 2, 3]
    # nE goes 3 -> 5 (grew, +2) then 5 -> 4 (shrank, -1): the signed dnE trend
    steps = traj["steps"]
    assert steps[0]["d"]["nE"] == 2 and steps[1]["d"]["nE"] == -1
    # each step carries a relational match (how close consecutive versions are) in [0,1]
    assert 0.0 <= steps[0]["match"] <= 1.0
    d = drift(st, "g")
    assert "trajectory" in d or "steps" in d          # drift now carries signed movement


def test_get_ver_falls_back_for_backend_without_get_version():
    from agent.agent.rcdb import MemoryStore, _get_ver, RCStore
    class NoGetVersion(MemoryStore):
        backend = "nogv"
    # remove the override so it inherits the ABC stub path (simulate a custom backend)
    NoGetVersion.get_version = RCStore.get_version
    st = NoGetVersion()
    st.put("g", _rex(3)); st.put("g", _rex(5))
    # _get_ver must NOT raise NotImplementedError; it returns the right version via fallback
    r1 = _get_ver(st, "g", 1)
    assert r1 is not None and r1.nE == 3
    r2 = _get_ver(st, "g", 2)
    assert r2.nE == 5


# ---------------------------------------------------------------------------
# Task 10: comprehensive suite, parametrized over all three backends, plus a
# versioned-store dogfood test. Everything below exercises behavior already
# implemented in Tasks 1 to 9; the point is coverage and cross-backend proof,
# not new production code.
# ---------------------------------------------------------------------------

def _open(backend, tmp_path, name="db"):
    """Open a fresh store of the given kind, isolated per test via tmp_path."""
    from agent.agent.rcdb import open_store
    if backend == "memory":
        return open_store("memory://")
    if backend == "file":
        return open_store("file://%s/%s" % (tmp_path, name))
    if backend == "sql":
        return open_store("sqlite:///%s/%s.db" % (tmp_path, name))
    raise ValueError(backend)


_ALL_BACKENDS = ["memory", "file", "sql"]


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_versioning_and_time_travel_all_backends(backend, tmp_path):
    st = _open(backend, tmp_path)
    st.put("g", _rex(3))                          # version 1
    st.put("g", _rex(5))                          # version 2 (more edges)
    hist = st.history("g")
    assert [r.version for r in hist] == [1, 2]
    assert hist[0].tx_to == hist[1].tx_from        # v1 closed exactly when v2 opened
    assert hist[1].tx_to is None                    # v2 is the live row
    assert st.get("g").nE == 5                       # current read
    assert st.get("g", as_of=hist[0].tx_from + 1e-9).nE == 3   # time travel into v1
    assert st.next_version("g") == 3
    assert st.get("missing") is None
    st.close()


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_bitemporal_valid_time_windows_all_backends(backend, tmp_path):
    """put with an explicit backdated valid_from, get(valid_at=...), and a
    combined (as_of, valid_at) point read, over all three backends."""
    st = _open(backend, tmp_path)
    now = time.time()
    v1_valid_from, v1_valid_to = now - 1000.0, now - 500.0
    rec1 = st.put("bt", _rex(3), valid_from=v1_valid_from, valid_to=v1_valid_to)
    v2_valid_from = now - 500.0
    st.put("bt", _rex(5), valid_from=v2_valid_from, valid_to=None)

    # valid time only: a point inside v1's real world window reads v1's data
    assert st.get("bt", valid_at=now - 700.0).nE == 3
    # a point inside v2's (open ended) real world window reads v2's data
    assert st.get("bt", valid_at=now - 100.0).nE == 5
    # a point before either fact was ever true: no row satisfies the window
    assert st.get("bt", valid_at=now - 2000.0) is None

    # combined (as_of, valid_at): pin transaction time to when only v1 had
    # been recorded (before v2's put), then ask about v2's real world window,
    # which that transaction time has never heard of; no candidate satisfies
    # both selectors at once.
    only_v1_known = rec1.tx_from + 1e-6
    assert st.get("bt", as_of=only_v1_known, valid_at=now - 100.0) is None
    # same transaction time, but a valid_at inside v1's own window: satisfies both
    assert st.get("bt", as_of=only_v1_known, valid_at=now - 700.0).nE == 3
    st.close()


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_legacy_read_backfills_to_version_1_all_backends(backend, tmp_path):
    """A pre-Slice-C record shape (no version/tx_from/tx_to/valid_from/valid_to)
    reads back as version 1 on every backend."""
    import json
    from agent.agent.rcdb import MemoryStore, FileStore, open_store, ComplexRecord, serialize_complex
    rex = _rex(3)
    legacy_sig = {"nV": 4, "nE": 3}
    if backend == "memory":
        st = MemoryStore()
        rec = ComplexRecord.from_dict({"id": "old", "signature": legacy_sig,
                                       "created": 50.0, "meta": {}})
        st._recs["old"] = [rec]
        st._blobs[("old", 1)] = serialize_complex(rex)
    elif backend == "file":
        root = str(tmp_path / "db")
        os.makedirs(os.path.join(root, "blobs"), exist_ok=True)
        with open(os.path.join(root, "blobs", "old.safetensors"), "wb") as f:
            f.write(serialize_complex(rex))
        legacy = {"id": "old", "signature": legacy_sig, "created": 50.0, "meta": {}}
        with open(os.path.join(root, "index.json"), "w") as f:
            json.dump({"old": legacy}, f)
        st = FileStore(root)
    else:
        import sqlalchemy as sa
        dbfile = tmp_path / "legacy.db"
        uri = "sqlite:///%s" % dbfile
        eng = sa.create_engine(uri)
        with eng.begin() as c:
            c.exec_driver_sql(
                "CREATE TABLE rc_complexes (id TEXT PRIMARY KEY, signature TEXT, "
                "meta TEXT, created FLOAT, blob BLOB, nV INTEGER, nE INTEGER, "
                "betti1 INTEGER, kappa_mean FLOAT, chain_valid BOOLEAN, source VARCHAR(256))")
            blob = serialize_complex(rex)
            c.exec_driver_sql(
                "INSERT INTO rc_complexes (id, signature, meta, created, blob, nV, nE) "
                "VALUES ('old', '{}', '{}', 50.0, :b, 4, 3)", {"b": blob})
        eng.dispose()
        st = open_store(uri)
    hist = st.history("old")
    assert len(hist) == 1 and hist[0].version == 1
    assert hist[0].tx_to is None
    assert st.get("old") is not None
    assert st.get("old").nE == 3
    st.close()


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_incremental_version_index_all_backends(backend, tmp_path):
    """next_version/lineage are keyed per id (an O(1) lookup on that id's own
    chain), not a rescan of every stored complex: an unrelated id's writes
    never perturb another lineage's version numbering."""
    from agent.agent.rcdb import put_version, lineage
    st = _open(backend, tmp_path)
    put_version(st, "L", _rex(3)); put_version(st, "L", _rex(4)); put_version(st, "L", _rex(5))
    assert [x["version"] for x in lineage(st, "L")] == [1, 2, 3]
    put_version(st, "OTHER", _rex(2))
    assert [x["version"] for x in lineage(st, "L")] == [1, 2, 3]
    assert st.next_version("L") == 4
    st.close()


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_opt_in_temporalrex_payload_round_trip_all_backends(backend, tmp_path):
    """The TemporalRex payload is opt-in: put a TemporalRex, get it back as a
    TemporalRex (not a plain RexGraph), on every backend."""
    from rexgraph.graph import TemporalRex
    st = _open(backend, tmp_path)
    trex = TemporalRex([])
    trex.append_snapshot(_rex(3)); trex.append_snapshot(_rex(5))
    st.put("seq", trex)
    got = st.get("seq")
    assert isinstance(got, TemporalRex)
    assert got.reconstruct_at(0).nE == 3
    assert got.reconstruct_at(1).nE == 5
    rec = st.get_record("seq")
    assert rec.signature.get("object_type") == "TemporalRex" and rec.signature.get("T") == 2
    st.close()


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_trajectory_signed_movement_all_backends(backend, tmp_path):
    from agent.agent.rcdb import trajectory, drift
    st = _open(backend, tmp_path)
    st.put("g", _rex(3)); st.put("g", _rex(5)); st.put("g", _rex(4))
    traj = trajectory(st, "g")
    assert [s["version"] for s in traj["versions"]] == [1, 2, 3]
    steps = traj["steps"]
    # nE goes 3 -> 5 (grew, +2) then 5 -> 4 (shrank, -1): the signed dnE trend
    assert steps[0]["d"]["nE"] == 2 and steps[1]["d"]["nE"] == -1
    assert all(0.0 <= s["match"] <= 1.0 for s in steps)
    d = drift(st, "g")
    assert "trajectory_steps" in d
    assert d["trajectory_steps"][0]["d"]["nE"] == 2
    st.close()


# ---------------------------------------------------------------------------
# Slice C final-review fix: {base}@{version} display-id resolution as a
# fallback, plus the legacy meta.lineage scheme fallback for lineage/drift.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_display_id_resolves_via_get_all_backends(backend, tmp_path):
    """A lineage() display id "base@v" resolves through get/get_record as a
    fallback, on every backend, once the direct lookup misses."""
    from agent.agent.rcdb import lineage
    st = _open(backend, tmp_path)
    st.put("M", _rex(3))                          # version 1
    st.put("M", _rex(5))                          # version 2
    lin = lineage(st, "M")
    assert st.get("M@1").nE == 3
    assert st.get("M@2").nE == 5
    assert st.get_record("M@2").version == 2
    assert st.get("M@99") is None                 # nonexistent version
    assert st.get("nope@1") is None                # nonexistent lineage
    st.close()


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_compare_accepts_lineage_display_ids_all_backends(backend, tmp_path):
    """compare() takes the exact display ids lineage() hands back, on every
    backend, now that get/get_record resolve them (Fix A)."""
    from agent.agent.rcdb import lineage, compare
    st = _open(backend, tmp_path)
    st.put("M", _rex(3))
    st.put("M", _rex(5))
    lin = lineage(st, "M")
    c = compare(st, lin[0]["id"], lin[1]["id"])
    assert c is not None
    assert c["a"] == "M@1" and c["b"] == "M@2"
    st.close()


@pytest.mark.parametrize("backend", ["memory", "file"])
def test_lineage_legacy_meta_scheme_fallback(backend, tmp_path):
    """A pre Slice C store never wrote a native version chain under a bare
    lineage id; instead each version was its own record, id "L@1"/"L@2", each
    carrying meta["lineage"]. lineage()/drift() must still read it: history("L")
    is empty (no such id was ever put), so the legacy meta.lineage scan is the
    only source, and its output shape must match the old native-chain shape."""
    from agent.agent.rcdb import lineage, drift
    st = _open(backend, tmp_path)
    t0 = time.time()
    st.put("L@1", _rex(3),
           meta={"lineage": {"id": "L", "version": 1, "parent_version": None, "created": t0}})
    st.put("L@2", _rex(5),
           meta={"lineage": {"id": "L", "version": 2, "parent_version": 1, "created": t0 + 1}})
    lin = lineage(st, "L")
    assert [x["id"] for x in lin] == ["L@1", "L@2"]
    assert [x["version"] for x in lin] == [1, 2]
    d = drift(st, "L")
    assert len(d["versions"]) == 2
    assert len(d["trajectory"]) > 0                # the from->to diff step exists
    st.close()


def test_lineage_native_chain_unchanged():
    """Guard against the fallback disturbing the happy path: a normal
    put_version chain still returns exactly the same display-id rows."""
    from agent.agent.rcdb import MemoryStore, put_version, lineage
    st = MemoryStore()
    put_version(st, "N", _rex(3))
    put_version(st, "N", _rex(5))
    lin = lineage(st, "N")
    assert [x["id"] for x in lin] == ["N@1", "N@2"]
    assert [x["version"] for x in lin] == [1, 2]
    assert [x["parent_version"] for x in lin] == [None, 1]
    assert all("created" in x for x in lin)


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_get_real_at_id_still_wins_over_split(backend, tmp_path):
    """An id that literally contains "@" and is a REAL stored record must
    resolve to itself directly; the split fallback only fires on a miss."""
    st = _open(backend, tmp_path)
    st.put("real@1", _rex(7))
    got = st.get("real@1")
    assert got is not None and got.nE == 7
    rec = st.get_record("real@1")
    assert rec is not None and rec.version == 1
    st.close()


def test_dogfood_versioned_store_timetravel_and_trend(tmp_path):
    """Slice C dogfood: stream an evolving complex through a real (file-backed)
    versioned store, time travel to prior states, confirm the change feed rode
    the shared activity journal, and confirm trajectory shows the trend."""
    from agent.agent.rcdb import open_store, trajectory
    from agent import activity
    st = open_store("file://%s/db" % tmp_path)
    times = []
    for k in (2, 3, 4, 5, 4):
        rec = st.put("dev", _rex(k))
        times.append(rec.tx_from)
    assert st.get("dev").nE == 4                                  # current
    assert st.get("dev", as_of=times[0] + 1e-9).nE == 2          # time-travel to v1
    assert st.get("dev", as_of=times[2] + 1e-9).nE == 4          # v3
    traj = trajectory(st, "dev")
    assert [s["d"]["nE"] for s in traj["steps"]] == [1, 1, 1, -1]  # grew then shrank
    feed = [e for e in activity.get_log().events(limit=100)
            if e.get("action") == "rcdb.put" and e.get("detail", {}).get("id") == "dev"]
    assert len(feed) == 5
    st.close()
