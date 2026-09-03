import numpy as np
import pytest
from agent.rcdb import open_store
from agent.temporal_loop import DERIVED_TAG, ChangeEvent, ChangeSource

from rexgraph.graph import RexGraph


def _rex(src, tgt):
    return RexGraph(sources=np.asarray(src, np.int32), targets=np.asarray(tgt, np.int32))


def test_change_source_fetches_put_via_get_version():
    store = open_store("memory://")
    rec = store.put("svc", _rex([0, 1], [1, 2]))
    src = ChangeSource(store)
    pub = {"action": "rcdb.put", "scope": "network",
           "detail": {"id": "svc", "version": rec.version, "tags": []}}
    ev = src._event_from_pub(pub)
    assert isinstance(ev, ChangeEvent)
    assert ev.id == "svc" and ev.version == rec.version and ev.action == "rcdb.put"
    assert ev.rex is not None and ev.rex.nE == 2       # _rex([0,1],[1,2]) is two edges


def test_change_source_skips_derived_tag():
    store = open_store("memory://")
    # put REAL derived data so get_version would succeed: the skip must be the guard,
    # not an incidental get_version miss.
    rec = store.put("svc::online", _rex([0, 1], [1, 2]), tags=[DERIVED_TAG])
    # control (fresh source): the SAME id/version WITHOUT the tag genuinely delivers
    ctrl = ChangeSource(store)._event_from_pub(
        {"action": "rcdb.put", "scope": "network",
         "detail": {"id": "svc::online", "version": rec.version, "tags": []}})
    assert ctrl is not None and ctrl.rex is not None
    # with the derived tag present, the guard skips it (proven distinct from the control)
    guarded = ChangeSource(store)._event_from_pub(
        {"action": "rcdb.put", "scope": "network",
         "detail": {"id": "svc::online", "version": rec.version, "tags": [DERIVED_TAG]}})
    assert guarded is None


def test_change_source_skips_get_version_miss():
    store = open_store("memory://")
    src = ChangeSource(store)
    pub = {"action": "rcdb.put", "scope": "network",
           "detail": {"id": "ghost", "version": 99, "tags": []}}
    assert src._event_from_pub(pub) is None          # missing blob: skip, do not crash


def test_change_source_idempotent_on_duplicate():
    store = open_store("memory://")
    rec = store.put("svc", _rex([0, 1], [1, 2]))
    src = ChangeSource(store)
    pub = {"action": "rcdb.put", "scope": "network",
           "detail": {"id": "svc", "version": rec.version, "tags": []}}
    assert src._event_from_pub(pub) is not None
    assert src._event_from_pub(pub) is None          # same (id, version) not re-delivered


def test_change_source_emits_delete_as_removal():
    store = open_store("memory://")
    src = ChangeSource(store)
    pub = {"action": "rcdb.delete", "scope": "network",
           "detail": {"id": "svc", "version": 2, "tags": []}}
    ev = src._event_from_pub(pub)
    assert ev is not None and ev.action == "rcdb.delete" and ev.rex is None


from agent.temporal_loop import OnlineLoop, StepResult


def test_on_change_advances_and_returns_stepresult():
    store = open_store("memory://")
    calls = {}

    class MockLearner:
        def predict_then_observe(self, t, change, trex):
            calls["t"] = t
            return {"pred": None, "target": None, "error": 0.0, "updated": True}

    loop = OnlineLoop(store, learner=MockLearner())
    ev = ChangeEvent("svc", 1, "rcdb.put", _rex([0, 1], [1, 2]))
    res = loop.on_change(ev)
    assert isinstance(res, StepResult)
    assert res.t == 0 and res.id == "svc" and res.version == 1
    assert res.learn["updated"] is True and calls["t"] == 0
    assert res.wrote_back is None                    # write_back defaults off
    assert loop.history()[-1] is res


def test_on_change_native_learner_default_is_torch_free():
    store = open_store("memory://")
    loop = OnlineLoop(store)                          # no learner -> native GreensCochainField
    from rexgraph.flow.online import GreensCochainField
    assert isinstance(loop.learner, GreensCochainField)
    r1 = loop.on_change(ChangeEvent("svc", 1, "rcdb.put", _rex([0, 1], [1, 2])))
    r2 = loop.on_change(ChangeEvent("svc", 2, "rcdb.put", _rex([0, 1, 2], [1, 2, 3])))
    assert r1.t == 0 and r2.t == 1
    assert r2.change.added.size >= 1                  # second snapshot added an edge


def test_on_change_delete_drops_state_without_append():
    store = open_store("memory://")
    loop = OnlineLoop(store)
    loop.on_change(ChangeEvent("svc", 1, "rcdb.put", _rex([0, 1], [1, 2])))
    before_T = loop.trex.T
    res = loop.on_change(ChangeEvent("svc", 2, "rcdb.delete", None))
    assert res.t == -1 and res.change is None
    assert before_T == loop.trex.T                    # a delete does not append a snapshot


def test_write_back_creates_guarded_derived_version():
    store = open_store("memory://")
    loop = OnlineLoop(store, write_back=True)
    res = loop.on_change(ChangeEvent("svc", 1, "rcdb.put", _rex([0, 1], [1, 2])))
    assert res.wrote_back == "svc::online"
    got = store.get_version("svc::online", 1)
    assert got is not None                            # a derived version was persisted

    # the guard: the derived put's feed event carries DERIVED_TAG and is skipped
    src = ChangeSource(store)
    derived_pub = {"action": "rcdb.put", "scope": "network",
                   "detail": {"id": "svc::online", "version": 1, "tags": [DERIVED_TAG]}}
    assert src._event_from_pub(derived_pub) is None


def test_write_back_off_persists_nothing():
    store = open_store("memory://")
    loop = OnlineLoop(store, write_back=False)
    res = loop.on_change(ChangeEvent("svc", 1, "rcdb.put", _rex([0, 1], [1, 2])))
    assert res.wrote_back is None
    assert store.get_version("svc::online", 1) is None


def test_save_and_reload_field_and_trex(tmp_path):
    pytest.importorskip("safetensors")
    from rexgraph.io.safetensors_bridge import safetensors_to_temporal_rex
    store = open_store("memory://")
    loop = OnlineLoop(store)
    for k in range(2, 6):
        loop.on_change(ChangeEvent("svc", k, "rcdb.put",
                                   _rex(list(range(k)), list(range(1, k + 1)))))
    assert loop.learner.phi                           # the native field is populated after stepping

    tp, fp = loop.save(str(tmp_path / "state"))
    back_trex = safetensors_to_temporal_rex(tp)
    assert back_trex.T == loop.trex.T                 # running TemporalRex round-trips via Slice B

    loop2 = OnlineLoop(store)
    loop2.load_field(fp)
    assert loop2.learner.phi == loop.learner.phi      # the field round-trips by canonical key


def test_poll_replays_oldest_first_and_skips_derived():
    store = open_store("memory://")
    r1 = store.put("svc", _rex([0, 1], [1, 2]))
    r2 = store.put("svc", _rex([0, 1, 2], [1, 2, 3]))
    # a derived-tagged put must not surface through poll
    store.put("svc::online", _rex([0, 1], [1, 2]), tags=[DERIVED_TAG])

    src = ChangeSource(store)
    evs = src.poll()
    ids_versions = [(e.id, e.version) for e in evs]
    assert ("svc", r1.version) in ids_versions
    assert ("svc", r2.version) in ids_versions
    assert all(e.id != "svc::online" for e in evs)    # derived events are guarded out
    # oldest-first: version 1 of svc appears before version 2
    order = [v for (i, v) in ids_versions if i == "svc"]
    assert order == sorted(order)
    # idempotent: a second poll re-delivers nothing already seen
    assert src.poll() == []


def _drive_dogfood(store):
    from rexgraph.core._temporal import cell_keys_of
    loop = OnlineLoop(store, write_back=True)
    src = ChangeSource(store)
    loop.run_stream(src)                              # subscribe on_change to the live bus
    try:
        seqs = [([0, 1], [1, 2]), ([0, 1, 2], [1, 2, 3]),
                ([0, 2], [1, 3]), ([0, 2, 3], [1, 3, 4])]   # step 3 removes edge (1,2)
        for s, t in seqs:
            store.put("svc", _rex(s, t))             # external put -> feed drives on_change synchronously

        processed = [r for r in loop.history() if r.t >= 0]
        # the guard: exactly the external puts were processed, no derived re-entry (no runaway)
        assert len(processed) == len(seqs)

        # stable ids across the intervening deletion: edge (0,1)'s canonical key survives
        r0 = loop.trex.at(0)
        rN = loop.trex.at(loop.trex.T - 1)
        k0 = cell_keys_of(r0.boundary_ptr, r0.boundary_idx, r0._directed)
        kN = set(cell_keys_of(rN.boundary_ptr, rN.boundary_idx, rN._directed).tolist())
        assert int(k0[0]) in kN

        # derived lineage: one version per cycle, every cycle time-travelable
        for v in range(1, len(seqs) + 1):
            assert store.get_version("svc::online", v) is not None
    finally:
        src.stop()
        store.close()


def test_online_loop_closes_over_memory_feed():
    _drive_dogfood(open_store("memory://"))


def test_online_loop_closes_over_file_feed(tmp_path):
    _drive_dogfood(open_store(f"file://{tmp_path}/rcdb"))
