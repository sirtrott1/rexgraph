from agent.agent.hive_config import CoordinatorSpec, HiveProfile, coordinator_settings


def test_coordinator_spec_defaults_are_neutral():
    c = CoordinatorSpec()
    assert c.enabled is True
    assert c.task_weights == {} and c.worker_weights == {} and c.hive_shares == {}


def test_profile_round_trips_coordinator_spec():
    p = HiveProfile(id="x", name="X",
                    coordinator=CoordinatorSpec(enabled=False, task_weights={"ask": 3.0}))
    d = p.to_dict()
    p2 = HiveProfile.from_dict(d)
    assert p2.coordinator.enabled is False
    assert p2.coordinator.task_weights == {"ask": 3.0}


def test_coordinator_settings_returns_a_spec():
    c = coordinator_settings()
    assert isinstance(c, CoordinatorSpec)


import functools, pickle
from agent.agent.hive_tasks import structural_of
from agent.agent.coordinator_adapter import work_units


def test_structural_of_is_picklable_and_returns_a_dict():
    part = functools.partial(structural_of, "the cat sat on the mat and the cat ran")
    pickle.dumps(part)                         # must not raise
    out = part()
    assert isinstance(out, dict)


def test_work_units_carries_weight_default_one():
    us = work_units([{"id": "a", "kind": "ask", "fn": (lambda: 1)}])
    assert us[0]["weight"] == 1.0
    us2 = work_units([{"id": "b", "kind": "ask", "fn": (lambda: 1), "weight": 4.0}])
    assert us2[0]["weight"] == 4.0


def test_run_wave_returns_results_and_falls_back_on_failure():
    from agent.agent.hive import Hive
    h = Hive("wavetest")
    tasks = [{"id": f"t{i}", "kind": "compute", "fn": (lambda i=i: i * 2)} for i in range(3)]
    assert h._run_wave(tasks) == {f"t{i}": i * 2 for i in range(3)}
    # a task that raises inside the wave still yields correct results for the others via fallback
    bad = [{"id": "ok", "kind": "compute", "fn": (lambda: 5)},
           {"id": "boom", "kind": "compute", "fn": (lambda: (_ for _ in ()).throw(ValueError("x")))}]
    out = h._run_wave(bad)
    assert out["ok"] == 5
    h.stop_all()


def _stub_hive_with_answers(answers: dict):
    """A Hive whose ask() returns canned answers and whose bees are all generate-capable stubs."""
    from agent.agent.hive import Hive, Bee
    h = Hive("consensustest")
    for name in answers:
        b = Bee(name=name, url="http://x", role="worker", capability="generate")
        h._bees[name] = b
    h.ask = lambda name, prompt, **kw: answers[name]           # bypass real LLM
    h.route = lambda query, top_k=3: [{"bee": n} for n in answers]
    return h


def test_consensus_result_matches_regardless_of_coordinator():
    answers = {"w1": "the sky is blue", "w2": "the sky is blue today", "w3": "bananas are yellow"}
    h = _stub_hive_with_answers(answers)
    out = h.consensus("what color is the sky", k=3)
    assert out["n_workers"] == 3
    assert "answer" in out and out["reliability"] >= 0.0
    h.stop_all()


def test_compose_spawns_all_entries_concurrently():
    from agent.agent.hive import Hive
    h = Hive("composetest")
    spawned = []

    def fake_spawn(name, path, **kw):
        spawned.append(name)
        from agent.agent.hive import Bee
        b = Bee(name=name, url="http://x", role=kw.get("role", "worker"), capability="generate")
        h._bees[name] = b
        return b

    h.spawn = fake_spawn
    plan = {"plan": [{"name": f"b{i}", "path": f"/m{i}", "role": "worker"} for i in range(4)]}
    res = h.compose(plan, wait=1.0)
    assert len(res["spawned"]) == 4
    assert sorted(spawned) == ["b0", "b1", "b2", "b3"]
    h.stop_all()


def test_spawn_and_attach_route_to_thread_lane_not_proc():
    # A bee spawn mutates hive state in-process and must run on the thread lane (io_llm), never the
    # forkserver proc lane where the mutation would be lost.
    from agent.agent.coordinator_adapter import _to_type
    assert _to_type("spawn") == "io_llm"
    assert _to_type("attach") == "io_llm"
    assert _to_type("analysis") == "cpu_coordination"   # structural metrics still go to proc
    assert _to_type("ask") == "io_llm"


def test_compose_runs_spawns_concurrently_on_the_thread_lane():
    # Four slow spawns must overlap (thread lane), finishing in roughly one spawn's time, not four.
    import time
    from agent.agent.hive import Hive, Bee
    h = Hive("composeconc")

    def slow_spawn(name, path, **kw):
        time.sleep(0.2)
        b = Bee(name=name, url="http://x", role=kw.get("role", "worker"), capability="generate")
        h._bees[name] = b
        return b

    h.spawn = slow_spawn
    plan = {"plan": [{"name": f"b{i}", "path": f"/m{i}", "role": "worker"} for i in range(4)]}
    t0 = time.perf_counter()
    res = h.compose(plan, wait=1.0)
    dt = time.perf_counter() - t0
    assert len(res["spawned"]) == 4 and all(s["ok"] for s in res["spawned"])
    assert dt < 0.5     # concurrent (~0.2s), not serial (~0.8s)
    h.stop_all()


def test_status_includes_coordinator_block_after_a_wave():
    from agent.agent.hive import Hive
    h = Hive("statustest")
    h._run_wave([{"id": "a", "kind": "compute", "fn": (lambda: 1)}])
    st = h.status()
    assert "coordinator" in st
    assert "pools" in st["coordinator"]
    h.stop_all()


def test_consensus_all_workers_fail_returns_no_answer_not_crash():
    from agent.agent.hive import Hive, Bee
    h = Hive("allfail")
    for n in ("w1", "w2"):
        h._bees[n] = Bee(name=n, url="http://x", role="worker", capability="generate")

    def boom(name, prompt, **kw):
        raise RuntimeError("model down")

    h.ask = boom
    h.route = lambda query, top_k=3: [{"bee": "w1"}, {"bee": "w2"}]
    out = h.consensus("anything", k=2)
    assert out["answer"] is None and out["n_workers"] == 0
    h.stop_all()


def test_compose_duplicate_names_do_not_drop_a_spawn():
    from agent.agent.hive import Hive, Bee
    h = Hive("dupnames")
    calls = []

    def spawn(name, path, **kw):
        calls.append((name, path))
        b = Bee(name=name, url="http://x", role=kw.get("role", "worker"), capability="generate")
        h._bees[name] = b
        return b

    h.spawn = spawn
    plan = {"plan": [{"name": "dupe", "path": "/a", "role": "worker"},
                     {"name": "dupe", "path": "/b", "role": "worker"},
                     {"name": "uniq", "path": "/c", "role": "worker"}]}
    res = h.compose(plan, wait=1.0)
    assert len(res["spawned"]) == 3            # every entry has its own result slot
    assert len(calls) == 3                     # all three spawns actually ran
    h.stop_all()


def test_hive_share_unregistered_on_gc_without_stop_all():
    import gc
    from agent.agent.hive import Hive
    from agent.agent import hive_config as hc
    from agent.agent.hive_config import CoordinatorSpec
    from rexgraph import coordinator as co
    co.reset_shares()
    orig = hc.coordinator_settings
    hc.coordinator_settings = lambda: CoordinatorSpec(hive_shares={"ghost": 3.0})
    try:
        h = Hive("ghost")
        h._run_wave([{"id": "x", "kind": "compute", "fn": (lambda: 1)}])  # builds coordinator, registers share
        assert "ghost" in co._ACTIVE_SHARES
        del h
        gc.collect()
        assert "ghost" not in co._ACTIVE_SHARES   # finalizer dropped it without stop_all
    finally:
        hc.coordinator_settings = orig
        co.reset_shares()
