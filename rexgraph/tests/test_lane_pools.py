import functools

from rexgraph.coordinator import CostModel, LanePools


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_pools_are_lazy_cold_at_rest():
    p = LanePools("h", now=FakeClock())
    st = p.status()
    assert st["thread"]["state"] == "cold"
    assert st["proc"]["state"] == "cold"
    p.shutdown()


def test_thread_wave_runs_and_warms_only_thread():
    p = LanePools("h", now=FakeClock())
    units = [{"id": f"t{i}", "type": "io_llm", "fn": (lambda i=i: i * i)} for i in range(4)]
    res = p.run(units, {u["id"]: "thread" for u in units}, cost=CostModel())
    assert res == {f"t{i}": i * i for i in range(4)}
    assert p.status()["thread"]["state"] == "warm"
    assert p.status()["proc"]["state"] == "cold"
    p.shutdown()


def test_proc_wave_runs_picklable_fn_and_warms_proc():
    p = LanePools("h", now=FakeClock())
    units = [{"id": f"c{i}", "type": "cpu_coordination", "fn": functools.partial(pow, i, 2)}
             for i in range(4)]
    res = p.run(units, {u["id"]: "proc" for u in units}, cost=CostModel())
    assert res == {f"c{i}": i * i for i in range(4)}
    assert p.status()["proc"]["state"] == "warm"
    p.shutdown()


def test_unpicklable_proc_fn_spills_to_thread():
    p = LanePools("h", now=FakeClock())
    seen = []

    def make(i):
        return lambda: (seen.append(i), i)[1]

    units = [{"id": f"c{i}", "type": "cpu_coordination", "fn": make(i)} for i in range(3)]
    res = p.run(units, {u["id"]: "proc" for u in units}, cost=CostModel())
    assert res == {f"c{i}": i for i in range(3)}
    assert sorted(seen) == [0, 1, 2]  # ran in-process on the thread lane
    p.shutdown()


def test_warm_pool_is_reused_across_waves():
    p = LanePools("h", now=FakeClock())
    u = [{"id": "a", "type": "io_llm", "fn": (lambda: 1)}]
    p.run(u, {"a": "thread"})
    ex1 = p._pools["thread"]
    p.run(u, {"a": "thread"})
    assert p._pools["thread"] is ex1  # same executor object, not re-created
    p.shutdown()


def test_idle_reap_closes_pool_and_reaper_self_exits():
    clk = FakeClock()
    p = LanePools("h", now=clk, idle_ttl_proc=10.0, idle_ttl_thread=10.0, reaper_tick=0.01)
    p.run([{"id": "a", "type": "io_llm", "fn": (lambda: 1)}], {"a": "thread"})
    assert p.status()["thread"]["state"] == "warm"
    clk.advance(20.0)                 # push both lanes past their TTL
    p._reap_once()                    # deterministic single reap pass (no sleep)
    assert p.status()["thread"]["state"] == "cold"
    assert p.status()["proc"]["state"] == "cold"
    p.shutdown()


def test_run_folds_timing_into_cost():
    cm = CostModel()
    before, _ = cm.cost("io_llm", "thread")
    p = LanePools("h", now=FakeClock())
    p.run([{"id": "a", "type": "io_llm", "fn": (lambda: 1)}], {"a": "thread"}, cost=cm)
    after, _ = cm.cost("io_llm", "thread")
    assert after != before
    p.shutdown()


def test_reaper_restarts_after_flag_cleared_even_if_thread_lingers():
    # Regression: _ensure must key off the reaper_alive flag, not a stale thread.is_alive().
    # After _reap_once clears the flag (deciding to exit), a new wave must start a FRESH reaper so a
    # newly created pool is never left unwatched.
    p = LanePools("h", now=FakeClock())
    p.run([{"id": "a", "type": "io_llm", "fn": (lambda: 1)}], {"a": "thread"})
    with p._lock:
        p.reaper_alive = False          # simulate the reaper having decided to exit
    p.run([{"id": "b", "type": "io_llm", "fn": (lambda: 1)}], {"b": "thread"})
    assert p.reaper_alive is True       # a new reaper was started off the flag
    p.shutdown()


def test_background_reaper_actually_reaps_and_self_exits_real_clock():
    # End-to-end with the REAL background thread (not _reap_once): a short TTL + tick must close the
    # warm pool and the reaper thread must terminate, leaving nothing running.
    import time
    p = LanePools("h", idle_ttl_proc=0.05, idle_ttl_thread=0.05, reaper_tick=0.02)
    p.run([{"id": "a", "type": "io_llm", "fn": (lambda: 1)}], {"a": "thread"})
    assert p.status()["thread"]["state"] == "warm"
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and p.status()["thread"]["state"] == "warm":
        time.sleep(0.02)
    assert p.status()["thread"]["state"] == "cold"      # reaped by the background thread
    time.sleep(0.05)
    assert p.reaper_alive is False                      # reaper self-exited, nothing lingers
    p.shutdown()


def test_coordinator_uses_managed_pools_when_given():
    from rexgraph.coordinator import Coordinator, CostModel, LanePools
    pools = LanePools("h", now=FakeClock())
    co_ = Coordinator(cost=CostModel(), pools=pools)
    units = [{"id": f"t{i}", "type": "io_llm", "fn": (lambda i=i: i + 1)} for i in range(3)]
    res = co_.run_wave(units)
    assert res == {f"t{i}": i + 1 for i in range(3)}
    assert pools.status()["thread"]["state"] == "warm"  # went through the managed pool
    pools.shutdown()


def test_coordinator_without_pools_uses_per_wave_execute():
    from rexgraph.coordinator import Coordinator, CostModel
    co_ = Coordinator(cost=CostModel())
    units = [{"id": "a", "type": "io_llm", "fn": (lambda: 7)}]
    assert co_.run_wave(units) == {"a": 7}


def test_run_isolates_a_failing_task_no_rerun():
    # A single task raising must NOT abort the wave or re-run its peers: the failed id is omitted,
    # every fn runs exactly once.
    calls = {"a": 0, "b": 0, "c": 0}

    def mk(k):
        def f():
            calls[k] += 1
            if k == "b":
                raise ValueError("boom")
            return k.upper()
        return f

    p = LanePools("h", now=FakeClock())
    units = [{"id": k, "type": "io_llm", "fn": mk(k)} for k in ("a", "b", "c")]
    res = p.run(units, {u["id"]: "thread" for u in units})
    assert res == {"a": "A", "c": "C"}          # b omitted, no exception raised
    assert calls == {"a": 1, "b": 1, "c": 1}    # each ran exactly once (no full-wave re-run)
    p.shutdown()


def test_wave_longer_than_ttl_is_not_reaped_midflight():
    # The background reaper must not close a lane that is actively running a wave, even when the wave
    # outlasts the TTL. After completion the pool stays warm (idle clock refreshed).
    import time
    p = LanePools("h", idle_ttl_thread=0.05, idle_ttl_proc=0.05, reaper_tick=0.02)

    def slow():
        time.sleep(0.25)      # far longer than the 0.05s TTL; reaper ticks during it
        return 1

    res = p.run([{"id": "a", "type": "io_llm", "fn": slow}], {"a": "thread"})
    assert res == {"a": 1}
    assert p.status()["thread"]["state"] == "warm"   # survived: busy-guard + completion refresh
    p.shutdown()


def test_inner_threads_budget_prevents_oversubscription():
    import rexgraph.coordinator as co
    cores = __import__("os").cpu_count() or 8
    assert co._inner_threads(1) == cores          # a lone worker may use all cores
    assert co._inner_threads(cores) == 1          # cores workers -> 1 BLAS thread each (no oversub)
    assert co._inner_threads(4) == max(1, cores // 4)
    assert co._inner_threads(0) >= 1              # never below 1


def test_inner_threads_budget_scales_with_a_hive_share():
    # Cross-coordinator: several hives share the machine, so each pool's inner-thread budget must be
    # its SHARE of cores, not all cores. 3 equal hives on 32 cores -> ~10-core budget each.
    import rexgraph.coordinator as co
    assert co._inner_threads(5, cores_budget=10) == 2      # 10-core share / 5 workers = 2 BLAS each
    assert co._inner_threads(8, cores_budget=32) == 4
    assert co._inner_threads(2, cores_budget=10) == 5
    # 3 equal hives never collectively exceed the machine: sum(workers*inner) <= cores
    import os
    cores = os.cpu_count() or 8
    per = cores // 3
    assert co._inner_threads(max(1, (cores // 2) // 3), cores_budget=per) * ((cores // 2) // 3 or 1) <= cores + 2
