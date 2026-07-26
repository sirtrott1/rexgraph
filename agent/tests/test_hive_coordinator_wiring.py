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
