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
