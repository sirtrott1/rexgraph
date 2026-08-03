"""Hive profiles (agent.hive_config): presets, save/clone/delete, active pointer, apply."""
import pytest
from agent.hive_config import BeeSpec, HiveProfile, ProfileStore

from agent import hive, hive_config


@pytest.fixture
def store(tmp_path):
    return ProfileStore(directory=tmp_path / "hive_profiles")


@pytest.fixture(autouse=True)
def clean():
    hive.reset_hive()
    yield
    hive.reset_hive()


def test_builtins_always_present(store):
    ids = {p.id for p in store.list()}
    assert {"solo", "research", "coding", "lean", "attach", "max"} <= ids
    assert all(p.builtin for p in store.list() if p.id in ("solo", "research"))


def test_get_builtin_and_roundtrip():
    p = hive_config._BUILTIN_BY_ID["research"]
    d = p.to_dict()
    assert HiveProfile.from_dict(d).id == "research"
    assert d["compose"] == "auto" and d["monitor_embed"] is True


def test_create_clone_and_persist(store):
    p = store.create("My Coder", base="coding", max_workers=2, optimizer="adam")
    assert p.id == "my-coder" and not p.builtin
    assert p.max_workers == 2 and p.optimizer == "adam"
    # persisted + listed alongside builtins
    got = store.get("my-coder")
    assert got is not None and got.name == "My Coder"
    assert "my-coder" in {x.id for x in store.list()}


def test_user_profile_shadows_builtin(store):
    # saving with a built-in id overrides it in listings; delete restores the preset
    store.save(HiveProfile(id="solo", name="My Solo", compose="attach-live"))
    got = store.get("solo")
    assert got.name == "My Solo" and got.compose == "attach-live"
    assert store.delete("solo") is True
    assert store.get("solo").name == "Solo driver"       # preset back


def test_active_pointer(store):
    assert store.active_id() is None
    store.set_active("research")
    assert store.active_id() == "research"
    assert store.active().id == "research"


def test_delete_clears_active(store):
    store.create("Temp", base="lean")
    store.set_active("temp")
    assert store.delete("temp") is True
    assert store.active_id() is None


def test_apply_attach_profile_stands_up_hive(store, monkeypatch):
    # 'attach' profile enrolls whatever is 'running' - stub the probe so no process is needed
    monkeypatch.setattr("agent.local_runtime.probe_endpoints", lambda timeout=0.4: [
        {"url": "http://127.0.0.1:8000", "kind": "openai", "models": ["Qwen2.5-7B"], "n_models": 1},
    ])
    res = store.apply("attach")
    assert res["profile"] == "attach"
    assert res["status"]["n_bees"] == 1
    assert store.active_id() == "attach"                  # apply sets active
    assert res["engine"]["monitor_embed"] is True


def test_apply_manual_bees(store, monkeypatch):
    prof = HiveProfile(id="mine", name="Mine", compose="manual",
                       bees=[BeeSpec(name="q", role="queen", source="attach",
                                     url="http://127.0.0.1:9001", specialties=["plan"])])
    store.save(prof)
    res = store.apply("mine")
    assert "q" in res["attached"]
    assert res["status"]["queen"] == "q"


def test_apply_switches_setups(store, monkeypatch):
    monkeypatch.setattr("agent.local_runtime.probe_endpoints", lambda timeout=0.4: [
        {"url": "http://127.0.0.1:8000", "kind": "openai", "models": ["m"], "n_models": 1},
    ])
    store.apply("attach")
    assert hive.get_hive().status()["n_bees"] == 1
    # switching to a manual profile with no reachable bees resets the swarm first
    store.save(HiveProfile(id="empty", name="Empty", compose="manual"))
    res = store.apply("empty")
    assert res["status"]["n_bees"] == 0                   # prior swarm was cleared on switch
