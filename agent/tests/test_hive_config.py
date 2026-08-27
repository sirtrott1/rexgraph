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


# specialty rules: the model-name -> specialty table as CONFIG, not a hardcoded list

def test_builtin_specialty_rules_cover_the_shipped_families():
    rules = hive_config.load_specialty_rules()
    bases = {r.base for r in rules}
    assert {"coder", "math", "bio", "sql", "vision", "legal"} <= bases


def test_user_rules_file_overrides_the_builtins(tmp_path, monkeypatch):
    """A new model family must be teachable without editing hive.py."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    (tmp_path / "specialty_rules.json").write_text(
        '[{"match": ["myco"], "base": "mycology", "specialties": ["fungi", "spore"]}]')
    rules = hive_config.load_specialty_rules()
    assert [r.base for r in rules] == ["mycology"]
    assert hive._specialty_of("MycoLLM-7B", rules=rules) == ("mycology", ["fungi", "spore"])


def test_a_rule_can_exclude_a_family_member(tmp_path, monkeypatch):
    """`exclude` is what lets a broad `match` stay safe: match the family, veto the variants
    that are not actually that specialty."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    (tmp_path / "specialty_rules.json").write_text(
        '[{"match": ["qwen"], "base": "coder", "specialties": ["code"], "exclude": ["-vl", "audio"]}]')
    rules = hive_config.load_specialty_rules()
    assert hive._specialty_of("Qwen3-Coder-30B", rules=rules)[0] == "coder"
    assert hive._specialty_of("Qwen2-VL-7B", rules=rules) == (None, [])      # vetoed
    assert hive._specialty_of("Qwen2-Audio-7B", rules=rules) == (None, [])   # vetoed


def test_malformed_rules_file_falls_back_to_builtins(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    (tmp_path / "specialty_rules.json").write_text("{ not json")
    assert {r.base for r in hive_config.load_specialty_rules()} >= {"coder", "math"}


def test_plan_hive_accepts_explicit_rules():
    """plan_hive threads rules through rather than reaching for a global."""
    rules = [hive_config.SpecialtyRule(match=["zzz"], base="zed", specialties=["zeta"])]
    models = [{"name": "zzz-7b", "path": "/m/zzz.gguf", "format": "gguf", "size_gb": 4.0}]
    plan = hive.plan_hive(models, budget_gb=32.0, rules=rules)["plan"]
    assert plan[0]["specialties"] == ["zeta"]


def test_an_unmatched_worker_still_gets_a_general_specialty():
    """The queen already falls back to general specialties when nothing matches; a worker got
    an EMPTY list, so it scored 0 on every cold-hive routing query."""
    models = [{"name": "big-generalist-70b", "path": "/m/a.gguf", "format": "gguf", "size_gb": 20.0},
              {"name": "small-generalist-3b", "path": "/m/b.gguf", "format": "gguf", "size_gb": 2.0}]
    plan = hive.plan_hive(models, budget_gb=64.0)["plan"]
    worker = next(p for p in plan if p["role"] == "worker")
    assert worker["specialties"], "an unmatched worker must not be left with no specialties"
