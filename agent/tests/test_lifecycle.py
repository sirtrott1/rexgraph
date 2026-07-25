"""The lifecycle spine (agent.lifecycle): phase registry, run-logging, and setup-driven phases."""
import pytest

from agent import hive, hive_config, lifecycle


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    hive_config.reset_store()
    lifecycle.reset_store()
    hive.reset_hive()
    yield
    hive_config.reset_store()
    lifecycle.reset_store()
    hive.reset_hive()


def test_builtin_phases_registered():
    names = {p["name"] for p in lifecycle.phases()}
    assert {"serve", "finetune", "build", "deploy", "test"} <= names


def test_run_persists_and_lists():
    rl = lifecycle.run("test")
    assert rl.status == "ok" and rl.ended
    assert rl.steps and rl.steps[0]["msg"].startswith("phase 'test' start")
    # persisted + retrievable + listed
    got = lifecycle.get_store().get(rl.id)
    assert got is not None and got.status == "ok"
    assert rl.id in {r.id for r in lifecycle.get_store().list()}


def test_unknown_phase_raises():
    with pytest.raises(KeyError):
        lifecycle.run("frobnicate")



def test_deploy_phase_generates_real_bundle():
    hive_config.get_store().set_active("solo")
    rl = lifecycle.run("deploy", name="myagent")
    assert rl.status == "ok"
    files = rl.result["files"]
    assert any(f == "Dockerfile" for f in files)     # real deploy.generate_bundle output
    assert rl.result["n_files"] >= 3


def test_test_phase_reports_checks():
    hive_config.get_store().set_active("research")
    rl = lifecycle.run("test")
    checks = {c["check"]: c["ok"] for c in rl.result["checks"]}
    assert checks["core import"] is True
    assert "hive" in checks and "active setup" in checks


def test_register_custom_phase_extensibility():
    @lifecycle.register_phase("eval", "custom eval suite")
    def _eval(ctx):
        ctx.log("running custom eval")
        return {"score": 0.9, "profile": ctx.profile.id if ctx.profile else None}
    try:
        assert "eval" in {p["name"] for p in lifecycle.phases()}
        hive_config.get_store().set_active("solo")
        rl = lifecycle.run("eval")
        assert rl.status == "ok" and rl.result["score"] == 0.9
        assert rl.result["profile"] == "solo"
    finally:
        lifecycle.PHASES.pop("eval", None)


def test_phase_error_is_captured_not_raised():
    @lifecycle.register_phase("boom", "always fails")
    def _boom(ctx):
        raise ValueError("kaboom")
    try:
        rl = lifecycle.run("boom")
        assert rl.status == "error"
        assert "kaboom" in rl.error and rl.ended
        # a failed run is still persisted and inspectable
        assert lifecycle.get_store().get(rl.id).status == "error"
    finally:
        lifecycle.PHASES.pop("boom", None)
