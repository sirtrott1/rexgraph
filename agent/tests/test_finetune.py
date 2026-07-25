"""The finetune module and phase: dep reporting, graceful degradation, phase wiring, A/B shape.

The fine-tune itself needs the [finetune] extra (transformers/peft), which is not in the test
env, so these verify the harness, the setup-driven wiring, and the clean-degradation contract."""
import pytest

from agent import finetune, hive_config, lifecycle


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    hive_config.reset_store()
    lifecycle.reset_store()
    yield
    hive_config.reset_store()
    lifecycle.reset_store()


def test_deps_available_reports_shape():
    d = finetune.deps_available()
    assert "ready" in d and "have" in d and "need" in d
    assert isinstance(d["have"], dict) and "transformers" in d["have"]


def test_default_model_and_dataset():
    assert finetune.DEFAULT_MODEL == "Qwen/Qwen2.5-0.5B-Instruct"
    assert len(finetune._TINY_DATA) >= 8
    assert all("instruction" in e and "response" in e for e in finetune._TINY_DATA)


def test_load_data_point_at_anything(tmp_path):
    # built-in
    assert len(finetune.load_data()) >= 8
    # jsonl instruction/response
    jl = tmp_path / "d.jsonl"
    jl.write_text('{"instruction":"a","response":"b"}\n{"instruction":"c","response":"d"}\n')
    r = finetune.load_data(str(jl))
    assert r == [{"instruction": "a", "response": "b"}, {"instruction": "c", "response": "d"}]
    # plain text file -> one text example per line
    tx = tmp_path / "d.txt"
    tx.write_text("line one\nline two\n")
    assert finetune.load_data(str(tx)) == [{"text": "line one"}, {"text": "line two"}]
    # csv with custom field mapping
    cv = tmp_path / "d.csv"
    cv.write_text("q,a\nx,y\n")
    assert finetune.load_data(str(cv), instruction_field="q", response_field="a") == \
        [{"instruction": "x", "response": "y"}]
    # limit
    assert len(finetune.load_data(str(jl), limit=1)) == 1


def test_finetune_graceful_without_deps(monkeypatch):
    # force the deps-missing path deterministically
    monkeypatch.setattr(finetune, "deps_available",
                        lambda: {"have": {}, "ready": False, "need": "pip install -e '.[finetune]'",
                                 "missing": ["transformers", "peft"]})
    r = finetune.finetune(steps=1)
    assert "skipped" in r and "pip install" in r["skipped"]
    ab = finetune.finetune_ab(steps=1)
    assert "skipped" in ab and ab["ab"] == []


def test_finetune_phase_registered():
    assert "finetune" in {p["name"] for p in lifecycle.phases()}


def test_finetune_phase_degrades_cleanly_and_reads_setup(monkeypatch):
    monkeypatch.setattr(finetune, "deps_available",
                        lambda: {"have": {}, "ready": False, "need": "pip install -e '.[finetune]'",
                                 "missing": ["transformers"]})
    hive_config.get_store().save(hive_config.HiveProfile(id="p", name="P", optimizer="hodge"))
    hive_config.get_store().set_active("p")
    rl = lifecycle.run("finetune", steps=1)
    assert rl.status == "ok"                      # a missing extra is not a failure
    assert "skipped" in rl.result
    assert any("deps missing" in s["msg"] for s in rl.steps)


def test_finetune_ab_runs_both_optimizers(monkeypatch):
    # stub the single-run trainer so we can verify the A/B orchestration without torch/transformers
    calls = []

    def fake_single(*, optimizer, on_step=None, label=None, steps=10, **kw):
        calls.append(optimizer)
        if on_step:
            on_step(label or optimizer, 0, 2.0, steps)
        # hodge generalizes clearly better on the held-out eval here
        ev = 0.8 if optimizer == "hodge" else 1.2
        return {"optimizer": optimizer, "optimizer_class": optimizer.title(),
                "loss_start": 2.0, "loss_final": 1.0, "eval_start": 2.0, "eval_final": ev,
                "eval_trajectory": [2.0, 1.5, ev], "trajectory": [2.0, 1.5, 1.0], "adapter": None}

    monkeypatch.setattr(finetune, "deps_available",
                        lambda: {"have": {}, "ready": True, "need": "", "missing": []})
    monkeypatch.setattr(finetune, "finetune", fake_single)
    res = finetune.finetune_ab(optimizers=("hodge", "adam"), steps=3)
    assert calls == ["hodge", "adam"]
    assert len(res["ab"]) == 2
    assert res["best"] == "hodge"                 # lower held-out EVAL loss wins the verdict
    assert res["eval_losses"] == {"hodge": 0.8, "adam": 1.2}
    assert "hodge" in res["verdict"]              # a clear gap -> named winner, not "tie"


def test_ab_reports_tie_within_noise(monkeypatch):
    def fake_single(*, optimizer, on_step=None, label=None, steps=10, **kw):
        ev = 0.231 if optimizer == "hodge" else 0.230   # sub-1% gap = noise
        return {"optimizer": optimizer, "optimizer_class": optimizer.title(),
                "loss_final": 0.2, "eval_final": ev, "eval_start": 2.0,
                "eval_trajectory": [2.0, ev], "trajectory": [2.0, 0.2], "adapter": None}
    monkeypatch.setattr(finetune, "deps_available",
                        lambda: {"have": {}, "ready": True, "need": "", "missing": []})
    monkeypatch.setattr(finetune, "finetune", fake_single)
    res = finetune.finetune_ab(optimizers=("hodge", "adam"), steps=2)
    assert "tie" in res["verdict"]                 # a noise-level gap is called a tie
