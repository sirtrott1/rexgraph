"""The agent builder, executed rather than described.

`builder.py` is 784 lines and 205 of them never ran under test. It is what the Agent
Builder screen's run button calls, so a config that composes on screen and fails on
execution is a defect the UI cannot show.

The steps that need a model, an OCR backend or LangChain are exercised for their
failure behaviour instead: a missing dependency has to come back as a step with
status "error" and a message, not as an exception out of `run`.
"""
from __future__ import annotations

import json

import pytest
from agent.builder import AgentBuilder

TEMPLATES = ("default", "rag", "training", "langgraph", "langchain")

DOC = ("Alpha connects beta. Beta connects gamma. Gamma connects alpha.\n"
       "Delta relates to alpha. Epsilon relates to delta. Alpha relates to epsilon.\n"
       "The alpha pathway regulates the beta response through gamma.\n")


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text(DOC)
    return str(p)


#### the registry


def test_every_registered_step_is_listed():
    steps = AgentBuilder.available_steps()
    assert steps == sorted(steps), "available_steps is not sorted"
    assert "corpus" in steps and "chunk" in steps and "query" in steps


def test_every_registered_step_documents_itself():
    """The builder screen shows `step_help` as the step's description. A step with no
    docstring shows an empty box."""
    missing = [s for s in AgentBuilder.available_steps()
               if not (AgentBuilder.step_help(s) or "").strip()]
    assert not missing, f"steps with no docstring: {', '.join(missing)}"


def test_help_for_an_unknown_step_says_so_rather_than_raising():
    assert "nknown" in AgentBuilder.step_help("no-such-step")


#### templates


@pytest.mark.parametrize("name", TEMPLATES)
def test_a_template_names_only_registered_steps(name):
    """A template offering a step the registry does not have is a dead menu entry."""
    known = set(AgentBuilder.available_steps())
    used = {s["type"] for s in AgentBuilder.template(name)["steps"]}
    assert used <= known, f"{name} uses unregistered steps: {used - known}"


@pytest.mark.parametrize("name", TEMPLATES)
def test_a_template_is_well_formed(name):
    tpl = AgentBuilder.template(name)
    assert tpl.get("name"), f"{name} has no name"
    assert tpl.get("steps"), f"{name} has no steps"
    for s in tpl["steps"]:
        assert isinstance(s.get("type"), str) and s["type"]
        assert isinstance(s.get("params", {}), dict)


def test_an_unknown_template_falls_back_to_default():
    assert AgentBuilder.template("no-such-template") == AgentBuilder.template("default")


#### config round-trip


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_a_config_round_trips_through_a_file(tmp_path, suffix):
    cfg = AgentBuilder.template("rag")
    path = AgentBuilder(cfg).save(str(tmp_path / f"agent{suffix}"))
    back = AgentBuilder.load(path)
    assert back.name == cfg["name"]
    assert [s["type"] for s in back.steps] == [s["type"] for s in cfg["steps"]]


def test_save_returns_the_path_it_actually_wrote(tmp_path):
    """With no yaml installed, `save` rewrites a .yaml request as .json. The returned
    path is the one to reopen, so it has to be the real one."""
    import os
    path = AgentBuilder(AgentBuilder.template("default")).save(
        str(tmp_path / "agent.yaml"))
    assert os.path.exists(path), f"save returned {path}, which does not exist"


def test_a_json_config_written_by_hand_loads(tmp_path):
    p = tmp_path / "hand.json"
    p.write_text(json.dumps({"name": "hand-written",
                             "steps": [{"type": "corpus"}, {"type": "chunk"}]}))
    b = AgentBuilder.load(str(p))
    assert b.name == "hand-written" and len(b.steps) == 2


#### execution


def test_the_default_template_runs_on_a_real_document(doc):
    """corpus -> chunk -> query, the path the run button takes."""
    res = AgentBuilder(AgentBuilder.template("default")).run(
        files=[doc], query="what connects alpha?")
    assert [s.step_type for s in res.steps] == ["corpus", "chunk", "query"]
    failed = [(s.step_type, s.error) for s in res.steps if s.status != "ok"]
    assert not failed, f"steps failed: {failed}"


def test_a_run_builds_a_complex_from_the_document(doc):
    res = AgentBuilder(AgentBuilder.template("default")).run(
        files=[doc], query="what connects alpha?")
    assert res.documents, "the run reported no documents"
    d = res.documents[0]
    assert d["nE"] > 0, "the document produced no relations"
    assert d["nV"] > 0


def test_a_run_returns_chunks_and_query_results(doc):
    res = AgentBuilder(AgentBuilder.template("default")).run(
        files=[doc], query="what connects alpha?")
    assert res.chunks, "chunk ran but no chunks reached the result"
    assert res.query_results, "query ran but no results reached the result"


def test_a_run_reports_elapsed_time_per_step(doc):
    res = AgentBuilder(AgentBuilder.template("default")).run(files=[doc], query="q")
    assert res.elapsed > 0
    assert all(s.elapsed >= 0 for s in res.steps)


def test_an_unknown_step_is_reported_not_raised(doc):
    res = AgentBuilder({"name": "bad", "steps": [
        {"type": "corpus"}, {"type": "not-a-step"}, {"type": "chunk"}]}).run(
        files=[doc], query="q")
    bad = [s for s in res.steps if s.step_type == "not-a-step"]
    assert bad and bad[0].status == "error"
    assert "nknown" in bad[0].error
    assert any(s.step_type == "chunk" for s in res.steps), (
        "an unknown step stopped the run instead of being skipped")


def test_a_required_step_that_fails_stops_the_run(doc):
    """A required step that raises must end the run rather than let the rest proceed
    on state it did not produce."""
    res = AgentBuilder({"name": "x", "steps": [
        {"type": "corpus"},
        {"type": "export", "required": True,
         "params": {"output": "/proc/nonexistent/out.rex"}},
        {"type": "chunk"}]}).run(files=[doc], query="q")
    assert [x.step_type for x in res.steps] == ["corpus", "export"], \
        "a failed required step did not stop the run"
    assert res.steps[-1].status == "error"


def test_running_with_no_files_does_not_raise():
    res = AgentBuilder(AgentBuilder.template("default")).run(files=[], query="q")
    assert res is not None
    assert isinstance(res.steps, list)


def test_an_empty_config_runs_and_reports_nothing():
    res = AgentBuilder({"name": "empty", "steps": []}).run(files=[], query="q")
    assert res.steps == [] and res.chunks == []


def test_defaults_reach_every_step(doc):
    """`defaults` is the config-level parameter block. A step's own params win."""
    res = AgentBuilder({
        "name": "d", "defaults": {"depth": "quick"},
        "steps": [{"type": "corpus"}, {"type": "chunk", "params": {"min_chars": 20}}],
    }).run(files=[doc], query="q")
    assert all(s.status == "ok" for s in res.steps), \
        [(s.step_type, s.error) for s in res.steps]


#### steps that need something not installed here


@pytest.mark.parametrize("step", ["model", "langchain_tools", "langgraph_analyze",
                                  "hallucination_check", "export", "ocr"])
def test_a_step_runs_after_a_corpus(doc, step, tmp_path):
    """Each remaining step, executed once with a corpus in front of it. These all
    complete here; a step that needs a backend has to say so as a step error with a
    message, never as an exception escaping `run`."""
    params = {"output": str(tmp_path / f"{step}.rex")} if step == "export" else {}
    res = AgentBuilder({"name": "x", "steps": [
        {"type": "corpus"},
        {"type": step, "required": False, "params": params}]}).run(
        files=[doc], query="what connects alpha?")
    got = [s for s in res.steps if s.step_type == step]
    assert got, f"{step} did not appear in the result at all"
    assert got[0].status == "ok" or got[0].error, \
        f"{step} failed with an empty message"


def test_the_langgraph_template_runs_its_structural_steps(doc):
    """langgraph_init and langgraph_record are pure structure and need no LLM, so
    they must run even where the model step cannot."""
    res = AgentBuilder(AgentBuilder.template("langgraph")).run(
        files=[doc], query="what connects alpha?")
    by_type = {}
    for s in res.steps:
        by_type.setdefault(s.step_type, []).append(s)
    assert by_type["langgraph_init"][0].status == "ok", \
        by_type["langgraph_init"][0].error
    assert any(s.status == "ok" for s in by_type.get("langgraph_record", [])), \
        "no state was recorded against the execution complex"


def test_the_training_template_writes_its_export(doc, tmp_path):
    cfg = AgentBuilder.template("training")
    for s in cfg["steps"]:
        if s["type"] == "training_export":
            s["params"]["output"] = str(tmp_path / "training.safetensors")
    res = AgentBuilder(cfg).run(files=[doc], query="q")
    exp = [s for s in res.steps if s.step_type == "training_export"]
    assert exp and exp[0].status == "ok", exp and exp[0].error
    assert (tmp_path / "training.safetensors").exists(), \
        "the export reported success but wrote no file"
    assert exp[0].data.get("n_examples", 0) > 0, "the export wrote no examples"


#### what a run reports about itself


def test_a_training_export_reports_where_it_wrote(doc, tmp_path):
    cfg = AgentBuilder.template("training")
    for s in cfg["steps"]:
        if s["type"] == "training_export":
            s["params"]["output"] = str(tmp_path / "training.safetensors")
    res = AgentBuilder(cfg).run(files=[doc], query="q")
    assert res.export_path, "the run wrote a file and reported no path for it"


def test_a_corpus_over_a_missing_file_does_not_invent_a_complex(tmp_path):
    missing = tmp_path / "alpha-beta-gamma-delta.txt"
    res = AgentBuilder({"name": "x", "steps": [{"type": "corpus"}]}).run(
        files=[str(missing)], query="q")
    step = res.steps[0]
    if step.status == "error":
        return                                  # the acceptable outcome
    docs = step.data.get("documents", [])
    built = [d for d in docs if d.get("nE", 0) > 0]
    assert not built, (
        f"a file that does not exist produced a complex: {built[0]}. The vertices "
        f"are the words of the path, not of any document.")
