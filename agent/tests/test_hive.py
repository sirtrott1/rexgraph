"""The agent beehive (agent.hive): membership, routing, the message-to-complex seam, monitor.

Uses attached bees and a stubbed chat call so the orchestration runs without spawning
llama.cpp subprocesses."""
import pytest

from agent import agent_complex, hive


@pytest.fixture(autouse=True)
def clean():
    hive.reset_hive()
    agent_complex.reset_live()
    yield
    hive.reset_hive()
    agent_complex.reset_live()


def _beehive():
    h = hive.get_hive()
    h.attach("queen", "http://127.0.0.1:9001", role="queen",
             specialties=["plan", "orchestrate", "summarize"])
    h.attach("coder", "http://127.0.0.1:9002", role="worker",
             specialties=["python", "code", "function", "refactor"])
    h.attach("bio", "http://127.0.0.1:9003", role="worker",
             specialties=["protein", "cell", "receptor", "binding"])
    h.attach("embedder", "http://127.0.0.1:9004", role="embedder")
    return h


def test_membership_roles():
    h = _beehive()
    st = h.status()
    assert st["n_bees"] == 4
    assert st["queen"] == "queen"
    assert st["embedder"] == "embedder"
    assert set(st["workers"]) == {"coder", "bio"}
    # queen sorts first in the bee list
    assert st["bees"][0]["role"] == "queen"


def test_invalid_role_rejected():
    h = hive.get_hive()
    with pytest.raises(ValueError):
        h.attach("x", "http://y", role="drone")


def test_route_by_specialty_cold_hive():
    h = _beehive()
    # no messages yet - routing must fall back to declared specialty
    r = h.route("refactor this python function")
    assert r[0]["bee"] == "coder"
    r2 = h.route("analyze the protein receptor binding site")
    assert r2[0]["bee"] == "bio"
    # the embedder is never a routing target
    assert all(x["bee"] != "embedder" for x in r + r2)


def test_route_falls_back_to_queen():
    h = _beehive()
    r = h.route("something with no specialist overlap zzz")
    assert r and r[0]["bee"] == "queen" and r[0].get("fallback")


def test_ask_records_both_directions_into_complex(monkeypatch):
    h = _beehive()
    monkeypatch.setattr(hive, "_chat", lambda url, model, prompt, **k: "hydrophobic pocket found")
    reply = h.ask("bio", "describe the binding pocket", sender="queen")
    assert reply == "hydrophobic pocket found"
    msgs = agent_complex.get_live()._msgs
    # both the prompt (queen->bio) and the reply (bio->queen) are in the complex
    pairs = {(m["from"], m["to"]) for m in msgs}
    assert ("queen", "bio") in pairs and ("bio", "queen") in pairs


def test_dispatch_routes_then_acts_and_monitor_sees_it(monkeypatch):
    h = _beehive()
    monkeypatch.setattr(hive, "_chat", lambda url, model, prompt, **k: "def f(): return 1")
    out = h.dispatch("write a python function")
    assert out["bee"] == "coder"
    assert out["reply"] == "def f(): return 1"
    # the dispatch populated the live complex -> the monitor now reports real bees
    mon = h.monitor()
    names = {a["agent"] for a in mon["agents"]}
    assert "coder" in names and "user" in names
    assert mon["edges"]  # directed flow exists for the graph


def test_history_reweights_routing(monkeypatch):
    h = _beehive()
    # a realistic reply carries the domain vocabulary (as a real code model's would)
    monkeypatch.setattr(hive, "_chat",
                        lambda url, model, prompt, **k: "refactored the python function cleanly")
    # drive several code tasks through coder so it accrues interaction history
    for _ in range(3):
        h.ask("coder", "python code refactor task", sender="user")
    r = h.route("python refactor")
    coder = next(x for x in r if x["bee"] == "coder")
    assert coder["history"] > 0        # history component now contributes
    assert r[0]["bee"] == "coder"


def _disk(*specs):
    # specs: (name, size_gb) -> discover-shaped gguf entries
    return [{"name": n, "path": f"/m/{n}.gguf", "size_gb": s, "format": "gguf",
             "loadable": "llama.cpp", "source": "dir"} for n, s in specs]


def test_plan_picks_largest_queen_and_fits_workers():
    models = _disk(("qwen3-coder-7b", 6.0), ("phi-4-mini", 2.5),
                   ("gpt-oss-120b", 65.0), ("nomic-embed-text", 0.3))
    plan = hive.plan_hive(models, budget_gb=96.0)
    byrole = {p["role"]: p for p in plan["plan"] if p["role"] != "worker"}
    assert byrole["queen"]["model"] == "gpt-oss-120b"          # largest chat model that fits
    assert byrole["embedder"]["model"] == "nomic-embed-text"   # embedder always included
    workers = {p["model"] for p in plan["plan"] if p["role"] == "worker"}
    assert "phi-4-mini" in workers                             # smallest worker fits after a 65GB queen
    assert plan["planned_gb"] <= plan["usable_gb"] + 0.5       # respects the usable budget


def test_plan_respects_tight_budget():
    models = _disk(("big-70b", 40.0), ("small-3b", 2.5), ("nomic-embed", 0.3))
    plan = hive.plan_hive(models, budget_gb=8.0)               # 70b can't fit
    q = next(p for p in plan["plan"] if p["role"] == "queen")
    assert q["model"] == "small-3b"                             # queen = biggest that fits
    assert all(p["model"] != "big-70b" for p in plan["plan"])  # oversized model excluded


def test_plan_infers_specialties_and_names():
    # a large generalist becomes queen; the coder/math models become named worker bees
    models = _disk(("llama-3.3-70b", 40.0), ("qwen3-coder-14b", 9.0), ("mathstral-7b", 6.0))
    plan = hive.plan_hive(models, budget_gb=96.0)
    named = {p["name"]: p for p in plan["plan"]}
    assert named["queen"]["model"] == "llama-3.3-70b"
    assert "coder" in named and "python" in named["coder"]["specialties"]
    assert "math" in named and "calculus" in named["math"]["specialties"]


def test_plan_no_spawnable_models():
    # only a transformers snapshot on disk - nothing llama.cpp can launch
    models = [{"name": "Qwen/Qwen2.5-7B", "path": "/hf/x", "size_gb": 15.0,
               "format": "transformers", "loadable": "vllm/transformers", "source": "hf-cache"}]
    plan = hive.plan_hive(models, budget_gb=96.0)
    assert plan["plan"] == []


def test_plan_flags_when_embedder_pushes_past_budget():
    # queen alone fits the usable budget, but the always-included embedder tips it over -
    # the plan must say so honestly rather than silently over-committing memory.
    models = _disk(("big-chat-23b", 23.0), ("nomic-embed-text", 1.0))
    plan = hive.plan_hive(models, budget_gb=32.0)
    assert plan["planned_gb"] > plan["usable_gb"]          # the embedder does overflow it
    assert plan["over_budget"] is True                     # ...and the plan admits it


def test_plan_not_over_budget_when_it_fits():
    models = _disk(("qwen3-coder-7b", 6.0), ("phi-4-mini", 2.5),
                   ("gpt-oss-120b", 65.0), ("nomic-embed-text", 0.3))
    plan = hive.plan_hive(models, budget_gb=96.0)
    assert plan["over_budget"] is False


def test_auto_plan_uses_detected_and_budget(monkeypatch):
    monkeypatch.setattr("agent.local_runtime.discover_local_models",
                        lambda: _disk(("qwen-7b", 6.0), ("nomic-embed", 0.3)))
    monkeypatch.setattr("agent.local_runtime.detect_hardware",
                        lambda: {"model_budget_gb": 32.0})
    plan = hive.get_hive().auto_plan()
    assert plan["budget_gb"] == 32.0
    assert any(p["role"] == "queen" for p in plan["plan"])


def test_attach_live_infers_roles(monkeypatch):
    h = hive.get_hive()
    monkeypatch.setattr("agent.local_runtime.probe_endpoints", lambda timeout=0.4: [
        {"url": "http://127.0.0.1:8000", "kind": "openai", "models": ["Qwen2.5-7B"], "n_models": 1},
        {"url": "http://127.0.0.1:11434", "kind": "ollama", "models": ["nomic-embed-text"], "n_models": 1},
    ])
    added = h.attach_live()
    by = {b.url: b for b in added}
    assert by["http://127.0.0.1:8000"].role == "queen"        # first non-embed -> queen
    assert by["http://127.0.0.1:11434"].role == "embedder"    # embed name -> embedder
    # idempotent: attaching again adds nothing new
    assert h.attach_live() == []


def test_model_worker_joins_hive_and_is_monitored(tmp_path):
    """A trained model registers as a worker member, is invokable, and shows in the complex."""
    import pytest
    pytest.importorskip("torch")
    from agent import models
    h = _beehive()
    ckpt = str(tmp_path / "m")
    models.run("mlp", steps=20, save_to=ckpt)
    bee = h.add_model("clf", ckpt, capability="predict", specialties=["classify"])
    assert bee.capability == "predict" and bee.model == "mlp"
    st = {b["name"]: b for b in h.status()["bees"]}
    assert "clf" in st and st["clf"]["local"] is True and st["clf"]["capability"] == "predict"
    out = h.invoke("clf", None)                              # inference on synth data
    assert out["n"] > 0 and out["predictions"].shape[0] == out["n"]
    assert "clf" in {a["agent"] for a in h.monitor()["agents"]}   # a monitored cell
    with pytest.raises(ValueError):                         # ask() is for chat bees only
        h.ask("clf", "hello")


def test_heterogeneous_workers_dispatch_and_type_complex():
    """Any callable joins as a typed worker; dispatch routes by capability; types form a complex."""
    import numpy as np
    import pytest
    h = hive.get_hive()
    h.add_worker("stat", lambda x, **k: float(np.mean(x)), capability="score",
                 worker_type="analyzer:stat:summary", specialties=["score"])
    h.add_worker("rexan", lambda r, **k: {"n": r}, capability="analyze",
                 worker_type="analyzer:rexgraph:hodge", specialties=["structure"])
    assert h.providers("score") == ["stat"] and h.providers("analyze") == ["rexan"]
    r = h.dispatch_capability("score", np.array([1.0, 2.0, 3.0]))
    assert r["worker"] == "stat" and r["result"] == 2.0
    with pytest.raises(ValueError):                          # no provider for a capability
        h.dispatch_capability("embed", [1, 2])
    rex, meta = h.type_complex()                             # worker types as a relational complex
    assert rex.nV > 0 and "analyzer" in meta["vertex_labels"]
    assert {"stat", "rexan"} <= set(meta["vertex_labels"])


def test_hive_monitor_track_exposes_drift():
    """monitor(track=True) accumulates the field over time and reports the drift trend."""
    from agent import agent_complex
    agent_complex.reset_drift()
    h = _beehive()
    for _ in range(2):
        h.relay("coder", "bio", "python task"); h.relay("bio", "coder", "protein result")
        m = h.monitor(track=True)
    assert "drift" in m and "drifting" in m["drift"] and "strain_trend" in m["drift"]


def test_hive_persist_to_rcdb_by_signature():
    """The hive's worker-type structure is catalogued in the RCDB by structural signature."""
    from agent.rcdb import open_store
    store = open_store("memory://")
    h = hive.get_hive()
    h.add_worker("nn", lambda d, **k: 0, capability="predict", worker_type="analyzer:nn:mlp")
    h.add_worker("stat", lambda d, **k: 0, capability="score", worker_type="analyzer:stat:x")
    h.persist(store, name="hv")
    rec = store.get_record("hv")
    assert rec is not None and rec.meta["kind"] == "hive"
    assert {w["name"] for w in rec.meta["workers"]} == {"nn", "stat"}
    assert "predict" in rec.signature.get("tags", []) and "score" in rec.signature.get("tags", [])


# --- collaborate(): dynamic delegation with topological deadlock breaking --------

def _team_hive():
    h = hive.get_hive()
    h.attach("lead", "http://x", role="queen", model="m-lead", specialties=["coordinate", "resolve"])
    h.attach("planner", "http://x", role="worker", model="m-planner", specialties=["plan", "design"])
    h.attach("coder", "http://x", role="worker", model="m-coder", specialties=["code", "implement"])
    h.attach("reviewer", "http://x", role="worker", model="m-reviewer", specialties=["review", "test"])
    return h


def _scripted(script):
    def _chat(url, model, prompt, system=None, **k):
        if "Break the deadlock" in (system or ""):
            return script.get(model + "@break", "resolved directly")
        return script.get(model, "done")
    return _chat


def test_collaborate_breaks_circular_deadlock(monkeypatch):
    # planner -> coder -> reviewer -> planner: b1 of the hand-off complex hits 1
    monkeypatch.setattr(hive, "_chat", _scripted({
        "m-planner": "HANDOFF coder: need the code first",
        "m-coder": "HANDOFF reviewer: need the review first",
        "m-reviewer": "HANDOFF planner: need the plan first",
        "m-lead@break": "resolved in one pass",
    }))
    res = _team_hive().collaborate("Design and build the feature.")
    assert res["deadlock_broken"] is True
    assert res["cycle_at_hop"] == 3                     # the hand-off that closes the loop
    assert res["bee"] == "lead"                         # re-routed to a bee outside the cycle
    assert {"planner", "coder", "reviewer"} <= set(res["cycle_bees"])
    assert res["answer"] == "resolved in one pass"


def test_collaborate_direct_answer_no_loop(monkeypatch):
    monkeypatch.setattr(hive, "_chat", _scripted({"m-planner": "here is the full answer"}))
    res = _team_hive().collaborate("Design and build the feature.")
    assert res["deadlock_broken"] is False
    assert res["hops"] == 1
    assert res["answer"] == "here is the full answer"


def test_collaborate_linear_chain_completes(monkeypatch):
    # a chain (no repeat) never closes a cycle, so no false positive
    monkeypatch.setattr(hive, "_chat", _scripted({
        "m-planner": "HANDOFF coder: implement then finish",
        "m-coder": "implemented and done",
    }))
    res = _team_hive().collaborate("Design and build the feature.")
    assert res["deadlock_broken"] is False
    assert res["answer"] == "implemented and done"
    assert [s["bee"] for s in res["trail"]] == ["planner", "coder"]


# --- consensus(): aggregate workers by the structure of their agreement ----------

def test_consensus_flags_the_divergent_worker(monkeypatch):
    monkeypatch.setattr(hive, "_chat", _scripted({
        "m-planner": "the capital of france is paris a major european city on the seine river",
        "m-coder": "paris is the capital city of france a major european city on the seine",
        "m-reviewer": "bananas are a yellow tropical fruit rich in potassium and grown near the equator",
    }))
    res = _team_hive().consensus("What is the capital of France?",
                                 workers=["planner", "coder", "reviewer"])
    assert res["flagged"] == ["reviewer"]                 # the odd one out is the likely hallucination
    assert res["by"] in ("planner", "coder")
    assert "paris" in (res["answer"] or "").lower()
    assert res["reliability"] > 0.3


def test_consensus_reports_structural_reliability(monkeypatch):
    monkeypatch.setattr(hive, "_chat", _scripted({
        "m-planner": "receptors bind ligands and activate signalling pathways in cells",
        "m-coder": "ligands bind receptors which activate cellular signalling pathways",
        "m-reviewer": "quarterly revenue exceeded projections across every regional market",
    }))
    res = _team_hive().consensus("how do receptors signal?",
                                 workers=["planner", "coder", "reviewer"])
    # each worker carries its own structural reliability read (varentropy gap), independent of agreement
    assert all("varentropy_gap" in r and "reliable" in r for r in res["responders"])
    assert "reviewer" in res["flagged"]                       # off-topic -> shares no content -> flagged


def test_consensus_all_agree_no_flags(monkeypatch):
    monkeypatch.setattr(hive, "_chat", _scripted({
        "m-planner": "the answer is forty two computed from the running sum",
        "m-coder": "the answer is forty two from the running sum",
        "m-reviewer": "the running sum gives the answer forty two",
    }))
    res = _team_hive().consensus("q", workers=["planner", "coder", "reviewer"])
    assert res["flagged"] == []
    assert res["n_workers"] == 3
    assert res["reliability"] > 0.3
