"""agent.reactive_hive: the monitor->schema loop (deadlock->mediator, divergence->guard)."""
from agent.reactive_hive import ReactiveHive

from agent import agent_complex, rcdb
from agent import hive as hivemod


def _hive():
    hivemod.reset_hive(); agent_complex.reset_live()
    h = hivemod.get_hive()
    h.attach("lead", "http://x", role="queen", model="m", specialties=["coordinate"])
    return h


def _scripted(script):
    def _chat(url, model, prompt, system=None, **k):
        return script.get(model, "ok")
    return _chat


def test_deadlock_deploys_mediator_and_versions_schema():
    h = _hive()
    rh = ReactiveHive(h, store=rcdb.MemoryStore())
    rh.schema.snapshot(cause="init")
    # inject a circular coordination wait -> beta_1 of the interaction complex = 1
    for a, b in [("planner", "coder"), ("coder", "reviewer"), ("reviewer", "planner")]:
        h.relay(a, b, "waiting on you")
    assert h.monitor()["deadlock_cycles"] == 1

    actions = rh.react()
    assert any(x["rule"] == "deadlock" for x in actions)
    assert h.get("mediator") is not None                    # the hive mutated its own structure
    assert any("deadlock" in e["cause"] for e in rh.schema.evolution())  # tracked, with cause


def test_healthy_hive_takes_no_action():
    h = _hive()
    rh = ReactiveHive(h, store=rcdb.MemoryStore())
    for w in ["a", "b", "c"]:                                # a star (no cycle) = no deadlock
        h.relay("lead", w, "do the task")
        h.relay(w, "lead", "done")
    assert rh.react() == []


def test_reaction_is_idempotent():
    h = _hive()
    rh = ReactiveHive(h, store=rcdb.MemoryStore())
    for a, b in [("x", "y"), ("y", "z"), ("z", "x")]:
        h.relay(a, b, "waiting")
    first = rh.react()
    assert len(first) == 1
    assert rh.react() == []                                  # mediator already deployed, no repeat


def test_require_grows_the_missing_specialists():
    h = _hive()                                              # just a lead
    rh = ReactiveHive(h, store=rcdb.MemoryStore())
    acts = rh.require("review", "test")
    assert {a["need"] for a in acts} == {"review", "test"}
    assert h.get("reviewer") is not None and h.get("tester") is not None
    # a satisfied need is a no-op (exact set-membership), and it is idempotent
    assert rh.require("review") == []
    assert any("review" in e["cause"] for e in rh.schema.evolution())


def test_on_consensus_deploys_verifier_on_reliability_gap():
    h = _hive()
    rh = ReactiveHive(h, store=rcdb.MemoryStore())
    # a flagged worker OR a structurally unreliable answer triggers it (structural facts)
    acts = rh.on_consensus({"flagged": ["rogue"], "responders": []})
    assert acts and acts[0]["rule"] == "reliability" and h.get("verifier") is not None

    h2 = _hive()
    rh2 = ReactiveHive(h2, store=rcdb.MemoryStore())
    acts2 = rh2.on_consensus({"flagged": [], "responders": [{"reliable": False}]})
    assert acts2 and h2.get("verifier") is not None
    # a clean consensus does nothing
    assert rh2.on_consensus({"flagged": [], "responders": [{"reliable": True}]}) == []


def test_on_query_attaches_db_for_unbound_tables():
    h = _hive()
    rh = ReactiveHive(h, store=rcdb.MemoryStore())

    class FakeDB:
        def table_names(self):
            return ["suppliers", "inventory"]

        def attach_to_hive(self, hive, prefix="db"):
            hive.add_worker("wh.search", lambda d, **k: d, capability="analyze",
                            worker_type="db:search")
            return ["wh.search"]

    q = {"linked": True, "disconnected_tables": ["suppliers"], "unmatched_concepts": []}
    acts = rh.on_query(q, available={"warehouse": FakeDB()})
    assert acts and acts[0]["attached"] == "warehouse"
    assert h.get("wh.search") is not None                    # the db's bees joined the hive
    assert any("warehouse" in e["cause"] for e in rh.schema.evolution())


def test_run_grows_team_does_work_and_verifies(monkeypatch):
    h = _hive()                                              # a lone lead
    monkeypatch.setattr(hivemod, "_chat",
                        lambda url, model, prompt, **k:
                        "the parser reads tokens and builds an ast from grammar rules")
    rh = ReactiveHive(h, store=rcdb.MemoryStore())
    out = rh.run("plan, review, and test the parser")
    assert out["answer"] and "parser" in out["answer"]
    # the team grew the exact roles the task implied, while working
    assert {"planner", "reviewer", "tester"} <= set(out["team"])
    caps = {a.get("need") for a in out["reactions"] if a.get("rule") == "capability"}
    assert {"plan", "review", "test"} <= caps
    assert out["verification"] is not None                   # a cross-check ran


def test_run_deploys_verifier_on_reliability_gap(monkeypatch):
    hivemod.reset_hive(); agent_complex.reset_live()
    h = hivemod.get_hive()
    for name, tag, role in [("a", "m-a", "queen"), ("b", "m-b", "worker"), ("c", "m-c", "worker")]:
        h.attach(name, "http://x", role=role, model=tag, specialties=["answer"])
    monkeypatch.setattr(hivemod, "_chat", _scripted({
        "m-a": "the parser reads tokens and builds an ast from grammar rules",
        "m-b": "parser tokens build an ast using the grammar rules",
        "m-c": "quarterly revenue exceeded projections in every region",
    }))
    rh = ReactiveHive(h, store=rcdb.MemoryStore())
    out = rh.run("answer the parser question", needs=[])     # skip capability inference
    assert out["verification"]["flagged"]                    # the off-topic worker was caught
    assert h.get("verifier") is not None                     # -> reactive layer deployed a verifier
    assert any(a.get("rule") == "reliability" for a in out["reactions"])


def test_divergent_worker_deploys_guard(monkeypatch):
    h = _hive()
    rh = ReactiveHive(h, store=rcdb.MemoryStore())
    monkeypatch.setattr(h, "monitor", lambda **k: {
        "deadlock_cycles": 0, "interaction_hodge": {"persistent": 0.0},
        "alignment_mode": "embedding",                       # the reliable semantic signal
        "agents": [{"agent": "rogue", "flag": "divergent", "load_bearing": 1.0}],
    })
    actions = rh.react()
    assert any(x["rule"] == "divergence" for x in actions)
    assert h.get("guard.rogue") is not None


def _embedder_hive(monkeypatch):
    """A hive whose embedder is ATTACHED (a live server this process does not own) - the
    normal case when llama-server is started outside the agent."""
    import numpy as np
    from agent import model_introspect
    h = _hive()
    h.attach("embedder", "http://127.0.0.1:8081", role="embedder", model="bge")

    def fake_embed(texts, url=None, model=None, timeout=60.0):
        # distinct-but-close vectors: every agent is on-topic, nobody is divergent
        return np.array([[1.0, 0.1 * i] for i, _ in enumerate(texts)], dtype=float)

    monkeypatch.setattr(model_introspect, "embed", fake_embed)
    monkeypatch.setattr("agent.local_runtime.embed_url", lambda: None)   # nothing managed
    return h


def test_monitor_uses_an_attached_embedder(monkeypatch):
    """hive.monitor(embed=True) must use the embedder BEE, not only a locally-managed server."""
    h = _embedder_hive(monkeypatch)
    for a, b in [("planner", "coder"), ("coder", "reviewer"), ("reviewer", "planner")]:
        h.relay(a, b, "waiting on you")
    assert h.monitor(embed=True)["alignment_mode"] == "embedding"


def test_observe_requests_the_semantic_signal(monkeypatch):
    """react() only acts on divergence in embedding mode, so observe() must ask for it.
    Otherwise the divergence rule is permanently dead whenever an embedder is available."""
    h = _embedder_hive(monkeypatch)
    for a, b in [("planner", "coder"), ("coder", "reviewer"), ("reviewer", "planner")]:
        h.relay(a, b, "waiting on you")
    assert ReactiveHive(h).observe()["alignment_mode"] == "embedding"
