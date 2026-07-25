"""
Tests for agent_complex - the agentic relational complex + monitor. Asserts the stable signals
(load-bearing centrality, cross-agent alignment ordering, query routing), not the binary flag.
Needs the compiled rexgraph core (RexGraph); skips cleanly if unavailable.
"""
import pytest

pytest.importorskip("rexgraph.graph")
from agent.agent_complex import AgentComplex      # noqa: E402


def _swarm():
    convo = [
        ("router", "bio", "tumor suppressor gene mutation apoptosis pathway"),
        ("bio", "router", "p53 tumor suppressor mutation disrupts apoptosis cell cycle"),
        ("router", "chem", "binding affinity inhibitor molecule receptor enzyme"),
        ("chem", "router", "inhibitor binds receptor high affinity blocking enzyme"),
        ("bio", "chem", "mutation affects receptor protein inhibitor targets"),
        ("chem", "bio", "receptor binding site altered by protein mutation"),
        ("drift", "router", "i enjoy pizza sunny weather beach vacation today"),
        ("drift", "bio", "favorite color blue long walks music movies"),
    ]
    return AgentComplex().add_messages([{"from": a, "to": b, "text": t} for a, b, t in convo])


def test_monitor_shapes_and_router_is_load_bearing():
    m = _swarm().monitor()
    assert m["n_agents"] >= 4 and m["n_interactions"] >= 4
    assert m["interaction_hodge"] is not None
    # the hubs (router + bio, both touching 3 agents) carry the load; drift is peripheral
    agents_by_load = m["agents"]  # already sorted by load_bearing desc
    top2 = {agents_by_load[0]["agent"], agents_by_load[1]["agent"]}
    assert "router" in top2
    assert agents_by_load[-1]["agent"] != "router"       # the coordinator is never least load-bearing
    # β₁ interaction cycles present (bio<->chem<->router coordination loop)
    assert m["deadlock_cycles"] >= 1


def test_drift_agent_has_lowest_alignment():
    m = _swarm().monitor()
    align = {a["agent"]: a["alignment"] for a in m["agents"]}
    # the off-topic (pizza/beach) agent's output diverges most from the swarm
    assert align["drift"] == min(align.values())
    assert align["drift"] < align["bio"] and align["drift"] < align["chem"]


def test_routing_surfaces_the_right_specialist():
    ac = _swarm()
    r1 = ac.route("receptor inhibitor binding affinity")
    r2 = ac.route("apoptosis tumor mutation")
    assert r1 and r1[0]["agent"] == "chem"        # chemistry query -> chem agent
    assert r2 and r2[0]["agent"] == "bio"         # biology query -> bio agent


def test_embedding_alignment_recognizes_disjoint_specialist():
    """A math specialist shares no vocabulary with bio/chem but is semantically technical.
    Semantic embeddings recognize it as aligned (a valid specialist) where the lexical signal
    nearly misses it."""
    import re
    import numpy as np
    convo = [
        ("router", "bio", "tumor suppressor gene mutation apoptosis cell"),
        ("bio", "router", "p53 tumor suppressor mutation disrupts apoptosis"),
        ("router", "chem", "inhibitor molecule receptor enzyme binding"),
        ("chem", "router", "inhibitor binds receptor affinity enzyme"),
        ("router", "math", "prove the theorem using the integral and eigenvalue decomposition"),
        ("math", "router", "the eigenvalue decomposition gives the integral via matrix diagonalization"),
        ("drift", "router", "i enjoy pizza sunny weather beach vacation"),
    ]
    ac = AgentComplex().add_messages([{"from": a, "to": b, "text": t} for a, b, t in convo])
    TECH = set("tumor suppressor gene mutation apoptosis cell p53 disrupts inhibitor molecule "
               "receptor enzyme binding affinity theorem integral eigenvalue decomposition matrix "
               "diagonalization prove".split())
    CAS = set("pizza sunny weather beach vacation enjoy".split())

    def mock_embed(texts):
        return np.array([[len(set(re.findall(r"[a-z]+", t.lower())) & TECH) + 1e-3,
                          len(set(re.findall(r"[a-z]+", t.lower())) & CAS) + 1e-3] for t in texts], dtype=float)

    lex = {a["agent"]: a["alignment"] for a in ac.monitor()["agents"]}
    memb = ac.monitor(embed_fn=mock_embed)
    emb = {a["agent"]: a["alignment"] for a in memb["agents"]}
    assert memb["alignment_mode"] == "embedding"
    assert emb["math"] > lex["math"] + 0.3        # embedding recognizes the disjoint specialist
    assert emb["drift"] < emb["math"]             # off-topic drift still clearly less aligned


def test_empty_and_singleton():
    assert AgentComplex().monitor()["n_agents"] == 0
    solo = AgentComplex().add_message("a", "a", "self note").monitor()   # no inter-agent edge
    assert solo["n_interactions"] == 0


def test_monitor_reports_rcfe_field():
    """The monitor surfaces the RCFE field: flat (no strain) on a pairwise complex, a real field
    once agents coordinate (a face); per-agent curvature localizes the geometry."""
    flat = AgentComplex().add_messages(
        [{"from": "a", "to": "b", "text": "x"}, {"from": "b", "to": "c", "text": "y"}]).monitor()
    assert flat["strain"] in (0.0, None)                     # a 1-complex is flat, no curvature
    triad = AgentComplex().add_messages(
        [{"from": "a", "to": "b", "text": "x"}, {"from": "b", "to": "c", "text": "y"},
         {"from": "c", "to": "a", "text": "z"}]).monitor()
    assert triad["strain"] is not None and triad["strain"] > 0    # coordination -> a field
    assert all("curvature" in ag for ag in triad["agents"])


def test_drift_tracker_flags_rising_curvature_and_falling_alignment():
    """A worker whose curvature climbs or alignment falls over snapshots is 'starting to detract'."""
    from agent.agent_complex import DriftTracker
    dt = DriftTracker()
    for i in range(5):
        dt.snapshot({"strain": 1.0 + 0.2 * i, "agents": [
            {"agent": "rise", "curvature": 0.2 + 0.15 * i, "alignment": 0.8},
            {"agent": "stable", "curvature": 0.5, "alignment": 0.8},
            {"agent": "fall", "curvature": 0.4, "alignment": 0.9 - 0.1 * i}]})
    drifting = set(dt.drifting())
    assert "rise" in drifting and "fall" in drifting and "stable" not in drifting
    assert dt.strain_trend() > 0 and dt.trends()["rise"]["curvature_slope"] > 0
