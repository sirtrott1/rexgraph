"""rexgraph.mesh_health: draining-vs-circulating flow, loop localization, bottlenecks."""
import numpy as np
import pytest

from rexgraph import harmonic_health, mesh_health
from rexgraph.graph import RexGraph
from rexgraph.mesh_health import mesh_health as mh


def _mesh():
    tier = {"gateway": 0, "auth": 1, "catalog": 1, "orders": 1, "inventory": 2,
            "pricing": 2, "payments": 2, "shipping": 2, "ledger": 3, "notify": 3}
    calls = [("gateway", "auth"), ("gateway", "catalog"), ("gateway", "orders"),
             ("orders", "inventory"), ("orders", "payments"), ("orders", "shipping"),
             ("inventory", "pricing"), ("payments", "ledger"), ("payments", "notify"),
             ("shipping", "notify"), ("catalog", "pricing"), ("pricing", "orders")]
    healthy = np.array([tier[b] - tier[a] for a, b in calls], float) * 40.0
    return tier, calls, healthy


def test_exported_from_package():
    assert mesh_health is mh


def test_healthy_flow_all_draining():
    tier, calls, healthy = _mesh()
    r = mh(calls, healthy)
    assert r["status"] == "draining"   # cycles exist but the harmonic field vanishes on this flow
    assert r["circulating"] < 1e-6
    assert r["draining"] > 0.999
    assert r["stuck_loops"] == []
    assert r["n_cycles"] == 3          # structural cycles exist but carry no stuck load


def test_harmonic_health_splits_the_character():
    # a triangle carries harmonic content; the readout is exact-structural
    rex = RexGraph.from_graph(np.array([0, 1, 2], np.int32), np.array([1, 2, 0], np.int32))
    hh = harmonic_health(rex, np.array([1.0, 1.0, 1.0]))
    assert hh["dim_H"] == 1
    assert hh["harm_per_edge"].shape == (3,)
    assert hh["frustration_total"] is not None            # channel split present, no threshold

    # an acyclic complex has no harmonic content -> no health ratio (dim_H == 0)
    tree = harmonic_health(RexGraph.from_graph(np.array([0, 0], np.int32), np.array([1, 2], np.int32)))
    assert tree["dim_H"] == 0 and tree["health_ratio"] is None


def test_health_ratio_is_the_frustration_over_coparticipation_channels():
    """health_ratio must weigh L_SG against L_C, the two channels it is named for.

    Reading L1_down/L_O instead gives the constant 1.0 on every unweighted complex,
    because those two channels share a diagonal and chi is built from diagonals only.
    """
    # K4: every edge carries harmonic content, and L_SG/L_C differs from 1
    src = np.array([0, 0, 0, 1, 1, 2], np.int32)
    tgt = np.array([1, 2, 3, 2, 3, 3], np.int32)
    rex = RexGraph.from_graph(src, tgt)
    hh = harmonic_health(rex, np.ones(6))

    names = list(rex.hat_names)
    chi = np.asarray(rex.structural_character) * np.asarray(rex._rl4_sparse.diagonal())[:, None]
    harm = np.abs(hh["harm_per_edge"])
    expected = (float((harm * chi[:, names.index("L_SG")]).sum())
                / float((harm * chi[:, names.index("L_C")]).sum()))

    assert hh["health_ratio"] == pytest.approx(expected, rel=1e-9)
    assert hh["health_ratio"] != pytest.approx(1.0, abs=1e-9)   # not the degenerate pair


def test_health_ratio_channels_are_resolved_by_name_not_position():
    """The two channels must be looked up in hat_names, so a different channel
    ordering or count cannot silently shift which pair is reported."""
    src = np.array([0, 0, 0, 1, 1, 2], np.int32)
    tgt = np.array([1, 2, 3, 2, 3, 3], np.int32)
    rex = RexGraph.from_graph(src, tgt)
    hh = harmonic_health(rex, np.ones(6))

    names = list(rex.hat_names)
    chi = np.asarray(rex.structural_character) * np.asarray(rex._rl4_sparse.diagonal())[:, None]
    harm = np.abs(hh["harm_per_edge"])
    # the totals are reported rounded to 6 decimals, so compare at that precision
    assert hh["frustration_total"] == pytest.approx(
        float((harm * chi[:, names.index("L_SG")]).sum()), abs=5e-7)
    assert hh["coparticipation_total"] == pytest.approx(
        float((harm * chi[:, names.index("L_C")]).sum()), abs=5e-7)


def test_retry_storm_is_flagged_and_localized():
    tier, calls, healthy = _mesh()
    storm = healthy.copy()
    for a, b in [("orders", "inventory"), ("inventory", "pricing"), ("pricing", "orders")]:
        storm[calls.index((a, b))] += 220.0
    r = mh(calls, storm)
    assert r["status"] == "circulating"
    assert r["circulating"] > 0.9
    # the character of the circulation is reported, and the loop is classified
    assert "health_ratio" in r and r["frustration"] is not None
    assert r["stuck_loops"][0]["kind"] in ("irreducible", "fillable")
    assert len(r["stuck_loops"]) == 1
    loop = r["stuck_loops"][0]
    assert set(loop["services"]) == {"orders", "inventory", "pricing"}
    # a benign path off the same node is not part of the stuck loop
    involved = {(e["from"], e["to"]) for e in loop["edges"]}
    assert ("orders", "payments") not in involved


def test_reports_coherence_and_implied_groups():
    tier, calls, healthy = _mesh()
    storm = healthy.copy()
    for a, b in [("orders", "inventory"), ("inventory", "pricing"), ("pricing", "orders")]:
        storm[calls.index((a, b))] += 220.0
    r = mh(calls, storm)
    assert r["coherence"] and all("kappa" in c and "node" in c for c in r["coherence"])
    # least-coherent first (structural centrality ordering)
    assert r["coherence"][0]["kappa"] <= r["coherence"][-1]["kappa"]
    assert isinstance(r["implied_groups"], list)              # void-derived completion candidates


def test_early_warning_before_saturation():
    tier, calls, healthy = _mesh()
    lo = healthy.copy()
    for a, b in [("orders", "inventory"), ("inventory", "pricing"), ("pricing", "orders")]:
        lo[calls.index((a, b))] += 30.0     # small circulating load
    r = mh(calls, lo)
    # the signal is the APPEARANCE of a nonzero harmonic field, not crossing a tuned band
    assert r["status"] == "circulating"
    assert 0.0 < r["circulating"] < 1.0


def test_labels_dedup_and_selfloops():
    edges = [("a", "b"), ("b", "c"), ("c", "a"),   # a 3-cycle
             ("a", "b"),                            # duplicate -> aggregated
             ("a", "a")]                            # self-loop -> dropped
    flow = [1.0, 1.0, 1.0, 1.0, 5.0]
    r = mh(edges, flow)
    assert r["n_nodes"] == 3
    assert r["n_edges"] == 3                        # dup merged, self-loop gone
    assert r["circulating"] > 0.0                   # a pure cycle circulates
    assert r["bottlenecks"]                          # centrality reported


def test_empty_is_acyclic():
    r = mh([], None)
    assert r["status"] == "acyclic"
    assert r["n_edges"] == 0 and r["stuck_loops"] == []


def test_default_uniform_flow_on_dag_has_no_stuck_loops():
    # a tree/DAG with uniform flow: cycles from undirected structure carry no net circulation beyond harmonic
    r = mh([("root", "a"), ("root", "b"), ("a", "leaf"), ("b", "leaf")])
    assert r["n_cycles"] == 1
    # status is structural: acyclic (no cycles), draining (harmonic ~ 0), or circulating (nonzero)
    assert r["status"] in ("draining", "circulating")          # a cycle exists, so not acyclic
    assert isinstance(r["circulating"], float)
