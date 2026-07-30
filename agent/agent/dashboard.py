"""agent.dashboard: the whole hive network as a live statistics snapshot.

Assembles every signal we already compute into one structured dashboard: the roster, the
coordination health (deadlocks, the Hodge split, per-worker load/coherence/curvature/alignment), the
networks the LMs forged, and - crucially - how information MOVES. Information propagation is signal
exchange on the tensor field of the hive: the directed message flow (who -> whom, how much) plus the
Hodge reading of it (is work draining, or circulating in a stuck loop). `hive_dashboard()` returns
the data; `render()` prints terminal panels; the server route serves it live.
"""
from __future__ import annotations

from typing import Any, Dict, List


def hive_dashboard(hive) -> Dict[str, Any]:
    """A full snapshot of the hive network's state and information flow."""
    m = hive.monitor()
    st = hive.status()
    from .foundry import hierarchy
    hier = hierarchy(hive)

    # information flow health: the directed coordination edges, read by the same Hodge machinery
    edges = m.get("edges", []) or []
    flow: Dict[str, Any] = {}
    if edges:
        try:
            from rexgraph import mesh_health
            flow = mesh_health([(e["from"], e["to"]) for e in edges],
                               [e.get("weight", 1) for e in edges])
        except Exception:
            flow = {}

    hodge = m.get("interaction_hodge") or {}
    return {
        "overview": {
            "bees": st["n_bees"], "queen": st["queen"], "embedder": st["embedder"],
            "workers": st["workers"],
            "controllers": [c["name"] for c in hier["controllers"]],
            "networks": [n["name"] for n in hier["networks"]],
        },
        "coordination": {
            "interactions": m.get("n_interactions"),
            "deadlocks": m.get("deadlock_cycles"),
            "draining": hodge.get("coherent"), "circulating": hodge.get("circulating"),
            "harmonic": hodge.get("persistent"), "strain": m.get("strain"),
        },
        "information_flow": {
            "edges": edges,
            "status": flow.get("status"), "draining": flow.get("draining"),
            "circulating": flow.get("circulating"), "health_ratio": flow.get("health_ratio"),
            "stuck_loops": flow.get("stuck_loops", []),
            "bottlenecks": flow.get("bottlenecks", []),
        },
        "workers": [{
            "name": a["agent"], "load_bearing": a.get("load_bearing"),
            "coherence": a.get("coherence"), "curvature": a.get("curvature"),
            "alignment": a.get("alignment"), "flag": a.get("flag"), "messages": a.get("messages"),
        } for a in m.get("agents", [])],
        "networks": hier["networks"],
    }


def render(dash: Dict[str, Any]) -> str:
    """Terminal panels for a dashboard snapshot."""
    o, c, f = dash["overview"], dash["coordination"], dash["information_flow"]
    L = []
    L.append("  ╔══ HIVE NETWORK ═══════════════════════════════════════════")
    L.append("  ║ bees %d   queen %s   controllers %s   networks %s"
             % (o["bees"], o["queen"], o["controllers"] or "-", o["networks"] or "-"))
    L.append("  ╟── coordination ───────────────────────────────────────────")
    L.append("  ║ interactions %s   deadlocks %s   strain %s"
             % (c["interactions"], c["deadlocks"], c["strain"]))
    L.append("  ║ flow: draining %s  circulating %s  harmonic %s"
             % (c["draining"], c["circulating"], c["harmonic"]))
    L.append("  ╟── information flow ───────────────────────────────────────")
    L.append("  ║ status %s   circulating %s   health_ratio %s"
             % (f.get("status"), f.get("circulating"), f.get("health_ratio")))
    for loop in (f.get("stuck_loops") or [])[:2]:
        L.append("  ║   stuck loop %s (%s)" % (loop.get("services"), loop.get("kind")))
    L.append("  ╟── workers ────────────────────────────────────────────────")
    for w in sorted(dash["workers"], key=lambda x: -(x.get("load_bearing") or 0))[:8]:
        L.append("  ║ %-12s load %-5s coh %-5s align %-5s %s"
                 % (w["name"], w.get("load_bearing"), w.get("coherence"),
                    w.get("alignment"), w.get("flag") or ""))
    if dash["networks"]:
        L.append("  ╟── networks (forged NNs) ──────────────────────────────────")
        for n in dash["networks"]:
            L.append("  ║ %-12s %s" % (n["name"], n["type"]))
    L.append("  ╚═══════════════════════════════════════════════════════════")
    return "\n".join(L)
