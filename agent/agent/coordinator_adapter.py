"""Adapter: turn hive tasks into coordinator work units by mapping the hive's task `kind` onto a
coordinator TYPE. Keeps the relational coordinator (rexgraph) domain-agnostic; this is the agent glue."""
from __future__ import annotations

_IO = ("llm", "chat", "generate", "ask", "complete")
_GPU = ("kernel", "heat", "greens", "block_cg", "matvec", "dirac")


def _to_type(kind: str) -> str:
    k = (kind or "").lower()
    if any(s in k for s in _IO):
        return "io_llm"
    if any(s in k for s in _GPU):
        return "gpu_kernel"
    return "cpu_coordination"   # monitors / consensus / analysis / unknown default here


def work_units(tasks: list) -> list:
    return [{"id": t["id"], "type": _to_type(t.get("kind", "")),
             "fn": t["fn"], "weight": float(t.get("weight", 1.0))}
            for t in tasks]
