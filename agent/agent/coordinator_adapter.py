"""Adapter: turn hive tasks into coordinator work units by mapping the hive's task `kind` onto a
coordinator TYPE. Keeps the relational coordinator (rexgraph) domain-agnostic; this is the agent glue."""
from __future__ import annotations

# io_llm == I/O-bound, GIL-light work that runs IN-PROCESS on the thread lane: LLM calls, and also
# subprocess spawns / live-server attaches (they block on model load AND mutate hive state in place,
# so they MUST NOT run in a forkserver child where the mutation would be lost).
_IO = ("llm", "chat", "generate", "ask", "complete", "spawn", "attach")
_GPU = ("kernel", "heat", "greens", "block_cg", "matvec", "dirac")


def _to_type(kind: str) -> str:
    k = (kind or "").lower()
    if k.startswith("train:"):
        archetype = k.split(":", 1)[1]
        from agent.agent.foundry import _CPU_ONLY
        return "cpu_coordination" if archetype in _CPU_ONLY else "gpu_kernel"
    if any(s in k for s in _IO):
        return "io_llm"
    if any(s in k for s in _GPU):
        return "gpu_kernel"
    return "cpu_coordination"   # monitors / consensus / analysis / unknown default here


def work_units(tasks: list) -> list:
    return [{"id": t["id"], "type": _to_type(t.get("kind", "")),
             "fn": t["fn"], "weight": float(t.get("weight", 1.0))}
            for t in tasks]
