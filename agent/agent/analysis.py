"""Agent owned structural analysis orchestration.

The Agent pipeline is the live application orchestrator.  ``rexgraph.analysis`` remains
a dashboard/reference compatibility API, but Agent code must not route through it.
"""
from __future__ import annotations

from typing import Any, Callable

from agent.pipeline import AnalysisPipeline

_DEPTHS = {"quick", "standard", "full", "deep"}


def analyze(
    rex,
    *,
    depth: str = "standard",
    labels=None,
    draw: bool = True,
    draw_limit: int = 400,
    on_stage: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    """Run the live Agent analysis pipeline on one native complex."""
    if depth not in _DEPTHS:
        raise ValueError(f"unknown analysis depth {depth!r}; expected quick, standard, or full")
    pipeline = AnalysisPipeline(rex, draw=draw, draw_limit=draw_limit, labels=labels)
    if on_stage is not None:
        pipeline.on_stage(on_stage)
    return pipeline.run(depth=depth)


__all__ = ["analyze"]
