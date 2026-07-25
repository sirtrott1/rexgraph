"""Output/display settings for schema / ontology diagnosis.

There are (deliberately) no *decision* thresholds here. Every structural verdict is
driven by EXACT integer invariants - the harmonic dimension β₁ (persistent unfilled
cycles), rank(B₂) (co-participation-filled/curl cycles), and the combinatorial
directed feedback-arc cut (orderability) - and the one continuous classification
(conflict tables) uses a data-adaptive Tukey fence, not a fixed cutoff. Those are
mathematical facts, so there is nothing to tune.

What remains are genuine *output* preferences (how much to enumerate/show), which a
user legitimately toggles via `REXGRAPH_DIAG_*` - they change the size of the report,
never a verdict.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class DiagnosisOutput:
    """Output-size preferences for the diagnosis report (not decision thresholds)."""

    #: Maximum number of distinct circular-dependency chains to enumerate as
    #: examples. A display bound on the actionable output - it does not affect any
    #: verdict (the existence of directed cycles is decided exactly by the cut).
    max_cycles: int = _envi("REXGRAPH_DIAG_MAX_CYCLES", 20)


#: Process-wide output settings (read env once at import).
THRESHOLDS = DiagnosisOutput()
