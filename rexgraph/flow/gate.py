"""rexgraph.flow.gate: the MalaughGate, a condensed scalar gate on the Malaugh
harmonic log entropy H_T.

The gate stays quiet during expected change (steady growth of the complex
produces a steady, predictable change magnitude in H_T) and fires only when
the change magnitude itself is a SURPRISE relative to the recent baseline.
Closing a cycle (adding an edge between two vertices that already exist,
introducing no new vertex) is such a surprise: the change in H_T collapses
to something anomalously small compared to the steady leaf growth baseline,
even though the edit to the complex is not obviously small. The gate reacts
to the anomaly in the change PATTERN, never to the raw size of the change.

Everything here is scalar bookkeeping: one prior H_T value and a running
list of past absolute deltas of real changes (a median/MAD fence over that
history). No eigendecomposition, no dense operator. The fence itself is
O(len(history)) per real-change step, since observe() recomputes the
median/MAD over the full growing history each time rather than maintaining
an incremental statistic; a bounded/windowed history is a follow-on for
long-lived streams where that history would otherwise grow without limit.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from rexgraph.scale_propagator import malaugh_quantities

__all__ = ["malaugh_entropy", "MalaughGate"]


def malaugh_entropy(rex) -> float:
    """The Malaugh topology harmonic log entropy H_T for one rex snapshot.

    H_T is finite on any complex that has at least one edge (it depends only
    on T = B1^T B1, never on faces), so it is the stable channel to track
    across a stream of snapshots that may or may not have faces. Raises
    ValueError if H_T comes back nan (e.g. an empty complex with trace 0)."""
    h_t = malaugh_quantities(rex)["H_T"]
    if h_t != h_t:  # nan check without importing math.isnan
        raise ValueError("malaugh_entropy: H_T is nan for this rex (empty or degenerate complex)")
    return float(h_t)


class MalaughGate:
    """A condensed scalar gate that fires on a surprise in the change pattern
    of H_T, not on a large change.

    Each call to observe() computes the current H_T, compares it against the
    previous observation to get a signed delta, and checks the MAGNITUDE of
    that delta against a running median/MAD fence built from the magnitudes
    of past REAL changes (delta ~ 0, i.e. a no-op resubmission of the same
    complex, is reported but never folded into the fence history).

    An event fires only once there is a baseline (warmup real changes seen)
    and the current change magnitude sits more than fence_k robust deviations
    away from the median of the past change magnitudes, in either direction.
    That means both an anomalously LARGE and an anomalously SMALL change
    magnitude can fire the gate; the verified case for this subsystem is a
    cycle close producing an anomalously small delta against a steady
    leaf-growth baseline.

    Single-use instance: a MalaughGate carries state (_prev, _hist) across
    calls to observe(). It is meant to be run once per stream; reusing one
    instance across separate streams blends the baseline from the first
    stream into the second instead of starting fresh.
    """

    def __init__(self, fence_k: float = 3.0, warmup: int = 3, eps: float = 1e-9):
        self.fence_k = float(fence_k)
        self.warmup = int(warmup)
        self.eps = float(eps)
        self._prev: Optional[float] = None
        self._hist: List[float] = []

    def observe(self, rex) -> Dict[str, object]:
        h_t = malaugh_entropy(rex)

        if self._prev is None:
            self._prev = h_t
            return {"H_T": h_t, "delta": 0.0, "event": False}

        delta = h_t - self._prev
        self._prev = h_t
        mag = abs(delta)

        event = False
        if mag > self.eps:
            if len(self._hist) >= self.warmup:
                median = float(np.median(self._hist))
                mad = float(np.median(np.abs(np.asarray(self._hist) - median)))
                mad = max(mad, self.eps)
                event = abs(mag - median) > self.fence_k * mad
            self._hist.append(mag)

        return {"H_T": h_t, "delta": float(delta), "event": bool(event)}
