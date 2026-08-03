import numpy as np

from rexgraph.flow import MalaughGate
from rexgraph.graph import RexGraph


def _rex(src, tgt):
    return RexGraph(sources=np.asarray(src, np.int32), targets=np.asarray(tgt, np.int32))


# steady leaf growth (t0..t3) then a cycle-close (t4, an edge between existing vertices) then a leaf (t5)
_STREAM = [
    ([0, 0, 1], [1, 2, 3]),
    ([0, 0, 1, 2], [1, 2, 3, 4]),
    ([0, 0, 1, 2, 3], [1, 2, 3, 4, 5]),
    ([0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]),
    ([0, 0, 1, 2, 3, 4, 4], [1, 2, 3, 4, 5, 6, 0]),   # cycle-close: the surprise
    ([0, 0, 1, 2, 3, 4, 4, 5], [1, 2, 3, 4, 5, 6, 0, 7]),
]


def test_gate_fires_only_on_the_surprising_step():
    g = MalaughGate()
    events = [g.observe(_rex(s, t))["event"] for s, t in _STREAM]
    assert events[0] is False                 # first observation is never an event (no baseline)
    assert sum(events) == 1                    # exactly one surprise in this stream
    assert events[4] is True                   # and it is the cycle-close step, not the steady leaves


def test_gate_quiet_on_noop():
    g = MalaughGate()
    r = _rex([0, 0, 1, 2, 3], [1, 2, 3, 4, 5])
    g.observe(r)
    o = g.observe(r)                           # identical complex -> delta ~ 0 -> never an event
    assert abs(o["delta"]) < 1e-9 and o["event"] is False
