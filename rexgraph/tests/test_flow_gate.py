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


#### what the gate can and cannot see, and why a benchmark got it wrong ########
def test_H_T_of_disjoint_relations_is_exactly_ln_n_at_any_arity():
    """n relations sharing no vertex have H_T = ln(n) EXACTLY: a function of the
    count alone, carrying nothing about content or arity.

    This is why a stream of disjoint snapshots can never fire the gate: the delta
    sequence is ln(n+1) - ln(n), smooth and monotone, so no change magnitude is ever a
    surprise. A benchmark that fed the gate disjoint turns was measuring arithmetic.
    """
    import numpy as np
    from rexgraph.graph import RexGraph
    from rexgraph.flow.gate import malaugh_entropy

    for k in (2, 3, 5):
        for n in (1, 2, 3, 5, 8, 13):
            ptr = np.arange(n + 1, dtype=np.int64) * k
            idx = np.arange(n * k, dtype=np.int64)        # no vertex shared
            h = malaugh_entropy(RexGraph.from_hypergraph(ptr, idx))
            assert abs(h - float(np.log(n))) < 1e-12, (k, n, h)


def _stream(extra):
    """Eight steady-growth snapshots, then one more built from `extra` term ids."""
    import numpy as np
    from rexgraph.graph import RexGraph

    turns = []
    for i in range(8):
        turns.append([3 * i, 3 * i + 1, 3 * i + 2] + ([3 * i - 1] if i else []))
    turns.append(list(extra))
    out = []
    for j in range(1, len(turns) + 1):
        flat, ptr = [], [0]
        for t in turns[:j]:
            flat.extend(t); ptr.append(len(flat))
        out.append(RexGraph.from_hypergraph(np.asarray(ptr, np.int64),
                                            np.asarray(flat, np.int64)))
    return out


def test_the_event_is_a_cycle_close_which_means_CONTINUATION_not_departure():
    """The gate's documented case: an edit over vertices that already exist collapses
    the H_T delta against a steady leaf-growth baseline.

    The POLARITY is the part a caller gets wrong. In a conversation complex a cycle
    close is a turn RETURNING to what was already said, so the event means continuation.
    Reading it as a topic change inverts the signal.
    """
    from rexgraph.flow import MalaughGate

    g = MalaughGate(warmup=4)
    events = [g.observe(r)["event"] for r in _stream([0, 4, 8, 12])]
    assert events[-1] is True                      # cycle close: adds NO new vertex
    assert not any(events[:-1])                    # steady growth stayed quiet

    g2 = MalaughGate(warmup=4)
    fresh = [g2.observe(r)["event"] for r in _stream([90, 91, 92, 93])]
    assert not any(fresh)                          # leaf growth is never a surprise
