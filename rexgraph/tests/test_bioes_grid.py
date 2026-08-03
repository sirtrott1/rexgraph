"""BIOES per cell per moment: the grid.

O is the 0 of the existence condition. B/I/E/S all presuppose existence=1 and say
where in a contiguous life you are, so BIOES is the lifetime-position reading of the
existence channel -- not a separate scheme bolted on. Tagging TIMESTEPS by phase
could never use O, because phases partition the timeline and nothing is outside them.

Indexing cells against moments gives the grid: a row is what the whole complex is
doing at one moment, a column is one cell's life. Orientation rides alongside as its
own channel rather than inside the tag, because a cell that reverses has persisted.
"""

import numpy as np

from rexgraph.graph import RexGraph, TemporalRex

B, I, O, E, S = 0, 1, 2, 3, 4
LETTER = {B: "B", I: "I", O: "O", E: "E", S: "S"}


def _store(edge_sets, sign_sets=None):
    """Each entry is a list of (u, v) pairs present at that step."""
    tr = TemporalRex([])
    for k, edges in enumerate(edge_sets):
        src = np.array([e[0] for e in edges], np.int32)
        tgt = np.array([e[1] for e in edges], np.int32)
        kw = {}
        if sign_sets is not None:
            kw["signs"] = np.array(sign_sets[k], np.int32)
        tr.append_snapshot(RexGraph(sources=src, targets=tgt, **kw))
    return tr


def _trace(grid, key):
    col = list(grid["keys"]).index(key)
    return "".join(LETTER[int(t)] for t in grid["tags"][:, col])


def test_a_cell_present_throughout_is_bounded_by_B_and_E():
    tr = _store([[(0, 1)]] * 4)
    g = tr.bioes_grid()
    assert _trace(g, g["keys"][0]) == "BIIE"


def test_a_cell_present_for_one_moment_is_single():
    tr = _store([[(0, 1), (2, 3)], [(0, 1)], [(0, 1)]])
    g = tr.bioes_grid()
    # (2,3) exists only at t=0
    key = [k for k in g["keys"] if _trace(g, k).startswith("S")][0]
    assert _trace(g, key) == "SOO"


def test_absence_is_O():
    tr = _store([[(0, 1)], [(0, 1), (2, 3)], [(0, 1), (2, 3)]])
    g = tr.bioes_grid()
    key = [k for k in g["keys"] if _trace(g, k)[0] == "O"][0]
    assert _trace(g, key) == "OBE"


def test_a_flickering_cell_gets_two_spans():
    """The reason edge_intervals had to exist. first_seen/last_seen would report one
    unbroken life; two spans is what actually happened, exactly as NLP tags two
    entities in one sentence."""
    tr = _store([[(0, 1), (2, 3)], [(2, 3)], [(0, 1), (2, 3)]])
    g = tr.bioes_grid()
    flick = [k for k in g["keys"] if "O" in _trace(g, k)][0]
    assert _trace(g, flick) == "SOS"


def test_a_longer_flicker_bounds_each_span_separately():
    tr = _store([[(0, 1), (9, 9)], [(0, 1), (9, 9)], [(9, 9)],
                 [(0, 1), (9, 9)], [(0, 1), (9, 9)]])
    g = tr.bioes_grid()
    flick = [k for k in g["keys"] if "O" in _trace(g, k)][0]
    assert _trace(g, flick) == "BEOBE"


def test_every_moment_of_every_cell_is_tagged():
    tr = _store([[(0, 1)], [(0, 1), (2, 3)], [(2, 3)]])
    g = tr.bioes_grid()
    assert g["tags"].shape == (tr.T, len(g["keys"]))
    assert set(np.unique(g["tags"]).tolist()) <= {B, I, O, E, S}


def test_a_row_is_the_moment_and_a_column_is_a_life():
    tr = _store([[(0, 1)], [(0, 1), (2, 3)], [(2, 3)]])
    g = tr.bioes_grid()
    assert g["tags"].shape[0] == tr.T
    assert g["tags"].shape[1] == len(g["keys"])
    # the moment vector at t=1 has both cells present
    assert (g["tags"][1] != O).sum() == 2


def test_orientation_is_its_own_channel_not_part_of_the_tag():
    """A cell that reverses has persisted. Folding the reversal into the tag would
    collapse the two independent conditions back together."""
    tr = _store([[(0, 1)]] * 3, sign_sets=[[1], [-1], [-1]])
    g = tr.bioes_grid()
    assert _trace(g, g["keys"][0]) == "BIE", "a reversal changed the existence tag"
    assert list(g["orientation"][:, 0]) == [1, -1, -1]


def test_orientation_is_zero_where_the_cell_is_absent():
    tr = _store([[(0, 1), (2, 3)], [(2, 3)]], sign_sets=[[1, -1], [-1]])
    g = tr.bioes_grid()
    col = list(g["keys"]).index([k for k in g["keys"] if "O" in _trace(g, k)][0])
    assert g["orientation"][1, col] == 0


def test_moment_counts_summarise_each_row():
    """'what BIOES and when', read straight off the grid."""
    tr = _store([[(0, 1)], [(0, 1), (2, 3)], [(2, 3)]])
    g = tr.bioes_grid()
    m = g["moment"]
    assert m.shape == (tr.T, 5)
    assert m.sum(axis=1).tolist() == [len(g["keys"])] * tr.T
    assert m[0][B] == 1 and m[0][O] == 1


def test_an_empty_history_is_an_empty_grid():
    tr = TemporalRex([])
    g = tr.bioes_grid()
    assert g["tags"].shape[0] == 0
    assert len(g["keys"]) == 0


def test_the_grid_agrees_with_the_delta_tensor_on_when_things_change():
    """Two readings of the same history: a birth in the delta tensor is a B in the
    grid, a death is the moment after an E."""
    tr = _store([[(0, 1)], [(0, 1), (2, 3)], [(0, 1)]])
    g = tr.bioes_grid()
    d = tr.delta_tensor()
    born = {(int(t), int(k)) for t, k, e in zip(d["t"], d["key"], d["existence"], strict=False) if e > 0}
    for t, key in born:
        col = list(g["keys"]).index(key)
        assert g["tags"][t, col] in (B, S)
