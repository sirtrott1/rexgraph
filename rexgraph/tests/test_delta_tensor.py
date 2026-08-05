"""The temporal delta tensor: existence and orientation as independent channels.

A relational complex's entries are composite binary: an existence condition in
{0,1} and an orientation in {+1,-1}. They are independent: a cell can persist while
its orientation reverses, which is a real event and not a weaker form of the cell
appearing. Differencing each condition separately is what makes the history a
tensor with two channels rather than one churn count.

That distinction was not represented. The delta kernel records sign changes, but
folds them in with weight changes under a single MODIFIED bucket, and nothing
exposed the two conditions apart. The checkpoint heuristic counted only born and
died, so a history made entirely of orientation reversals registered no churn at
all and never checkpointed.
"""

import numpy as np

from rexgraph.graph import RexGraph, TemporalRex


def _ring(nE=4):
    src = np.arange(nE, dtype=np.int32)
    tgt = ((np.arange(nE) + 1) % nE).astype(np.int32)
    return src, tgt


def _store(sign_history, nE=4):
    src, tgt = _ring(nE)
    tr = TemporalRex([])
    for s in sign_history:
        tr.append_snapshot(RexGraph(sources=src, targets=tgt,
                                    signs=np.array(s, np.int32)))
    return tr


#### the churn counter
def test_orientation_churn_counts_toward_checkpoints():
    """200 reversals used to produce a single checkpoint (the seed), so the delta
    chain grew without bound and reconstruct_at replayed all of it."""
    rng = np.random.default_rng(0)
    nE = 60
    src = rng.integers(0, 30, nE).astype(np.int32)
    tgt = ((src + 1 + rng.integers(0, 5, nE)) % 30).astype(np.int32)
    tr = TemporalRex([])
    signs = np.ones(nE, np.int32)
    for _ in range(200):
        signs = signs.copy()
        signs[rng.integers(0, nE)] *= -1
        tr.append_snapshot(RexGraph(sources=src, targets=tgt, signs=signs))

    assert len(tr._index_cp_times) > 1, (
        f"only {len(tr._index_cp_times)} checkpoint(s) after 200 orientation flips")


def test_checkpointing_does_not_change_what_is_reconstructed():
    """A checkpoint is an optimisation. It must not alter the history."""
    hist = [[1, 1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1],
            [-1, -1, -1, -1], [1, -1, -1, -1]]
    tr = _store(hist)
    for t, want in enumerate(hist):
        got = np.asarray(tr.reconstruct_at(t)._signs)
        got = np.ones(4, np.int32) if got.shape == () else got
        assert got.ravel().tolist() == want, f"t={t}"


#### the tensor
def test_orientation_reversal_is_recorded_as_an_orientation_event():
    tr = _store([[1, 1, 1, 1], [-1, 1, 1, 1]])
    d = tr.delta_tensor()
    assert list(d["t"]) == [1]
    assert list(d["existence"]) == [0], "nothing was born or died"
    assert list(d["orientation"]) == [-1], "the reversal was not recorded"


def test_a_born_cell_is_an_existence_event_not_an_orientation_one():
    """A cell coming into being has no previous orientation to have changed."""
    src, tgt = _ring(3)
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=src[:2], targets=tgt[:2]))
    tr.append_snapshot(RexGraph(sources=src, targets=tgt))
    d = tr.delta_tensor()
    assert list(d["existence"]) == [1]
    assert list(d["orientation"]) == [0]


def test_a_died_cell_is_an_existence_event():
    src, tgt = _ring(3)
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=src, targets=tgt))
    tr.append_snapshot(RexGraph(sources=src[:2], targets=tgt[:2]))
    d = tr.delta_tensor()
    assert list(d["existence"]) == [-1]
    assert list(d["orientation"]) == [0]


def test_the_two_channels_are_independent():
    """The whole point. A step that reverses one cell and adds another must show
    one event in each channel, not two of the same kind."""
    src, tgt = _ring(4)
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=src[:3], targets=tgt[:3],
                                signs=np.array([1, 1, 1], np.int32)))
    tr.append_snapshot(RexGraph(sources=src, targets=tgt,
                                signs=np.array([-1, 1, 1, 1], np.int32)))
    d = tr.delta_tensor()
    assert sorted(d["existence"].tolist()) == [0, 1]
    assert sorted(d["orientation"].tolist()) == [-1, 0]
    # and they are carried on different cells
    ex_key = d["key"][d["existence"] != 0][0]
    or_key = d["key"][d["orientation"] != 0][0]
    assert ex_key != or_key


def test_an_unchanged_step_produces_no_events():
    tr = _store([[1, -1, 1, 1], [1, -1, 1, 1]])
    d = tr.delta_tensor()
    assert len(d["t"]) == 0


def test_keys_are_stable_identities_across_time():
    """A cell that reverses at t=1 and reverses back at t=2 is the same cell."""
    tr = _store([[1, 1, 1, 1], [-1, 1, 1, 1], [1, 1, 1, 1]])
    d = tr.delta_tensor()
    assert list(d["t"]) == [1, 2]
    assert d["key"][0] == d["key"][1]
    assert list(d["orientation"]) == [-1, 1]


def test_the_tensor_survives_a_checkpoint_boundary():
    """Deltas are not stored at checkpoints, so the tensor cannot simply read the
    stored delta list: it has to be correct across a checkpoint too."""
    hist = [[1, 1, 1, 1]] + [[(-1) ** (i > k) for i in range(4)] for k in range(4)]
    tr = _store(hist)
    tr._checkpoint_threshold = 0.0          # force a checkpoint at every step
    tr2 = _store(hist)
    a, b = tr.delta_tensor(), tr2.delta_tensor()
    assert list(a["t"]) == list(b["t"])
    assert list(a["orientation"]) == list(b["orientation"])


def test_tensor_shape_and_dtypes_are_a_stable_contract():
    tr = _store([[1, 1, 1, 1], [-1, 1, 1, 1]])
    d = tr.delta_tensor()
    for key in ("t", "key", "existence", "orientation"):
        assert key in d
    n = len(d["t"])
    assert all(len(d[k]) == n for k in ("key", "existence", "orientation"))
    assert set(np.unique(d["existence"]).tolist()) <= {-1, 0, 1}
    assert set(np.unique(d["orientation"]).tolist()) <= {-1, 0, 1}


def test_dense_form_is_available_for_small_histories():
    """(T, n_cells, 2): the tensor as a tensor, for callers that want to contract
    over it rather than iterate events."""
    tr = _store([[1, 1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1]])
    dense, keys = tr.delta_tensor(dense=True)
    assert dense.shape == (tr.T, len(keys), 2)
    assert dense[0].sum() == 0, "t=0 has no predecessor, so no delta"
    assert dense[1, :, 1].sum() == -1


#### mutation as one event with a magnitude
def test_a_swap_sharing_a_vertex_is_one_mutation_not_two_events():
    """A cell dying as another is born, on the same vertices, is a topology
    mutating. Reading it as an unrelated death and an unrelated birth loses exactly
    the thing worth knowing."""
    src, tgt = _ring(4)
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=np.array([0, 1, 2], np.int32),
                                targets=np.array([1, 2, 3], np.int32)))
    # (2,3) dies, (2,0) is born: both bounded by vertex 2
    tr.append_snapshot(RexGraph(sources=np.array([0, 1, 2], np.int32),
                                targets=np.array([1, 2, 0], np.int32)))
    m = tr.mutations()
    assert len(m["t"]) == 1
    assert m["t"][0] == 1
    assert m["shared"][0] >= 1


def test_the_magnitude_is_how_much_boundary_survived():
    """Same currency as the face correspondence: an exact count of shared boundary,
    not a similarity score."""
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=np.array([0, 5], np.int32),
                                targets=np.array([1, 6], np.int32)))
    tr.append_snapshot(RexGraph(sources=np.array([0, 5], np.int32),
                                targets=np.array([2, 6], np.int32)))
    m = tr.mutations()
    # (0,1) -> (0,2) keeps vertex 0
    assert list(m["shared"]) == [1]


def test_an_unrelated_birth_and_death_is_not_a_mutation():
    """Sharing no boundary means nothing turned into anything."""
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=np.array([0, 4], np.int32),
                                targets=np.array([1, 5], np.int32)))
    tr.append_snapshot(RexGraph(sources=np.array([2, 4], np.int32),
                                targets=np.array([3, 5], np.int32)))
    m = tr.mutations()
    assert len(m["t"]) == 0, "unrelated cells were paired as a mutation"


def test_a_birth_with_no_death_is_not_a_mutation():
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=np.array([0], np.int32),
                                targets=np.array([1], np.int32)))
    tr.append_snapshot(RexGraph(sources=np.array([0, 1], np.int32),
                                targets=np.array([1, 2], np.int32)))
    assert len(tr.mutations()["t"]) == 0


def test_mutations_name_both_ends():
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=np.array([0, 5], np.int32),
                                targets=np.array([1, 6], np.int32)))
    tr.append_snapshot(RexGraph(sources=np.array([0, 5], np.int32),
                                targets=np.array([2, 6], np.int32)))
    m = tr.mutations()
    assert m["died_key"][0] != m["born_key"][0]
    d = tr.delta_tensor()
    died = {int(k) for k, e in zip(d["key"], d["existence"], strict=False) if e < 0}
    born = {int(k) for k, e in zip(d["key"], d["existence"], strict=False) if e > 0}
    assert int(m["died_key"][0]) in died and int(m["born_key"][0]) in born


def test_mutations_report_the_moment_on_the_real_clock():
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=np.array([0, 5], np.int32),
                                targets=np.array([1, 6], np.int32)), at=100.0)
    tr.append_snapshot(RexGraph(sources=np.array([0, 5], np.int32),
                                targets=np.array([2, 6], np.int32)), at=250.0)
    m = tr.mutations()
    assert list(m["when"]) == [250.0]


def test_each_cell_is_paired_at_most_once():
    """One death cannot be the origin of two births, or the count of what happened
    stops meaning anything."""
    tr = TemporalRex([])
    tr.append_snapshot(RexGraph(sources=np.array([0, 7], np.int32),
                                targets=np.array([1, 8], np.int32)))
    tr.append_snapshot(RexGraph(sources=np.array([0, 0, 7], np.int32),
                                targets=np.array([2, 3, 8], np.int32)))
    m = tr.mutations()
    assert len(set(m["died_key"].tolist())) == len(m["died_key"])
    assert len(set(m["born_key"].tolist())) == len(m["born_key"])
