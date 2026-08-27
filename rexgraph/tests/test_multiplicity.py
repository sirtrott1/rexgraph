"""Separating multiplicity from topology in the cycle space.

Two relations can be distinct and still have the same boundary: two occurrences of
one token are two spans, so two witnesses on one vertex. Their difference has zero
boundary, so it is a cycle, but it records an occurrence count rather than a hole.
On the Gutenberg store that is 39 to 85 percent of beta_1, so a cycle reading that
does not separate them is mostly reading multiplicity.
"""
import itertools

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.harmonic_sparse import (
    multiplicity_cycles,
    multiplicity_dimension,
    multiplicity_groups,
)


def _g(src, tgt):
    r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
    r._ensure_clean()
    return r


def test_parallel_relations_are_found_and_counted():
    """Three copies of one relation contribute two multiplicity cycles, not three."""
    r = _g([0, 0, 0, 1], [1, 1, 1, 2])
    groups = multiplicity_groups(r)
    assert len(groups) == 1
    idx, _ = groups[0]
    assert sorted(idx.tolist()) == [0, 1, 2]
    assert multiplicity_dimension(r) == 2


def test_a_reversed_relation_cancels_by_sum_not_difference():
    """The sign is load-bearing. (2,0) and (0,2) have the same support but opposite
    orientation, so they are negatives: their SUM has zero boundary and their
    difference does not. Dropping the sign emits a vector that is not a cycle."""
    r = _g([2, 0], [0, 2])
    groups = multiplicity_groups(r)
    assert len(groups) == 1
    idx, sign = groups[0]
    assert sign[0] * sign[1] < 0, "the pair is sign-flipped"
    C = np.asarray(multiplicity_cycles(r).todense())
    B1 = np.asarray(r.B1_dense, float)
    assert np.abs(B1 @ C).max() < 1e-12
    naive = np.zeros((2, 1))
    naive[0, 0], naive[1, 0] = 1.0, -1.0
    assert np.abs(B1 @ naive).max() > 0.5, "the sign-blind version is not a cycle"


@pytest.mark.parametrize("src,tgt,mult,genuine", [
    ([0, 0, 1, 2, 0], [1, 1, 2, 0, 2], 2, 1),        # a double edge and a triangle
    ([0, 1, 2], [1, 2, 0], 0, 1),                     # a bare triangle
    ([0, 0], [1, 1], 1, 0),                           # nothing but multiplicity
])
def test_the_split_adds_up_to_betti_one(src, tgt, mult, genuine):
    """multiplicity + genuine = beta_1, exactly, with beta_1 from the library."""
    r = _g(src, tgt)
    d = multiplicity_dimension(r)
    assert d == mult
    assert int(r.betti[1]) - d == genuine


@pytest.mark.parametrize("src,tgt", [
    ([0, 0, 1, 2, 0], [1, 1, 2, 0, 2]),
    ([0, 0, 0, 1, 1], [1, 1, 1, 2, 2]),
])
def test_the_cycles_close_and_are_independent(src, tgt):
    """Groups have disjoint support, so the differences within them are independent
    and the count is exact rather than an upper bound."""
    r = _g(src, tgt)
    C = np.asarray(multiplicity_cycles(r).todense())
    B1 = np.asarray(r.B1_dense, float)
    assert np.abs(B1 @ C).max() < 1e-12, "every column is a cycle"
    assert np.linalg.matrix_rank(C) == C.shape[1] == multiplicity_dimension(r)


def test_a_simple_complex_has_no_multiplicity():
    r = _g(*zip(*itertools.combinations(range(5), 2), strict=False))
    assert multiplicity_groups(r) == []
    assert multiplicity_dimension(r) == 0
    assert multiplicity_cycles(r).shape[1] == 0


def test_branching_relations_group_at_their_own_arity():
    """Arity-general: identical 3-ary relations group exactly as 2-ary ones do, and
    a relation over the same vertices with a DIFFERENT head is not the same column."""
    ptr = np.array([0, 3, 6, 9], np.int64)
    idx = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2], np.int64)   # third one re-heads
    r = RexGraph.from_hypergraph(ptr, idx)
    r._ensure_clean()
    groups = multiplicity_groups(r)
    assert len(groups) == 1, groups
    assert sorted(groups[0][0].tolist()) == [0, 1]
    assert multiplicity_dimension(r) == 1
    C = np.asarray(multiplicity_cycles(r).todense())
    assert np.abs(np.asarray(r.B1_dense, float) @ C).max() < 1e-12


def test_the_limit_bounds_the_column_count():
    """d runs to 1.5e6 on a full book, so materialising every cycle is not the
    default a caller should get by accident."""
    r = _g([0] * 20, [1] * 20)
    assert multiplicity_dimension(r) == 19
    assert multiplicity_cycles(r, limit=5).shape[1] == 5


def test_multiplicity_cycles_carry_no_winding_against_genuine_cycles():
    """The point of the split. A multiplicity cycle and a genuine one are different
    directions of ker(B1): pairing a gradient against either still gives zero, but
    the multiplicity part is what an occurrence count moves and topology does not."""
    r = _g([0, 0, 1, 2, 0], [1, 1, 2, 0, 2])
    B1 = np.asarray(r.B1_dense, float)
    C = multiplicity_cycles(r)
    rng = np.random.default_rng(0)
    for _ in range(4):
        g = B1.T @ rng.integers(-9, 10, B1.shape[0]).astype(float)
        assert np.abs(r.harmonic_winding(g, cycles=C)).max() < 1e-9


#### the fast grouping must agree with the obvious one
def _reference_groups(rex, min_size=2):
    """np.unique(axis=0) per arity block: slow, and obviously right."""
    from rexgraph.harmonic_sparse import _b1_csc
    B1 = _b1_csc(rex)
    B1.sort_indices()
    arity = np.diff(B1.indptr)
    out = []
    for k in np.unique(arity):
        k = int(k)
        if k == 0:
            continue
        cols = np.where(arity == k)[0]
        if cols.size < min_size:
            continue
        span = B1.indptr[cols][:, None] + np.arange(k)[None, :]
        idx = B1.indices[span].astype(np.int64)
        dat = B1.data[span].astype(float)
        sign = np.where(dat[:, 0] < 0, -1.0, 1.0)
        dat = np.round(dat * sign[:, None], 12)
        key = np.concatenate([idx.astype(float), dat], axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        for g in np.where(cnt >= min_size)[0]:
            out.append(sorted(cols[np.where(inv == g)[0]].tolist()))
    return sorted(out)


@pytest.mark.parametrize("seed", range(12))
def test_the_hash_grouping_agrees_with_unique(seed):
    """Differential, on random complexes with deliberate duplicates and reversals.
    The hash is what makes this affordable; agreement with the obvious method is
    what makes it trustworthy."""
    rng = np.random.default_rng(seed)
    nV = int(rng.integers(4, 12))
    m = int(rng.integers(6, 30))
    s = rng.integers(0, nV, m)
    t = rng.integers(0, nV, m)
    keep = s != t
    s, t = s[keep], t[keep]
    if s.size < 2:
        pytest.skip("degenerate draw")
    # force duplicates and reversals into the sample
    s = np.concatenate([s, s[:3], t[:2]])
    t = np.concatenate([t, t[:3], s[:2]])
    r = _g(s.tolist(), t.tolist())
    got = sorted(sorted(i.tolist()) for i, _ in multiplicity_groups(r))
    assert got == _reference_groups(r), (got, _reference_groups(r))


@pytest.mark.parametrize("seed", range(6))
def test_the_grouping_is_arity_general_under_random_branching(seed):
    """Same check where the relations are k-ary, so the block loop is exercised at
    more than one width."""
    rng = np.random.default_rng(100 + seed)
    nV = int(rng.integers(5, 10))
    ptr, idx = [0], []
    for _ in range(int(rng.integers(4, 12))):
        a = int(rng.integers(2, min(nV, 5) + 1))
        idx += [int(x) for x in rng.choice(nV, size=a, replace=False)]
        ptr.append(len(idx))
    span = ptr[1] - ptr[0]                       # duplicate the first relation twice
    for _ in range(2):
        idx += idx[ptr[0]:ptr[0] + span]
        ptr.append(len(idx))
    r = RexGraph.from_hypergraph(np.array(ptr, np.int64), np.array(idx, np.int64))
    r._ensure_clean()
    got = sorted(sorted(i.tolist()) for i, _ in multiplicity_groups(r))
    assert got == _reference_groups(r)
    C = np.asarray(multiplicity_cycles(r).todense())
    if C.shape[1]:
        assert np.abs(np.asarray(r.B1_dense, float) @ C).max() < 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_the_dimension_shortcut_agrees_with_the_groups(seed):
    """multiplicity_dimension skips building the groups (sum of size-1 is exactly
    columns - runs). It must still return what counting the groups would."""
    rng = np.random.default_rng(200 + seed)
    nV = int(rng.integers(4, 10))
    m = int(rng.integers(6, 24))
    s = rng.integers(0, nV, m)
    t = rng.integers(0, nV, m)
    keep = s != t
    s, t = s[keep], t[keep]
    if s.size < 3:
        pytest.skip("degenerate draw")
    s = np.concatenate([s, s[:3], t[:2]])
    t = np.concatenate([t, t[:3], s[:2]])
    r = _g(s.tolist(), t.tolist())
    g = multiplicity_groups(r)
    assert multiplicity_dimension(r) == sum(int(i.size) - 1 for i, _ in g)
    assert multiplicity_dimension(r, groups=g) == multiplicity_dimension(r)


def test_a_forced_hash_collision_is_repaired():
    """The fallback path, exercised directly: two DIFFERENT rows given the same hash
    must not be merged. _identical_runs cuts on the full rows, so they land in
    separate runs, and the repair fires because one hash then spans two runs."""
    from rexgraph.harmonic_sparse import _identical_runs
    idx = np.array([[0, 1], [0, 1], [2, 3], [0, 1]], dtype=np.int64)
    dat = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    order, bounds = _identical_runs(idx, dat)
    got = sorted(sorted(order[a:b].tolist())
                 for a, b in zip(bounds[:-1], bounds[1:], strict=False))
    assert got == [[0, 1, 3], [2]], got


#### the split of H1 itself, which is a quotient and not a subtraction
def test_a_face_can_fill_a_multiplicity_cycle():
    """Why multiplicity_dimension is not the H1 answer. W lives in Z1, not H1: put
    a face on a bigon and W still has dimension 1 while beta_1 is 0. Subtracting the
    chain-level count would report more multiplicity than there is homology."""
    from rexgraph.harmonic_sparse import multiplicity_homology_dimension, simple_cycle_dimension
    r = _g([0, 0], [1, 1])
    assert int(r.betti[1]) == 1 and multiplicity_dimension(r) == 1
    r.add_faces([[0, 1]], signs=[[1.0, -1.0]])
    r._ensure_clean()
    assert int(r.betti[1]) == 0
    assert multiplicity_dimension(r) == 1, "the chain-level subspace is still there"
    assert multiplicity_homology_dimension(r) == 0, "but it carries no homology"
    assert simple_cycle_dimension(r) == 0


@pytest.mark.parametrize("src,tgt,faces,signs", [
    ([0, 0], [1, 1], [[0, 1]], [[1.0, -1.0]]),
    ([0, 1, 2, 0, 0], [1, 2, 0, 1, 1], [[0, 1, 2]], [[1.0, 1.0, 1.0]]),
    ([0, 1, 2, 0, 0], [1, 2, 0, 1, 1], [[3, 4]], [[1.0, -1.0]]),
    ([0, 0, 0], [1, 1, 1], [[0, 1]], [[1.0, -1.0]]),
])
def test_the_homology_split_always_sums_to_betti_one(src, tgt, faces, signs):
    """The property that `dim_H_genuine` did not have. Z1/(W + B1) is isomorphic to
    H1 of the collapsed complex, so the two parts sum exactly, with faces or
    without."""
    from rexgraph.harmonic_sparse import multiplicity_homology_dimension, simple_cycle_dimension
    r = _g(src, tgt)
    r.add_faces(faces, signs=signs)
    r._ensure_clean()
    b1 = int(r.betti[1])
    m = multiplicity_homology_dimension(r)
    s = simple_cycle_dimension(r)
    assert m >= 0 and s >= 0
    assert m + s == b1, (m, s, b1)
    assert m <= multiplicity_dimension(r), "homology cannot exceed the chain space"


def test_the_collapse_is_a_chain_map():
    """pi has to commute with the boundary or X' is a relabelling rather than a
    complex: B1' pi = B1, and B2' = pi B2 keeps B1' B2' = 0."""
    from rexgraph.harmonic_sparse import _b1_csc, collapse_map
    r = _g([0, 1, 2, 0, 0], [1, 2, 0, 1, 1])
    r.add_faces([[0, 1, 2]], signs=[[1.0, 1.0, 1.0]])
    r._ensure_clean()
    pi, keep = collapse_map(r)
    B1 = _b1_csc(r)
    B1p = B1[:, keep]
    assert np.abs((B1p @ pi - B1).toarray()).max() < 1e-12, "B1' pi = B1"
    B2 = np.asarray(r.B2_hodge, dtype=float)
    B2p = pi @ B2
    assert np.abs(np.asarray(B1p.todense()) @ B2p).max() < 1e-12, "B1' B2' = 0"


def test_a_complex_with_no_repeats_is_its_own_collapse():
    from rexgraph.harmonic_sparse import simple_cycle_dimension
    r = _g(*zip(*itertools.combinations(range(5), 2), strict=False))
    assert simple_cycle_dimension(r) == int(r.betti[1]) == 6
