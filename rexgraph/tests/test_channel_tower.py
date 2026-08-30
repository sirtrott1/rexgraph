"""The four channel diagonals at any arity, in O(nnz), by accumulating at the vertex.

`channel_diagonals`' closed form is exact for a signed pairwise unweighted complex and
says so; past that the disagreement count it uses is standing in for a magnitude, and
the magnitude accumulates at every arity. Both off-diagonal channels sum over pairs
that share a vertex, and a pair contributes only there, so the sum reorders onto the
vertex and no pair is ever formed.
"""
import numpy as np
import pytest
from fractions import Fraction as F

from rexgraph.core._channel_tower import channel_diagonals_any_arity as fast
from rexgraph.graph import RexGraph
from rexgraph.rational_trig import exact_channel_diagonals
from rexgraph.sparse_character import channel_diagonals, closed_form_applies


NAMES = ("L1_down", "L_O", "L_SG", "L_C")

CASES = {
    "pairwise triangle":  [(0, 1), (1, 2), (2, 0)],
    "pairwise with legs": [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)],
    "witness and pairs":  [(3,), (0, 1), (1, 2), (2, 0)],
    "witness only":       [(0,), (1,), (2,)],
    "5-ary with legs":    [(0, 1, 2, 3, 4), (0, 5), (1, 6), (2, 7)],
    "mixed arity 1 to 5": [(3,), (0, 1), (1, 2, 5), (0, 2, 4, 6), (2, 3), (1, 4, 5, 6, 7)],
    "branching only":     [(0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 3, 4)],
    "star":               [(0, i) for i in range(1, 7)],
    "one wide relation":  [(0, 1, 2, 3, 4, 5, 6, 7)],
}


def _rex(cells):
    ptr, idx = [0], []
    for c in cells:
        idx.extend(c)
        ptr.append(len(idx))
    r = RexGraph.from_hypergraph(np.array(ptr, np.int32), np.array(idx, np.int32))
    r._ensure_clean()
    return r


def _exact(r):
    d, _ = exact_channel_diagonals(r)
    return {k: np.array([float(F(x)) for x in d[k]]) for k in NAMES}


@pytest.mark.parametrize("name", list(CASES))
def test_the_kernel_matches_the_exact_rational_tower(name):
    """Including the cases the pairwise derivation refuses, which is the point."""
    r = _rex(CASES[name])
    T, G, Fq, C = fast(np.asarray(r._boundary_ptr), np.asarray(r._boundary_idx), int(r.nV))
    want = _exact(r)
    for got, key in ((T, "L1_down"), (G, "L_O"), (Fq, "L_SG"), (C, "L_C")):
        assert np.allclose(got, want[key]), (name, key, got, want[key])


@pytest.mark.parametrize("name", list(CASES))
def test_the_public_entry_point_returns_the_same(name):
    r = _rex(CASES[name])
    d = channel_diagonals(r)
    assert d is not None, "the tower now answers at any arity"
    want = _exact(r)
    for key in NAMES:
        assert np.allclose(d[key], want[key]), (name, key)


def test_a_witness_joins_the_positive_mass_not_the_head():
    """A witness column is (+1) and does not follow the head rule. Reading it as a head
    puts it in the wrong accumulator and only F moves, which is the same failure the
    exact tower had before it was fixed."""
    r = _rex([(0,), (0, 1), (0, 2)])
    d = channel_diagonals(r)
    want = _exact(r)
    assert np.allclose(d["L_SG"], want["L_SG"])
    assert np.allclose(d["L1_down"], want["L1_down"])


def test_T_and_G_agree_because_squaring_kills_the_sign():
    for name in CASES:
        r = _rex(CASES[name])
        d = channel_diagonals(r)
        assert np.array_equal(d["L1_down"], d["L_O"]), name


def test_T_is_one_plus_the_share_and_needs_no_accumulator():
    """1 + 1/(k-1) at arity k, and 1 at a witness: the support size decides it."""
    r = _rex([(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)])
    d = channel_diagonals(r)
    assert np.allclose(d["L1_down"], [1.0, 2.0, 1.5, 1 + 1 / 3])


def test_it_agrees_with_assembling_the_channels():
    """The path this replaces: assembling costs sum_v deg(v)^2 to read nE numbers."""
    from rexgraph.sparse_character import build_sparse_channels
    r = _rex(CASES["mixed arity 1 to 5"])
    d = channel_diagonals(r)
    asm = {n: L.diagonal() for n, L in dict(build_sparse_channels(r)).items()}
    for k in asm:
        if k in d:
            assert np.allclose(d[k], asm[k]), k


def test_edge_weighting_is_carried_not_refused():
    cells = CASES["mixed arity 1 to 5"]
    ptr, idx = [0], []
    for c in cells:
        idx.extend(c)
        ptr.append(len(idx))
    w = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.25])
    r = RexGraph.from_hypergraph(np.array(ptr, np.int32), np.array(idx, np.int32))
    r = RexGraph(boundary_ptr=np.array(ptr, np.int32),
                 boundary_idx=np.array(idx, np.int32), w_E=w)
    r._ensure_clean()
    T, G, Fq, C = fast(np.asarray(r._boundary_ptr), np.asarray(r._boundary_idx),
                       int(r.nV), w)
    # T scales as w^2, which is the definition and needs no reference to run
    k = np.diff(np.asarray(r._boundary_ptr))
    base = np.where(k == 1, 1.0, 1.0 + 1.0 / np.maximum(k - 1, 1))
    assert np.allclose(T, w ** 2 * base)


def test_a_vertex_weighting_is_refused_rather_than_approximated():
    """There diag(G) stops equalling diag(T), so the channels separate and the caller
    has to assemble. Refusing is the honest answer, not a fallback value."""
    from rexgraph.sparse_character import _any_arity_diagonals
    r = _rex(CASES["branching only"])
    r.w_V = np.full(int(r.nV), 2.0)
    try:
        assert _any_arity_diagonals(r) is None
    finally:
        del r.w_V


def test_it_answers_exactly_where_the_pairwise_derivation_refuses():
    """The point of the kernel. `closed_form_applies` is False on every branching or
    witness-carrying complex, and those used to fall through to assembling."""
    refused = [n for n in CASES if not closed_form_applies(_rex(CASES[n]))]
    assert refused, "some case must exercise the branching path"
    for name in refused:
        r = _rex(CASES[name])
        d = channel_diagonals(r)
        assert d is not None, name
        want = _exact(r)
        for key in NAMES:
            assert np.allclose(d[key], want[key]), (name, key)


def test_an_empty_complex_reads_empty():
    ptr = np.array([0], np.int32)
    idx = np.array([], np.int32)
    T, G, Fq, C = fast(ptr, idx, 0)
    assert T.shape == G.shape == Fq.shape == C.shape == (0,)


#### the lanes
def test_the_tower_is_reachable_through_every_registered_lane():
    """Same registry as the ternary operator, so a new architecture is a register_op
    call and nothing in the tower moves."""
    from rexgraph import compute
    import rexgraph.sparse_character  # noqa: F401  (registers cpu/openmp)
    lanes = [e["backends"] for e in compute.ops() if e["name"] == "channel_tower"]
    assert lanes and "cpu" in lanes[0] and "openmp" in lanes[0]
    r = _rex(CASES["mixed arity 1 to 5"])
    want = _exact(r)
    bp, bi = np.asarray(r._boundary_ptr), np.asarray(r._boundary_idx)
    for lane in lanes[0]:
        out = compute.dispatch("channel_tower", bp, bi, int(r.nV), None, prefer=lane)
        for got, key in zip(out, NAMES, strict=True):
            assert np.allclose(got, want[key]), (lane, key)


def test_the_thread_count_is_a_scheduling_choice_not_a_numerical_one():
    """Each thread owns the vertices it writes, so widening changes nothing."""
    r = _rex(CASES["mixed arity 1 to 5"])
    bp, bi = np.asarray(r._boundary_ptr), np.asarray(r._boundary_idx)
    ref = fast(bp, bi, int(r.nV), None, 1)
    for t in (2, 4, 8):
        for got, want in zip(fast(bp, bi, int(r.nV), None, t), ref, strict=True):
            assert np.array_equal(got, want)


def test_a_supplied_transpose_gives_the_same_answer():
    """It is reusable across readings of one complex, which is why it is an argument."""
    from rexgraph.core._channel_tower import transpose_incidence
    r = _rex(CASES["branching only"])
    bp, bi = np.asarray(r._boundary_ptr), np.asarray(r._boundary_idx)
    tp = transpose_incidence(bp, bi, int(r.nV))
    for got, want in zip(fast(bp, bi, int(r.nV), None, 4, tp),
                         fast(bp, bi, int(r.nV), None, 1), strict=True):
        assert np.array_equal(got, want)


def test_the_transpose_places_every_entry_under_its_vertex():
    from rexgraph.core._channel_tower import transpose_incidence
    r = _rex(CASES["mixed arity 1 to 5"])
    bp, bi = np.asarray(r._boundary_ptr), np.asarray(r._boundary_idx)
    vptr, owner, is_head = transpose_incidence(bp, bi, int(r.nV))
    assert vptr[-1] == len(bi)
    for v in range(int(r.nV)):
        for q in range(vptr[v], vptr[v + 1]):
            e = owner[q]
            span = list(bi[bp[e]:bp[e + 1]])
            assert v in span
            assert bool(is_head[q]) == (span[0] == v and span.count(v) == 1 or
                                        (span[0] == v))


def test_the_parallel_transpose_is_byte_identical_to_the_serial_one():
    """Not "close": the fill order inside a bucket has to be exactly the serial one.

    The accumulation sums float magnitudes per vertex, so reordering a bucket changes
    the last bits of every reading built on it. Each thread takes a contiguous range of
    RELATIONS, so its entries all precede the next thread's inside every bucket, which
    is what makes the split stable rather than merely correct.
    """
    from rexgraph.core._channel_tower import transpose_incidence
    rng = np.random.default_rng(0)
    nV, nE = 400, 2000
    ar = rng.integers(1, 7, nE)
    bp = np.concatenate([[0], np.cumsum(ar)]).astype(np.int32)
    bi = rng.integers(0, nV, bp[-1]).astype(np.int32)
    ref = transpose_incidence(bp, bi, nV, 1)
    for t in (2, 4, 8, 16):
        got = transpose_incidence(bp, bi, nV, t)
        for a, b in zip(got, ref, strict=True):
            assert np.array_equal(a, b), f"threads={t} reordered a bucket"


def test_the_transpose_caps_its_threads_by_the_scratch_it_would_need():
    """The per-thread histogram is nthr x nV against nnz of data, so the width is capped
    at nnz // nV. A complex with more vertices than entries stays serial rather than
    allocating a histogram larger than the array it is permuting, and the answer is the
    same either way."""
    from rexgraph.core._channel_tower import transpose_incidence
    bp = np.array([0, 2, 4], np.int32)
    bi = np.array([0, 1, 2, 3], np.int32)
    nV = 400                                    # nnz // nV == 0, so no parallel split
    for t in (1, 8, 32):
        for a, b in zip(transpose_incidence(bp, bi, nV, t),
                        transpose_incidence(bp, bi, nV, 1), strict=True):
            assert np.array_equal(a, b)


def test_the_default_width_is_physical_cores_not_logical():
    """The tower is memory-bound, so SMT siblings add contention and not parallelism.
    An explicit set_threads still wins, since that is where a measured per-host optimum
    goes."""
    from rexgraph import compute
    from rexgraph.hardware import cpu_count, physical_cores
    from rexgraph.sparse_character import _tower_width
    assert compute.get_threads() is None, "this test assumes no explicit width is set"
    assert _tower_width() == physical_cores()
    assert physical_cores() <= cpu_count()
    try:
        compute.set_threads(3)
        assert _tower_width() == 3, "an explicit width has to win"
    finally:
        compute.set_threads(None)


#### which tower the numbers actually need
def test_a_pairwise_complex_needs_no_denominator_at_all():
    """The share is 1/(k-1), which is 1 at a witness and at a pairwise relation, so
    such a complex has no denominator and its whole tower is exact in int64."""
    from rexgraph.sparse_character import channel_tower_precision
    for name in ("pairwise triangle", "pairwise with legs", "witness and pairs",
                 "witness only", "star"):
        r = _rex(CASES[name])
        p = channel_tower_precision(r._boundary_ptr, r._boundary_idx)
        assert p["tower"] == "integer" and p["scale"] == 1, (name, p)


def test_branching_moves_it_to_the_rational_tower():
    """k-1 enters, and C and F multiply two shares, so the denominator is lcm(k-1)^2."""
    from rexgraph.sparse_character import channel_tower_precision
    for name in ("5-ary with legs", "branching only", "one wide relation"):
        r = _rex(CASES[name])
        p = channel_tower_precision(r._boundary_ptr, r._boundary_idx)
        assert p["tower"] == "rational" and p["scale"] > 1, (name, p)
        assert p["dtype"] == "int64", "still exact, just over a denominator"


@pytest.mark.parametrize("name", list(CASES))
def test_the_integer_tower_equals_the_rational_one_exactly(name):
    """Numerator over the reported scale IS the Fraction, at every arity."""
    from fractions import Fraction as F
    from rexgraph.sparse_character import channel_diagonals_integer
    r = _rex(CASES[name])
    got, scale = channel_diagonals_integer(r)
    assert got is not None
    d, names = exact_channel_diagonals(r)
    for n in names:
        for i in range(int(r.nE)):
            assert F(int(got[n][i]), scale) == F(d[n][i]), (name, n, i)


def test_the_integer_tower_carries_no_float():
    import numpy as _np
    from rexgraph.sparse_character import channel_diagonals_integer
    got, scale = channel_diagonals_integer(_rex(CASES["mixed arity 1 to 5"]))
    assert isinstance(scale, int)
    for v in got.values():
        assert v.dtype == _np.int64


def test_an_unrepresentable_scale_says_so_rather_than_rounding():
    """Past int64 the approximation tower is the honest answer, and it is named."""
    from rexgraph.sparse_character import channel_tower_precision
    r = _rex(CASES["branching only"])
    p = channel_tower_precision(r._boundary_ptr, r._boundary_idx, int64_max=4)
    assert p["tower"] == "float" and p["scale"] is None
    assert "int64" in p["reason"]
