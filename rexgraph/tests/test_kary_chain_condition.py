"""The k-ary fan and chain-condition fulfilment, checked against spore.

Every expected value here was produced by running `spore-probes/04_kary_fan_and_chain.spore`
against the real engine. rexgraph and spore agree on every row at every arity, which is
what makes them usable together rather than merely similar.
"""
import numpy as np
import pytest

from rexgraph.core._sparse import to_scipy_csr
from rexgraph.faces import auto_hyperface
from rexgraph.graph import RexGraph


def fan(k, *, legs=True):
    """H over b0..b(k-1) as ONE branching column, plus the legs b0-bi."""
    ptr, idx = [0], list(range(k))
    ptr.append(len(idx))
    if legs:
        for i in range(1, k):
            idx += [0, i]
            ptr.append(len(idx))
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int64),
                                    np.asarray(idx, np.int64))


def chain_residual(rex):
    """|B1 B2|, the quantity spore's `verify hodge` prints."""
    b2 = getattr(rex, "_B2_hodge_dual", None)
    if b2 is None or int(rex.nF_hodge) == 0:
        return None
    B1 = to_scipy_csr(rex._B1_dual).tocsr()
    B2 = to_scipy_csr(b2).tocsr()
    return float(np.abs((B1 @ B2).toarray()).max())


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_the_fan_opens_a_hole_and_the_hyperface_closes_it(k):
    """spore, every k: H alone cycles=0 dim_H=0; + legs cycles=1 dim_H=1;
    + hyperface nF=1 cycles=1 dim_H=0 curl=1. The cycle SURVIVES: it stops
    being a hole and starts bounding, which is curl_dim = cycles - dim_H."""
    lone = fan(k, legs=False)
    lone._ensure_clean()
    assert len(lone.cycle_basis) == 0 and int(lone.betti[1]) == 0

    rex = fan(k)
    rex._ensure_clean()
    assert len(rex.cycle_basis) == 1 and int(rex.betti[1]) == 1

    assert auto_hyperface(rex) == 1
    rex._ensure_clean()
    cycles, dim_h = len(rex.cycle_basis), int(rex.betti[1])
    assert (cycles, dim_h, cycles - dim_h) == (1, 0, 1)


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_the_chain_condition_is_fulfilled_exactly(k):
    """|B1 B2| = 0.00e+00 in spore. Exactly zero, not merely small."""
    rex = fan(k)
    auto_hyperface(rex)
    rex._ensure_clean()
    assert chain_residual(rex) == 0.0


@pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
def test_the_solved_face_coefficient_is_minus_one_over_k_minus_one(k):
    """spore solves dH + sum c_i d(e_i) = 0 and recovers c_i = -1/(k-1) for the fan.
    rexgraph solves the same column and gets the same number."""
    rex = fan(k)
    auto_hyperface(rex)
    rex._ensure_clean()
    col = to_scipy_csr(rex._B2_hodge_dual).tocsc().toarray()[:, 0]
    normalised = col / col[0]
    assert normalised[0] == pytest.approx(1.0)
    for c in normalised[1:]:
        assert c == pytest.approx(-1.0 / (k - 1))


def test_a_lone_hyperedge_closes_nothing():
    """There is no enclosed area to bound, so refusing is correct rather than a gap."""
    assert auto_hyperface(fan(4, legs=False)) == 0
    assert auto_hyperface(fan(6, legs=False)) == 0


@pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
def test_the_zero_sum_share_is_the_same_number_that_makes_the_fan_bound(k):
    """The identity underneath all of the above, and the reason arity never enters the
    boundary condition.

        B1(c) for c = (k-1)H - sum legs  cancels iff (k-1)*share == 1

    So `share = 1/(k-1)` has two roles at once: it is what makes the column sum to zero
    AND what makes the fan bound. The old unsigned form (share 1) fails by exactly k-2,
    which is also its column sum.
    """
    def B1(share):
        B = np.zeros((k, k))
        B[0, 0] = -1.0
        for i in range(1, k):
            B[i, 0] = share
            B[0, i] = -1.0
            B[i, i] = 1.0
        return B

    c = np.concatenate([[k - 1], -np.ones(k - 1)])

    right = 1.0 / (k - 1)
    assert abs(B1(right)[:, 0].sum()) < 1e-12            # zero-sum column
    assert np.abs(B1(right) @ c).max() < 1e-12           # and it bounds

    unsigned = B1(1.0)
    assert unsigned[:, 0].sum() == pytest.approx(k - 2)  # the known column-sum defect
    assert np.abs(unsigned @ c).max() == pytest.approx(k - 2)   # and it does NOT bound
