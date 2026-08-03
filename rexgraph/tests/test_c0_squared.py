"""c0^2, the exchange-rate invariant, as an exact rational.

Two coupling constants are in circulation and they are the two sides of the
energy/entropy duality, not rival definitions of one quantity:

    c2_E = tr(L1^2) / tr(T^2)                          the ENERGY side  (RexGraph alpha_G)
    c2_H = [tr(T^2)/tr(T)^2] / [tr(L1^2)/tr(L1)^2]     the ENTROPY side, = e^{H_S - H_T}

They coincide on K_k by regularity, so a complete-graph fixture cannot separate them, and
they differ on any non-regular complex. What is invariant is the geometric mean, and the
square root is exact because the product telescopes:

    c2_E * c2_H = [trL2/trT2] * [trT2/trT^2] * [trL^2/trL2] = (trL / trT)^2

so

    c0^2 = tr(L1) / tr(T) = ||B2||_F^2 / ||B1||_F^2

with no square root, no matmul and no eigensolve: two sums of squared entries, O(nnz).
On K_k, tr(T) = k(k-1) and tr(L1) = 3*C(k,3) = k(k-1)(k-2)/2, giving (k-2)/2 directly.

The value is returned as a `Fraction`. That is the point: it is on the integer/exact tower,
and a float would discard the exactness for nothing. The weighted (Riemann) side is where
floats become unavoidable, and this quantity is not on it.
"""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _complete(k):
    """K_k with every triangle attached, built explicitly so the fixture does not depend
    on the face-detection path under test elsewhere."""
    E = list(combinations(range(k), 2))
    ei = {e: i for i, e in enumerate(E)}
    src = np.array([i for i, j in E], np.int32)
    tgt = np.array([j for i, j in E], np.int32)
    rex = RexGraph(sources=src, targets=tgt)
    faces, signs = [], []
    for (i, j, l) in combinations(range(k), 3):
        faces.append(np.array([ei[(i, j)], ei[(j, l)], ei[(i, l)]], np.int32))
        signs.append(np.array([1.0, 1.0, -1.0]))
    rex.add_faces(faces, signs)
    return rex


def _p4_tri():
    """A path a-b-c-d with a chord a-c closed by one triangle. Non-regular, so it
    separates c2_E from c2_H."""
    rex = RexGraph(sources=np.array([0, 1, 2, 0], np.int32),
                   targets=np.array([1, 2, 3, 2], np.int32))
    rex.add_faces([np.array([0, 1, 3], np.int32)], [np.array([1.0, 1.0, -1.0])])
    return rex


def _bowtie():
    """Two triangles sharing an edge."""
    ed = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)]
    ei = {e: i for i, e in enumerate(ed)}
    rex = RexGraph(sources=np.array([e[0] for e in ed], np.int32),
                   targets=np.array([e[1] for e in ed], np.int32))
    rex.add_faces(
        [np.array([ei[(0, 1)], ei[(1, 2)], ei[(0, 2)]], np.int32),
         np.array([ei[(1, 3)], ei[(2, 3)], ei[(1, 2)]], np.int32)],
        [np.array([1.0, 1.0, -1.0]), np.array([1.0, -1.0, -1.0])])
    return rex


@pytest.mark.parametrize("k", [4, 5, 6, 7])
def test_c0_squared_is_k_minus_two_over_two_on_complete_graphs(k):
    """The reference value, exact."""
    assert _complete(k).c0_squared == Fraction(k - 2, 2)


def test_c0_squared_is_exact_rational_not_float():
    assert isinstance(_complete(5).c0_squared, Fraction)


@pytest.mark.parametrize("fixture,want", [(_p4_tri, Fraction(3, 8)),
                                          (_bowtie, Fraction(3, 5))])
def test_c0_squared_on_non_regular_complexes(fixture, want):
    """Where the two sides genuinely differ, the invariant is still a clean rational."""
    assert fixture().c0_squared == want


@pytest.mark.parametrize("fixture", [_p4_tri, _bowtie, lambda: _complete(5)])
def test_the_geometric_mean_identity(fixture):
    """c0^2 squared equals c2_E * c2_H. This is the claim that the two constants are the
    two sides of one duality, checked rather than asserted."""
    rex = fixture()
    assert rex.c0_squared ** 2 == rex.c2_E * rex.c2_H


def test_c2_E_and_c2_H_coincide_on_K_k_and_differ_otherwise():
    """Why a complete-graph fixture cannot validate either one alone."""
    reg = _complete(5)
    assert reg.c2_E == reg.c2_H == Fraction(3, 2)
    non = _p4_tri()
    assert non.c2_E != non.c2_H
    assert (non.c2_E, non.c2_H) == (Fraction(9, 26), Fraction(13, 32))


def test_c2_E_agrees_with_the_float_alpha_G():
    """alpha_G is the same quantity on the approximation tower; the exact one must not
    drift from it."""
    for fx in (_complete(5), _p4_tri(), _bowtie()):
        assert abs(float(fx.c2_E) - float(fx.alpha_G)) < 1e-12


def test_no_faces_means_no_exchange_rate():
    """With nF = 0 the curl tier is empty, so tr(L1) = 0 and the rate is 0, not NaN."""
    rex = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32))
    assert rex.c0_squared == 0
    assert rex.c2_E == 0


def test_branching_arity_is_carried_exactly():
    """tr(T) = sum_e ||c_e||^2 and a branching column contributes 1 + 1/(k-1), not 2.
    A k=3 relation therefore contributes 3/2 where a pairwise edge contributes 2."""
    rex = RexGraph.from_hypergraph(np.array([0, 3], np.int32), np.array([0, 1, 2], np.int32))
    assert rex.trace_T == Fraction(3, 2)
    pair = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32))
    assert pair.trace_T == Fraction(2)
