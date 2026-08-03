"""The channels must see the metric. G is geometric BECAUSE B1 is weighted.

Two axes carry structure beyond raw topology and they are independent:

    VERTEX-ORDER OVERLAP   the share 1/(k-1): a wide relation contributes less overlap
                           mass per leg than a narrow one. Carried by the boundary
                           column, so it is present on an unweighted complex.
    THE METRIC             the edge weights. An unweighted G is a COUNT of shared
                           vertices, and a count cannot carry a geometry.

The second was missing entirely: `w_E` was accepted by the constructor, carried through
IO and incremental mutation, asserted in six tests, and consumed by no operator. Every
channel read identically at every weight.

The canonical form weights per relation, not per square root:

    T[i,j] = sum_v s_i(v) w_i s_j(v) w_j          signed
    G[i,j] = sum_v |s_i(v)| w_i |s_j(v)| w_j      unsigned twin

so a rational weight keeps the channels rational. It is sqrt(w), which appears in the
normalized G and nowhere here, that forces a float.

The expected values below are spore's, from test/channel_probe.c, computed independently
of this library. An external oracle is the point: agreeing with our own reimplementation
of the formula would prove nothing.
"""

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _triangle(w_e1):
    return RexGraph(sources=np.array([0, 1, 2], np.int32),
                    targets=np.array([1, 2, 0], np.int32),
                    w_E=np.array([w_e1, 1.0, 1.0]))


# (weight on e1, chi_T, chi_G, chi_F, chi_C) from spore channel_probe.c
SPORE = [
    (1.0,   0.250000, 0.250000, 0.250000, 0.250000),
    (5.0,   0.350765, 0.350765, 0.172194, 0.126276),
    (100.0, 0.353231, 0.353231, 0.175772, 0.117767),
]


@pytest.mark.parametrize("w,T,G,F,C", SPORE)
def test_chi_matches_the_external_oracle(w, T, G, F, C):
    chi = np.asarray(_triangle(w).structural_character)[0]
    assert np.allclose(chi, [T, G, F, C], atol=1e-6), (w, chi)


def test_the_channels_move_with_the_metric():
    """The property the old path lacked: it returned 0.250000 at every weight."""
    a = np.asarray(_triangle(1.0).structural_character)[0]
    b = np.asarray(_triangle(5.0).structural_character)[0]
    assert abs(a[0] - b[0]) > 1e-6


def test_G_diagonal_still_equals_T_diagonal():
    """|s|^2 = s^2, so weighting cannot separate them on the diagonal. That is not a
    defect to normalise away: it is why F exists, since all the sign content is
    off-diagonal."""
    for w in (1.0, 5.0, 100.0):
        chi = np.asarray(_triangle(w).structural_character)[0]
        assert abs(chi[0] - chi[1]) < 1e-12


@pytest.mark.parametrize("w", [1.0, 2.0, 5.0, 100.0])
def test_the_channels_still_partition_unity_under_weighting(w):
    chi = np.asarray(_triangle(w).structural_character)
    assert np.allclose(chi.sum(axis=1), 1.0, atol=1e-9)


def test_the_overlap_gramian_carries_the_weight():
    """K = |B1|^T W |B1|. Unweighted it is a shared-vertex count, which is
    combinatorial; the metric is what makes it geometric."""
    K1 = RexGraph(sources=np.array([0, 1, 2], np.int32),
                  targets=np.array([1, 2, 0], np.int32)).overlap_gramian_sparse.toarray()
    K5 = _triangle(5.0).overlap_gramian_sparse.toarray()
    assert K1[0, 0] == 2.0
    assert np.isclose(K5[0, 0], 2.0 * 25.0)            # both legs scale by w^2
    assert np.isclose(K5[0, 1], 1.0 * 5.0)             # one leg weighted, one not


def test_the_two_axes_are_independent():
    """Arity and the metric are separate structure. A branching relation at unit weight
    already differs from a pairwise one, and weighting moves it again."""
    def K00(ptr, idx, w):
        rex = RexGraph.from_hypergraph(np.asarray(ptr, np.int32), np.asarray(idx, np.int32))
        if w is not None:
            rex._w_E = np.asarray(w, float)     # from_hypergraph takes no w_E; set it after
        return rex.overlap_gramian_sparse.toarray()[0, 0]

    pair = K00([0, 2], [0, 1], None)
    k4 = K00([0, 4], [0, 1, 2, 3], None)
    k4_w = K00([0, 4], [0, 1, 2, 3], [3.0])
    assert pair == 2.0
    assert np.isclose(k4, 1.0 + 3 * (1 / 3) ** 2)      # arity alone changes it
    assert np.isclose(k4_w, k4 * 9.0)                  # and the metric changes it again


def test_unweighted_is_unchanged():
    """Every complex without an explicit w_E must read exactly as before."""
    rex = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    chi = np.asarray(rex.structural_character)[0]
    assert np.allclose(chi, [0.25, 0.25, 0.25, 0.25], atol=1e-12)


def test_a_rational_weight_stays_exact():
    """Weighting is per relation, not per square root, so a rational weight keeps the
    channels rational. sqrt(w) appears in the normalized G and nowhere here."""
    from fractions import Fraction

    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32),
                   w_E=np.array([0.5, 1.0, 1.0]))
    K = rex.overlap_gramian_sparse.toarray()
    assert Fraction(K[0, 0]).limit_denominator(1000) == Fraction(1, 2)
