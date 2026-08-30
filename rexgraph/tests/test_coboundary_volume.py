"""How much room a field confined to the coboundary has, one grade at a time.

Confining a field to `im(B_{k+1}^T)` restricts the space it may occupy, and the
question a determinant answers about any spanning set is how much volume it spans and
whether it spans any. The Gram of the coboundary is the UP Laplacian, which
`graded_laplacians` folds into its combined form; `build_L0_sparse` / `build_L1_up_sparse`
are where the up half is exposed on its own, and this module reads it from there.
"""
import itertools
from math import comb

import numpy as np
import pytest

from rexgraph.coboundary_volume import coboundary_volume, volume_tower
from rexgraph.graph import RexGraph
from rexgraph.hodge_coords import harmonic_gram_det


def _g(src, tgt):
    r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
    r._ensure_clean()
    return r


def _complete2(n):
    """Every edge and every triangle on n vertices."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    ei = {e: k for k, e in enumerate(edges)}
    r = _g([a for a, b in edges], [b for a, b in edges])
    faces, signs = [], []
    for a, b, c in itertools.combinations(range(n), 3):
        faces.append([ei[(a, b)], ei[(a, c)], ei[(b, c)]])
        signs.append([1.0, -1.0, 1.0])
    r.add_faces(faces, signs)
    r._ensure_clean()
    return r


@pytest.mark.parametrize("name,src,tgt", [
    ("triangle", [0, 1, 2], [1, 2, 0]),
    ("K4", [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]),
    ("C6", [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]),
    ("path", [0, 1, 2, 3], [1, 2, 3, 4]),
    ("two triangles", [0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]),
])
def test_grade_zero_agrees_with_the_cycle_route(name, src, tgt):
    """The same integer from the coboundary side and the cycle side.

    Only WHERE BOTH REDUCE TO THE CYCLE SPACE, which is face-free and pairwise, as every
    fixture here is. They are different objects and diverge as soon as either condition
    fails: `test_the_cycle_route_is_a_different_object` pins that."""
    r = _g(src, tgt)
    assert coboundary_volume(r, 0) == harmonic_gram_det(r), name


@pytest.mark.parametrize("n", [4, 5, 6])
def test_the_tower_is_the_graded_matrix_tree_theorem(n):
    """Grade 0 is Cayley's n^(n-2) and grade 1 is Kalai's n^C(n-2,2), so the tower is
    that theorem read one grade at a time rather than an analogy carried upward."""
    r = _complete2(n)
    g0, g1 = volume_tower(r)
    assert g0 == n ** (n - 2)
    assert g1 == n ** comb(n - 2, 2)


def test_grade_one_is_zero_without_faces():
    """No faces means no grade-1 coboundary, so a field confined there has nowhere to
    be. Zero is the answer, not an error."""
    assert volume_tower(_g([0, 1, 2], [1, 2, 0]))[1] == 0


def test_a_face_makes_the_grade_one_volume_nonzero():
    r = _g([0, 1, 2], [1, 2, 0])
    r.add_faces([[0, 1, 2]], [[1.0, 1.0, 1.0]])
    r._ensure_clean()
    assert volume_tower(r)[1] != 0


def test_the_up_laplacian_it_reads_is_the_librarys_own():
    """The operator is `L0 = B1 B1^T` from the library, not a local reconstruction, and
    a zero-sum column is what makes its row sums vanish."""
    r = _g([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
    L = np.asarray(r.L0_sparse.todense())
    assert L.shape == (r.nV, r.nV)
    assert np.allclose(L, L.T)
    assert np.allclose(L.sum(1), 0.0), "a zero-sum column makes the row sums vanish"


def test_the_volume_is_exact_and_integral_on_an_integral_boundary():
    r = _g([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
    v = coboundary_volume(r, 0)
    assert isinstance(v, int) and v == 16


def test_the_cycle_route_is_a_different_object():
    """`harmonic_gram_det` reads the harmonic frame, which faces shrink; this reads the
    1-skeleton, which they cannot touch. Equal only where both are the cycle space."""
    base = _g([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
    assert coboundary_volume(base, 0) == harmonic_gram_det(base) == 16
    faced = _g([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
    faced.add_faces([[0, 3, 1]], [[1.0, 1.0, -1.0]])
    faced._ensure_clean()
    assert coboundary_volume(faced, 0) == 16, "faces cannot move a 1-skeleton quantity"
    assert harmonic_gram_det(faced) != 16, "and they do move the harmonic frame"


def test_branching_reads_the_integer_representative():
    """The stored column carries the share 1/(k-1); the Gram must see the integer
    representative instead. A k=3 relation carrying a triangle reads 9, where the shares
    would give 9/2, and neither the pairwise anchors nor the Kalai tower move."""
    ptr = np.array([0, 2, 4, 6, 9], np.int32)
    idx = np.array([0, 1, 1, 2, 0, 2, 0, 1, 2], np.int32)
    r = RexGraph(boundary_ptr=ptr, boundary_idx=idx)
    r._ensure_clean()
    v = coboundary_volume(r, 0)
    assert isinstance(v, int) and v == 9


def test_a_representative_dependent_volume_is_refused():
    """A lone k=4 relation has a 3-dimensional kernel over one component, so different
    maximal independent row sets give 1 and 9. Returning either silently is the
    silent-wrong-answer shape, so it is refused."""
    ptr = np.array([0, 4], np.int32)
    idx = np.array([0, 1, 2, 3], np.int32)
    r = RexGraph(boundary_ptr=ptr, boundary_idx=idx)
    r._ensure_clean()
    with pytest.raises(ValueError, match="representative-independent"):
        coboundary_volume(r, 0)


def test_a_grade_the_complex_does_not_carry_reads_zero():
    """Past the top grade there is no coboundary, so a field confined there has nowhere
    to be. Zero is the answer; the old 2-grade cap raised instead and could not reach a
    3-rex at all."""
    assert coboundary_volume(_g([0, 1], [1, 0]), 2) == 0
