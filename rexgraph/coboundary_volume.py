"""rexgraph.coboundary_volume: how much room a field confined to the coboundary has.

A field confined to the coboundary at grade k lives in `im(delta_k) = im(B_{k+1}^T)`, and
that is a restriction of the space the field may occupy rather than a storage decision.
The question a determinant answers about any set of vectors is how much volume they span
and whether they span any at all, so the question to ask about the confined space is the
determinant of its Gram, one grade at a time.

The Gram of the coboundary is the UP Laplacian, `B_{k+1} B_{k+1}^T`. It is singular by
construction, since whatever the coboundary does not reach is its kernel, so the volume is
taken off that kernel and what is left is exact.

GRADE 0 IS AN ANCHOR, NOT AN ANALOGY. There the Gram is `B1 B1^T = L0`, its kernel is one
indicator per component, and the reduced determinant is the number of spanning trees, the
product over components: triangle 3, K4 16, K5 125, path 1, C6 6, two triangles 9. K4 and
K5 are Cayley's `n^(n-2)`.

AND GRADE 1 IS ALSO ANCHORED. On the complete 2-complex the tower reproduces the graded
matrix-tree theorem at both grades:

    n     grade 0            grade 1
    4     16 = 4^2           4      = 4^C(2,2)
    5    125 = 5^3         125      = 5^C(3,2)
    6   1296 = 6^4       46656      = 6^C(4,2)

Grade 0 is Cayley and grade 1 is Kalai, so the tower is that theorem read one grade at a
time rather than an analogy carried up from grade 0 by hope.

THE INTEGER REPRESENTATIVE IS THE ONE TO READ. The stored `B1` column carries the share
`1/(k-1)`; `RexGraph._integer_B1` scales it to `(-(k-1), +1, ..., +1)`, which is the same
column in the normalisation the integer tower works in and is identical at k=2. Taking the
Gram of the shares instead gives a different number on every branching complex: a k=3
relation carrying a triangle reads 9/2 against 9. Pairwise complexes are unaffected,
which is why every anchor above still holds.

WHAT IT DOES NOT SAY. `hodge_coords.harmonic_gram_det` reads the same kind of quantity from
the CYCLE side and is NOT this one. The two agree exactly where the harmonic frame is the
whole cycle space, meaning face-free and pairwise, and diverge as soon as either fails:
K4 with one face reads 16 here against 432 there, with two faces 16 against 8, and the
branching case above 9 against 18. Faces cannot move the grade-0 volume, since it is fixed
by the 1-skeleton, and they do move the harmonic frame. Neither is a check on the other
outside the overlap.

WHICH INDEPENDENT ROWS, AND WHEN IT MATTERS. The volume is a determinant on a maximal
independent set of rows, and in general its value DEPENDS on which set. Enumerated
exhaustively: a lone k=4 relation admits both 1 and 9, and a k=3 relation with a pendant
admits both 1 and 4. It is representative-independent exactly when the kernel is spanned
by indicators partitioning the cells, since then every admissible complement picks one
cell per part and the Gram of the kernel basis on it is the same. At grade 0 that
condition is `nullity == components`, which is checked and refused when it fails: every
pairwise complex satisfies it, and so does a branching k=3 relation carrying a triangle
(9). Higher grades are read on the stated convention, `_exact_rank_reduction`'s pivot
rows, which reproduces the Kalai anchors above; no cheap test decides invariance there,
and a two-set check was measured to MISS the pendant case, so none is offered.

WHAT IT MEANS FOR A CONFINED FIELD. Zero says the coboundary is degenerate at that grade
and confining a field there destroys it. Large says the confined space is roomy. The number
is a squared volume, so it compares across complexes of the same shape and not across
different ones, which is the ordinary caution about determinants rather than a defect of
this one.

SCOPE. The reduction to the pivot rows is sparse and exact, but a determinant is an
elimination and the final Bareiss pass is dense in the RANK. That is `nV - components` at
grade 0, so this is a reading for a complex you can eliminate on, not a corpus-scale one.
`graded_boundary._exact_rank_reduction` is the sparse route when only the rank is wanted.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

__all__ = ["coboundary_volume", "volume_tower"]


def _integer_coboundary(rex, grade: int):
    """`B_{grade+1}` in its exact integer representative, sparse, or None past the top.

    Grade 0 goes through `RexGraph._integer_B1` so the share never reaches the Gram.
    Higher grades read the stored coefficients, which `faces.solve_face_column` has
    already cleared of denominators.
    """
    from rexgraph.graded_boundary import graded_boundaries_from_rex

    if int(grade) == 0:
        return rex._integer_B1()
    bounds = graded_boundaries_from_rex(rex)
    g = int(grade)
    return bounds[g] if g < len(bounds) else None


def coboundary_volume(rex, grade: int = 0, *, exact: bool = True):
    """The squared volume of the coboundary image at `grade`, off its kernel.

    Returns 0 when the coboundary reaches nothing at that grade, which is the honest
    answer rather than an error: a complex with no faces has no grade-1 coboundary and a
    field confined there has nowhere to be.

    Raises when the boundary is not integral in its own normalisation. Recovering a
    rational from a float would be answering a question that should not have been asked,
    so it is refused here the same way `hodge_coords._exact_ints` refuses it.
    """
    from rexgraph.graded_boundary import _exact_rank_reduction
    from rexgraph.rational_trig import bareiss_determinant

    rex._ensure_clean()
    B = _integer_coboundary(rex, grade)
    if B is None or B.shape[1] == 0 or B.nnz == 0:
        return 0
    rank, rows = _exact_rank_reduction(B, with_pivots=True)
    if rank is None:
        raise ValueError(
            f"the grade-{int(grade)} boundary is not integral in its own normalisation, "
            "so its volume would be a rounded guess. Supply integer face coefficients "
            "(faces.solve_face_column clears denominators) or ask for the rank instead")
    if not rows:
        return 0
    if int(grade) == 0:
        from rexgraph.graded_boundary import _beta0_components
        comps = int(_beta0_components(B))
        nullity = int(B.shape[0]) - int(rank)
        if nullity != comps:
            raise ValueError(
                f"the grade-0 volume is not representative-independent here: the kernel "
                f"is {nullity}-dimensional over {comps} component(s), so different "
                f"maximal independent row sets give different determinants (a lone k=4 "
                f"relation admits both 1 and 9). Only a kernel spanned by component "
                f"indicators makes the cofactor a single number, which is what the "
                f"matrix-tree reading needs")
    # the pivot rows are a maximal independent set, so this block is invertible and the
    # Gram is only ever formed on them: no nV x nV anything
    L = (B @ B.T).tocsr()[rows][:, rows]
    A = np.asarray(L.todense())
    if not exact:
        return float(np.linalg.det(A))
    det = bareiss_determinant([[Fraction(int(round(float(x)))) for x in row] for row in A])
    # an integral boundary gives an integral volume, so hand back the integer rather than
    # a Fraction whose denominator is 1
    if isinstance(det, Fraction) and det.denominator == 1:
        return int(det)
    return det


def volume_tower(rex, *, exact: bool = True) -> list:
    """The coboundary volume at every grade the complex carries.

    Grade 0 is the spanning-tree count and grade 1 is its analogue over the faces, 0
    without them. The length follows the complex rather than a fixed cap, so a 3-rex
    reports three entries.
    """
    from rexgraph.graded_boundary import graded_boundaries_from_rex

    n = len(graded_boundaries_from_rex(rex))
    # grades 0 and 1 are always reported: "no faces, so no room at grade 1" is a reading
    # and not a missing entry. Past that the length follows the complex, so a 3-rex
    # reports three rather than being capped at two.
    return [coboundary_volume(rex, g, exact=exact) for g in range(max(n, 2))]
