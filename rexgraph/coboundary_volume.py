"""Exact reduced Gram determinants for each coboundary grade.

At grade k the module reads B_(k+1), selects a maximal independent set of
rows and computes the determinant of the reduced Gram matrix over integers
and rational numbers. The result is a squared volume for that row selection.
An absent or rank zero boundary returns zero.

Grade 0 uses RexGraph._integer_B1. A stored relation column with shares
1/(k-1) becomes (-(k-1), +1, ..., +1) before the Gram is formed. For pairwise
relations this is the stored column. Higher grades use the integral stored
boundary coefficients and the pivot rows selected by exact rank reduction.

At grade 0 the implementation checks that the boundary nullity equals the
number of connected components. It raises ValueError when that condition
fails because different independent row selections can then give different
determinants. Higher grades use the stated pivot convention; their results
can depend on the selected rows.

For an ordinary pairwise graph, grade 0 returns the product of the spanning
tree counts of its connected components. Examples are 3 for a triangle,
16 for K4, 125 for K5 and 9 for two disconnected triangles. On complete
triangle complexes, the grade 1 regression values are 4, 125 and 46656 for
K4, K5 and K6 respectively.

This calculation uses boundary rows, while hodge_coords.harmonic_gram_det
uses a harmonic cycle frame. Their outputs agree for face free pairwise
graphs but can differ when faces or branching relations are present.
Faces affect higher grade boundaries and do not change the grade 0 input.

Pivot selection and determinant evaluation use sparse exact elimination.
Their cost depends on elimination fill and coefficient growth as well as
the input size. Use a rank query when the determinant is not required.
"""
from __future__ import annotations

from fractions import Fraction

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
    """Compute the reduced Gram determinant at the selected grade.

    The calculation selects independent rows from the integral boundary
    and evaluates their Gram determinant with exact elimination. It returns
    zero for an absent or rank zero boundary. Nonintegral boundary data
    raises ValueError instead of being rounded to integers.

    At grade 0, a boundary nullity different from the component count also
    raises ValueError. exact=True returns an integer or Fraction;
    exact=False converts the exact result to float.
    """
    from rexgraph.graded_boundary import _exact_rank_reduction, _integer_columns
    from rexgraph.rational_trig import _sparse_gram_determinant

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
    # Multiply primary Python integers only after selecting the independent rows.
    # A float or int64 Gram can lose coefficients before exact elimination begins.
    det = _sparse_gram_determinant(_integer_columns(B), rows)
    if not exact:
        return float(det)
    # an integral boundary gives an integral volume, so hand back the integer rather than
    # a Fraction whose denominator is 1
    if isinstance(det, Fraction) and det.denominator == 1:
        return int(det)
    return det


def volume_tower(rex, *, exact: bool = True) -> list:
    """Return reduced Gram determinants for all carried coboundary grades.

    Grades 0 and 1 are always included. An absent upper boundary gives
    zero at that grade. Higher grades follow the stored boundary tower,
    and exact controls the return type as in coboundary_volume.
    """
    from rexgraph.graded_boundary import graded_boundaries_from_rex

    n = len(graded_boundaries_from_rex(rex))
    # grades 0 and 1 are always reported: "no faces, so no room at grade 1" is a reading
    # and not a missing entry. Past that the length follows the complex, so a 3 rex
    # reports three rather than being capped at two.
    return [coboundary_volume(rex, g, exact=exact) for g in range(max(n, 2))]
