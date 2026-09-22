"""Effective mode counts and the harmonic log of the graded Hodge operators, exactly.

For a nonnegative operator X that is self adjoint in a positive metric, the effective
mode count is

    n(X) = tr(X)^2 / tr(X^2),

the number of equal eigenvalues that would reproduce the first two traces of its
powers, and the harmonic log is H2(X) = log n(X), the collision entropy of the
normalized spectrum. Both traces are sums of products of boundary coefficients and
metric weights, so n is an exact rational and only the logarithm leaves Q. No operator
is assembled and no eigenvalue is computed.

At grade k the Hodge operator L_k = L_k^down + L_k^up has the sectors
L_k^down = B_k^dagger B_k and L_k^up = B_(k+1) B_(k+1)^dagger, where the metric adjoint
is B^dagger = M^-1 B^T M. The chain law B_k B_(k+1) = 0 makes tr(L^down L^up) vanish,
so both traces of L_k are the sums of the sector traces.

L_k alone is blind to its kernel: a harmonic field has eigenvalue zero and adds nothing
to either trace. The harmonic completion L_k + w Pi^h places every harmonic direction
at eigenvalue w. Because L_k Pi^h = 0 and Pi^h is a metric orthogonal projector of
trace beta_k,

    tr(L_k + w Pi^h) = tr(L_k) + w beta_k,    tr((L_k + w Pi^h)^2) = tr(L_k^2) + w^2 beta_k,

so a completed count needs only the sector traces and the exact Betti number, and no
harmonic frame is built. Three weights are carried:

    unit    w = 1                   the complete field coordinates of the paper
    mean    w = tr L_k / rank L_k   the mean nonzero eigenvalue
    energy  w = tr L_k^2 / tr L_k   the energy weighted mean eigenvalue

The unit weight reads the harmonic directions against the scale of L_k, which the
composite binary shares and the ratio of adjacent grade metrics set; a common rescale of
every grade metric leaves L_k unchanged. The mean and energy weights are invariant under
scaling L_k, and each is the unique weight whose completion keeps its own trace ratio:
under the mean weight tr X / rank X stays tr L_k / rank L_k, and under the energy weight
tr X^2 / tr X stays tr L_k^2 / tr L_k. Every weight gives n <= n(L_k) + beta_k, with
equality exactly at the energy weight, where each harmonic direction adds one mode.

This is a graded reading of the Hodge tower. `RexGraph.harmonic_entropy` is a different
object, the float collision entropy of the four channel relational Laplacian, which is
positive definite and so has no kernel to complete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

__all__ = ["SECTORS", "WEIGHTS", "GradeTraces", "grade_traces", "effective_modes", "harmonic_log"]

SECTORS = ("completed", "hodge", "down", "up")
WEIGHTS = ("unit", "mean", "energy")


def _check(sector: str, weight: str) -> None:
    if sector not in SECTORS:
        raise ValueError(f"sector must be one of {', '.join(SECTORS)}; got {sector!r}")
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {', '.join(WEIGHTS)}; got {weight!r}")
    if weight != "unit" and sector != "completed":
        raise ValueError(f"a harmonic weight applies only to the completed sector; "
                         f"the {sector} sector has no harmonic part to weigh")


@dataclass(frozen=True)
class GradeTraces:
    """The exact first and second traces of one grade's Hodge sectors.

    `down` and `up` are each (tr X, tr X^2) for L_k^down and L_k^up in the grade
    metrics; `betti` is beta_k = dim ker L_k and `size` the number of grade k cells.
    """

    grade: int
    size: int
    betti: int
    down: tuple[Fraction, Fraction]
    up: tuple[Fraction, Fraction]

    @property
    def rank(self) -> int:
        """rank L_k = size - beta_k, the number of modes that carry energy."""
        return self.size - self.betti

    def harmonic_weight(self, weight: str = "unit") -> Fraction:
        """The eigenvalue the completion gives each harmonic direction.

        When L_k vanishes there is no scale to read; every weight then gives the same
        count beta_k, and the weight is reported as 1.
        """
        _check("completed", weight)
        first, second = self.down[0] + self.up[0], self.down[1] + self.up[1]
        if weight == "unit" or first == 0:
            return Fraction(1)
        return first / self.rank if weight == "mean" else second / first

    def traces(self, sector: str = "completed", weight: str = "unit") -> tuple[Fraction, Fraction]:
        """(tr X, tr X^2) for the sector X named by `sector`."""
        _check(sector, weight)
        if sector == "down":
            return self.down
        if sector == "up":
            return self.up
        first, second = self.down[0] + self.up[0], self.down[1] + self.up[1]
        if sector == "hodge":
            return first, second
        w = self.harmonic_weight(weight)
        return first + w * self.betti, second + w * w * self.betti

    def effective_modes(self, sector: str = "completed", weight: str = "unit") -> Fraction:
        """n(X) = tr(X)^2 / tr(X^2); zero when the sector is the zero operator."""
        first, second = self.traces(sector, weight)
        return Fraction(0) if second == 0 else first * first / second

    def harmonic_log(self, sector: str = "completed", weight: str = "unit") -> float:
        """H2 = log n, taken as log(numerator) minus log(denominator) so no rational is rounded first."""
        count = self.effective_modes(sector, weight)
        if count == 0:
            raise ValueError(f"the {sector} sector at grade {self.grade} carries no modes; "
                             "its harmonic log is undefined")
        return math.log(count.numerator) - math.log(count.denominator)


def _metric_form(metric, grade: int):
    """None for the identity, the diagonal when the form is diagonal, else the form itself."""
    if any(i != j for i, j in metric.entries):
        return metric
    diagonal = [metric.entries.get((i, i), Fraction(0)) for i in range(metric.nrows)]
    if any(value <= 0 for value in diagonal):
        raise ValueError(f"the grade {grade} metric must be positive")
    return None if all(value == 1 for value in diagonal) else diagonal


def _as_form(form, size: int):
    from rexgraph.exact_green import ExactSparse
    if form is None:
        return ExactSparse.identity(size)
    return form if isinstance(form, ExactSparse) else ExactSparse.diagonal(form)


def _sector_traces(columns, rows: int, row_form, column_form):
    """(tr G, tr G^2) for G = N^-1 B^T M B, with M on the rows of B and N on its columns.

    Diagonal forms read the sparse Gram accumulation. A form that couples cells is solved,
    never inverted: one exact elimination of [N | B^T M B] gives N^-1 B^T M B.
    """
    from rexgraph.exact_green import ExactSparse
    from rexgraph.graded_boundary import _exact_gram_traces
    if not isinstance(row_form, ExactSparse) and not isinstance(column_form, ExactSparse):
        return _exact_gram_traces(columns, row_form, column_form)
    size = len(columns)
    if size == 0:
        return Fraction(0), Fraction(0)
    boundary = ExactSparse.from_columns(rows, columns)
    row_form, column_form = _as_form(row_form, rows), _as_form(column_form, size)
    entries = dict(column_form.entries)
    for j in range(size):
        column = [Fraction(0)] * size
        column[j] = Fraction(1)
        for i, value in enumerate(boundary.T.apply(row_form.apply(boundary.apply(column)))):
            if value:
                entries[i, size + j] = value
    reduced, pivots = ExactSparse(size, 2 * size, entries).rref()
    if pivots[:size] != tuple(range(size)):
        raise ValueError("the column metric must be positive definite")
    solved = {(i, j - size): v for (i, j), v in reduced.entries.items() if j >= size}
    return (sum((solved.get((i, i), Fraction(0)) for i in range(size)), Fraction(0)),
            sum((v * solved.get((j, i), Fraction(0)) for (i, j), v in solved.items()), Fraction(0)))


def _rex_tower(rex):
    from rexgraph.exact_green import exact_grade_metric
    from rexgraph.native_rank import exact_tower
    betti = rex.betti_tower
    shapes, columns = exact_tower(rex)
    sizes = [shapes[0][0]] + [shape[1] for shape in shapes]
    forms = [_metric_form(exact_grade_metric(rex, g, n), g) for g, n in enumerate(sizes)]
    return sizes, list(columns), forms, [int(b) for b in betti]


def _calculus_tower(calculus):
    """Columns, metric forms and Betti numbers of a NativeFieldCalculus, whose chain law is verified."""
    from rexgraph.graded_boundary import _rank_integer_columns
    from rexgraph.native_rank import clear_column_denominators
    complex_ = calculus.complex
    sizes = list(complex_.sizes)
    columns = []
    for k, triples in enumerate(complex_.boundaries):
        cols = [{} for _ in range(sizes[k + 1])]
        for i, j, value in triples:
            cols[j][i] = Fraction(value)
        columns.append(cols)
    ranks = [0] + [_rank_integer_columns((sizes[k], sizes[k + 1]), clear_column_denominators(cols))[0]
                   for k, cols in enumerate(columns)] + [0]
    forms = [_metric_form(m.as_sparse(), g) for g, m in enumerate(calculus.metrics)]
    return sizes, columns, forms, [n - ranks[g] - ranks[g + 1] for g, n in enumerate(sizes)]


def grade_traces(source, grade: int = 1) -> GradeTraces:
    """The exact sector traces of L_grade under the declared grade metrics.

    `source` is a RexGraph, whose declared edge metric is read at grade 1 and the identity
    elsewhere, or a NativeFieldCalculus, whose declared metric at every grade is read,
    including a metric that couples cells. Both read the original rational tower, so a
    branching relation contributes its declared shares, and both certify the chain law
    before any trace is returned, since the additivity of the sectors depends on it.
    """
    from numbers import Integral

    from rexgraph.native_field import NativeFieldCalculus
    if isinstance(grade, bool) or not isinstance(grade, Integral):
        raise TypeError("grade must be an integer")
    grade = int(grade)
    if isinstance(source, NativeFieldCalculus):
        source.check_state()
        sizes, columns, forms, betti = _calculus_tower(source)
    else:
        sizes, columns, forms, betti = _rex_tower(source)
    if not 0 <= grade < len(sizes):
        raise ValueError(f"grade {grade} is not present; the complex carries grades 0 to {len(sizes) - 1}")
    zero = (Fraction(0), Fraction(0))
    down = (_sector_traces(columns[grade - 1], sizes[grade - 1], forms[grade - 1], forms[grade])
            if grade > 0 else zero)
    up = (_sector_traces(columns[grade], sizes[grade], forms[grade], forms[grade + 1])
          if grade < len(columns) else zero)
    return GradeTraces(grade, int(sizes[grade]), betti[grade],
                       tuple(Fraction(v) for v in down), tuple(Fraction(v) for v in up))


def effective_modes(rex, grade: int = 1, sector: str = "completed", weight: str = "unit") -> Fraction:
    """The exact effective mode count n = tr(X)^2 / tr(X^2) of one Hodge sector.

    `sector` is "completed" for L_k + w Pi^h (the default), "hodge" for L_k, and "down"
    or "up" for one side. `weight` chooses w for the completed sector: "unit",
    "mean" or "energy". On a nonempty grade a completed count lies between 1 and the
    number of cells. The other sectors ignore the harmonic directions and give zero when
    their operator vanishes, and on an empty grade every sector gives zero.
    """
    _check(sector, weight)
    return grade_traces(rex, grade).effective_modes(sector, weight)


def harmonic_log(rex, grade: int = 1, sector: str = "completed", weight: str = "unit") -> float:
    """H2 = log n of one Hodge sector, the float logarithm of the exact count.

    The count is the exact carrier and this is its one transcendental step. A sector
    with no modes has no logarithm and is refused.
    """
    _check(sector, weight)
    return grade_traces(rex, grade).harmonic_log(sector, weight)
