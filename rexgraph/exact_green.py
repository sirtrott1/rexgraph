"""Exact augmented Green solve: the resolvent without assembling L or an inverse.

`(I + lambda L_k) x = j` is evaluated by solving one sparse coupled system in the
auxiliary coordinates of its own definition, over Q. The Hodge product `B* M B`, the
metric inverse and the resolvent itself are never assembled: the system's coefficients
are the original boundary and metric maps, their transposes, and rational constants.

That is not only cheaper, it is the reason the weighted case is well posed at all.
Forming `L1w = (B1 W)* (B1 W)` puts the solve on the normal equations, whose range is
`W . im B1*` while a boundary source lies in the UNWEIGHTED `im B1*`; those coincide
only at `W = cI`, so under any other weighting the assembled system is inconsistent and
a conjugate gradient cannot return what the pseudoinverse would. The augmented system
never forms that product and so never acquires the mismatch. Measured on a unit
triangle with one weight at 1 + 1e-6, the assembled float path fails outright while
this returns 8000003/16000008.

Specification: Exact Relational Field Calculus, "Exact action contracts without
eigensolves"; reference implementation `verification/exact_actions.py`.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

__all__ = ["ExactSparse", "BlockSystem", "exact_adjoint_apply",
           "exact_hodge_apply", "exact_green_apply", "exact_projector_apply",
           "exact_hodge_frames", "exact_hodge_decompose", "exact_boundary",
           "exact_grade_metric", "exact_hodge", "exact_green"]


def _rational(value) -> Fraction:
    """Reject inexact input rather than silently adopting a binary float."""
    if isinstance(value, bool):
        raise TypeError("a boolean is not an exact rational coefficient")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise TypeError(
        f"exact action coefficients must be int or Fraction; {type(value).__name__} "
        "is approximate. Convert at the source, where the exact value is known."
    )


class ExactSparse:
    """A rational sparse matrix that solves by elimination and never inverts."""

    __slots__ = ("nrows", "ncols", "entries")

    def __init__(self, nrows: int, ncols: int, entries):
        if nrows < 0 or ncols < 0:
            raise ValueError("matrix dimensions must be nonnegative")
        clean = {}
        for (i, j), value in dict(entries).items():
            if not (0 <= i < nrows and 0 <= j < ncols):
                raise IndexError((i, j))
            q = _rational(value)
            if q:
                clean[i, j] = q
        self.nrows, self.ncols, self.entries = int(nrows), int(ncols), clean

    @classmethod
    def identity(cls, n: int) -> ExactSparse:
        return cls(n, n, {(i, i): Fraction(1) for i in range(n)})

    @classmethod
    def diagonal(cls, values) -> ExactSparse:
        values = list(values)
        return cls(len(values), len(values),
                   {(i, i): _rational(v) for i, v in enumerate(values)})

    @classmethod
    def from_columns(cls, nrows: int, columns) -> ExactSparse:
        """Columns as {row: coefficient} dicts, which is how a boundary is stored."""
        columns = list(columns)
        return cls(nrows, len(columns),
                   {(int(i), j): v for j, col in enumerate(columns) for i, v in col.items()})

    @property
    def T(self) -> ExactSparse:
        return ExactSparse(self.ncols, self.nrows,
                           {(j, i): v for (i, j), v in self.entries.items()})

    def apply(self, vector):
        vector = [_rational(v) for v in vector]
        if len(vector) != self.ncols:
            raise ValueError(f"expected {self.ncols} coordinates, got {len(vector)}")
        out = [Fraction(0)] * self.nrows
        for (i, j), v in self.entries.items():
            out[i] += v * vector[j]
        return tuple(out)

    def _rows(self):
        rows = [dict() for _ in range(self.nrows)]
        for (i, j), v in self.entries.items():
            rows[i][j] = v
        return rows

    def solve(self, rhs):
        """Exact sparse Gaussian elimination. No inverse is constructed.

        The returned solution is checked by EXACT equality against the right hand
        side before it is returned: an exact method that reports an unverified
        answer has given up the only thing it was for.
        """
        if self.nrows != self.ncols:
            raise ValueError("an exact solve needs a square system")
        if len(rhs) != self.nrows:
            raise ValueError("right hand side does not match the system")
        n = self.ncols
        rows = self._rows()
        b = [_rational(v) for v in rhs]
        for k in range(n):
            choices = [i for i in range(k, n) if rows[i].get(k)]
            if not choices:
                raise ValueError("singular system: no unique solution")
            # Sparsity aware pivot: fewest entries first. A policy, not a fill bound.
            p = min(choices, key=lambda i: len(rows[i]))
            rows[k], rows[p] = rows[p], rows[k]
            b[k], b[p] = b[p], b[k]
            pivot = rows[k][k]
            for i in range(k + 1, n):
                if not rows[i].get(k):
                    continue
                ratio = rows[i].pop(k) / pivot
                for j, value in rows[k].items():
                    if j > k:
                        new = rows[i].get(j, Fraction(0)) - ratio * value
                        if new:
                            rows[i][j] = new
                        else:
                            rows[i].pop(j, None)
                b[i] -= ratio * b[k]
        x = [Fraction(0)] * n
        for i in range(n - 1, -1, -1):
            total = sum((v * x[j] for j, v in rows[i].items() if j > i), Fraction(0))
            x[i] = (b[i] - total) / rows[i][i]
        result = tuple(x)
        if self.apply(result) != tuple(_rational(v) for v in rhs):
            raise ArithmeticError("exact solve residual is nonzero")
        return result


    def rref(self):
        """Exact reduced rows and their pivot columns; the basis for the frames."""
        rows = self._rows()
        pivots, k = [], 0
        for j in range(self.ncols):
            candidates = [i for i in range(k, self.nrows) if rows[i].get(j)]
            if not candidates:
                continue
            p = min(candidates, key=lambda i: len(rows[i]))
            rows[k], rows[p] = rows[p], rows[k]
            scale = rows[k][j]
            rows[k] = {c: v / scale for c, v in rows[k].items()}
            for i in range(self.nrows):
                if i == k or not rows[i].get(j):
                    continue
                ratio = rows[i][j]
                for c, v in rows[k].items():
                    updated = rows[i].get(c, Fraction(0)) - ratio * v
                    if updated:
                        rows[i][c] = updated
                    else:
                        rows[i].pop(c, None)
            pivots.append(j)
            k += 1
            if k == self.nrows:
                break
        return (ExactSparse(self.nrows, self.ncols,
                            {(i, j): v for i, row in enumerate(rows) for j, v in row.items()}),
                tuple(pivots))

    def kernel_frame(self) -> ExactSparse:
        """An independent rational frame for ker(self), one column per free variable."""
        reduced, pivots = self.rref()
        free = [j for j in range(self.ncols) if j not in pivots]
        data = {}
        for a, j in enumerate(free):
            data[j, a] = Fraction(1)
            for i, pivot in enumerate(pivots):
                value = -reduced.entries.get((i, j), Fraction(0))
                if value:
                    data[pivot, a] = value
        return ExactSparse(self.ncols, len(free), data)

    def image_frame(self) -> ExactSparse:
        """The pivot columns: an INDEPENDENT frame spanning im(self).

        A projector needs independence, not merely a spanning set: the Gram of a
        redundant frame is singular and the solve refuses rather than inventing a
        pseudoinverse.
        """
        _, pivots = self.rref()
        index = {j: a for a, j in enumerate(pivots)}
        return ExactSparse(self.nrows, len(pivots),
                           {(i, index[j]): v for (i, j), v in self.entries.items() if j in index})

    def stack_rows(self, other: ExactSparse) -> ExactSparse:
        """[self ; other] on a shared column space, for a simultaneous kernel."""
        if self.ncols != other.ncols:
            raise ValueError("stacked maps must share a column space")
        data = dict(self.entries)
        for (i, j), v in other.entries.items():
            data[self.nrows + i, j] = v
        return ExactSparse(self.nrows + other.nrows, self.ncols, data)


class BlockSystem:
    """Assemble a coupled system from the original maps, placed at offsets."""

    __slots__ = ("dimension", "data")

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.data: dict[tuple[int, int], Fraction] = {}

    def put(self, row: int, col: int, matrix: ExactSparse, scale=1) -> None:
        scale = _rational(scale)
        for (i, j), v in matrix.entries.items():
            key = (row + i, col + j)
            value = self.data.get(key, Fraction(0)) + scale * v
            if value:
                self.data[key] = value
            else:
                self.data.pop(key, None)

    def matrix(self) -> ExactSparse:
        return ExactSparse(self.dimension, self.dimension, self.data)


def exact_adjoint_apply(boundary, source_metric, target_metric, y):
    """`B^dagger y = M_source^-1 B* M_target y`, by a transpose action and a solve.

    The metric solve is the whole adjoint. Dropping it leaves `B* M y`, which
    coincides with the adjoint only when the source metric is the identity -- so a
    fixture without weights cannot see the difference, and a weighted one fails
    outright rather than drifting.
    """
    return source_metric.solve(boundary.T.apply(target_metric.apply(y)))


def exact_hodge_apply(lower, upper, lower_metric, metric, upper_metric, x):
    """`L_k x` through its factors: down through B_k, up through B_{k+1}."""
    down = exact_adjoint_apply(lower, metric, lower_metric, lower.apply(x))
    if upper.ncols == 0:
        return tuple(down)
    up = upper.apply(exact_adjoint_apply(upper, upper_metric, metric, x))
    return tuple(a + b for a, b in zip(down, up, strict=True))


def exact_green_apply(lower, upper, lower_metric, metric, upper_metric, lam, source):
    """Solve `(I + lam L_k) x = j` exactly, forming neither L nor any inverse.

    The auxiliaries are the definition read one step at a time:
    `a = Mx`, `b = Bx`, `c = M_{k-1}b`, `d = C*a`, `M_{k+1}e = d`, `f = Ce`,
    `g = Mf`, and finally `a + lam B*c + lam g = Mj`. Eliminating them returns
    exactly `(I + lam L_k) x = j`, so the augmented system is equivalent on the
    same domain and uses only the original maps.
    """
    lam = _rational(lam)
    n, below, above = metric.nrows, lower.nrows, upper.ncols
    if (metric.ncols != n or lower.ncols != n or upper.nrows != n
            or (lower_metric.nrows, lower_metric.ncols) != (below, below)
            or (upper_metric.nrows, upper_metric.ncols) != (above, above)
            or len(source) != n):
        raise ValueError("boundary, metric and source dimensions disagree")
    sizes = [n, n, below, below, above, above, n, n]       # x a b c d e f g
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    ix, ia, ib, ic, id_, ie, if_, ig = offsets
    block = BlockSystem(total)
    row = 0
    block.put(row, ia, ExactSparse.identity(n)); block.put(row, ix, metric, -1); row += n
    block.put(row, ib, ExactSparse.identity(below)); block.put(row, ix, lower, -1); row += below
    block.put(row, ic, ExactSparse.identity(below)); block.put(row, ib, lower_metric, -1); row += below
    block.put(row, id_, ExactSparse.identity(above)); block.put(row, ia, upper.T, -1); row += above
    block.put(row, ie, upper_metric); block.put(row, id_, ExactSparse.identity(above), -1); row += above
    block.put(row, if_, ExactSparse.identity(n)); block.put(row, ie, upper, -1); row += n
    block.put(row, ig, ExactSparse.identity(n)); block.put(row, if_, metric, -1); row += n
    block.put(row, ia, ExactSparse.identity(n)); block.put(row, ic, lower.T, lam)
    block.put(row, ig, ExactSparse.identity(n), lam)
    rhs = (Fraction(0),) * row + metric.apply(source)
    solution = block.matrix().solve(rhs)[:n]
    check = exact_hodge_apply(lower, upper, lower_metric, metric, upper_metric, solution)
    residual = tuple(s + lam * h for s, h in zip(solution, check, strict=True))
    if residual != tuple(_rational(v) for v in source):
        raise ArithmeticError("Green residual is nonzero")
    return solution

def exact_projector_apply(frame, metric, x, *, assemble_gram=False):
    """Apply `Pi_F = F (F* M F)^-1 F* M` without assembling the projector.

    Two routes to the same exact value. The default keeps the Gram unassembled as
    well, solving the coupled system `y - Fa = 0`, `z - My = 0`, `F* z = F* M x`
    in `(a, y, z)`; `assemble_gram=True` forms `F* M F` and solves the smaller
    r x r system directly. Both return the same rationals; the coupled route is
    the one that needs no product of the original maps.

    `F` must be INDEPENDENT. A redundant frame has a singular Gram, and the exact
    solve refuses instead of quietly supplying a pseudoinverse -- take
    `image_frame()` first if the columns may be dependent.
    """
    n, r = frame.nrows, frame.ncols
    if metric.nrows != n or metric.ncols != n or len(x) != n:
        raise ValueError("projector frame, metric and vector dimensions disagree")
    if r == 0:
        return (Fraction(0),) * n                      # the zero projector
    rhs_small = frame.T.apply(metric.apply(x))
    if assemble_gram:
        gram_entries = {}
        for a in range(r):
            fa = [frame.entries.get((i, a), Fraction(0)) for i in range(n)]
            Mfa = metric.apply(fa)
            for b in range(r):
                fb = [frame.entries.get((i, b), Fraction(0)) for i in range(n)]
                value = sum((fb[i] * Mfa[i] for i in range(n)), Fraction(0))
                if value:
                    gram_entries[b, a] = value
        coefficients = ExactSparse(r, r, gram_entries).solve(rhs_small)
        return frame.apply(coefficients)
    total = r + 2 * n
    block = BlockSystem(total)
    block.put(0, 0, frame, -1)                         # y - F a = 0
    block.put(0, r, ExactSparse.identity(n))
    block.put(n, r, metric, -1)                        # z - M y = 0
    block.put(n, r + n, ExactSparse.identity(n))
    block.put(2 * n, r + n, frame.T)                   # F* z = F* M x
    rhs = (Fraction(0),) * (2 * n) + rhs_small
    return block.matrix().solve(rhs)[r:r + n]


def exact_hodge_frames(lower, upper, lower_metric, metric, upper_metric):
    """Independent rational frames for the three sectors of `C_k`.

    `C_k = im B_k^dagger (+) im B_{k+1} (+) H_k` in the declared metric, with
    `H_k = ker B_k ^ ker B_{k+1}^dagger`. The harmonic frame is a kernel solve of
    the STACKED adjacent maps, so it is constructed rather than read off a
    spectrum: no eigenvector and no thresholded eigenvalue appears anywhere here.

    Returns `(gradient, curl, harmonic)`, each an independent ExactSparse frame.
    """
    n = metric.nrows
    adjoint_entries = {}
    for j in range(lower.nrows):                       # B^dagger applied to each basis vector
        column = exact_adjoint_apply(
            lower, metric, lower_metric,
            [Fraction(1) if i == j else Fraction(0) for i in range(lower.nrows)])
        for i, v in enumerate(column):
            if v:
                adjoint_entries[i, j] = v
    gradient = ExactSparse(n, lower.nrows, adjoint_entries).image_frame()
    curl = upper.image_frame() if upper.ncols else ExactSparse(n, 0, {})
    if upper.ncols:
        upper_adjoint_entries = {}
        for j in range(n):
            column = exact_adjoint_apply(
                upper, upper_metric, metric,
                [Fraction(1) if i == j else Fraction(0) for i in range(n)])
            for i, v in enumerate(column):
                if v:
                    upper_adjoint_entries[i, j] = v
        upper_adjoint = ExactSparse(upper.ncols, n, upper_adjoint_entries)
        stacked = lower.stack_rows(upper_adjoint)
    else:
        stacked = lower
    harmonic = stacked.kernel_frame()
    return gradient, curl, harmonic


def exact_hodge_decompose(lower, upper, lower_metric, metric, upper_metric, x):
    """Split `x` into its gradient, curl and harmonic parts, exactly.

    The three parts sum back to `x` by exact equality, which is checked here: a
    decomposition whose parts do not reconstruct is not a decomposition.
    """
    gradient, curl, harmonic = exact_hodge_frames(
        lower, upper, lower_metric, metric, upper_metric)
    parts = tuple(exact_projector_apply(frame, metric, x)
                  for frame in (gradient, curl, harmonic))
    total = tuple(sum(values, Fraction(0)) for values in zip(*parts, strict=True))
    if total != tuple(_rational(v) for v in x):
        raise ArithmeticError("Hodge parts do not reconstruct the field")
    return parts

# Bridging a RexGraph into the exact carriers.

def exact_boundary(rex, grade: int) -> ExactSparse:
    """`B_grade` as exact rationals, rebuilt from the declared incidence.

    Never read back from the assembled float operator: the coefficients are -1 at the
    head and the declared shares elsewhere, and recovering those from a float would put
    an exact solve on the approximation tower for no reason.
    """
    rex._ensure_clean()
    if grade == 1:
        from rexgraph.faces import _exact_b1_block
        columns = _exact_b1_block(rex, range(int(rex.nE)))
        return ExactSparse.from_columns(int(rex.nV), columns)
    if grade == 2:
        b2 = getattr(rex, "_B2_hodge_dual", None)
        if b2 is None or int(getattr(rex, "nF_hodge", 0)) == 0:
            return ExactSparse(int(rex.nE), 0, {})
        # Read the stored CSC directly. The exact path must not require scipy: it is an
        # optional dependency and the native platform disables it outright, so routing
        # the face columns through a scipy conversion made the exact reading unavailable
        # exactly where being native matters most.
        col_ptr = np.asarray(b2.col_ptr)
        row_idx = np.asarray(b2.row_idx)
        values = np.asarray(b2.vals_csc)
        entries = {}
        for f in range(int(b2.ncol)):
            for k in range(int(col_ptr[f]), int(col_ptr[f + 1])):
                value = float(values[k])
                exact = Fraction(int(round(value)))
                if float(exact) != value:
                    raise ValueError(
                        "stored B2 coefficient is not an exact integer; the exact path "
                        "refuses a rounded face column rather than adopting it")
                if exact:
                    entries[int(row_idx[k]), f] = exact
        return ExactSparse(int(rex.nE), int(b2.ncol), entries)
    raise ValueError(f"exact boundary is declared for grades 1 and 2; got {grade}")


def exact_grade_metric(rex, grade: int, size: int) -> ExactSparse:
    """The declared grade metric, exactly. Absent means the identity, not a guess."""
    if grade == 1:
        weights = getattr(rex, "edge_metric_exact", None)
        if weights is None:
            return ExactSparse.identity(size)
        weights = list(weights)
        if len(weights) != size:
            raise ValueError("edge metric does not match the relation count")
        return ExactSparse.diagonal([_rational(w) for w in weights])
    return ExactSparse.identity(size)


def exact_hodge(rex, field, *, grade: int = 1):
    """`field` split into its gradient, curl and harmonic parts over Q.

    The exact reading of `RexGraph.hodge`. Its float counterpart is an oracle: it
    answers the same question in the approximation tower and its parts agree only to
    rounding. Inputs must be exact -- a float field is refused rather than adopted,
    because the binary value of 0.1 is not the number the caller wrote.
    """
    if grade != 1:
        raise ValueError("the exact Hodge split is declared at grade 1")
    lower = exact_boundary(rex, 1)
    upper = exact_boundary(rex, 2)
    nE = int(rex.nE)
    return exact_hodge_decompose(
        lower, upper,
        exact_grade_metric(rex, 0, int(rex.nV)),
        exact_grade_metric(rex, 1, nE),
        exact_grade_metric(rex, 2, upper.ncols),
        list(field))


def exact_green(rex, source, lam=1, *, grade: int = 1):
    """`(I + lam L_grade)^-1 source` over Q, forming neither L nor an inverse.

    This is the REGULARIZED resolvent, which preserves the harmonic component. It is a
    different object from `vertex_green`'s Moore Penrose action, which kills it; the two
    names must not be given each other's semantics.
    """
    if grade != 1:
        raise ValueError("the exact Green solve is declared at grade 1")
    lower = exact_boundary(rex, 1)
    upper = exact_boundary(rex, 2)
    return exact_green_apply(
        lower, upper,
        exact_grade_metric(rex, 0, int(rex.nV)),
        exact_grade_metric(rex, 1, int(rex.nE)),
        exact_grade_metric(rex, 2, upper.ncols),
        lam, list(source))


def _field_rational(value) -> Fraction:
    """One field coordinate as a rational, taking a float at its exact binary value.

    Deliberately weaker than `_rational`, and the two must not be merged. `_rational`
    guards OPERATOR coefficients, where a float means someone recovered a share from
    the assembled float operator and the exact value was already lost. This guards a
    FIELD the caller supplied, where -- by the same convention as `edge_metric_exact`
    -- a stored double means the rational it actually holds. 0.1 enters as
    3602879701896397/36028797018963968, which is the number in memory, not 1/10.
    """
    if isinstance(value, bool):
        raise TypeError("a boolean is not an exact field coordinate")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{value} is not a finite field coordinate")
        return Fraction(value)
    if isinstance(value, np.integer):
        return Fraction(int(value))
    if isinstance(value, np.floating):
        return _field_rational(float(value))
    raise TypeError(
        f"a field coordinate must be int, float or Fraction; got {type(value).__name__}")


def exact_field(values):
    """A supplied field as exact rationals, by `_field_rational` on every coordinate."""
    return [_field_rational(v) for v in values]


def exact_path_available(rex, grade: int = 1) -> bool:
    """Whether the exact path answers at this size, under the declared ceiling.

    The ceiling is policy, not mathematics: `configure_algorithms(exact_field_limit=...)`
    moves it. Rational elimination is cubic in the grade dimension with growing
    coefficients, so a large complex is answered by the float tower and reported as
    approximate rather than run for an unbounded time.
    """
    from rexgraph.core._common import get_algorithm_config
    limit = int(get_algorithm_config()["exact_field_limit"])
    if limit <= 0:
        return False
    rex._ensure_clean()
    size = int(rex.nV) if grade == 0 else int(rex.nE)
    return size <= limit
