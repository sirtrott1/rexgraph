"""rexgraph.graded_boundary: a general, graded, mixed-arity boundary builder.

A relational complex (rex) is a finite sequence of sparse boundary maps

    C_G --B_G--> C_{G-1} --> ... --> C_1 --B_1--> C_0

with entries on the exact integer/rational tower, satisfying the chain condition
``B_{d-1} B_d = 0`` at every consecutive pair. A C1 relation is special: its declared
column is ``(-1, 1/(k-1), ..., 1/(k-1))`` (or ``(+1)`` for a witness), with exact
integer representative ``(-(k-1), +1, ..., +1)``. Higher-grade columns are declared
as signed/integer coefficients. Relations are primitive and vertices are derived: a
vertex exists exactly when some column of B_1 is nonzero in its row.

Each ``B_d : C_d -> C_{d-1}`` is a *signed* sparse matrix whose COLUMNS carry any number
of nonzeros. The number of nonzeros in a column is the cell's *arity* and is INDEPENDENT
of its *grade* (dimension):

    nnz = 1  -> a "witness" cell (single face),
    nnz = 2  -> an ordinary cell (pairwise edge / bigon),
    nnz = k  -> a "branching" cell (k-ary edge, n-gon face, ...).

Pairwise and branching edges coexist; triangle, pentagon, hexagon and n-gon faces
coexist; and grades stack arbitrarily high. The only law is the chain condition

    d o d = 0   <=>   B_d @ B_{d+1} = 0   for every consecutive pair,

which is a *structural* (sparse) zero, never a densified product.

This module is the single, kernel-free source of truth for building, verifying and
reading graded boundaries. It is pure Python + scipy.sparse; it does not touch the
Cython core, and it is the generalization of
``rexgraph.dirac_propagator._boundaries_from_rex``.

Sign convention: a C1 relation given as a plain participant list ``[v0, v1, ...]`` has
``v0`` distinguished and receives the canonical shares above. At grades >= 2, a plain
boundary list ``[i0, i1, ...]`` has the first lower cell signed ``-1`` and every
remaining lower cell signed ``+1``. Explicit ``[(index, coefficient), ...]`` form is
accepted at every grade; C1 coefficients select orientation only and must have one
negative distinguished participant with positive shares.
"""
from __future__ import annotations

from collections import OrderedDict as _OrderedDict
from collections.abc import Sequence
from fractions import Fraction
from math import gcd
from numbers import Integral

import numpy as np
import scipy.sparse as sp

__all__ = [
    "build_graded_boundaries",
    "verify_chain",
    "graded_laplacians",
    "betti_numbers",
    "graded_boundaries_from_rex",
    "truncated_icosahedron_3rex",
    "solid_octahedron_3rex",
    "square_pyramid_3rex",
]

_f64 = np.float64
_I64 = np.iinfo(np.int64)


#### -
# Cell parsing
#### -
def _is_signed_cell(cell) -> bool:
    """True for explicit ``[(index, coefficient), ...]`` instead of bare indices.

    The builder carries higher-grade coefficients as sparse numeric values. Exact
    rational normalisation for a primary branching C1 is performed by
    :meth:`RexGraph.from_cells`; this generic builder preserves the supplied numeric
    carrier for inspection and chain verification.
    """
    if len(cell) == 0:
        return False
    for x in cell:
        if not isinstance(x, (tuple, list, np.ndarray)):
            return False
        if len(x) != 2:
            return False
        try:
            s = float(Fraction(x[1]))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(s) or s == 0.0:
            return False
    return True


def _exact_cell_index(value, *, context: str) -> int:
    """Read a graded basis address without coercing a float or boolean."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{context} must be an exact integral basis index")
    integer = int(value)
    if integer < _I64.min or integer > _I64.max:
        raise ValueError(f"{context} is outside int64")
    return integer


def _cell_entries(cell) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, signs)`` for one cell.

    Plain form ``[i0, i1, ...]`` -> first index ``-1``, the rest ``+1`` (positional).
    Explicit form ``[(i, c), ...]`` -> exactly as given in its numeric carrier.
    """
    if _is_signed_cell(cell):
        idx = np.fromiter(
            (_exact_cell_index(x[0], context="explicit cell index") for x in cell),
            dtype=np.int64,
            count=len(cell),
        )
        sgn = np.fromiter((float(Fraction(x[1])) for x in cell), dtype=_f64, count=len(cell))
        return idx, sgn
    raw = np.asarray(cell)
    if raw.ndim != 1:
        raise ValueError("plain cell boundary must be a one-dimensional basis-index list")
    idx = np.fromiter(
        (_exact_cell_index(value, context="plain cell index") for value in raw),
        dtype=np.int64,
        count=raw.size,
    )
    sgn = np.ones(idx.shape[0], dtype=_f64)
    if idx.shape[0] >= 1:
        sgn[0] = -1.0
    return idx, sgn


def _canonical_c1_entries(cell, cell_index: int):
    """Return one declared C1 boundary in display and exact forms.

    ``float64`` is only the sparse display carrier for the canonical share.  The
    accompanying dict is built straight over :class:`Fraction` and is what the
    branching C2 importer solves against, so no coefficient is recovered from a
    decimal approximation.
    """
    idx, sgn = _cell_entries(cell)
    if idx.size == 0:
        raise ValueError(f"grade-1 cell {cell_index} has empty boundary support")
    self_loop = idx.size == 2 and int(idx[0]) == int(idx[1])
    if not self_loop and np.unique(idx).size != idx.size:
        raise ValueError(f"grade-1 cell {cell_index} repeats a C0 boundary participant")
    if self_loop:
        # A deliberate [v, v] relation exists but carries no C0 boundary.  Keep both
        # occurrences in the sparse display carrier so the primary support survives;
        # scipy sums the signed entries to its exact zero column.
        v = int(idx[0])
        return idx, np.asarray([-1.0, 1.0], dtype=_f64), {}
    if idx.size == 1:
        v = int(idx[0])
        return idx, np.ones(1, dtype=_f64), {v: Fraction(1)}

    head = np.flatnonzero(sgn < 0)
    shares = np.flatnonzero(sgn > 0)
    if (head.size != 1 or shares.size != idx.size - 1
            or not np.all(np.abs(sgn) == 1.0)):
        raise ValueError(
            f"grade-1 cell {cell_index} must have one negative distinguished "
            "participant and positive unit orientation signs"
        )
    h = int(head[0])
    ordered = np.concatenate((idx[h:h + 1], np.delete(idx, h)))
    display = np.full(ordered.shape[0], 1.0 / (ordered.shape[0] - 1), dtype=_f64)
    display[0] = -1.0
    share = Fraction(1, int(ordered.shape[0] - 1))
    exact = {int(ordered[0]): Fraction(-1)}
    exact.update({int(v): share for v in ordered[1:]})
    return ordered, display, exact


def _exact_higher_coefficient(value) -> Fraction:
    """Read an explicit branching C2 coefficient without float reconstruction."""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(
                "explicit branching C2 coefficients must be integers, Fraction values, "
                "or rational strings; a nonintegral float is not an exact carrier"
            )
        q = Fraction(int(value))
    else:
        try:
            q = Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("explicit branching C2 coefficients must be exact rationals") from exc
    if q == 0:
        raise ValueError("an explicit branching C2 coefficient cannot be zero")
    return q


def _exact_face_column(c1_columns, edge_ids):
    """One primitive exact null vector for C1 support, or ``None`` if none closes."""
    cols = [c1_columns[int(e)] for e in edge_ids]
    verts = sorted({v for col in cols for v in col})
    m = len(cols)
    if not verts or m == 0:
        return None
    row_of = {v: i for i, v in enumerate(verts)}
    A = [[Fraction(0)] * m for _ in verts]
    for j, col in enumerate(cols):
        for v, coefficient in col.items():
            A[row_of[v]][j] = coefficient

    pivot_columns: list[int] = []
    row = 0
    for column in range(m):
        pivot = next((i for i in range(row, len(verts)) if A[i][column] != 0), None)
        if pivot is None:
            continue
        A[row], A[pivot] = A[pivot], A[row]
        inv = Fraction(1) / A[row][column]
        A[row] = [value * inv for value in A[row]]
        for i in range(len(verts)):
            if i != row and A[i][column] != 0:
                factor = A[i][column]
                A[i] = [a - factor * b for a, b in zip(A[i], A[row], strict=True)]
        pivot_columns.append(column)
        row += 1
        if row == len(verts):
            break

    pivots = set(pivot_columns)
    free = [column for column in range(m) if column not in pivots]
    if not free:
        return None
    vector = [Fraction(0)] * m
    vector[free[0]] = Fraction(1)
    for i, pivot in enumerate(pivot_columns):
        vector[pivot] = -A[i][free[0]]
    lead = next((value for value in vector if value != 0), None)
    if lead is None:
        return None
    vector = [value / lead for value in vector]
    return _primitive_integer_column(vector)


def _primitive_integer_column(coefficients) -> list[int]:
    """Clear rational denominators and common integer scale, retaining orientation."""
    denominator = 1
    for coefficient in coefficients:
        denominator = denominator * coefficient.denominator // gcd(
            denominator, coefficient.denominator
        )
    integers = [int(coefficient * denominator) for coefficient in coefficients]
    common = 0
    for value in integers:
        common = gcd(common, abs(value))
    if common > 1:
        integers = [value // common for value in integers]
    if any(abs(value) > 2 ** 53 for value in integers):
        raise ValueError(
            "grade-2 coefficient exceeds the exact integer range of the current B2 storage"
        )
    return integers


def _build_branching_c2(n_edges: int, cells, c1_columns) -> sp.csr_matrix:
    """Derive/validate C2 over canonical branching C1, exactly and sparsely."""
    rows: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    for face, cell in enumerate(cells):
        explicit = (
            len(cell) > 0
            and all(isinstance(item, (tuple, list, np.ndarray)) and len(item) == 2
                    for item in cell)
        )
        if explicit:
            edge_ids = np.fromiter((int(item[0]) for item in cell), dtype=np.int64,
                                   count=len(cell))
            coefficients = [_exact_higher_coefficient(item[1]) for item in cell]
        else:
            edge_ids = np.asarray(cell, dtype=np.int64).ravel()
            coefficients = None
        if edge_ids.size == 0:
            raise ValueError(f"grade-2 cell {face} has empty relation support")
        if np.any(edge_ids < 0) or np.any(edge_ids >= n_edges):
            raise ValueError(f"grade-2 cell {face} refers to a relation outside C1")
        if np.unique(edge_ids).size != edge_ids.size:
            raise ValueError(f"grade-2 cell {face} repeats a relation in one boundary")

        if coefficients is None:
            integers = _exact_face_column(c1_columns, edge_ids)
            if integers is None:
                raise ValueError(
                    f"grade-2 cell {face} does not close over the declared C1 relations; "
                    "no face is attached"
                )
        else:
            residual: dict[int, Fraction] = {}
            for coefficient, edge in zip(coefficients, edge_ids, strict=True):
                for vertex, value in c1_columns[int(edge)].items():
                    residual[vertex] = residual.get(vertex, Fraction(0)) + coefficient * value
            if any(residual.values()):
                raise ValueError(
                    f"grade-2 cell {face} does not satisfy the exact canonical C1 "
                    "chain condition"
                )
            integers = _primitive_integer_column(coefficients)

        for edge, value in zip(edge_ids, integers, strict=True):
            if value:
                rows.append(int(edge))
                columns.append(face)
                values.append(value)
    return sp.coo_matrix(
        (np.asarray(values, dtype=np.int64),
         (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64))),
        shape=(n_edges, len(cells)),
    ).tocsr()


def build_graded_boundaries(cells_by_grade) -> list[sp.csr_matrix]:
    """Build the signed boundary maps ``[B_1, B_2, ..., B_G]`` of a graded complex.

    Parameters
    ----------
    cells_by_grade : sequence
        ``cells_by_grade[0]`` is the vertex count ``n_V`` (an int). For ``d >= 1``,
        ``cells_by_grade[d]`` is a list of d-cells; each d-cell is either a plain
        list of ``(d-1)``-cell indices (positional signs: first ``-1``, rest ``+1``)
        or an explicit ``[(index, sign), ...]`` list. Mixed arity within a grade is
        allowed and expected.

    Returns
    -------
    list of scipy.sparse.csr_matrix
        ``B_1`` is the canonical C1 share boundary.  When any C1 relation is
        branching or a witness, C2 is derived/validated over that exact boundary and
        returned with primitive integer coefficients. Higher maps have shape
        ``(n_{d-1}, n_d)`` with arbitrary column arity. Length ``G`` where ``G`` is
        the top grade present.
    """
    if len(cells_by_grade) == 0:
        raise ValueError("cells_by_grade must at least declare the vertex count")
    n_prev = _exact_cell_index(cells_by_grade[0], context="vertex count")
    if n_prev < 0:
        raise ValueError("vertex count must be nonnegative")
    boundaries: list[sp.csr_matrix] = []
    c1_columns = None
    c1_is_pairwise = True

    for d in range(1, len(cells_by_grade)):
        cells = cells_by_grade[d]
        n_cells = len(cells)
        if d == 2 and not c1_is_pairwise:
            # C2 is not a positional sign template over a primary branching C1.
            # Derive the only coefficients that satisfy the actual declared C1
            # boundary, or validate an explicit exact coefficient vector.
            B = _build_branching_c2(n_prev, cells, c1_columns)
            boundaries.append(B)
            n_prev = n_cells
            continue
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for j, cell in enumerate(cells):
            if d == 1:
                idx, sgn, exact = _canonical_c1_entries(cell, j)
                if c1_columns is None:
                    c1_columns = []
                c1_columns.append(exact)
                c1_is_pairwise = c1_is_pairwise and idx.shape[0] == 2
            else:
                idx, sgn = _cell_entries(cell)
                if idx.shape[0] == 0:
                    raise ValueError(f"grade-{d} cell {j} has empty boundary support")
            if np.any(idx < 0) or np.any(idx >= n_prev):
                raise ValueError(
                    f"grade-{d} cell {j} refers to a lower-grade basis index outside C{d - 1}"
                )
            rows.append(idx)
            cols.append(np.full(idx.shape[0], j, dtype=np.int64))
            vals.append(sgn)
        if rows:
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            v = np.concatenate(vals)
        else:
            r = np.zeros(0, dtype=np.int64)
            c = np.zeros(0, dtype=np.int64)
            v = np.zeros(0, dtype=_f64)
        B = sp.coo_matrix((v, (r, c)), shape=(n_prev, n_cells)).tocsr()
        boundaries.append(B)
        n_prev = n_cells

    return boundaries


#### -
# Verification, Laplacians, homology
#### -
def _canonical_c1_columns(B1: sp.spmatrix):
    """Exact C1 columns if ``B1`` is the declared canonical carrier, else ``None``.

    This does not recover rationals from arbitrary floats. It recognises only the
    unique C1 representation declared by the relational complex axiom: one literal
    ``-1`` and literal equal ``1/(k-1)`` shares, or one literal ``+1`` witness. Once
    recognised, the exact fractions are reconstructed from arity, not from decimals.
    """
    A = B1.tocsc(copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    columns = []
    for edge in range(A.shape[1]):
        lo, hi = int(A.indptr[edge]), int(A.indptr[edge + 1])
        rows = A.indices[lo:hi]
        values = A.data[lo:hi]
        arity = int(rows.size)
        if arity == 0:
            columns.append({})
            continue
        if arity == 1:
            if values[0] != 1.0:
                return None
            columns.append({int(rows[0]): Fraction(1)})
            continue
        heads = np.flatnonzero(values == -1.0)
        if heads.size != 1:
            return None
        share = 1.0 / (arity - 1)
        shares = np.flatnonzero(values == share)
        if shares.size != arity - 1:
            return None
        head = int(heads[0])
        column = {int(rows[head]): Fraction(-1)}
        column.update({int(rows[i]): Fraction(1, arity - 1) for i in shares})
        columns.append(column)
    return columns


def _integer_columns(B: sp.spmatrix):
    """Sparse columns as exact integer dictionaries, or ``None`` for numeric input."""
    A = B.tocsc(copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    if A.data.size and not np.all(A.data == np.round(A.data)):
        return None
    out = []
    for column in range(A.shape[1]):
        lo, hi = int(A.indptr[column]), int(A.indptr[column + 1])
        out.append({int(row): Fraction(int(round(float(value))))
                    for row, value in zip(A.indices[lo:hi], A.data[lo:hi], strict=True)})
    return out


def _exact_composition_residual(lower_columns, upper_columns) -> Fraction:
    """Maximum coefficient of a sparse exact column composition."""
    maximum = Fraction(0)
    for upper in upper_columns:
        total: dict[int, Fraction] = {}
        for middle, upper_value in upper.items():
            for row, lower_value in lower_columns[middle].items():
                total[row] = total.get(row, Fraction(0)) + lower_value * upper_value
        for value in total.values():
            if abs(value) > maximum:
                maximum = abs(value)
    return maximum


def _exact_chain_residual(boundaries: Sequence[sp.spmatrix]):
    """Exact residual for canonical C1 + integral higher maps, else ``None``."""
    if len(boundaries) < 2:
        return Fraction(0)
    lower = _canonical_c1_columns(boundaries[0])
    if lower is None:
        return None
    maximum = Fraction(0)
    for upper_boundary in boundaries[1:]:
        upper = _integer_columns(upper_boundary)
        if upper is None or len(lower) != upper_boundary.shape[0]:
            return None
        residual = _exact_composition_residual(lower, upper)
        if residual > maximum:
            maximum = residual
        lower = upper
    return maximum


def verify_chain(boundaries: Sequence[sp.spmatrix], tol: float = 1e-9) -> tuple[bool, float]:
    """Sparsely check ``B_d @ B_{d+1} == 0`` for every consecutive pair.

    A canonical C1 plus integral higher tower takes the exact Fraction/sparse-column
    route: C1 shares are reconstructed from declared arity, never from a decimal, and
    every composition must vanish at literal zero. Genuinely non-exact matrices retain
    the numerical sparse fallback, whose ``tol`` is therefore an explicit oracle
    contract rather than the ordinary relational complex path. Neither route densifies.

    Returns
    -------
    (ok, max_residual)
        ``ok`` is True iff ``max_residual <= tol``.
    """
    exact = _exact_chain_residual(boundaries)
    if exact is not None:
        return exact == 0, float(exact)

    max_res = 0.0
    for d in range(len(boundaries) - 1):
        prod = (boundaries[d].tocsr() @ boundaries[d + 1].tocsr())
        if prod.nnz:
            max_res = max(max_res, float(np.abs(prod.data).max()))
    return (max_res <= tol), max_res


def graded_laplacians(boundaries: Sequence[sp.spmatrix]) -> list[sp.csr_matrix]:
    """The Hodge Laplacian ``L_d`` per grade, sparse.

    ``L_d = B_d^T B_d + B_{d+1} B_{d+1}^T`` with the boundary terms dropped where
    they do not exist, i.e. ``L_0 = B_1 B_1^T`` and the top-grade Laplacian is
    ``L_G = B_G^T B_G``.

    Returns a list of length ``G + 1`` (one operator per grade ``0..G``).
    """
    B = [b.tocsr() for b in boundaries]
    G = len(B)                          # top grade index; grades run 0..G
    sizes = [B[0].shape[0]] + [b.shape[1] for b in B] if B else [0]
    out: list[sp.csr_matrix] = []
    for g in range(G + 1):
        n_g = sizes[g]
        L = sp.csr_matrix((n_g, n_g), dtype=_f64)
        if g >= 1:                       # down: B_g^T B_g,  B_g = B[g-1]
            L = L + (B[g - 1].T @ B[g - 1])
        if g <= G - 1:                   # up: B_{g+1} B_{g+1}^T,  B_{g+1} = B[g]
            L = L + (B[g] @ B[g].T)
        out.append(L.tocsr())
    return out


def _is_integer_matrix(M: sp.spmatrix) -> bool:
    """True if every stored entry is an integer."""
    d = M.data
    return d.size == 0 or bool(np.all(d == np.round(d)))


def _rational_data(M: sp.spmatrix):
    """The stored entries as exact Python ints, or None if any is not exactly an integer.

    Deliberately NOT a float-to-rational reconstruction. A boundary column carries the
    share 1/(k-1), so it looks rational, but it has an exact INTEGER representative:
    scaling by (k-1) gives (-(k-1), +1, ..., +1), still zero-sum and still (-1,+1) at
    k=2. Rank is invariant under column scaling, so the rank path should be handed that
    integer form and never see a fraction. Recovering 1/3 from its nearest double by
    continued fraction would be answering a question that should not have been asked.
    See `RexGraph._integer_B1`.

    These were Fractions, which was carrying a denominator that is provably always 1:
    the integrality check on the line above is what makes every entry an integer, and
    the reduction below keeps them integral by cross-multiplying rather than dividing.
    """
    d = M.data
    if d.size and not bool(np.all(d == np.round(d))):
        return None
    return [int(round(float(x))) for x in d]


# Content-addressed memo for the exact integer rank. The rational column reduction below
# is the dominant cost on a large monitor step, and the SAME integer boundary map is
# reduced more than once per step (e.g. the pairwise interaction complex and the faced
# coordination complex share an identical B1). The key is the matrix's exact canonical
# content (shape + CSC structure + rounded integer data), so a hit returns a value that is
# byte-for-byte the same matrix - zero collision/staleness risk (dict compares keys
# exactly). Bounded so it never grows without limit; a race only ever costs a redundant
# (correct) recompute, so it is safe under the coordinator's thread lane too.
_RANK_MEMO: _OrderedDict[tuple, int] = _OrderedDict()
_RANK_MEMO_MAX = 64


def _exact_rank_reduction(M: sp.spmatrix, *, with_pivots: bool = False):
    """EXACT rank of an INTEGER sparse matrix by column reduction over Z: eigen-free,
    NO SVD, no eigendecomposition, no dense operator. Each column is reduced against the
    registered pivots (lowest-nonzero-row 'low' convention, as in persistence
    reduction); rank = number of columns that keep a pivot. This is the canon's
    `rank(B_k) via Z/Q elimination` (Part III). Columns are sparse dicts, so cost tracks
    FILL, not n^3, which is what the two choices below are both about.

    **The arithmetic stays in Z.** `col <- piv[low]*col - col[low]*piv` clears the
    pivot entry the same way `col -= (col[low]/piv[low])*piv` does, without ever
    forming a quotient, and dividing the result through by its gcd keeps the integers
    from growing. A column op scaled by a nonzero integer is still a column op, so the
    rank is untouched. The Fraction form was calling `math.gcd` twice per elementary
    operation to normalise denominators that the integrality check guarantees are 1.

    **Columns are reduced sparsest-first.** Rank does not depend on the order columns
    are presented in, but fill very much does: a wide column reduced early becomes a
    dense pivot that every later column must then reduce against. Persistence needs the
    input order because it is pairing births with deaths; this routine returns one
    integer and is free to choose. Measured over three Gutenberg documents (nE 7,899 to
    12,500), against the Fraction path: 14.5x, 30.1x and 45.3x for the same rank, the
    ratio growing with size because the fill it avoids is superlinear.

    `with_pivots` also returns the pivot ROW indices, which are a maximal independent
    set of rows: each reduced column has its lowest nonzero at its own pivot row, so the
    pivot columns restricted to the pivot rows are triangular with a nonzero diagonal.
    Anything wanting an invertible submatrix wants exactly that set, and it is already
    built here, so asking for it costs nothing. The memo carries the rank only, so a
    request for pivots reduces rather than reading it back.

    Memoized on exact matrix content (see :data:`_RANK_MEMO`)."""
    from math import gcd
    A = M.tocsc()
    A.sum_duplicates()                      # MUST precede the read: a self-loop stores -1
                                            # and +1 at the same (row, col), and the column
                                            # build below is a dict, so an unsummed pair
                                            # OVERWRITES rather than cancels and a zero
                                            # column registers a spurious pivot. Measured:
                                            # rank 2 on a matrix of rank 1.
    A.sort_indices()                        # canonical CSC for a stable content key
    indptr, indices, data = A.indptr, A.indices, A.data
    exact = _rational_data(A)
    if exact is None:
        return (None, None) if with_pivots else None   # caller falls back to the float path
    key = (A.shape, indptr.tobytes(), indices.tobytes(), data.tobytes())
    hit = _RANK_MEMO.get(key)
    if hit is not None and not with_pivots:
        _RANK_MEMO.move_to_end(key)
        return hit

    pivots: dict = {}                       # pivot_row -> reduced column {row: int}
    rank = 0
    widths = np.diff(indptr)
    for j in np.argsort(widths, kind="stable"):
        j = int(j)
        col = {}
        for k in range(indptr[j], indptr[j + 1]):
            v = exact[k]
            if v:                           # never register an explicit zero as a pivot
                col[int(indices[k])] = v
        while col:
            low = max(col)                  # 'low' pivot = highest row index present
            piv = pivots.get(low)
            if piv is None:
                pivots[low] = col
                rank += 1
                break
            a, b = piv[low], col[low]       # col <- a*col - b*piv, exact and integral
            new = {r: a * val for r, val in col.items()}
            for r, val in piv.items():
                nv = new.get(r, 0) - b * val
                if nv:
                    new[r] = nv
                else:
                    new.pop(r, None)
            g = 0
            for val in new.values():        # scale out the common factor: same column
                g = gcd(g, val)             # direction, integers that stay small
            col = {r: v // g for r, v in new.items()} if g > 1 else new

    _RANK_MEMO[key] = rank
    _RANK_MEMO.move_to_end(key)
    if len(_RANK_MEMO) > _RANK_MEMO_MAX:
        _RANK_MEMO.popitem(last=False)
    return (rank, sorted(pivots)) if with_pivots else rank


def _pairwise_rank(M: sp.spmatrix):
    """``rank = n_rows - components`` when every column is a signed pairwise edge.

    An arity-two column is ``(-c, +c)`` on two rows, which is the incidence matrix of
    a graph, and there the rank identity is combinatorial: each component contributes
    one dimension to the kernel of ``B_1^T`` and nothing else does. Union-find answers
    it in near-linear time, where the general column reduction is quadratic in the
    fill it creates and carries Fraction arithmetic per entry, so on a large complex
    the shortcut is the difference between seconds and minutes for the same integer.

    The identity is NOT general, which is why this is a guarded pre-check rather than
    the path. An arity-k relation touches k vertices while contributing rank one, so
    a lone arity-4 relation is one component with rank 1, not 3. Any column with more
    than two entries, or two entries that are not negatives of each other, returns
    None and the exact reduction runs.

    Returns None when the shortcut does not apply.
    """
    A = M.tocsc()
    A.sum_duplicates()                      # a self-loop stores -1 and +1 on one row
    counts = np.diff(A.indptr)
    if counts.size and counts.max() > 2:
        return None                         # a branching relation: rank is not n - c
    data, indptr = A.data, A.indptr
    for j in np.nonzero(counts == 2)[0]:
        a, b = data[indptr[j]], data[indptr[j] + 1]
        if a + b != 0:
            return None                     # not a boundary column: unsigned, or scaled
    if counts.size and (counts == 1).any():
        return None                         # a column reaching one vertex is not zero-sum
    return int(M.shape[0]) - _beta0_components(A)



def _spanned_branching_rank(M: sp.spmatrix):
    """`nV - components(pairwise part)` when the pairs SPAN every branching column.

    `_pairwise_rank` refuses any column with more than two entries, correctly: a lone
    arity-4 relation has rank 1 while its support is one component, so `n - c` is wrong
    there. But a MIXED complex, one carrying each group together with pairwise contacts
    that connect it, is a different case. A connected set's zero-sum space has dimension
    `k - 1`, so a branching column whose support lies inside ONE component of the pairwise
    subgraph is already spanned by it and adds no rank. What is left is a pairwise
    boundary map, where the combinatorial identity does hold.

    The condition is read off `M` alone and needs nothing about how it was built: take
    the arity-2 columns, union-find them, and check every wider column lands in one
    component. Refuses otherwise, so the exact reduction still runs wherever this does
    not apply.

    This is the difference between 14.3s and half an hour on a real lexical complex of
    1,626,490 relations, for the same integer.
    """
    A = M.tocsc()
    A.sum_duplicates()
    counts = np.diff(A.indptr)
    if counts.size == 0:
        return None
    two = np.flatnonzero(counts == 2)
    wide = np.flatnonzero(counts > 2)
    if two.size == 0 or wide.size == 0:
        return None                         # no mix: the existing paths already cover it
    if (counts == 1).any():
        return None                         # a witness column is not zero-sum
    data, indptr, indices = A.data, A.indptr, A.indices
    for j in two:                           # every 2-column must be a boundary column
        a, b = data[indptr[j]], data[indptr[j] + 1]
        if a + b != 0:
            return None
    nV = int(A.shape[0])
    parent = np.arange(nV, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for j in two:
        r0, r1 = int(indices[indptr[j]]), int(indices[indptr[j] + 1])
        a, b = find(r0), find(r1)
        if a != b:
            parent[b] = a
    for j in wide:
        rows = indices[indptr[j]:indptr[j + 1]]
        root = find(int(rows[0]))
        for r in rows[1:]:
            if find(int(r)) != root:
                return None                 # this group is fragmented: it DOES add rank
    return nV - _beta0_components(A[:, two])


def _column_integer_form(M: sp.spmatrix, atol: float = 1e-9):
    """`M` with each column scaled to integer entries, or None if that is not possible.

    Rank is invariant under nonzero column scaling, so this changes nothing the rank
    path cares about. The scale tried is the reciprocal of the column's smallest
    magnitude, which is exactly what clears the share: a column (-1, s, ..., s) with
    s = 1/(k-1) becomes (-(k-1), 1, ..., 1).
    """
    A = M.tocsc(copy=True)
    A.sum_duplicates()
    data = A.data.astype(float, copy=True)
    for j in range(A.shape[1]):
        lo, hi = A.indptr[j], A.indptr[j + 1]
        if hi <= lo:
            continue
        seg = data[lo:hi]
        nz = np.abs(seg[np.abs(seg) > atol])
        if nz.size == 0:
            continue
        scaled = seg / nz.min()
        if not np.all(np.abs(scaled - np.round(scaled)) <= 1e-6 * np.maximum(1.0, np.abs(scaled))):
            return None
        data[lo:hi] = np.round(scaled)
    out = sp.csc_matrix((data, A.indices.copy(), A.indptr.copy()), shape=A.shape)
    out.eliminate_zeros()
    return out


def _sparse_rank(M: sp.spmatrix, tol: float = 1e-9) -> int:
    """Rank of a sparse matrix. Betti comes from RANKS (the canon), not spectra.

    For INTEGER boundary maps (the unweighted topology) rank is computed EXACTLY and
    EIGEN-FREE by rational column reduction (:func:`_exact_rank_reduction`), the
    canon's Z/Q-elimination path - no SVD, no dense operator. Only genuinely
    non-integer (float-weighted) matrices fall back to the dense/truncated SVD.

    A pairwise boundary map takes the combinatorial identity first
    (:func:`_pairwise_rank`), and a MIXED map whose pairs span its branching columns
    takes the same identity through :func:`_spanned_branching_rank`. Both are the same
    exact integer by a cheaper route, and both refuse rather than guess.
    """
    if M.nnz == 0 or min(M.shape) == 0:
        return 0
    quick = _pairwise_rank(M)
    if quick is not None:
        return quick
    quick = _spanned_branching_rank(M)      # the mixed construction, same identity
    if quick is not None:
        return quick
    exact = _exact_rank_reduction(M)
    if exact is not None:
        return exact
    # A boundary column carries the share 1/(k-1), so it reads as a float matrix while
    # having an exact integer representative: scaling the column by (k-1) gives
    # (-(k-1), +1, ..., +1). Rank is invariant under column scaling, so clear the
    # denominators and retry the exact path rather than falling to a float estimate.
    scaled = _column_integer_form(M)
    if scaled is not None:
        exact = _exact_rank_reduction(scaled)
        if exact is not None:
            return exact
    m, n = M.shape
    # Densify only when the matrix is small enough to be harmless; boundary maps of
    # the complexes this module builds are far below this bound.
    if min(m, n) <= 1500:
        s = np.linalg.svd(M.toarray(), compute_uv=False)
        if s.size == 0:
            return 0
        thresh = tol * s[0] * max(m, n)
        return int(np.sum(s > max(thresh, tol)))
    # Large, and neither exact path applied. A truncated SVD can only CONFIRM a rank
    # below its own k; if every computed singular value clears the threshold the rank is
    # at least k and this routine does not know it. Returning k there reports a cap as a
    # measurement, which is how a branching BindingDB complex read rank 400 against a
    # true 4673. So it answers when it can and raises when it cannot.
    k = min(min(m, n) - 1, 400)
    s = sp.linalg.svds(M.asfptype(), k=k, return_singular_vectors=False)
    thresh = tol * s.max() * max(m, n)
    kept = int(np.sum(s > max(thresh, tol)))
    if kept >= k:
        raise ValueError(
            f"rank of a {m}x{n} operator is at least {k} and is not determined by a "
            f"truncated SVD at k={k}. Hand this path an exact integer representative "
            f"(see RexGraph._integer_B1) or densify deliberately; it will not return a "
            f"cap as if it were the rank.")
    return kept


def _beta0_components(B1: sp.spmatrix) -> int:
    """Number of connected components over the vertices, from the 0/1 incidence
    pattern of ``B_1`` (combinatorial, via union-find on the graph whose cliques are
    the edge supports). Isolated vertices count as components.

    This reads the SUPPORT of B_1, so it is a reading of the EXISTENCE tensor, which
    is a first-class object and not a degraded one. The composite datum factors into
    three separable axes and each is a usable operator:

        existence     which cells bound which           supp(B_1)
        orientation   the sign                          the whole of the F channel
        share         how the mass is divided           1/(k-1), and what carries arity

    The component count is the correct and complete answer for the first of them: how
    many pieces the incidence pattern falls into. What it is NOT is ``beta_0``, which
    is a reading of the BOUNDARY: ``n_0 - rank(B_1)``, how many directions the boundary
    fails to reach. The two coincide exactly when every relation has arity two, because
    an arity-k relation touches k vertices while contributing rank one. A lone arity-4
    relation is one component and has ``beta_0 = 3``, and both numbers are right about
    their own object.

    So use this when the question is about existence, and :func:`betti_numbers` when it
    is about homology. Answering one with the other is the error, not calling this.
    """
    nV = B1.shape[0]
    if nV == 0:
        return 0
    Bc = B1.tocsc()
    parent = list(range(nV))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    indptr, indices = Bc.indptr, Bc.indices
    for e in range(Bc.shape[1]):
        support = indices[indptr[e]:indptr[e + 1]]
        for k in range(1, len(support)):
            union(int(support[0]), int(support[k]))
    return len({find(v) for v in range(nV)})


def betti_numbers(boundaries: Sequence[sp.spmatrix], tol: float = 1e-9) -> list[int]:
    """Betti numbers ``[beta_0, ..., beta_G]`` from ranks.

    ``beta_g = dim ker(B_g) - rank(B_{g+1}) = n_g - rank(B_g) - rank(B_{g+1})`` with
    ``rank(B_0) = rank(B_{G+1}) = 0``. Grade 0 is the same formula:
    ``beta_0 = dim ker(B_1^T) = n_0 - rank(B_1)``.

    That is NOT a component count once any relation has arity above two.
    ``rank(B_1) = n_0 - c`` is a graph identity: a relation of arity k touches k
    vertices while contributing rank one, so reaching a new vertex stops meaning
    reaching a new direction, and only the second is what beta_0 counts. A lone
    arity-4 relation is one component and has ``beta_0 = 3``. Taking the component
    count there breaks the Euler characteristic, which is the property that fixes the
    convention without appeal to taste. The two agree on every pairwise complex.
    """
    B = [b.tocsr() for b in boundaries]
    G = len(B)
    if G == 0:
        return []
    sizes = [B[0].shape[0]] + [b.shape[1] for b in B]
    ranks = [_sparse_rank(b, tol) for b in B]        # ranks[d] = rank(B_{d+1})

    betti: list[int] = []
    for g in range(G + 1):
        n_g = sizes[g]
        rank_down = ranks[g - 1] if g >= 1 else 0    # rank(B_g)
        rank_up = ranks[g] if g <= G - 1 else 0      # rank(B_{g+1})
        if g == 0:
            betti.append(int(n_g - rank_up))         # n_0 - rank(B_1)
        else:
            betti.append(int(n_g - rank_down - rank_up))
    return betti


#### -
# Reading graded boundaries off a RexGraph (single source of truth)
#### -
def graded_boundaries_from_rex(rex) -> list[sp.csr_matrix]:
    """The full sparse boundary list ``[B_1, B_2, B_3, ...]`` of a RexGraph.

    This is the generalization of ``dirac_propagator._boundaries_from_rex`` and the
    single source of truth for reading a rex's graded structure:

      * ``B_1`` always, from the rex's own signed vertex-edge incidence;
      * ``B_2`` when ``nF > 0``, from the chain-consistent Hodge slice
        (``_B2_hodge_dual``), so whatever face arity the complex carries is kept;
        an explicitly empty B2 is retained when higher grades exist;
      * ``B_3, B_4, ...`` when the rex additionally stores higher boundaries in the
        optional ``_graded_duals`` attribute (populated by ``RexGraph.from_cells``).

    Every returned matrix is scipy CSR; nothing is densified.
    """
    from rexgraph.core._sparse import to_scipy_csr

    B1 = _rex_b1_csr(rex)
    boundaries: list[sp.csr_matrix] = [B1]

    duals = getattr(rex, "_graded_duals", None)
    if int(getattr(rex, "nF", 0)) > 0 and getattr(rex, "_B2_hodge_dual", None) is not None:
        boundaries.append(to_scipy_csr(rex._B2_hodge_dual).tocsr())
    elif duals and int(getattr(rex, "nF", 0)) == 0:
        # A higher tower with an empty grade two still carries B2 as an empty operator.
        # Omitting it relabels stored B3 as B2 and shifts every subsequent grade down.
        boundaries.append(sp.csr_matrix((int(rex.nE), int(duals[0].shape[0]))))

    if duals:
        for Bd in duals:
            boundaries.append(sp.csr_matrix(Bd))
    return boundaries


def _rex_b1_csr(rex) -> sp.csr_matrix:
    """B_1 (nV x nE, signed) of a rex as scipy CSR, via the rex's own DualCSR."""
    from rexgraph.core._sparse import to_scipy_csr
    return to_scipy_csr(rex._B1_dual).tocsr()


#### -
# Constructor helpers: genuine grade-3 complexes (d^2 = 0)
#### -
def _order_face_ccw(points: np.ndarray, face_idx: Sequence[int],
                    center: np.ndarray) -> list[int]:
    """Order a convex, planar face's vertices CCW as seen from OUTSIDE the solid.

    The outward normal is the direction from the solid's centroid to the face
    centroid; sorting the (coplanar, convex) face vertices by their polar angle in
    the plane orthogonal to that normal yields the boundary loop with a globally
    consistent (outward) orientation, which is exactly what makes the closed
    surface orientable, hence ``B_2 @ 1 = 0`` and ``B_2 B_3 = 0``.
    """
    fi = list(face_idx)
    pts = points[fi]
    fc = pts.mean(axis=0)
    normal = fc - center
    nrm = np.linalg.norm(normal)
    normal = np.array([0.0, 0.0, 1.0]) if nrm < 1e-12 else normal / nrm
    # An in-plane basis (e1, e2) with e2 = normal x e1, so angle increases CCW
    # about the outward normal.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, normal) * normal
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    ang = []
    for p in pts:
        d = p - fc
        ang.append(np.arctan2(np.dot(d, e2), np.dot(d, e1)))
    order = np.argsort(ang)
    return [fi[k] for k in order]


def _polyhedron_3rex(points: np.ndarray, face_vertex_sets: Sequence[Sequence[int]]):
    """Assemble a SOLID convex polyhedron as a 3-rex ``cells_by_grade``.

    ``points`` are the vertex coordinates; ``face_vertex_sets`` lists, per face, the
    (unordered) vertex indices bounding it. Faces are oriented outward, edges are
    derived from the oriented face loops, and a single volume (grade-3) cell is added
    bounded by all faces with ``+1`` signs - which closes as ``B_2 B_3 = 0`` because
    the outward orientation makes every edge cancel between its two faces.

    Returns ``cells_by_grade = [nV, edges, faces_signed, [volume]]``.
    """
    points = np.asarray(points, dtype=_f64)
    nV = points.shape[0]
    center = points.mean(axis=0)

    # Order each face CCW outward.
    ordered_faces = [_order_face_ccw(points, fs, center) for fs in face_vertex_sets]

    # Derive edges from the oriented face loops; store each with a fixed orientation
    # (first-seen direction) so face signs are relative to that stored direction.
    edge_index = {}
    edges: list[list[int]] = []
    for loop in ordered_faces:
        L = len(loop)
        for k in range(L):
            a, b = loop[k], loop[(k + 1) % L]
            key = frozenset((a, b))
            if key not in edge_index:
                edge_index[key] = len(edges)
                edges.append([a, b])

    # Signed grade-2 faces in edge space.
    faces_signed: list[list[tuple[int, float]]] = []
    for loop in ordered_faces:
        L = len(loop)
        col: list[tuple[int, float]] = []
        for k in range(L):
            a, b = loop[k], loop[(k + 1) % L]
            eidx = edge_index[frozenset((a, b))]
            stored = edges[eidx]
            sign = 1.0 if (stored[0] == a and stored[1] == b) else -1.0
            col.append((eidx, sign))
        faces_signed.append(col)

    # Single volume bounded by every face (+1); outward orientation => B2 @ 1 = 0.
    volume = [[(f, 1.0) for f in range(len(faces_signed))]]

    return [nV, edges, faces_signed, volume]


_PHI = (1.0 + 5.0 ** 0.5) / 2.0


def _icosahedron():
    """Icosahedron combinatorics from golden-ratio coordinates: returns
    ``(points[12x3], neighbors: list[set], faces: list[(a,b,c)])``.

    Edges are vertex pairs at the (minimal) squared distance; triangular faces are
    triples that are pairwise adjacent. Coordinate-driven, so exact and orientation-
    agnostic: the truncation and orientation are handled downstream.
    """
    p = _PHI
    verts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts.append((0.0, s1 * 1.0, s2 * p))
            verts.append((s1 * 1.0, s2 * p, 0.0))
            verts.append((s1 * p, 0.0, s2 * 1.0))
    P = np.array(verts, dtype=_f64)
    # Deduplicate should not be needed (12 distinct), but guard against ordering.
    nV = P.shape[0]
    # Pairwise squared distances; edge length^2 == 4 for a unit icosahedron here.
    d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
    off = d2 + np.eye(nV) * 1e9
    emin = off.min()
    adj = np.abs(d2 - emin) < 1e-6
    neighbors = [set(np.nonzero(adj[i])[0].tolist()) for i in range(nV)]
    # Triangular faces: mutually adjacent triples.
    faces = []
    for a in range(nV):
        for b in neighbors[a]:
            if b <= a:
                continue
            for c in neighbors[a] & neighbors[b]:
                if c <= b:
                    continue
                faces.append((a, b, c))
    return P, neighbors, faces


def truncated_icosahedron_3rex():
    """The SOLID truncated icosahedron (soccer ball) as a 3-rex ``cells_by_grade``.

    60 vertices, 90 edges, 32 faces (12 pentagons + 20 hexagons = mixed grade-2
    arity), 1 volume. Built programmatically by truncating the icosahedron: each
    icosahedron vertex ``v`` with an incident edge to neighbor ``n`` becomes a
    "corner" point ``v + (n - v)/3``; the 5 corners around ``v`` form a pentagon and
    the 6 corners of each icosahedron triangle form a hexagon. Faces are oriented
    outward so the shell is orientable and the single enclosed volume closes with
    ``B_2 B_3 = 0``.

    This is exactly "a topological 2-sphere with its boundary encoded as a
    5-6-gon-3-rex", promoted to a solid by the enclosing 3-cell.
    """
    P, neighbors, faces = _icosahedron()

    # Corner points, indexed by the ordered pair (vertex, neighbor).
    corner_index = {}
    corner_pts: list[np.ndarray] = []
    for v in range(P.shape[0]):
        for n in neighbors[v]:
            corner_index[(v, n)] = len(corner_pts)
            corner_pts.append(P[v] + (P[n] - P[v]) / 3.0)
    pts = np.array(corner_pts, dtype=_f64)

    face_vertex_sets: list[list[int]] = []
    # Pentagons: the 5 corners around each icosahedron vertex.
    for v in range(P.shape[0]):
        face_vertex_sets.append([corner_index[(v, n)] for n in neighbors[v]])
    # Hexagons: the 6 corners of each icosahedron triangle {a,b,c}.
    for (a, b, c) in faces:
        face_vertex_sets.append([
            corner_index[(a, b)], corner_index[(b, a)],
            corner_index[(b, c)], corner_index[(c, b)],
            corner_index[(c, a)], corner_index[(a, c)],
        ])

    return _polyhedron_3rex(pts, face_vertex_sets)


def solid_octahedron_3rex():
    """The SOLID octahedron as a 3-rex ``cells_by_grade``: 6 vertices, 12 edges,
    8 triangular faces (arity-3), 1 volume. A simple, fully triangulated grade-3
    complex with ``d^2 = 0``."""
    pts = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ], dtype=_f64)
    # 8 faces, one per (x-sign, y-sign, z-sign) octant.
    faces = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ]
    return _polyhedron_3rex(pts, faces)


def square_pyramid_3rex():
    """A SOLID square pyramid as a small MIXED-ARITY 3-rex ``cells_by_grade``:
    5 vertices, 8 edges, 5 faces (4 triangles of arity 3 + 1 square base of arity 4),
    1 volume. Demonstrates mixed grade-2 arity in a genuine grade-3 complex."""
    pts = np.array([
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],   # square base
        [0.0, 0.0, 1.5],                        # apex
    ], dtype=_f64)
    faces = [
        [0, 1, 2, 3],       # square base (arity 4)
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],   # 4 triangles (arity 3)
    ]
    return _polyhedron_3rex(pts, faces)
