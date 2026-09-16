"""Sparse linear operators between graded Rex cell spaces.

Laplacians retain their boundary factorization and apply it directly.  The
corresponding sparse product is materialized only when a caller asks for it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any

import numpy as np

from rexgraph.compute import sparse_mm
from rexgraph.native_sparse import NativeSparse, empty_native, native_boundaries

__all__ = [
    "RexOperator",
    "MetricAdjointOperator",
    "metric_adjoint",
    "boundary_operator",
    "coboundary_operator",
    "down_laplacian",
    "hodge_operator",
    "up_laplacian",
]


@dataclass(frozen=True)
class RexOperator:
    """A linear map whose domain and codomain are named Rex grades."""

    name: str
    shape: tuple[int, int]
    domain_grade: int
    codomain_grade: int
    matvec: Callable[[Any], Any]
    matrix: Any = None
    matrix_factory: Callable[[], Any] | None = None
    torch_factory: Callable[[Any, Any], Callable[[Any], Any]] | None = None
    source: Any = None
    symmetric: bool = False
    psd: bool = False
    arithmetic: str = "float"
    construction: str = "external-operator"
    variance: str = "cochain"
    transpose_matvec: Callable[[Any], Any] | None = None
    exact_matvec: Callable[[Any], Any] | None = None
    exact_transpose_matvec: Callable[[Any], Any] | None = None
    parameters: tuple[tuple[str, object], ...] = ()
    # Explicit construction time certificate, never inferred from a display name.
    exact_rank_factory: Callable[[], tuple[int, str]] | None = None
    # Canonical sparse boundary coefficients captured with the exact action.
    # None means no such certificate; an empty tuple certifies the zero map.
    boundary_entries: tuple | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(x) for x in self.shape)
        domain_grade = int(self.domain_grade)
        codomain_grade = int(self.codomain_grade)
        if len(shape) != 2 or min(shape) < 0:
            raise ValueError("shape must contain two nonnegative axes")
        if domain_grade < 0 or codomain_grade < 0:
            raise ValueError("operator grades must be >= 0")
        if not callable(self.matvec):
            raise TypeError("matvec must be callable")
        if self.variance not in {"chain", "cochain"}:
            raise ValueError("operator variance must be chain or cochain")
        if self.symmetric and (shape[0] != shape[1] or domain_grade != codomain_grade):
            raise ValueError("a symmetric operator must act within one square grade")
        if self.psd and not self.symmetric:
            raise ValueError("a positive semidefinite operator must be symmetric")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "domain_grade", domain_grade)
        object.__setattr__(self, "codomain_grade", codomain_grade)

    def apply(self, values, *, exact=False):
        """Apply the operator to a vector or a block whose first axis is its domain."""
        shape = getattr(values, "shape", None)
        if not isinstance(exact, (bool, np.bool_)):
            raise TypeError("exact must be a boolean")
        if shape is None or len(shape) not in (1, 2):
            raise ValueError("operator input must be a vector or block with a cell axis")
        if int(shape[0]) != self.shape[1]:
            raise ValueError(
                f"{self.name} expects first axis {self.shape[1]}, got {shape[0]}"
            )
        if exact:
            if self.exact_matvec is None:
                raise TypeError(f"{self.name} has no certified exact action")
            return self.exact_matvec(_exact_array(values))
        return self.matvec(values)

    @property
    def has_transpose(self):
        return self.symmetric or self.transpose_matvec is not None or self.matrix is not None

    def transpose_apply(self, values, *, exact=False):
        """Hermitian transpose action; never materialize a factored Gram matrix."""
        if not isinstance(exact, (bool, np.bool_)):
            raise TypeError("exact must be a boolean")
        shape = getattr(values, "shape", ())
        if len(shape) not in (1, 2) or shape[0] != self.shape[0]:
            raise ValueError("transpose input must match the codomain vector/block axis")
        if exact:
            if self.exact_transpose_matvec is None:
                raise TypeError(f"{self.name} has no certified exact transpose action")
            return self.exact_transpose_matvec(_exact_array(values))
        if self.transpose_matvec is not None:
            return self.transpose_matvec(values)
        if self.symmetric:
            return self.apply(values)
        if self.matrix is not None:
            if isinstance(self.matrix, NativeSparse):
                return self.matrix.transpose_apply(values)
            import scipy.sparse as sp
            return sp.csr_matrix(self.matrix).conjugate().T @ values
        raise TypeError(f"{self.name} has no declared transpose action")

    def __call__(self, values):
        return self.apply(values)

    def as_scipy(self):
        """Materialize the operator as SciPy CSR."""
        import scipy.sparse as sp
        if self.matrix is not None:
            return self.matrix.as_scipy() if isinstance(self.matrix, NativeSparse) else sp.csr_matrix(self.matrix)
        if self.matrix_factory is None:
            raise TypeError(f"{self.name} has no sparse matrix representation")
        matrix = self.matrix_factory()
        return matrix.as_scipy() if isinstance(matrix, NativeSparse) else sp.csr_matrix(matrix)

    def as_native(self):
        """Materialize through the declared matrix factory into core storage."""
        from rexgraph.native_sparse import as_native
        if self.matrix is not None:
            return as_native(self.matrix)
        if self.matrix_factory is None:
            raise TypeError(f"{self.name} has no sparse matrix representation")
        return as_native(self.matrix_factory())

    def torch_matvec(self, *, dtype=None, device=None):
        """Return a torch action without requiring a dense matrix."""
        if self.torch_factory is not None:
            return self.torch_factory(dtype, device)
        tensor = self.as_torch(dtype=dtype, device=device)
        return lambda values: _torch_apply(tensor, values)

    def as_torch(self, *, dtype=None, device=None):
        """Materialize the operator as a coalesced torch sparse tensor."""
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional torch
            raise ImportError("as_torch requires PyTorch") from exc
        matrix = self.as_scipy().tocoo()
        index = torch.as_tensor(
            np.vstack([matrix.row, matrix.col]), dtype=torch.long, device=device
        )
        if dtype is None:
            dtype = torch.get_default_dtype()
        values = torch.as_tensor(matrix.data, dtype=dtype, device=device)
        from rexgraph.compute import sparse_coo_tensor

        return sparse_coo_tensor(
            index, values, matrix.shape, device=device
        ).coalesce()


def _boundaries(rex) -> list[NativeSparse]:
    return native_boundaries(rex)


def _torch_sparse(matrix, dtype, device):
    return matrix.torch_sparse(dtype=dtype, device=device)


def _torch_apply(matrix, values):
    one = values.dim() == 1
    block = values.unsqueeze(1) if one else values
    out = sparse_mm(matrix, block)
    return out[:, 0] if one else out


def _matrix_torch_factory(matrix, *, transpose=False):
    def factory(dtype, device):
        sparse = matrix.torch_sparse(dtype=dtype, device=device, transpose=transpose)
        return lambda values: _torch_apply(sparse, values)

    return factory


def _grade_sizes(boundaries: list[NativeSparse]) -> list[int]:
    return [boundaries[0].shape[0]] + [matrix.shape[1] for matrix in boundaries]


def _exact_array(values):
    from rexgraph.graded_metric import _fraction
    array = np.asarray(values)
    return np.asarray([_fraction(v) for v in array.flat], dtype=object).reshape(array.shape)


def _numeric_array(values, *, operation):
    """Validate real/complex vector coefficients before a numeric factored action."""
    values = np.asarray(values)
    if values.dtype.kind not in "iufcO" or (values.dtype.kind == "O" and any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, complex, np.complexfloating, Fraction))
        for v in values.flat
    )):
        raise TypeError(f"{operation} requires numeric coefficients, not booleans or strings")
    complex_values = np.iscomplexobj(values) or (values.dtype.kind == "O" and any(
        isinstance(v, (complex, np.complexfloating)) for v in values.flat))
    try:
        values = np.asarray(values, dtype=complex if complex_values else float)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(f"{operation} input is outside numerical range") from exc
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"{operation} input is outside numerical range")
    return values


def _exact_incidence_actions(rex, grade, matrix):
    """Capture canonical rational C1 shares or certified integer higher maps.

    The exact and numerical actions describe the same construction time tower.
    No rational reconstruction from the float backed primary boundary is used.
    """
    from rexgraph.graded_boundary import _rank_integer_columns
    from rexgraph.native_rank import boundary_columns, clear_column_denominators
    try:
        shape, columns = boundary_columns(rex, grade)
    except ValueError:
        return None, None, None, None
    if shape != matrix.shape:
        raise ValueError("exact boundary carrier differs from numerical operator axes")
    entries = tuple((i, j, Fraction(c)) for j, col in enumerate(columns) for i, c in col.items())

    def action(values, transpose=False):
        shape = (matrix.shape[1 if transpose else 0],) + values.shape[1:]
        result = np.full(shape, Fraction(0), dtype=object)
        for i, j, c in entries:
            row, col = (j, i) if transpose else (i, j)
            result[row] += c * values[col]
        return result
    def rank():
        return _rank_integer_columns(shape, clear_column_denominators(columns))
    return action, lambda values: action(values, True), rank, entries


def boundary_operator(rex, grade: int) -> RexOperator:
    """Return ``B_grade`` as a sparse graded operator."""
    grade = int(grade)
    boundaries = _boundaries(rex)
    if grade < 1 or grade > len(boundaries):
        raise ValueError(f"boundary grade {grade} is not present")
    matrix = boundaries[grade - 1]
    exact_action, exact_transpose, exact_rank, entries = _exact_incidence_actions(rex, grade, matrix)
    return RexOperator(
        f"B{grade}",
        matrix.shape,
        grade,
        grade - 1,
        matrix.apply,
        matrix_factory=lambda: matrix,
        transpose_matvec=matrix.transpose_apply,
        torch_factory=_matrix_torch_factory(matrix),
        source=rex,
        construction="boundary", variance="chain",
        exact_matvec=exact_action, exact_transpose_matvec=exact_transpose,
        exact_rank_factory=exact_rank,
        boundary_entries=entries,
    )


def coboundary_operator(rex, grade: int) -> RexOperator:
    """Return ``B_(grade+1)^T`` from one grade to the next.

    At the top carried grade the next cochain space is empty, so the
    coboundary is the rectangular zero map ``C^d -> C^(d+1)``.  It is a
    genuine typed sector of the boundary tower, rather than an attempt to
    manufacture another nonempty cell grade.
    """
    grade = int(grade)
    boundaries = _boundaries(rex)
    sizes = _grade_sizes(boundaries)
    if grade < 0 or grade >= len(sizes):
        raise ValueError(f"grade {grade} is not present")
    if grade == len(boundaries):
        n_cells = sizes[grade]
        matrix = empty_native((0, n_cells))
        return RexOperator(
            f"B{grade + 1}T",
            matrix.shape,
            grade,
            grade + 1,
            matrix.apply,
            matrix_factory=lambda: matrix,
            transpose_matvec=matrix.transpose_apply,
            torch_factory=_matrix_torch_factory(matrix),
            source=rex,
            arithmetic="structural",
            construction="coboundary",
            exact_matvec=lambda v: np.full((0,) + v.shape[1:], Fraction(0), dtype=object),
            exact_transpose_matvec=lambda v: np.full((n_cells,) + v.shape[1:], Fraction(0), dtype=object),
            exact_rank_factory=lambda: (0, "integer-zero"),
        )
    boundary = boundary_operator(rex, grade + 1)
    matrix = boundaries[grade]
    return RexOperator(
        f"B{grade + 1}T",
        boundary.shape[::-1],
        grade,
        grade + 1,
        boundary.transpose_apply,
        transpose_matvec=boundary.apply,
        matrix_factory=lambda: boundary.as_native().T,
        torch_factory=_matrix_torch_factory(matrix, transpose=True),
        source=rex,
        arithmetic=boundary.arithmetic,
        construction="coboundary",
        exact_matvec=boundary.exact_transpose_matvec,
        exact_transpose_matvec=boundary.exact_matvec,
        exact_rank_factory=boundary.exact_rank_factory,
    )


def _zero_operator(rex, grade: int, n_cells: int, name: str) -> RexOperator:
    matrix = empty_native((n_cells, n_cells))
    return RexOperator(
        name,
        matrix.shape,
        grade,
        grade,
        matrix.apply,
        matrix_factory=lambda: matrix,
        torch_factory=_matrix_torch_factory(matrix),
        source=rex,
        symmetric=True,
        psd=True,
    )


def down_laplacian(rex, grade: int) -> RexOperator:
    """Return ``B_grade^T B_grade``, with the exact zero operator at grade 0."""
    grade = int(grade)
    boundaries = _boundaries(rex)
    sizes = _grade_sizes(boundaries)
    if grade < 0 or grade >= len(sizes):
        raise ValueError(f"grade {grade} is not present")
    n_cells = sizes[grade]
    if grade == 0:
        return _zero_operator(rex, grade, n_cells, "L0_down")
    boundary = boundaries[grade - 1]

    def torch_factory(dtype, device):
        matrix = _torch_sparse(boundary, dtype, device)
        transpose = matrix.transpose(0, 1).coalesce()
        return lambda values: _torch_apply(
            transpose, _torch_apply(matrix, values)
        )

    return RexOperator(
        f"L{grade}_down",
        (n_cells, n_cells),
        grade,
        grade,
        lambda values: boundary.transpose_apply(boundary.apply(values)),
        matrix_factory=lambda: _gram_export(boundary, down=True),
        torch_factory=torch_factory,
        source=rex,
        symmetric=True,
        psd=True,
    )


def up_laplacian(rex, grade: int) -> RexOperator:
    """Return ``B_(grade+1) B_(grade+1)^T``, or exact zero at top grade."""
    grade = int(grade)
    boundaries = _boundaries(rex)
    sizes = _grade_sizes(boundaries)
    if grade < 0 or grade >= len(sizes):
        raise ValueError(f"grade {grade} is not present")
    n_cells = sizes[grade]
    if grade >= len(boundaries):
        return _zero_operator(rex, grade, n_cells, f"L{grade}_up")
    boundary = boundaries[grade]

    def torch_factory(dtype, device):
        matrix = _torch_sparse(boundary, dtype, device)
        transpose = matrix.transpose(0, 1).coalesce()
        return lambda values: _torch_apply(
            matrix, _torch_apply(transpose, values)
        )

    return RexOperator(
        f"L{grade}_up",
        (n_cells, n_cells),
        grade,
        grade,
        lambda values: boundary.apply(boundary.transpose_apply(values)),
        matrix_factory=lambda: _gram_export(boundary, down=False),
        torch_factory=torch_factory,
        source=rex,
        symmetric=True,
        psd=True,
    )


def _gram_export(boundary, *, down):
    return boundary.T.product(boundary) if down else boundary.product(boundary.T)


def hodge_operator(rex, grade: int, *, alpha=1) -> RexOperator:
    """Return ``L_down + alpha L_up`` for a finite, nonnegative ``alpha``."""
    grade = int(grade)
    coefficient = float(alpha)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("alpha must be finite and >= 0")
    down = down_laplacian(rex, grade)
    up = up_laplacian(rex, grade)

    def torch_factory(dtype, device):
        down_action = down.torch_matvec(dtype=dtype, device=device)
        up_action = up.torch_matvec(dtype=dtype, device=device)
        return lambda values: down_action(values) + coefficient * up_action(values)

    return RexOperator(
        f"L{grade}",
        down.shape,
        grade,
        grade,
        lambda values: down.apply(values) + coefficient * up.apply(values),
        matrix_factory=lambda: down.as_native().add(up.as_native(), coefficient),
        torch_factory=torch_factory,
        source=rex,
        symmetric=True,
        psd=True,
        construction="hodge_operator",
        parameters=(("alpha", coefficient),),
    )


@dataclass(frozen=True)
class MetricAdjointOperator(RexOperator):
    """M_domain^-1 A* M_codomain, in the original cell coordinates.

    Metrics name the ORIGINAL operator's endpoints, not the reversed result.
    Variance is inherited: an adjoint of B acts on Chains, not dual Cochains.
    Like other Rex handles, keep the source unchanged while reusing the handle.
    """

    primal: RexOperator | None = None
    domain_metric: Any = None
    codomain_metric: Any = None


def metric_adjoint(operator, domain_metric=None, codomain_metric=None):
    """Factored diagonal metric adjoint; missing endpoint metrics mean identity.

    No matrix inversion, square root, eigensolve, dense or sparse Gram is built.
    Exact action requires rational metrics and a certified Q transpose action.
    A general metric adjoint is NOT asserted Euclidean symmetric or PSD.
    """
    from rexgraph.graded_metric import DiagonalMetric, diagonal_metric
    if not isinstance(operator, RexOperator) or operator.source is None:
        raise TypeError("adjoint requires a source-bound RexOperator")
    if not operator.has_transpose:
        raise TypeError("adjoint requires a declared transpose action; materialization is not a fallback")
    metrics = []
    for supplied, grade, n in ((domain_metric, operator.domain_grade, operator.shape[1]),
                               (codomain_metric, operator.codomain_grade, operator.shape[0])):
        metric = diagonal_metric(operator.source, grade) if supplied is None else supplied
        if (not isinstance(metric, DiagonalMetric) or metric.source is not operator.source
                or metric.grade != grade or len(metric.weights) != n):
            raise ValueError("adjoint metric must match its operator source, grade and population")
        if metric.cell_keys is not None:
            raise ValueError("adjoint requires the canonical ordered basis")
        metrics.append(metric)
    md, mc = metrics
    cached_weights = {}

    def weights(metric, inverse, exact):
        key = (id(metric), inverse, exact)
        if key in cached_weights:
            return cached_weights[key]
        if exact:
            result = np.asarray([1/w if inverse else w for w in metric.weights], dtype=object)
        else:
            try:
                result = np.asarray([1/w if inverse else w for w in metric.weights], dtype=float)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError("adjoint metric is outside numerical range") from exc
            if np.any(result == 0) or not np.all(np.isfinite(result)):
                raise FloatingPointError("adjoint metric is outside numerical range")
        result.setflags(write=False)
        cached_weights[key] = result
        return result

    def action(values, *, transpose=False, exact=False):
        # (Md^-1 A* Mc)* = Mc A Md^-1. The adjoint of the adjoint is
        # instead constructed with the SWAPPED endpoint metrics.
        before = weights(md if transpose else mc, transpose, exact)
        after = weights(mc if transpose else md, not transpose, exact)
        if not exact:
            values = _numeric_array(values, operation="adjoint")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled = (before[:, None] if values.ndim == 2 else before) * values
            out = (operator.apply(scaled, exact=exact) if transpose
                   else operator.transpose_apply(scaled, exact=exact))
            if np.shape(out) != (len(after),) + values.shape[1:]:
                raise ValueError("adjoint action returned the wrong codomain vector/block shape")
            if exact:
                out = _exact_array(out)
            result = (after[:, None] if values.ndim == 2 else after) * out
        if not exact and not np.all(np.isfinite(result)):
            raise FloatingPointError("adjoint result is outside numerical range")
        return result

    exact = md.exact and mc.exact and operator.exact_transpose_matvec is not None
    exact_transpose = md.exact and mc.exact and operator.exact_matvec is not None
    return MetricAdjointOperator(
        f"ADJOINT({operator.name})", operator.shape[::-1], operator.codomain_grade,
        operator.domain_grade, action, source=operator.source, construction="metric-adjoint",
        variance=operator.variance, transpose_matvec=lambda v: action(v, transpose=True),
        exact_matvec=(lambda v: action(v, exact=True)) if exact else None,
        exact_transpose_matvec=(lambda v: action(v, transpose=True, exact=True)) if exact_transpose else None,
        primal=operator, domain_metric=md, codomain_metric=mc,
    )
