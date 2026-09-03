"""Sparse linear operators between graded Rex cell spaces.

Laplacians retain their boundary factorization and apply it directly.  The
corresponding sparse product is materialized only when a caller asks for it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from rexgraph.graded_boundary import graded_boundaries_from_rex
from rexgraph.compute import sparse_mm

__all__ = [
    "RexOperator",
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
        if self.symmetric and (shape[0] != shape[1] or domain_grade != codomain_grade):
            raise ValueError("a symmetric operator must act within one square grade")
        if self.psd and not self.symmetric:
            raise ValueError("a positive semidefinite operator must be symmetric")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "domain_grade", domain_grade)
        object.__setattr__(self, "codomain_grade", codomain_grade)

    def apply(self, values):
        """Apply the operator to a vector or a block whose first axis is its domain."""
        shape = getattr(values, "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError("operator input must have a cell axis")
        if int(shape[0]) != self.shape[1]:
            raise ValueError(
                f"{self.name} expects first axis {self.shape[1]}, got {shape[0]}"
            )
        return self.matvec(values)

    def __call__(self, values):
        return self.apply(values)

    def as_scipy(self) -> sp.csr_matrix:
        """Materialize the operator as SciPy CSR."""
        if self.matrix is not None:
            return sp.csr_matrix(self.matrix)
        if self.matrix_factory is None:
            raise TypeError(f"{self.name} has no sparse matrix representation")
        return sp.csr_matrix(self.matrix_factory())

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


def _boundaries(rex) -> list[sp.csr_matrix]:
    rex._ensure_clean()
    return graded_boundaries_from_rex(rex)


def _torch_sparse(matrix, dtype, device):
    import torch

    coo = sp.coo_matrix(matrix)
    index = torch.as_tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
    )
    if dtype is None:
        dtype = torch.get_default_dtype()
    values = torch.as_tensor(coo.data, dtype=dtype, device=device)
    from rexgraph.compute import sparse_coo_tensor

    return sparse_coo_tensor(
        index, values, coo.shape, device=device
    ).coalesce()


def _torch_apply(matrix, values):
    one = values.dim() == 1
    block = values.unsqueeze(1) if one else values
    out = sparse_mm(matrix, block)
    return out[:, 0] if one else out


def _matrix_torch_factory(matrix):
    def factory(dtype, device):
        sparse = _torch_sparse(matrix, dtype, device)
        return lambda values: _torch_apply(sparse, values)

    return factory


def _grade_sizes(boundaries: list[sp.csr_matrix]) -> list[int]:
    return [boundaries[0].shape[0]] + [matrix.shape[1] for matrix in boundaries]


def boundary_operator(rex, grade: int) -> RexOperator:
    """Return ``B_grade`` as a sparse graded operator."""
    grade = int(grade)
    boundaries = _boundaries(rex)
    if grade < 1 or grade > len(boundaries):
        raise ValueError(f"boundary grade {grade} is not present")
    matrix = boundaries[grade - 1].tocsr()
    return RexOperator(
        f"B{grade}",
        matrix.shape,
        grade,
        grade - 1,
        lambda values, matrix=matrix: matrix @ values,
        matrix=matrix,
        torch_factory=_matrix_torch_factory(matrix),
        source=rex,
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
        matrix = sp.csr_matrix((0, n_cells), dtype=np.float64)
        return RexOperator(
            f"B{grade + 1}T",
            matrix.shape,
            grade,
            grade + 1,
            lambda values, matrix=matrix: matrix @ values,
            matrix=matrix,
            torch_factory=_matrix_torch_factory(matrix),
            source=rex,
            arithmetic="structural",
        )
    boundary = boundary_operator(rex, grade + 1)
    matrix = boundary.as_scipy().T.tocsr()
    return RexOperator(
        f"B{grade + 1}T",
        matrix.shape,
        grade,
        grade + 1,
        lambda values, matrix=matrix: matrix @ values,
        matrix=matrix,
        torch_factory=_matrix_torch_factory(matrix),
        source=rex,
        arithmetic=boundary.arithmetic,
    )


def _zero_operator(rex, grade: int, n_cells: int, name: str) -> RexOperator:
    matrix = sp.csr_matrix((n_cells, n_cells), dtype=np.float64)
    return RexOperator(
        name,
        matrix.shape,
        grade,
        grade,
        lambda values: np.zeros_like(values),
        matrix=matrix,
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
    boundary = boundaries[grade - 1].tocsr()

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
        lambda values, boundary=boundary: boundary.T @ (boundary @ values),
        matrix_factory=lambda boundary=boundary: (boundary.T @ boundary).tocsr(),
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
    boundary = boundaries[grade].tocsr()

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
        lambda values, boundary=boundary: boundary @ (boundary.T @ values),
        matrix_factory=lambda boundary=boundary: (boundary @ boundary.T).tocsr(),
        torch_factory=torch_factory,
        source=rex,
        symmetric=True,
        psd=True,
    )


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
        matrix_factory=lambda: (
            down.as_scipy() + coefficient * up.as_scipy()
        ).tocsr(),
        torch_factory=torch_factory,
        source=rex,
        symmetric=True,
        psd=True,
    )
