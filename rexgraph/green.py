"""Matrix-free Green and resolvent actions on Rex operators."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rexgraph.linear_operator import RexOperator, hodge_operator

__all__ = ["GreenOperator", "vertex_green"]


@dataclass(frozen=True)
class GreenOperator:
    """A solve action associated with a square Rex operator."""

    operator: RexOperator
    solver: Callable[[Any], Any]
    kind: str = "green"

    def __post_init__(self) -> None:
        if not isinstance(self.operator, RexOperator):
            raise TypeError("Green operator must wrap a RexOperator")
        if self.operator.shape[0] != self.operator.shape[1]:
            raise ValueError("Green operator requires a square operator")
        if not callable(self.solver):
            raise TypeError("Green solver must be callable")

    def solve(self, values):
        """Apply the Green solve to a vector or block of vectors."""
        shape = getattr(values, "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError("Green input must have a cell axis")
        if int(shape[0]) != self.operator.shape[0]:
            raise ValueError(
                f"{self.kind} expects first axis {self.operator.shape[0]}, got {shape[0]}"
            )
        return self.solver(values)

    def quadrance(self, source, field=None):
        """Return ``source^* G source`` for a vector or each column of a block."""
        field = self.solve(source) if field is None else field
        source = np.asarray(source)
        field = np.asarray(field)
        if source.shape != field.shape:
            raise ValueError("source and field must have the same shape")
        if source.ndim == 1:
            return np.vdot(source, field).real.item()
        if source.ndim != 2:
            raise ValueError("quadrance expects a vector or a two-dimensional block")
        return np.einsum("ij,ij->j", source.conj(), field).real

    def gram(self, sources):
        """Return the Hermitian Green Gram of source columns."""
        sources = np.asarray(sources)
        if sources.ndim == 1:
            sources = sources[:, None]
        if sources.ndim != 2:
            raise ValueError("gram expects a vector or a two-dimensional block")
        field = np.asarray(self.solve(sources))
        gram = sources.conj().T @ field
        return 0.5 * (gram + gram.conj().T)

    def spread(self, sources):
        """Return the normalized pairwise spread induced by the Green Gram."""
        gram = self.gram(sources)
        quadrances = np.real(np.diag(gram)).copy()
        safe = np.where(np.abs(quadrances) > 1e-300, quadrances, 1.0)
        spread = 1.0 - np.abs(gram) ** 2 / np.outer(safe, safe)
        zero = np.abs(quadrances) <= 1e-300
        spread[zero, :] = 0.0
        spread[:, zero] = 0.0
        np.fill_diagonal(spread, 0.0)
        return np.clip(0.5 * (spread + spread.T), 0.0, 1.0)

    @classmethod
    def resolvent(
        cls, operator: RexOperator, alpha=1.0, *, tol=1e-10, maxiter=1000
    ):
        """Build ``(I + alpha L)^-1`` for a symmetric PSD operator."""
        import scipy.sparse.linalg as spla

        if not operator.symmetric or not operator.psd:
            raise ValueError(
                "resolvent requires a symmetric positive semidefinite operator"
            )
        coefficient = float(alpha)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        n_cells = operator.shape[0]

        def apply(values):
            return values + coefficient * operator.apply(values)

        shifted = spla.LinearOperator(
            (n_cells, n_cells),
            matvec=apply,
            rmatvec=apply,
            matmat=apply,
            dtype=np.float64,
        )

        def solve(values):
            block = np.asarray(values, dtype=np.float64)
            one = block.ndim == 1
            if one:
                block = block[:, None]
            elif block.ndim != 2:
                raise ValueError("Green solve expects a vector or two-dimensional block")
            out = np.empty_like(block)
            for column in range(block.shape[1]):
                solution, info = spla.cg(
                    shifted,
                    block[:, column],
                    rtol=tol,
                    atol=0.0,
                    maxiter=maxiter,
                )
                if info != 0:
                    raise RuntimeError(f"Green solve did not converge, info={info}")
                out[:, column] = solution
            return out[:, 0] if one else out

        return cls(operator, solve, kind="resolvent")


def _least_norm_l0_solve(boundary, block, *, tol: float, maxiter: int):
    """Apply ``(B B^T)^+`` one RHS at a time without forming ``B B^T``."""
    import scipy.sparse.linalg as spla

    transpose = boundary.T.tocsr()
    n_vertices = boundary.shape[0]

    def apply(values):
        return boundary @ (transpose @ values)

    laplacian = spla.LinearOperator(
        (n_vertices, n_vertices),
        matvec=apply,
        rmatvec=apply,
        dtype=np.float64,
    )
    out = np.empty_like(block)
    for column in range(block.shape[1]):
        solution = spla.lsmr(
            laplacian,
            block[:, column],
            atol=tol,
            btol=tol,
            conlim=0.0,
            maxiter=maxiter,
        )
        if solution[1] not in (0, 1, 2):
            raise RuntimeError(
                f"pseudoinverse Green solve did not converge, istop={solution[1]}"
            )
        out[:, column] = solution[0]
    return out


def vertex_green(rex, *, tol=1e-12, maxiter=500) -> GreenOperator:
    """Return the grade-0 Moore-Penrose Green action ``(B1 B1^T)^+``.

    Pairwise complexes use the core's factored, deflated block-CG path.  General
    branching boundaries use a minimum-norm LSMR action because connected-component
    indicators need not span their larger kernel.
    """
    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.fiedler import deflated_operator
    from rexgraph.graded_boundary import _pairwise_rank
    from rexgraph.sparse_character import _block_cg

    rex._ensure_clean()
    boundary = to_scipy_csr(rex._B1_dual).tocsr()
    laplacian = hodge_operator(rex, 0)
    pairwise = _pairwise_rank(boundary) is not None
    if pairwise:
        apply_deflated, diagonal_inverse, kernel, n_kernel = deflated_operator(boundary)

    def solve(values):
        block = np.asarray(values, dtype=np.float64)
        one = block.ndim == 1
        if one:
            block = block[:, None]
        elif block.ndim != 2:
            raise ValueError("Green solve expects a vector or two-dimensional block")
        if boundary.shape[0] == 0:
            out = np.zeros_like(block)
        elif pairwise:
            out = _block_cg(
                apply_deflated,
                np.ascontiguousarray(block),
                diagonal_inverse,
                tol=tol,
                maxit=maxiter,
            )
            if n_kernel:
                out = out - kernel @ (kernel.T @ out)
            residual = boundary @ (boundary.T @ out) - block
            residual = residual - kernel @ (kernel.T @ residual) if n_kernel else residual
            scale = np.maximum(np.linalg.norm(block, axis=0), 1e-300)
            if np.any(np.linalg.norm(residual, axis=0) / scale > max(10.0 * tol, 1e-10)):
                out = _least_norm_l0_solve(
                    boundary, block, tol=tol, maxiter=maxiter
                )
        else:
            out = _least_norm_l0_solve(boundary, block, tol=tol, maxiter=maxiter)
        return out[:, 0] if one else out

    return GreenOperator(laplacian, solve, kind="pseudoinverse")
