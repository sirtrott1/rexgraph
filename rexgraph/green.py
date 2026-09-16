"""Matrix free Green and resolvent actions on Rex operators."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from rexgraph.linear_operator import RexOperator, _numeric_array, hodge_operator
from rexgraph.weighted_hodge import WeightedHodgeOperator

__all__ = ["GreenOperator", "vertex_green"]


@dataclass(frozen=True)
class GreenOperator:
    """A solve action associated with a square Rex operator."""

    operator: RexOperator
    solver: Callable[[Any], Any]
    kind: str = "green"
    observed_solver: Callable[[Any], tuple[Any, dict]] | None = None
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operator, RexOperator):
            raise TypeError("Green operator must wrap a RexOperator")
        if self.operator.shape[0] != self.operator.shape[1]:
            raise ValueError("Green operator requires a square operator")
        if not callable(self.solver):
            raise TypeError("Green solver must be callable")
        if self.observed_solver is not None and not callable(self.observed_solver):
            raise TypeError("observed Green solver must be callable")

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

    def solve_with_info(self, values):
        """Solve and report the method from this invocation, never a global last run.

        Custom actions without an observation hook remain explicitly unreported.
        No method or error bound is inferred from the returned array's dtype.
        """
        shape = getattr(values, "shape", None)
        if shape is None or len(shape) == 0 or int(shape[0]) != self.operator.shape[0]:
            raise ValueError("Green input must match the operator's cell axis")
        if self.observed_solver is None:
            return self.solver(values), {"kernel": None, "status": "unreported"}
        return self.observed_solver(values)

    @property
    def metric(self):
        """Declared solve form, or None for the existing Euclidean actions."""
        if self.kind == "metric-resolvent" and isinstance(self.operator, WeightedHodgeOperator):
            return self.operator.grade_metric
        return None

    def _metric_field(self, field):
        if self.metric is None:
            return field
        diagonal = _numeric_array(self.metric.weights, operation="Green metric")
        if np.any(diagonal <= 0):
            raise FloatingPointError("Green metric is outside numerical range")
        with np.errstate(over="ignore", invalid="ignore"):
            result = (diagonal[:, None] if field.ndim == 2 else diagonal) * field
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Green metric contraction is outside numerical range")
        return result

    def quadrance(self, source, field=None):
        """Return ``source^* M G source`` per column (M=I for Euclidean actions)."""
        field = self.solve(source) if field is None else field
        source = np.asarray(source)
        field = np.asarray(field)
        if source.shape != field.shape:
            raise ValueError("source and field must have the same shape")
        field = self._metric_field(field)
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
        field = self._metric_field(np.asarray(self.solve(sources)))
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
        """Build ``(I + alpha L)^-1`` for Euclidean or native metric PSD L.

        Weighted Hodge uses CG on M(I+alpha L), preconditioned by M^-1.
        Only diagonal scaling and the original factored action are required.
        The numerical backend is an execution choice, not the inverse definition.
        """
        from rexgraph import sparse_character

        if not isinstance(operator, RexOperator) or (operator.shape[0] != operator.shape[1]
                or operator.domain_grade != operator.codomain_grade):
            raise TypeError("resolvent requires a grade-preserving square RexOperator")
        weighted = isinstance(operator, WeightedHodgeOperator)
        metric = operator.grade_metric if weighted else None
        if weighted:
            if (operator.sector not in {"down", "up", "sum"} or not operator.metric_self_adjoint
                    or operator.metric_psd is not True or metric is None):
                raise ValueError("resolvent requires a metric positive semidefinite Hodge sector")
        elif not operator.symmetric or not operator.psd:
            raise ValueError(
                "resolvent requires a symmetric positive semidefinite operator"
            )
        if isinstance(alpha, (bool, np.bool_)):
            raise TypeError("alpha must be a real scalar, not a boolean")
        coefficient = float(alpha)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or not 0 < tol < 1:
            raise ValueError("tol must be finite and in (0, 1)")
        if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, Integral) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer")
        n_cells = operator.shape[0]

        def apply(values):
            return values + coefficient * operator.apply(values)

        def solve(values, *, report=False):
            try:
                block = _numeric_array(values, operation="Green solve")
            except FloatingPointError as exc:
                raise ValueError("Green solve requires finite, numerically representable coefficients") from exc
            if np.iscomplexobj(block):
                raise TypeError("this resolvent solver requires real coefficients")
            one = block.ndim == 1
            if one:
                block = block[:, None]
            elif block.ndim != 2:
                raise ValueError("Green solve expects a vector or two-dimensional block")
            if block.shape[0] != n_cells:
                raise ValueError("Green input must match the operator's cell axis")
            if metric is not None:
                from rexgraph.cochain import Chain
                metric._carrier(Chain(operator.domain_grade, block, source=operator.source))
            if not np.all(np.isfinite(block)):
                raise ValueError("Green solve requires finite coefficients")
            if coefficient == 0 or n_cells == 0 or block.shape[1] == 0:
                result = block[:, 0].copy() if one else block.copy()
                details = {"kernel": "identity" if coefficient == 0 else "empty-zero",
                           "status": "observed", "alpha": coefficient, "tol": tol,
                           "maxiter": int(maxiter), "relative_residuals": [0.0] * block.shape[1],
                           "residual_norm": "euclidean-original-system", "iterations": [0] * block.shape[1],
                           "metric_digest": None if metric is None else metric.coefficient_digest}
                return (result, details) if report else result
            diagonal, solver_tol = None, float(tol)
            reciprocal = np.ones(n_cells)
            if metric is not None:
                diagonal = _numeric_array(metric.weights, operation="Green metric")
                # A scalar multiple of M gives the same solution. No square root
                # coordinates: keep the primary Chain basis throughout.
                with np.errstate(over="ignore", under="ignore", divide="ignore"):
                    diagonal = diagonal / np.max(diagonal)
                    reciprocal = 1.0 / diagonal
                    solver_tol = float(tol) * float(np.min(diagonal))
                if (np.any(diagonal <= 0) or not np.all(np.isfinite(reciprocal))
                        or solver_tol == 0):
                    raise FloatingPointError("Green metric range cannot support this numerical tolerance")

            def system_apply(v):
                result = apply(v)
                return result if diagonal is None else (diagonal[:, None] if v.ndim == 2 else diagonal) * result

            out = np.empty_like(block)
            residuals, iterations = [], []
            for column in range(block.shape[1]):
                # Scale each RHS before CG, whose internal norms otherwise square
                # arbitrary user magnitudes. Check the returned physical field too.
                rhs_scale = float(np.max(np.abs(block[:, column])))
                rhs = block[:, column] / rhs_scale if rhs_scale else block[:, column]
                try:
                    solution, info = sparse_character._block_cg(
                        system_apply,
                        (rhs if diagonal is None else diagonal * rhs)[:, None],
                        reciprocal, tol=solver_tol, maxit=maxiter, return_info=True)
                except ArithmeticError as exc:
                    raise RuntimeError(f"Green solve did not converge: {exc}") from exc
                solution = solution[:, 0]
                with np.errstate(over="ignore", invalid="ignore"):
                    solution = solution * rhs_scale
                    error = apply(solution) - block[:, column]
                if not np.all(np.isfinite(solution)) or not np.all(np.isfinite(error)):
                    raise RuntimeError("Green solve returned nonfinite coefficients or residual")
                # A common scale avoids squaring huge/tiny coefficients in the
                # norm and accepting an inaccurate solve through inf/0 arithmetic.
                scale = max(np.max(np.abs(error)), np.max(np.abs(block[:, column])))
                if scale == 0:
                    relative = 0.0
                else:
                    residual = np.linalg.norm(error / scale)
                    norm = np.linalg.norm(block[:, column] / scale)
                    relative = float(residual / norm) if norm else float("inf")
                if not np.isfinite(relative) or relative > tol:
                    raise RuntimeError("Green solve did not meet the measured residual tolerance")
                residuals.append(relative)
                iterations.append(info["iterations"][0])
                out[:, column] = solution
            result = out[:, 0] if one else out
            details = {"kernel": "native-metric-block-cg" if weighted else "native-block-cg",
                       "status": "observed", "alpha": coefficient, "tol": tol,
                       "maxiter": int(maxiter), "relative_residuals": residuals,
                       "residual_norm": "euclidean-original-system", "iterations": iterations,
                       "linear_system_tol": solver_tol,
                       "metric_digest": None if metric is None else metric.coefficient_digest}
            return (result, details) if report else result

        return cls(operator, solve, kind="metric-resolvent" if weighted else "resolvent",
                   observed_solver=lambda values: solve(values, report=True),
                   parameters=(("alpha", coefficient), ("tol", tol), ("maxiter", int(maxiter))))


def _least_norm_l0_solve(boundary, block, *, tol: float, maxiter: int):
    """Apply ``(B B^T)^+`` one RHS at a time without forming ``B B^T``."""
    from rexgraph.fiedler import minimum_norm_gram_solve
    return minimum_norm_gram_solve(boundary, block, tol=tol, maxit=maxiter)


def vertex_green(rex, *, tol=1e-12, maxiter=500) -> GreenOperator:
    """Return the grade 0 Moore Penrose Green action ``(B1 B1^T)^+``.

    Pairwise complexes use the core's factored, deflated block CG path.  General
    branching boundaries use a minimum norm LSQR action because connected component
    indicators need not span their larger kernel.
    """
    from rexgraph.native_sparse import NativeSparse
    from rexgraph.fiedler import deflated_operator
    from rexgraph.sparse_character import _block_cg

    rex._ensure_clean()
    if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("Green tolerance must lie strictly between zero and one")
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError("Green maxiter must be a positive integer")
    boundary = NativeSparse(rex._B1_dual)
    laplacian = hodge_operator(rex, 0)
    try:
        apply_deflated, diagonal_inverse, kernel, n_kernel = deflated_operator(boundary, native=True)
        pairwise = True
    except ValueError:
        pairwise = False

    def solve(values, *, report=False):
        block = _numeric_array(values, operation="Green solve")
        if np.iscomplexobj(block):
            raise TypeError("pseudoinverse Green requires real coefficients")
        one = block.ndim == 1
        if one:
            block = block[:, None]
        elif block.ndim != 2:
            raise ValueError("Green solve expects a vector or two-dimensional block")
        fallback = False
        if boundary.shape[0] == 0:
            kernel_name = "empty-zero"
            out = np.zeros_like(block)
        elif pairwise:
            kernel_name = "deflated-block-cg"
            scales = np.max(np.abs(block), axis=0)
            scales = np.where(scales == 0, 1.0, scales)
            normalized = block / scales
            cg_failed = False
            try:
                out = _block_cg(
                    apply_deflated, np.ascontiguousarray(normalized), diagonal_inverse,
                    tol=tol, maxit=maxiter)
            except ArithmeticError:
                out = np.zeros_like(block)
                cg_failed = True
            if n_kernel:
                out = out - kernel.apply(kernel.transpose_apply(out))
            residual = boundary.apply(boundary.transpose_apply(out)) - normalized
            residual = residual - kernel.apply(kernel.transpose_apply(residual)) if n_kernel else residual
            scale = np.maximum(np.linalg.norm(normalized, axis=0), 1e-300)
            relative = np.linalg.norm(residual, axis=0) / scale
            if cg_failed or not np.all(np.isfinite(relative)) or np.any(relative > max(10.0 * tol, 1e-10)):
                kernel_name, fallback = "native-factor-lsqr", True
                out = _least_norm_l0_solve(
                    boundary, block, tol=tol, maxiter=maxiter
                )
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    out *= scales
                if not np.all(np.isfinite(out)):
                    raise FloatingPointError("Green solution is outside float64")
        else:
            kernel_name = "native-factor-lsqr"
            out = _least_norm_l0_solve(boundary, block, tol=tol, maxiter=maxiter)
        result = out[:, 0] if one else out
        return (result, {"kernel": kernel_name, "status": "observed", "fallback": fallback,
                         "tol": tol, "maxiter": maxiter}) if report else result

    return GreenOperator(laplacian, solve, kind="pseudoinverse",
                         observed_solver=lambda values: solve(values, report=True))
