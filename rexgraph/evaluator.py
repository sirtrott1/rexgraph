"""Evaluator policy for sparse production paths and explicit dense oracles.

RexGraph keeps dense linear algebra as an independent numerical oracle.  That is useful,
but a sparse solver failure must never silently turn into an unbounded dense allocation.
This module is the one policy boundary between those two worlds.
"""
from __future__ import annotations

SMALL_DENSE_EIGEN_MAX = 64
_FALLBACK_EIGEN_DENSE_LIMIT = 2000


class DenseEvaluationRefused(RuntimeError):
    """A production path refused an implicit dense evaluation."""


def eigen_dense_limit() -> int:
    """Configured upper dimension for deliberate full dense eigensolves.

    The compiled configuration is authoritative when available.  The fallback exists so
    source inspection/documentation imports remain deterministic before native modules are
    built; numerical entry points themselves still require their native prerequisites.
    """
    try:
        from rexgraph.core import _common
        return int(_common.get_algorithm_config().get("eigen_dense_limit", _FALLBACK_EIGEN_DENSE_LIMIT))
    except Exception:  # pragma: no cover: source tree before native build
        return _FALLBACK_EIGEN_DENSE_LIMIT


def check_dense_allocation(operation: str, nrows: int, ncols: int) -> None:
    """Apply RexGraph's configured dense memory ceiling to one float64 matrix."""
    try:
        from rexgraph.core import _common
        _common.check_dense_allocation(str(operation), int(nrows), int(ncols))
    except ImportError:  # pragma: no cover: source tree before native build
        return


def require_small_dense_eigen(operation: str, n: int, *, max_dimension: int = SMALL_DENSE_EIGEN_MAX) -> None:
    """Permit an implicit dense eigensolve only for an explicitly tiny matrix.

    This is for capability fallbacks such as ARPACK refusing a 2x2 or 8x8 problem.  The
    bound is deliberately independent of available RAM: a solver exception on a large
    sparse operator must not change the algorithmic complexity of the call.
    """
    n = int(n)
    limit = min(int(max_dimension), eigen_dense_limit())
    if n > limit:
        raise DenseEvaluationRefused(
            f"{operation} sparse evaluator failed at dimension {n}; refusing implicit "
            f"dense eigensolve above the small-matrix limit {limit}. Use an explicit "
            "dense/oracle evaluator if that cost is intended."
        )
    check_dense_allocation(operation, n, n)


def require_bounded_full_eigen(operation: str, n: int) -> None:
    """Guard a deliberate full eigenbasis operation by the configured dense limit."""
    n = int(n)
    limit = eigen_dense_limit()
    if n > limit:
        raise DenseEvaluationRefused(
            f"{operation} requires a full eigenbasis of dimension {n}, above the "
            f"configured dense eigen limit {limit}. Precompute/provide a basis or use a "
            "sparse observable instead."
        )
    check_dense_allocation(operation, n, n)


def require_explicit_dense_oracle(operation: str, n: int) -> None:
    """Memory check an explicitly requested dense oracle without an implicit size gate."""
    check_dense_allocation(operation, int(n), int(n))


__all__ = [
    "DenseEvaluationRefused",
    "SMALL_DENSE_EIGEN_MAX",
    "check_dense_allocation",
    "eigen_dense_limit",
    "require_bounded_full_eigen",
    "require_explicit_dense_oracle",
    "require_small_dense_eigen",
]
