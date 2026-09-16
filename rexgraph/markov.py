"""Native C0 participation action with compiled PageRank iteration.

The bound Rex remains the source. No branching relation is replaced by a pair,
and no channel or higher boundary is identified with a stochastic transition.
"""
from fractions import Fraction as Q

import numpy as np

from rexgraph.core import _standard, _sparse
from rexgraph.io.catalog import object_digest
from rexgraph.linear_operator import RexOperator, _exact_array, _numeric_array


def validate_markov_source(source, grade=0):
    if isinstance(grade, (bool, np.bool_)) or grade != 0 or not isinstance(grade, (int, np.integer)):
        raise ValueError("Markov action requires grade zero")
    weights = source.edge_metric_exact
    if weights is not None and any(w < 0 for w in weights):
        raise ValueError("Markov relation weights must be nonnegative")
    return weights


def _participation_entries(source, weights):
    """Read the exact unsigned weighted primary tensor without changing its axes."""
    from rexgraph.native_rank import primary_columns
    return tuple((v, e, abs(c)*(Q(1) if weights is None else weights[e]))
                 for e, col in enumerate(primary_columns(source)) for v, c in col.items()
                 if weights is None or weights[e])


class MarkovView(RexOperator):
    """Native C0 mass action through the primary participation tensor."""

    def __init__(self, source, grade=0):
        weights = validate_markov_source(source, grade)
        from rexgraph.native_sparse import native_coo
        n, m = int(source.nV), int(source.nE)
        digest = object_digest(source)
        entries = _participation_entries(source, weights)
        dv, de = [Q(0)]*n, [Q(0)]*m
        for v, e, u in entries:
            dv[v] += u
            de[e] += u
        # Normalize in Q before conversion, so finite ratios survive huge metrics.
        entries = tuple((v, e, u/de[e], u/dv[v]) for v, e, u in entries)
        dangling = np.array([not d for d in dv], dtype=bool)
        cache = {}

        def check():
            if object_digest(source) != digest:
                raise ValueError("Markov source state changed; bind a fresh view")

        def factors():
            if "numeric" not in cache:
                rows = np.array([v for v, e, a, b in entries], dtype=np.int64)
                cols = np.array([e for v, e, a, b in entries], dtype=np.int64)
                a = _numeric_array([a for v, e, a, b in entries], operation="tensor column probabilities")
                b = _numeric_array([b for v, e, a, b in entries], operation="tensor row probabilities")
                if np.any(a == 0) or np.any(b == 0):
                    raise FloatingPointError("tensor probabilities underflow float64; use the exact view action")
                cache["numeric"] = native_coo(rows, cols, a, (n, m)), native_coo(rows, cols, b, (n, m))
            return cache["numeric"]

        def raw_action(x, exact=False, transpose=False):
            if not n:
                return x.copy()
            if exact:
                tmp = np.full((m,)+x.shape[1:], Q(0), object)
                out = np.full(x.shape, Q(0), object)
                for v, e, a, b in entries:
                    tmp[e] += (a if transpose else b)*x[v]
                for v, e, a, b in entries:
                    out[v] += (b if transpose else a)*tmp[e]
            else:
                a, b = factors()
                left, right = (b, a) if transpose else (a, b)
                if x.ndim == 1 and not np.iscomplexobj(x):
                    # Captured probabilities are finite and checked once at binding.
                    out = _sparse.matvec(left.dual, _sparse.rmatvec(right.dual, np.ascontiguousarray(x)))
                else:
                    out = left.apply(right.transpose_apply(x))
            selected = x if transpose else x[dangling]
            mass = (sum(selected, np.full(x.shape[1:], Q(0), object))
                    if exact else np.sum(selected, axis=0))/n
            if transpose:
                out[dangling] += mass
            else:
                out += mass
            return out

        def action(values, exact=False, transpose=False):
            check()
            x = _exact_array(values) if exact else _numeric_array(values, operation="tensor Markov action")
            result = raw_action(x, exact, transpose)
            check()
            return result if exact else _numeric_array(result, operation="tensor Markov result")

        super().__init__("MARKOV_VIEW", (n, n), 0, 0, action, source=source,
            construction="markov-view", variance="cochain", transpose_matvec=lambda x: action(x, transpose=True),
            exact_matvec=lambda x: action(x, True), exact_transpose_matvec=lambda x: action(x, True, True),
            parameters=(("participation", "abs(B1) W"), ("dangling", "uniform"), ("direction", "column-mass")))
        object.__setattr__(self, "check_state", check)
        object.__setattr__(self, "tensor_action", raw_action)


def pagerank(view, damping=0.85, seed=None, *, tol=1e-10, maxiter=1000, report=False):
    """Numerical fixed point with measured L1 contraction bound at most tol.

The view supplies uniform dangling columns independently of the restart seed.
No exact fixed point or spectral interpretation is claimed by this interface.
"""
    if not isinstance(view, MarkovView):
        raise TypeError("PageRank requires an explicit native MARKOV_VIEW")
    if not isinstance(report, (bool, np.bool_)):
        raise TypeError("report must be boolean")
    view.check_state()
    if seed is not None:
        seed = _numeric_array(seed, operation="PageRank seed")
    result, info = _standard._pagerank_action(view.tensor_action, view.shape[0], damping, maxiter, tol, seed)
    view.check_state()
    if not info["converged"]:
        raise RuntimeError("PageRank did not meet its measured fixed point error bound")
    return (result, info) if report else result
