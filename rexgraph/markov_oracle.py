"""Explicit endpoint comparison oracle, outside native RCQL execution."""
from fractions import Fraction as Q

import numpy as np

from rexgraph.core import _standard, _sparse
from rexgraph.io.catalog import object_digest
from rexgraph.linear_operator import RexOperator, _exact_array, _numeric_array
from rexgraph.native_sparse import NativeSparse
from rexgraph.markov import validate_markov_source


class PairwiseMarkovOracle(RexOperator):
    """Column stochastic mass action; dangling columns are uniform.

The transpose acts on observables. This is generally not a symmetric map.
Parallel relations and stored loop multiplicity retain the endpoint convention.
"""
    def __init__(self, source, grade=0):
        weights = validate_markov_source(source, grade)
        source._require_pairwise_c1("pairwise Markov oracle")
        ptr, idx, edges = source._adjacency_bundle
        n = int(source.nV)
        digest = object_digest(source)
        exact_weights = tuple(Q(1) for _ in range(source.nE)) if weights is None else tuple(weights)
        cache = {}

        def check():
            if object_digest(source) != digest:
                raise ValueError("Markov source state changed; bind a fresh view")

        def data():
            check()
            if "data" not in cache:
                metric = _numeric_array(exact_weights, operation="Markov weights")
                if any(w and metric[i] == 0 for i, w in enumerate(exact_weights)):
                    raise FloatingPointError("Markov metric underflows float64; use the exact view action")
                cache["data"] = np.ascontiguousarray(metric[edges])
            return ptr, idx, cache["data"]

        def action(values, exact=False, transpose=False):
            check()
            x = _exact_array(values) if exact else _numeric_array(values, operation="Markov action")
            if not n:
                return x.copy()
            if exact:
                if "exact" not in cache:
                    rows = []
                    for u in range(n):
                        terms = tuple((int(idx[k]), exact_weights[int(edges[k])])
                                      for k in range(int(ptr[u]), int(ptr[u+1])))
                        degree = sum((w for _, w in terms), Q(0))
                        rows.append(tuple((v, w/degree) for v, w in terms if w) if degree else ())
                    cache["exact"] = tuple(rows)
                out = np.full(x.shape, Q(0), object)
                total = sum(x, np.full(x.shape[1:], Q(0), object))/n if transpose else None
                dangling_mass = np.full(x.shape[1:], Q(0), object)
                for u, terms in enumerate(cache["exact"]):
                    if not terms:
                        if transpose:
                            out[u] = total
                        else:
                            dangling_mass += x[u]
                    for v, w in terms:
                        if transpose:
                            out[u] += w*x[v]
                        else:
                            out[v] += w*x[u]
                if not transpose:
                    out += dangling_mass/n
            else:
                if "numeric" not in cache:
                    p, dangling = _standard.markov_weights(*data(), n)
                    csr = _sparse.CSRMatrix(ptr, idx, p, n, n)
                    cache["numeric"] = NativeSparse(_sparse.dual_from_csr(csr)), dangling.astype(bool)
                walk, dangling = cache["numeric"]
                out = walk.apply(x) if transpose else walk.transpose_apply(x)
                if transpose:
                    out[dangling] += np.sum(x, axis=0)/n
                else:
                    out += np.sum(x[dangling], axis=0)/n
            check()
            return out if exact else _numeric_array(out, operation="Markov result")

        super().__init__("PAIRWISE_MARKOV_ORACLE", (n, n), 0, 0, action, source=source,
            construction="pairwise-markov-oracle", variance="cochain", transpose_matvec=lambda x: action(x, transpose=True),
            exact_matvec=lambda x: action(x, True), exact_transpose_matvec=lambda x: action(x, True, True),
            parameters=(("reading", "endpoint"), ("dangling", "uniform"), ("direction", "column-mass")))
        object.__setattr__(self, "check_state", check)
        object.__setattr__(self, "adjacency", data)


def pairwise_pagerank_oracle(view, damping=0.85, seed=None, *, tol=1e-10, maxiter=1000, report=False):
    """Evaluate the legacy endpoint equation for comparisons only."""
    if not isinstance(view, PairwiseMarkovOracle):
        raise TypeError("endpoint solver requires PairwiseMarkovOracle")
    if not isinstance(report, (bool, np.bool_)):
        raise TypeError("report must be boolean")
    view.check_state()
    seed = None if seed is None else _numeric_array(seed, operation="PageRank oracle seed")
    result, info = _standard.pagerank(*view.adjacency(), view.shape[0], int(view.source.nE),
                                     damping, maxiter, tol, seed=seed, report=True)
    view.check_state()
    if not info["converged"]:
        raise RuntimeError("PageRank oracle did not meet its measured fixed point error bound")
    return (result, info) if report else result
