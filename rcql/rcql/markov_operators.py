"""Thin Markov view and PageRank adapters to native Core."""
from rexgraph.cochain import Cochain

from .execution_trace import record_method


def markov_view(source, grade=0):
    from rexgraph.markov import MarkovView
    result = MarkovView(source, grade)
    record_method("native-tensor-markov-view", dangling="uniform")
    return result


def pagerank(source, view, damping=0.85, seed=None, tol=1e-10, maxiter=1000):
    from rexgraph.markov import pagerank as solve
    from .operators import _typed_value
    if view.source is not source:
        raise ValueError("PageRank view must belong to its bound source")
    if seed is not None:
        seed = _typed_value(source, seed, operator="PAGERANK", variance="cochain", grade=0)
        if seed.cell_keys is not None:
            raise ValueError("PageRank seed requires the canonical basis")
    values, info = solve(view, damping, None if seed is None else seed.values,
                         tol=tol, maxiter=maxiter, report=True)
    record_method(info["kernel"], **{k: v for k, v in info.items() if k != "kernel"})
    return Cochain(0, values, source=source)


def install(register):
    register("MARKOV_VIEW")(markov_view)
    register("PAGERANK")(pagerank)
