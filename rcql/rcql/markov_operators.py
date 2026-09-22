"""Thin participation walk and PageRank adapters to native Core, named for their implementation."""
from fractions import Fraction
from rexgraph.cochain import Cochain

from .execution_trace import record_method


def participation_walk(source, grade=0):
    from rexgraph.markov import ParticipationWalk
    result = ParticipationWalk(source, grade)
    record_method("native-participation-walk", dangling="uniform")
    return result


def pagerank_iteration(source, view, damping=0.85, seed=None, tol=1e-10, maxiter=1000):
    from rexgraph.markov import pagerank_iteration as solve
    from .operators import _typed_value
    if view.source is not source:
        raise ValueError("the PageRank iteration view must belong to its bound source")
    if seed is not None:
        seed = _typed_value(source, seed, operator="PAGERANK_ITERATION", variance="cochain", grade=0)
        if seed.cell_keys is not None:
            raise ValueError("the PageRank iteration seed requires the canonical basis")
    values, info = solve(view, damping, None if seed is None else seed.values,
                         tol=tol, maxiter=maxiter, report=True)
    record_method(info["kernel"], **{k: v for k, v in info.items() if k != "kernel"})
    return Cochain(0, values, source=source)


def pagerank_solve(source, view, damping=Fraction(17,20), seed=None):
    from rexgraph.ranking_response import pagerank_solve as solve
    from .operators import _typed_value
    if view.source is not source:
        raise ValueError("the PageRank solve view must belong to its bound source")
    if seed is not None:
        seed = _typed_value(source, seed, operator="PAGERANK_SOLVE", variance="cochain", grade=0)
        if seed.cell_keys is not None:
            raise ValueError("the PageRank solve seed requires the canonical basis")
    values, info = solve(view, damping, None if seed is None else seed.values, report=True)
    record_method(info["method"], **{k: v for k, v in info.items() if k != "method"})
    return Cochain(0, values, source=source)


def install(register):
    register("PARTICIPATION_WALK")(participation_walk)
    register("PAGERANK_ITERATION")(pagerank_iteration)
    register("PAGERANK_SOLVE")(pagerank_solve)
