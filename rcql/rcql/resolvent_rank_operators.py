"""Read the core resolvent rank; RCQL builds no transition or operator of its own."""
from fractions import Fraction

from .execution_trace import record_method


def resolvent_rank(source, grade, seed=None, damping=Fraction(17, 20), metric="declared", calculus=None,
                   metrics=None):
    from rexgraph.cochain import Chain
    from rexgraph.graded_metric import DiagonalMetric
    from rexgraph.resolvent_ranking import resolvent_rank as rank
    from .operators import _typed_value
    if calculus is not None and calculus.source is not None and calculus.source is not source:
        raise ValueError("field calculus belongs to another selected source")
    values = None
    if seed is not None:
        seed = _typed_value(source, seed, operator="RESOLVENT_RANK", grade=grade)
        if seed.cell_keys is not None:
            raise ValueError("the resolvent rank seed requires the canonical basis")
        values = seed.values
    by_grade = None
    if metrics is not None:
        by_grade = {}
        for metric_value in metrics:
            if not isinstance(metric_value, DiagonalMetric) or metric_value.source is not source:
                raise TypeError("explicit metrics must be METRIC values of the bound source")
            if metric_value.grade in by_grade:
                raise ValueError(f"two explicit metrics were given for grade {metric_value.grade}")
            by_grade[metric_value.grade] = metric_value
        by_grade = [by_grade.get(g) for g in range(max(by_grade, default=-1) + 1)]
    result, info = rank(source if calculus is None else calculus, grade, values, damping, metric,
                        metrics=by_grade, report=True)
    record_method(info["method"], **{k: v for k, v in info.items() if k != "method"})
    return Chain(grade, result, source=source)


def install(register):
    register("RESOLVENT_RANK")(resolvent_rank)
