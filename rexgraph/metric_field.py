"""Metric curvature fields on the primary C1 relations of a relational complex.

The metric belongs to declared relations.  This module reads the resulting
local curvature through their actual boundary incidences, rather than through
an adjacency or a pairwise source/target projection.  A C1 relation of arity
``r`` contributes boundary magnitude one at its distinguished participant and
``1/(r-1)`` at each sharing participant; a witness contributes one at its sole
participant.  Repeated incidence is retained as repeated boundary magnitude.

For a C1 metric cochain ``m`` the local mean and curvature at a C0 participant
``v`` are::

    mean_v = sum_e a_ve m_e / sum_e a_ve
    kappa_v = sum_e a_ve |m_e - mean_v|

where ``a_ve`` is the declared boundary magnitude.  The C1 contribution is the
sum of its incident terms, so summing relation contributions exactly reproduces
the total C0 curvature.  This is a field-to-metric reading on the relational
complex; it is not a fabricated C2 curvature and it does not alter the chain
condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from rexgraph.cochain import Cochain
from rexgraph.cells import cell_count

__all__ = ["MetricCurvature", "relation_metric_curvature"]


@dataclass(frozen=True)
class MetricCurvature:
    """A C1 metric's direct local curvature readings.

    ``curvature`` and ``local_mean`` are C0 cochains on the derived boundary
    participant basis.  ``relation_contribution`` is a C1 cochain on the same
    primary relation basis as ``metric``.  Exact integer or rational metrics
    retain rational arithmetic; measured floating metrics remain approximate.
    """

    metric: Cochain
    curvature: Cochain
    local_mean: Cochain
    relation_contribution: Cochain
    total: Fraction | float

    @property
    def source(self) -> Any:
        """The relational complex that declares both metric and boundaries."""
        return self.metric.source


def _exact_metric(values: np.ndarray) -> bool:
    """Whether a scalar C1 metric can retain exact rational arithmetic."""
    if values.dtype.kind in "iu":
        return True
    if values.dtype.kind != "O":
        return False
    return all(isinstance(value, (Fraction, int, np.integer)) for value in values.tolist())


def _as_exact(value: Any) -> Fraction:
    """Read one certified integer/rational coefficient without float conversion."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    raise TypeError("exact metric coefficients must be integers or Fractions")


def _incidences(rex) -> tuple[tuple[tuple[int, Fraction], ...], ...]:
    """Return exact per-relation boundary magnitudes, retaining repetitions.

    This deliberately does not read the assembled floating B1: a repeated
    incidence can cancel in that signed matrix but is still an actual primary
    relation incidence for a metric field.  The share coefficients are read as
    Fractions from the declared relation support.
    """
    rows: list[tuple[tuple[int, Fraction], ...]] = []
    for support in rex.relation_supports():
        arity = len(support)
        if arity == 0:
            rows.append(())
            continue
        if arity == 1:
            rows.append(((int(support[0]), Fraction(1)),))
            continue
        share = Fraction(1, arity - 1)
        rows.append(tuple(
            (int(vertex), Fraction(1) if position == 0 else share)
            for position, vertex in enumerate(support)
        ))
    return tuple(rows)


def relation_metric_curvature(rex, metric: Cochain) -> MetricCurvature:
    """Read exact-boundary local curvature of a scalar C1 metric field.

    The operation is linear in declared C1 boundary incidences before its local
    absolute-value reading: time and memory are Theta(total relation arity plus
    number of C0 participants plus number of C1 relations).  It neither walks
    derived vertices nor expands a branching relation into pairwise cells.
    """
    if not isinstance(metric, Cochain):
        raise TypeError("relation_metric_curvature expects a typed C1 Cochain")
    if metric.source is not rex:
        raise ValueError("metric cochain must be bound to its source Rex")
    if metric.grade != 1:
        raise ValueError(f"relation_metric_curvature expects grade 1, got grade {metric.grade}")

    values = np.asarray(metric.values)
    if values.ndim != 1:
        raise ValueError("relation_metric_curvature expects one scalar value per C1 relation")
    n_relations = cell_count(rex, 1)
    if values.size != n_relations:
        raise ValueError(f"metric has {values.size} values for {n_relations} C1 relations")
    n_participants = cell_count(rex, 0)
    exact = _exact_metric(values)
    incidence = _incidences(rex)

    if exact:
        metrics = tuple(_as_exact(value) for value in values.tolist())
        weights = [Fraction(0) for _ in range(n_participants)]
        weighted_sum = [Fraction(0) for _ in range(n_participants)]
    else:
        metrics = tuple(float(value) for value in values.tolist())
        weights = np.zeros(n_participants, dtype=np.float64)
        weighted_sum = np.zeros(n_participants, dtype=np.float64)

    for relation, row in enumerate(incidence):
        for participant, weight in row:
            if exact:
                weights[participant] += weight
                weighted_sum[participant] += weight * metrics[relation]
            else:
                magnitude = float(weight)
                weights[participant] += magnitude
                weighted_sum[participant] += magnitude * metrics[relation]

    if exact:
        means = tuple(
            weighted_sum[index] / weights[index] if weights[index] else Fraction(0)
            for index in range(n_participants)
        )
        curvature = [Fraction(0) for _ in range(n_participants)]
        relation = [Fraction(0) for _ in range(n_relations)]
    else:
        means = np.divide(
            weighted_sum, weights, out=np.zeros(n_participants, dtype=np.float64), where=weights != 0
        )
        curvature = np.zeros(n_participants, dtype=np.float64)
        relation = np.zeros(n_relations, dtype=np.float64)

    for relation_index, row in enumerate(incidence):
        for participant, weight in row:
            contribution = weight * abs(metrics[relation_index] - means[participant])
            if exact:
                curvature[participant] += contribution
                relation[relation_index] += contribution
            else:
                value = float(contribution)
                curvature[participant] += value
                relation[relation_index] += value

    dtype = object if exact else np.float64
    curvature_values = np.asarray(curvature, dtype=dtype)
    mean_values = np.asarray(means, dtype=dtype)
    relation_values = np.asarray(relation, dtype=dtype)
    total = sum(curvature) if exact else float(curvature_values.sum())
    return MetricCurvature(
        metric=metric,
        curvature=Cochain(0, curvature_values, source=rex),
        local_mean=Cochain(0, mean_values, source=rex),
        relation_contribution=Cochain(1, relation_values, source=rex),
        total=total,
    )
