"""Finite compositions and variations of the core's explicit typed actions."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import DiagonalMetric, _fraction
from rexgraph.linear_operator import RexOperator, _numeric_array, boundary_operator, coboundary_operator

from .execution_trace import record_method


def _action(source, value):
    if not isinstance(value, RexOperator) or value.source is not source:
        raise TypeError("action requires a RexOperator bound to this source")
    from rexgraph.cells import cell_count
    expected = (cell_count(source, value.codomain_grade, allow_empty_upper=True),
                cell_count(source, value.domain_grade, allow_empty_upper=True))
    if value.shape != expected:
        raise ValueError("operator axes no longer match the source")
    return value


class ComposedAction(RexOperator):
    """Ordered factors, stored and applied without constructing their product."""

    def __init__(self, source, factors):
        if not isinstance(factors, (list, tuple)) or not factors:
            raise ValueError("CHAIN requires a nonempty explicit sequence of actions")
        factors = tuple(_action(source, factor) for factor in factors)
        for left, right in zip(factors, factors[1:], strict=False):
            if (left.codomain_grade != right.domain_grade or left.shape[0] != right.shape[1]
                    or left.variance != right.variance):
                raise ValueError("CHAIN factors require matching intermediate grades and variance")

        def forward(values, exact=False):
            for factor in factors:
                _action(source, factor)
                values = factor.apply(values, exact=exact)
            return values

        def transpose(values, exact=False):
            for factor in reversed(factors):
                _action(source, factor)
                values = factor.transpose_apply(values, exact=exact)
            return values

        exact = all(f.exact_matvec is not None for f in factors)
        exact_t = all(f.exact_transpose_matvec is not None for f in factors)
        super().__init__("cell-chain", (factors[-1].shape[0], factors[0].shape[1]),
                         factors[0].domain_grade, factors[-1].codomain_grade, forward,
                         source=source, variance=factors[0].variance, construction="cell-chain",
                         transpose_matvec=transpose if all(f.has_transpose for f in factors) else None,
                         exact_matvec=(lambda x: forward(x, True)) if exact else None,
                         exact_transpose_matvec=(lambda x: transpose(x, True)) if exact_t else None)
        object.__setattr__(self, "factors", factors)


def chain(source, actions):
    return ComposedAction(source, actions)


def apply_composition(source, action, values, exact=False):
    from .operators import _typed_value
    _action(source, action)
    value = _typed_value(source, values, operator="CHAIN", variance=action.variance,
                         grade=action.domain_grade, allow_empty_upper=True)
    if value.cell_keys is not None:
        raise ValueError("CHAIN requires the canonical ordered basis")
    coefficients = np.asarray(value.values) if exact else _numeric_array(value.values, operation="CHAIN")
    result = action.apply(coefficients, exact=exact)
    record_method("exact-factored-cell-chain" if exact else "native-factored-cell-chain",
                  factors=len(action.factors))
    return (Chain if action.variance == "chain" else Cochain)(action.codomain_grade, result, source=source)


def transfer(source, action, values, target=None, exact=True):
    """Apply a cell chain, optionally contracting against a target in its range."""
    from .operators import moment
    if not isinstance(action, ComposedAction):
        raise TypeError("TRANSFER requires an explicit CHAIN")
    out = apply_composition(source, action, values, exact)
    return out if target is None else moment(source, target, out, None, exact)


def dependence(source, values):
    """Exact relations among an ordered family, in sparse family coordinates."""
    from rexgraph.graded_boundary import _exact_kernel_columns
    from .value_operators import _family
    values = _family(source, values, None, True)
    columns = [{i: _fraction(x) for i, x in enumerate(np.asarray(v.values).flat) if x != 0}
               for v in values]
    kernel = tuple(tuple(sorted(c.items())) for c in _exact_kernel_columns(columns))
    record_method("exact-sparse-family-kernel", members=len(values), nullity=len(kernel))
    return {"rank": len(values) - len(kernel), "nullity": len(kernel), "kernel": kernel,
            "coordinates": "ordered-input-family", "dependent": bool(kernel)}


def strain(source, grade, weight):
    """B_k W_k B_(k+1), retaining the full two grade map and primary shares."""
    from .operators import _grade_argument
    grade = _grade_argument(grade)
    if grade < 1:
        raise ValueError("STRAIN requires a positive interface grade")
    if not isinstance(weight, DiagonalMetric) or weight.source is not source or weight.grade != grade:
        raise TypeError("STRAIN requires a diagonal metric at its source interface grade")
    if weight.cell_keys is not None:
        raise ValueError("STRAIN requires the canonical ordered basis")
    lower = boundary_operator(source, grade)
    from rexgraph.native_sparse import native_boundaries
    if grade < len(native_boundaries(source)):
        upper = boundary_operator(source, grade + 1)
    else:
        dual = coboundary_operator(source, grade)
        upper = RexOperator("empty-upper-boundary", (dual.shape[1], dual.shape[0]), grade + 1, grade,
                            dual.transpose_apply, source=source, variance="chain",
                            construction="boundary", transpose_matvec=dual.apply,
                            exact_matvec=lambda x: dual.transpose_apply(x, exact=True),
                            exact_transpose_matvec=lambda x: dual.apply(x, exact=True))

    def scale(values, exact=False):
        if exact:
            values = np.asarray([_fraction(v) for v in np.asarray(values).flat], dtype=object).reshape(values.shape)
            weights = np.asarray(weight.weights, dtype=object)
        else:
            values = _numeric_array(values, operation="STRAIN")
            weights = np.asarray(weight.weights, dtype=float)
            if not np.isfinite(weights).all() or np.any(weights == 0):
                raise FloatingPointError("STRAIN weights are outside numerical range")
        return values * (weights[:, None] if values.ndim == 2 else weights)

    n = len(weight.weights)
    diagonal = RexOperator("strain-weight", (n, n), grade, grade, scale, source=source,
                           variance="chain", symmetric=True, psd=True,
                           construction="explicit-diagonal", transpose_matvec=scale,
                           exact_matvec=(lambda x: scale(x, True)) if weight.exact else None,
                           exact_transpose_matvec=(lambda x: scale(x, True)) if weight.exact else None)
    result = ComposedAction(source, (upper, diagonal, lower))
    object.__setattr__(result, "strain_weight", weight)
    object.__setattr__(result, "strain_grade", grade)
    return result


def metric_homotopy(source, left, right, parameter):
    """The convex segment between two explicit positive diagonal forms."""
    from .calculus_contracts import interpolation_parameter
    parameter = interpolation_parameter(parameter)
    if not all(isinstance(m, DiagonalMetric) and m.source is source for m in (left, right)):
        raise TypeError("METRIC_HOMOTOPY requires two bound diagonal metrics")
    if (left.grade, left.cell_keys, len(left.weights)) != (right.grade, right.cell_keys, len(right.weights)):
        raise ValueError("METRIC_HOMOTOPY requires the same ordered grade and basis")
    if left.cell_keys is not None:
        raise ValueError("METRIC_HOMOTOPY requires the canonical ordered basis")
    weights = tuple((1 - parameter) * a + parameter * b for a, b in zip(left.weights, right.weights, strict=True))
    return DiagonalMetric(source, left.grade, weights)


def cross_metric(source, left, right, entries):
    """Declare an ordered sparse cross block, without asserting positivity."""
    from rexgraph.type_accession import CrossMetric, TypeAccession
    if not all(isinstance(a, TypeAccession) and a.source is source and a.cell_keys is None for a in (left, right)):
        raise TypeError("CROSS_METRIC requires canonical accessions bound to its source")
    return CrossMetric(left, right, entries)


def _quadratic(source, value, operator, forcing, exact):
    from .operators import _same_space, _typed_value, moment
    from rexgraph.weighted_hodge import WeightedHodgeOperator
    if not isinstance(exact, bool):
        raise TypeError("exact must be boolean")
    operator = _action(source, operator)
    metric = operator.grade_metric if isinstance(operator, WeightedHodgeOperator) else None
    if (not operator.symmetric and metric is None) or operator.domain_grade != operator.codomain_grade:
        raise TypeError("ACTION requires a declared self adjoint endomorphism and its form")
    value = _typed_value(source, value, operator="ACTION", variance=operator.variance,
                         grade=operator.domain_grade, allow_empty_upper=True)
    if value.cell_keys is not None or np.iscomplexobj(value.values):
        raise TypeError("ACTION requires canonical real coefficient carriers")
    if forcing is not None:
        value, forcing = _same_space(source, value, forcing, operator="ACTION")
        forcing_array = (np.asarray([_fraction(v) for v in forcing.values.flat], dtype=object).reshape(forcing.values.shape)
                         if exact else _numeric_array(forcing.values, operation="ACTION"))
        if np.iscomplexobj(forcing_array):
            raise TypeError("ACTION requires real forcing")
        forcing = type(value)(value.grade, forcing_array, source=source)
    # Validate all coefficients before applying factors that could annihilate them.
    coefficients = np.asarray(value.values)
    if exact:
        coefficients = np.asarray([_fraction(v) for v in coefficients.flat], dtype=object).reshape(coefficients.shape)
    else:
        coefficients = _numeric_array(coefficients, operation="ACTION")
        if np.iscomplexobj(coefficients):
            raise TypeError("ACTION requires real coefficients")
    output = np.asarray(operator.apply(coefficients, exact=exact))
    output = (np.asarray([_fraction(v) for v in output.flat], dtype=object).reshape(output.shape) if exact else
              _numeric_array(output, operation="ACTION"))
    if output.shape != coefficients.shape or np.iscomplexobj(output):
        raise ValueError("ACTION output must preserve the real vector or block shape")
    applied = type(value)(value.grade, output, source=source)
    linear = Fraction(0) if exact else 0.0
    if forcing is not None:
        linear = moment(source, forcing, value, metric, exact)
    return value, applied, forcing, linear, metric


def action(source, value, operator, forcing=None, exact=True):
    """One half <x,A x> minus <j,x>; A includes any requested regularization."""
    from .operators import moment
    value, applied, _, linear, metric = _quadratic(source, value, operator, forcing, exact)
    half = Fraction(1, 2) if exact else 0.5
    record_method("exact-quadratic-action" if exact else "native-quadratic-action")
    return half * moment(source, value, applied, metric, exact) - linear


def variation(source, operator, value, direction=None, forcing=None, exact=True):
    """A x minus j in the declared form, or its pairing with a direction."""
    from .operators import moment
    value, applied, forcing, _, metric = _quadratic(source, value, operator, forcing, exact)
    coefficients = applied.values
    if forcing is not None:
        other = np.asarray(forcing.values, dtype=object if exact else float)
        coefficients = coefficients - other
    if not exact and not np.isfinite(coefficients).all():
        raise FloatingPointError("variation is outside numerical range")
    residual = type(value)(value.grade, coefficients, source=source)
    record_method("exact-quadratic-variation" if exact else "native-quadratic-variation")
    return residual if direction is None else moment(source, direction, residual, metric, exact)


def differential(source, value, direction, exact=True):
    """d <x,L_k x>[eta] = 2 <eta,L_k x>, with identity endpoint metrics."""
    from rexgraph.weighted_hodge import weighted_hodge
    from .operators import _typed_value
    value = _typed_value(source, value, operator="DIFFERENTIAL", variance="chain")
    return 2 * variation(source, weighted_hodge(source, value.grade), value, direction, None, exact)


ADAPTERS = {"CHAIN": chain, "TRANSFER": transfer, "DEPENDENCE": dependence,
            "STRAIN": strain, "METRIC_HOMOTOPY": metric_homotopy,
            "ACTION": action, "VARIATION": variation, "DIFFERENTIAL": differential,
            "CROSS_METRIC": cross_metric}


def install(register):
    for name, function in ADAPTERS.items():
        register(name)(function)
