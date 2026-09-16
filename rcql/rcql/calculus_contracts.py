"""Static contracts for composed actions and finite real quadratic calculus."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from math import isfinite

from .types import Domain, Exactness, MetricDescriptor, OperatorDescriptor, RCType, ShapeRef, ValueKind


ARGUMENTS = {
    "CHAIN": (("actions",), ()),
    "TRANSFER": (("action", "values", "target", "exact"), (None, True)),
    "DEPENDENCE": (("values",), ()),
    "STRAIN": (("grade", "weight"), ()),
    "METRIC_HOMOTOPY": (("left", "right", "parameter"), ()),
    "CROSS_METRIC": (("left", "right", "entries"), ()),
    "ACTION": (("value", "operator", "forcing", "exact"), (None, True)),
    "VARIATION": (("operator", "value", "direction", "forcing", "exact"), (None, None, True)),
    "DIFFERENTIAL": (("value", "direction", "exact"), (True,)),
}


def chain_result(args):
    actions = args[0]
    if not isinstance(actions, (tuple, list)) or not actions:
        raise ValueError("CHAIN requires a nonempty explicit sequence of actions")
    if any(not isinstance(a, RCType) or a.kind is not ValueKind.OPERATOR or a.operator is None for a in actions):
        raise TypeError("CHAIN requires explicit operator descriptors")
    first = actions[0]
    if any(a.source != first.source or a.temporal != first.temporal for a in actions):
        raise ValueError("CHAIN requires one source and temporal state")
    factors = tuple(a.operator for a in actions)
    if any(a.domain.ordering != "canonical" or a.codomain.ordering != "canonical" for a in factors):
        raise ValueError("CHAIN requires canonical ordered bases")
    for left, right in zip(factors, factors[1:], strict=False):
        if left.codomain != right.domain or left.shape[0] != right.shape[1] or left.action_variance != right.action_variance:
            raise ValueError("CHAIN factors require matching intermediate grades, bases and variance")
    exact = all(a.exact_action for a in factors)
    domain = Domain.COMPLEX if any(a.coefficient_domain is Domain.COMPLEX for a in factors) else (
        Domain.RATIONAL if exact else Domain.REAL)
    desc = OperatorDescriptor("cell-chain", factors[0].domain, factors[-1].codomain,
                              (factors[-1].shape[0], factors[0].shape[1]), domain, Exactness.APPROXIMATE,
                              metric="not-inferred", symmetric=False, psd=False,
                              transpose_available=all(a.transpose_available for a in factors),
                              exact_action=exact, exact_transpose=all(a.exact_transpose for a in factors),
                              action_variance=factors[0].action_variance, operands=factors)
    return first.with_(name="CellChain", grade=desc.domain.grade, basis=desc.domain,
                       operator=desc, shape=ShapeRef(desc.shape))


def _transfer(args):
    from .signatures import _apply_result, _signed_moment_result
    action, value = args[:2]
    if action.operator is None or action.operator.construction != "cell-chain":
        raise TypeError("TRANSFER requires an explicit CHAIN")
    target, exact = args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else True
    result = _apply_result((action, value, exact)).with_(source=value.source, temporal=value.temporal)
    return result if target is None else _signed_moment_result((target, result, None, exact))


def _dependence(args):
    from .value_contracts import _gram
    _gram(args, rank=True)
    return RCType("Dependence", kind=ValueKind.STRUCTURAL_DESCRIPTION,
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def strain_result(args):
    grade, weight = args
    metric = weight.metric
    if grade < 1 or grade != weight.grade or metric is None:
        raise ValueError("STRAIN requires a positive interface grade and its explicit metric")
    if metric.basis.ordering != "canonical" or metric.positive_definite is not True:
        raise ValueError("STRAIN requires a canonical positive diagonal metric")
    domain, codomain = replace(metric.basis, grade=grade + 1), replace(metric.basis, grade=grade - 1)
    exact = metric.coefficient_domain in {Domain.INTEGER, Domain.RATIONAL}
    desc = OperatorDescriptor("cell-chain", domain, codomain, (None, None),
                              Domain.RATIONAL if exact else Domain.REAL, Exactness.APPROXIMATE,
                              symmetric=False, psd=False, transpose_available=True,
                              exact_action=exact, exact_transpose=exact, action_variance="chain",
                              grade_metrics=(metric,), parameters=(("strain_grade", grade),))
    return weight.with_(name="CellChain", grade=grade + 1, kind=ValueKind.OPERATOR,
                        domain=Domain.METADATA, basis=domain, metric=None,
                        operator=desc, shape=ShapeRef(desc.shape))


def interpolation_parameter(parameter):
    if isinstance(parameter, bool) or not isinstance(parameter, (int, float, Fraction)):
        raise TypeError("metric parameter must be a real scalar")
    if not 0 <= parameter <= 1 or isinstance(parameter, float) and not isfinite(parameter):
        raise ValueError("metric parameter must lie in [0, 1]")
    return Fraction(parameter) if isinstance(parameter, int) else parameter


def _metric_homotopy(args):
    left, right, parameter = args
    parameter = interpolation_parameter(parameter)
    if not left.same_space(right) or left.shape != right.shape or left.metric is None or right.metric is None:
        raise ValueError("METRIC_HOMOTOPY requires the same ordered grade and basis")
    a, b = left.metric, right.metric
    if a.positive_definite is not True or b.positive_definite is not True or a.basis.ordering != "canonical":
        raise ValueError("METRIC_HOMOTOPY requires canonical positive diagonal metrics")
    exact = isinstance(parameter, Fraction) and all(m.coefficient_domain in {
        Domain.INTEGER, Domain.RATIONAL} for m in (a, b))
    domain = Domain.RATIONAL if exact else Domain.REAL
    identity = a.construction == b.construction == "identity"
    digest = sha256(repr((a, b, parameter)).encode()).hexdigest()
    metric = MetricDescriptor("identity" if identity else "convex-diagonal", a.basis, a.shape, domain,
                              Exactness.RATIONAL if exact else Exactness.APPROXIMATE, True, digest)
    return left.with_(domain=domain, metric=metric)


def _cross_metric(args):
    from rexgraph.type_accession import _digest, _entries
    from .types import CrossMetricDescriptor
    left, right, entries = args
    if not left.same_space(right) or len(left.accessions) != 1 or len(right.accessions) != 1:
        raise ValueError("CROSS_METRIC requires one common ambient source, grade and basis")
    a, b = left.accessions[0], right.accessions[0]
    if a.basis.ordering != "canonical":
        raise ValueError("CROSS_METRIC requires the canonical ambient basis")
    shape = (a.shape[0], b.shape[0])
    entries, exact = _entries(entries, shape)
    domain = Domain.RATIONAL if exact else Domain.REAL
    digest = _digest(entries, f"cross-metric-v1|{shape}|{exact}|")
    desc = CrossMetricDescriptor(a, b, shape, domain,
                                 Exactness.RATIONAL if exact else Exactness.APPROXIMATE, digest, len(entries))
    return left.with_(name="CrossMetric", kind=ValueKind.CROSS_METRIC, domain=domain,
                      shape=ShapeRef(shape), accessions=(a, b), cross_metric=desc)


def _quadratic(args, *, derivative=False):
    from .signatures import _coefficient_carrier, _signed_moment_result
    if derivative:
        operator, value = args[:2]
        direction = args[2] if len(args) > 2 else None
        forcing = args[3] if len(args) > 3 else None
        exact = args[4] if len(args) > 4 else True
    else:
        value, operator = args[:2]
        direction = None
        forcing = args[2] if len(args) > 2 else None
        exact = args[3] if len(args) > 3 else True
    value = _coefficient_carrier(value)
    desc = operator.operator
    weighted = desc is not None and desc.construction == "weighted-hodge" and desc.metric_self_adjoint is True
    if (desc is None or not (desc.symmetric is True or weighted) or desc.domain != desc.codomain
            or desc.domain != value.basis or desc.action_variance != value.variance.value):
        raise TypeError("ACTION requires a declared self adjoint endomorphism on its field space")
    if exact and not desc.exact_action:
        raise TypeError("quadratic exact action requires certified exact factors")
    allowed = {Domain.INTEGER, Domain.RATIONAL} if exact else {Domain.INTEGER, Domain.RATIONAL, Domain.REAL}
    if value.domain not in allowed or desc.coefficient_domain is Domain.COMPLEX:
        raise TypeError("quadratic actions require real coefficients, and Q in exact mode")
    metric = None
    if weighted:
        form = desc.grade_metrics[0]
        metric = RCType("Metric", grade=value.grade, kind=ValueKind.METRIC, source=value.source,
                        temporal=value.temporal, basis=value.basis, metric=form, shape=ShapeRef(form.shape),
                        domain=form.coefficient_domain, exactness=Exactness.STRUCTURAL)
    scalar = _signed_moment_result((value, value, metric, exact))
    for other in (forcing, direction):
        if other is not None:
            if other.domain not in allowed or other.shape != value.shape:
                raise TypeError("quadratic forcing and direction require matching real vector or block shapes")
            _signed_moment_result((value, other, metric, exact))
    if derivative and direction is None:
        return value.with_(domain=scalar.domain, exactness=scalar.exactness)
    return scalar


def _differential(args):
    from .signatures import _signed_moment_result
    left, direction = args[:2]
    if left.kind is not ValueKind.CHAIN or direction.kind is not ValueKind.CHAIN:
        raise TypeError("DIFFERENTIAL requires Chains with identity endpoint metrics")
    exact = args[2] if len(args) > 2 else True
    if Domain.COMPLEX in {left.domain, direction.domain} or left.shape != direction.shape:
        raise TypeError("DIFFERENTIAL requires matching real vector or block shapes")
    return _signed_moment_result((left, direction, None, exact))


def refine(typed, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native Rex source")
    pending = list(typed.args)
    while pending:
        value = pending.pop()
        if isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, RCType) and value.source is not None:
            if value.source != typed.binding.ref or value.basis is not None and value.basis.ordering != "canonical":
                raise ValueError("calculus values require their bound source and canonical ordered basis")
            if value.grade is not None:
                context.grade(value.grade, allow_empty_upper=True)
    result = typed.result
    if typed.operator == "STRAIN":
        from rexgraph.native_rank import boundary_columns
        grade = typed.args[0]
        n = context.grade(grade, lower=1)
        if n != typed.args[1].metric.shape[0]:
            raise ValueError("STRAIN metric population changed")
        shape = (context.grade(grade - 1), context.grade(grade + 1, allow_empty_upper=True))
        exact = result.operator.exact_action
        for k in (grade, grade + 1):
            if k < len(context.sizes):
                try:
                    boundary_columns(context.binding.value, k)
                except ValueError:
                    exact = False
        result = result.with_(shape=ShapeRef(shape), operator=replace(result.operator, shape=shape,
                               exact_action=exact, exact_transpose=exact))
    return result


def install(register):
    from .signatures import OperatorSignature, TypePattern, _CHAIN_OR_COCHAIN
    bound = dict(source_bound=True, basis_bound=True)
    action = TypePattern("action", kind=ValueKind.OPERATOR, **bound)
    metric = TypePattern("metric", kind=ValueKind.METRIC, **bound)
    accession = TypePattern("accession", kind=ValueKind.TYPE_ACCESSION, **bound)
    option = TypePattern("optional carrier", kind=(ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.FIELD),
                         literal=type(None), optional=True, **bound)
    exact = TypePattern("exact", literal=bool, optional=True)
    sequence = TypePattern("explicit sequence", literal=(list, tuple))
    specs = {
        "CHAIN": ((sequence,), chain_result),
        "TRANSFER": ((action, _CHAIN_OR_COCHAIN, option, exact), _transfer),
        "DEPENDENCE": ((sequence,), _dependence),
        "STRAIN": ((TypePattern("grade", literal=int), metric), strain_result),
        "METRIC_HOMOTOPY": ((metric, metric, TypePattern("parameter", literal=(int, float, Fraction))), _metric_homotopy),
        "CROSS_METRIC": ((accession, accession, sequence), _cross_metric),
        "ACTION": ((_CHAIN_OR_COCHAIN, action, option, exact), _quadratic),
        "VARIATION": ((action, _CHAIN_OR_COCHAIN, option, option, exact), lambda args: _quadratic(args, derivative=True)),
        "DIFFERENTIAL": ((_CHAIN_OR_COCHAIN, _CHAIN_OR_COCHAIN, exact), _differential),
    }
    for name, (inputs, result) in specs.items():
        register(OperatorSignature(name=name, source_kind=ValueKind.REX, inputs=inputs, result=result,
                                   implementation_key=f"rcql.calculus.{name.lower()}", memoizable=True,
                                   preconditions=("explicit canonical source, basis, variance and arithmetic",)))
