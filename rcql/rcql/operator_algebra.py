"""Non executing bracket contracts; matching dimensions alone are insufficient."""
from __future__ import annotations

from .types import (
    Domain,
    Exactness,
    GradedOperatorDescriptor,
    OperatorDescriptor,
    ShapeRef,
    ValueKind,
)


def _form(desc):
    if desc.metric_self_adjoint is True:
        return desc.grade_metrics, 1
    if desc.metric_skew_adjoint is True:
        return desc.grade_metrics, -1
    if isinstance(desc, OperatorDescriptor) and desc.euclidean_skew_adjoint is True:
        return (), -1
    if desc.symmetric is True:
        return (), 1
    return None


def _certificate(left, right, anti):
    a, b = _form(left), _form(right)
    if a is None or b is None or len(a[0]) != len(b[0]):
        return (), None
    for x, y in zip(a[0], b[0], strict=True):
        if (x.basis != y.basis or x.shape != y.shape or x.positive_definite is not True
                or y.positive_definite is not True):
            return (), None
        if not (x.construction == y.construction == "identity" or (
                x.coefficient_digest is not None and x.coefficient_digest == y.coefficient_digest)):
            return (), None
    return a[0], (1 if anti else -1)*a[1]*b[1]


def bracket_result(args, *, anti=False):
    left, right = args
    if left.kind != right.kind or left.source != right.source or left.temporal != right.temporal:
        raise TypeError("bracket requires matching operator kinds, source and temporal state")
    graded = left.kind is ValueKind.GRADED_OPERATOR
    a, b = (left.graded_operator, right.graded_operator) if graded else (left.operator, right.operator)
    if a is None or b is None:
        raise TypeError("bracket requires explicit native operator descriptors")
    if graded:
        if a.bases != b.bases or a.sizes != b.sizes or a.action_variance != b.action_variance:
            raise TypeError("graded bracket requires the same full tower, bases and variance")
    elif (a.domain != a.codomain or b.domain != b.codomain or a.domain != b.domain
            or a.shape != b.shape or a.shape[0] != a.shape[1] or a.action_variance != b.action_variance
            or a.domain.ordering != "canonical"):
        raise TypeError("bracket requires endomorphisms of the same grade, canonical basis and variance")
    exact = a.exact_action and b.exact_action
    domain = Domain.COMPLEX if Domain.COMPLEX in {a.coefficient_domain, b.coefficient_domain} else (
        Domain.RATIONAL if exact else Domain.REAL)
    metrics, sign = _certificate(a, b, anti)
    kind = "anticommutator" if anti else "commutator"
    if graded:
        desc = GradedOperatorDescriptor("graded-operator-bracket", a.bases, a.sizes, metrics, None,
            tuple(sorted(set(a.active_boundaries) | set(b.active_boundaries))), exact, domain,
            Exactness.APPROXIMATE, transpose_available=a.transpose_available and b.transpose_available,
            metric_self_adjoint=True if metrics and sign == 1 else None,
            metric_skew_adjoint=True if metrics and sign == -1 else None,
            operands=(a, b), bracket_kind=kind)
        return left.with_(name="GradedOperatorBracket", graded_operator=desc)
    desc = OperatorDescriptor("operator-bracket", a.domain, a.codomain, a.shape, domain,
        Exactness.APPROXIMATE, metric="explicit-common-diagonal" if metrics else "not-inferred",
        symmetric=not metrics and sign == 1, psd=False, parameters=(("bracket", kind),),
        transpose_available=a.transpose_available and b.transpose_available,
        exact_action=exact, exact_transpose=a.exact_transpose and b.exact_transpose,
        action_variance=a.action_variance, grade_metrics=metrics, operands=(a, b),
        metric_self_adjoint=True if metrics and sign == 1 else None,
        metric_skew_adjoint=True if metrics and sign == -1 else None,
        euclidean_skew_adjoint=True if not metrics and sign == -1 else None)
    return left.with_(name="OperatorBracket", operator=desc, shape=ShapeRef(desc.shape))
