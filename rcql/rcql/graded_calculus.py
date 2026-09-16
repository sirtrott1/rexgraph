"""Non executing full tower Chain/operator contracts.

No boundary action, Gram construction or eigensolve runs during planning.
"""
from __future__ import annotations

from math import isfinite

from .types import (
    BasisRef,
    Domain,
    Exactness,
    GradedOperatorDescriptor,
    MetricDescriptor,
    PredicateResult,
    RCType,
    ShapeRef,
    ValueKind,
    Variance,
)


def dirac_descriptor(ref, sizes, metrics, anti, active, exact):
    positive = all(m.positive_definite is True for m in metrics)
    return GradedOperatorDescriptor(
        "weighted-dirac", tuple(BasisRef(ref.name, k) for k in range(len(sizes))),
        tuple(sizes), tuple(metrics), anti, tuple(active), exact,
        Domain.RATIONAL if exact else Domain.REAL, Exactness.APPROXIMATE,
        metric_self_adjoint=(not anti) if positive else None,
        metric_skew_adjoint=anti if positive else None,
    )


def graded_application(args):
    action, value = args[:2]
    desc = action.graded_operator
    exact = len(args) > 2 and args[2] is True
    if desc is None or not isinstance(value, RCType) or value.kind is not ValueKind.GRADED_CHAIN:
        raise TypeError("graded operator APPLY requires an explicit GradedChain")
    if (value.graded_bases != desc.bases or value.source != action.source
            or value.temporal != action.temporal or value.variance is not Variance.CHAIN):
        raise TypeError("graded operator APPLY requires the same source, full graded bases, Chain variance and time")
    if exact and not desc.exact_action:
        raise TypeError("graded operator has no certified exact action for these boundaries and metrics")
    allowed = {Domain.INTEGER, Domain.RATIONAL} if exact else {
        Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX}
    if value.domain not in allowed:
        raise TypeError("graded operator exact=True requires integer/rational coefficients" if exact else
                        "graded operator requires numeric coefficients")
    return value.with_(domain=Domain.RATIONAL if exact else Domain.COMPLEX if value.domain is Domain.COMPLEX else Domain.REAL,
                       exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE)


def component_result(args):
    state, grade = args
    if len(state.member_shapes) != len(state.graded_bases):
        raise ValueError("graded Chain must declare every component shape")
    if not 0 <= grade < len(state.graded_bases):
        raise ValueError("component grade is not carried")
    return RCType("Chain", grade=grade, kind=ValueKind.CHAIN, variance=Variance.CHAIN,
                  domain=state.domain, exactness=state.exactness, source=state.source, temporal=state.temporal,
                  basis=state.graded_bases[grade], shape=ShapeRef(state.member_shapes[grade]))


def validate_tower(value, context):
    """Reject wrong direct sum spaces before any adapter runs."""
    bases = tuple(BasisRef(context.binding.ref.name, k) for k in range(len(context.sizes)))
    if value.source != context.binding.ref or value.temporal not in (None, context.binding.temporal):
        raise TypeError("graded value requires the bound source and temporal state")
    if value.kind is ValueKind.GRADED_OPERATOR:
        desc = value.graded_operator
        if desc is None or desc.bases != bases or desc.sizes != tuple(context.sizes):
            raise ValueError("graded operator requires the complete canonical source tower")
        if desc.operands:
            for operand in desc.operands:
                validate_tower(value.with_(graded_operator=operand), context)
        if (not desc.operands or desc.grade_metrics) and (len(desc.grade_metrics) != len(bases) or any(
            m.basis != b or m.shape != (n, n)
            for m, b, n in zip(desc.grade_metrics, bases, context.sizes, strict=True)
        )):
            raise ValueError("graded operator metric axes do not match the source tower")
    else:
        if value.graded_bases != bases or len(value.member_shapes) != len(bases):
            raise ValueError("graded Chain requires the complete canonical source tower")
        trailing = value.member_shapes[0][1:]
        if value.variance is not Variance.CHAIN or any(
            len(s) not in (1, 2) or s[0] != n or s[1:] != trailing
            for s, n in zip(value.member_shapes, context.sizes, strict=True)
        ):
            raise ValueError("graded Chain component cell/block axes do not match the source tower")


def refine_graded(name, args, result, context):
    sizes, ref = context.sizes, context.binding.ref
    bases = tuple(BasisRef(ref.name, k) for k in range(len(sizes)))

    def sequence(items, kind):
        if not isinstance(items, (tuple, list)):
            raise TypeError("graded inputs require an explicit list or tuple")
        seen = set()
        for item in items:
            if (not isinstance(item, RCType) or item.kind is not kind or item.source != ref
                    or item.temporal not in (None, context.binding.temporal)):
                raise TypeError("graded entries require the declared kind, source and temporal state")
            if item.grade in seen:
                raise ValueError("duplicate graded entry")
            if not isinstance(item.grade, int) or isinstance(item.grade, bool) or not 0 <= item.grade < len(sizes):
                raise ValueError("graded entry must name a carried grade")
            if item.basis != bases[item.grade]:
                raise TypeError("graded entries require canonical ordered bases")
            seen.add(item.grade)
        return items

    facts = []
    if name in {"DIRAC", "ANTI_DIRAC"}:
        from rexgraph.graded_boundary import _integer_columns
        items = sequence(() if not args or args[0] is None else args[0], ValueKind.METRIC)
        selected = {}
        for m in items:
            if m.metric is None or m.metric.basis != m.basis or m.metric.shape != (sizes[m.grade],)*2:
                raise ValueError("Dirac metric axes must match the source grade population")
            selected[m.grade] = m.metric
        metrics = tuple(selected.get(k) or MetricDescriptor(
            "identity", bases[k], (n, n), Domain.RATIONAL, Exactness.RATIONAL, positive_definite=True)
            for k, n in enumerate(sizes))
        active, exact = [], True
        for k, b in enumerate(context.boundaries, 1):
            if any(not isfinite(float(v)) for v in b.data):
                raise ValueError("Dirac requires finite boundary coefficients")
            if b.nnz:
                active.append(k)
                exact &= (k == 1 or _integer_columns(b) is not None) and all(
                    metrics[g].coefficient_domain in {Domain.INTEGER, Domain.RATIONAL} for g in (k-1, k))
        desc = dirac_descriptor(ref, sizes, metrics, name == "ANTI_DIRAC", active, exact)
        result = result.with_(graded_operator=desc, shape=ShapeRef((sum(sizes),)*2))
        facts += [PredicateResult("graded_metric_adjointness", "verified" if all(
            m.positive_definite is True for m in metrics) else "deferred",
            "boundary +/- metric adjoint; known positive diagonals or runtime validation of computed metrics"),
            PredicateResult("dirac_chain_identities", "not-asserted",
                "squares/anticommutator require the source chain law; construction does not run that verification"),
            PredicateResult("euclidean_psd", "not-asserted", "no Euclidean symmetry or PSD solver dispatch")]
    elif name == "GRADED_CHAIN":
        items = sequence(args[0], ValueKind.CHAIN)
        trailing = ()
        domains = set()
        for c in items:
            if c.variance is not Variance.CHAIN or c.domain not in {
                    Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX}:
                raise TypeError("graded Chain requires numeric Chain coefficients")
            shape = () if c.shape is None else c.shape.dims
            if len(shape) not in (1, 2) or shape[0] != sizes[c.grade]:
                raise ValueError("graded Chain component has wrong cell/block axes")
            if domains and trailing != shape[1:]:
                raise ValueError("graded Chain components require matching vector/block shapes")
            trailing = shape[1:]
            domains.add(c.domain)
        domain = Domain.COMPLEX if Domain.COMPLEX in domains else Domain.REAL if Domain.REAL in domains else Domain.RATIONAL
        result = result.with_(domain=domain, exactness=Exactness.RATIONAL if domain is Domain.RATIONAL else Exactness.APPROXIMATE,
                              graded_bases=bases, member_shapes=tuple((n, *trailing) for n in sizes))
        facts.append(PredicateResult("graded_chain_spaces", "verified",
            "canonical Chain components; unique grades; common block shape; omitted grades are exact zero seeds"))
    for value in args:
        if isinstance(value, RCType) and value.kind in {ValueKind.GRADED_CHAIN, ValueKind.GRADED_OPERATOR}:
            validate_tower(value, context)
    return result, facts
