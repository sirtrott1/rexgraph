"""Contracts for explicit coordinate actions and temporal channel results."""
from .types import (CoordinateDescriptor, CoordinateActionDescriptor, Domain, Exactness,
                    PredicateResult, RCType, ShapeRef, SourceRef, ValueKind)

ARGUMENTS = {
    "COORDINATE_APPLY": (("action", "field"), ()),
    "OPERATION_DELTA": (("operation", "old", "new", "metric"), (None,)),
    "INJECTION_DELTA": (("operation", "old", "new", "metric"), (None,)),
    "WORD_DELTA": (("word", "field", "metric"), (None,)),
    "KERNEL_MOMENTS": (("kernel", "field"), ()),
}


def _coordinate(space):
    return CoordinateDescriptor(space.name, space.keys)


def coordinate_literal(binding, value):
    from rexgraph.coordinate_map import CoordinateMap, CoordinateWord, CoordinateDifference, CoordinateMetric
    from rexgraph.temporal_calculus import TemporalMetrics, TemporalOperation, TemporalWord, MomentKernel
    from rexgraph.type_accession import CoordinateField
    from .planning import _coefficient_contract
    from .types import Variance

    if isinstance(value, CoordinateField):
        domain, arithmetic = _coefficient_contract(value.values)
        ref = binding.ref if value.source is binding.value else SourceRef("foreign")
        return RCType("CoordinateField", grade=value.grade, kind=ValueKind.COORDINATE_FIELD,
                      domain=domain, exactness=arithmetic, variance=Variance(value.variance),
                      source=ref, shape=ShapeRef(value.values.shape), coordinates=_coordinate(value.space))
    if isinstance(value, TemporalMetrics):
        ref = binding.ref if value.domain.source is binding.value else SourceRef("foreign")
        return RCType("TemporalMetrics", kind=ValueKind.TEMPORAL_METRICS, source=ref,
                      domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest)
    if isinstance(value, CoordinateMetric):
        n = len(value.space.keys)
        return RCType("CoordinateMetric", kind=ValueKind.COORDINATE_METRIC,
                      domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL, shape=ShapeRef((n, n)),
                      coordinates=_coordinate(value.space), declaration_digest=value.coefficient_digest)
    if isinstance(value, (CoordinateMap, CoordinateWord, CoordinateDifference)):
        kind, left, right, shape = ValueKind.COORDINATE_MAP, value.domain, value.codomain, value.shape
    elif isinstance(value, TemporalOperation):
        kind, left, right = ValueKind.TEMPORAL_OPERATION, value.old.domain, value.new.codomain
        shape = (len(right.keys), len(left.keys))
    elif isinstance(value, TemporalWord):
        kind, left, right = ValueKind.TEMPORAL_WORD, value.domain, value.codomain
        shape = (len(right.keys), len(left.keys))
    elif isinstance(value, MomentKernel):
        kind, left, right = ValueKind.MOMENT_KERNEL, value.domain, value.channels[0].codomain
        shape = (len(value.channels), len(value.channels))
    else:
        return None
    desc = CoordinateActionDescriptor(_coordinate(left), _coordinate(right),
                                       (len(right.keys), len(left.keys)), value.coefficient_digest,
                                       type(value).__name__)
    return RCType(kind.value, kind=kind, domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL,
                  shape=ShapeRef(shape), coordinate_action=desc, declaration_digest=value.coefficient_digest)


def result_type(name, args):
    if name == "COORDINATE_APPLY":
        action, field = args
        desc = action.coordinate_action
        if desc is None or field.coordinates != desc.domain:
            raise ValueError("coordinate action requires its declared input coordinates")
        return field.with_(shape=ShapeRef((desc.shape[0], *field.shape.dims[1:])),
                           coordinates=desc.codomain, domain=Domain.RATIONAL,
                           exactness=Exactness.RATIONAL)
    return RCType("ChannelMoments" if name == "KERNEL_MOMENTS" else "CoordinateDelta",
                  kind=ValueKind.RECORD, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def refine(typed, children, context):
    if not context.native:
        raise TypeError("coordinate operations require a native bound source")
    name, args = typed.operator, typed.args
    declaration = context.known_value(children[0])
    from rexgraph.temporal_calculus import TemporalOperation, TemporalWord, MomentKernel
    from rexgraph.coordinate_map import CoordinateMetric, _is_action
    expected = (TemporalOperation if name in {"OPERATION_DELTA", "INJECTION_DELTA"}
                else TemporalWord if name == "WORD_DELTA" else MomentKernel)
    if name == "COORDINATE_APPLY":
        if not _is_action(declaration):
            raise TypeError("coordinate action requires an explicit retained declaration")
    elif not isinstance(declaration, expected):
        raise TypeError("temporal operation requires an explicit retained declaration")
    domain = declaration.old.domain if isinstance(declaration, TemporalOperation) else declaration.domain
    field = args[1]
    if field.coordinates != _coordinate(domain):
        raise ValueError("field coordinates differ from the declared operation input")
    metric_index = None
    if isinstance(declaration, TemporalOperation):
        other = args[2]
        if (other.coordinates != _coordinate(declaration.new.domain)
                or other.grade != field.grade or other.variance != field.variance
                or other.shape.dims[1:] != field.shape.dims[1:]):
            raise ValueError("new field must retain the declared coordinates, grade, variance and block axes")
        metric_index = 3
        target = declaration.new.codomain
    elif name == "WORD_DELTA":
        metric_index = 2
        target = declaration.codomain
    if metric_index is not None and len(args) > metric_index and args[metric_index] is not None:
        metric = context.known_value(children[metric_index])
        if not isinstance(metric, CoordinateMetric) or metric.space != target:
            raise ValueError("measurement metric must belong to the declared target coordinates")
    return [PredicateResult("coordinate_endpoints", "verified", "named spaces and rational coefficient contracts"),
            PredicateResult("temporal_action", "deferred", "exact factored actions and retained interaction moments")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    exact = (Domain.INTEGER, Domain.RATIONAL)
    for name in ARGUMENTS:
        if name == "COORDINATE_APPLY":
            kind = ValueKind.COORDINATE_MAP
        elif name in {"OPERATION_DELTA", "INJECTION_DELTA"}:
            kind = ValueKind.TEMPORAL_OPERATION
        elif name == "WORD_DELTA":
            kind = ValueKind.TEMPORAL_WORD
        else:
            kind = ValueKind.MOMENT_KERNEL
        patterns = [TypePattern(ARGUMENTS[name][0][0], kind=kind),
                    TypePattern(ARGUMENTS[name][0][1], kind=ValueKind.COORDINATE_FIELD,
                                domain=exact, source_bound=True)]
        if name in {"OPERATION_DELTA", "INJECTION_DELTA"}:
            patterns.append(TypePattern("new", kind=ValueKind.COORDINATE_FIELD, domain=exact))
        if name in {"OPERATION_DELTA", "INJECTION_DELTA", "WORD_DELTA"}:
            patterns.append(TypePattern("metric", kind=ValueKind.COORDINATE_METRIC,
                                        literal=type(None), optional=True))
        register(OperatorSignature(name=name, source_kind=ValueKind.REX, inputs=tuple(patterns),
            result=lambda args, name=name: result_type(name, args), memoizable=True,
            implementation_key={
                "COORDINATE_APPLY": "rexgraph.coordinate_map.CoordinateMap.apply",
                "OPERATION_DELTA": "rexgraph.temporal_calculus.TemporalOperation.field_delta",
                "INJECTION_DELTA": "rexgraph.temporal_calculus.TemporalOperation.field_delta",
                "WORD_DELTA": "rexgraph.temporal_calculus.TemporalWord.delta",
                "KERNEL_MOMENTS": "rexgraph.temporal_calculus.MomentKernel.evaluate",
            }[name],
            preconditions=("explicit coordinate spaces and rational input fields",
                           "no coordinate correspondence is inferred from dimension alone")))
