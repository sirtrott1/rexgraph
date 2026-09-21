"""Static contracts for retained native tensor fields and moments."""
from fractions import Fraction

from .types import (RCType, ValueKind, Exactness, Domain, Variance, SourceRef, ShapeRef,
                    CoordinateDescriptor, CoordinateActionDescriptor, PredicateResult)

ARGUMENTS = {
    "TENSOR_APPLY": (("action", "field"), ()),
    "NATIVE_RESPONSE": (("field", "parameter", "calculus"), (1, None)),
    "TENSOR_MOMENTS": (("kernel", "left", "right"), (None,)),
    "FIELD_PAIR": (("left", "right", "pairing"), ()),
    "MOMENT_PAIR": (("moments", "left", "right"), ()),
    "MOMENT_SUPPORT": (("moment",), ()),
    "MOMENT_CONTRACT": (("moment",), ()),
    "TENSOR_SCALAR": (("field",), ()),
    "TENSOR_DELTA": (("evolution", "old", "new"), ()),
    "CHANNEL_FIELD": (("channels", "name"), ()),
    "CHANNEL_TOTAL": (("channels", "weights"), (None,)),
    "ATTACHMENTS": (("annotation_ids", "roles"), (None, None)),
    "SPAN_FIELD": (("attachments", "support", "local", "mode", "amplitudes", "refinement", "scopes"),
                   ("time", True, "sum", None, None, None)),
    "SPAN_DELTA": (("old", "new", "old_values", "new_values", "support", "local", "mode"),
                   (None, None, "time", True, "sum")),
    "SECTOR_FIELDS": (("transport", "old", "new"), (None,)),
    "MOMENT_CHANGE": (("old_left", "new_left", "left_map", "old_right", "new_right", "right_map", "old_form", "new_form"), ()),
}


def _coordinate(space):
    return CoordinateDescriptor(space.name, space.keys)


def _source_ref(binding, reference):
    if reference is None:
        return None
    if reference.source is binding.value:
        return binding.ref
    return SourceRef("native_state/" + reference.state_digest[:16], state_digest=reference.state_digest,
                     record_id=reference.record_id, record_version=reference.version)


def tensor_literal(binding, value):
    from rexgraph.coordinate_map import CoordinateMap, CoordinateWord, CoordinateDifference, _is_action
    from rexgraph.tensor_field import TensorField, TensorChannels, _native_sources
    from rexgraph.tensor_moment import TensorMomentKernel, TensorMoments, MomentSpan, CoordinatePairing, RealizedPairing
    from rexgraph.temporal_field import TensorEvolution, ResolvedEvolution, NativeFieldEvolution, SectorTransport, MomentChange
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.attachment_field import AttachmentField, AttachmentObservation
    from rexgraph.reconstruction import ReconstructionFamily
    if isinstance(value, TensorField):
        return RCType("TensorField", value.grade, Exactness.RATIONAL, kind=ValueKind.TENSOR_FIELD,
                      variance={"chain": Variance.CHAIN, "cochain": Variance.COCHAIN, "coordinate": Variance.NEUTRAL}[value.variance],
                      shape=ShapeRef(value.values.shape), domain=Domain.RATIONAL,
                      source=_source_ref(binding, value.source), coordinates=_coordinate(value.space),
                      tensor_axes=tuple(_coordinate(a) for a in value.axes), declaration_digest=value.coefficient_digest)
    classes = ((NativeFieldCalculus, ValueKind.NATIVE_FIELD_CALCULUS),
               (TensorMomentKernel, ValueKind.TENSOR_KERNEL), (TensorMoments, ValueKind.TENSOR_MOMENTS),
               (MomentSpan, ValueKind.MOMENT_SPAN), (TensorChannels, ValueKind.TENSOR_CHANNELS),
               (TensorEvolution, ValueKind.TENSOR_EVOLUTION), (ResolvedEvolution, ValueKind.TENSOR_EVOLUTION),
               (NativeFieldEvolution, ValueKind.TENSOR_EVOLUTION),
               (SectorTransport, ValueKind.SECTOR_TRANSPORT), (MomentChange, ValueKind.MOMENT_CHANGE),
               (AttachmentField, ValueKind.ATTACHMENT_FIELD), (AttachmentObservation, ValueKind.ATTACHMENT_OBSERVATION),
               (CoordinatePairing, ValueKind.TENSOR_PAIRING), (RealizedPairing, ValueKind.TENSOR_PAIRING),
               (ReconstructionFamily, ValueKind.RECONSTRUCTION_FAMILY))
    for cls, kind in classes:
        if isinstance(value, cls):
            return RCType(type(value).__name__, exactness=Exactness.STRUCTURAL, kind=kind, domain=Domain.RATIONAL,
                          declaration_digest=getattr(value, "coefficient_digest", None))
    if isinstance(value, (CoordinateMap, CoordinateWord, CoordinateDifference)):
        return None
    if _is_action(value):
        sources = _native_sources(value)
        reference = binding.ref if sources and all(s is binding.value for s in sources) else None
        desc = CoordinateActionDescriptor(_coordinate(value.domain), _coordinate(value.codomain), value.shape,
                                          value.coefficient_digest, type(value).__name__)
        return RCType(type(value).__name__, exactness=Exactness.STRUCTURAL, kind=ValueKind.COORDINATE_MAP,
                      shape=ShapeRef(value.shape), domain=Domain.RATIONAL, source=reference,
                      coordinate_action=desc, declaration_digest=value.coefficient_digest)
    return None


def _result(kind):
    return RCType(kind.value, exactness=Exactness.RATIONAL if kind == ValueKind.TENSOR_FIELD else Exactness.STRUCTURAL,
                  kind=kind, domain=Domain.RATIONAL)


def _bound(reference, binding):
    if reference is None:
        return
    reference.check()
    if reference.source is not binding.value:
        raise ValueError("tensor input belongs to a different selected native source")
    if reference.record_id is not None and binding.ref.record_id is not None and reference.record_id != binding.ref.record_id:
        raise ValueError("tensor input belongs to a different selected record")
    if reference.version is not None and binding.ref.record_version is not None and reference.version != binding.ref.record_version:
        raise ValueError("tensor input belongs to a different selected record version")


def refine(typed, children, context):
    if not context.native:
        raise TypeError("native tensor operations require a bound relational source")
    name, args, result = typed.operator, typed.args, typed.result
    values = tuple(context.known_value(child) for child in children)
    from rexgraph.tensor_field import TensorField
    from rexgraph.coordinate_map import _is_action
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.graded_metric import _fraction
    if name in {"TENSOR_APPLY", "NATIVE_RESPONSE"}:
        i = 1 if name == "TENSOR_APPLY" else 0
        value = values[i]
        if isinstance(value, TensorField):
            _bound(value.source, context.binding)
            if name == "TENSOR_APPLY" and _is_action(values[0]):
                action = values[0]
                if value.space != action.domain:
                    raise ValueError("named tensor action coordinates differ")
                expected = getattr(action, "domain_variance", None)
                if expected is not None and value.variance != expected:
                    raise ValueError("tensor variance differs from the declared action domain")
                grade = getattr(action, "domain_grade", None)
                if grade is not None and value.grade is not None and grade != value.grade:
                    raise ValueError("tensor grade differs from the action domain")
                result = result.with_(grade=getattr(action, "codomain_grade", value.grade),
                    coordinates=_coordinate(action.codomain), tensor_axes=args[i].tensor_axes,
                    shape=ShapeRef((len(action.codomain.keys), *value.values.shape[1:])),
                    variance={"chain": Variance.CHAIN, "cochain": Variance.COCHAIN, "coordinate": Variance.NEUTRAL}[
                        getattr(action, "codomain_variance", value.variance)], source=args[i].source)
            elif name == "NATIVE_RESPONSE":
                if value.grade is None or value.variance != "chain":
                    raise ValueError("native response requires a declared chain field grade")
                calculus = values[2] if len(values) > 2 else None
                if calculus is not None:
                    if not isinstance(calculus, NativeFieldCalculus):
                        raise TypeError("native response requires a declared field calculus")
                    if calculus.source is not None and calculus.source is not context.binding.value:
                        raise ValueError("field calculus belongs to another selected source")
                    calculus._grade(value.grade)
                    if value.space != calculus.complex.spaces[value.grade]:
                        raise ValueError("native response requires its named grade coordinates")
                result = result.with_(grade=value.grade, coordinates=args[0].coordinates,
                    tensor_axes=args[0].tensor_axes, shape=ShapeRef(value.values.shape),
                    variance=args[0].variance, source=args[0].source)
        if name == "NATIVE_RESPONSE" and len(values) > 1 and values[1] is not None:
            if _fraction(values[1]) < 0:
                raise ValueError("Green parameter must be nonnegative")
    if name in {"SPAN_FIELD", "SPAN_DELTA"}:
        from rexgraph.attachment_field import AttachmentField
        if isinstance(values[0], AttachmentField):
            _bound(values[0].source, context.binding)
    if name == "TENSOR_MOMENTS":
        from rexgraph.tensor_moment import TensorMomentKernel
        kernel = values[0]
        if isinstance(kernel, TensorMomentKernel):
            for side in values[1:]:
                if side is None:
                    continue
                fields = (side,)*len(kernel.channels) if isinstance(side, TensorField) else tuple(side)
                if len(fields) != len(kernel.channels) or any(not isinstance(f, TensorField) for f in fields):
                    raise TypeError("moment inputs must be explicit tensor fields for each channel")
                if any(f.space != a.domain for f,a in zip(fields,kernel.channels,strict=True)):
                    raise ValueError("each moment channel requires its named input coordinates")
    if name == "TENSOR_SCALAR" and isinstance(values[0], TensorField):
        if values[0].axes or len(values[0].space.keys) != 1:
            raise ValueError("scalar output requires one coordinate and no retained field axes")
    if name == "FIELD_PAIR" and all(v is not None for v in values):
        from rexgraph.tensor_moment import MomentSpan
        MomentSpan(*values)
    if name == "TENSOR_DELTA" and all(v is not None for v in values):
        from rexgraph.temporal_field import TensorEvolution, ResolvedEvolution, NativeFieldEvolution
        evolution, old, new = values
        _bound(old.source, context.binding)
        if isinstance(evolution, TensorEvolution):
            evolution.validate(old, new)
        elif isinstance(evolution, ResolvedEvolution):
            evolution._injection_evolution().validate(old, new)
        elif isinstance(evolution, NativeFieldEvolution):
            evolution.validate(old, new)
    return result, [PredicateResult("tensor_coordinates", "verified", "declared exact types and named axes"),
                    PredicateResult("tensor_action", "deferred", "retained field and support evaluation")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    v = ValueKind
    kinds = {
        "TENSOR_APPLY": (v.COORDINATE_MAP, v.TENSOR_FIELD),
        "NATIVE_RESPONSE": (v.TENSOR_FIELD, (int, Fraction), v.NATIVE_FIELD_CALCULUS),
        "TENSOR_MOMENTS": (v.TENSOR_KERNEL, None, None),
        "FIELD_PAIR": (v.TENSOR_FIELD, v.TENSOR_FIELD, v.TENSOR_PAIRING),
        "MOMENT_PAIR": (v.TENSOR_MOMENTS, str, str),
        "MOMENT_SUPPORT": (v.MOMENT_SPAN,), "MOMENT_CONTRACT": (v.MOMENT_SPAN,),
        "TENSOR_SCALAR": (v.TENSOR_FIELD,),
        "TENSOR_DELTA": (v.TENSOR_EVOLUTION, v.TENSOR_FIELD, v.TENSOR_FIELD),
        "CHANNEL_FIELD": (v.TENSOR_CHANNELS, str), "CHANNEL_TOTAL": (v.TENSOR_CHANNELS, None),
        "ATTACHMENTS": (None, None),
        "SPAN_FIELD": (v.ATTACHMENT_FIELD, str, bool, str, None, None, None),
        "SPAN_DELTA": (v.ATTACHMENT_FIELD, v.ATTACHMENT_FIELD, None, None, str, bool, str),
        "SECTOR_FIELDS": (v.SECTOR_TRANSPORT, v.TENSOR_FIELD, v.TENSOR_FIELD),
        "MOMENT_CHANGE": (v.TENSOR_FIELD, v.TENSOR_FIELD, v.COORDINATE_MAP, v.TENSOR_FIELD,
                          v.TENSOR_FIELD, v.COORDINATE_MAP, v.TENSOR_PAIRING, v.TENSOR_PAIRING),
    }
    outputs = {
        "TENSOR_APPLY": v.TENSOR_FIELD, "NATIVE_RESPONSE": v.TENSOR_FIELD, "TENSOR_MOMENTS": v.TENSOR_MOMENTS,
        "FIELD_PAIR": v.MOMENT_SPAN, "MOMENT_PAIR": v.MOMENT_SPAN, "MOMENT_SUPPORT": v.TENSOR_FIELD,
        "MOMENT_CONTRACT": v.TENSOR_FIELD, "TENSOR_SCALAR": v.EXACT_RATIONAL, "TENSOR_DELTA": v.TENSOR_CHANNELS,
        "CHANNEL_FIELD": v.TENSOR_FIELD, "CHANNEL_TOTAL": v.TENSOR_FIELD, "ATTACHMENTS": v.ATTACHMENT_FIELD,
        "SPAN_FIELD": v.TENSOR_FIELD, "SPAN_DELTA": v.TENSOR_CHANNELS, "SECTOR_FIELDS": v.TENSOR_CHANNELS,
        "MOMENT_CHANGE": v.MOMENT_CHANGE,
    }
    for name, (labels, defaults) in ARGUMENTS.items():
        inputs = []
        for i, (label, kind) in enumerate(zip(labels, kinds[name], strict=True)):
            optional = i >= len(labels) - len(defaults)
            if isinstance(kind, ValueKind):
                inputs.append(TypePattern(label, kind=kind, literal=type(None) if optional else None, optional=optional))
            else:
                inputs.append(TypePattern(label, literal=kind, optional=optional))
        register(OperatorSignature(name=name, source_kind=v.REX, inputs=tuple(inputs),
                 result=lambda args, kind=outputs[name]: _result(kind),
                 implementation_key="rexgraph.retained_tensor."+name.lower(), memoizable=True,
                 preconditions=("named coordinates and exact coefficients", "no implicit field or support contraction")))
