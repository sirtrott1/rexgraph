"""Types and validation for exact section equations and observations."""
from .types import RCType, ValueKind, Domain, Exactness, PredicateResult, SourceRef

ARGUMENTS = {
    "SECTION_SYSTEM": (("section", "name", "stalk_spaces", "mediator_spaces"), ("section", None, None)),
    "SECTION_FIELD": (("system", "values", "axes"), (None, ())),
    "SECTION_RESIDUAL": (("system", "field"), ()),
    "SECTION_COMPLETE": (("system", "observation", "observed"), (None, None)),
    "SECTION_OBSERVE": (("family", "action"), ()),
    "SECTION_VALUE": (("image",), ()),
    "SECTION_RECIPE": (("system",), ()),
    "SECTION_RESTORE": (("recipe", "contributors"), ((),)),
    "SECTION_DELTA": (("old", "new", "input_map", "output_map", "old_field", "new_field"), ()),
}


def result_type(kind):
    return RCType(kind.value, kind=kind, domain=Domain.RATIONAL,
                  exactness=Exactness.RATIONAL if kind in {ValueKind.TENSOR_FIELD, ValueKind.TENSOR_CHANNELS} else Exactness.STRUCTURAL)


def section_literal(binding, value):
    from rexgraph.section_calculus import SectionSystem, SectionRecipe, SectionFamily, SectionImage
    kinds = ((SectionSystem, ValueKind.SECTION_SYSTEM), (SectionRecipe, ValueKind.SECTION_RECIPE),
             (SectionFamily, ValueKind.SECTION_FAMILY), (SectionImage, ValueKind.SECTION_IMAGE))
    for cls, kind in kinds:
        if isinstance(value, cls):
            reference = (value.source if isinstance(value, (SectionRecipe, SectionSystem))
                         else value.particular.source)
            source = binding.ref if reference is not None and reference.source is binding.value else SourceRef("foreign")
            return result_type(kind).with_(source=source, declaration_digest=value.coefficient_digest)
    return None


def refine(typed, children, context):
    if not context.native:
        raise TypeError("section queries require their selected native source")
    from rexgraph.section_calculus import SectionSystem, SectionFamily, SectionImage
    from rexgraph.coordinate_map import CoordinateMap, _is_action
    from rexgraph.tensor_field import TensorField
    from rexgraph.sheaf import ExactSheaf
    name = typed.operator
    values = tuple(context.known_value(c) for c in children)
    labels, defaults = ARGUMENTS[name]
    missing = len(labels) - len(values)
    if missing:
        values += defaults[-missing:]
    first = values[0]
    if isinstance(first, SectionSystem):
        from .section_operators import authorize_system
        authorize_system(first, context.binding)
    elif isinstance(first, (SectionFamily, SectionImage)) and name != "SECTION_RESTORE":
        from .section_operators import authorize_family
        authorize_family(first.family if isinstance(first, SectionImage) else first, context.binding)
    elif isinstance(first, ExactSheaf):
        if first.rex is not context.binding.value:
            raise ValueError("section belongs to another native source")
        first.check_state()
        if hasattr(first, "_require_section"):
            first._require_section()
        if name == "SECTION_SYSTEM" and (not isinstance(values[1], str) or not values[1]):
            raise ValueError("section system requires a name")
    if name == "SECTION_RESIDUAL" and isinstance(first, SectionSystem) and isinstance(values[1], TensorField):
        first.check_field(values[1])
    if name == "SECTION_COMPLETE" and isinstance(first, SectionSystem):
        observation, observed = values[1:3]
        if isinstance(observation, CoordinateMap):
            if observation.domain != first.space:
                raise ValueError("section observation has the wrong domain")
            if isinstance(observed, TensorField) and observed.space != observation.codomain:
                raise ValueError("section observation and field coordinates disagree")
    if name == "SECTION_OBSERVE" and isinstance(first, SectionFamily) and _is_action(values[1]):
        if values[1].domain != first.particular.space:
            raise ValueError("section observation has the wrong domain")
    if name == "SECTION_DELTA" and isinstance(first, SectionSystem) and isinstance(values[1], SectionSystem):
        from .section_operators import authorize_destination
        new = values[1]
        authorize_destination(new, context.binding)
        for action, domain, codomain in ((values[2], first.space, new.space),
                                         (values[3], first.residual_space, new.residual_space)):
            if isinstance(action, CoordinateMap) and (action.domain != domain or action.codomain != codomain):
                raise ValueError("temporal section correspondence coordinates disagree")
        for system, tensor in ((first, values[4]), (new, values[5])):
            if isinstance(tensor, TensorField):
                system.check_field(tensor)
    if name == "SECTION_RESTORE":
        from rexgraph.section_calculus import SectionRecipe
        if isinstance(first, (SectionRecipe, SectionFamily)):
            from .section_operators import validate_restore
            validate_restore(first, values[1], context.binding)
    return typed.result.with_(source=context.binding.ref), [
        PredicateResult("section_source", "verified", "declared source and local spaces"),
        PredicateResult("section_equations", "deferred", "exact equations and complete affine family")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    v = ValueKind
    kinds = {
        "SECTION_SYSTEM": (v.EXACT_SHEAF, str, None, None),
        "SECTION_FIELD": (v.SECTION_SYSTEM, None, None),
        "SECTION_RESIDUAL": (v.SECTION_SYSTEM, v.TENSOR_FIELD),
        "SECTION_COMPLETE": (v.SECTION_SYSTEM, v.COORDINATE_MAP, v.TENSOR_FIELD),
        "SECTION_OBSERVE": (v.SECTION_FAMILY, v.COORDINATE_MAP),
        "SECTION_VALUE": (v.SECTION_IMAGE,),
        "SECTION_RECIPE": (v.SECTION_SYSTEM,),
        "SECTION_RESTORE": ((v.SECTION_RECIPE, v.SECTION_FAMILY), None),
        "SECTION_DELTA": (v.SECTION_SYSTEM, v.SECTION_SYSTEM, v.COORDINATE_MAP,
                          v.COORDINATE_MAP, v.TENSOR_FIELD, v.TENSOR_FIELD),
    }
    outputs = {"SECTION_SYSTEM": v.SECTION_SYSTEM, "SECTION_FIELD": v.TENSOR_FIELD,
               "SECTION_RESIDUAL": v.TENSOR_FIELD, "SECTION_COMPLETE": v.SECTION_FAMILY,
               "SECTION_OBSERVE": v.SECTION_IMAGE, "SECTION_VALUE": v.TENSOR_FIELD,
               "SECTION_RECIPE": v.SECTION_RECIPE, "SECTION_RESTORE": v.SECTION_SYSTEM,
               "SECTION_DELTA": v.TENSOR_CHANNELS}
    for name, (labels, defaults) in ARGUMENTS.items():
        inputs = []
        for i, (label, kind) in enumerate(zip(labels, kinds[name], strict=True)):
            optional = i >= len(labels) - len(defaults)
            if isinstance(kind, ValueKind) or isinstance(kind, tuple):
                inputs.append(TypePattern(label, kind=kind, optional=optional,
                                          literal=type(None) if optional else None))
            else:
                inputs.append(TypePattern(label, literal=kind, optional=optional))
        register(OperatorSignature(name=name, source_kind=v.REX, inputs=tuple(inputs),
            result=lambda args, kind=outputs[name], name=name: result_type(
                v.SECTION_FAMILY if name == "SECTION_RESTORE" and args[0].kind is v.SECTION_FAMILY else kind),
            implementation_key="rexgraph.section_calculus." + name.lower(),
            memoizable=True,
            preconditions=("declared exact incidence maps and native source identities",
                           "observations constrain an affine family without statistical selection")))
