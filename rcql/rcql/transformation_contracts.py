"""Typed stages for finite certified name transformations."""
from .types import RCType, ValueKind, Exactness, PredicateResult

ARGUMENTS = {
    "TRANSFORM_NAME": (("operation", "rule", "arguments"), ()),
    "TRANSFORM_VERIFY": (("transformation", "operation"), ()),
    "TRANSFORM_COMPILE": (("transformation", "operation", "arguments"), ()),
    "TRANSFORM_RECORD": (("transformation",), ()),
    "TRANSFORM_READ": (("record",), (None,)),
    "TRANSFORM_TOPOLOGY": (("transformation",), ()),
    "TRANSFORM_SECTION": (("family", "left", "right"), ()),
    "TRANSFORM_SOURCE": (("transformation",), ()),
    "TRANSFORM_SPECIALIZE": (("operation", "bindings"), ()),
    "TRANSFORM_COMPOSE": (("operation", "following", "ports"), ()),
    "TRANSFORM_PROGRAM": (("program", "rule", "arguments"), ()),
    "TRANSFORM_PROGRAM_COMPILE": (("transformation", "program", "sources", "parameters"), ()),
    "TRANSFORM_PROGRAM_TOPOLOGY": (("transformation", "program", "sources", "parameters"), ()),
    "SECTION_CERTIFIED_OBSERVE": (("family", "left", "right", "certificate", "side"), ("right",)),
}


def literal_type(value):
    from .program_transformation import ProgramTransformation
    if isinstance(value, ProgramTransformation):
        return RCType("ProgramTransformation", kind=ValueKind.PROGRAM_TRANSFORMATION,
                      exactness=Exactness.STRUCTURAL, declaration_digest=value.coefficient_digest,
                      transformation_declaration=value.to_bytes())
    return None


def install(register, OperatorSignature, TypePattern):
    v = ValueKind
    t, n = v.PROGRAM_TRANSFORMATION, v.NAME_RELATION
    table = {"TRANSFORM_NAME": ((n, str, None), t),
             "TRANSFORM_VERIFY": ((t, (n, v.PROGRAM)), v.RECORD),
             "TRANSFORM_COMPILE": ((t, n, None), n),
             "TRANSFORM_RECORD": ((t,), v.REX),
             "TRANSFORM_READ": ((None,), t),
             "TRANSFORM_TOPOLOGY": ((t,), v.REX),
             "TRANSFORM_SOURCE": ((t,), v.UNKNOWN),
             "TRANSFORM_PROGRAM": ((v.PROGRAM, str, None), t),
             "TRANSFORM_PROGRAM_COMPILE": ((t, v.PROGRAM, None, None), v.PROGRAM),
             "TRANSFORM_PROGRAM_TOPOLOGY": ((t, v.PROGRAM, None, None), v.REX),
             "TRANSFORM_SPECIALIZE": ((n, None), t),
             "TRANSFORM_COMPOSE": ((n, n, None), t),
             "TRANSFORM_SECTION": ((v.SECTION_FAMILY, v.COORDINATE_MAP, v.COORDINATE_MAP), t),
             "SECTION_CERTIFIED_OBSERVE": ((v.SECTION_FAMILY, v.COORDINATE_MAP,
                                            v.COORDINATE_MAP, bytes, str), v.SECTION_IMAGE)}
    for name, (labels, defaults) in ARGUMENTS.items():
        kinds, output = table[name]
        inputs = tuple(TypePattern(label, kind=kind, optional=i >= len(labels)-len(defaults))
                       if isinstance(kind, (ValueKind, tuple)) else
                       TypePattern(label, literal=kind, optional=i >= len(labels)-len(defaults))
                       for i, (label, kind) in enumerate(zip(labels, kinds, strict=True)))
        register(OperatorSignature(name, v.REX, inputs,
            RCType(output.value, kind=output, exactness=Exactness.STRUCTURAL),
            "rcql.transformation_operators."+name.lower(), memoizable=True))


def refine(typed, children, context):
    from .ast import Literal, ListExpr
    from .name_relation import NameRelation
    from .program import Program
    from .program_transformation import ProgramTransformation
    from .recursion_contracts import literal_type as name_type
    unknown = object()

    def known(child):
        value = context.known_value(child)
        if value is not None and not isinstance(value, RCType):
            return value
        if isinstance(child.expr, Literal) and not isinstance(child.expr.value, RCType):
            return child.expr.value
        if isinstance(child.result, RCType):
            if child.result.transformation_declaration is not None:
                return ProgramTransformation.from_bytes(child.result.transformation_declaration)
            if child.result.name_declaration is not None:
                return NameRelation.from_bytes(child.result.name_declaration)
            if child.result.program_declaration is not None:
                return Program.from_bytes(child.result.program_declaration)
        if isinstance(child.expr, ListExpr):
            items = tuple(known(c) for c in child.children)
            return items if all(v is not unknown for v in items) else tuple(c.result for c in child.children)
        return unknown

    values = tuple(known(c) for c in children)
    result, facts = typed.result, []
    name = typed.operator
    if name == "TRANSFORM_PROGRAM" and all(v is not unknown for v in values):
        result = literal_type(ProgramTransformation.program(*values))
    elif name in {"TRANSFORM_SECTION", "SECTION_CERTIFIED_OBSERVE"}:
        from rexgraph.section_calculus import SectionFamily
        from .section_operators import authorize_family
        if isinstance(values[0], SectionFamily):
            authorize_family(values[0], context.binding)
        concrete = all(v is not unknown and not isinstance(v, RCType) for v in values)
        if concrete and name == "TRANSFORM_SECTION":
            result = literal_type(ProgramTransformation.section(*values))
        elif concrete:
            from .transformation_operators import validate_readout
            args = values if len(values) == 5 else (*values, "right")
            validate_readout(context.binding, *args)
        facts.append(PredicateResult("section_readout_equivalence", "verified" if concrete else "deferred",
                                     "exact offset and all free directions on the declared family"))
    elif name == "TRANSFORM_NAME" and all(v is not unknown for v in values):
        result = literal_type(ProgramTransformation.name(*values))
    elif name in {"TRANSFORM_SPECIALIZE", "TRANSFORM_COMPOSE"} and all(v is not unknown for v in values):
        constructor = ProgramTransformation.specialize if name == "TRANSFORM_SPECIALIZE" else ProgramTransformation.compose
        result = literal_type(constructor(*values))
    elif values and isinstance(values[0], ProgramTransformation):
        transform = values[0]
        transform.verify(values[1] if name in {"TRANSFORM_COMPILE", "TRANSFORM_VERIFY", "TRANSFORM_PROGRAM_COMPILE", "TRANSFORM_PROGRAM_TOPOLOGY"}
                         and values[1] is not unknown else None)
        facts.append(PredicateResult("operation_transformation_structure", "verified",
                                     "finite declaration correspondence; topology requires explicit source bindings"
                                     if transform.is_program else
                                     "declared construction and exact boundary checks; live readout claims remain guarded"))
        if name == "TRANSFORM_SOURCE":
            from .planning import _carrier_literal
            result = _carrier_literal(context.binding, transform.original())
        if name == "TRANSFORM_COMPILE" and all(v is not unknown for v in values):
            result = name_type(context.binding, transform.compile(context.binding, values[1], values[2]))
        if name in {"TRANSFORM_PROGRAM_COMPILE", "TRANSFORM_PROGRAM_TOPOLOGY"} and all(v is not unknown for v in values):
            from .execution_trace import capture_methods
            from .planning import _carrier_literal
            with capture_methods(binding=context.binding):
                candidate = transform.compile_program(*values[1:])
            if name == "TRANSFORM_PROGRAM_COMPILE":
                result = _carrier_literal(context.binding, candidate)
    return result, facts
