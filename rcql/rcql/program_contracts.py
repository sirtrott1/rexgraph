"""Contracts for explicit tensor axes and section readout certificates."""
from .types import RCType, ValueKind, Exactness, Domain, ShapeRef, CoordinateDescriptor

ARGUMENTS = {
    "TENSOR_SELECT": (("field", "axis", "key"), ()),
    "TENSOR_CONTRACT": (("field", "axis", "weights"), ()),
    "TENSOR_DIAGONAL": (("field", "first", "second", "name"), ()),
    "SAMPLE_RATES": (("field",), ()),
    "SECTION_EQUIVALENT": (("family", "left", "right"), ()),
    "PROGRAM_READ": (("record",), (None,)),
    "PROGRAM_RECORD": (("program",), ()),
    "PROGRAM_RUN": (("program", "sources", "parameters"), ()),
    "PROGRAM_EXPLAIN": (("program", "sources", "parameters"), ()),
    "PROGRAM_OUTPUT": (("result", "index"), ()),
    "PROGRAM_TOPOLOGY": (("program", "sources", "parameters"), ()),
    "PROGRAM_ASSEMBLY": (("name", "fragments", "coefficients", "inputs", "links", "outputs"), ((), (), ())),
    "PROGRAM_GLUE": (("assembly", "system", "observation", "observed"), (None, None)),
    "PROGRAM_FAMILY_OBSERVE": (("family",), ()),
    "PROGRAM_FAMILY_COMPILE": (("family", "sources", "parameters"), ()),
    "PROGRAM_FAMILY_RECORD": (("family",), ()),
    "PROGRAM_FAMILY_READ": (("record", "contributors"), ((),)),
    "PROGRAM_ASSEMBLY_RECORD": (("assembly",), ()),
    "PROGRAM_ASSEMBLY_READ": (("record",), ()),
    "PROGRAM_COMPARE": (("old", "new", "matches"), ()),
    "PROGRAM_EVOLUTION_INFO": (("evolution",), ()),
    "PROGRAM_BOUNDARY_CHANGE": (("evolution", "grade"), ()),
    "PROGRAM_EVOLUTION_RECORD": (("evolution",), ()),
    "PROGRAM_EVOLUTION_READ": (("record",), ()),
}


def refine(typed, children, context):
    name = typed.operator
    values = tuple(context.known_value(c) for c in children)
    result = typed.result
    if name.startswith("PROGRAM_"):
        from .program_family import ProgramFamily
        from .section_operators import authorize_family, authorize_system
        if values and isinstance(values[0], ProgramFamily):
            authorize_family(values[0].family, context.binding)
            if name == "PROGRAM_FAMILY_COMPILE" and all(v is not None for v in values):
                from .execution_trace import capture_methods
                from .planning import _carrier_literal
                with capture_methods(binding=context.binding):
                    result = _carrier_literal(context.binding, values[0].compile(*values[1:]))
        if name == "PROGRAM_GLUE":
            from rexgraph.section_calculus import SectionSystem
            if isinstance(values[1], SectionSystem):
                authorize_system(values[1], context.binding)
        if values and isinstance(children[0].result, RCType) and children[0].result.program_declaration is not None:
            from .program import Program
            values = (Program.from_bytes(children[0].result.program_declaration), *values[1:])
        if name in {"PROGRAM_RUN", "PROGRAM_EXPLAIN", "PROGRAM_TOPOLOGY"} and all(v is not None for v in values):
            from .program_operators import _program_executor
            from .execution_trace import capture_methods
            with capture_methods(binding=context.binding):
                execution = _program_executor(values[1], values[2])
            explanation = execution.execute_program(values[0], explain=True)
            if name == "PROGRAM_RUN":
                result = result.with_(program_outputs=explanation["output_types"])
        elif name == "PROGRAM_OUTPUT":
            outputs = getattr(typed.args[0], "program_outputs", None)
            if outputs is not None and values[1] is not None:
                index = values[1]
                if type(index) is not int or not 0 <= index < len(outputs):
                    raise ValueError("program output index is outside its declaration")
                result = outputs[index]
        return result, ()
    if name == "SECTION_EQUIVALENT":
        from rexgraph.section_calculus import SectionFamily
        from .section_operators import authorize_family
        if isinstance(values[0], SectionFamily):
            authorize_family(values[0], context.binding)
        return result, ()
    from rexgraph.tensor_field import TensorField
    field = values[0]
    if isinstance(field, TensorField):
        field.check_state()
    argument = typed.args[0]
    axes = argument.tensor_axes if isinstance(argument, RCType) else None
    if name in {"TENSOR_SELECT", "TENSOR_CONTRACT"} and axes is not None and values[1] is not None:
        selected = next((a for a in axes if a.name == values[1]), None)
        if selected is None:
            raise ValueError("requested axis is not a retained field axis")
        if name == "TENSOR_SELECT" and values[2] is not None and values[2] not in selected.keys:
            raise ValueError("requested coordinate is not present in the axis")
        if name == "TENSOR_CONTRACT" and values[2] is not None:
            from rexgraph.graded_metric import _fraction
            weights = tuple(_fraction(w) for w in values[2])
            if len(weights) != len(selected.keys):
                raise ValueError("contraction requires one weight per coordinate")
        axes = tuple(a for a in axes if a.name != selected.name)
        result = result.with_(tensor_axes=axes, coordinates=argument.coordinates,
                              source=argument.source, grade=argument.grade, variance=argument.variance,
                              shape=None if argument.coordinates is None else
                              ShapeRef((len(argument.coordinates.keys), *(len(a.keys) for a in axes))))
    elif name == "TENSOR_DIAGONAL" and axes is not None and all(v is not None for v in values[1:]):
        first, second, label = values[1:]
        names = tuple(a.name for a in axes)
        if first == second or first not in names or second not in names:
            raise ValueError("diagonal selection requires two distinct retained axes")
        a, b = axes[names.index(first)], axes[names.index(second)]
        if a.keys != b.keys:
            raise ValueError("diagonal selection requires equal ordered keys")
        if not isinstance(label, str) or not label or label in set(names) - {first, second}:
            raise ValueError("diagonal axis name must be distinct")
        axes = (CoordinateDescriptor(label, a.keys), *(v for v in axes if v.name not in {first, second}))
        result = result.with_(tensor_axes=axes, coordinates=argument.coordinates,
                              source=argument.source, grade=argument.grade, variance=argument.variance)
    return result, ()


def install(register, OperatorSignature, TypePattern):
    v = ValueKind
    for name, (labels, defaults) in ARGUMENTS.items():
        if name.startswith("PROGRAM_"):
            descriptions = {
                "PROGRAM_READ": ((None,), v.PROGRAM),
                "PROGRAM_RECORD": ((v.PROGRAM,), v.REX),
                "PROGRAM_RUN": ((v.PROGRAM, None, None), v.PROGRAM_RESULT),
                "PROGRAM_EXPLAIN": ((v.PROGRAM, None, None), v.RECORD),
                "PROGRAM_OUTPUT": ((v.PROGRAM_RESULT, int), v.UNKNOWN),
                "PROGRAM_TOPOLOGY": ((v.PROGRAM, None, None), v.REX),
                "PROGRAM_ASSEMBLY": ((str, None, None, None, None, None), v.PROGRAM_ASSEMBLY),
                "PROGRAM_GLUE": ((v.PROGRAM_ASSEMBLY, v.SECTION_SYSTEM, None, None), v.PROGRAM_FAMILY),
                "PROGRAM_FAMILY_OBSERVE": ((v.PROGRAM_FAMILY,), v.SECTION_IMAGE),
                "PROGRAM_FAMILY_COMPILE": ((v.PROGRAM_FAMILY, None, None), v.PROGRAM),
                "PROGRAM_FAMILY_RECORD": ((v.PROGRAM_FAMILY,), v.REX),
                "PROGRAM_FAMILY_READ": ((None, None), v.PROGRAM_FAMILY),
                "PROGRAM_ASSEMBLY_RECORD": ((v.PROGRAM_ASSEMBLY,), v.REX),
                "PROGRAM_ASSEMBLY_READ": ((None,), v.PROGRAM_ASSEMBLY),
                "PROGRAM_COMPARE": ((None, None, None), v.PROGRAM_EVOLUTION),
                "PROGRAM_EVOLUTION_INFO": ((v.PROGRAM_EVOLUTION,), v.RECORD),
                "PROGRAM_BOUNDARY_CHANGE": ((v.PROGRAM_EVOLUTION, int), v.COORDINATE_MAP),
                "PROGRAM_EVOLUTION_RECORD": ((v.PROGRAM_EVOLUTION,), v.REX),
                "PROGRAM_EVOLUTION_READ": ((None,), v.PROGRAM_EVOLUTION),
            }
            kinds, output = descriptions[name]
            result = RCType(output.value, kind=output, exactness=Exactness.STRUCTURAL)
        elif name == "SECTION_EQUIVALENT":
            kinds = (v.SECTION_FAMILY, v.COORDINATE_MAP, v.COORDINATE_MAP)
            result = RCType("ReadoutEquivalence", kind=v.RECORD, exactness=Exactness.STRUCTURAL)
        else:
            kinds = (v.TENSOR_FIELD, *(str for _ in labels[1:]))
            if name == "TENSOR_CONTRACT":
                kinds = (v.TENSOR_FIELD, str, None)
            result = RCType("TensorField", kind=v.TENSOR_FIELD, exactness=Exactness.RATIONAL, domain=Domain.RATIONAL)
        patterns = tuple(TypePattern(label, kind=kind, optional=i >= len(labels)-len(defaults))
                         if isinstance(kind, ValueKind) else
                         TypePattern(label, literal=kind, optional=i >= len(labels)-len(defaults))
                         for i, (label, kind) in enumerate(zip(labels, kinds, strict=True)))
        register(OperatorSignature(name, v.REX, patterns, result,
                 "rcql.program_operators." + name.lower(),
                 memoizable=name not in {"PROGRAM_RUN", "PROGRAM_READ"}))


def result_types(binding, values):
    from .planning import _carrier_literal
    from .types import INTEGER, RATIONAL, BOOLEAN, UNKNOWN
    from fractions import Fraction
    output = []
    for value in values:
        typed = value if isinstance(value, RCType) else _carrier_literal(binding, value)
        if not isinstance(typed, RCType):
            typed = BOOLEAN if type(value) is bool else INTEGER if type(value) is int else RATIONAL if isinstance(value, Fraction) else UNKNOWN
        output.append(typed)
    return tuple(output)
