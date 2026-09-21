"""Declarations for operation names, recursion and exact feedback."""
from .types import RCType, ValueKind, Exactness

ARGUMENTS = {
    "NAME": (("operator", "alias"), (None,)),
    "NAME_BIND": (("operation", "argument", "value"), ()),
    "NAME_REBIND": (("operation", "argument", "value"), ()),
    "NAME_ALIAS": (("operation", "name"), ()),
    "NAME_PORT": (("operation", "old", "new"), ()),
    "NAME_CHAIN": (("left", "right", "argument", "name"), (None,)),
    "NAME_MODIFY": (("base", "modifier", "argument", "name"), (None,)),
    "NAME_APPLY": (("operation", "arguments"), ()),
    "NAME_EXPLAIN": (("operation", "arguments"), ()),
    "NAME_RECORD": (("operation",), ()),
    "NAME_READ": (("record",), (None,)),
    "NAME_TOPOLOGY": (("operation",), ()),
    "NAME_ITERATE": (("operation", "initial", "steps", "argument", "parameters", "limits"), ((), None)),
    "RECURSIVE_RUN": (("program", "entry", "arguments", "limits", "history", "memoize"), (None, True, True)),
    "RECURSIVE_EXPLAIN": (("program", "entry", "arguments"), ()),
    "RECURSIVE_VALUE": (("result",), ()),
    "RECURSIVE_FIELDS": (("result", "definition", "order", "port"), (None, "completion", None)),
    "RECURSIVE_TRACE": (("result",), ()),
    "RECURSIVE_HISTORY": (("result",), ()),
    "RECURSIVE_RECORD": (("program",), ()),
    "RECURSIVE_READ": (("record",), (None,)),
    "RECURSIVE_TOPOLOGY": (("program",), ()),
    "RECURSIVE_RESULT_RECORD": (("result",), ()),
    "RECURSIVE_RESULT_READ": (("record", "contributors"), ((),)),
    "FEEDBACK_RECORD": (("system",), ()),
    "FEEDBACK_READ": (("record", "contributors"), ((),)),
    "FEEDBACK_COMPLETE": (("system",), ()),
    "FEEDBACK_SELECT": (("system", "variable"), ()),
    "FEEDBACK_ITERATE": (("system", "initial", "steps"), ()),
}


def literal_type(binding, value):
    from .name_relation import NameRelation
    from .recursive_program import RecursiveProgram, RecursionResult
    from rexgraph.affine_feedback import AffineFeedback
    if isinstance(value, NameRelation):
        return RCType("NameRelation", kind=ValueKind.NAME_RELATION, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest, name_declaration=value.to_bytes())
    if isinstance(value, RecursiveProgram):
        return RCType("RecursiveProgram", kind=ValueKind.RECURSIVE_PROGRAM, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest, recursive_declaration=value.to_bytes())
    if isinstance(value, RecursionResult):
        from .program_contracts import result_types
        return RCType("RecursionResult", kind=ValueKind.RECURSION_RESULT, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest, program_outputs=result_types(binding, (value.value,)))
    if isinstance(value, AffineFeedback):
        return RCType("AffineFeedback", kind=ValueKind.AFFINE_FEEDBACK, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest)
    return None


def install(register, OperatorSignature, TypePattern):
    v = ValueKind
    n, r, o = v.NAME_RELATION, v.RECURSIVE_PROGRAM, v.RECURSION_RESULT
    table = {
        "NAME": ((str, str), n), "NAME_REBIND": ((n, str, None), n), "NAME_BIND": ((n, str, None), n), "NAME_ALIAS": ((n, str), n),
        "NAME_PORT": ((n, str, str), n), "NAME_CHAIN": ((n, n, None, str), n), "NAME_MODIFY": ((n, n, None, str), n),
        "NAME_APPLY": ((n, None), v.UNKNOWN), "NAME_EXPLAIN": ((n, None), v.RECORD),
        "NAME_RECORD": ((n,), v.REX), "NAME_READ": ((None,), n), "NAME_TOPOLOGY": ((n,), v.REX),
        "NAME_ITERATE": ((n, None, int, str, None, None), o),
        "RECURSIVE_RUN": ((r, str, None, None, bool, bool), o), "RECURSIVE_EXPLAIN": ((r, str, None), v.RECORD),
        "RECURSIVE_FIELDS": ((o, str, str, str), v.TENSOR_CHANNELS),
        "RECURSIVE_TRACE": ((o,), v.REX),
        "RECURSIVE_VALUE": ((o,), v.UNKNOWN), "RECURSIVE_HISTORY": ((o,), v.SEQUENCE),
        "RECURSIVE_RECORD": ((r,), v.REX), "RECURSIVE_READ": ((None,), r), "RECURSIVE_TOPOLOGY": ((r,), v.REX),
        "RECURSIVE_RESULT_RECORD": ((o,), v.REX), "RECURSIVE_RESULT_READ": ((None, None), o),
        "FEEDBACK_RECORD": ((v.AFFINE_FEEDBACK,), v.REX),
        "FEEDBACK_READ": ((None, None), v.AFFINE_FEEDBACK),
        "FEEDBACK_COMPLETE": ((v.AFFINE_FEEDBACK,), v.SECTION_FAMILY),
        "FEEDBACK_SELECT": ((v.AFFINE_FEEDBACK, str), v.COORDINATE_MAP),
        "FEEDBACK_ITERATE": ((v.AFFINE_FEEDBACK, v.TENSOR_FIELD, int), v.TENSOR_CHANNELS),
    }
    for name, (labels, defaults) in ARGUMENTS.items():
        kinds, output = table[name]
        patterns = tuple(TypePattern(label, kind=kind, optional=i >= len(labels)-len(defaults))
                         if isinstance(kind, ValueKind) else
                         TypePattern(label, literal=kind, optional=i >= len(labels)-len(defaults))
                         for i, (label, kind) in enumerate(zip(labels, kinds, strict=True)))
        result = RCType(output.value, kind=output,
                        exactness=Exactness.RATIONAL if name in {"FEEDBACK_ITERATE", "RECURSIVE_FIELDS"} else Exactness.STRUCTURAL)
        register(OperatorSignature(name, v.REX, patterns, result,
                 "rcql.recursion_operators."+name.lower(), memoizable=True))


def refine(typed, children, context):
    from .name_relation import NameRelation
    from .recursive_program import RecursiveProgram, RecursionResult
    from .ast import Literal, ListExpr
    from .execution_trace import capture_methods
    unknown = object()
    def known(child):
        if isinstance(child.expr, Literal):
            return child.expr.value if not isinstance(child.expr.value, RCType) else unknown
        value = context.known_value(child)
        if value is not None:
            return value
        if isinstance(child.expr, ListExpr):
            return tuple(v.result for v in child.children)
        return unknown
    values = tuple(known(c) for c in children)
    labels, defaults = ARGUMENTS[typed.operator]
    if len(values) < len(labels):
        values += defaults[-(len(labels)-len(values)):]
    result = typed.result
    operation = values[0] if values else unknown
    if not isinstance(operation, (NameRelation, RecursiveProgram)) and typed.args:
        decl = typed.args[0]
        if isinstance(decl, RCType):
            if decl.name_declaration:
                operation = NameRelation.from_bytes(decl.name_declaration)
            elif decl.recursive_declaration:
                operation = RecursiveProgram.from_bytes(decl.recursive_declaration)
    name = typed.operator
    produced = None
    if name == "NAME" and all(v is not unknown for v in values):
        produced = NameRelation.operator(values[0], name=values[1])
    elif isinstance(operation, NameRelation):
        if name in {"NAME_APPLY", "NAME_EXPLAIN"} and values[1] is not unknown:
            plan = operation.explain(context.binding, values[1])
            if name == "NAME_APPLY":
                from .program_contracts import result_types
                result = result_types(context.binding, (plan.returns[0].result,))[0]
        elif name in {"NAME_BIND", "NAME_REBIND"} and all(v is not unknown for v in values[1:]):
            method = operation.bind if name == "NAME_BIND" else operation.rebind
            produced = method(values[1], values[2])
        elif name == "NAME_ALIAS" and values[1] is not unknown:
            produced = operation.named(values[1])
        elif name == "NAME_PORT" and all(v is not unknown for v in values[1:]):
            produced = operation.rename(values[1], values[2])
        elif name in {"NAME_CHAIN", "NAME_MODIFY"}:
            right = values[1]
            if not isinstance(right, NameRelation) and getattr(typed.args[1], "name_declaration", None):
                right = NameRelation.from_bytes(typed.args[1].name_declaration)
            if isinstance(right, NameRelation) and all(v is not unknown for v in values[2:]):
                produced = operation.then(right, values[2], name=values[3], modifier=name == "NAME_MODIFY")
        elif name == "NAME_ITERATE" and all(v is not unknown for v in values[1:]):
            if type(values[2]) is not int or values[2] < 0:
                raise ValueError("iteration requires a nonnegative horizon")
            supplied = dict(values[4])
            if values[3] in supplied:
                raise ValueError("iteration state port was supplied twice")
            plan = operation.explain(context.binding, {**supplied, values[3]: values[1]})
            result = result.with_(program_outputs=(plan.returns[0].result,))
    elif isinstance(operation, RecursiveProgram) and name in {"RECURSIVE_RUN", "RECURSIVE_EXPLAIN"}:
        if all(v is not unknown for v in values[1:3]):
            with capture_methods(binding=context.binding):
                plan = operation.explain(context.binding, values[1], values[2])
            if name == "RECURSIVE_RUN":
                from .relation_runtime import RecursionLimits
                if values[3] is not unknown:
                    RecursionLimits.from_value(values[3])
                result = result.with_(program_outputs=(plan["result_type"],))
    if produced is not None:
        result = literal_type(context.binding, produced)
    if name in {"RECURSIVE_VALUE", "RECURSIVE_FIELDS", "RECURSIVE_HISTORY", "RECURSIVE_TRACE", "RECURSIVE_RESULT_RECORD"}:
        if isinstance(values[0], RecursionResult):
            from .recursion_operators import _result_source
            with capture_methods(binding=context.binding):
                _result_source(context.binding.value, values[0])
        outputs = getattr(typed.args[0], "program_outputs", None)
        if outputs and name == "RECURSIVE_VALUE":
            result = outputs[0]
    if name.startswith("FEEDBACK_") and name != "FEEDBACK_READ" and values[0] is not unknown:
        from .recursion_operators import _feedback_source
        with capture_methods(binding=context.binding):
            _feedback_source(context.binding.value, values[0])
        if name == "FEEDBACK_SELECT" and values[1] is not unknown:
            from .planning import _carrier_literal
            result = _carrier_literal(context.binding, values[0].selection(values[1]))
    return result, ()
