"""Homology dimension readings, not chosen harmonic representatives."""
from .types import Domain, Exactness, RCType, ValueKind

ARGUMENTS = {"SIMPLE_HOMOLOGY": (("grade",), ()), "MULTIPLICITY_HOMOLOGY": (("grade",), ())}


def refine(typed, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native Rex source")
    context.grade(typed.args[0])
    return [context.homology()]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    for name in ARGUMENTS:
        register(OperatorSignature(name=name, source_kind=ValueKind.REX,
            inputs=(TypePattern("grade", literal=int),),
            result=RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
                          exactness=Exactness.INTEGER), implementation_key="rex." + name.lower(),
            memoizable=True, preconditions=(
                "exact dimension over Q, not a harmonic basis or representative",
                "identify nonzero boundary columns equal up to sign at the requested grade only",
                "project the upper map into that quotient; zero columns remain distinct",
                "full exact chain condition; missing upper map is zero")))
