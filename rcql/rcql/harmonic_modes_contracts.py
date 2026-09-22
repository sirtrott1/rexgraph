"""Effective mode counts of the graded Hodge sectors and their harmonic log."""
from .types import Domain, Exactness, PredicateResult, RCType, ValueKind

ARGUMENTS = {"EFFECTIVE_MODES": (("grade", "sector", "weight", "calculus"), ("completed", "unit", None)),
             "HARMONIC_LOG": (("grade", "sector", "weight", "calculus"), ("completed", "unit", None))}

# Mirrors rexgraph.harmonic_modes, which this stdlib contract layer does not import.
SECTORS = ("completed", "hodge", "down", "up")
WEIGHTS = ("unit", "mean", "energy")


def refine(typed, children, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native Rex source")
    calculus = context.field_calculus(children[3], typed.operator) if len(children) > 3 else None
    if calculus is not None:
        calculus._grade(typed.args[0])
    else:
        context.grade(typed.args[0])
    sector = typed.args[1] if len(typed.args) > 1 else "completed"
    weight = typed.args[2] if len(typed.args) > 2 else "unit"
    if sector not in SECTORS:
        raise ValueError(f"{typed.operator} sector must be one of {', '.join(SECTORS)}")
    if weight not in WEIGHTS:
        raise ValueError(f"{typed.operator} weight must be one of {', '.join(WEIGHTS)}")
    if weight != "unit" and sector != "completed":
        raise ValueError(f"{typed.operator} weight applies only to the completed sector")
    return [context.homology(), PredicateResult("sector_traces", "verified",
        "declared grade metrics, sparse Gram traces over Q and the exact Betti number; "
        "no operator, harmonic frame or eigenvalue")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    inputs = (TypePattern("grade", literal=int), TypePattern("sector", literal=str, optional=True),
              TypePattern("weight", literal=str, optional=True),
              TypePattern("calculus", kind=ValueKind.NATIVE_FIELD_CALCULUS, literal=type(None), optional=True))
    preconditions = (
        "full exact chain condition, which makes the sector traces add",
        "declared grade metrics through the metric adjoint, from the source or an explicit field calculus",
        "the completed sector is L_k + w Pi^h with w the unit, the mean nonzero eigenvalue "
        "or the energy weighted mean eigenvalue")
    register(OperatorSignature(name="EFFECTIVE_MODES", source_kind=ValueKind.REX, inputs=inputs,
        result=RCType("Rational", kind=ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL,
                      exactness=Exactness.RATIONAL),
        implementation_key="rexgraph.harmonic_modes.effective_modes", memoizable=True,
        preconditions=preconditions))
    register(OperatorSignature(name="HARMONIC_LOG", source_kind=ValueKind.REX, inputs=inputs,
        result=RCType("Real", kind=ValueKind.REAL, domain=Domain.REAL,
                      exactness=Exactness.APPROXIMATE),
        implementation_key="rexgraph.harmonic_modes.harmonic_log", memoizable=True,
        preconditions=preconditions + ("one float logarithm of the exact count; a sector with no modes is refused",)))
