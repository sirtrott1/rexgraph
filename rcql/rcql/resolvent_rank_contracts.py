"""The resolvent rank, the relational replacement for PageRank at any grade."""
from fractions import Fraction

from .types import BasisRef, Domain, Exactness, PredicateResult, RCType, ShapeRef, ValueKind, Variance

ARGUMENTS = {"RESOLVENT_RANK": (("grade", "seed", "damping", "metric", "calculus", "metrics"),
                                (None, Fraction(17, 20), "declared", None, None))}

# Mirrors rexgraph.resolvent_ranking, which this stdlib contract layer does not import.
POLICIES = ("declared", "walk", "degree", "completed")


def refine(typed, children, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native Rex source")
    args = (*typed.args, *ARGUMENTS["RESOLVENT_RANK"][1][len(typed.args) - 1:])
    grade, seed, damping, metric = args[:4]
    calculus = context.field_calculus(children[4], typed.operator) if len(children) > 4 else None
    size = calculus.complex.sizes[calculus._grade(grade)] if calculus is not None else context.grade(grade)
    if not size:
        raise ValueError(f"{typed.operator} requires a nonempty grade")
    if metric not in POLICIES:
        raise ValueError(f"{typed.operator} metric must be one of {', '.join(POLICIES)}")
    from rexgraph.ranking_response import _damping
    _damping(damping)
    basis = BasisRef(context.binding.ref.name, grade)
    if isinstance(seed, RCType) and (seed.grade != grade or seed.basis != basis or seed.shape.dims != (size,)
                                     or seed.variance not in (Variance.CHAIN, Variance.COCHAIN)):
        raise TypeError(f"{typed.operator} seed requires one canonical vector at grade {grade}")
    result = RCType("Chain", grade=grade, kind=ValueKind.CHAIN, variance=Variance.CHAIN, domain=Domain.RATIONAL,
                    exactness=Exactness.RATIONAL, source=context.binding.ref, basis=basis, shape=ShapeRef((size,)))
    return result, [context.homology(), PredicateResult("resolvent_rank", "deferred",
        "exact Green action (I + lam L_k)^-1 over Q in the selected metrics; mass and sign are reported")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    register(OperatorSignature(name="RESOLVENT_RANK", source_kind=ValueKind.REX,
        inputs=(TypePattern("grade", literal=int),
                TypePattern("seed", kind=(ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.FIELD), literal=type(None),
                            source_bound=True, basis_bound=True, domain=(Domain.INTEGER, Domain.RATIONAL), optional=True),
                TypePattern("damping", literal=(int, Fraction), optional=True),
                TypePattern("metric", literal=str, optional=True),
                TypePattern("calculus", kind=ValueKind.NATIVE_FIELD_CALCULUS, literal=type(None), optional=True),
                TypePattern("metrics", literal=(tuple, list, type(None)), optional=True)),
        result=RCType("Chain", kind=ValueKind.CHAIN, variance=Variance.CHAIN, domain=Domain.RATIONAL,
                      exactness=Exactness.RATIONAL),
        implementation_key="rexgraph.resolvent_ranking.resolvent_rank", memoizable=True,
        preconditions=("full exact chain condition and positive metrics at every grade",
                       "nonnegative seed of positive mass, normalized; uniform when omitted",
                       "rational damping in [0, 1), lam = damping / (1 - damping)",
                       "metric 'walk' at grade 0 on pair relations gives exactly personalized PageRank")))
