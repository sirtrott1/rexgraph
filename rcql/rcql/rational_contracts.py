"""Rational skew action contracts, distinct from spectral plane diagnostics."""
from dataclasses import replace
from fractions import Fraction

from .types import Domain, Exactness, PredicateResult, ShapeRef, ValueKind

ARGUMENTS = {"CAYLEY": (("generator", "parameter", "tol", "maxiter"), (1e-10, 1000)),
             "COMPLEX_STRUCTURE": (("generator", "scale"), (None,)),
             "RATIONAL_ROTATION": (("structure", "a", "b", "c"), ()),
             "RESOLVENT_GROUP": (("operators", "parameters", "word", "tol", "maxiter"), ((), 1e-10, 1000))}


def result(args, name):
    if name == "RESOLVENT_GROUP":
        operators = args[0]
        from .types import RCType
        from rexgraph.rational_operator import resolvent_word
        scales, word = resolvent_word(args[1], args[2] if len(args) > 2 else ())
        if not operators or len(operators) != len(scales):
            raise ValueError("one parameter is required per resolvent generator")
        primal = operators[0]
        for op in operators:
            if not isinstance(op, RCType) or op.kind is not ValueKind.OPERATOR:
                raise TypeError("resolvent group requires typed operators")
            desc = op.operator
            if (desc is None or desc.domain != desc.codomain or desc.shape[0] != desc.shape[1]
                    or desc.domain.ordering != "canonical" or not desc.exact_action or not desc.exact_transpose
                    or desc.symmetric is not True or desc.psd is not True):
                raise TypeError("resolvent generators require canonical Euclidean PSD square Q actions and transposes")
            if (op.source != primal.source or desc.domain != primal.operator.domain
                    or desc.shape != primal.operator.shape or desc.action_variance != primal.operator.action_variance):
                raise ValueError("resolvent generators require one source, grade, shape and variance")
        tol, maxiter = args[3] if len(args) > 3 else 1e-10, args[4] if len(args) > 4 else 1000
        if not 0 < tol < 1 or maxiter <= 0:
            raise ValueError("resolvent group requires 0<tol<1 and maxiter>0")
        desc = replace(primal.operator, construction="rational-operator", coefficient_domain=Domain.RATIONAL,
            arithmetic=Exactness.APPROXIMATE, symmetric=None, psd=None, metric_self_adjoint=None,
            metric_skew_adjoint=None, metric_psd=None, euclidean_skew_adjoint=None,
            operands=tuple(op.operator for op in operators), parameters=(("kind", name), ("scales", scales),
            ("word", word), ("tol", tol), ("maxiter", maxiter)))
        return primal.with_(name="ResolventGroup", operator=desc, shape=ShapeRef(desc.shape))
    primal = args[0]
    desc = primal.operator
    if (desc is None or desc.domain != desc.codomain or desc.shape[0] != desc.shape[1]
            or desc.domain.ordering != "canonical" or not desc.exact_action or not desc.exact_transpose):
        raise TypeError("rational transform requires a canonical square Q action and transpose")
    if name == "RATIONAL_ROTATION":
        if desc.construction != "rational-operator" or dict(desc.parameters).get("kind") != "COMPLEX_STRUCTURE":
            raise TypeError("rotation requires a certified COMPLEX_STRUCTURE")
        from rexgraph.rational_operator import rotation_triple
        rotation_triple(*args[1:])
        parameters = (("a", args[1]), ("b", args[2]), ("c", args[3]))
    else:
        if desc.euclidean_skew_adjoint is not True:
            raise TypeError("generator requires a native Euclidean skew adjoint certificate")
        from rexgraph.graded_metric import _fraction
        if name == "CAYLEY":
            _fraction(args[1])
            tol, maxiter = args[2] if len(args) > 2 else 1e-10, args[3] if len(args) > 3 else 1000
            if not 0 < tol < 1 or maxiter <= 0:
                raise ValueError("Cayley requires 0<tol<1 and maxiter>0")
            parameters = (("parameter", args[1]), ("tol", tol), ("maxiter", maxiter))
        else:
            scale = args[1] if len(args) > 1 else None
            if scale is not None and _fraction(scale) <= 0:
                raise ValueError("complex structure scale must be positive")
            parameters = (("scale", scale),)
    desc = replace(desc, construction="rational-operator", coefficient_domain=Domain.RATIONAL,
                   arithmetic=Exactness.APPROXIMATE, symmetric=None, psd=None,
                   metric_self_adjoint=None, metric_skew_adjoint=None, metric_psd=None,
                   euclidean_skew_adjoint=True if name == "COMPLEX_STRUCTURE" else None,
                   operands=(primal.operator,), parameters=(("kind", name), *parameters))
    return primal.with_(name="RationalOperator", operator=desc, shape=ShapeRef(desc.shape))


def refine(typed, context):
    if not context.native:
        raise TypeError("rational operators require a native Rex source")
    if typed.operator == "RESOLVENT_GROUP":
        if any(op.source != typed.binding.ref for op in typed.args[0]):
            raise ValueError("resolvent generators require the bound source")
        return [PredicateResult("resolvent_group", "declared",
            "finite words in PSD resolvents and their inverses; no commutation claim; exact Q or native numerical solves"),
            PredicateResult("group_state", "deferred", "Core checks source state at execution; EXPLAIN performs no solve")]
    return [PredicateResult("rational_transform", "deferred",
        "Core checks source state and exact normalization at construction; EXPLAIN runs no action or solver"),
        PredicateResult("solve_policy", "declared",
        "Cayley uses exact finite Q directions or the native numerical resolvent; no eigenbasis or dense inverse")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    action = TypePattern("generator", kind=ValueKind.OPERATOR, source_bound=True, basis_bound=True)
    rational = TypePattern("rational", literal=(int, Fraction))
    patterns = {"CAYLEY": (action, rational, TypePattern("tol", literal=(float, Fraction), optional=True),
                            TypePattern("maxiter", literal=int, optional=True)),
                "COMPLEX_STRUCTURE": (action, TypePattern("scale", literal=(int, Fraction, type(None)), optional=True)),
                "RATIONAL_ROTATION": (action, rational, rational, rational),
                "RESOLVENT_GROUP": (TypePattern("operators", literal=(tuple, list)),
                    TypePattern("parameters", literal=(tuple, list)),
                    TypePattern("word", literal=(tuple, list), optional=True),
                    TypePattern("tol", literal=(float, Fraction), optional=True),
                    TypePattern("maxiter", literal=int, optional=True))}
    for name in ARGUMENTS:
        register(OperatorSignature(name=name, source_kind=ValueKind.REX, inputs=patterns[name],
            result=lambda args, name=name: result(args, name), memoizable=True,
            implementation_key="rexgraph.rational_operator." + ("ResolventGroup" if name == "RESOLVENT_GROUP" else name.lower()),
            preconditions=("nonempty compatible Euclidean PSD Q generators, nonnegative rational parameters",
                           "signed word indices start at one; products apply from right to left; empty word is identity")
                if name == "RESOLVENT_GROUP" else ("one canonical grade and Euclidean skew certificate; exact Q primal action",
                           "complex normalization certifies a single frequency square with rational root; no spectral plane selection",
                           "rotation keeps the kernel fixed; a*a+b*b=c*c with c>0")))
