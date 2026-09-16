"""Static contracts for exact adjugates and explicit chain homotopy witnesses."""
from __future__ import annotations

from dataclasses import replace

from .types import Domain, Exactness, PredicateResult, RCType, ValueKind


ARGUMENTS = {"ADJUGATE": (("operator",), ()), "HOMOTOPY": (("left", "right", "witness"), ())}


def adjugate_result(args):
    value = args[0]
    desc = value.operator
    if (desc is None or desc.domain != desc.codomain or desc.shape[0] != desc.shape[1]
            or not desc.exact_action or desc.coefficient_domain is Domain.COMPLEX
            or desc.domain.ordering != "canonical"):
        raise TypeError("ADJUGATE requires a canonical square operator with a certified Q action")
    n = desc.shape[0]
    return value.with_(name="AdjugateOperator", operator=replace(desc, construction="exact-adjugate",
        coefficient_domain=Domain.RATIONAL, arithmetic=Exactness.APPROXIMATE, psd=False,
        primal_operator=desc, operands=(), metric_self_adjoint=None, metric_skew_adjoint=None,
        euclidean_skew_adjoint=None, parameters=(("coefficient_method", "exact-streamed-power-traces"),
            ("coefficient_actions", None if n is None else n * max(0, n-1)))))


def homotopy_result(args):
    from .signatures import _chain_map_result
    left, right, witness = args
    _chain_map_result((left,))
    _chain_map_result((right,))
    a, b = left.graded_map, right.graded_map
    if (left.source != right.source or left.temporal != right.temporal or a.domain != b.domain
            or a.codomain != b.codomain or a.domain_digest != b.domain_digest
            or a.codomain_digest != b.codomain_digest):
        raise ValueError("HOMOTOPY requires the same full ordered endpoint complexes and states")
    if len(witness) != len(a.shapes):
        raise ValueError("HOMOTOPY requires every witness grade, including the empty top")
    return RCType("ChainHomotopy", kind=ValueKind.CHAIN_HOMOTOPY, domain=Domain.RATIONAL,
                  exactness=Exactness.STRUCTURAL, source=left.source, temporal=left.temporal)


def refine(typed, children, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native Rex source")
    if typed.operator == "ADJUGATE":
        return typed.result, []
    from rexgraph.chain_map import ChainHomotopy, ChainMap, GradedMap, _exact_entries
    from .validation import graded_map_descriptor
    from .ast import Call
    args = typed.args
    for value in args[:2]:
        desc = value.graded_map
        if value.source != context.binding.ref or value.temporal is not None and value.temporal != context.binding.temporal:
            raise ValueError("homotopy maps must belong to the bound source and time")
        if tuple(s[1] for s in desc.shapes) != tuple(context.sizes) or any(
            s.name != f"C{k}" or s.keys != tuple(map(str, range(n)))
            for k, (s, n) in enumerate(zip(desc.domain, context.sizes, strict=True))
        ):
            raise ValueError("homotopy maps require the full canonical source tower")
    desc = args[0].graded_map
    shapes = tuple((desc.shapes[k+1][0] if k+1 < len(desc.shapes) else 0, n)
                   for k, (_, n) in enumerate(desc.shapes))
    witness = tuple(_exact_entries(h, shape) for h, shape in zip(args[2], shapes, strict=True))

    def known_map(expression):
        while isinstance(expression.expr, Call) and expression.expr.name == "CHAIN_MAP":
            expression = expression.children[0]
        value = context.known_value(expression)
        return value.declaration if isinstance(value, ChainMap) else value

    left, right = map(known_map, children[:2])
    if all(isinstance(m, GradedMap) for m in (left, right)):
        for value, expected in zip((left, right), args[:2], strict=True):
            if value.domain.source is not context.binding.value or graded_map_descriptor(value) != replace(
                    expected.graded_map, chain_preserving=None):
                raise ValueError("homotopy descriptor does not match its supplied map")
        proof = ChainHomotopy(left, right, witness)
        facts = [PredicateResult("homotopy_certificate", "verified",
                 f"both chain maps verified; exact G-F-BH-HB residuals = {proof.residuals}")]
    else:
        facts = [PredicateResult("homotopy_certificate", "deferred",
                 "execution must verify both endpoint maps and G-F=BH+HB over Q at every grade")]
    facts.append(PredicateResult("homotopy_witness_axes", "verified",
                 f"H_k:C_k to D_(k+1), shapes={shapes}; explicit empty top"))
    return typed.result, facts


def install(register):
    from .signatures import OperatorSignature, TypePattern
    register(OperatorSignature(name="ADJUGATE", source_kind=ValueKind.REX,
        inputs=(TypePattern("exact operator", kind=ValueKind.OPERATOR, source_bound=True, basis_bound=True),),
        result=adjugate_result, implementation_key="rex.exact_adjugate", memoizable=True,
        preconditions=("square canonical operator with a certified rational action",
                       "global scalar polynomial: n(n-1) exact primal actions per application; no inverse or eigenbasis")))
    endpoint = TypePattern("graded map", kind=(ValueKind.GRADED_MAP, ValueKind.CHAIN_MAP),
                           source_bound=True, domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL)
    register(OperatorSignature(name="HOMOTOPY", source_kind=ValueKind.REX,
        inputs=(endpoint, endpoint, TypePattern("witness", literal=(list, tuple))),
        result=homotopy_result, implementation_key="rex.chain_homotopy.verify", memoizable=True,
        preconditions=("identical declared endpoint complexes; explicit Q witness H at every grade",
                       "both chain maps and all G-F=BH+HB equations must be verified exactly")))
