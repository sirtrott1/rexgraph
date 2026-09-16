"""Declared Euclidean chain symmetries, not inferred combinatorial groups."""
from .types import Domain, Exactness, PredicateResult, RCType, ValueKind

ARGUMENTS = {"SYMMETRY": (("generators", "word"), ((),))}


def result_type():
    return RCType("SymmetryGroup", kind=ValueKind.RECORD, domain=Domain.METADATA,
                  exactness=Exactness.STRUCTURAL)


def refine(typed, children, context):
    from rexgraph.rational_operator import generator_word
    from rexgraph.chain_map import symmetry_generators
    if not context.native:
        raise TypeError("SYMMETRY requires a native Rex source")
    generators = typed.args[0]
    if not generators or any(not isinstance(g, RCType) or
        g.kind not in {ValueKind.GRADED_MAP, ValueKind.CHAIN_MAP} or g.source != typed.binding.ref
        for g in generators):
        raise TypeError("SYMMETRY requires nonempty graded maps bound to the source")
    generator_word(len(generators), typed.args[1] if len(typed.args) > 1 else ())
    known = context.known_value(children[0])
    if known is None and children[0].children:
        candidates = tuple(context.known_value(child) for child in children[0].children)
        if all(value is not None for value in candidates):
            known = candidates
    if known is not None:
        symmetry_generators(known)
    return [PredicateResult("euclidean_chain_symmetries", "verified" if known is not None else "deferred",
        "exact full chain law, commuting boundary squares and U transpose U = I; selected product is not constructed"),
        PredicateResult("symmetry_scope", "declared",
        "generated rational Euclidean subgroup; no weighted metric, G channel or combinatorial automorphism claim")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    register(OperatorSignature(name="SYMMETRY", source_kind=ValueKind.REX,
        inputs=(TypePattern("generators", literal=(tuple, list)),
                TypePattern("word", literal=(tuple, list), optional=True)),
        result=result_type(), implementation_key="rexgraph.chain_map.SymmetryGroup", memoizable=True,
        preconditions=("explicit full rational graded endomorphisms on one identical declared source complex",
                       "Euclidean chain symmetries; not automatic discovery or preservation of arbitrary weighted metrics",
                       "materialize map and bind it as a parameter before using its graded map descriptor")))
