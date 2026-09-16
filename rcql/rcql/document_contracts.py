"""Explicit C0 seeds, primary document fields and stored section partitions."""
from dataclasses import replace

from .types import BasisRef, Domain, Exactness, OperatorDescriptor, PredicateResult, RCType, ShapeRef, ValueKind, Variance

ARGUMENTS = {"TEXT_OVERLAP_VIEW": ((), ()),
             "DOCUMENT_FIELD": (("seeds", "reading", "seed_weight", "exact"), ("mass", "invdeg", True)),
             "SECTION_RESPONSE": (("layer", "seeds", "reading", "seed_weight", "exact"), ("mass", "invdeg", True)),
             "SEMANTIC_CLOSURE": (("seed", "max_depth", "grade"), (8, 0))}


def options(args, offset):
    tail = args[offset:]
    return tuple(tail) + ("mass", "invdeg", True)[len(tail):]


def section_layer(rex, layer):
    from rexgraph.sectioning import sectionings_of
    if not isinstance(layer, str) or not layer:
        raise TypeError("section response requires a stored layer name")
    store = sectionings_of(rex)
    if layer not in store:
        raise ValueError(f"source has no section layer {layer!r}")
    return store[layer].resolved(store)


def overlap_descriptor(source, n):
    basis = BasisRef(source.name, 1)
    return OperatorDescriptor("text-overlap", basis, basis, (n, n), Domain.RATIONAL,
        Exactness.APPROXIMATE, metric="participation", symmetric=True, psd=None,
        parameters=(("participation", "abs(B1) W"), ("diagonal", "removed")),
        transpose_available=True, exact_action=True, exact_transpose=True)


def response_result(args, section=False):
    reading, weight, exact = options(args, 2 if section else 1)
    if reading not in {"mass", "coverage"} or weight not in {"flat", "invdeg"}:
        raise ValueError("document reading must be mass or coverage, with flat or invdeg seeding")
    return RCType("SectionResponse" if section else "Cochain",
        kind=ValueKind.RECORD if section else ValueKind.COCHAIN,
        grade=None if section else 1, variance=None if section else Variance.COCHAIN,
        domain=Domain.RATIONAL if exact else Domain.REAL,
        exactness=Exactness.STRUCTURAL if section else Exactness.RATIONAL if exact else Exactness.APPROXIMATE)


def refine(typed, children, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native Rex source")
    args, rex = typed.args, context.binding.value
    if typed.operator == "TEXT_OVERLAP_VIEW":
        from rexgraph.markov import validate_markov_source
        validate_markov_source(rex)
        desc = overlap_descriptor(context.binding.ref, int(rex.nE))
        return typed.result.with_(operator=desc, shape=ShapeRef(desc.shape)), [
            PredicateResult("primary_overlap", "verified",
                "unsigned weighted primary C1 Gram without self overlap; no sentence graph or implicit section aggregation")]
    if typed.operator in {"CLOSURE", "SEMANTIC_CLOSURE"}:
        from rexgraph.tower import validate_closure
        validate_closure(rex, args[0], args[1] if len(args) > 1 else 8, args[2] if len(args) > 2 else 0)
        return typed.result, [PredicateResult("structural_closure", "verified",
            "present C0 seed, positive bound and full raw chain law; no expansion or rank executed")]
    from rexgraph.partition import response_seeds, response_owner
    index = int(typed.operator == "SECTION_RESPONSE")
    seeds = args[index]
    _, weight, _ = options(args, index+1)
    if isinstance(seeds, RCType):
        if seeds.source != typed.binding.ref or seeds.grade != 0 or seeds.basis.ordering != "canonical":
            raise ValueError("document seeds require canonical C0 cells of the bound source")
    else:
        response_seeds(rex, seeds, weight)
    result = typed.result
    if index:
        response_owner(rex, section_layer(rex, args[0]))
    else:
        result = result.with_(shape=ShapeRef((context.grade(1),)))
    return result, [PredicateResult("query_coordinates", "verified",
        "C0 seed set and source C1 axis; coalesced original shares and explicit incidence degree weighting"),
        PredicateResult("section_partition", "verified" if index else "not-applicable",
        "stored disjoint C1 layer; no text matching, corpus selection or inferred owner")]


def install(register):
    from .signatures import OperatorSignature, TypePattern, lookup
    seeds = TypePattern("seeds", kind=(ValueKind.CELL, ValueKind.CELL_SET), grade=0,
                        literal=(list, tuple), source_bound=True, basis_bound=True)
    optional = (TypePattern("reading", literal=str, optional=True),
                TypePattern("seed_weight", literal=str, optional=True),
                TypePattern("exact", literal=bool, optional=True))
    register(OperatorSignature(name="TEXT_OVERLAP_VIEW", source_kind=ValueKind.REX,
        inputs=(), result=RCType("TextOverlapView", kind=ValueKind.OPERATOR, grade=1,
                                 domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
        memoizable=True, implementation_key="rexgraph.text_overlap.TextOverlapView",
        preconditions=("original C1 relation axis; X=abs(B1) W with nonnegative relation metrics",
                       "symmetric off diagonal Gram action, not generally PSD or stochastic")))
    register(OperatorSignature(name="DOCUMENT_FIELD", source_kind=ValueKind.REX,
        inputs=(seeds, *optional), result=response_result, memoizable=True,
        implementation_key="rexgraph.partition.document_field",
        preconditions=("explicit C0 seed set, not raw text or automatic term alignment",
                       "C1 mass or coverage on original rational primary columns; source metrics not applied")))
    register(OperatorSignature(name="SECTION_RESPONSE", source_kind=ValueKind.REX,
        inputs=(TypePattern("layer", literal=str), seeds, *optional),
        result=lambda args: response_result(args, section=True), memoizable=True,
        implementation_key="rexgraph.partition.section_response", requires=frozenset({"read", "identity"}),
        preconditions=("one stored disjoint C1 section layer, including resolved coarsenings",
                       "exact rational scores or one rounding of each finished score; no channel profile inference")))
    register(replace(lookup("CLOSURE"), name="SEMANTIC_CLOSURE"))
