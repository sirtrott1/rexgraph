"""Exact declared C2 attachment and the existing Core harmonic shadow reading."""
from .types import Domain, Exactness, PredicateResult, RCType, ValueKind, Variance

ARGUMENTS = {"FILL": (("cycle",), ()), "HARMONIC_SHADOW": ((), ()),
             "VOID": (("region",), ()),
             "VALIDATE_RELATIONS": (("candidates", "grade"), (2,))}


def refine(typed, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native RexGraph")
    if typed.operator == "VOID":
        value = typed.args[0]
        if value.source != typed.binding.ref or value.basis.ordering != "canonical":
            raise ValueError("VOID requires the bound canonical C1 cell basis")
        return [PredicateResult("triangle_region", "deferred",
            "pairwise region and full raw chain law checked at execution; no triangle enumeration or rank during EXPLAIN")]
    if typed.operator == "VALIDATE_RELATIONS":
        from rexgraph.relation_validation import _proposals
        _proposals(context.binding.value, typed.args[0], typed.args[1] if len(typed.args) > 1 else 2)
        return [PredicateResult("candidate_axes", "verified",
            "explicit sparse integer or rational columns on the bound canonical lower grade"),
                PredicateResult("candidate_chain", "deferred",
            "exact sparse composition and individual storage checks run only at execution")]
    if typed.operator == "FILL":
        from rexgraph.io.partition_state import partition_tower
        value = typed.args[0]
        if value.source != typed.binding.ref or value.basis.ordering != "canonical":
            raise ValueError("FILL requires the bound canonical C1 Chain basis")
        if value.shape is not None and len(value.shape.dims) != 1:
            raise ValueError("FILL requires a vector, not a block")
        partition_tower(context.binding.value)
    facts = [PredicateResult("core_reading", "verified",
        "native Core attachment or exact boundary ranks; no dense face matrix or eigenbasis")]
    if typed.operator == "FILL":
        facts.append(PredicateResult("filling_coefficients", "deferred",
            "nonzero integral cycle checked over the original Q boundary at execution; no normalization"))
    return facts


def install(register):
    from .signatures import OperatorSignature, TypePattern
    for name in ARGUMENTS:
        if name == "VOID":
            register(OperatorSignature(name=name, source_kind=ValueKind.REX,
                inputs=(TypePattern("region", kind=ValueKind.CELL_SET, grade=1, source_bound=True, basis_bound=True),),
                result=RCType("VoidState", kind=ValueKind.RECORD, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
                implementation_key="rexgraph.void_state.void_state", memoizable=True,
                preconditions=("explicit distinct participant pairwise C1 region, not a branching expansion",
                               "missing triangular faces; exact joint filling rank is lazy; no metric or eigenbasis")))
            continue
        if name == "VALIDATE_RELATIONS":
            register(OperatorSignature(name=name, source_kind=ValueKind.REX,
                inputs=(TypePattern("candidates", literal=(list, tuple)),
                        TypePattern("grade", literal=int, optional=True)),
                result=RCType("RelationValidation", kind=ValueKind.RECORD,
                              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
                implementation_key="rexgraph.relation_validation.validate_relations", memoizable=True,
                preconditions=("declared grade 2 or higher boundaries; source is the FROM binding",
                               "closure and exact integral storage checked separately; no inferred scale or attachment")))
            continue
        fill = name == "FILL"
        register(OperatorSignature(name=name, source_kind=ValueKind.REX,
            inputs=(TypePattern("cycle", kind=ValueKind.CHAIN, grade=1, variance=Variance.CHAIN,
                    source_bound=True, basis_bound=True),) if fill else (),
            result=RCType("Rex" if fill else "HarmonicShadow", kind=ValueKind.REX if fill else ValueKind.RECORD,
                          domain=Domain.METADATA, exactness=Exactness.STRUCTURAL if fill else Exactness.INTEGER),
            implementation_key="rexgraph.graph.RexGraph." + ("fill_cycle" if fill else "harmonic_shadow"),
            requires=frozenset({"read", "identity"} if fill else {"read"}), memoizable=True,
            preconditions=("FILL returns an owned Rex; rebind its basis before further queries",
                           "harmonic shadow counts Hodge eligible C2 boundaries, not selected cycle representatives")))
