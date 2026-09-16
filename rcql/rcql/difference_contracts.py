"""Boundary differences and exact temporal correspondence defects."""
from .types import Domain, Exactness, PredicateResult, RCType, ValueKind, Variance

ARGUMENTS = {"DIFF": (("other", "ref_labels", "other_labels", "matching"), (None, None, "auto")),
             "ACCESSION_DELTA": (("old", "new", "correspondence"), ()),
             "FIELD_DELTA": (("field", "correspondence"), ()),
             "FIELD_DELTA_MOMENT": (("field", "correspondence"), ()),
             "ORIENTED_FIELD_DELTA_MOMENT": (("field", "correspondence"), ())}


def refine(typed, children, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native RexGraph source")
    if typed.operator == "ACCESSION_DELTA":
        old, new, mapping = typed.args
        if old.source != typed.binding.ref or mapping.source != typed.binding.ref:
            raise ValueError("old accession and correspondence require the bound source")
        if old.grade != new.grade or old.temporal != mapping.temporal:
            raise ValueError("accession delta requires matching grades and source states")
        for value in (old, new):
            if (value.domain not in {Domain.INTEGER, Domain.RATIONAL}
                    or value.basis is None or value.basis.ordering != "canonical"):
                raise TypeError("accession delta requires exact canonical measurements")
        from rexgraph.accession_delta import validate_accession_delta
        from rexgraph.type_accession import TypeAccession
        from rexgraph.chain_map import GradedMap, ChainMap
        values = tuple(context.known_value(child) for child in children)
        if not (all(isinstance(v, TypeAccession) for v in values[:2])
                and isinstance(values[2], (GradedMap, ChainMap))):
            raise TypeError("accession delta requires explicit measurement and correspondence declarations")
        validate_accession_delta(*values)
        return [PredicateResult("accession_correspondence", "deferred",
            "exact sparse accession defect; explicit ambient map or identical named output coordinates")]
    if typed.operator == "DIFF":
        from rexgraph.graph import RexGraph
        from rexgraph.boundary_difference import vertex_alignment
        args = (*typed.args, None, None, "auto")
        other = typed.args[0]
        if not (isinstance(other, RexGraph) or isinstance(other, RCType) and other.kind is ValueKind.REX):
            raise TypeError("DIFF requires another native RexGraph")
        matching = typed.args[3] if len(typed.args) > 3 else "auto"
        if matching not in {"auto", "identity", "support"}:
            raise ValueError("difference matching must be auto, identity or support")
        if isinstance(other, RexGraph):
            vertex_alignment(context.binding.value, other, args[1], args[2])
            a, b = context.binding.value.relation_ids, other.relation_ids
            if matching == "identity" and (a is None or b is None):
                raise ValueError("identity matching requires both relation ID axes")
            if matching == "auto" and (a is None) != (b is None):
                raise ValueError("one endpoint lacks relation IDs; choose support explicitly")
        return [PredicateResult("difference_coordinates", "verified" if isinstance(other, RexGraph) else "deferred",
            "complete vertex union, stable identities or explicit support multiset matching; exact original Q columns")]
    field, mapping = typed.args
    if field.source != typed.binding.ref or mapping.source != typed.binding.ref:
        raise ValueError("field and correspondence require the bound source")
    if field.temporal != mapping.temporal:
        raise ValueError("field and correspondence require the same source state")
    if field.basis.ordering != "canonical" or field.domain not in {Domain.INTEGER, Domain.RATIONAL}:
        raise TypeError("field delta requires canonical exact integer or rational Chain coefficients")
    desc = mapping.graded_map
    if desc is None or tuple(s[1] for s in desc.shapes) != tuple(context.sizes):
        raise ValueError("correspondence requires the complete source grade tower")
    context.grade(field.grade)
    from rexgraph.chain_map import GradedMap, ChainMap
    from rexgraph.field_delta import validate_correspondence
    known = context.known_value(children[1])
    if isinstance(known, (GradedMap, ChainMap)):
        validate_correspondence(known)
    return [PredicateResult("correspondence_defects", "deferred",
        "exact B'J-JB and transpose boundary defect actions; neither square is presumed to commute"),
        PredicateResult("endpoint_metrics", "verified", "identity coordinate metrics; no weighted adjoint is inferred")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    register(OperatorSignature(name="ACCESSION_DELTA", source_kind=ValueKind.REX,
        inputs=(TypePattern("old", kind=ValueKind.TYPE_ACCESSION, source_bound=True, basis_bound=True),
                TypePattern("new", kind=ValueKind.TYPE_ACCESSION),
                TypePattern("correspondence", kind=(ValueKind.GRADED_MAP, ValueKind.CHAIN_MAP), source_bound=True)),
        result=RCType("AccessionDelta", kind=ValueKind.RECORD, domain=Domain.METADATA,
                      exactness=Exactness.STRUCTURAL), memoizable=True,
        implementation_key="rexgraph.accession_delta.accession_delta",
        preconditions=("A_new J - J A_old for ambient endomorphisms at one grade",
                       "rectangular measurements require identical output coordinates and return A_new J - A_old",
                       "exact current source and target measurements; no inferred lineage or time division")))
    register(OperatorSignature(name="DIFF", source_kind=ValueKind.REX,
        inputs=(TypePattern("other"), TypePattern("ref_labels", literal=(list, tuple, type(None)), optional=True),
                TypePattern("other_labels", literal=(list, tuple, type(None)), optional=True),
                TypePattern("matching", literal=str, optional=True)),
        result=RCType("BoundaryDifference", kind=ValueKind.BOUNDARY_DIFFERENCE, domain=Domain.RATIONAL,
                      exactness=Exactness.STRUCTURAL), implementation_key="rexgraph.boundary_difference.boundary_difference",
        requires=frozenset({"read", "identity"}), memoizable=True,
        preconditions=("union coordinates are not the bound Rex cell basis; no canonical Rex reconstruction",
                       "C1 boundary change only; no metric, declared sign or higher grade change is inferred")))
    for name in ARGUMENTS.keys() - {"DIFF", "ACCESSION_DELTA"}:
        record = name == "FIELD_DELTA"
        register(OperatorSignature(name=name, source_kind=ValueKind.REX,
            inputs=(TypePattern("field", kind=ValueKind.CHAIN, variance=Variance.CHAIN,
                                source_bound=True, basis_bound=True),
                    TypePattern("correspondence", kind=(ValueKind.GRADED_MAP, ValueKind.CHAIN_MAP), source_bound=True)),
            result=RCType("FieldDelta" if record else "Rational", kind=ValueKind.RECORD if record else ValueKind.EXACT_RATIONAL,
                          domain=Domain.METADATA if record else Domain.RATIONAL,
                          exactness=Exactness.STRUCTURAL if record else Exactness.RATIONAL),
            implementation_key="rexgraph.field_delta." + ("field_delta" if record else "field_delta_moment"),
            memoizable=True, preconditions=("explicit exact correspondence, source Chain and identity endpoint metrics",
                "both endpoint chain laws hold; zero missing down or upper coordinate space; no time or lineage inference")))
