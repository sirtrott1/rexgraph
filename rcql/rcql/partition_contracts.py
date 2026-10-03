"""Explicit source selections, not inferred faces or clustering policies."""
from dataclasses import replace

from .types import Domain, Exactness, PredicateResult, RCType, ValueKind

ARGUMENTS = {"FACES": (("support",), ()),
             "GLUE_PARTITIONS": (("parts", "carried_state"), ("structural",)),
             "TRAINING_PARTITION": (("policy",), ()),
             "RESTRICT": (("selection", "closure", "carried_state"), ("subcomplex", "structural")),
             "QUOTIENT": (("selection", "closure"), ("subcomplex",)),
             "PARTITION": (("selection", "closure", "carried_state"), ("subcomplex", "structural"))}


def partition_result():
    return RCType("RexPartition", kind=ValueKind.REX_PARTITION,
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def faces_result(args):
    cell = args[0]
    return cell.with_(name="CellSet", kind=ValueKind.CELL_SET, grade=2,
                      basis=replace(cell.basis, grade=2))


def refine(typed, context):
    from rexgraph.partition_state import partition_tower
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native Rex source")
    if typed.operator == "GLUE_PARTITIONS":
        parts = typed.args[0]
        if not parts or any(not isinstance(part, RCType) or part.kind is not ValueKind.REX_PARTITION
                            or part.source != typed.binding.ref for part in parts):
            raise ValueError("GLUE_PARTITIONS requires nonempty partitions bound to the original source")
        if len(typed.args) > 1 and typed.args[1] not in {"structural", "all"}:
            raise ValueError("partition carried_state must be structural or all")
        partition_tower(context.binding.value)
        return [PredicateResult("partition_union", "verified",
            "bound original source; each lineage and carried state verified during execution; exact downward closure")]
    if typed.operator == "TRAINING_PARTITION":
        from rexgraph.partition_state import partition_policy
        partition_policy(context.binding.value, typed.args[0])
        return [PredicateResult("training_selection", "verified",
            "explicit state bound cell policy and raw exact chain; identities retained; no model fitting")]
    selected = typed.args[0]
    pattern = selected.kind is ValueKind.CELL_PATTERN
    if selected.source != typed.binding.ref or not pattern and selected.basis.ordering != "canonical":
        raise ValueError("partition selection requires the bound canonical cell basis")
    if not pattern:
        context.grade(selected.grade)
    if len(typed.args) > 1 and typed.args[1] != "subcomplex":
        raise ValueError("partition closure must be subcomplex")
    if len(typed.args) > 2 and typed.args[2] not in {"structural", "all"}:
        raise ValueError("partition carried_state must be structural or all")
    partition_tower(context.binding.value)
    return [PredicateResult("raw_exact_chain", "verified",
        "original C1 shares and integer upper boundaries compose to exact zero; no face filtering"),
        PredicateResult("partition_selection", "verified",
        "declared cell selection; closure retains full relation slots and all lower grades")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    from collections.abc import Mapping
    register(OperatorSignature(name="GLUE_PARTITIONS", source_kind=ValueKind.REX,
        inputs=(TypePattern("parts", literal=(tuple, list)),
                TypePattern("carried_state", literal=str, optional=True)), result=partition_result(),
        implementation_key="rexgraph.selection.glue", requires=frozenset({"read", "identity"}),
        memoizable=True, preconditions=("nonempty restrictions of the bound original source",
            "each exact restriction and carried state must reproduce before union",
            "original source is authority for overlap; modified or foreign parts refused")))
    register(OperatorSignature(name="TRAINING_PARTITION", source_kind=ValueKind.REX,
        inputs=(TypePattern("policy", literal=Mapping),), result=partition_result(),
        implementation_key="rexgraph.partition_state.partition_from_policy",
        requires=frozenset({"read", "identity", "train"}), memoizable=True,
        preconditions=("explicit policy mapping bound to the current source state",
                       "full downward closure; identities and metrics retained; application state is explicit",
                       "selection only, not deidentification, a disjoint split or model fitting")))
    selection = dict(kind=(ValueKind.CELL, ValueKind.CELL_SET), source_bound=True, basis_bound=True)
    register(OperatorSignature(name="FACES", source_kind=ValueKind.REX,
        inputs=(TypePattern("support", grade=1, **selection),), result=faces_result,
        implementation_key="rexgraph.partition_state.faces_in_support", memoizable=True,
        preconditions=("stored C2 cells with nonempty boundary wholly contained in the C1 selection",
                       "no inferred filling or pairwise expansion")))
    for name, kind in (("RESTRICT", ValueKind.REX), ("PARTITION", ValueKind.REX_PARTITION),
                       ("QUOTIENT", ValueKind.RECORD)):
        register(OperatorSignature(name=name, source_kind=ValueKind.REX,
            inputs=(TypePattern("selection", kind=(ValueKind.CELL, ValueKind.CELL_SET, ValueKind.CELL_PATTERN),
                                source_bound=True),
                    TypePattern("closure", literal=str, optional=True)) + (() if name == "QUOTIENT" else
                    (TypePattern("carried_state", literal=str, optional=True),)),
            result=RCType("RelativeQuotient" if name == "QUOTIENT" else kind.value,
                          kind=kind, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
            implementation_key="rexgraph.relative_quotient.relative_quotient" if name == "QUOTIENT" else
                               "rexgraph.partition_state.build_rex_partition",
            requires=frozenset({"read", "identity"}), memoizable=True,
            preconditions=(("relative coordinate tower and certified projection, not a renormalized Rex",
                            "complete original boundary coefficients; exact P B = Bbar P; ranks lazy",
                            "materialize projection before supplying it as an explicit graded map parameter")
                           if name == "QUOTIENT" else ("one explicit selection, not automatic clustering",
                           "owned downward closed result; structural default or explicit carried_state=all",
                           "rebind the result Rex before querying its new cell basis"))))
