"""Adapters to the Core partition builder; no duplicate restriction algorithm."""
from .execution_trace import current_policy_digest, record_method


def faces(source, support):
    from rexgraph.partition_state import faces_in_support
    from .operators import _typed_cells
    result = faces_in_support(_typed_cells(source, support, operator="FACES"))
    record_method("native-stored-face-containment", grade=2, count=len(result.indices))
    return result


def partition(source, selection, closure="subcomplex", carried_state="structural"):
    from rexgraph.selection import Selection, restrict
    if not isinstance(selection, Selection):
        selection = Selection.from_cells(selection)
    result = restrict(source, selection, closure=closure,
                      policy_digest=current_policy_digest(), carried_state=carried_state)
    grades = tuple(grade for grade, indices in enumerate(selection.indices) if indices)
    record_method("native-exact-subcomplex-restriction", grades=grades,
                  chain_condition="exact-zero", selection_digest=result.state.selection_digest,
                  result_state=result.state.result_state, policy_digest=result.state.policy_digest,
                  carried_state=carried_state)
    return result


def restrict(source, selection, closure="subcomplex", carried_state="structural"):
    return partition(source, selection, closure, carried_state).rex


def training_partition(source, policy):
    from rexgraph.partition_state import partition_from_policy
    result = partition_from_policy(source, policy, authority_digest=current_policy_digest())
    record_method("native-exact-training-partition", chain_condition="exact-zero",
                  selection_digest=result.state.selection_digest,
                  result_state=result.state.result_state, policy_digest=result.state.policy_digest)
    return result


def glue_partitions(source, parts, carried_state="structural"):
    from rexgraph.selection import glue
    result = glue(parts, source=source, carried_state=carried_state,
                  policy_digest=current_policy_digest())
    record_method("native-exact-partition-union", chain_condition="exact-zero",
                  parents=result.lineage_parents, carried_state=carried_state,
                  result_state=result.state.result_state, policy_digest=result.state.policy_digest)
    return result


def quotient(source, selection, closure="subcomplex"):
    from rexgraph.relative_quotient import relative_quotient
    from rexgraph.cells import GradedCellPattern
    from rexgraph.selection import Selection
    from .operators import _typed_cells
    if closure != "subcomplex":
        raise ValueError("relative quotient closure must be subcomplex")
    if isinstance(selection, Selection):
        selection.check_state(source)
        selection = selection.as_pattern()
    if isinstance(selection, GradedCellPattern):
        if selection.source is not source:
            raise ValueError("QUOTIENT requires a pattern bound to its source Rex")
    else:
        selection = _typed_cells(source, selection, operator="QUOTIENT")
    result = relative_quotient(selection)
    record_method("core-exact-relative-quotient", sizes=result.sizes, residuals=result.residuals,
                  homology="lazy", normalization="original-coefficients")
    return result


def install(register):
    register("FACES")(faces)
    register("RESTRICT")(restrict)
    register("PARTITION")(partition)
    register("QUOTIENT")(quotient)
    register("TRAINING_PARTITION")(training_partition)
    register("GLUE_PARTITIONS")(glue_partitions)
