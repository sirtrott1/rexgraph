"""Adapters to the Core partition builder; no duplicate restriction algorithm."""
from .execution_trace import current_policy_digest, record_method


def faces(source, support):
    from rexgraph.io.partition_state import faces_in_support
    from .operators import _typed_cells
    result = faces_in_support(_typed_cells(source, support, operator="FACES"))
    record_method("native-stored-face-containment", grade=2, count=len(result.indices))
    return result


def partition(source, selection, closure="subcomplex"):
    import numpy as np
    from rexgraph.cells import Cell, CellSet
    from rexgraph.io.partition_state import build_rex_partition
    from .operators import _typed_cells
    selection = _typed_cells(source, selection, operator="PARTITION")
    grade = selection.grade
    indices = (selection.index,) if isinstance(selection, Cell) else selection.indices
    CellSet(source, grade, indices)
    # Population comes from the Core cell contract, not inferred incidence ranks.
    from rexgraph.cells import cell_count
    mask = np.zeros(cell_count(source, grade), dtype=np.uint8)
    mask[list(indices)] = 1
    edges = mask if grade == 1 else np.zeros(int(source.nE), dtype=np.uint8)
    result = build_rex_partition(source, edges,
        v_mask=mask if grade == 0 else None,
        grade_masks={grade: mask} if grade >= 2 else None,
        closure=closure, policy_digest=current_policy_digest())
    record_method("native-exact-subcomplex-restriction", grade=grade,
                  chain_condition="exact-zero", selection_digest=result.state.selection_digest,
                  result_state=result.state.result_state, policy_digest=result.state.policy_digest)
    return result


def restrict(source, selection, closure="subcomplex"):
    return partition(source, selection, closure).rex


def training_partition(source, policy):
    from rexgraph.io.partition_state import partition_from_policy
    result = partition_from_policy(source, policy, authority_digest=current_policy_digest())
    record_method("native-exact-training-partition", chain_condition="exact-zero",
                  selection_digest=result.state.selection_digest,
                  result_state=result.state.result_state, policy_digest=result.state.policy_digest)
    return result


def quotient(source, selection, closure="subcomplex"):
    from rexgraph.relative_quotient import relative_quotient
    from rexgraph.cells import GradedCellPattern
    from .operators import _typed_cells
    if closure != "subcomplex":
        raise ValueError("relative quotient closure must be subcomplex")
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
