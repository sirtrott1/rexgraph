"""Typed adapters for core primary factorization and local incidence views."""
from .execution_trace import record_method


def column_expansion(source, boundary):
    from rexgraph.column_expansion import ColumnExpansion
    if boundary.source is not source:
        raise ValueError("COLUMN_EXPANSION requires its bound source")
    result = ColumnExpansion(boundary)
    record_method("exact-primary-column-expansion", grade=result.grade,
                  legs=result.legs.shape[1], expansion_digest=result.coefficient_digest)
    return result


def primary_lift(source, legs, lift):
    from rexgraph.column_expansion import primary_lift as reconstruct
    if legs.source is not source or lift.source is not source:
        raise ValueError("PRIMARY_LIFT requires its bound source")
    result = reconstruct(legs, lift)
    record_method("exact-primary-lift-handle", expansion_digest=legs.expansion.coefficient_digest)
    return result


def hyperslice(source, cell):
    from rexgraph.cell_neighborhood import hyperslice as neighborhood
    from .operators import _typed_cells
    result = neighborhood(_typed_cells(source, cell, operator="HYPERSLICE"))
    record_method("native-graded-incidence-neighborhood", grade=result.grade)
    return result


ADAPTERS = {"COLUMN_EXPANSION": column_expansion, "PRIMARY_LIFT": primary_lift, "HYPERSLICE": hyperslice}


def install(register):
    for name, function in ADAPTERS.items():
        register(name)(function)
