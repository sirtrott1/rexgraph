"""Adapters to Core boundary and correspondence difference operations."""
from .execution_trace import record_method


def accession_delta(source, old, new, correspondence):
    from rexgraph.accession_delta import accession_delta as core_delta
    if old.source is not source:
        raise ValueError("old accession must belong to the bound source")
    result = core_delta(old, new, correspondence)
    record_method("core-exact-accession-difference", grade=result["grade"], formula=result["formula"],
                  entries=len(result["entries"]))
    return result


def diff(source, other, ref_labels=None, other_labels=None, matching="auto"):
    from rexgraph.boundary_difference import boundary_difference
    result = boundary_difference(source, other, ref_labels=ref_labels, other_labels=other_labels, matching=matching)
    record_method("core-exact-boundary-difference", matching=result.matching, entries=result.nnz)
    return result


def field_delta(source, field, correspondence):
    from rexgraph.field_delta import field_delta as core_delta
    from .operators import _typed_value
    field = _typed_value(source, field, operator="FIELD_DELTA", variance="chain")
    result = core_delta(field, correspondence)
    record_method("core-exact-correspondence-defects", grade=field.grade, metrics="identity")
    return result


def field_delta_moment(source, field, correspondence):
    return _moment(source, field, correspondence, oriented=False)


def oriented_field_delta_moment(source, field, correspondence):
    return _moment(source, field, correspondence, oriented=True)


def _moment(source, field, correspondence, *, oriented):
    from rexgraph.field_delta import field_delta_moment as core_moment
    from .operators import _typed_value
    field = _typed_value(source, field, operator="FIELD_DELTA_MOMENT", variance="chain")
    result = core_moment(field, correspondence, oriented=oriented)
    record_method("core-exact-correspondence-defects", grade=field.grade, metrics="identity", oriented=oriented)
    return result


def install(register):
    register("ACCESSION_DELTA")(accession_delta)
    for name, fn in (("DIFF", diff), ("FIELD_DELTA", field_delta), ("FIELD_DELTA_MOMENT", field_delta_moment),
                     ("ORIENTED_FIELD_DELTA_MOMENT", oriented_field_delta_moment)):
        register(name)(fn)
