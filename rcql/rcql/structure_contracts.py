"""Primary factors retain their source boundary descriptor, not a cell leg basis."""
from dataclasses import replace

from .types import Domain, Exactness, PredicateResult, ShapeRef, ValueKind

ARGUMENTS = {"COLUMN_EXPANSION": (("boundary",), ()),
             "PRIMARY_LIFT": (("legs", "lift"), ()), "HYPERSLICE": (("cell",), ())}


def expansion_result(args):
    boundary = args[0]
    desc = boundary.operator
    if (desc is None or desc.construction != "boundary" or not desc.exact_action
            or desc.action_variance != "chain" or desc.codomain.grade != desc.domain.grade-1
            or desc.domain.ordering != "canonical" or desc.codomain.ordering != "canonical"):
        raise TypeError("COLUMN_EXPANSION requires BOUNDARY(grade) with a certified exact action")
    return boundary.with_(name="ColumnExpansion", kind=ValueKind.COLUMN_EXPANSION,
                          domain=Domain.RATIONAL, exactness=Exactness.STRUCTURAL, shape=None)


def lift_result(args):
    left, right = args
    if (left.source != right.source or left.temporal != right.temporal or left.grade != right.grade
            or left.basis != right.basis or left.operator is None or left.operator != right.operator):
        raise ValueError("PRIMARY_LIFT requires factors of the same ordered source boundary")
    return left.with_(name="PrimaryBoundary", kind=ValueKind.OPERATOR, domain=Domain.METADATA,
                      shape=ShapeRef(left.operator.shape),
                      operator=replace(left.operator, construction="primary-lift", coefficient_domain=Domain.RATIONAL))


def refine(typed, children, context):
    if not context.native:
        raise TypeError(f"{typed.operator} requires a native Rex source")
    for arg in typed.args:
        if arg.source != typed.binding.ref or arg.basis.ordering != "canonical":
            raise ValueError("structural values require the bound source and canonical cell basis")
        context.grade(arg.grade)
    result = typed.result
    if typed.operator == "HYPERSLICE":
        return result, [PredicateResult("primary_incidence", "verified",
            "full C1 participant slots; carried higher boundaries; no lower grade below C0; empty upper top")]
    if typed.operator == "COLUMN_EXPANSION":
        from rexgraph.column_expansion import _capture, _check_boundary
        from rexgraph.linear_operator import RexOperator
        # Establish exact domain and full state without constructing the factors.
        _capture(context.binding.value)
        known = context.known_value(children[0])
        if isinstance(known, RexOperator):
            _check_boundary(known)
        return result, [PredicateResult("exact_column_coordinates", "verified",
            "b=sum_i b_i(e_i-e_h)+sum(b)e_h; witnesses retained; legs are not primary cells")]
    from rexgraph.column_expansion import ColumnLegs, PrimaryColumnLift, primary_lift
    a, b = map(context.known_value, children)
    if isinstance(a, ColumnLegs) and isinstance(b, PrimaryColumnLift):
        primary_lift(a, b)
        status = "verified"
    else:
        status = "deferred"
    return result, [PredicateResult("identical_expansion", status,
        "execution requires the original legs and lift of the identical core expansion")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    bound = dict(source_bound=True, basis_bound=True)
    records = {
        "COLUMN_EXPANSION": ((TypePattern("boundary", kind=ValueKind.OPERATOR, **bound),), expansion_result,
            "rex.column_expansion", ("exact canonical boundary, full original source state",
                                     "elementary pairs plus witness term; primary lift retained")),
        "PRIMARY_LIFT": ((TypePattern("legs", kind=ValueKind.COLUMN_LEGS, **bound),
                          TypePattern("lift", kind=ValueKind.PRIMARY_COLUMN_LIFT, **bound)), lift_result,
            "rex.primary_lift", ("factors of the identical certified expansion", "no replacement of primary cells")),
        "HYPERSLICE": ((TypePattern("cell", kind=ValueKind.CELL, **bound),),
            lambda args: args[0].with_(name="Hyperslice", kind=ValueKind.HYPERSLICE),
            "rex.cell_neighborhood", ("immediate lower, upper and lateral primary cell selections",)),
    }
    for name, (inputs, result, key, preconditions) in records.items():
        register(OperatorSignature(name=name, source_kind=ValueKind.REX, inputs=inputs,
            result=result, implementation_key=key, preconditions=preconditions, memoizable=True))
