"""Static contracts for incidence, field and finite arithmetic readings."""
from __future__ import annotations

from fractions import Fraction

from .types import Domain, Exactness, RCType, ShapeRef, ValueKind, Variance


ARGUMENTS = {
    "SUPPORT": (("value",), ()),
    "DEGREE": (("value",), ()),
    "SHARED_BOUNDARY": (("left", "right"), ()),
    "FIELD": (("value", "grade"), (None,)),
    "GRADIENT": (("value",), ()), "CURL": (("value",), ()),
    "GRAM": (("values", "metric", "exact"), (None, True)),
    "GRAM_RANK": (("values", "metric"), (None,)),
    "ORIENTED_MOMENT": (("left", "right", "exact"), (True,)),
    "COBOUNDARY_MOMENT": (("left", "right", "exact"), (True,)),
    "FIELD_QUOTIENT": (("left", "right", "sector", "exact"), ("down", True)),
    "COFIELD_QUOTIENT": (("left", "right", "exact"), (True,)),
    "RATE": (("delta", "interval"), ()),
    "TEMPORAL_RATE": (("delta", "interval"), ()),
    "MOMENT_RATE": (("delta", "interval"), ()),
    "MASS": (("value", "grade"), (None,)),
    "ARGMIN": (("values", "measures"), ()), "ARGMAX": (("values", "measures"), ()),
    "TRACE": (("action", "exact"), (False,)),
    "TYPE_VIEW": (("value", "accession", "exact"), (False,)),
    "TYPES": (("family", "grade"), (None,)), "CURVATURE": (("value",), ()),
    "WEIGHT": (("value", "exact"), (True,)), "SIGNING": (("value",), ()),
    "ORIENTATION": (("value",), ()), "PARITY": (("values",), ()),
    "CHAIN_VALID": ((), ()),
    "GREEN_OPERATOR": (("grade", "alpha", "tol", "maxiter"), (1.0, 1e-10, 1000)),
    "GREEN_FIELD": (("value", "alpha", "tol", "maxiter"), (1.0, 1e-10, 1000)),
    "GREEN_GRAM": (("values", "action"), (None,)),
    "GREEN_SPREAD": (("left", "right", "action"), (None,)),
}


def _integer():
    return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER, exactness=Exactness.INTEGER)


def _support(args):
    value = args[0]
    if value.grade is None or value.grade < 1:
        raise ValueError("SUPPORT requires a positive grade")
    return RCType("CellSet", kind=ValueKind.CELL_SET, grade=value.grade - 1,
                  variance=Variance.CELL, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def _degree(args):
    value = args[0]
    if isinstance(value, RCType):
        return _integer()
    return _integer().with_(name="Cochain", kind=ValueKind.COCHAIN, grade=value, variance=Variance.COCHAIN)


def _shared(args):
    if args[0].grade != args[1].grade:
        raise ValueError("SHARED_BOUNDARY requires equal boundary grades")
    return _support(args)


def _field(args):
    from .signatures import _coefficient_carrier
    value = _coefficient_carrier(args[0])
    if len(args) > 1 and args[1] is not None and args[1] != value.grade:
        raise ValueError("FIELD grade must equal its cochain grade")
    return value.with_(name="Field", kind=ValueKind.FIELD)


def _gram(args, *, rank=False):
    from .signatures import _metric_check, _signed_moment_result
    values = args[0]
    if not isinstance(values, (tuple, list)):
        raise TypeError("GRAM requires an explicit sequence of typed values")
    metric = args[1] if len(args) > 1 else None
    exact = True if rank or len(args) < 3 else args[2]
    if not values:
        if metric is not None:
            raise ValueError("an empty Gram family has no grade for a metric")
        result = RCType("Rational", domain=Domain.RATIONAL, exactness=Exactness.RATIONAL) if exact else RCType(
            "Real", domain=Domain.REAL, exactness=Exactness.APPROXIMATE)
    else:
        result = None
        for value in values:
            if not isinstance(value, RCType) or value.kind not in {ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.FIELD}:
                raise TypeError("GRAM requires typed Chains or Cochains")
            if value.shape != values[0].shape:
                raise TypeError("GRAM requires matching vector or block shapes")
            if rank:
                _metric_check(value, metric, False)
            current = _signed_moment_result((values[0], value, None if rank else metric, exact))
            if result is None or current.domain is Domain.COMPLEX:
                result = current
    return _integer() if rank else result.with_(name="Gram", kind=ValueKind.GRAM,
                                                shape=ShapeRef((len(values), len(values))))


def _sector(args, *, quotient=False):
    from .signatures import _signed_moment_result
    if quotient and len(args) > 2 and args[2] not in {"down", "up"}:
        raise ValueError("FIELD_QUOTIENT sector must be down or up")
    position = 3 if quotient else 2
    exact = args[position] if len(args) > position else True
    if args[0].shape != args[1].shape:
        raise TypeError("boundary moments require matching vector or block shapes")
    return _signed_moment_result((*args[:2], None, exact))


def _scalar(value):
    from .aggregates import result_type
    return result_type((value,))


def _rate(args):
    value, interval = args
    interval_type = _scalar(interval)
    if not isinstance(interval, RCType) and interval == 0:
        raise ZeroDivisionError("rate interval must be nonzero")
    if isinstance(value, RCType) and value.kind in {ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.FIELD}:
        if value.domain not in {Domain.INTEGER, Domain.RATIONAL, Domain.REAL, Domain.COMPLEX}:
            raise TypeError("RATE requires numeric coefficients")
        result = value
    else:
        result = _scalar(value)
    exact = all(v.domain in {Domain.INTEGER, Domain.RATIONAL} and v.exactness in {
        Exactness.INTEGER, Exactness.RATIONAL} for v in (result, interval_type))
    scalar = result.kind in {ValueKind.EXACT_INTEGER, ValueKind.EXACT_RATIONAL, ValueKind.REAL}
    return result.with_(name=("Rational" if exact else "Real") if scalar else result.name,
                        kind=(ValueKind.EXACT_RATIONAL if exact else ValueKind.REAL) if scalar else result.kind,
                        domain=Domain.RATIONAL if exact else Domain.COMPLEX if result.domain is Domain.COMPLEX else Domain.REAL,
                        exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE)


def _mass(args):
    from .signatures import _coefficient_carrier
    value = _coefficient_carrier(args[0])
    if value.domain not in {Domain.INTEGER, Domain.RATIONAL, Domain.REAL}:
        raise TypeError("MASS requires real coefficients")
    if len(args) > 1 and args[1] is not None and value.grade != args[1]:
        raise ValueError("MASS grade must equal its carrier grade")
    kind = {Domain.INTEGER: ValueKind.EXACT_INTEGER, Domain.RATIONAL: ValueKind.EXACT_RATIONAL,
            Domain.REAL: ValueKind.REAL}[value.domain]
    return RCType(kind.value, kind=kind, domain=value.domain, exactness=value.exactness)


def _argextreme(args):
    from .aggregates import numeric_kind
    values, measures = args
    if not isinstance(values, (list, tuple)) or not isinstance(measures, (list, tuple)):
        raise TypeError("ARGMIN and ARGMAX require explicit value and scalar measure sequences")
    if not values or len(values) != len(measures):
        raise ValueError("values and measures must have the same nonzero length")
    for measure in measures:
        numeric_kind(measure)
    contracts = []
    for value in values:
        if isinstance(value, str):
            current = RCType("Text", kind=ValueKind.TEXT, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
        elif isinstance(value, RCType) and value.kind in {
                ValueKind.CELL, ValueKind.CELL_SET, ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.FIELD}:
            current = value
        else:
            current = _scalar(value)
        contracts.append(current)
    first = contracts[0]
    if any(not first.same_space(other) or first.kind != other.kind or first.domain != other.domain
           or first.exactness != other.exactness or first.shape != other.shape for other in contracts[1:]):
        raise TypeError("extreme values must have one type, arithmetic contract and cell space")
    return first


def _types(args):
    family = args[0]
    if len(args) > 1 and args[1] is not None and args[1] != family.grade:
        raise ValueError("TYPES grade must equal the family grade")
    return RCType("TypeNames", kind=ValueKind.SEQUENCE, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def _weight(args):
    value = args[0]
    exact = args[1] if len(args) > 1 else True
    scalar = value.kind is ValueKind.CELL
    return RCType("Rational" if scalar and exact else "Real" if scalar else "Cochain",
                  kind=(ValueKind.EXACT_RATIONAL if exact else ValueKind.REAL) if scalar else ValueKind.COCHAIN,
                  grade=None if scalar else 1, variance=None if scalar else Variance.COCHAIN,
                  exactness=Exactness.RATIONAL if exact else Exactness.APPROXIMATE,
                  domain=Domain.RATIONAL if exact else Domain.REAL)


def _parity(args):
    value = args[0]
    if isinstance(value, (list, tuple)):
        if any(not isinstance(v, RCType) or v.kind is not ValueKind.CELL or v.grade != 1 for v in value):
            raise TypeError("PARITY requires primary C1 Cells")
    elif value.grade != 1:
        raise TypeError("PARITY requires primary C1 Cells")
    return _integer()


def _green_input(value):
    if value.kind in {ValueKind.CELL, ValueKind.CELL_SET}:
        return value.with_(name="Cochain", kind=ValueKind.COCHAIN, variance=Variance.COCHAIN,
                           domain=Domain.INTEGER, exactness=Exactness.INTEGER)
    if value.domain not in {Domain.INTEGER, Domain.RATIONAL, Domain.REAL}:
        raise TypeError("Green solves require real coefficients")
    from .signatures import _coefficient_carrier
    return _coefficient_carrier(value)


def _green_parameters(args):
    from math import isfinite
    alpha = args[1] if len(args) > 1 else 1.0
    tol = args[2] if len(args) > 2 else 1e-10
    maxiter = args[3] if len(args) > 3 else 1000
    if isinstance(alpha, bool) or not isfinite(float(alpha)) or alpha < 0:
        raise ValueError("Green alpha must be finite and nonnegative")
    if isinstance(tol, bool) or not isfinite(float(tol)) or not 0 < tol < 1:
        raise ValueError("Green tolerance must lie in (0, 1)")
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter <= 0:
        raise ValueError("Green maxiter must be a positive integer")
    return alpha, tol, maxiter


def _green_field(args):
    _green_parameters(args)
    return _green_input(args[0]).with_(name="Field", kind=ValueKind.FIELD,
                                      domain=Domain.REAL, exactness=Exactness.APPROXIMATE)


def _green_operator(args):
    _green_parameters(args)
    return RCType("GreenAction", kind=ValueKind.GREEN_ACTION, grade=args[0],
                  domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def _green_gram(args, *, spread=False):
    values, action = ((args[:2], args[2] if len(args) > 2 else None) if spread else
                      (args[0], args[1] if len(args) > 1 else None))
    if not isinstance(values, (tuple, list)) or not values:
        raise TypeError("GREEN_GRAM requires a nonempty explicit sequence")
    converted = []
    for value in values:
        if not isinstance(value, RCType) or value.kind not in {
                ValueKind.CELL, ValueKind.CELL_SET, ValueKind.COCHAIN, ValueKind.FIELD}:
            raise TypeError("Green sources must be cells or cochain vectors")
        converted.append(_green_input(value))
    first = converted[0]
    for value in converted:
        if not first.same_space(value):
            raise ValueError("Green sources require one grade, source and basis")
        if value.shape is not None and len(value.shape.dims) != 1:
            raise TypeError("GREEN_GRAM requires vectors, not blocks")
    if action is not None:
        desc = action.operator
        if desc is None or desc.domain != first.basis or desc.action_variance != "cochain":
            raise ValueError("Green Gram requires a matching Euclidean cochain action")
    return RCType("Real" if spread else "Gram", kind=ValueKind.REAL if spread else ValueKind.GRAM,
                  domain=Domain.REAL, exactness=Exactness.APPROXIMATE,
                  shape=None if spread else ShapeRef((len(values), len(values))))


def refine(typed, context):
    """Check nested carrier ownership and the grades before any adapter runs."""
    pending = list(typed.args)
    while pending:
        value = pending.pop()
        if isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, RCType) and value.source is not None:
            if value.source != typed.binding.ref:
                raise ValueError(f"{typed.operator} requires values bound to its source")
            if value.basis is not None and value.basis.ordering != "canonical":
                raise ValueError(f"{typed.operator} requires the canonical ordered basis")
            if value.grade is not None and context.native:
                context.grade(value.grade)
    name, args = typed.operator, typed.args
    if name == "DEGREE" and isinstance(args[0], int):
        context.grade(args[0])
    result = typed.result
    if name == "GREEN_OPERATOR":
        from .inference import infer
        hodge = infer(typed.binding, "HODGE_OPERATOR", (args[0],), context=context).result
        result = infer(typed.binding, "RESOLVENT", (hodge, *_green_parameters(args)), context=context).result
    if name == "GREEN_FIELD":
        context.grade(args[0].grade)
    if name in {"DEGREE", "FIELD", "GRADIENT", "CURL", "WEIGHT"} and result.grade is not None and context.native:
        n = context.grade(result.grade)
        if name in {"DEGREE", "WEIGHT"}:
            result = result.with_(shape=ShapeRef((n,)))
    if name in {"ORIENTED_MOMENT", "COBOUNDARY_MOMENT", "FIELD_QUOTIENT", "COFIELD_QUOTIENT"}:
        if not context.native:
            raise TypeError("boundary moments require a native Rex source")
    return result


def install(register):
    from .signatures import (
        OperatorSignature, TypePattern, _access_result, _moment_result, _C1_COCHAIN,
        _CHAIN_OR_COCHAIN, _METRIC, _metric_curvature_result,
    )
    bound = dict(source_bound=True, basis_bound=True)
    chain = TypePattern("Chain", kind=ValueKind.CHAIN, **bound)
    support_input = TypePattern("cell or Chain", kind=(ValueKind.CELL, ValueKind.CHAIN), **bound)
    cochain = TypePattern("cochain", kind=(ValueKind.COCHAIN, ValueKind.FIELD), **bound)
    boolean = TypePattern("exact", literal=bool, optional=True)
    grade = TypePattern("grade", literal=(int, type(None)), optional=True)
    sequence = TypePattern("values", literal=(list, tuple))
    scalar = TypePattern("scalar", literal=(int, float, Fraction), kind=(ValueKind.EXACT_INTEGER, ValueKind.EXACT_RATIONAL, ValueKind.REAL))
    green_input = TypePattern("Green source", kind=(ValueKind.CELL, ValueKind.CELL_SET, ValueKind.COCHAIN, ValueKind.FIELD), **bound)
    green_action = TypePattern("Green action", kind=ValueKind.GREEN_ACTION, literal=type(None), optional=True, **bound)
    green_options = (TypePattern("alpha", literal=(int, float, Fraction), optional=True),
                     TypePattern("tol", literal=(int, float, Fraction), optional=True),
                     TypePattern("maxiter", literal=int, optional=True))
    specs = {
        "SUPPORT": ((support_input,), _support),
        "DEGREE": ((TypePattern("cell or grade", kind=ValueKind.CELL, literal=int, **bound),), _degree),
        "SHARED_BOUNDARY": ((support_input, support_input), _shared),
        "FIELD": ((cochain, grade), _field),
        "GRADIENT": ((_C1_COCHAIN,), lambda args: args[0].with_(name="Cochain", kind=ValueKind.COCHAIN, domain=Domain.REAL, exactness=Exactness.APPROXIMATE)),
        "CURL": ((_C1_COCHAIN,), lambda args: args[0].with_(name="Cochain", kind=ValueKind.COCHAIN, domain=Domain.REAL, exactness=Exactness.APPROXIMATE)),
        "GRAM": ((sequence, _METRIC, boolean), _gram),
        "GRAM_RANK": ((sequence, _METRIC), lambda args: _gram(args, rank=True)),
        "ORIENTED_MOMENT": ((chain, chain, boolean), _sector),
        "COBOUNDARY_MOMENT": ((chain, chain, boolean), _sector),
        "FIELD_QUOTIENT": ((chain, chain, TypePattern("sector", literal=str, optional=True), boolean), lambda args: _sector(args, quotient=True)),
        "COFIELD_QUOTIENT": ((chain, chain, boolean), _sector),
        "MASS": ((_CHAIN_OR_COCHAIN, grade), _mass),
        "ARGMIN": ((sequence, sequence), _argextreme),
        "ARGMAX": ((sequence, sequence), _argextreme),
        "TRACE": ((TypePattern("operator", kind=ValueKind.OPERATOR, **bound), boolean),
                  lambda args: _moment_result((args[0], 1, False, args[1] if len(args) > 1 else False))),
        "TYPE_VIEW": ((_CHAIN_OR_COCHAIN, TypePattern("accession", kind=ValueKind.TYPE_ACCESSION, **bound), boolean), _access_result),
        "TYPES": ((TypePattern("family", kind=ValueKind.ACCESSION_FAMILY, **bound), grade), _types),
        "CURVATURE": ((_C1_COCHAIN,), _metric_curvature_result),
        "WEIGHT": ((TypePattern("C1 cell or field", kind=(ValueKind.CELL, ValueKind.COCHAIN, ValueKind.FIELD), grade=1, **bound), boolean), _weight),
        "SIGNING": ((TypePattern("C1 cell", kind=ValueKind.CELL, grade=1, **bound),), lambda args: _integer()),
        "ORIENTATION": ((TypePattern("C1 cell", kind=ValueKind.CELL, grade=1, **bound),),
                        lambda args: _integer().with_(name="Chain", kind=ValueKind.CHAIN, grade=0, variance=Variance.CHAIN)),
        "PARITY": ((TypePattern("C1 cells", kind=(ValueKind.CELL, ValueKind.CELL_SET), literal=(tuple, list), **bound),), _parity),
        "CHAIN_VALID": ((), lambda args: RCType("Boolean", kind=ValueKind.BOOLEAN, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)),
        "GREEN_OPERATOR": ((TypePattern("grade", literal=int), *green_options), _green_operator),
        "GREEN_FIELD": ((green_input, *green_options), _green_field),
        "GREEN_GRAM": ((sequence, green_action), _green_gram),
        "GREEN_SPREAD": ((green_input, green_input, green_action), lambda args: _green_gram(args, spread=True)),
    }
    for name in ("RATE", "TEMPORAL_RATE", "MOMENT_RATE"):
        specs[name] = ((TypePattern("delta", literal=(int, float, Fraction), kind=(
            ValueKind.CHAIN, ValueKind.COCHAIN, ValueKind.FIELD, ValueKind.EXACT_INTEGER,
            ValueKind.EXACT_RATIONAL, ValueKind.REAL), source_bound=True), scalar), _rate)
    for name, (inputs, result) in specs.items():
        register(OperatorSignature(name=name, source_kind=ValueKind.REX, inputs=inputs,
            result=result, implementation_key=f"rcql.value.{name.lower()}", memoizable=True,
            preconditions=("explicit typed carriers; source, variance, grade and basis are retained",
                           "exact arithmetic is certified from integer or rational inputs only")))
