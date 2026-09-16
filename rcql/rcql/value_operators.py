"""Typed value readings using the existing incidence and metric operations."""
from __future__ import annotations

from fractions import Fraction
from numbers import Integral, Real

import numpy as np

from rexgraph.cells import Cell, CellSet, cell_count
from rexgraph.cochain import Chain, Cochain, Field

from .aggregates import reduce_scalars
from .execution_trace import record_method


def support(source, value):
    """Lower support, retaining primary participants for a Cell, including a loop."""
    from . import operators as op
    if isinstance(value, Cell):
        op._typed_cells(source, value, operator="SUPPORT")
        if value.grade == 0:
            raise ValueError("SUPPORT requires a positive grade")
        if value.grade == 1:
            source._ensure_clean()
            lo, hi = source._boundary_ptr[value.index:value.index + 2]
            return CellSet(source, 0, tuple(map(int, source._boundary_idx[int(lo):int(hi)])))
        return op.boundary(source, value).cells
    value = op._typed_value(source, value, operator="SUPPORT", variance="chain")
    if value.grade == 0:
        raise ValueError("SUPPORT requires a positive grade")
    out = op.boundary(source, value.grade, value)
    coefficients = np.asarray(out.values)
    mask = coefficients != 0
    if mask.ndim == 2:
        mask = mask.any(axis=1)
    return CellSet(source, value.grade - 1, tuple(np.flatnonzero(mask)))


def degree(source, value):
    """Count primary upper incidences, once per cell, with no adjacency projection."""
    from . import operators as op
    from rexgraph.native_sparse import boundary_carriers, sparse_arrays
    one = isinstance(value, Cell)
    if one:
        op._typed_cells(source, value, operator="DEGREE")
        grade = value.grade
    else:
        grade = op._grade_argument(value)
    n = cell_count(source, grade)
    if grade == 0:
        ptr, indices = source._v2e

        def count_row(index):
            lo, hi = int(ptr[index]), int(ptr[index + 1])
            return len(set(map(int, indices[lo:hi])))
    elif grade < len(boundary_carriers(source)):
        ptr, indices, coefficients, _ = sparse_arrays(boundary_carriers(source)[grade])

        def count_row(index):
            row = {}
            for position in range(int(ptr[index]), int(ptr[index + 1])):
                cell = int(indices[position])
                row[cell] = row.get(cell, 0) + coefficients[position]
            return sum(value != 0 for value in row.values())
    else:
        def count_row(index):
            return 0
    record_method("primary-upper-incidence-count", grade=grade)
    if one:
        return int(count_row(value.index))
    return Cochain(grade, np.fromiter((count_row(i) for i in range(n)), dtype=np.int64, count=n), source=source)


def shared_boundary(source, left, right):
    a, b = support(source, left), support(source, right)
    if a.grade != b.grade:
        raise ValueError("SHARED_BOUNDARY requires equal boundary grades")
    return CellSet(source, a.grade, tuple(set(a.indices) & set(b.indices)))


def field(source, value, grade=None):
    """Give an existing cochain field meaning without changing its variance."""
    from . import operators as op
    carrier = op._typed_value(source, value, operator="FIELD", variance="cochain")
    if grade is not None and op._grade_argument(grade) != carrier.grade:
        raise ValueError("FIELD grade must equal its cochain grade")
    return Field(carrier, value.operator if isinstance(value, Field) else None)


def gradient(source, value):
    from . import operators as op
    return op.hodge(source, value)["gradient"]


def curl(source, value):
    from . import operators as op
    return op.hodge(source, value)["curl"]


def _family(source, values, metric, exact):
    from . import operators as op
    if not isinstance(values, (list, tuple)):
        raise TypeError("GRAM requires an explicit sequence of typed values")
    if not isinstance(exact, bool):
        raise TypeError("exact must be boolean")
    values = tuple(op._typed_value(source, value, operator="GRAM") for value in values)
    for value in values:
        op._same_space(source, values[0], value, operator="GRAM")
        op._contraction_metric(source, value, metric)
        if exact:
            op._require_exact_coefficients(value.values)
    if not values and metric is not None:
        raise ValueError("an empty Gram family has no grade for a metric")
    return values


def gram(source, values, metric=None, exact=True):
    from . import operators as op
    values = _family(source, values, metric, exact)
    n = len(values)
    dtype = object if exact else complex if any(np.iscomplexobj(v.values) for v in values) else float
    result = np.empty((n, n), dtype=dtype)
    for i, left in enumerate(values):
        for j in range(i, n):
            entry = op.moment(source, left, values[j], metric, exact)
            result[i, j] = entry
            result[j, i] = entry if exact else np.conjugate(entry)
    record_method("typed-gram-contractions", exact=exact, fields=n)
    return result


def gram_rank(source, values, metric=None):
    """Exact rank of the input span; a positive metric does not change that rank."""
    from rexgraph.graded_boundary import _rank_integer_columns
    from rexgraph.native_rank import clear_column_denominators
    values = _family(source, values, metric, True)
    columns = [{i: Fraction(int(x)) if isinstance(x, Integral) else x
                for i, x in enumerate(np.asarray(v.values).flat) if x != 0} for v in values]
    n = int(np.size(values[0].values)) if values else 0
    rank, info = _rank_integer_columns((n, len(columns)), clear_column_denominators(columns))
    record_method("certified-exact-gram-rank", rank_info=info)
    return int(rank)


def _sector_moment(source, left, right, sector, exact):
    from . import operators as op
    from rexgraph.weighted_hodge import weighted_hodge
    left, right = op._same_space(source, left, right, operator="FIELD_QUOTIENT")
    if not isinstance(left, Chain):
        raise TypeError("boundary moments require Chains; use an explicit metric identification for Cochains")
    action = weighted_hodge(source, left.grade, sector=sector)
    transformed = op.apply(source, action, right, exact)
    return op.moment(source, left, transformed, None, exact)


def oriented_moment(source, left, right, exact=True):
    return _sector_moment(source, left, right, "difference", exact)


def coboundary_moment(source, left, right, exact=True):
    return _sector_moment(source, left, right, "up", exact)


def field_quotient(source, left, right, sector="down", exact=True):
    if sector not in {"down", "up"}:
        raise ValueError("FIELD_QUOTIENT sector must be down or up")
    numerator = _sector_moment(source, left, right, sector, exact)
    denominator = _sector_moment(source, right, right, sector, exact)
    if denominator == 0:
        raise ZeroDivisionError("field quotient denominator is zero")
    return numerator / denominator


def cofield_quotient(source, left, right, exact=True):
    return field_quotient(source, left, right, "up", exact)


def _interval(interval):
    if isinstance(interval, bool) or not isinstance(interval, (Integral, Real, Fraction)):
        raise TypeError("rate interval must be a finite real scalar")
    if not isinstance(interval, (Integral, Fraction)) and not np.isfinite(interval):
        raise ValueError("rate interval must be finite")
    if interval == 0:
        raise ZeroDivisionError("rate interval must be nonzero")
    return Fraction(int(interval)) if isinstance(interval, Integral) else interval


def rate(source, delta, interval):
    from . import operators as op
    interval = _interval(interval)
    if isinstance(delta, (Chain, Cochain, Field)):
        carrier = op._typed_value(source, delta, operator="RATE")
        values = np.asarray(carrier.values)
        exact = all(isinstance(v, (Integral, Fraction)) and not isinstance(v, (bool, np.bool_))
                    for v in values.flat) and isinstance(interval, Fraction)
        if exact:
            result = np.array([Fraction(int(v)) / interval if isinstance(v, Integral) else v / interval
                               for v in values.flat], dtype=object).reshape(values.shape)
        else:
            from rexgraph.linear_operator import _numeric_array
            numeric = _numeric_array(values, operation="RATE")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                result = numeric / float(interval)
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("rate is outside numerical range")
        result = carrier.with_values(result)
        return Field(result, delta.operator, delta.kind) if isinstance(delta, Field) else result
    if isinstance(delta, bool) or not isinstance(delta, (Integral, Real, Fraction)):
        raise TypeError("RATE requires a real scalar or typed coefficient carrier")
    delta = int(delta) if isinstance(delta, Integral) else delta
    if not isinstance(delta, (int, Fraction)) and not np.isfinite(delta):
        raise ValueError("rate delta must be finite")
    result = delta / interval
    if not isinstance(result, Fraction) and not np.isfinite(result):
        raise FloatingPointError("rate is outside numerical range")
    return result


def temporal_rate(source, delta, interval):
    return rate(source, delta, interval)


def moment_rate(source, delta, interval):
    return rate(source, delta, interval)


def mass(source, value, grade=None):
    """Total absolute coefficient mass; not a Hodge trace or a cell count."""
    from . import operators as op
    carrier = op._typed_value(source, value, operator="MASS")
    if grade is not None and op._grade_argument(grade) != carrier.grade:
        raise ValueError("MASS grade must equal its carrier grade")
    values = np.asarray(carrier.values)
    if np.iscomplexobj(values) or any(not isinstance(x, (Real, Integral, Fraction))
                                     or isinstance(x, (bool, np.bool_)) for x in values.flat):
        raise TypeError("MASS requires real coefficients")
    return reduce_scalars([abs(int(x)) if isinstance(x, Integral) else abs(x)
                           for x in values.flat])


def argmin(source, values, measures):
    return _argextreme(source, values, measures, maximum=False)


def argmax(source, values, measures):
    return _argextreme(source, values, measures, maximum=True)


def _argextreme(source, values, measures, *, maximum):
    from . import operators as op
    if not isinstance(values, (list, tuple)) or not isinstance(measures, (list, tuple)):
        raise TypeError("ARGMIN and ARGMAX require explicit value and scalar measure sequences")
    if not values or len(values) != len(measures):
        raise ValueError("values and measures must have the same nonzero length")
    from .aggregates import numeric_kind
    from .executor import value_exactness
    contracts = []
    for value in values:
        if isinstance(value, (Cell, CellSet)):
            op._typed_cells(source, value, operator="ARGMIN/ARGMAX")
        elif isinstance(value, (Chain, Cochain, Field)):
            op._typed_value(source, value, operator="ARGMIN/ARGMAX")
        elif isinstance(value, bool) or not isinstance(value, (str, Integral, Real, Fraction)):
            raise TypeError("ARGMIN and ARGMAX values must be scalars or typed cell values")
        carrier = value.cochain if isinstance(value, Field) else value
        contracts.append((
            type(value) if isinstance(value, (Cell, CellSet, Chain, Cochain, Field, str)) else "scalar",
            getattr(carrier, "grade", None), getattr(carrier, "cell_keys", None),
            getattr(getattr(carrier, "values", None), "shape", None), value_exactness(value),
        ))
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise TypeError("extreme values must have one type, arithmetic contract and cell space")
    for value in measures:
        numeric_kind(value)
    index = (max if maximum else min)(range(len(measures)), key=measures.__getitem__)
    return values[index]


def trace(source, action, exact=False):
    from . import operators as op
    return op.scale_moment(source, action, 1, False, exact)


def type_view(source, value, accession, exact=False):
    from . import operators as op
    return op.access(source, value, accession, exact)


def types(source, family, grade=None):
    from . import operators as op
    from rexgraph.type_accession import AccessionFamily
    if not isinstance(family, AccessionFamily):
        raise TypeError("TYPES requires an explicit AccessionFamily")
    for accession in family.accessions:
        if accession.source is not source:
            raise ValueError("TYPES requires accessions bound to its source")
        if grade is not None and op._grade_argument(grade) != accession.grade:
            raise ValueError("TYPES grade must equal the family grade")
    return tuple(a.name for a in family.accessions)


def curvature(source, value):
    from . import operators as op
    return op.metric_curvature(source, value)


def weight(source, value, exact=True):
    from . import operators as op
    if not isinstance(exact, bool):
        raise TypeError("exact must be boolean")
    if isinstance(value, Cell):
        op._typed_cells(source, value, operator="WEIGHT")
        grade = value.grade
    else:
        carrier = op._typed_value(source, value, operator="WEIGHT", variance="cochain", grade=1)
        grade = carrier.grade
    if grade != 1:
        raise ValueError("WEIGHT reads the declared C1 relation metric")
    weights = source.edge_metric_exact if exact else source.edge_metric
    if weights is None:
        weights = (Fraction(1),) * int(source.nE) if exact else np.ones(source.nE)
    if isinstance(value, Cell):
        return weights[value.index] if exact else float(weights[value.index])
    return Cochain(1, np.asarray(weights, dtype=object if exact else float), source=source)


def signing(source, value):
    from . import operators as op
    value = op._typed_cells(source, value, operator="SIGNING")
    if not isinstance(value, Cell) or value.grade != 1:
        raise TypeError("SIGNING requires one primary C1 Cell")
    signs = source._signs
    declared = 1 if signs is None else int(signs[value.index])
    if declared not in {-1, 1}:
        raise ValueError("declared signing must be +1 or -1")
    return int(declared < 0)


def orientation(source, value):
    """The primary C1 distinguished participant mask, separate from gauge signing."""
    from . import operators as op
    return op.head(source, value)


def parity(source, values):
    """Product of declared C1 signs, not an upper grade orientation invariant."""
    from . import operators as op
    if isinstance(values, Cell):
        values = (values,)
    elif isinstance(values, CellSet):
        op._typed_cells(source, values, operator="PARITY")
        values = values.cells
    if not isinstance(values, (tuple, list)):
        raise TypeError("PARITY requires a Cell, CellSet or explicit cell sequence")
    result = 1
    for value in values:
        result *= -1 if signing(source, value) else 1
    return result


def chain_valid(source):
    """Check declared C2 and all carried higher chain identities over Q."""
    from rexgraph.native_rank import exact_tower, tower_chain_residual
    source._ensure_clean()
    shapes, columns = exact_tower(source)
    return bool(source.chain_valid and tower_chain_residual(shapes, columns) == 0)


def green_operator(source, grade, alpha=1.0, tol=1e-10, maxiter=1000):
    from . import operators as op
    return op.resolvent(source, op.hodge_op(source, op._grade_argument(grade)), alpha, tol, maxiter)


def _green_input(source, value):
    from . import operators as op
    if isinstance(value, (Cell, CellSet)):
        value = op.indicator(source, value)
    return op._typed_value(source, value, operator="GREEN_FIELD", variance="cochain")


def green_field(source, value, alpha=1.0, tol=1e-10, maxiter=1000):
    from . import operators as op
    value = _green_input(source, value)
    return op.apply(source, green_operator(source, value.grade, alpha, tol, maxiter), value)


def green_gram(source, values, action=None):
    from . import operators as op
    from rexgraph.green import GreenOperator
    if not isinstance(values, (tuple, list)) or not values:
        raise TypeError("GREEN_GRAM requires a nonempty explicit sequence")
    carriers = tuple(_green_input(source, value) for value in values)
    for value in carriers:
        op._same_space(source, carriers[0], value, operator="GREEN_GRAM")
        if np.asarray(value.values).ndim != 1 or np.iscomplexobj(value.values):
            raise TypeError("GREEN_GRAM requires real vectors")
    grade = carriers[0].grade
    action = green_operator(source, grade) if action is None else action
    if not isinstance(action, GreenOperator) or action.operator.source is not source:
        raise TypeError("GREEN_GRAM requires a Green action bound to its source")
    if action.operator.domain_grade != grade or action.metric is not None:
        raise ValueError("GREEN_GRAM requires a matching Euclidean cochain Green action")
    from rexgraph.linear_operator import _numeric_array
    block = np.column_stack([_numeric_array(value.values, operation="GREEN_GRAM") for value in carriers])
    if np.iscomplexobj(block):
        raise TypeError("GREEN_GRAM requires real vectors")
    if not np.all(np.isfinite(block)):
        raise ValueError("GREEN_GRAM requires finite coefficients")
    fields, info = action.solve_with_info(block)
    result = block.T @ fields
    result = (result + result.T) * 0.5
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("GREEN_GRAM is outside numerical range")
    record_method("green-gram-contraction", **info)
    return result


def green_spread(source, left, right, action=None):
    matrix = green_gram(source, (left, right), action)
    a, b = float(matrix[0, 0]), float(matrix[1, 1])
    if a == 0 or b == 0:
        return None
    if a < 0 or b < 0:
        raise ValueError("Green spread requires nonnegative quadrances")
    # Divide separately so a finite normalized reading need not overflow a*b.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = float(1 - (matrix[0, 1] / a) * (matrix[1, 0] / b))
    if not np.isfinite(result):
        raise FloatingPointError("Green spread is outside numerical range")
    return result


ADAPTERS = {
    "SUPPORT": support, "DEGREE": degree, "SHARED_BOUNDARY": shared_boundary,
    "FIELD": field, "GRADIENT": gradient, "CURL": curl,
    "GRAM": gram, "GRAM_RANK": gram_rank,
    "ORIENTED_MOMENT": oriented_moment, "COBOUNDARY_MOMENT": coboundary_moment,
    "FIELD_QUOTIENT": field_quotient, "COFIELD_QUOTIENT": cofield_quotient,
    "RATE": rate, "TEMPORAL_RATE": temporal_rate, "MOMENT_RATE": moment_rate,
    "MASS": mass, "ARGMIN": argmin, "ARGMAX": argmax,
    "TRACE": trace, "TYPE_VIEW": type_view,
    "TYPES": types, "CURVATURE": curvature, "WEIGHT": weight,
    "SIGNING": signing, "ORIENTATION": orientation, "PARITY": parity,
    "CHAIN_VALID": chain_valid,
    "GREEN_OPERATOR": green_operator, "GREEN_FIELD": green_field,
    "GREEN_GRAM": green_gram, "GREEN_SPREAD": green_spread,
}


def install(register):
    for name, adapter in ADAPTERS.items():
        register(name)(adapter)
