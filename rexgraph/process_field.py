"""Exact sampled processes and declared factor response fields."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np

from rexgraph.coordinate_map import _identity, _is_action
from rexgraph.graded_metric import _fraction
from rexgraph.tensor_field import TensorField, TensorChannels, apply_tensor
from rexgraph.type_accession import CoordinateSpace

__all__ = ["sampled_field", "trajectory_comparison", "sample_rates", "diagonal_axes",
           "factor_contrast", "factorial_contrast", "response_direction"]


def _references(fields):
    refs = {s.coefficient_digest: s for f in fields for s in
            (*f.dependencies, *((f.source,) if f.source else ()))}
    return tuple(refs.values())


def _compatible(fields):
    fields = tuple(fields)
    if not fields or any(not isinstance(f, TensorField) for f in fields):
        raise TypeError("process operations require tensor fields")
    first = fields[0]
    for f in fields:
        f.check_state()
        if (f.space, f.axes, f.grade, f.variance) != (first.space, first.axes, first.grade, first.variance):
            raise ValueError("process fields require explicitly matching observation coordinates")
    return fields


def _result(template, values, fields, declaration, *, space=None, axes=None):
    return TensorField(space or template.space, values, template.axes if axes is None else axes,
                       grade=template.grade, variance=template.variance, dependencies=_references(fields),
                       provenance=tuple(f.coefficient_digest for f in fields)+(declaration,))


def sampled_field(fields, times, *, axis, unit, maps=None):
    """Retain supplied samples without interpolation or inferred time alignment."""
    fields = tuple(fields)
    if maps is not None:
        maps = tuple(maps)
        if len(maps) != len(fields):
            raise ValueError("each sample requires one explicit observation map")
        fields = tuple(apply_tensor(m, f) for m, f in zip(maps, fields, strict=True))
    fields = _compatible(fields)
    times = tuple(_fraction(t) for t in times)
    if len(times) != len(fields) or any(a >= b for a, b in zip(times[:-1], times[1:], strict=True)):
        raise ValueError("sample times must match the fields and increase strictly")
    if not all(isinstance(s, str) and s for s in (axis, unit)):
        raise ValueError("samples require an axis and unit")
    time_axis = CoordinateSpace("sample/"+axis+"/"+unit, tuple(str(t) for t in times))
    if time_axis.name in {a.name for a in fields[0].axes}:
        raise ValueError("sample axis is already present")
    values = np.stack([f.values for f in fields], axis=1)
    declaration = _identity(("sampled_field_v1", axis, unit, tuple(str(t) for t in times)))
    return _result(fields[0], values, fields, declaration, axes=(time_axis, *fields[0].axes))


def trajectory_comparison(left, right, times, *, axis, unit, left_maps=None, right_maps=None):
    """Compare supplied pathway observations on a declared common sampling schedule."""
    a = sampled_field(left, times, axis=axis, unit=unit, maps=left_maps)
    b = sampled_field(right, times, axis=axis, unit=unit, maps=right_maps)
    _compatible((a, b))
    declaration = _identity(("trajectory_comparison_v1", a.coefficient_digest, b.coefficient_digest))
    delta = _result(a, b.values-a.values, (a, b), declaration)
    return TensorChannels(("left", "right", "difference"), (a, b, delta), declaration, _references((a, b)))


def sample_rates(field):
    """Compute interval secants, not an interpolation or an instantaneous velocity."""
    if not isinstance(field, TensorField) or not field.axes or not field.axes[0].name.startswith("sample/"):
        raise ValueError("rates require an explicit leading sample axis")
    field.check_state()
    times = tuple(Fraction(t) for t in field.axes[0].keys)
    if len(times) < 2 or any(a >= b for a, b in zip(times[:-1], times[1:], strict=True)):
        raise ValueError("rates require increasing sample times")
    intervals = tuple(b-a for a, b in zip(times[:-1], times[1:], strict=True))
    shape = (1, len(intervals), *([1]*(field.values.ndim-2)))
    values = (field.values[:, 1:]-field.values[:, :-1])/np.asarray(intervals, dtype=object).reshape(shape)
    rate_axis = CoordinateSpace("interval/"+field.axes[0].name, tuple(str(a)+":"+str(b) for a, b in zip(times[:-1], times[1:], strict=True)))
    space = CoordinateSpace(field.space.name+"/per/"+field.axes[0].name, field.space.keys)
    return _result(field, values, (field,), _identity(("sample_rates_v1", field.coefficient_digest)),
                   space=space, axes=(rate_axis, *field.axes[1:]))


def diagonal_axes(field, first, second, name):
    """Select matching coordinate pairs while retaining the selected axis."""
    if not isinstance(field, TensorField):
        raise TypeError("axis selection requires a tensor field")
    field.check_state()
    names = [a.name for a in field.axes]
    if first == second or first not in names or second not in names:
        raise ValueError("select two different retained axes")
    i, j = names.index(first), names.index(second)
    if field.axes[i].keys != field.axes[j].keys:
        raise ValueError("diagonal selection requires identical ordered keys")
    if not isinstance(name, str) or not name or name in set(names)-{first, second}:
        raise ValueError("selected axis name must be distinct")
    rest = tuple(k for k in range(len(field.axes)) if k not in (i, j))
    values = np.diagonal(field.values, axis1=i+1, axis2=j+1)
    values = np.moveaxis(values, -1, 1)
    axes = (CoordinateSpace(name, field.axes[i].keys), *(field.axes[k] for k in rest))
    return _result(field, values, (field,), _identity(("diagonal_axes_v1", first, second, name)), axes=axes)


def factor_contrast(old, new, step, *, parameter, unit):
    """Retain a finite contrast and its secant under an explicitly supplied factor step."""
    fields = _compatible((old, new))
    step = _fraction(step)
    if not step:
        raise ValueError("factor step must be nonzero")
    if not all(isinstance(s, str) and s for s in (parameter, unit)):
        raise ValueError("factor response requires a parameter name and unit")
    declaration = _identity(("factor_contrast_v1", parameter, unit, str(step), *(f.coefficient_digest for f in fields)))
    delta = _result(old, new.values-old.values, fields, declaration)
    space = CoordinateSpace(old.space.name+"/per/"+parameter+"/"+unit, old.space.keys)
    secant = _result(old, delta.values/step, fields, declaration, space=space)
    return TensorChannels(("difference", "per_unit"), (delta, secant), declaration, _references(fields))


def factorial_contrast(base, first, second, joint, *, first_step, second_step, first_parameter, second_parameter):
    """Retain a finite four corner interaction without assigning causal meaning."""
    fields = _compatible((base, first, second, joint))
    p, q = _fraction(first_step), _fraction(second_step)
    if not p or not q:
        raise ValueError("factor steps must be nonzero")
    if any(not isinstance(v, tuple) or len(v) != 2 or not all(isinstance(s, str) and s for s in v)
           for v in (first_parameter, second_parameter)):
        raise ValueError("each parameter needs its name and unit")
    if first_parameter[0] == second_parameter[0]:
        raise ValueError("interaction requires two distinct factor names")
    declaration = _identity(("factorial_contrast_v1", first_parameter, second_parameter, str(p), str(q),
                            *(f.coefficient_digest for f in fields)))
    a = _result(base, first.values-base.values, fields, declaration)
    b = _result(base, second.values-base.values, fields, declaration)
    c = _result(base, joint.values-first.values-second.values+base.values, fields, declaration)
    return TensorChannels(("first", "second", "interaction"), (a, b, c), declaration, _references(fields))


def response_direction(action, response, rhs, operator_direction, rhs_direction, *, parameter, unit, scale_direction=0):
    """Differentiate a verified native Green equation in a declared operator direction."""
    from rexgraph.native_field import FieldAction
    if not isinstance(action, FieldAction) or action.operation != "green" or action.transposed:
        raise TypeError("response derivative requires a native Green action")
    fields = _compatible((response, rhs, rhs_direction))
    if response.space != action.domain or response.grade != action.grade or response.variance != "chain":
        raise ValueError("response must use the declared native grade and chain coordinates")
    if (not _is_action(operator_direction) or operator_direction.domain != action.domain
            or operator_direction.codomain != action.codomain):
        raise ValueError("operator derivative must use the same native coordinates")
    if not all(isinstance(v, str) and v for v in (parameter, unit)):
        raise ValueError("response derivative requires a parameter name and unit")
    for f in fields:
        if action.owner.source is not None and (f.source is None or f.source.source is not action.owner.source):
            raise ValueError("response fields must belong to the native action source")
    hodge = apply_tensor(action.owner.hodge(action.grade), response)
    if not np.array_equal(response.values+action.parameter*hodge.values, rhs.values):
        raise ValueError("candidate does not solve the declared native response equation")
    applied = apply_tensor(operator_direction, response)
    residual = _result(rhs, rhs_direction.values-action.parameter*applied.values-_fraction(scale_direction)*hodge.values,
                       (*fields, applied), _identity(("response_direction_rhs", operator_direction.coefficient_digest)))
    residual = replace(residual, source=rhs.source)
    result = apply_tensor(action, residual)
    space = CoordinateSpace(result.space.name+"/per/"+parameter+"/"+unit, result.space.keys)
    return replace(result, space=space, grade=None,
                   provenance=(*result.provenance, _identity((parameter, unit, str(_fraction(scale_direction))))))
