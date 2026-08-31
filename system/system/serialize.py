"""JSON values returned by the System API."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from rexgraph.cochain import Cochain, Field
from rexgraph.green import GreenOperator
from rexgraph.linear_operator import RexOperator


def _array(value: Any, max_values: int) -> dict[str, Any]:
    arr = np.asarray(value)
    out = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size <= max_values:
        out["values"] = arr.tolist()
    elif arr.size:
        flat = arr.ravel()
        out["sample"] = flat[:max_values].tolist()
        if np.issubdtype(arr.dtype, np.number):
            out["min"] = float(np.nanmin(arr))
            out["max"] = float(np.nanmax(arr))
    return out


def json_value(value: Any, *, max_values: int = 256) -> Any:
    """Convert one RCQL result to a bounded JSON value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _array(value, max_values)
    if isinstance(value, Cochain):
        return {
            "kind": "Cochain",
            "grade": value.grade,
            "cells": value.n_cells,
            "values": _array(value.values, max_values),
        }
    if isinstance(value, Field):
        return {
            "kind": "Field",
            "field": value.kind,
            "grade": value.grade,
            "values": _array(value.values, max_values),
            "operator": json_value(value.operator, max_values=max_values),
        }
    if isinstance(value, RexOperator):
        return {
            "kind": "RexOperator",
            "name": value.name,
            "shape": list(value.shape),
            "domain_grade": value.domain_grade,
            "codomain_grade": value.codomain_grade,
            "symmetric": value.symmetric,
            "psd": value.psd,
            "arithmetic": value.arithmetic,
        }
    if isinstance(value, GreenOperator):
        return {
            "kind": "GreenOperator",
            "green": value.kind,
            "operator": json_value(value.operator, max_values=max_values),
        }
    if isinstance(value, dict):
        return {str(k): json_value(v, max_values=max_values) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v, max_values=max_values) for v in value]
    if is_dataclass(value):
        return json_value(asdict(value), max_values=max_values)
    if hasattr(value, "_asdict"):
        return json_value(value._asdict(), max_values=max_values)
    return repr(value)
