"""Exact logical SQL columns using the common coefficient/presence carrier."""
from __future__ import annotations

from numbers import Integral
import numpy as np

from rexgraph._binary import Cursor, integer
from rexgraph.exact_array import ExactArray
from .columnar import exact_column, _column

_FIELDS = ("numerator", "denominator", "presence", "kind", "big_numerator", "big_denominator")


def encode_columns(arrays):
    import sqlalchemy as sa
    columns, types, declarations = {}, {}, {}
    logical = set(arrays)
    lengths = set()
    def put(name, array, dtype=None, owner=None):
        if name in columns or (name in logical and name != owner):
            raise ValueError(f"SQL logical/physical column collision at {name!r}")
        columns[name] = array
        if dtype is not None:
            types[name] = dtype
    for name, raw in arrays.items():
        if type(name) is not str or not name:
            raise ValueError("SQL column names must be nonempty strings")
        value = exact_column(raw)
        if isinstance(value, np.ndarray) and value.dtype.kind == "u" and value.size and int(value.max()) >= 2**63:
            value = ExactArray.from_values(value)
        shape = value.shape
        if len(shape) not in (1, 2) or (len(shape) == 2 and not shape[1]):
            raise ValueError(f"SQL column {name!r} must have one or two nonempty dimensions")
        lengths.add(shape[0])
        exact = isinstance(value, ExactArray)
        if exact or len(shape) == 2:
            declarations[name] = {"shape": list(shape), "split": len(shape) == 2, "codec": "exact-array-v1" if exact else "numpy"}
        for axis in range(shape[1] if len(shape) == 2 else 1):
            physical = f"{name}_{axis}" if len(shape) == 2 else name
            part = _column(value, axis) if len(shape) == 2 else value
            if not exact:
                if part.dtype.kind == "O" and all(v is None or isinstance(v, bytes) for v in part.flat):
                    put(physical, part, sa.LargeBinary(), name)
                    continue
                if part.dtype.kind not in "biufSU":
                    raise TypeError(f"SQL column {name!r} requires an explicit native value codec")
                put(physical, part, owner=name)
                continue
            big_n, big_d = [None]*part.size, [None]*part.size
            for position, numerator, denominator in part.bigint:
                big_n[position], big_d[position] = integer(numerator), integer(denominator)
            values = (part.numerator, part.denominator, part.presence.astype(np.uint8), part.kind,
                      np.asarray(big_n, dtype=object), np.asarray(big_d, dtype=object))
            storage = (sa.BigInteger(), sa.BigInteger(), sa.SmallInteger(), sa.SmallInteger(), sa.LargeBinary(), sa.LargeBinary())
            for field, array, dtype in zip(_FIELDS, values, storage, strict=True):
                put(physical+"__rex_"+field, array, dtype, name)
    if len(lengths) > 1:
        raise ValueError("SQL columns must have equal row counts")
    return columns, types, declarations


def project_columns(names, declarations):
    result = []
    for name in names:
        info = declarations.get(name)
        parts = ([f"{name}_{axis}" for axis in range(info["shape"][1])] if info and info["split"] else [name])
        for part in parts:
            result.extend([part+"__rex_"+field for field in _FIELDS] if info and info["codec"] == "exact-array-v1" else [part])
    return result


def decode_columns(columns, declarations):
    result = dict(columns)
    for name, info in declarations.items():
        if (not isinstance(info, dict) or set(info) != {"shape", "split", "codec"}
                or type(info["split"]) is not bool or info["codec"] not in {"numpy", "exact-array-v1"}
                or not isinstance(info["shape"], list) or len(info["shape"]) != (2 if info["split"] else 1)
                or any(type(n) is not int or n < 0 for n in info["shape"])
                or (info["split"] and not info["shape"][1])):
            raise ValueError("invalid SQL logical column declaration")
        parts = [f"{name}_{axis}" for axis in range(info["shape"][1])] if info["split"] else [name]
        expected = project_columns([name], {name: info})
        present = set(expected) & set(result)
        if not present:
            continue
        if present != set(expected):
            raise ValueError(f"incomplete SQL physical column {name!r}")
        values = []
        for part in parts:
            if info["codec"] == "numpy":
                values.append(result.pop(part))
                continue
            raw = [result.pop(part+"__rex_"+field) for field in _FIELDS]
            for index, column in enumerate(raw[:4]):
                if any(not isinstance(v, (Integral, np.bool_)) for v in column):
                    raise ValueError("SQL exact coefficient fields require integer values")
                if index in (2, 3) and any(int(v) not in (0, 1) for v in column):
                    raise ValueError("invalid SQL exact presence or kind")
            bigint = []
            for position, (numerator, denominator) in enumerate(zip(raw[4], raw[5], strict=True)):
                if (numerator is None) != (denominator is None):
                    raise ValueError("incomplete SQL bigint escape")
                if numerator is not None:
                    n, d = Cursor(numerator), Cursor(denominator)
                    bigint.append((position, n.integer(), d.integer()))
                    n.finish(); d.finish()
            values.append(ExactArray(*(np.asarray(array, dtype=dtype) for array, dtype in
                                     zip(raw[:4], ("<i8", "<i8", "bool", "u1"), strict=True)), tuple(bigint)).values())
        result[name] = np.column_stack(values) if info["split"] else values[0]
    return result
