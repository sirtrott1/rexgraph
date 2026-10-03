"""Declared exact Arrow columns shared by Parquet writers and record readers.

Fixed width coefficients use typed struct children. Only oversized coefficients
use minimal binary integer frames; presence and integer/rational kind are explicit.
Arrow and Parquet never choose a floating representation for an exact column.
"""
from __future__ import annotations

from fractions import Fraction
import json
from numbers import Integral

import numpy as np

from rexgraph._binary import Cursor, integer
from rexgraph.exact_array import ExactArray
from rexgraph.value import Absent

EXACT_CODEC = "exact-array-v1"


def exact_column(value):
    """Recognize declared exact carriers, without interpreting numeric text."""
    if isinstance(value, ExactArray):
        return value
    array = np.asarray(value)
    if array.dtype.kind == "O" and any(
        v is Absent or isinstance(v, Fraction)
        or (isinstance(v, Integral) and not isinstance(v, (bool, np.bool_))
            and not -(2**63) <= int(v) < 2**63) for v in array.flat
    ):
        return ExactArray.from_values(array)
    return array


def exact_to_arrow(value):
    import pyarrow as pa
    if not isinstance(value, ExactArray) or len(value.shape) != 1:
        raise TypeError("an exact Arrow column requires a one-dimensional ExactArray")
    big_num, big_den = [None]*value.size, [None]*value.size
    for i, n, d in value.bigint:
        big_num[i], big_den[i] = integer(n), integer(d)
    return pa.StructArray.from_arrays(
        [pa.array(value.numerator, type=pa.int64()),
         pa.array(value.denominator, type=pa.int64()),
         pa.array(value.presence, type=pa.bool_()),
         pa.array(value.kind, type=pa.uint8()),
         pa.array(big_num, type=pa.binary()), pa.array(big_den, type=pa.binary())],
        names=["numerator", "denominator", "presence", "kind", "big_numerator", "big_denominator"],
    )


def exact_from_arrow(column):
    import pyarrow as pa
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    expected = pa.struct([
        ("numerator", pa.int64()), ("denominator", pa.int64()),
        ("presence", pa.bool_()), ("kind", pa.uint8()),
        ("big_numerator", pa.binary()), ("big_denominator", pa.binary()),
    ])
    if column.type != expected or column.null_count:
        raise ValueError("invalid exact Arrow column schema or null record")
    children = [column.field(i) for i in range(6)]
    if any(child.null_count for child in children[:4]):
        raise ValueError("exact Arrow coefficient/presence fields cannot be null")
    big = []
    for i, (n, d) in enumerate(zip(children[4].to_pylist(), children[5].to_pylist(), strict=True)):
        if (n is None) != (d is None):
            raise ValueError("incomplete exact Arrow bigint escape")
        if n is not None:
            nc, dc = Cursor(n), Cursor(d)
            big.append((i, nc.integer(), dc.integer()))
            nc.finish()
            dc.finish()
    return ExactArray(*(np.asarray(child.to_numpy(zero_copy_only=False), dtype=dtype)
                        for child, dtype in zip(children[:4], ("<i8", "<i8", "bool", "u1"), strict=True)), tuple(big))


def _column(value, index):
    if not isinstance(value, ExactArray):
        return value[:, index]
    width = value.shape[1]
    return ExactArray(*(getattr(value, name)[:, index] for name in
                        ("numerator", "denominator", "presence", "kind")),
                      tuple((i//width, n, d) for i, n, d in value.bigint if i % width == index))


def physical_columns(arrays):
    """Validate logical columns and produce physical Arrow columns and their schema."""
    import pyarrow as pa
    columns, declarations = {}, {}
    logical_names = set(arrays)
    lengths = set()
    for name, raw in arrays.items():
        if type(name) is not str or not name:
            raise ValueError("Parquet column names must be nonempty strings")
        array = exact_column(raw)
        shape = array.shape
        if len(shape) not in (1, 2) or (len(shape) == 2 and shape[1] == 0):
            raise ValueError(f"column {name!r} must have one or two nonempty dimensions")
        lengths.add(shape[0])
        exact = isinstance(array, ExactArray)
        if len(shape) == 2 or exact:
            declarations[name] = {"shape": list(shape), "split": len(shape) == 2}
            if exact:
                declarations[name]["codec"] = EXACT_CODEC
        for index in range(shape[1] if len(shape) == 2 else 1):
            physical = f"{name}_{index}" if len(shape) == 2 else name
            if physical in columns or (physical in logical_names and physical != name):
                raise ValueError(f"logical Parquet column {name!r} collides with {physical!r}")
            part = _column(array, index) if len(shape) == 2 else array
            columns[physical] = exact_to_arrow(part) if exact else pa.array(part)
    if len(lengths) > 1:
        raise ValueError("Parquet columns must have equal row counts")
    return columns, declarations


def column_declarations(schema):
    raw = (schema.metadata or {}).get(b"rex_col_meta")
    if raw is None:
        return {}
    declarations = json.loads(raw.decode("utf-8"))
    if not isinstance(declarations, dict):
        raise ValueError("invalid logical Parquet column declarations")
    for name, info in declarations.items():
        if (type(name) is not str or not isinstance(info, dict) or set(info) - {"shape", "split", "codec"}
                or type(info.get("split")) is not bool or not isinstance(info.get("shape"), list)
                or len(info["shape"]) != (2 if info["split"] else 1)
                or any(type(n) is not int or n < 0 for n in info["shape"])
                or (info["split"] and info["shape"][1] == 0)
                or info.get("codec", EXACT_CODEC) != EXACT_CODEC):
            raise ValueError("unsupported logical Parquet column declaration")
    return declarations


def decoded_columns(table, declarations, *, columns=None):
    """Decode projected physical columns, also for sliced streaming batches."""
    result, consumed = {}, set()
    available = set(table.column_names)
    codecs = {}
    for name, info in declarations.items():
        physical = [f"{name}_{j}" for j in range(info["shape"][1])] if info["split"] else [name]
        for cn in physical:
            codecs[cn] = info.get("codec")
        if columns is not None and name not in columns:
            continue
        present = available.intersection(physical)
        if not present:
            continue
        if present != set(physical):
            raise ValueError(f"incomplete physical projection for logical column {name!r}")
        parts = [exact_from_arrow(table.column(cn)).values() if codecs[cn] == EXACT_CODEC
                 else table.column(cn).to_numpy(zero_copy_only=False) for cn in physical]
        result[name] = np.column_stack(parts) if info["split"] else parts[0]
        consumed.update(physical)
    for name in table.column_names:
        if name not in consumed and (columns is None or name in columns):
            result[name] = (exact_from_arrow(table.column(name)).values() if codecs.get(name) == EXACT_CODEC
                            else table.column(name).to_numpy(zero_copy_only=False))
    return result


def projected_columns(schema, declarations, columns):
    if columns is None:
        return None
    available = set(schema.names)
    result = []
    for name in columns:
        info = declarations.get(name)
        candidates = ([f"{name}_{j}" for j in range(info["shape"][1])]
                      if info is not None and info["split"] else [name])
        for candidate in candidates:
            if candidate in available and candidate not in result:
                result.append(candidate)
    return result
