# rexgraph/io/arrow_bridge.py
"""
Arrow/IPC bridge for RexGraph arrays and analysis results.

Provides zero copy columnar export of RexGraph data through Apache Arrow,
suitable for interop with Polars, DuckDB, Spark, and any Arrow compatible
tool.

Core API:

    # RexGraph <-> Arrow
    table = rex_to_arrow(rex)
    rex   = arrow_to_rex(table)

    # Named array dicts <-> Arrow IPC files
    write_arrow_ipc({"L0": rex.L0, "layout": rex.layout}, "data.arrow")
    arrays = read_arrow_ipc("data.arrow")

    # Streaming large files
    for batch in read_arrow_batches("data.arrow"):
        process(batch["L0"])

All `pyarrow` imports are lazy, so the module can be imported without
pyarrow installed.  An `ImportError` is raised only when a function
is actually called.

Design notes
~~~~~~~~~~~~
- RexGraph arrays are dense `NDArray` (not scipy sparse), so Arrow
  columns map directly to 1D or flattened-2D arrays.
- Complex arrays (Hamiltonian eigenvectors, etc.) are split into
  `<name>_real` / `<name>_imag` columns.
- 2D array shapes are stored in Arrow schema metadata under the key
  `rex_array_meta` so round-trip reshape is exact.
- `rex_to_arrow`/`arrow_to_rex` delegate to the canonical
  `rex_state.to_state()`/`from_state()`: every state tensor is stored
  through `arrays_to_arrow`, and the state header is carried as this
  table's `rex_user_meta` schema metadata, enabling exact
  reconstruction.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "rex_to_arrow",
    "arrow_to_rex",
    "arrays_to_arrow",
    "arrow_to_arrays",
    "write_arrow_ipc",
    "read_arrow_ipc",
    "read_arrow_batches",
]


# Lazy pyarrow import


def _pa():
    """Lazily import pyarrow + ipc.  Raises ImportError if missing."""
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        return pa, ipc
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Arrow features: pip install pyarrow"
        ) from exc


# Low-level: dict-of-arrays <-> Arrow Table


def arrays_to_arrow(
    arrays: dict[str, NDArray],
    *,
    metadata: dict[str, Any] | None = None,
):
    """Convert a dict of NumPy arrays to a `pyarrow.Table`.

    Each array becomes one or two columns (two for complex dtype).
    Original shapes and dtypes are stored in schema metadata under
    `rex_array_meta` so arrow_to_arrays() can reconstruct them
    exactly.

    Parameters
    arrays : dict
        Mapping of name -> ndarray.
    metadata : dict, optional
        Extra metadata embedded in the Arrow schema.

    Returns
    pyarrow.Table
    """
    pa, _ = _pa()

    columns: dict[str, np.ndarray] = {}
    array_meta: dict[str, dict] = {}

    # All columns must have the same length for a valid Arrow table.
    # We pad shorter arrays with NaN/0 to the max length.
    max_len = 0

    for name, arr in arrays.items():
        if type(name) is not str or not name:
            raise ValueError("Arrow array names must be nonempty strings")
        arr = np.asarray(arr)
        array_meta[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "is_complex": bool(np.iscomplexobj(arr)),
        }
        flat = arr.astype(arr.dtype.newbyteorder("="), copy=False).ravel()
        max_len = max(max_len, len(flat))

        if np.iscomplexobj(arr):
            if any(f"{name}{suffix}" in arrays or f"{name}{suffix}" in columns
                   for suffix in ("__real", "__imag")):
                raise ValueError(f"Arrow complex column {name!r} collides with a physical column")
            columns[f"{name}__real"] = flat.real
            columns[f"{name}__imag"] = flat.imag
        else:
            if name in columns:
                raise ValueError(f"Arrow column {name!r} collides with a complex column")
            columns[name] = flat

    # Pad to uniform length
    pa_columns: dict[str, Any] = {}
    for col_name, col_arr in columns.items():
        if len(col_arr) < max_len:
            padded = np.empty(max_len, dtype=col_arr.dtype)
            padded[: len(col_arr)] = col_arr
            padded[len(col_arr) :] = 0
            col_arr = padded
        pa_columns[col_name] = pa.array(col_arr)

    table = pa.table(pa_columns)

    # Attach metadata
    schema_meta = {
        b"rex_array_meta": _dumps(array_meta).encode("utf-8"),
    }
    if metadata:
        schema_meta[b"rex_user_meta"] = _dumps(metadata).encode("utf-8")

    return table.replace_schema_metadata(schema_meta)


def _array_metadata(schema):
    """Validate declared array shapes and their distinct physical columns."""
    if len(set(schema.names)) != len(schema.names):
        raise ValueError("Arrow columns must have distinct names")
    metadata = json.loads((schema.metadata or {}).get(b"rex_array_meta", b"{}"))
    if type(metadata) is not dict:
        raise ValueError("invalid Arrow array metadata")
    claimed = set()
    for name, info in metadata.items():
        if (type(name) is not str or not name or type(info) is not dict
                or set(info) != {"shape", "dtype", "is_complex"}
                or type(info["shape"]) is not list or len(info["shape"]) > 64
                or any(type(n) is not int or n < 0 for n in info["shape"])
                or type(info["is_complex"]) is not bool):
            raise ValueError("invalid Arrow array declaration")
        dtype = np.dtype(info["dtype"])
        if (dtype.kind == "c") != info["is_complex"]:
            raise ValueError("Arrow complex declaration differs from its dtype")
        physical = {name + "__real", name + "__imag"} if info["is_complex"] else {name}
        if claimed.intersection(physical) or not physical <= set(schema.names):
            raise ValueError("missing or colliding Arrow physical columns")
        claimed.update(physical)
    return metadata


def arrow_to_arrays(table) -> dict[str, np.ndarray]:
    """Convert a `pyarrow.Table` back to a dict of NumPy arrays.

    Reconstructs original shapes, dtypes, and complex values from
    the `rex_array_meta` schema metadata written by
    arrays_to_arrow().

    Parameters
    table : pyarrow.Table

    Returns
    dict of name -> ndarray
    """
    from math import prod
    _pa()
    array_meta = _array_metadata(table.schema)
    result, consumed = {}, set()
    for name, info in array_meta.items():
        shape, dtype = tuple(info["shape"]), np.dtype(info["dtype"])
        n_elem = prod(shape)
        if n_elem > table.num_rows:
            raise ValueError("Arrow array shape exceeds its stored rows")
        # Slice before materializing NumPy data, so a short tensor never reads
        # the padding belonging to a longer unrelated tensor.
        if info["is_complex"]:
            rk, ik = name + "__real", name + "__imag"
            array = np.empty(n_elem, dtype=dtype)
            array.real = table.column(rk).slice(0, n_elem).to_numpy()
            array.imag = table.column(ik).slice(0, n_elem).to_numpy()
            consumed.update((rk, ik))
        else:
            array = table.column(name).slice(0, n_elem).to_numpy().astype(dtype, copy=False)
            consumed.add(name)
        result[name] = array.reshape(shape)
    for name in table.column_names:
        if name not in consumed:
            result[name] = table.column(name).to_numpy()
    return result


# RexGraph <-> Arrow


def rex_to_arrow(rex):
    """Export a RexGraph as a `pyarrow.Table`.

    Delegates the reconstruction contract to the canonical
    `rex_state.to_state()`: every state tensor is passed straight through
    to the existing `arrays_to_arrow()` low-level bridge (raw tensor
    names, which may contain '/' for nested rexes; arrow field names are
    not filesystem paths so no encoding is needed), which already knows
    how to pad ragged, differently-shaped tensors into one Arrow table and
    record each one's true shape/dtype in the `rex_array_meta` schema
    entry. The state header is passed through unchanged as this table's
    `rex_user_meta` metadata.

    Parameters
    rex : RexGraph
        The graph to export.

    Returns
    pyarrow.Table

    Examples
    >>> table = rex_to_arrow(rex)
    """
    from rexgraph.state import to_state

    st = to_state(rex)
    return arrays_to_arrow(st.tensors, metadata=st.header)


def arrow_to_rex(table):
    """Reconstruct a RexGraph from a `pyarrow.Table`.

    The table must have been created by rex_to_arrow() (or contain the
    same columns and `rex_user_meta` schema metadata). The tensors come
    back through the existing `arrow_to_arrays()` low-level bridge; the
    state header comes back from `rex_user_meta`. Delegates
    reconstruction to `rex_state.from_state()`.

    Parameters
    table : pyarrow.Table

    Returns
    RexGraph
    """
    from rexgraph.state import RexState, from_state

    schema_meta = table.schema.metadata or {}
    hdr = json.loads(schema_meta[b"rex_user_meta"]) if b"rex_user_meta" in schema_meta else {}
    tensors = arrow_to_arrays(table)
    return from_state(RexState(tensors, hdr))


# IPC file I/O


def write_arrow_ipc(
    arrays: dict[str, NDArray],
    path: str | os.PathLike,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a dict of arrays to an Arrow IPC file.

    Parameters
    arrays : dict
        Mapping of name -> ndarray.
    path : str or path-like
        Output file path.
    metadata : dict, optional
        Extra metadata to embed.
    """
    pa, ipc = _pa()

    table = arrays_to_arrow(arrays, metadata=metadata)
    fpath = os.fspath(path)

    from .publication import staged_publication
    with staged_publication(fpath) as staged:
        with pa.OSFile(str(staged), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_arrow_ipc(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Read arrays from an Arrow IPC file.

    Parameters
    path : str or path-like

    Returns
    dict of name -> ndarray
    """
    pa, ipc = _pa()

    fpath = os.fspath(path)
    with pa.OSFile(fpath, "rb") as source:
        reader = ipc.open_file(source)
        table = reader.read_all()

    return arrow_to_arrays(table)


# Streaming reads


def read_arrow_batches(
    path: str | os.PathLike,
    *,
    batch_rows: int = 100_000,
) -> Iterator[dict[str, np.ndarray]]:
    """Stream an Arrow IPC file as batches of arrays.

    Each yielded dict contains the same array names as a full
    read_arrow_ipc() call, but with fewer rows.  This is useful
    for processing large Laplacian or eigenvector exports without
    loading everything into memory.

    Parameters
    path : str or path-like
        Arrow IPC file.
    batch_rows : int
        Positive bound on flattened file rows per chunk. Multidimensional
        arrays retain partial rows until complete, so one output may also
        include a previously buffered row. Scalars yield a length-one vector.

    Yields
    dict of name -> ndarray
    """
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError("Arrow batch_rows must be a positive integer")
    pa, ipc = _pa()
    with pa.memory_map(os.fspath(path), "r") as source:
        reader = ipc.open_file(source)
        array_meta = _array_metadata(reader.schema)
        tails = {}
        row_offset = 0
        emitted = False
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            for start in range(0, batch.num_rows, batch_rows):
                piece = batch.slice(start, batch_rows)
                table = pa.Table.from_batches([piece], schema=reader.schema)
                yield _arrow_batch_to_arrays(table, array_meta, row_offset, tails=tails)
                row_offset += piece.num_rows
                emitted = True
        from math import prod
        if (any(value.size for value in tails.values())
                or any(prod(info["shape"]) > row_offset for info in array_meta.values())):
            raise ValueError("Arrow stream ended before its complete declared arrays")
        if not emitted:
            table = pa.Table.from_batches([], schema=reader.schema)
            yield _arrow_batch_to_arrays(table, array_meta, 0, tails=tails)


def _arrow_batch_to_arrays(
    table, array_meta: dict[str, dict], row_offset: int, *, tails=None
) -> dict[str, np.ndarray]:
    """Reconstruct arrays from a single streamed sub-table.

    Unlike :func:`arrow_to_arrays`, which assumes the table contains every
    flattened row of every array, this reshapes each column against only the
    rows present in this batch.  ``row_offset`` is the number of flattened
    rows already emitted by earlier batches; it lets us skip padding rows that
    fall past an array's true element count.  Concatenating the per-batch
    results along axis 0 reproduces the full arrays from
    :func:`arrow_to_arrays`.

    Parameters
    table : pyarrow.Table
        The record batch(es) for this streamed chunk.
    array_meta : dict
        The `rex_array_meta` shape/dtype map for the whole file.
    row_offset : int
        Count of flattened rows emitted by previous batches.

    Returns
    dict of name -> ndarray
    """
    raw: dict[str, np.ndarray] = {}
    for col_name in table.column_names:
        raw[col_name] = table.column(col_name).to_numpy()

    tails = {} if tails is None else tails
    batch_rows = table.num_rows
    result: dict[str, np.ndarray] = {}
    consumed: set = set()

    for name, info in array_meta.items():
        shape = tuple(info["shape"])
        dtype = np.dtype(info["dtype"])
        is_complex = info.get("is_complex", False)

        n_elem = 1
        for s in shape:
            n_elem *= s
        inner = 1  # elements per leading-axis row (1 for 1-D arrays)
        for s in shape[1:]:
            inner *= s

        # Non-padding elements of this array present in this batch: the global
        # flat range [row_offset, row_offset + batch_rows) intersected with
        # this array's real extent [0, n_elem).
        valid = max(0, min(row_offset + batch_rows, n_elem) - row_offset)
        def rows(values):
            previous = tails.pop(name, None)
            if previous is not None and previous.size:
                values = np.concatenate((previous, values))
            if inner == 0:
                return np.empty(shape if row_offset == 0 else (0,) + shape[1:], dtype=dtype)
            complete = values.size // inner * inner
            tails[name] = values[complete:].copy()
            return values[:complete].reshape((complete // inner,) + shape[1:]).astype(dtype)

        if is_complex:
            rk = f"{name}__real"
            ik = f"{name}__imag"
            if rk in raw and ik in raw:
                arr = np.empty(valid, dtype=dtype)
                arr.real = raw[rk][:valid]
                arr.imag = raw[ik][:valid]
                result[name] = rows(arr)
                consumed.update((rk, ik))
        else:
            if name in raw:
                result[name] = rows(raw[name][:valid])
                consumed.add(name)

    # Passthrough any unknown columns (no metadata) as-is.
    for col_name, col_arr in raw.items():
        if col_name not in consumed:
            result[col_name] = col_arr

    return result


# Helpers


#: the one encoder (rexgraph.io._compat). Re-exported under the local name so the
#: existing call sites keep working; `dumps` is what applies the non-finite policy.
from ._compat import dumps as _dumps
