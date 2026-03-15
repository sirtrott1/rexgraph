# rexgraph/io/arrow_bridge.py
"""
Arrow/IPC bridge for RexGraph arrays and analysis results.

Provides zero-copy columnar export of RexGraph data through Apache Arrow,
suitable for interop with Polars, DuckDB, Spark, and any Arrow-compatible
tool.

Core API::

    # RexGraph <-> Arrow
    table = rex_to_arrow(rex)
    rex   = arrow_to_rex(table)

    # Named array dicts <-> Arrow IPC files
    write_arrow_ipc({"L0": rex.L0, "layout": rex.layout}, "data.arrow")
    arrays = read_arrow_ipc("data.arrow")

    # Streaming large files
    for batch in read_arrow_batches("data.arrow"):
        process(batch["L0"])

All `pyarrow` imports are lazy - the module can be imported without
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
- The `rex_to_arrow` function stores the full `from_dict()`
  contract plus optional computed properties, enabling reconstruction
  via `arrow_to_rex`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

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
    arrays: Dict[str, NDArray],
    *,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convert a dict of NumPy arrays to a `pyarrow.Table`.

    Each array becomes one or two columns (two for complex dtype).
    Original shapes and dtypes are stored in schema metadata under
    `rex_array_meta` so arrow_to_arrays() can reconstruct them
    exactly.

    Parameters
    ----------
    arrays : dict
        Mapping of name -> ndarray.
    metadata : dict, optional
        Extra metadata embedded in the Arrow schema.

    Returns
    -------
    pyarrow.Table
    """
    pa, _ = _pa()

    columns: Dict[str, np.ndarray] = {}
    array_meta: Dict[str, dict] = {}

    # All columns must have the same length for a valid Arrow table.
    # We pad shorter arrays with NaN/0 to the max length.
    max_len = 0

    for name, arr in arrays.items():
        arr = np.asarray(arr)
        array_meta[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "is_complex": bool(np.iscomplexobj(arr)),
        }
        flat = arr.ravel()
        max_len = max(max_len, len(flat))

        if np.iscomplexobj(arr):
            columns[f"{name}__real"] = flat.real
            columns[f"{name}__imag"] = flat.imag
        else:
            columns[name] = flat

    # Pad to uniform length
    pa_columns: Dict[str, Any] = {}
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
        b"rex_array_meta": json.dumps(array_meta).encode("utf-8"),
    }
    if metadata:
        schema_meta[b"rex_user_meta"] = json.dumps(
            metadata, default=_json_default
        ).encode("utf-8")

    return table.replace_schema_metadata(schema_meta)


def arrow_to_arrays(table) -> Dict[str, np.ndarray]:
    """Convert a `pyarrow.Table` back to a dict of NumPy arrays.

    Reconstructs original shapes, dtypes, and complex values from
    the `rex_array_meta` schema metadata written by
    arrays_to_arrow().

    Parameters
    ----------
    table : pyarrow.Table

    Returns
    -------
    dict of name -> ndarray
    """
    pa, _ = _pa()

    schema_meta = table.schema.metadata or {}
    array_meta: Dict[str, dict] = {}
    if b"rex_array_meta" in schema_meta:
        array_meta = json.loads(
            schema_meta[b"rex_array_meta"].decode("utf-8")
        )

    # Read all columns into numpy
    raw: Dict[str, np.ndarray] = {}
    for col_name in table.column_names:
        raw[col_name] = table.column(col_name).to_numpy()

    result: Dict[str, np.ndarray] = {}
    consumed: set = set()

    for name, info in array_meta.items():
        shape = tuple(info["shape"])
        dtype = np.dtype(info["dtype"])
        n_elem = 1
        for s in shape:
            n_elem *= s
        is_complex = info.get("is_complex", False)

        if is_complex:
            rk = f"{name}__real"
            ik = f"{name}__imag"
            if rk in raw and ik in raw:
                arr = raw[rk][:n_elem] + 1j * raw[ik][:n_elem]
                result[name] = arr.reshape(shape).astype(dtype)
                consumed.update((rk, ik))
        else:
            if name in raw:
                result[name] = raw[name][:n_elem].reshape(shape).astype(dtype)
                consumed.add(name)

    # Passthrough any unknown columns
    for col_name, col_arr in raw.items():
        if col_name not in consumed:
            result[col_name] = col_arr

    return result


# RexGraph <-> Arrow


def rex_to_arrow(
    rex,
    *,
    include: Optional[List[str]] = None,
):
    """Export a RexGraph as a `pyarrow.Table`.

    By default, stores the minimal reconstruction data (the
    `from_dict()` contract).  Pass *include* to add computed
    properties.

    Parameters
    ----------
    rex : RexGraph
        The graph to export.
    include : list of str, optional
        Additional properties to include as columns, e.g.
        `["layout", "betti", "eigenvalues_L0", "edge_types"]`.

    Returns
    -------
    pyarrow.Table

    Examples
    --------
    >>> table = rex_to_arrow(rex)
    >>> table = rex_to_arrow(rex, include=["layout", "eigenvalues_L0"])
    """
    arrays: Dict[str, NDArray] = {}

    # Core reconstruction arrays
    arrays["boundary_ptr"] = rex._boundary_ptr
    arrays["boundary_idx"] = rex._boundary_idx
    arrays["B2_col_ptr"] = rex._B2_col_ptr
    arrays["B2_row_idx"] = rex._B2_row_idx
    arrays["B2_vals"] = rex._B2_vals
    if rex._w_E is not None:
        arrays["w_E"] = rex._w_E

    # Graph metadata (stored in Arrow schema metadata)
    graph_meta: Dict[str, Any] = {
        "object_type": "RexGraph",
        "nV": int(rex.nV),
        "nE": int(rex.nE),
        "nF": int(rex.nF),
        "directed": bool(rex._directed),
        "dimension": int(rex.dimension),
    }
    if rex._w_boundary:
        graph_meta["w_boundary"] = {
            str(k): v for k, v in rex._w_boundary.items()
        }

    # Optional computed properties
    if include:
        for prop in include:
            try:
                val = getattr(rex, prop)
                if isinstance(val, np.ndarray):
                    arrays[prop] = val
                elif isinstance(val, tuple):
                    # Betti, fiedler_overlap, etc.
                    if prop == "betti":
                        graph_meta["betti"] = list(val)
                    elif prop == "fiedler_overlap":
                        graph_meta["fiedler_overlap_value"] = float(val[0])
                        arrays["fiedler_overlap_vector"] = val[1]
                    else:
                        # Generic tuple of arrays
                        for i, item in enumerate(val):
                            if isinstance(item, np.ndarray):
                                arrays[f"{prop}_{i}"] = item
                elif isinstance(val, (int, float, bool)):
                    graph_meta[prop] = val
            except Exception:
                pass  # skip properties that fail

    return arrays_to_arrow(arrays, metadata=graph_meta)


def arrow_to_rex(table):
    """Reconstruct a RexGraph from a `pyarrow.Table`.

    The table must have been created by rex_to_arrow() (or
    contain the same columns and metadata).

    Parameters
    ----------
    table : pyarrow.Table

    Returns
    -------
    RexGraph
    """
    from ..graph import RexGraph

    pa, _ = _pa()
    arrays = arrow_to_arrays(table)

    # Read graph metadata
    schema_meta = table.schema.metadata or {}
    graph_meta: dict = {}
    if b"rex_user_meta" in schema_meta:
        graph_meta = json.loads(
            schema_meta[b"rex_user_meta"].decode("utf-8")
        )

    kw: dict = {
        "boundary_ptr": arrays["boundary_ptr"],
        "boundary_idx": arrays["boundary_idx"],
        "directed": graph_meta.get("directed", False),
    }

    for name in ("B2_col_ptr", "B2_row_idx", "B2_vals"):
        if name in arrays:
            kw[name] = arrays[name]

    if "w_E" in arrays:
        kw["w_E"] = arrays["w_E"]

    wb = graph_meta.get("w_boundary")
    if wb:
        kw["w_boundary"] = {int(k): v for k, v in wb.items()}

    return RexGraph(**kw)


# IPC file I/O


def write_arrow_ipc(
    arrays: Dict[str, NDArray],
    path: Union[str, os.PathLike],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a dict of arrays to an Arrow IPC file.

    Parameters
    ----------
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

    with pa.OSFile(fpath, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_arrow_ipc(path: Union[str, os.PathLike]) -> Dict[str, np.ndarray]:
    """Read arrays from an Arrow IPC file.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
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
    path: Union[str, os.PathLike],
    *,
    batch_rows: int = 100_000,
) -> Iterator[Dict[str, np.ndarray]]:
    """Stream an Arrow IPC file as batches of arrays.

    Each yielded dict contains the same array names as a full
    read_arrow_ipc() call, but with fewer rows.  This is useful
    for processing large Laplacian or eigenvector exports without
    loading everything into memory.

    Parameters
    ----------
    path : str or path-like
        Arrow IPC file.
    batch_rows : int
        Target rows per batch.

    Yields
    ------
    dict of name -> ndarray
    """
    pa, ipc = _pa()

    fpath = os.fspath(path)
    with pa.OSFile(fpath, "rb") as source:
        reader = ipc.open_file(source)

        pending_batches = []
        pending_rows = 0

        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pending_batches.append(batch)
            pending_rows += batch.num_rows

            if pending_rows >= batch_rows:
                table = pa.Table.from_batches(
                    pending_batches, schema=reader.schema
                )
                yield arrow_to_arrays(table)
                pending_batches = []
                pending_rows = 0

        if pending_batches:
            table = pa.Table.from_batches(
                pending_batches, schema=reader.schema
            )
            yield arrow_to_arrays(table)


# Helpers


def _json_default(o):
    """JSON fallback for numpy types."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
