# rexgraph/io/parquet_bridge.py
"""
Parquet bridge for RexGraph tabular data.

Exports the mathematically meaningful structures of a relational complex
as columnar Parquet tables.  Each table type maps to a specific part of
the algebraic/topological framework:

Boundary table - the general boundary operator d_1
(Definition 3.1).  One row per (edge, boundary_vertex) pair - handles
standard, self-loop, branching, and witness edges (Definition 3.2).

Edge table - per-edge data: source/target (for standard edges),
edge type (Definition 3.2), weight, and optional Hodge components
(Theorem 3.8/4.5).

Vertex table - per-vertex data: layout from overlap-correct
spectral embedding (Definition 6.7), degree from L_0,
Fiedler vector entries, etc.

Face table - the B_2 boundary operator in CSC format
(Definition 4.1).  One row per nonzero in B_2, giving
(face_idx, edge_idx, orientation).

Persistence table - persistence pairs from column reduction over
Z/2.  Columns: birth, death, dim, birth_cell,
death_cell, lifetime.

Filtration table - filtration values f: C_k -> R
on the chain complex.

Metrics table - generic per-cell numeric metrics.

All `pyarrow` imports are lazy.  No pandas dependency.

Usage::

    from rexgraph.io.parquet_bridge import (
        write_edge_table, write_boundary_table,
        write_persistence_table, write_face_table,
    )

    write_edge_table(rex, "edges.parquet")
    write_boundary_table(rex, "boundary.parquet")
    write_persistence_table(rex.persistence(*rex.filtration("dimension")),
                            "persistence.parquet")
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "write_parquet",
    "read_parquet",
    "write_boundary_table",
    "read_boundary_table",
    "write_edge_table",
    "read_edge_table",
    "write_vertex_table",
    "read_vertex_table",
    "write_face_table",
    "read_face_table",
    "write_persistence_table",
    "read_persistence_table",
    "write_filtration_table",
    "read_filtration_table",
    "write_metrics_table",
    "read_metrics_table",
    "read_parquet_batches",
]


# Lazy import


def _pq():
    """Lazily import pyarrow + parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Parquet features: pip install pyarrow"
        ) from exc


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


# Edge type names matching types.py EdgeType enum (Definition 3.2)
_EDGE_TYPE_NAMES = {0: "standard", 1: "self_loop", 2: "branching", 3: "witness"}


# Generic Parquet I/O


def write_parquet(
    data: Dict[str, NDArray],
    path: Union[str, os.PathLike],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a dict of equal-length 1D arrays to Parquet.

    For 2D arrays, columns are split into `{name}_0`, `{name}_1`,
    etc. and reassembled by read_parquet().
    """
    pa, pq = _pq()

    columns: Dict[str, Any] = {}
    col_meta: Dict[str, dict] = {}

    for name, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            columns[name] = pa.array(arr)
        elif arr.ndim == 2:
            col_meta[name] = {"shape": list(arr.shape), "split": True}
            for j in range(arr.shape[1]):
                columns[f"{name}_{j}"] = pa.array(arr[:, j])
        else:
            raise ValueError(f"Column '{name}': {arr.ndim}D not supported")

    table = pa.table(columns)

    schema_meta: Dict[bytes, bytes] = {}
    if col_meta:
        schema_meta[b"rex_col_meta"] = json.dumps(col_meta).encode("utf-8")
    if metadata:
        schema_meta[b"rex_metadata"] = json.dumps(
            metadata, default=_json_default
        ).encode("utf-8")
    if schema_meta:
        table = table.replace_schema_metadata(schema_meta)

    pq.write_table(table, os.fspath(path))


def read_parquet(
    path: Union[str, os.PathLike],
    *,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read a Parquet file into a dict of arrays.

    Reassembles 2D arrays that were split by write_parquet().
    """
    pa, pq = _pq()

    table = pq.read_table(os.fspath(path))

    schema_meta = table.schema.metadata or {}
    col_meta: Dict[str, dict] = {}
    if b"rex_col_meta" in schema_meta:
        col_meta = json.loads(schema_meta[b"rex_col_meta"].decode("utf-8"))

    result: Dict[str, np.ndarray] = {}
    consumed: set = set()

    for name, info in col_meta.items():
        if columns and name not in columns:
            continue
        if info.get("split"):
            n_cols = info["shape"][1]
            parts = []
            for j in range(n_cols):
                cn = f"{name}_{j}"
                if cn in table.column_names:
                    parts.append(table.column(cn).to_numpy())
                    consumed.add(cn)
            if parts:
                result[name] = np.column_stack(parts)

    for cn in table.column_names:
        if cn in consumed:
            continue
        if columns and cn not in columns:
            continue
        result[cn] = table.column(cn).to_numpy()

    return result


def _read_metadata(path: Union[str, os.PathLike]) -> dict:
    """Read rex_metadata from a Parquet file without loading data."""
    pa, pq = _pq()
    schema = pq.read_schema(os.fspath(path))
    meta = schema.metadata or {}
    if b"rex_metadata" in meta:
        return json.loads(meta[b"rex_metadata"].decode("utf-8"))
    return {}


# Boundary table (Definition 3.1 - the general boundary operator)


def write_boundary_table(
    rex,
    path: Union[str, os.PathLike],
) -> None:
    """Write the general boundary operator d_1 to Parquet.

    One row per (edge, boundary_vertex) pair.  This is the fundamental
    representation - it handles all edge types from Definition 3.2:

    - Standard edges: 2 rows per edge (source, target)
    - Self-loops: 2 rows, same vertex
    - Branching edges: geq 3 rows
    - Witness edges: 1 row

    Columns: `edge_idx`, `vertex_idx`, `position`
    (position within the edge's boundary).
    """
    bp = rex._boundary_ptr
    bi = rex._boundary_idx

    n_entries = int(bp[-1])
    edge_idx = np.empty(n_entries, dtype=np.int32)
    position = np.empty(n_entries, dtype=np.int32)

    for e in range(rex.nE):
        lo, hi = int(bp[e]), int(bp[e + 1])
        edge_idx[lo:hi] = e
        position[lo:hi] = np.arange(hi - lo, dtype=np.int32)

    data = {
        "edge_idx": edge_idx,
        "vertex_idx": bi[:n_entries].astype(np.int32),
        "position": position,
    }

    meta = {
        "rex_table_type": "boundary_table",
        "nV": int(rex.nV),
        "nE": int(rex.nE),
        "n_entries": n_entries,
        "directed": bool(rex._directed),
    }
    write_parquet(data, path, metadata=meta)


def read_boundary_table(
    path: Union[str, os.PathLike],
) -> Dict[str, Any]:
    """Read boundary table and reconstruct `boundary_ptr`/`boundary_idx`.

    Returns
    -------
    dict with `boundary_ptr`, `boundary_idx`, `nV`, `nE`,
    `directed`, and the raw columns.
    """
    raw = read_parquet(path)
    meta = _read_metadata(path)

    nE = meta.get("nE", int(raw["edge_idx"].max()) + 1)

    edge_idx = raw["edge_idx"]
    vertex_idx = raw["vertex_idx"]
    position = raw["position"]

    order = np.lexsort((position, edge_idx))
    vertex_idx_sorted = vertex_idx[order]

    boundary_ptr = np.zeros(nE + 1, dtype=np.int32)
    for e in edge_idx:
        boundary_ptr[e + 1] += 1
    np.cumsum(boundary_ptr, out=boundary_ptr)

    return {
        "boundary_ptr": boundary_ptr,
        "boundary_idx": vertex_idx_sorted.astype(np.int32),
        "nV": meta.get("nV"),
        "nE": nE,
        "directed": meta.get("directed", False),
        **raw,
    }


# Edge table (Definition 3.2 - per-edge properties)


def write_edge_table(
    rex,
    path: Union[str, os.PathLike],
    *,
    include: Optional[List[str]] = None,
) -> None:
    r"""Write per-edge data to Parquet.

    One row per edge.  Columns:

    - `edge_idx`
    - `source`, `target` - endpoints (`-1` for witness edges)
    - `boundary_size` - |supp(d_1(e))|
    - `edge_type` - code from EdgeType enum (Def 3.2)
    - `weight` - if weighted

    Parameters
    ----------
    rex : RexGraph
    path : str or path-like
    include : list of str, optional
        Extra per-edge arrays: `"hodge_gradient"`,
        `"hodge_curl"`, `"hodge_harmonic"` (Theorem 3.8/4.5).
    """
    bp = rex._boundary_ptr
    bi = rex._boundary_idx
    nE = rex.nE

    boundary_size = np.diff(bp).astype(np.int32)

    src = np.full(nE, -1, dtype=np.int32)
    tgt = np.full(nE, -1, dtype=np.int32)
    for e in range(nE):
        lo, hi = int(bp[e]), int(bp[e + 1])
        if hi - lo >= 2:
            src[e] = bi[lo]
            tgt[e] = bi[lo + 1]
        elif hi - lo == 1:
            src[e] = bi[lo]

    etypes = rex.edge_types

    data: Dict[str, NDArray] = {
        "edge_idx": np.arange(nE, dtype=np.int32),
        "source": src,
        "target": tgt,
        "boundary_size": boundary_size,
        "edge_type": etypes.astype(np.int32),
    }

    if rex._w_E is not None:
        data["weight"] = rex._w_E

    if include:
        w = rex.w_E if rex.w_E is not None else np.ones(nE)
        hodge_done = False
        grad = curl = harm = None

        for prop in include:
            if prop.startswith("hodge_") and not hodge_done:
                try:
                    grad, curl, harm = rex.hodge(w)
                    hodge_done = True
                except Exception:
                    pass

            if prop == "hodge_gradient" and grad is not None:
                data["hodge_gradient"] = grad
            elif prop == "hodge_curl" and curl is not None:
                data["hodge_curl"] = curl
            elif prop == "hodge_harmonic" and harm is not None:
                data["hodge_harmonic"] = harm
            else:
                try:
                    val = getattr(rex, prop)
                    if isinstance(val, np.ndarray) and val.shape[0] == nE:
                        data[prop] = val
                except Exception:
                    pass

    meta = {
        "rex_table_type": "edge_table",
        "nV": int(rex.nV),
        "nE": nE,
        "nF": int(rex.nF),
        "directed": bool(rex._directed),
    }
    write_parquet(data, path, metadata=meta)


def read_edge_table(path: Union[str, os.PathLike]) -> Dict[str, np.ndarray]:
    """Read edge table from Parquet."""
    return read_parquet(path)


# Vertex table


def write_vertex_table(
    rex,
    path: Union[str, os.PathLike],
    *,
    include: Optional[List[str]] = None,
) -> None:
    r"""Write per-vertex data to Parquet.

    Default columns:

    - `vertex_idx`
    - `degree` - from diag(L_0)
    - `x`, `y` - spectral layout (Definition 6.7)

    Parameters
    ----------
    rex : RexGraph
    path : str or path-like
    include : list of str, optional
        `"fiedler_vector_L0"`, `"eigenvalues_L0"`, `"layout_3d"`.
    """
    nV = rex.nV
    data: Dict[str, NDArray] = {
        "vertex_idx": np.arange(nV, dtype=np.int32),
    }

    try:
        data["degree"] = np.diag(rex.L0).astype(np.int32)
    except Exception:
        pass

    try:
        layout = rex.layout
        data["x"] = layout[:, 0]
        data["y"] = layout[:, 1]
    except Exception:
        pass

    if include:
        for prop in include:
            if prop == "layout_3d":
                try:
                    l3 = rex.layout_3d
                    data["x3"] = l3[:, 0]
                    data["y3"] = l3[:, 1]
                    data["z3"] = l3[:, 2]
                except Exception:
                    pass
            else:
                try:
                    val = getattr(rex, prop)
                    if isinstance(val, np.ndarray) and val.shape[0] == nV:
                        data[prop] = val
                except Exception:
                    pass

    meta = {"rex_table_type": "vertex_table", "nV": nV, "nE": int(rex.nE)}
    write_parquet(data, path, metadata=meta)


def read_vertex_table(path: Union[str, os.PathLike]) -> Dict[str, np.ndarray]:
    """Read vertex table from Parquet."""
    return read_parquet(path)


# Face table (Definition 4.1 - B2 boundary operator)


def write_face_table(
    rex,
    path: Union[str, os.PathLike],
) -> None:
    """Write the face boundary operator B_2 to Parquet.

    One row per nonzero entry in the CSC representation of B_2.
    The chain complex condition B_1 B_2 = 0 guarantees each
    face boundary is a cycle in the edge basis.

    Columns: `face_idx`, `edge_idx`, `orientation` (±1).
    """
    b2cp = rex._B2_col_ptr
    b2ri = rex._B2_row_idx
    b2v = rex._B2_vals

    nnz = int(b2cp[-1]) if len(b2cp) > 1 else 0
    if nnz == 0:
        data = {
            "face_idx": np.array([], dtype=np.int32),
            "edge_idx": np.array([], dtype=np.int32),
            "orientation": np.array([], dtype=np.float64),
        }
    else:
        face_idx = np.empty(nnz, dtype=np.int32)
        for f in range(rex.nF):
            lo, hi = int(b2cp[f]), int(b2cp[f + 1])
            face_idx[lo:hi] = f
        data = {
            "face_idx": face_idx,
            "edge_idx": b2ri[:nnz].astype(np.int32),
            "orientation": b2v[:nnz],
        }

    meta = {
        "rex_table_type": "face_table",
        "nE": int(rex.nE),
        "nF": int(rex.nF),
        "nnz_B2": nnz,
        "chain_valid": bool(rex.chain_valid) if rex.nF > 0 else True,
    }
    write_parquet(data, path, metadata=meta)


def read_face_table(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Read face table and reconstruct B2 CSC arrays.

    Returns `B2_col_ptr`, `B2_row_idx`, `B2_vals`, plus raw columns.
    """
    raw = read_parquet(path)
    meta = _read_metadata(path)
    nF = meta.get("nF", 0)
    nE = meta.get("nE", 0)

    if nF == 0 or len(raw.get("face_idx", [])) == 0:
        return {
            "B2_col_ptr": np.zeros(1, dtype=np.int32),
            "B2_row_idx": np.array([], dtype=np.int32),
            "B2_vals": np.array([], dtype=np.float64),
            "nF": 0, "nE": nE, **raw,
        }

    face_idx = raw["face_idx"]
    edge_idx = raw["edge_idx"]
    orientation = raw["orientation"]

    B2_col_ptr = np.zeros(nF + 1, dtype=np.int32)
    for f in face_idx:
        B2_col_ptr[f + 1] += 1
    np.cumsum(B2_col_ptr, out=B2_col_ptr)

    order = np.lexsort((edge_idx, face_idx))
    return {
        "B2_col_ptr": B2_col_ptr,
        "B2_row_idx": edge_idx[order].astype(np.int32),
        "B2_vals": orientation[order].astype(np.float64),
        "nF": nF, "nE": nE, **raw,
    }


# Persistence table (column reduction over Z/2)


def write_persistence_table(
    result: Any,
    path: Union[str, os.PathLike],
) -> None:
    r"""Write persistence pairs to Parquet.

    Accepts the output of `rex.persistence(filt_v, filt_e, filt_f)`
    - a dict with keys `pairs`, `essential`, `betti`.

    The `pairs` array has shape `(k, 5)`:
    `[birth, death, dim, birth_cell, death_cell]`.
    """
    if isinstance(result, dict):
        pairs = np.asarray(result["pairs"])
    elif hasattr(result, "pairs"):
        pairs = np.asarray(result.pairs)
    else:
        raise TypeError("Expected persistence result dict or PersistenceDiagram")

    data: Dict[str, NDArray] = {}
    if pairs.ndim == 2 and pairs.shape[1] >= 5:
        data["birth"] = pairs[:, 0]
        data["death"] = pairs[:, 1]
        data["dim"] = pairs[:, 2].astype(np.int32)
        data["birth_cell"] = pairs[:, 3].astype(np.int32)
        data["death_cell"] = pairs[:, 4].astype(np.int32)
        data["lifetime"] = pairs[:, 1] - pairs[:, 0]
    elif pairs.ndim == 2 and pairs.shape[1] >= 3:
        data["birth"] = pairs[:, 0]
        data["death"] = pairs[:, 1]
        data["dim"] = pairs[:, 2].astype(np.int32)
        data["lifetime"] = pairs[:, 1] - pairs[:, 0]
    else:
        raise ValueError(f"Unexpected pairs shape: {pairs.shape}")

    meta: Dict[str, Any] = {
        "rex_table_type": "persistence_table",
        "n_pairs": int(len(pairs)),
    }

    essential = result.get("essential") if isinstance(result, dict) else getattr(result, "essential", None)
    if essential is not None:
        essential = np.asarray(essential)
        meta["n_essential"] = int(len(essential))
        meta["essential"] = essential.tolist()

    betti = result.get("betti") if isinstance(result, dict) else getattr(result, "betti", None)
    if betti is not None:
        meta["betti"] = list(betti)

    write_parquet(data, path, metadata=meta)


def read_persistence_table(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Read persistence table with metadata (betti, essential)."""
    result = read_parquet(path)
    meta = _read_metadata(path)
    if "betti" in meta:
        result["betti"] = tuple(meta["betti"])
    if "essential" in meta:
        result["essential"] = np.array(meta["essential"])
    return result


# Filtration table


def write_filtration_table(
    rex,
    filt_v: NDArray,
    filt_e: NDArray,
    filt_f: NDArray,
    path: Union[str, os.PathLike],
    *,
    kind: str = "",
) -> None:
    r"""Write filtration values on the chain complex to Parquet.

    One row per cell.  A valid filtration satisfies
    f(tau) leq f(sigma) for tau in partial(sigma).

    Columns: `cell_idx`, `cell_dim`, `filtration_value`.
    """
    filt_v = np.asarray(filt_v)
    filt_e = np.asarray(filt_e)
    filt_f = np.asarray(filt_f)

    cell_idx = np.concatenate([
        np.arange(len(filt_v), dtype=np.int32),
        np.arange(len(filt_e), dtype=np.int32),
        np.arange(len(filt_f), dtype=np.int32),
    ])
    cell_dim = np.concatenate([
        np.zeros(len(filt_v), dtype=np.int32),
        np.ones(len(filt_e), dtype=np.int32),
        np.full(len(filt_f), 2, dtype=np.int32),
    ])
    filt_val = np.concatenate([filt_v, filt_e, filt_f])

    data = {
        "cell_idx": cell_idx,
        "cell_dim": cell_dim,
        "filtration_value": filt_val,
    }

    meta = {
        "rex_table_type": "filtration_table",
        "nV": int(len(filt_v)),
        "nE": int(len(filt_e)),
        "nF": int(len(filt_f)),
        "kind": kind,
    }
    write_parquet(data, path, metadata=meta)


def read_filtration_table(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Read filtration table -> `filt_v`, `filt_e`, `filt_f`."""
    raw = read_parquet(path)
    meta = _read_metadata(path)

    cell_dim = raw["cell_dim"]
    filt_val = raw["filtration_value"]

    return {
        "filt_v": filt_val[cell_dim == 0],
        "filt_e": filt_val[cell_dim == 1],
        "filt_f": filt_val[cell_dim == 2],
        "kind": meta.get("kind", ""),
    }


# Metrics table


def write_metrics_table(
    metrics: Dict[str, NDArray],
    path: Union[str, os.PathLike],
    *,
    index_name: str = "cell_idx",
) -> None:
    """Write per-cell metrics to Parquet.  All arrays must have equal length."""
    if not metrics:
        raise ValueError("metrics dict cannot be empty")
    lengths = {len(np.asarray(v)) for v in metrics.values()}
    if len(lengths) > 1:
        raise ValueError(f"All arrays must have equal length, got {lengths}")

    n = lengths.pop()
    data = {index_name: np.arange(n, dtype=np.int32)}
    data.update({k: np.asarray(v) for k, v in metrics.items()})

    meta = {
        "rex_table_type": "metrics_table",
        "n_cells": n,
        "metric_names": list(metrics.keys()),
    }
    write_parquet(data, path, metadata=meta)


def read_metrics_table(
    path: Union[str, os.PathLike],
    *,
    exclude_index: bool = True,
) -> Dict[str, np.ndarray]:
    """Read metrics table.  Omits index column by default."""
    result = read_parquet(path)
    if exclude_index:
        result.pop("cell_idx", None)
    return result


# Streaming


def read_parquet_batches(
    path: Union[str, os.PathLike],
    *,
    batch_rows: int = 100_000,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[Dict[str, np.ndarray]]:
    """Stream a Parquet file as batches of arrays.

    Yields dict of name -> ndarray per batch.
    """
    pa, pq = _pq()

    pf = pq.ParquetFile(os.fspath(path))
    pending: List[Dict[str, np.ndarray]] = []
    pending_rows = 0

    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        chunk = {col: batch.column(col).to_numpy() for col in batch.column_names}
        pending.append(chunk)
        pending_rows += batch.num_rows

        if pending_rows >= batch_rows:
            yield _merge_dicts(pending)
            pending = []
            pending_rows = 0

    if pending:
        yield _merge_dicts(pending)


def _merge_dicts(batches: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate a list of column dicts."""
    if len(batches) == 1:
        return batches[0]
    return {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}


# RCF Character and Void tables (new in v2)


def write_character_table(
    rex,
    path,
):
    """Write per-edge structural character to Parquet.

    One row per edge. Columns: edge_idx, chi_0, chi_1, ..., chi_{nhats-1}.
    Column names use hat_names if available (e.g. chi_L1_down, chi_L_O, chi_L_SG).
    """
    import pyarrow as pa
    pq = _pq()

    chi = rex.structural_character
    nE = rex.nE
    nhats = chi.shape[1] if chi.ndim == 2 else 0

    arrays = [pa.array(np.arange(nE, dtype=np.int32))]
    names = ["edge_idx"]

    hat_names = getattr(rex, '_rcf_bundle', {}).get('hat_names', [])
    for k in range(nhats):
        col_name = f"chi_{hat_names[k]}" if k < len(hat_names) else f"chi_{k}"
        arrays.append(pa.array(chi[:, k].astype(np.float64)))
        names.append(col_name)

    table = pa.table(arrays, names=names)
    pq.write_table(table, path)


def read_character_table(path):
    """Read per-edge structural character from Parquet.

    Returns dict with edge_idx and chi columns.
    """
    pq = _pq()
    table = pq.read_table(path)
    result = {}
    for name in table.column_names:
        col = table.column(name)
        if name == "edge_idx":
            result[name] = col.to_numpy().astype(np.int32)
        else:
            result[name] = col.to_numpy().astype(np.float64)
    return result


def write_vertex_character_table(
    rex,
    path,
):
    """Write per-vertex character (phi, kappa) to Parquet.

    One row per vertex. Columns: vertex_idx, phi_0..phi_{nhats-1}, kappa.
    """
    import pyarrow as pa
    pq = _pq()

    phi = rex.vertex_character
    kappa = rex.coherence
    nV = rex.nV
    nhats = phi.shape[1] if phi.ndim == 2 else 0

    arrays = [pa.array(np.arange(nV, dtype=np.int32))]
    names = ["vertex_idx"]

    hat_names = getattr(rex, '_rcf_bundle', {}).get('hat_names', [])
    for k in range(nhats):
        col_name = f"phi_{hat_names[k]}" if k < len(hat_names) else f"phi_{k}"
        arrays.append(pa.array(phi[:, k].astype(np.float64)))
        names.append(col_name)

    arrays.append(pa.array(kappa.astype(np.float64)))
    names.append("kappa")

    table = pa.table(arrays, names=names)
    pq.write_table(table, path)


def read_vertex_character_table(path):
    """Read per-vertex character from Parquet."""
    pq = _pq()
    table = pq.read_table(path)
    result = {}
    for name in table.column_names:
        col = table.column(name)
        if name == "vertex_idx":
            result[name] = col.to_numpy().astype(np.int32)
        else:
            result[name] = col.to_numpy().astype(np.float64)
    return result


def write_void_table(
    rex,
    path,
):
    """Write void complex data to Parquet.

    One row per void triangle. Columns: void_idx, eta, fills_beta,
    chi_void_0..chi_void_{nhats-1}.
    """
    import pyarrow as pa
    pq = _pq()

    vc = rex.void_complex
    n_voids = vc.get('n_voids', 0)

    if n_voids == 0:
        # Write empty table with schema
        schema = pa.schema([
            ("void_idx", pa.int32()),
            ("eta", pa.float64()),
            ("fills_beta", pa.int32()),
        ])
        table = pa.table({name: pa.array([], type=typ) for name, typ in zip(
            schema.names, schema.types)})
        pq.write_table(table, path)
        return

    eta = vc['eta']
    fills = vc.get('fills_beta', np.zeros(n_voids, dtype=np.int32))
    chi_void = vc.get('chi_void', np.zeros((n_voids, 1)))
    nhats = chi_void.shape[1]

    arrays = [
        pa.array(np.arange(n_voids, dtype=np.int32)),
        pa.array(eta.astype(np.float64)),
        pa.array(fills.astype(np.int32)),
    ]
    names = ["void_idx", "eta", "fills_beta"]

    hat_names = getattr(rex, '_rcf_bundle', {}).get('hat_names', [])
    for k in range(nhats):
        col_name = f"chi_void_{hat_names[k]}" if k < len(hat_names) else f"chi_void_{k}"
        arrays.append(pa.array(chi_void[:, k].astype(np.float64)))
        names.append(col_name)

    table = pa.table(arrays, names=names)
    pq.write_table(table, path)


def read_void_table(path):
    """Read void complex data from Parquet."""
    pq = _pq()
    table = pq.read_table(path)
    result = {}
    for name in table.column_names:
        col = table.column(name)
        if name in ("void_idx", "fills_beta"):
            result[name] = col.to_numpy().astype(np.int32)
        else:
            result[name] = col.to_numpy().astype(np.float64)
    return result
