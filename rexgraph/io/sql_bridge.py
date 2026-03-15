# rexgraph/io/sql_bridge.py
"""
SQL bridge for RexGraph analysis results.

Stores the same table types as :mod:`parquet_bridge` into SQL databases
(SQLite, PostgreSQL, or any SQLAlchemy-compatible backend).  Each table
maps to a specific part of the algebraic/topological framework:

Boundary table - the general boundary operator d_1
(Definition 3.1).  One row per (edge, boundary_vertex) pair.

Edge table - per-edge data: source/target, boundary size, edge type
(Definition 3.2), weight, optional Hodge components (Theorem 3.8/4.5).

Vertex table - per-vertex data: degree from L_0, spectral
layout (Definition 6.7), Fiedler vector entries.

Face table - the B_2 operator (Definition 4.1).  One row
per nonzero: (face_idx, edge_idx, orientation).

Persistence table - persistence pairs from column reduction over
Z/2.

Filtration table - filtration values f: C_k -> R.

Temporal table - per-timestep Betti numbers and cell counts from a
TemporalRex.

All `sqlalchemy` and `pandas` imports are lazy.

Usage::

    from rexgraph.io.sql_bridge import write_edge_sql, write_boundary_sql

    engine = get_engine("sqlite:///my_graph.db")
    write_boundary_sql(rex, engine, "boundary")
    write_edge_sql(rex, engine, "edges")
    write_persistence_sql(persistence_result, engine, "persistence")

    # Read back
    data = read_edge_sql(engine, "edges")
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "get_engine",
    "write_boundary_sql",
    "read_boundary_sql",
    "write_edge_sql",
    "read_edge_sql",
    "write_vertex_sql",
    "read_vertex_sql",
    "write_face_sql",
    "read_face_sql",
    "write_persistence_sql",
    "read_persistence_sql",
    "write_filtration_sql",
    "read_filtration_sql",
    "write_temporal_sql",
    "read_temporal_sql",
    "write_metrics_sql",
    "read_metrics_sql",
    "read_sql_batches",
]

# Edge type names matching types.py EdgeType enum (Definition 3.2)
_EDGE_TYPE_NAMES = {0: "standard", 1: "self_loop", 2: "branching", 3: "witness"}


# Lazy imports


def _sa():
    """Lazily import SQLAlchemy."""
    try:
        from sqlalchemy import create_engine, text, MetaData, Table, Column
        from sqlalchemy import types as satypes
        from sqlalchemy.engine import Engine
        from sqlalchemy.pool import StaticPool
        return create_engine, text, Engine, StaticPool, satypes
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy is required for SQL features: pip install sqlalchemy"
        ) from exc


def _pd():
    """Lazily import pandas."""
    try:
        import pandas as pd
        return pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for SQL bridge: pip install pandas"
        ) from exc


# Engine management


_ENGINE_CACHE: Dict[str, Any] = {}


def get_engine(conn_str: str):
    """Get a SQLAlchemy engine.

    For in-memory SQLite, returns a shared `StaticPool`-backed engine
    so multiple calls see the same database.

    Parameters
    ----------
    conn_str : str
        SQLAlchemy connection string, e.g. `"sqlite:///graph.db"`
        or `"sqlite:///:memory:"`.
    """
    create_engine, _, _, StaticPool, _ = _sa()

    low = conn_str.lower()
    if low.startswith("sqlite") and (":memory:" in low or "file::memory:" in low):
        mapped = "sqlite+pysqlite:///file::memory:?cache=shared"
        if mapped not in _ENGINE_CACHE:
            _ENGINE_CACHE[mapped] = create_engine(
                mapped,
                connect_args={"check_same_thread": False, "uri": True},
                poolclass=StaticPool,
            )
        return _ENGINE_CACHE[mapped]

    return create_engine(conn_str)


def _ensure_engine(conn):
    """Convert string to engine if needed."""
    _, _, Engine, _, _ = _sa()
    if isinstance(conn, str):
        return get_engine(conn)
    return conn


def _write_df(data: Dict[str, NDArray], engine, table: str,
              dtype: Dict[str, Any], *, if_exists: str = "replace") -> None:
    """Build a DataFrame from arrays and write to SQL."""
    pd = _pd()
    df = pd.DataFrame(data)
    df.to_sql(table, engine, if_exists=if_exists, index=False, dtype=dtype)


def _read_table(engine, table: str, *, where: str = "",
                order_by: str = "") -> Dict[str, np.ndarray]:
    """Read a SQL table into a dict of arrays."""
    pd = _pd()
    _, text_fn, _, _, _ = _sa()

    query = f"SELECT * FROM {table}"
    if where:
        query += f" WHERE {where}"
    if order_by:
        query += f" ORDER BY {order_by}"

    df = pd.read_sql(text_fn(query), engine)
    return {col: df[col].to_numpy() for col in df.columns}


def _write_meta(engine, table: str, meta: dict) -> None:
    """Write metadata to a companion `<table>_meta` table."""
    pd = _pd()
    meta_table = f"{table}_meta"
    # Store as single-row JSON
    pd.DataFrame([{"meta_json": json.dumps(meta, default=_json_default)}]).to_sql(
        meta_table, engine, if_exists="replace", index=False
    )


def _read_meta(engine, table: str) -> dict:
    """Read metadata from the companion `<table>_meta` table."""
    pd = _pd()
    _, text_fn, _, _, _ = _sa()
    meta_table = f"{table}_meta"
    try:
        df = pd.read_sql(text_fn(f"SELECT meta_json FROM {meta_table}"), engine)
        if len(df) > 0:
            return json.loads(df.iloc[0]["meta_json"])
    except Exception:
        pass
    return {}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


# Boundary table (Definition 3.1)


def write_boundary_sql(
    rex,
    conn: Any,
    table: str = "boundary",
    *,
    if_exists: str = "replace",
) -> None:
    """Write the general boundary operator d_1 to SQL.

    One row per (edge, boundary_vertex) pair.  Handles all edge types
    from Definition 3.2: standard, self-loop, branching, witness.

    Columns: `edge_idx`, `vertex_idx`, `position`.
    """
    _, _, _, _, sat = _sa()
    engine = _ensure_engine(conn)

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
    dtype = {
        "edge_idx": sat.Integer(),
        "vertex_idx": sat.Integer(),
        "position": sat.SmallInteger(),
    }
    _write_df(data, engine, table, dtype, if_exists=if_exists)
    _write_meta(engine, table, {
        "rex_table_type": "boundary_table",
        "nV": int(rex.nV), "nE": int(rex.nE),
        "n_entries": n_entries,
        "directed": bool(rex._directed),
    })


def read_boundary_sql(
    conn: Any,
    table: str = "boundary",
) -> Dict[str, Any]:
    """Read boundary table and reconstruct `boundary_ptr`/`boundary_idx`."""
    engine = _ensure_engine(conn)
    raw = _read_table(engine, table, order_by="edge_idx, position")
    meta = _read_meta(engine, table)

    nE = meta.get("nE", int(raw["edge_idx"].max()) + 1)

    boundary_ptr = np.zeros(nE + 1, dtype=np.int32)
    for e in raw["edge_idx"]:
        boundary_ptr[e + 1] += 1
    np.cumsum(boundary_ptr, out=boundary_ptr)

    return {
        "boundary_ptr": boundary_ptr,
        "boundary_idx": raw["vertex_idx"].astype(np.int32),
        "nV": meta.get("nV"),
        "nE": nE,
        "directed": meta.get("directed", False),
        **raw,
    }


# Edge table (Definition 3.2)


def write_edge_sql(
    rex,
    conn: Any,
    table: str = "edges",
    *,
    include: Optional[List[str]] = None,
    if_exists: str = "replace",
) -> None:
    r"""Write per-edge data to SQL.

    Columns: `edge_idx`, `source`, `target`, `boundary_size`,
    `edge_type`, `edge_type_name`, `weight` (if weighted).

    Uses the general boundary d_1 to derive source/target,
    with `-1` for witness edges (Definition 3.2).
    """
    _, _, _, _, sat = _sa()
    engine = _ensure_engine(conn)

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
    etype_names = [_EDGE_TYPE_NAMES.get(int(t), "unknown") for t in etypes]

    data: Dict[str, Any] = {
        "edge_idx": np.arange(nE, dtype=np.int32),
        "source": src,
        "target": tgt,
        "boundary_size": boundary_size,
        "edge_type": etypes.astype(np.int32),
        "edge_type_name": etype_names,
    }

    if rex._w_E is not None:
        data["weight"] = rex._w_E

    # Optional Hodge components (Theorem 3.8/4.5)
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

    dtype = {
        "edge_idx": sat.Integer(),
        "source": sat.Integer(),
        "target": sat.Integer(),
        "boundary_size": sat.SmallInteger(),
        "edge_type": sat.SmallInteger(),
        "edge_type_name": sat.String(20),
    }
    if rex._w_E is not None:
        dtype["weight"] = sat.Float(precision=53)
    for k in data:
        if k not in dtype and isinstance(data[k], np.ndarray):
            if np.issubdtype(data[k].dtype, np.floating):
                dtype[k] = sat.Float(precision=53)
            elif np.issubdtype(data[k].dtype, np.integer):
                dtype[k] = sat.Integer()

    _write_df(data, engine, table, dtype, if_exists=if_exists)
    _write_meta(engine, table, {
        "rex_table_type": "edge_table",
        "nV": int(rex.nV), "nE": nE, "nF": int(rex.nF),
        "directed": bool(rex._directed),
    })


def read_edge_sql(
    conn: Any,
    table: str = "edges",
) -> Dict[str, np.ndarray]:
    """Read edge table from SQL."""
    engine = _ensure_engine(conn)
    return _read_table(engine, table, order_by="edge_idx")


# Vertex table


def write_vertex_sql(
    rex,
    conn: Any,
    table: str = "vertices",
    *,
    include: Optional[List[str]] = None,
    if_exists: str = "replace",
) -> None:
    r"""Write per-vertex data to SQL.

    Default columns: `vertex_idx`, `degree` (from L_0),
    `x`, `y` (spectral layout, Definition 6.7).
    """
    _, _, _, _, sat = _sa()
    engine = _ensure_engine(conn)
    nV = rex.nV

    data: Dict[str, Any] = {
        "vertex_idx": np.arange(nV, dtype=np.int32),
    }
    dtype: Dict[str, Any] = {"vertex_idx": sat.Integer()}

    try:
        data["degree"] = np.diag(rex.L0).astype(np.int32)
        dtype["degree"] = sat.Integer()
    except Exception:
        pass

    try:
        layout = rex.layout
        data["x"] = layout[:, 0]
        data["y"] = layout[:, 1]
        dtype["x"] = sat.Float(precision=53)
        dtype["y"] = sat.Float(precision=53)
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
                    dtype.update({f: sat.Float(precision=53) for f in ("x3", "y3", "z3")})
                except Exception:
                    pass
            else:
                try:
                    val = getattr(rex, prop)
                    if isinstance(val, np.ndarray) and val.shape[0] == nV:
                        data[prop] = val
                        dtype[prop] = sat.Float(precision=53)
                except Exception:
                    pass

    _write_df(data, engine, table, dtype, if_exists=if_exists)
    _write_meta(engine, table, {
        "rex_table_type": "vertex_table",
        "nV": nV, "nE": int(rex.nE),
    })


def read_vertex_sql(
    conn: Any,
    table: str = "vertices",
) -> Dict[str, np.ndarray]:
    """Read vertex table from SQL."""
    engine = _ensure_engine(conn)
    return _read_table(engine, table, order_by="vertex_idx")


# Face table (Definition 4.1 - B2)


def write_face_sql(
    rex,
    conn: Any,
    table: str = "faces",
    *,
    if_exists: str = "replace",
) -> None:
    """Write the B_2 boundary operator to SQL.

    One row per nonzero in the CSC representation.  The chain complex
    condition B_1 B_2 = 0 guarantees each face boundary is a
    cycle in the edge basis.

    Columns: `face_idx`, `edge_idx`, `orientation`.
    """
    _, _, _, _, sat = _sa()
    engine = _ensure_engine(conn)

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

    dtype = {
        "face_idx": sat.Integer(),
        "edge_idx": sat.Integer(),
        "orientation": sat.Float(precision=53),
    }
    _write_df(data, engine, table, dtype, if_exists=if_exists)
    _write_meta(engine, table, {
        "rex_table_type": "face_table",
        "nE": int(rex.nE), "nF": int(rex.nF),
        "nnz_B2": nnz,
        "chain_valid": bool(rex.chain_valid) if rex.nF > 0 else True,
    })


def read_face_sql(
    conn: Any,
    table: str = "faces",
) -> Dict[str, Any]:
    """Read face table and reconstruct B2 CSC arrays."""
    engine = _ensure_engine(conn)
    raw = _read_table(engine, table, order_by="face_idx, edge_idx")
    meta = _read_meta(engine, table)

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
    B2_col_ptr = np.zeros(nF + 1, dtype=np.int32)
    for f in face_idx:
        B2_col_ptr[f + 1] += 1
    np.cumsum(B2_col_ptr, out=B2_col_ptr)

    return {
        "B2_col_ptr": B2_col_ptr,
        "B2_row_idx": raw["edge_idx"].astype(np.int32),
        "B2_vals": raw["orientation"].astype(np.float64),
        "nF": nF, "nE": nE, **raw,
    }


# Persistence table (column reduction over Z/2)


def write_persistence_sql(
    result: Any,
    conn: Any,
    table: str = "persistence",
    *,
    if_exists: str = "replace",
) -> None:
    r"""Write persistence pairs to SQL.

    Accepts the output of `rex.persistence(filt_v, filt_e, filt_f)`
    - a dict with `pairs` shape `(k, 5)`:
    `[birth, death, dim, birth_cell, death_cell]`.
    """
    _, _, _, _, sat = _sa()
    engine = _ensure_engine(conn)

    if isinstance(result, dict):
        pairs = np.asarray(result["pairs"])
    elif hasattr(result, "pairs"):
        pairs = np.asarray(result.pairs)
    else:
        raise TypeError("Expected persistence result dict or PersistenceDiagram")

    data: Dict[str, Any] = {}
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

    dtype = {
        "birth": sat.Float(precision=53),
        "death": sat.Float(precision=53),
        "dim": sat.SmallInteger(),
        "lifetime": sat.Float(precision=53),
    }
    if "birth_cell" in data:
        dtype["birth_cell"] = sat.Integer()
        dtype["death_cell"] = sat.Integer()

    _write_df(data, engine, table, dtype, if_exists=if_exists)

    meta: Dict[str, Any] = {
        "rex_table_type": "persistence_table",
        "n_pairs": int(len(pairs)),
    }
    essential = result.get("essential") if isinstance(result, dict) else getattr(result, "essential", None)
    if essential is not None:
        meta["essential"] = np.asarray(essential).tolist()
    betti = result.get("betti") if isinstance(result, dict) else getattr(result, "betti", None)
    if betti is not None:
        meta["betti"] = list(betti)

    _write_meta(engine, table, meta)


def read_persistence_sql(
    conn: Any,
    table: str = "persistence",
) -> Dict[str, Any]:
    """Read persistence pairs with metadata (betti, essential)."""
    engine = _ensure_engine(conn)
    result = _read_table(engine, table, order_by="dim, birth")
    meta = _read_meta(engine, table)

    if "betti" in meta:
        result["betti"] = tuple(meta["betti"])
    if "essential" in meta:
        result["essential"] = np.array(meta["essential"])
    return result


# Filtration table


def write_filtration_sql(
    rex,
    filt_v: NDArray,
    filt_e: NDArray,
    filt_f: NDArray,
    conn: Any,
    table: str = "filtration",
    *,
    kind: str = "",
    if_exists: str = "replace",
) -> None:
    r"""Write filtration values on the chain complex to SQL.

    One row per cell.  A valid filtration satisfies
    f(tau) leq f(sigma) for tau in partial(sigma).

    Columns: `cell_idx`, `cell_dim`, `filtration_value`.
    """
    _, _, _, _, sat = _sa()
    engine = _ensure_engine(conn)

    filt_v = np.asarray(filt_v)
    filt_e = np.asarray(filt_e)
    filt_f = np.asarray(filt_f)

    data = {
        "cell_idx": np.concatenate([
            np.arange(len(filt_v), dtype=np.int32),
            np.arange(len(filt_e), dtype=np.int32),
            np.arange(len(filt_f), dtype=np.int32),
        ]),
        "cell_dim": np.concatenate([
            np.zeros(len(filt_v), dtype=np.int32),
            np.ones(len(filt_e), dtype=np.int32),
            np.full(len(filt_f), 2, dtype=np.int32),
        ]),
        "filtration_value": np.concatenate([filt_v, filt_e, filt_f]),
    }

    dtype = {
        "cell_idx": sat.Integer(),
        "cell_dim": sat.SmallInteger(),
        "filtration_value": sat.Float(precision=53),
    }
    _write_df(data, engine, table, dtype, if_exists=if_exists)
    _write_meta(engine, table, {
        "rex_table_type": "filtration_table",
        "nV": int(len(filt_v)), "nE": int(len(filt_e)),
        "nF": int(len(filt_f)), "kind": kind,
    })


def read_filtration_sql(
    conn: Any,
    table: str = "filtration",
) -> Dict[str, Any]:
    """Read filtration table -> `filt_v`, `filt_e`, `filt_f`."""
    engine = _ensure_engine(conn)
    raw = _read_table(engine, table, order_by="cell_dim, cell_idx")
    meta = _read_meta(engine, table)

    cell_dim = raw["cell_dim"]
    filt_val = raw["filtration_value"]

    return {
        "filt_v": filt_val[cell_dim == 0],
        "filt_e": filt_val[cell_dim == 1],
        "filt_f": filt_val[cell_dim == 2],
        "kind": meta.get("kind", ""),
    }


# Temporal table (TemporalRex per-timestep summaries)


def write_temporal_sql(
    trex,
    conn: Any,
    table: str = "temporal",
    *,
    if_exists: str = "replace",
) -> None:
    r"""Write per-timestep summary from a TemporalRex to SQL.

    Columns: `timestep`, `nE`, `nF`, `beta_0`, `beta_1`,
    `beta_2`, `euler_characteristic`.
    """
    _, _, _, _, sat = _sa()
    engine = _ensure_engine(conn)

    T = trex.T
    timesteps = np.arange(T, dtype=np.int32)

    data: Dict[str, Any] = {"timestep": timesteps}
    nE_arr = np.empty(T, dtype=np.int32)
    nF_arr = np.empty(T, dtype=np.int32)
    b0_arr = np.empty(T, dtype=np.int32)
    b1_arr = np.empty(T, dtype=np.int32)
    b2_arr = np.empty(T, dtype=np.int32)
    euler_arr = np.empty(T, dtype=np.int32)

    for t in range(T):
        snap = trex.at(t)
        nE_arr[t] = snap.nE
        nF_arr[t] = snap.nF
        try:
            b = snap.betti
            b0_arr[t], b1_arr[t], b2_arr[t] = b
        except Exception:
            b0_arr[t] = b1_arr[t] = b2_arr[t] = -1
        try:
            euler_arr[t] = snap.euler_characteristic
        except Exception:
            euler_arr[t] = snap.nV - snap.nE + snap.nF

    data["nE"] = nE_arr
    data["nF"] = nF_arr
    data["beta_0"] = b0_arr
    data["beta_1"] = b1_arr
    data["beta_2"] = b2_arr
    data["euler_characteristic"] = euler_arr

    dtype = {k: sat.Integer() for k in data}

    _write_df(data, engine, table, dtype, if_exists=if_exists)
    _write_meta(engine, table, {
        "rex_table_type": "temporal_table",
        "T": T,
        "directed": bool(trex._directed),
    })


def read_temporal_sql(
    conn: Any,
    table: str = "temporal",
) -> Dict[str, Any]:
    """Read temporal summary table.

    Returns dict with `timestep`, `betti` (T×3 array), `nE`, `nF`,
    `euler_characteristic`.
    """
    engine = _ensure_engine(conn)
    raw = _read_table(engine, table, order_by="timestep")

    result: Dict[str, Any] = {"timestep": raw["timestep"]}

    betti_cols = [c for c in raw if c.startswith("beta_")]
    if betti_cols:
        betti_cols.sort(key=lambda c: int(c.split("_")[1]))
        result["betti"] = np.column_stack([raw[c] for c in betti_cols])

    for col in ("nE", "nF", "euler_characteristic"):
        if col in raw:
            result[col] = raw[col]

    return result


# Metrics table (generic per-cell numerics)


def write_metrics_sql(
    metrics: Dict[str, NDArray],
    conn: Any,
    table: str = "metrics",
    *,
    cell_dim: int = 0,
    if_exists: str = "replace",
) -> None:
    """Write per-cell metrics to SQL.

    Parameters
    ----------
    metrics : dict of arrays
        All equal length.
    conn : str or Engine
    table : str
    cell_dim : int
        Cell dimension (0=vertex, 1=edge, 2=face).
    """
    _, _, _, _, sat = _sa()
    engine = _ensure_engine(conn)

    if not metrics:
        raise ValueError("metrics dict cannot be empty")
    lengths = {len(np.asarray(v)) for v in metrics.values()}
    if len(lengths) > 1:
        raise ValueError(f"All arrays must have equal length, got {lengths}")

    n = lengths.pop()
    data: Dict[str, Any] = {
        "cell_idx": np.arange(n, dtype=np.int32),
        "cell_dim": np.full(n, cell_dim, dtype=np.int32),
    }
    data.update({k: np.asarray(v) for k, v in metrics.items()})

    dtype: Dict[str, Any] = {
        "cell_idx": sat.Integer(),
        "cell_dim": sat.SmallInteger(),
    }
    for name, arr in metrics.items():
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.integer):
            dtype[name] = sat.Integer()
        else:
            dtype[name] = sat.Float(precision=53)

    _write_df(data, engine, table, dtype, if_exists=if_exists)
    _write_meta(engine, table, {
        "rex_table_type": "metrics_table",
        "n_cells": n, "cell_dim": cell_dim,
        "metric_names": list(metrics.keys()),
    })


def read_metrics_sql(
    conn: Any,
    table: str = "metrics",
    *,
    cell_dim: Optional[int] = None,
    exclude_index: bool = True,
) -> Dict[str, np.ndarray]:
    """Read metrics table from SQL."""
    engine = _ensure_engine(conn)

    where = f"cell_dim = {cell_dim}" if cell_dim is not None else ""
    result = _read_table(engine, table, where=where, order_by="cell_idx")

    if exclude_index:
        result.pop("cell_idx", None)
        result.pop("cell_dim", None)
    return result


# Streaming


def read_sql_batches(
    conn: Any,
    table_or_query: str,
    *,
    chunksize: int = 100_000,
) -> Iterator[Dict[str, np.ndarray]]:
    """Stream SQL results as batches of arrays.

    Parameters
    ----------
    conn : str or Engine
    table_or_query : str
        Table name or SQL query.
    chunksize : int
        Rows per batch.

    Yields
    ------
    dict of name -> ndarray per batch
    """
    pd = _pd()
    _, text_fn, _, _, _ = _sa()
    engine = _ensure_engine(conn)

    if " " not in table_or_query:
        query = f"SELECT * FROM {table_or_query}"
    else:
        query = table_or_query

    with engine.connect() as connection:
        for chunk in pd.read_sql(text_fn(query), connection, chunksize=chunksize):
            yield {col: chunk[col].to_numpy() for col in chunk.columns}


# RCF Character and Void SQL tables (new in v2)


def write_character_sql(
    rex,
    conn,
    *,
    table: str = "character",
    if_exists: str = "replace",
):
    """Write per-edge structural character to SQL.

    Columns: edge_idx, chi_L1_down, chi_L_O, chi_L_SG.
    """
    engine = _ensure_engine(conn)
    chi = rex.structural_character
    nE = rex.nE
    nhats = chi.shape[1] if chi.ndim == 2 else 0
    hat_names = getattr(rex, '_rcf_bundle', {}).get('hat_names', [])

    data = {"edge_idx": np.arange(nE, dtype=np.int32)}
    for k in range(nhats):
        col = f"chi_{hat_names[k]}" if k < len(hat_names) else f"chi_{k}"
        data[col] = chi[:, k].astype(np.float64)

    _write_df(data, engine, table, if_exists=if_exists)


def read_character_sql(conn, *, table: str = "character"):
    """Read per-edge character from SQL."""
    engine = _ensure_engine(conn)
    return _read_table(engine, table)


def write_vertex_character_sql(
    rex,
    conn,
    *,
    table: str = "vertex_character",
    if_exists: str = "replace",
):
    """Write per-vertex character (phi, kappa) to SQL."""
    engine = _ensure_engine(conn)
    phi = rex.vertex_character
    kappa = rex.coherence
    nV = rex.nV
    nhats = phi.shape[1] if phi.ndim == 2 else 0
    hat_names = getattr(rex, '_rcf_bundle', {}).get('hat_names', [])

    data = {"vertex_idx": np.arange(nV, dtype=np.int32)}
    for k in range(nhats):
        col = f"phi_{hat_names[k]}" if k < len(hat_names) else f"phi_{k}"
        data[col] = phi[:, k].astype(np.float64)
    data["kappa"] = kappa.astype(np.float64)

    _write_df(data, engine, table, if_exists=if_exists)


def read_vertex_character_sql(conn, *, table: str = "vertex_character"):
    """Read per-vertex character from SQL."""
    engine = _ensure_engine(conn)
    return _read_table(engine, table)


def write_void_sql(
    rex,
    conn,
    *,
    table: str = "void",
    if_exists: str = "replace",
):
    """Write void complex data to SQL."""
    engine = _ensure_engine(conn)
    vc = rex.void_complex
    n_voids = vc.get('n_voids', 0)

    if n_voids == 0:
        data = {"void_idx": np.array([], dtype=np.int32),
                "eta": np.array([], dtype=np.float64),
                "fills_beta": np.array([], dtype=np.int32)}
    else:
        chi_void = vc.get('chi_void', np.zeros((n_voids, 1)))
        nhats = chi_void.shape[1]
        hat_names = getattr(rex, '_rcf_bundle', {}).get('hat_names', [])

        data = {
            "void_idx": np.arange(n_voids, dtype=np.int32),
            "eta": vc['eta'].astype(np.float64),
            "fills_beta": vc.get('fills_beta', np.zeros(n_voids, dtype=np.int32)),
        }
        for k in range(nhats):
            col = f"chi_void_{hat_names[k]}" if k < len(hat_names) else f"chi_void_{k}"
            data[col] = chi_void[:, k].astype(np.float64)

    _write_df(data, engine, table, if_exists=if_exists)


def read_void_sql(conn, *, table: str = "void"):
    """Read void complex data from SQL."""
    engine = _ensure_engine(conn)
    return _read_table(engine, table)
