# rexgraph/io/__init__.py
"""
Serialization and storage for relational complexes.

Backends: Zarr (.zarr), HDF5 (.h5), bundle (.rex), Arrow IPC,
Parquet, SQL (via SQLAlchemy).

    from rexgraph.io import save, load
    save("graph.zarr", rex, cache="all")
    rex = load("graph.zarr")

Labeled vector corpora (embeddings, structural fingerprints) share one container:

    from rexgraph.io import save_vectors, load_vectors
    save_vectors(matrix, labels, "emb.safetensors", feature_names=..., block_offsets=...)
    matrix, labels, names, meta = load_vectors("emb.safetensors")
"""

from ._compat import ZARR_V3, HAS_ZARR, HAS_HDF5

__all__ = ["ZARR_V3", "HAS_ZARR", "HAS_HDF5", "save", "load"]

if HAS_ZARR:
    from .zarr_format import RexZarrFormat, save_zarr, load_zarr
    __all__ += ["RexZarrFormat", "save_zarr", "load_zarr"]

if HAS_HDF5:
    from .hdf5_format import RexHDF5Format, save_hdf5, load_hdf5
    __all__ += ["RexHDF5Format", "save_hdf5", "load_hdf5"]

from .bundle import RexBundle, save_rex, load_rex
__all__ += ["RexBundle", "save_rex", "load_rex"]

try:
    from .arrow_bridge import (
        rex_to_arrow, arrow_to_rex, arrays_to_arrow, arrow_to_arrays,
        write_arrow_ipc, read_arrow_ipc, read_arrow_batches,
    )
    __all__ += [
        "rex_to_arrow", "arrow_to_rex", "arrays_to_arrow", "arrow_to_arrays",
        "write_arrow_ipc", "read_arrow_ipc", "read_arrow_batches",
    ]
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

try:
    from .parquet_bridge import (
        write_parquet, read_parquet,
        write_boundary_table, read_boundary_table,
        write_edge_table, read_edge_table,
        write_vertex_table, read_vertex_table,
        write_face_table, read_face_table,
        write_persistence_table, read_persistence_table,
        write_filtration_table, read_filtration_table,
        write_metrics_table, read_metrics_table,
        read_parquet_batches,
        write_character_table, read_character_table,
        write_vertex_character_table, read_vertex_character_table,
        write_void_table, read_void_table,
    )
    __all__ += [
        "write_parquet", "read_parquet",
        "write_boundary_table", "read_boundary_table",
        "write_edge_table", "read_edge_table",
        "write_vertex_table", "read_vertex_table",
        "write_face_table", "read_face_table",
        "write_persistence_table", "read_persistence_table",
        "write_filtration_table", "read_filtration_table",
        "write_metrics_table", "read_metrics_table",
        "read_parquet_batches",
        "write_character_table", "read_character_table",
        "write_vertex_character_table", "read_vertex_character_table",
        "write_void_table", "read_void_table",
    ]
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    from .sql_bridge import (
        get_engine,
        write_boundary_sql, read_boundary_sql,
        write_edge_sql, read_edge_sql,
        write_vertex_sql, read_vertex_sql,
        write_face_sql, read_face_sql,
        write_persistence_sql, read_persistence_sql,
        write_filtration_sql, read_filtration_sql,
        write_temporal_sql, read_temporal_sql,
        write_metrics_sql, read_metrics_sql,
        read_sql_batches,
        write_character_sql, read_character_sql,
        write_vertex_character_sql, read_vertex_character_sql,
        write_void_sql, read_void_sql,
    )
    __all__ += [
        "get_engine",
        "write_boundary_sql", "read_boundary_sql",
        "write_edge_sql", "read_edge_sql",
        "write_vertex_sql", "read_vertex_sql",
        "write_face_sql", "read_face_sql",
        "write_persistence_sql", "read_persistence_sql",
        "write_filtration_sql", "read_filtration_sql",
        "write_temporal_sql", "read_temporal_sql",
        "write_metrics_sql", "read_metrics_sql",
        "read_sql_batches",
        "write_character_sql", "read_character_sql",
        "write_vertex_character_sql", "read_vertex_character_sql",
        "write_void_sql", "read_void_sql",
    ]
    HAS_SQL = True
except ImportError:
    HAS_SQL = False

__all__ += ["HAS_ARROW", "HAS_PARQUET", "HAS_SQL"]

try:
    from .safetensors_bridge import (
        rex_to_safetensors, safetensors_to_rex,
        temporal_rex_to_safetensors, safetensors_to_temporal_rex,
        save_safetensors, load_safetensors,
        fingerprints_to_safetensors, safetensors_to_fingerprints,
    )
    # Discoverable front door for the labeled-vector-corpus container. The stored schema
    # (object_type="FingerprintCorpus") is unchanged; these are the general names for the
    # same primitive - a stacked (n, d) matrix + labels + feature_names + block_offsets +
    # metadata. Any embedding matrix (model token embeddings, sentence embeddings, the
    # agent's structural fingerprints) round-trips through here without a new subsystem.
    save_vectors = fingerprints_to_safetensors
    load_vectors = safetensors_to_fingerprints
    __all__ += [
        "rex_to_safetensors", "safetensors_to_rex",
        "temporal_rex_to_safetensors", "safetensors_to_temporal_rex",
        "save_safetensors", "load_safetensors",
        "fingerprints_to_safetensors", "safetensors_to_fingerprints",
        "save_vectors", "load_vectors",
    ]
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

__all__ += ["HAS_SAFETENSORS"]

from .csv_loader import load_edge_csv, classify_columns, GraphData, ColumnProfile
__all__ += ["load_edge_csv", "classify_columns", "GraphData", "ColumnProfile"]

from .json_loader import (
    load_json, load_rexgraph_json, load_edge_list_json,
    load_cytoscape_json, load_networkx_json, load_adjacency_json,
    load_matrix_csv,
)
__all__ += [
    "load_json", "load_rexgraph_json", "load_edge_list_json",
    "load_cytoscape_json", "load_networkx_json", "load_adjacency_json",
    "load_matrix_csv",
]


def save(path, obj, *, format=None, **kwargs):
    """Save a RexGraph or TemporalRex to disk."""
    fmt = _detect_format(path, format)
    if fmt == "zarr":
        if not HAS_ZARR:
            raise ImportError("zarr is required: pip install zarr")
        save_zarr(path, obj, **kwargs)
    elif fmt == "hdf5":
        if not HAS_HDF5:
            raise ImportError("h5py is required: pip install h5py")
        save_hdf5(path, obj, **kwargs)
    elif fmt == "rex":
        save_rex(path, obj, **kwargs)
    elif fmt == "json":
        # Symmetric with load(format="json"): write RexGraph native JSON.
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(obj.to_json(), f)
    else:
        raise ValueError(f"Unknown format {fmt!r}.")


def load(path, *, format=None, **kwargs):
    """Load a RexGraph or TemporalRex from disk."""
    fmt = _detect_format(path, format)
    if fmt == "zarr":
        if not HAS_ZARR:
            raise ImportError("zarr is required: pip install zarr")
        return load_zarr(path, **kwargs)
    elif fmt == "hdf5":
        if not HAS_HDF5:
            raise ImportError("h5py is required: pip install h5py")
        return load_hdf5(path, **kwargs)
    elif fmt == "rex":
        return load_rex(path, **kwargs)
    elif fmt == "json":
        return load_json(path, **kwargs)
    elif fmt == "safetensors":
        # A .safetensors file can hold either a rex complex or a vector corpus; route by
        # the stored object_type instead of assuming a rex (which used to TypeError on a
        # vector/embedding file).
        from .safetensors_bridge import (
            _load_meta, safetensors_to_rex, safetensors_to_fingerprints,
        )
        if _load_meta(str(path)).get("object_type") == "FingerprintCorpus":
            return safetensors_to_fingerprints(path)
        return safetensors_to_rex(path)
    else:
        raise ValueError(f"Unknown format {fmt!r}.")


def _detect_format(path, override=None):
    import os
    if override is not None:
        return override.lower()
    if path.endswith(".zarr"):
        return "zarr"
    if path.endswith((".h5", ".hdf5")):
        return "hdf5"
    if path.endswith(".rex"):
        return "rex"
    if path.endswith(".json"):
        return "json"
    if path.endswith(".safetensors"):
        return "safetensors"
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "MANIFEST.json")):
            return "rex"
        return "zarr"
    if os.path.isfile(path):
        return "hdf5"
    return "zarr"
