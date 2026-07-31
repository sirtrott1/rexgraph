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


# --- format registry -----------------------------------------------------------
#
# Adding a format means registering one, not editing this module. Mirrors
# agent.rcdb.register_backend, which is the same pattern one layer up.

from ..registry import Registry


class _Format:
    __slots__ = ("name", "save", "load", "extensions")

    def __init__(self, name, save, load, extensions):
        self.name = name
        self.save = save
        self.load = load
        self.extensions = tuple(extensions)


_FORMATS = Registry("format")


def register_format(name, *, save=None, load=None, extensions=()):
    """Register a storage format under `name`.

    `save(path, obj, **kwargs)` and `load(path, **kwargs)` are the handlers; either may
    be None for a read-only or write-only format, in which case the corresponding entry
    point raises. `extensions` are lowercase suffixes (".rex") mapped to this format by
    `_detect_format`.
    """
    fmt = _Format(name, save, load, extensions)
    _FORMATS.register(name, fmt, extensions=tuple(extensions))
    return fmt


def unregister_format(name):
    """Remove a registered format. Returns it, or None if it was not registered."""
    return _FORMATS.unregister(name)


def available_formats():
    """Names of every registered format."""
    return _FORMATS.available()


def format_extensions():
    """Mapping of extension -> format name, built from the registry."""
    return {e: name for name, f in _FORMATS.items() for e in f.extensions}


def _require(fmt_name, verb):
    fmt = _FORMATS.get(fmt_name)
    if fmt is None:
        raise ValueError(
            f"Unknown format {fmt_name!r}. Registered: {', '.join(available_formats())}.")
    handler = getattr(fmt, verb)
    if handler is None:
        raise ValueError(f"Format {fmt_name!r} does not support {verb}.")
    return handler


def save(path, obj, *, format=None, **kwargs):
    """Save a RexGraph or TemporalRex to disk."""
    return _require(_detect_format(path, format), "save")(path, obj, **kwargs)


def load(path, *, format=None, **kwargs):
    """Load a RexGraph or TemporalRex from disk."""
    return _require(_detect_format(path, format), "load")(path, **kwargs)


def _detect_format(path, override=None):
    """Resolve a path to a format name.

    An UNRECOGNIZED extension is an error, not a default. The fallback used to be
    "zarr", so `save("graph.saftensors", rex)` silently wrote a Zarr store under a
    misspelled name and reported success. An extensionless path keeps the directory
    heuristics, because that is how a Zarr store is normally named.
    """
    import os
    if override is not None:
        return override.lower()
    exts = format_extensions()
    _, ext = os.path.splitext(path)
    if ext.lower() in exts:
        return exts[ext.lower()]
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "MANIFEST.json")):
            return "rex"
        return "zarr"
    if os.path.isfile(path):
        return "hdf5"
    if ext:
        raise ValueError(
            f"Unknown format for extension {ext!r} in {path!r}. Known extensions: "
            f"{', '.join(sorted(exts))}. Pass format= to override.")
    return "zarr"


# --- builtin formats -----------------------------------------------------------

def _save_json(path, obj, **kwargs):
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(obj.to_json(), f)


def _load_safetensors(path, **kwargs):
    """A .safetensors file holds a rex, a TemporalRex or a vector corpus; route on the
    stored object_type rather than assuming."""
    from .safetensors_bridge import (
        _load_meta, safetensors_to_rex, safetensors_to_fingerprints,
    )
    if _load_meta(str(path)).get("object_type") == "FingerprintCorpus":
        return safetensors_to_fingerprints(path)
    return safetensors_to_rex(path)


def _save_safetensors(path, obj, **kwargs):
    from .safetensors_bridge import save_safetensors as _s
    return _s(obj, path, **kwargs)


def _needs(pkg, extra):
    def _raise(*a, **k):
        raise ImportError(f"{pkg} is required: pip install {extra}")
    return _raise


register_format("rex", save=save_rex, load=load_rex, extensions=[".rex"])
register_format("json", save=_save_json, load=load_json, extensions=[".json"])
register_format(
    "zarr",
    save=save_zarr if HAS_ZARR else _needs("zarr", "zarr"),
    load=load_zarr if HAS_ZARR else _needs("zarr", "zarr"),
    extensions=[".zarr"],
)
register_format(
    "hdf5",
    save=save_hdf5 if HAS_HDF5 else _needs("h5py", "h5py"),
    load=load_hdf5 if HAS_HDF5 else _needs("h5py", "h5py"),
    extensions=[".h5", ".hdf5"],
)
register_format(
    "safetensors",
    save=_save_safetensors if HAS_SAFETENSORS else _needs("safetensors", "safetensors"),
    load=_load_safetensors if HAS_SAFETENSORS else _needs("safetensors", "safetensors"),
    extensions=[".safetensors"],
)

__all__ += ["register_format", "unregister_format", "available_formats",
            "format_extensions"]
