# rexgraph/io/safetensors_bridge.py
"""
Safetensors bridge for RexGraph and TemporalRex.

Exports the same cell-complex reconstruction data and optional cache
groups that `bundle.py` writes to `.rex` directories, but packed into a
single `.safetensors` file. The goals are:

- Cross-ecosystem transport: safetensors is the idiomatic format for
  sharing tensors with PyTorch, JAX, HuggingFace Hub, and downstream ML
  tooling without requiring numpy-level deserialization.
- Fast cold reads: the safetensors format is memory-mappable and has a
  small fixed JSON header, so large caches load in constant time plus
  the cost of the tensor slice actually requested.
- No pickle, no arbitrary code execution: the on-disk layout is a JSON
  header followed by packed byte buffers. Loading a file cannot run
  user code.

This bridge is parallel to `arrow_bridge.py` and `parquet_bridge.py`.
It is not the primary storage for RexGraphs - use `bundle.py` (`.rex`)
or `zarr_format.py` (`.zarr`) for that. Use this bridge when shipping
a rex to an ML environment or when you want the ML-ecosystem loader
path.

Layout
~~~~~~
All arrays live in a single flat dict with `/`-separated keys. The
grouping mirrors `bundle.py` cache groups:

    # Core reconstruction arrays (always present)
    boundary_ptr           int32 (nE+1,)
    boundary_idx           int32 (nnz,)
    B2_col_ptr             int32 (nF+1,)
    B2_row_idx             int32 (nnz_B2,)
    B2_vals                float64 (nnz_B2,)
    w_E                    float64 (nE,)      [if weighted]
    attribution            float64 (nV, ...)  [if set]

    # Optional cached properties (same names as bundle.py)
    cache/B1               float64 (nV, nE)
    cache/L0               float64 (nV, nV)
    cache/layout           float64 (nV, 2)
    cache/hodge_gradient   float64 (nE,)
    cache/...

    # For TemporalRex
    snapshot/0/sources     int32 (nE_0,)
    snapshot/0/targets     int32 (nE_0,)
    snapshot/1/sources     ...
    face_snapshot/0/B2_col_ptr  int32 (nF_0+1,)  [if has face snapshots]
    face_snapshot/0/B2_row_idx  int32 (nnz,)
    ...

Metadata
~~~~~~~~
Safetensors metadata is `Dict[str, str]`. We encode a JSON blob under
the single key `rex_meta` with the full object description:

    {
      "object_type": "RexGraph",
      "nV": 42, "nE": 128, "nF": 17,
      "directed": false, "dimension": 2,
      "weighted": true, "has_attribution": false,
      "cache_scalars": {"betti": [1, 3, 0], "euler_characteristic": -2,
                        "chain_valid": true,
                        "hodge_pct_gradient": 0.41, ...},
      "core_arrays": ["boundary_ptr", "boundary_idx", "B2_col_ptr", ...],
      "cached_arrays": ["B1", "L0", "layout", ...],
      "w_boundary": {"0": 1.0, "1": 0.5},
      "bridge_version": 1
    }

Scalar cache entries (betti tuple, Euler characteristic, chain_valid
flag, Hodge percentages) are stored in `cache_scalars` inside
`rex_meta`, not as tensors. This matches bundle.py's manifest
`cache_scalars` field exactly.

Usage
~~~~~
    from rexgraph.io.safetensors_bridge import (
        rex_to_safetensors, safetensors_to_rex,
    )

    rex_to_safetensors(rex, "graph.safetensors")
    rex2 = safetensors_to_rex("graph.safetensors")
    assert rex2.nV == rex.nV

    # With precomputed cache
    rex_to_safetensors(rex, "graph.safetensors", cache="all")
    rex_to_safetensors(rex, "graph.safetensors", cache=["topology", "spectral"])

Fingerprint corpus export for ML consumption:

    from rexgraph.io.safetensors_bridge import (
        fingerprints_to_safetensors, safetensors_to_fingerprints,
    )
    fingerprints_to_safetensors(
        feature_matrix,            # float32 (n_spans, n_features)
        labels,                    # object/str (n_spans,)
        "fingerprints.safetensors",
        feature_names=feature_names,
        metadata={"corpus": "demo"},
    )
    m, labs, names, meta = safetensors_to_fingerprints("fingerprints.safetensors")
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "rex_to_safetensors",
    "safetensors_to_rex",
    "temporal_rex_to_safetensors",
    "safetensors_to_temporal_rex",
    "save_safetensors",
    "load_safetensors",
    "fingerprints_to_safetensors",
    "safetensors_to_fingerprints",
]


_BRIDGE_VERSION = 1


# Lazy safetensors import


def _st():
    """Lazily import safetensors.numpy. Raise ImportError if missing."""
    try:
        from safetensors.numpy import save_file, load_file
        from safetensors import safe_open
        return save_file, load_file, safe_open
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for safetensors features: "
            "pip install safetensors"
        ) from exc


# Cache group resolution (mirrors bundle._CACHE_GROUPS exactly)


_CACHE_GROUPS: Dict[str, List[str]] = {
    "algebra": [
        "B1", "B2", "L0", "L1", "L2",
        "overlap_adjacency", "L_overlap",
    ],
    "spectral": [
        "eigenvalues_L0", "fiedler_vector_L0",
        "fiedler_overlap_value", "fiedler_overlap_vector",
        "layout", "layout_3d",
    ],
    "topology": [
        "betti", "euler_characteristic", "chain_valid",
        "edge_types", "harmonic_space",
    ],
    "hodge": [
        "hodge_gradient", "hodge_curl", "hodge_harmonic",
    ],
}

_ALL_CACHEABLE: Set[str] = set()
for _entries in _CACHE_GROUPS.values():
    _ALL_CACHEABLE.update(_entries)
_ALL_CACHEABLE.update(_CACHE_GROUPS.keys())


def _resolve_cache(cache) -> Set[str]:
    """Expand a cache spec into the set of individual property names."""
    if cache is None:
        return set()
    if isinstance(cache, str):
        if cache == "all":
            return set(_ALL_CACHEABLE)
        if cache in _CACHE_GROUPS:
            return set(_CACHE_GROUPS[cache])
        return {cache}
    out: Set[str] = set()
    for c in cache:
        if c == "all":
            return set(_ALL_CACHEABLE)
        if c in _CACHE_GROUPS:
            out.update(_CACHE_GROUPS[c])
        else:
            out.add(c)
    return out


# Helpers for safetensors dtype hygiene


def _as_storable(arr: NDArray) -> NDArray:
    """Return a C-contiguous view of *arr* suitable for safetensors.

    Safetensors requires contiguous buffers of a fixed set of dtypes.
    Object/unicode arrays are rejected. We narrow numpy bool arrays to
    uint8 to avoid dtype-identity subtleties across versions (some
    safetensors builds serialize bool as 1-byte, some as native bool;
    uint8 is unambiguous).
    """
    arr = np.asarray(arr)
    if arr.dtype.kind in ("O", "U"):
        raise TypeError(
            f"Cannot store dtype {arr.dtype!r} in safetensors. "
            "Convert object/string arrays to numeric first."
        )
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _coerce_path(path: Union[str, os.PathLike]) -> pathlib.Path:
    """Normalize path to `pathlib.Path` with `.safetensors` suffix."""
    p = pathlib.Path(path)
    if p.suffix != ".safetensors":
        p = pathlib.Path(str(p) + ".safetensors")
    return p


# RexGraph <-> safetensors


def rex_to_safetensors(
    rex,
    path: Union[str, os.PathLike],
    *,
    cache: Union[None, str, List[str]] = None,
) -> pathlib.Path:
    """Write a RexGraph to a `.safetensors` file.

    Stores the core reconstruction arrays (boundary_ptr, boundary_idx,
    B2 triples, and optional w_E/attribution), plus any requested cache
    groups. Returns the resolved output path.

    Parameters
    ----------
    rex : RexGraph
        Source graph. Accessed via its public properties and private
        `_boundary_ptr`/`_boundary_idx`/`_B2_*` storage.
    path : str or os.PathLike
        Output file. `.safetensors` suffix appended if missing.
    cache : None, str, or list of str
        Precomputed property groups to include. Same vocabulary as
        `bundle.save_rex`:

          - `None` (default) -> no cache, only the reconstruction contract
          - `"all"` -> every property in `_ALL_CACHEABLE`
          - `"algebra"`, `"spectral"`, `"topology"`, `"hodge"` -> that group
          - list of names or groups -> union

    Returns
    -------
    pathlib.Path
        The file path that was written.
    """
    save_file, _, _ = _st()
    out = _coerce_path(path)

    tensors: Dict[str, NDArray] = {}
    core_arrays: List[str] = []

    # Core reconstruction arrays
    tensors["boundary_ptr"] = _as_storable(rex._boundary_ptr)
    core_arrays.append("boundary_ptr")
    tensors["boundary_idx"] = _as_storable(rex._boundary_idx)
    core_arrays.append("boundary_idx")
    tensors["B2_col_ptr"] = _as_storable(rex._B2_col_ptr)
    core_arrays.append("B2_col_ptr")
    tensors["B2_row_idx"] = _as_storable(rex._B2_row_idx)
    core_arrays.append("B2_row_idx")
    tensors["B2_vals"] = _as_storable(rex._B2_vals)
    core_arrays.append("B2_vals")

    if rex._w_E is not None:
        tensors["w_E"] = _as_storable(rex._w_E)
        core_arrays.append("w_E")

    has_attribution = False
    attribution = getattr(rex, "_attribution", None)
    if attribution is not None:
        tensors["attribution"] = _as_storable(attribution)
        core_arrays.append("attribution")
        has_attribution = True

    # Metadata
    meta: Dict[str, Any] = {
        "object_type": "RexGraph",
        "nV": int(rex.nV),
        "nE": int(rex.nE),
        "nF": int(rex.nF),
        "directed": bool(rex._directed),
        "dimension": int(rex.dimension),
        "weighted": rex._w_E is not None,
        "has_attribution": has_attribution,
        "core_arrays": core_arrays,
        "bridge_version": _BRIDGE_VERSION,
    }
    if rex._w_boundary:
        meta["w_boundary"] = {str(k): float(v) for k, v in rex._w_boundary.items()}

    # Optional cache groups
    names = _resolve_cache(cache)
    cached_arrays: List[str] = []
    scalar_cache: Dict[str, Any] = {}

    if names:
        _collect_cache(rex, names, tensors, cached_arrays, scalar_cache)

    if cached_arrays:
        meta["cached_arrays"] = cached_arrays
    if scalar_cache:
        meta["cache_scalars"] = scalar_cache

    # Safetensors metadata is strict Dict[str, str]; encode as JSON
    st_meta = {"rex_meta": json.dumps(meta)}

    save_file(tensors, str(out), metadata=st_meta)
    return out


def safetensors_to_rex(path: Union[str, os.PathLike]):
    """Reconstruct a RexGraph from a `.safetensors` file.

    Only the core reconstruction arrays plus weights are consumed by the
    RexGraph constructor. Any cached arrays present in the file are
    silently ignored on reconstruction (RexGraph recomputes them
    lazily); if you want the cached data too, use
    :func:`load_safetensors_full`.

    Parameters
    ----------
    path : str or os.PathLike
        Input file.

    Returns
    -------
    RexGraph
    """
    from ..graph import RexGraph

    _, load_file, _ = _st()
    p = _coerce_path(path)
    tensors = load_file(str(p))
    meta = _load_meta(str(p))

    if meta.get("object_type") != "RexGraph":
        raise TypeError(
            f"Safetensors file contains {meta.get('object_type')!r}, "
            "not RexGraph."
        )

    kw: Dict[str, Any] = {
        "boundary_ptr": tensors["boundary_ptr"],
        "boundary_idx": tensors["boundary_idx"],
        "directed": bool(meta.get("directed", False)),
    }
    for name in ("B2_col_ptr", "B2_row_idx", "B2_vals"):
        if name in tensors:
            kw[name] = tensors[name]
    if "w_E" in tensors:
        kw["w_E"] = tensors["w_E"]

    wb = meta.get("w_boundary")
    if wb:
        kw["w_boundary"] = {int(k): float(v) for k, v in wb.items()}

    rex = RexGraph(**kw)

    if "attribution" in tensors and hasattr(rex, "set_vertex_attribution"):
        rex.set_vertex_attribution(tensors["attribution"])

    return rex


# Full load (returns both the rex and any cached arrays/scalars)


def load_safetensors(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Load the full contents of a safetensors file as a dict.

    Returns a dict with keys:

        "object"     -> RexGraph or TemporalRex
        "tensors"    -> dict of all loaded arrays (including cache/*)
        "metadata"   -> the decoded `rex_meta` header
        "scalars"    -> contents of `cache_scalars` from metadata

    This is the escape hatch when a caller needs both the object and
    its precomputed cache without recomputing.
    """
    _, load_file, _ = _st()
    p = _coerce_path(path)
    tensors = load_file(str(p))
    meta = _load_meta(str(p))
    obj_type = meta.get("object_type")
    if obj_type == "RexGraph":
        obj = _rex_from_loaded(tensors, meta)
    elif obj_type == "TemporalRex":
        obj = _temporal_from_loaded(tensors, meta)
    else:
        raise ValueError(
            f"Unknown object_type {obj_type!r} in {p} (expected "
            "'RexGraph' or 'TemporalRex')."
        )
    return {
        "object": obj,
        "tensors": tensors,
        "metadata": meta,
        "scalars": meta.get("cache_scalars", {}),
    }


def save_safetensors(
    obj: Any,
    path: Union[str, os.PathLike],
    *,
    cache: Union[None, str, List[str]] = None,
) -> pathlib.Path:
    """Save a RexGraph or TemporalRex to `.safetensors`.

    Dispatches to `rex_to_safetensors` or `temporal_rex_to_safetensors`
    based on object type.
    """
    from ..graph import RexGraph, TemporalRex
    if isinstance(obj, TemporalRex):
        return temporal_rex_to_safetensors(obj, path)
    if isinstance(obj, RexGraph):
        return rex_to_safetensors(obj, path, cache=cache)
    raise TypeError(
        f"save_safetensors expects RexGraph or TemporalRex, got {type(obj).__name__}"
    )


# Cache collection (mirrors bundle._write_cache structure)


def _collect_cache(
    rex,
    names: Set[str],
    tensors: Dict[str, NDArray],
    cached_arrays: List[str],
    scalar_cache: Dict[str, Any],
) -> None:
    """Populate `tensors` and `scalar_cache` with requested cache entries.

    The name and access pattern for each entry matches
    `bundle._write_cache` exactly so cache data transfers between
    formats without translation.
    """

    def _try_array(prop_name: str, cache_name: Optional[str] = None) -> None:
        cn = cache_name or prop_name
        try:
            arr = getattr(rex, prop_name)
            if isinstance(arr, np.ndarray):
                tensors[f"cache/{cn}"] = _as_storable(arr)
                cached_arrays.append(cn)
        except Exception:
            # Property may be unavailable for this rex shape (e.g. nF=0)
            pass

    # algebra
    for key in ("B1", "B2", "L0", "L1", "L2",
                "overlap_adjacency", "L_overlap"):
        if names & {"algebra", key}:
            _try_array(key)

    # spectral
    for key in ("eigenvalues_L0", "fiedler_vector_L0",
                "layout", "layout_3d"):
        if names & {"spectral", key}:
            _try_array(key)

    if names & {"spectral", "fiedler_overlap_value", "fiedler_overlap_vector"}:
        try:
            val, vec = rex.fiedler_overlap
            scalar_cache["fiedler_overlap_value"] = float(val)
            tensors["cache/fiedler_overlap_vector"] = _as_storable(vec)
            cached_arrays.append("fiedler_overlap_vector")
        except Exception:
            pass

    # topology
    if names & {"topology", "betti"}:
        try:
            scalar_cache["betti"] = list(rex.betti)
        except Exception:
            pass
    if names & {"topology", "euler_characteristic"}:
        try:
            scalar_cache["euler_characteristic"] = int(rex.euler_characteristic)
        except Exception:
            pass
    if names & {"topology", "chain_valid"}:
        try:
            scalar_cache["chain_valid"] = bool(rex.chain_valid)
        except Exception:
            pass
    if names & {"topology", "edge_types"}:
        _try_array("edge_types")
    if names & {"topology", "harmonic_space"}:
        _try_array("harmonic_space")

    # hodge
    if names & {"hodge", "hodge_gradient", "hodge_curl", "hodge_harmonic"}:
        try:
            w = rex.w_E if rex.w_E is not None else np.ones(rex.nE)
            grad, curl, harm = rex.hodge(w)
            tensors["cache/hodge_gradient"] = _as_storable(grad)
            tensors["cache/hodge_curl"] = _as_storable(curl)
            tensors["cache/hodge_harmonic"] = _as_storable(harm)
            cached_arrays.extend(["hodge_gradient", "hodge_curl", "hodge_harmonic"])
            total = float(np.dot(w, w))
            if total > 0:
                scalar_cache["hodge_pct_gradient"] = float(np.dot(grad, grad) / total)
                scalar_cache["hodge_pct_curl"] = float(np.dot(curl, curl) / total)
                scalar_cache["hodge_pct_harmonic"] = float(np.dot(harm, harm) / total)
        except Exception:
            pass


# TemporalRex <-> safetensors


def temporal_rex_to_safetensors(
    trex,
    path: Union[str, os.PathLike],
) -> pathlib.Path:
    """Write a TemporalRex to a `.safetensors` file.

    Each snapshot is stored under `snapshot/<t>/sources` +
    `snapshot/<t>/targets` (or `snapshot/<t>/boundary_ptr` +
    `boundary_idx` in general mode). Face snapshots, if present, go
    under `face_snapshot/<t>/B2_col_ptr` + `B2_row_idx`.
    """
    save_file, _, _ = _st()
    out = _coerce_path(path)

    tensors: Dict[str, NDArray] = {}

    T = trex.T
    general = bool(getattr(trex, "_general", False))
    for t in range(T):
        snap = trex._snapshots[t]
        if general:
            tensors[f"snapshot/{t}/boundary_ptr"] = _as_storable(snap[0])
            tensors[f"snapshot/{t}/boundary_idx"] = _as_storable(snap[1])
        else:
            tensors[f"snapshot/{t}/sources"] = _as_storable(snap[0])
            tensors[f"snapshot/{t}/targets"] = _as_storable(snap[1])

    has_face_snapshots = bool(getattr(trex, "_face_snapshots", False))
    if has_face_snapshots:
        for t, fsnap in enumerate(trex._face_snapshots):
            tensors[f"face_snapshot/{t}/B2_col_ptr"] = _as_storable(fsnap[0])
            tensors[f"face_snapshot/{t}/B2_row_idx"] = _as_storable(fsnap[1])

    meta: Dict[str, Any] = {
        "object_type": "TemporalRex",
        "T": int(T),
        "directed": bool(getattr(trex, "_directed", False)),
        "general": general,
        "has_face_snapshots": has_face_snapshots,
        "bridge_version": _BRIDGE_VERSION,
    }

    st_meta = {"rex_meta": json.dumps(meta)}
    save_file(tensors, str(out), metadata=st_meta)
    return out


def safetensors_to_temporal_rex(path: Union[str, os.PathLike]):
    """Reconstruct a TemporalRex from a `.safetensors` file."""
    from ..graph import TemporalRex

    _, load_file, _ = _st()
    p = _coerce_path(path)
    tensors = load_file(str(p))
    meta = _load_meta(str(p))

    if meta.get("object_type") != "TemporalRex":
        raise TypeError(
            f"Safetensors file contains {meta.get('object_type')!r}, "
            "not TemporalRex."
        )
    return _temporal_from_loaded(tensors, meta)


# Internal reconstructors used by load_safetensors


def _rex_from_loaded(tensors: Dict[str, NDArray], meta: Dict[str, Any]):
    from ..graph import RexGraph

    kw: Dict[str, Any] = {
        "boundary_ptr": tensors["boundary_ptr"],
        "boundary_idx": tensors["boundary_idx"],
        "directed": bool(meta.get("directed", False)),
    }
    for name in ("B2_col_ptr", "B2_row_idx", "B2_vals"):
        if name in tensors:
            kw[name] = tensors[name]
    if "w_E" in tensors:
        kw["w_E"] = tensors["w_E"]
    wb = meta.get("w_boundary")
    if wb:
        kw["w_boundary"] = {int(k): float(v) for k, v in wb.items()}

    rex = RexGraph(**kw)

    if "attribution" in tensors and hasattr(rex, "set_vertex_attribution"):
        rex.set_vertex_attribution(tensors["attribution"])

    return rex


def _temporal_from_loaded(tensors: Dict[str, NDArray], meta: Dict[str, Any]):
    from ..graph import TemporalRex

    T = int(meta["T"])
    directed = bool(meta.get("directed", False))
    general = bool(meta.get("general", False))

    snapshots = []
    for t in range(T):
        if general:
            snapshots.append((
                tensors[f"snapshot/{t}/boundary_ptr"],
                tensors[f"snapshot/{t}/boundary_idx"],
            ))
        else:
            snapshots.append((
                tensors[f"snapshot/{t}/sources"],
                tensors[f"snapshot/{t}/targets"],
            ))

    face_snapshots: List[Tuple[NDArray, NDArray]] = []
    if meta.get("has_face_snapshots"):
        t = 0
        while f"face_snapshot/{t}/B2_col_ptr" in tensors:
            face_snapshots.append((
                tensors[f"face_snapshot/{t}/B2_col_ptr"],
                tensors[f"face_snapshot/{t}/B2_row_idx"],
            ))
            t += 1

    trex = TemporalRex(
        snapshots=snapshots,
        directed=directed,
    )
    if face_snapshots:
        trex._face_snapshots = face_snapshots
    return trex


def _load_meta(path: str) -> Dict[str, Any]:
    """Load and parse the `rex_meta` JSON from a safetensors file header."""
    _, _, safe_open = _st()
    with safe_open(path, framework="numpy") as f:
        raw = f.metadata() or {}
    if "rex_meta" not in raw:
        raise ValueError(
            f"File {path} has no `rex_meta` key in its header; "
            "it was not written by rex_to_safetensors."
        )
    return json.loads(raw["rex_meta"])


# Fingerprint corpus export
#
# Downstream consumers may produce collections of fingerprint rows. For
# ML consumption we want a single flat feature matrix plus labels plus
# per-column names, which matches exactly what safetensors was built
# for. This function serves that use case without coupling to any
# external dataclass - it just takes arrays.


def fingerprints_to_safetensors(
    feature_matrix: NDArray,
    labels: Optional[NDArray],
    path: Union[str, os.PathLike],
    *,
    feature_names: Optional[List[str]] = None,
    block_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> pathlib.Path:
    """Write a fingerprint corpus to a `.safetensors` file.

    Parameters
    ----------
    feature_matrix : ndarray, shape (n_spans, n_features)
        The stacked fingerprint vectors. Must be a numeric dtype.
    labels : ndarray or None
        Per-span labels. If string-valued, they are encoded as a numeric
        label id array with the `label_names` mapping stored in
        metadata. If numeric, stored as-is.
    path : str or os.PathLike
        Output file. `.safetensors` suffix appended if missing.
    feature_names : list of str, optional
        Per-feature column names. Stored under `feature_names` in
        metadata for round-trip.
    block_offsets : dict, optional
        Named `{block_name: (start, end)}` slices of the feature vector,
        mirroring the standard fingerprint vector block layout. Stored
        in metadata for downstream use.
    metadata : dict, optional
        Additional JSON-serializable metadata (corpus name, build date,
        fingerprint schema version, etc.).

    Returns
    -------
    pathlib.Path
        The file path that was written.
    """
    save_file, _, _ = _st()
    out = _coerce_path(path)

    feature_matrix = np.asarray(feature_matrix)
    if feature_matrix.ndim != 2:
        raise ValueError(
            f"feature_matrix must be 2-D, got shape {feature_matrix.shape}"
        )

    tensors: Dict[str, NDArray] = {
        "features": _as_storable(feature_matrix),
    }

    fp_meta: Dict[str, Any] = {
        "object_type": "FingerprintCorpus",
        "n_spans": int(feature_matrix.shape[0]),
        "n_features": int(feature_matrix.shape[1]),
        "features_dtype": str(feature_matrix.dtype),
        "bridge_version": _BRIDGE_VERSION,
    }

    if labels is not None:
        labels_arr = np.asarray(labels)
        if labels_arr.dtype.kind in ("U", "O"):
            # Encode string labels as int32 label_ids + lookup table
            unique_labels, label_ids = np.unique(labels_arr, return_inverse=True)
            tensors["label_ids"] = _as_storable(label_ids.astype(np.int32))
            fp_meta["label_names"] = [str(x) for x in unique_labels.tolist()]
            fp_meta["labels_are_encoded"] = True
        else:
            tensors["labels"] = _as_storable(labels_arr)
            fp_meta["labels_are_encoded"] = False

    if feature_names is not None:
        if len(feature_names) != feature_matrix.shape[1]:
            raise ValueError(
                f"feature_names has length {len(feature_names)} but "
                f"feature_matrix has {feature_matrix.shape[1]} columns."
            )
        fp_meta["feature_names"] = list(feature_names)

    if block_offsets is not None:
        # JSON cannot store tuples as tuples; serialize as [start, end] lists
        fp_meta["block_offsets"] = {
            k: [int(v[0]), int(v[1])] for k, v in block_offsets.items()
        }

    if metadata is not None:
        for k, v in metadata.items():
            if k in fp_meta:
                raise KeyError(
                    f"metadata key {k!r} conflicts with reserved fingerprint "
                    "metadata"
                )
            fp_meta[k] = v

    st_meta = {"rex_meta": json.dumps(fp_meta)}
    save_file(tensors, str(out), metadata=st_meta)
    return out


def safetensors_to_fingerprints(
    path: Union[str, os.PathLike],
) -> Tuple[NDArray, Optional[NDArray], Optional[List[str]], Dict[str, Any]]:
    """Load a fingerprint corpus from a `.safetensors` file.

    Returns
    -------
    feature_matrix : ndarray (n_spans, n_features)
    labels : ndarray or None
        String-valued 1-D array if labels were stored as strings,
        numeric array if stored numerically, or None if no labels
        were stored.
    feature_names : list of str or None
    metadata : dict
        Remaining metadata keys (excluding reserved fields already
        surfaced as return values).
    """
    _, load_file, _ = _st()
    p = _coerce_path(path)
    tensors = load_file(str(p))
    meta = _load_meta(str(p))
    if meta.get("object_type") != "FingerprintCorpus":
        raise TypeError(
            f"Safetensors file contains {meta.get('object_type')!r}, "
            "not FingerprintCorpus."
        )

    features = tensors["features"]

    labels: Optional[NDArray] = None
    if meta.get("labels_are_encoded"):
        ids = tensors["label_ids"]
        names = meta.get("label_names", [])
        labels = np.array([names[int(i)] for i in ids], dtype=object)
    elif "labels" in tensors:
        labels = tensors["labels"]

    feature_names = meta.get("feature_names")

    # Strip reserved fields from the returned metadata dict
    reserved = {
        "object_type", "n_spans", "n_features", "features_dtype",
        "bridge_version", "label_names", "labels_are_encoded",
        "feature_names",
    }
    residual = {k: v for k, v in meta.items() if k not in reserved}

    # Restore tuple shape for block_offsets if present
    if "block_offsets" in residual:
        residual["block_offsets"] = {
            k: tuple(v) for k, v in residual["block_offsets"].items()
        }

    return features, labels, feature_names, residual
