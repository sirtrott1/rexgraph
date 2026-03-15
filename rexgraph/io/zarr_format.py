# rexgraph/io/zarr_format.py
"""
Zarr-based storage for the rex framework.

Stores RexGraph, TemporalRex, NamedTuples, and raw arrays in chunked,
compressed Zarr stores. Works with both Zarr v2 and v3.

On-disk layout for a RexGraph (graph.zarr/):

    .zattrs                 nV, nE, nF, directed, dimension, format_version
    boundary_ptr/           int32 (nE+1,)
    boundary_idx/           int32 (nnz,)
    B2_col_ptr/             int32 (nF+1,)
    B2_row_idx/             int32 (nnz,)
    B2_vals/                float64 (nnz,)
    w_E/                    float64 (nE,) [optional]
    attribution/            float64 (nV, d) [optional]
    algebra/                B1, B2, L0, L1, L2, overlap_adjacency, L_overlap
    spectral/               full spectral_bundle dict
    relational/             RL_1, coupling constants, L1_alpha, Lambda
    topology/               betti, euler, chain_valid, edge_types, cycle_basis
    hodge/                  gradient, curl, harmonic, rho, energy fractions
    faces/                  detected face data and metrics
    field/                  field operator M, eigenvalues, mode classification
    wave/                   density matrices
    signal/                 perturbation trajectories, cascade, BIOES
    quotient/               subcomplex masks, quotient B1/B2, relative betti
    persistence/            diagrams, barcodes, enrichment
    temporal/               edge/face lifecycle, betti matrix, BIOES result
    standard_metrics/       PageRank, betweenness, clustering, Louvain
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ._compat import (
    HAS_ZARR,
    ZARR_V3,
    ensure_zarr_suffix,
    rm_rf,
    open_root_group,
    create_root_group,
    default_zarr_compressor,
    normalize_zarr_compressor,
    g_create_array,
    g_store_complex,
    g_load_complex,
    g_store_sparse_csr,
    g_load_sparse_csr,
    g_store_dict,
    g_load_dict,
    g_store_bool_masks,
    g_load_bool_masks,
    write_text_array,
    read_text_array,
    to_native,
    json_default,
    as_str,
)

if HAS_ZARR:
    import zarr

__all__ = [
    "RexZarrFormat",
    "save_zarr",
    "load_zarr",
    "save_zarr_array",
    "load_zarr_array",
]

_FORMAT_VERSION = "2.0.0"

# Cache group definitions

_CACHE_GROUPS: Dict[str, List[str]] = {
    "algebra": [
        "B1", "B2", "L0", "L1", "L2",
        "L1_down", "L1_up",
        "overlap_adjacency", "L_overlap",
    ],
    "spectral": [
        "spectral_bundle",
        "eigenvalues_L0", "fiedler_vector_L0",
        "evals_L1", "evecs_L1",
        "evals_L2",
        "evals_L_O", "evecs_L_O",
        "diag_L1_down", "diag_L1_up",
        "fiedler_overlap",
        "layout", "layout_3d",
    ],
    "relational": [
        "relational_laplacian",
        "evals_RL1", "evecs_RL1",
        "alpha_G", "alpha_T",
        "L1_alpha", "evals_L1a", "evecs_L1a",
        "Lambda", "evals_Lambda", "evecs_Lambda",
    ],
    "topology": [
        "betti", "euler_characteristic", "chain_valid",
        "edge_types", "cycle_basis", "harmonic_space",
        "nF_hodge", "self_loop_face_indices", "B2_hodge",
    ],
    "hodge": [
        "hodge_decomposition",
        "hodge_rho",
    ],
    "faces": [
        "detected_faces", "face_metrics",
    ],
    "field": [
        "field_operator", "field_eigen", "mode_classification",
    ],
    "wave": [
        "density_matrix",
    ],
    "signal": [
        "perturbation_result", "field_perturbation_result",
    ],
    "quotient": [
        "subcomplex_masks", "quotient_result",
    ],
    "persistence": [
        "persistence_diagram", "persistence_enrichment",
    ],
    "temporal": [
        "edge_lifecycle", "edge_metrics",
        "face_lifecycle", "bioes_result",
        "betti_matrix",
    ],
    "standard_metrics": [
        "standard_metrics",
    ],
}

_ALL_CACHEABLE: Set[str] = set()
for _entries in _CACHE_GROUPS.values():
    _ALL_CACHEABLE.update(_entries)
_ALL_CACHEABLE.update(_CACHE_GROUPS.keys())


# Simple array save/load

def save_zarr_array(arr: NDArray, path: str) -> None:
    """Save a NumPy array into a .zarr store."""
    p = ensure_zarr_suffix(path)
    if os.path.exists(p):
        rm_rf(p)
    root = create_root_group(p)
    root.attrs["object_type"] = "array"
    root.attrs["format_version"] = _FORMAT_VERSION
    g_store_complex(root, "data", arr)


def load_zarr_array(path: str) -> np.ndarray:
    """Load a NumPy array from a .zarr store."""
    root = open_root_group(ensure_zarr_suffix(path), mode="r")
    return g_load_complex(root, "data")


# RexZarrFormat

class RexZarrFormat:
    """Zarr-based on-disk format for the rex framework.

    Parameters
    ----------
    compressor
        Compression codec. Accepts numcodecs.Blosc, string shorthands
        ("blosc", "zstd", "none"), or None.
        Default: Blosc(zstd, clevel=3, bitshuffle).
    chunks : bool or tuple
        Chunking strategy. True for automatic, False to disable.
    large_threshold : int
        Edge count above which chunked writes are used for large
        arrays. Default: 50000.
    """

    extension = ".zarr"

    def __init__(
        self,
        compressor: Any = "default",
        chunks: Union[bool, Tuple[int, ...]] = True,
        large_threshold: int = 50_000,
    ):
        if not HAS_ZARR:
            raise ImportError("zarr is required: pip install zarr")
        self.compressor = normalize_zarr_compressor(
            default_zarr_compressor() if compressor == "default" else compressor
        )
        self.chunks = chunks
        self.large_threshold = large_threshold

    # Internal helpers

    def _store(self, group, name: str, arr: NDArray, **kw) -> None:
        """Store a dense or complex array."""
        g_store_complex(group, name, np.asarray(arr),
                        compressor=self.compressor, chunks=self.chunks)

    def _store_chunked(
        self, group, name: str, arr: NDArray, chunk_rows: int = 10_000
    ) -> None:
        """Store a large 2D array with explicit row-chunking."""
        arr = np.asarray(arr)
        if arr.ndim == 2:
            cr = min(chunk_rows, arr.shape[0])
            cc = arr.shape[1]
            g_create_array(
                group, name, data=arr,
                compressor=self.compressor, chunks=(cr, cc),
            )
        elif arr.ndim == 1:
            cr = min(chunk_rows, arr.shape[0])
            g_create_array(
                group, name, data=arr,
                compressor=self.compressor, chunks=(cr,),
            )
        else:
            self._store(group, name, arr)

    def _load(self, group, name: str) -> np.ndarray:
        """Load an array, handling complex and sparse subgroups."""
        obj = group[name]
        if hasattr(obj, "attrs"):
            if obj.attrs.get("is_complex", False):
                return g_load_complex(group, name)
            if obj.attrs.get("is_sparse", False):
                return g_load_sparse_csr(group, name, dense=True)
        return np.asarray(obj)

    def _has(self, group, name: str) -> bool:
        try:
            return name in group
        except Exception:
            return False

    def _is_large(self, rex) -> bool:
        return rex.nE >= self.large_threshold

    def _get_or_create(self, group, name: str):
        """Get existing subgroup or create a new one."""
        try:
            return group.create_group(name)
        except Exception:
            return group[name]

    # Public API

    def write(
        self,
        path: str,
        obj: Any,
        *,
        cache: Union[None, str, List[str]] = None,
    ) -> None:
        """Write a RexGraph, TemporalRex, or ndarray to disk.

        Parameters
        ----------
        path : str
            Output path (.zarr suffix added if missing).
        obj : RexGraph, TemporalRex, or ndarray
        cache : None, "all", or list of str
            Precomputed results to include. "all" writes everything.
            Group names: "algebra", "spectral", "relational",
            "topology", "hodge", "faces", "field", "wave", "signal",
            "quotient", "persistence", "temporal", "standard_metrics".
        """
        from ..graph import RexGraph, TemporalRex

        path = ensure_zarr_suffix(path)
        if os.path.exists(path):
            rm_rf(path)

        root = create_root_group(path)
        root.attrs["format_version"] = _FORMAT_VERSION
        root.attrs["zarr_v3"] = ZARR_V3

        if isinstance(obj, TemporalRex):
            root.attrs["object_type"] = "TemporalRex"
            self._write_temporal_rex(root, obj, cache=cache)
        elif isinstance(obj, RexGraph):
            root.attrs["object_type"] = "RexGraph"
            self._write_rex_graph(root, obj, cache=cache)
        elif isinstance(obj, np.ndarray):
            root.attrs["object_type"] = "array"
            self._store(root, "data", obj)
        else:
            raise TypeError(f"Unsupported type: {type(obj).__name__}")

    def read(self, path: str) -> Any:
        """Read a RexGraph, TemporalRex, or ndarray from disk."""
        from ..graph import RexGraph, TemporalRex

        root = open_root_group(ensure_zarr_suffix(path), mode="r")
        obj_type = as_str(root.attrs.get("object_type"))

        if obj_type == "RexGraph":
            return self._read_rex_graph(root)
        if obj_type == "TemporalRex":
            return self._read_temporal_rex(root)
        if obj_type == "array":
            return self._load(root, "data")
        raise TypeError(f"Unknown object_type: {obj_type}")

    # Container (multi-object) API

    def write_to_group(self, path: str, name: str, obj: Any, **kw) -> None:
        """Write an object into /objects/<name> inside path."""
        from ..graph import RexGraph, TemporalRex

        path = ensure_zarr_suffix(path)
        if not os.path.exists(path):
            root = create_root_group(path)
            root.attrs["format_version"] = _FORMAT_VERSION
            root.attrs["object_type"] = "Container"
        else:
            root = open_root_group(path, mode="a")

        objs = root.require_group("objects")
        if name in objs:
            try:
                del objs[name]
            except Exception:
                pass

        g = objs.create_group(name)
        if isinstance(obj, TemporalRex):
            g.attrs["object_type"] = "TemporalRex"
            self._write_temporal_rex(g, obj, **kw)
        elif isinstance(obj, RexGraph):
            g.attrs["object_type"] = "RexGraph"
            self._write_rex_graph(g, obj, **kw)
        elif isinstance(obj, np.ndarray):
            g.attrs["object_type"] = "array"
            self._store(g, "data", obj)
        else:
            raise TypeError(f"Unsupported type: {type(obj).__name__}")

    def read_from_group(self, path: str, name: str) -> Any:
        """Read /objects/<name> from path."""
        root = open_root_group(ensure_zarr_suffix(path), mode="r")
        g = root["objects"][name]
        t = as_str(g.attrs.get("object_type"))
        from ..graph import RexGraph, TemporalRex
        if t == "RexGraph":
            return self._read_rex_graph(g)
        if t == "TemporalRex":
            return self._read_temporal_rex(g)
        if t == "array":
            return self._load(g, "data")
        raise TypeError(f"Unknown object_type in group '{name}': {t}")

    def list_groups(self, path: str) -> List[str]:
        """List sub-object names in a container store."""
        path = ensure_zarr_suffix(path)
        if not os.path.exists(path):
            return []
        root = open_root_group(path, mode="r")
        return list(root["objects"].keys()) if "objects" in root else []

    # RexGraph serialization

    def _write_rex_graph(self, g, rex, *, cache=None) -> None:
        """Serialize a RexGraph.

        Always writes the reconstruction arrays required by
        RexGraph.from_dict(): boundary_ptr, boundary_idx,
        B2_col_ptr, B2_row_idx, B2_vals, w_E, w_boundary, directed.
        """
        large = self._is_large(rex)

        # Metadata
        g.attrs["nV"] = int(rex.nV)
        g.attrs["nE"] = int(rex.nE)
        g.attrs["nF"] = int(rex.nF)
        g.attrs["directed"] = bool(rex._directed)
        g.attrs["dimension"] = int(rex.dimension)

        # Core reconstruction arrays
        self._store(g, "boundary_ptr", rex._boundary_ptr)
        self._store(g, "boundary_idx", rex._boundary_idx)

        self._store(g, "B2_col_ptr", rex._B2_col_ptr)
        self._store(g, "B2_row_idx", rex._B2_row_idx)
        self._store(g, "B2_vals", rex._B2_vals)

        if rex._w_E is not None:
            self._store(g, "w_E", rex._w_E)

        if rex._w_boundary:
            g.attrs["w_boundary"] = json.dumps(
                {str(k): v for k, v in rex._w_boundary.items()},
                default=json_default,
            )

        if hasattr(rex, "_attribution") and rex._attribution is not None:
            store_fn = self._store_chunked if large else self._store
            store_fn(g, "attribution", rex._attribution)

        if cache:
            self._write_cache(g, rex, cache, large)

    def _read_rex_graph(self, g) -> "RexGraph":
        """Reconstruct a RexGraph from a Zarr group."""
        from ..graph import RexGraph

        kw: dict = {
            "boundary_ptr": self._load(g, "boundary_ptr"),
            "boundary_idx": self._load(g, "boundary_idx"),
            "directed": bool(g.attrs.get("directed", False)),
        }

        if self._has(g, "B2_col_ptr"):
            kw["B2_col_ptr"] = self._load(g, "B2_col_ptr")
            kw["B2_row_idx"] = self._load(g, "B2_row_idx")
            kw["B2_vals"] = self._load(g, "B2_vals")

        if self._has(g, "w_E"):
            kw["w_E"] = self._load(g, "w_E")

        wb_raw = g.attrs.get("w_boundary")
        if wb_raw:
            wb_raw = as_str(wb_raw)
            kw["w_boundary"] = {int(k): v for k, v in json.loads(wb_raw).items()}

        rex = RexGraph(**kw)

        if self._has(g, "attribution"):
            rex.set_vertex_attribution(self._load(g, "attribution"))

        return rex

    # TemporalRex serialization

    def _write_temporal_rex(self, g, trex, *, cache=None) -> None:
        """Serialize a TemporalRex with all snapshots and optional cache."""
        T = trex.T
        g.attrs["T"] = T
        g.attrs["directed"] = bool(trex._directed)
        g.attrs["general"] = bool(trex._general)

        sg = g.create_group("snapshots")
        for t in range(T):
            tg = sg.create_group(str(t))
            snap = trex._snapshots[t]
            if trex._general:
                self._store(tg, "boundary_ptr", snap[0])
                self._store(tg, "boundary_idx", snap[1])
            else:
                self._store(tg, "sources", snap[0])
                self._store(tg, "targets", snap[1])

        if trex._face_snapshots:
            fg = g.create_group("face_snapshots")
            for t, fsnap in enumerate(trex._face_snapshots):
                ftg = fg.create_group(str(t))
                self._store(ftg, "B2_col_ptr", fsnap[0])
                self._store(ftg, "B2_row_idx", fsnap[1])

        if cache:
            # Cache on final snapshot
            rex_final = trex.at(T - 1)
            self._write_cache(g, rex_final, cache, self._is_large(rex_final))
            # Temporal-specific cache
            self._write_temporal_cache(g, trex, cache)

    def _write_temporal_cache(self, g, trex, cache) -> None:
        """Write TemporalRex-specific cached data."""
        names = self._resolve_cache_names(cache)
        if not names:
            return

        temporal_names = names & (set(_CACHE_GROUPS.get("temporal", [])) | {"temporal"})
        if not temporal_names:
            return

        tg = self._get_or_create(g, "temporal")

        if "edge_lifecycle" in names or "temporal" in names:
            try:
                first_seen, last_seen, duration = trex.edge_lifecycle
                self._store(tg, "first_seen", first_seen)
                self._store(tg, "last_seen", last_seen)
                self._store(tg, "duration", duration)
            except Exception:
                pass

        if "edge_metrics" in names or "temporal" in names:
            try:
                counts, born, died = trex.edge_metrics
                self._store(tg, "edge_counts", counts)
                self._store(tg, "edge_born", born)
                self._store(tg, "edge_died", died)
            except Exception:
                pass

        if "face_lifecycle" in names or "temporal" in names:
            try:
                fld = trex.face_lifecycle_data
                if fld is not None:
                    flg = self._get_or_create(tg, "face_lifecycle")
                    for i, arr in enumerate(fld):
                        self._store(flg, f"arr_{i}", arr)
                    flg.attrs["n_arrays"] = len(fld)
            except Exception:
                pass

        if "bioes_result" in names or "temporal" in names:
            try:
                from ._serialization import ZarrAdapter, write_namedtuple
                bioes = trex.bioes_joint()
                adapter = ZarrAdapter(tg, compressor=self.compressor,
                                      chunks=self.chunks)
                write_namedtuple(adapter, "bioes_result", bioes)
            except Exception:
                pass

        if "betti_matrix" in names or "temporal" in names:
            try:
                T = trex.T
                betti_rows = []
                for t in range(T):
                    snap = trex.at(t)
                    betti_rows.append(list(snap.betti))
                self._store(tg, "betti_matrix",
                            np.array(betti_rows, dtype=np.int64))
            except Exception:
                pass

    def _read_temporal_rex(self, g) -> "TemporalRex":
        """Reconstruct a TemporalRex from a Zarr group."""
        from ..graph import TemporalRex

        T = int(g.attrs["T"])
        directed = bool(g.attrs.get("directed", False))
        general = bool(g.attrs.get("general", False))

        sg = g["snapshots"]
        snapshots = []
        for t in range(T):
            tg = sg[str(t)]
            if general:
                snapshots.append((
                    self._load(tg, "boundary_ptr"),
                    self._load(tg, "boundary_idx"),
                ))
            else:
                snapshots.append((
                    self._load(tg, "sources"),
                    self._load(tg, "targets"),
                ))

        face_snapshots = []
        if self._has(g, "face_snapshots"):
            fg = g["face_snapshots"]
            keys = sorted(fg.keys(), key=int) if hasattr(fg, "keys") else [str(i) for i in range(T)]
            for k in keys:
                if k not in fg:
                    break
                ftg = fg[k]
                face_snapshots.append((
                    self._load(ftg, "B2_col_ptr"),
                    self._load(ftg, "B2_row_idx"),
                ))

        return TemporalRex(
            snapshots,
            face_snapshots=face_snapshots or None,
            directed=directed,
            general=general,
        )

    # Cache resolution

    def _resolve_cache_names(self, cache) -> Set[str]:
        """Expand cache spec into individual property names."""
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

    # Cache writing

    def _write_cache(self, g, rex, cache, large: bool) -> None:
        """Write precomputed properties into subgroups.

        Errors during computation are silently skipped.
        """
        names = self._resolve_cache_names(cache)
        if not names:
            return

        store_fn = self._store_chunked if large else self._store

        self._write_algebra_cache(g, rex, names, store_fn)
        self._write_spectral_cache(g, rex, names, store_fn)
        self._write_relational_cache(g, rex, names, store_fn)
        self._write_topology_cache(g, rex, names, store_fn)
        self._write_hodge_cache(g, rex, names)
        self._write_faces_cache(g, rex, names)
        self._write_field_cache(g, rex, names, store_fn)
        self._write_signal_cache(g, rex, names)
        self._write_quotient_cache(g, rex, names)
        self._write_persistence_cache(g, rex, names)
        self._write_standard_metrics_cache(g, rex, names)

    def _write_algebra_cache(self, g, rex, names, store_fn) -> None:
        algebra_props = {"B1", "B2", "L0", "L1", "L2",
                         "L1_down", "L1_up",
                         "overlap_adjacency", "L_overlap"}
        if not (names & (algebra_props | {"algebra"})):
            return
        ag = self._get_or_create(g, "algebra")
        for prop in algebra_props:
            if prop in names or "algebra" in names:
                try:
                    arr = getattr(rex, prop)
                    if arr is not None:
                        store_fn(ag, prop, arr)
                except Exception:
                    pass

    def _write_spectral_cache(self, g, rex, names, store_fn) -> None:
        spectral_props = {"eigenvalues_L0", "fiedler_vector_L0",
                          "evals_L1", "evecs_L1", "evals_L2",
                          "evals_L_O", "evecs_L_O",
                          "diag_L1_down", "diag_L1_up",
                          "fiedler_overlap", "layout", "layout_3d",
                          "spectral_bundle"}
        if not (names & (spectral_props | {"spectral"})):
            return
        sg = self._get_or_create(g, "spectral")

        # Full spectral_bundle dict
        if "spectral_bundle" in names or "spectral" in names:
            try:
                sb = rex.spectral_bundle
                g_store_dict(sg, "bundle", sb,
                             compressor=self.compressor, chunks=self.chunks)
            except Exception:
                pass

        # Individual spectral properties
        for prop in ("eigenvalues_L0", "fiedler_vector_L0",
                      "layout", "layout_3d"):
            if prop in names or "spectral" in names:
                try:
                    self._store(sg, prop, getattr(rex, prop))
                except Exception:
                    pass

        # Spectral bundle array fields
        for prop in ("evals_L1", "evecs_L1", "evals_L2",
                      "evals_L_O", "evecs_L_O",
                      "diag_L1_down", "diag_L1_up"):
            if prop in names or "spectral" in names:
                try:
                    val = rex.spectral_bundle.get(prop)
                    if val is not None:
                        store_fn(sg, prop, val)
                except Exception:
                    pass

        # Fiedler overlap (value + vector pair)
        if "fiedler_overlap" in names or "spectral" in names:
            try:
                val, vec = rex.fiedler_overlap
                fog = self._get_or_create(sg, "fiedler_overlap")
                fog.attrs["value"] = float(val)
                self._store(fog, "vector", vec)
            except Exception:
                pass

    def _write_relational_cache(self, g, rex, names, store_fn) -> None:
        rel_props = {"relational_laplacian",
                     "evals_RL1", "evecs_RL1",
                     "alpha_G", "alpha_T",
                     "L1_alpha", "evals_L1a", "evecs_L1a",
                     "Lambda", "evals_Lambda", "evecs_Lambda"}
        if not (names & (rel_props | {"relational"})):
            return
        rg = self._get_or_create(g, "relational")

        # RL_1 matrix
        if "relational_laplacian" in names or "relational" in names:
            try:
                rl = rex.rex_laplacian
                if rl is not None:
                    store_fn(rg, "RL_1", rl)
            except Exception:
                pass

        # Spectral bundle fields for relational block
        sb_keys = {
            "evals_RL1": "evals_RL_1",
            "evecs_RL1": "evecs_RL_1",
            "L1_alpha": "L1_alpha",
            "evals_L1a": "evals_L1a",
            "evecs_L1a": "evecs_L1a",
            "Lambda": "Lambda",
            "evals_Lambda": "evals_Lambda",
            "evecs_Lambda": "evecs_Lambda",
        }
        for cache_name, sb_key in sb_keys.items():
            if cache_name in names or "relational" in names:
                try:
                    val = rex.spectral_bundle.get(sb_key)
                    if val is not None:
                        store_fn(rg, cache_name, val)
                except Exception:
                    pass

        # Coupling constants
        if "alpha_G" in names or "relational" in names:
            try:
                sb = rex.spectral_bundle
                rg.attrs["alpha_G"] = to_native(sb.get("alpha_G", float("nan")))
                rg.attrs["alpha_T"] = to_native(sb.get("alpha_T", 0.0))
                rg.attrs["alpha_used"] = to_native(sb.get("alpha_used", float("nan")))
            except Exception:
                pass

    def _write_topology_cache(self, g, rex, names, store_fn) -> None:
        topo_props = {"betti", "euler_characteristic", "chain_valid",
                      "edge_types", "cycle_basis", "harmonic_space",
                      "nF_hodge", "self_loop_face_indices", "B2_hodge"}
        if not (names & (topo_props | {"topology"})):
            return
        tg = self._get_or_create(g, "topology")

        if "betti" in names or "topology" in names:
            try:
                b0, b1, b2 = rex.betti
                tg.attrs["betti"] = json.dumps([b0, b1, b2])
            except Exception:
                pass

        if "euler_characteristic" in names or "topology" in names:
            try:
                tg.attrs["euler_characteristic"] = int(rex.euler_characteristic)
            except Exception:
                pass

        if "chain_valid" in names or "topology" in names:
            try:
                tg.attrs["chain_valid"] = bool(rex.chain_valid)
            except Exception:
                pass

        if "edge_types" in names or "topology" in names:
            try:
                self._store(tg, "edge_types", rex.edge_types)
            except Exception:
                pass

        if "cycle_basis" in names or "topology" in names:
            try:
                cycles = rex.cycle_basis
                cbg = self._get_or_create(tg, "cycle_basis")
                cbg.attrs["n_cycles"] = len(cycles)
                for i, cyc in enumerate(cycles):
                    self._store(cbg, f"c{i}", np.asarray(cyc, dtype=np.int32))
            except Exception:
                pass

        if "harmonic_space" in names or "topology" in names:
            try:
                store_fn(tg, "harmonic_space", rex.harmonic_space)
            except Exception:
                pass

        if "nF_hodge" in names or "topology" in names:
            try:
                tg.attrs["nF_hodge"] = int(rex.nF_hodge)
            except Exception:
                pass

        if "self_loop_face_indices" in names or "topology" in names:
            try:
                indices = rex.self_loop_face_indices
                tg.attrs["self_loop_face_indices"] = json.dumps(
                    [int(i) for i in indices]
                )
            except Exception:
                pass

        if "B2_hodge" in names or "topology" in names:
            try:
                store_fn(tg, "B2_hodge", rex.B2_hodge)
            except Exception:
                pass

    def _write_hodge_cache(self, g, rex, names) -> None:
        hodge_props = {"hodge_decomposition", "hodge_rho"}
        if not (names & (hodge_props | {"hodge"})):
            return

        if "hodge_decomposition" in names or "hodge" in names:
            try:
                w = rex.w_E if rex.w_E is not None else np.ones(rex.nE)
                grad, curl, harm = rex.hodge(w)
                hg = self._get_or_create(g, "hodge")
                self._store(hg, "gradient", grad)
                self._store(hg, "curl", curl)
                self._store(hg, "harmonic", harm)
                total = np.dot(w, w)
                if total > 0:
                    hg.attrs["pct_gradient"] = float(np.dot(grad, grad) / total)
                    hg.attrs["pct_curl"] = float(np.dot(curl, curl) / total)
                    hg.attrs["pct_harmonic"] = float(np.dot(harm, harm) / total)

                # Full Hodge analysis with rho
                try:
                    analysis = rex.hodge_full(w)
                    if isinstance(analysis, dict) and "rho" in analysis:
                        self._store(hg, "rho", analysis["rho"])
                    elif hasattr(analysis, "rho"):
                        self._store(hg, "rho", analysis.rho)
                except Exception:
                    pass
            except Exception:
                pass

        if "hodge_rho" in names and not self._has(g, "hodge"):
            try:
                w = rex.w_E if rex.w_E is not None else np.ones(rex.nE)
                analysis = rex.hodge_full(w)
                hg = self._get_or_create(g, "hodge")
                if isinstance(analysis, dict) and "rho" in analysis:
                    self._store(hg, "rho", analysis["rho"])
                elif hasattr(analysis, "rho"):
                    self._store(hg, "rho", analysis.rho)
            except Exception:
                pass

    def _write_faces_cache(self, g, rex, names) -> None:
        if not (names & {"detected_faces", "face_metrics", "faces"}):
            return
        try:
            fg = self._get_or_create(g, "faces")
            if "detected_faces" in names or "faces" in names:
                try:
                    fdata = rex.face_data()
                    if hasattr(fdata, "faces"):
                        fg.attrs["face_data"] = json.dumps(
                            fdata.faces, default=json_default
                        )
                    if hasattr(fdata, "metrics"):
                        g_store_dict(fg, "metrics", fdata.metrics,
                                     compressor=self.compressor,
                                     chunks=self.chunks)
                except Exception:
                    pass
        except Exception:
            pass

    def _write_field_cache(self, g, rex, names, store_fn) -> None:
        field_props = {"field_operator", "field_eigen", "mode_classification"}
        if not (names & (field_props | {"field"})):
            return
        fg = self._get_or_create(g, "field")

        if "field_operator" in names or "field" in names:
            try:
                M, g_val, is_psd = rex.field_operator
                store_fn(fg, "M", M)
                fg.attrs["coupling_g"] = float(g_val)
                fg.attrs["is_psd"] = bool(is_psd)
            except Exception:
                pass

        if "field_eigen" in names or "field" in names:
            try:
                evals, evecs, freqs = rex.field_eigen
                self._store(fg, "field_evals", evals)
                store_fn(fg, "field_evecs", evecs)
                self._store(fg, "field_freqs", freqs)
            except Exception:
                pass

        if "mode_classification" in names or "field" in names:
            try:
                modes = rex.classify_modes()
                g_store_dict(fg, "modes", modes,
                             compressor=self.compressor, chunks=self.chunks)
            except Exception:
                pass

    def _write_signal_cache(self, g, rex, names) -> None:
        """Write signal/perturbation results via _serialization."""
        signal_props = {"perturbation_result", "field_perturbation_result"}
        if not (names & (signal_props | {"signal"})):
            return
        # Signal results are written by the caller via write_typed or
        # write_namedtuple since they require runtime arguments
        # (edge_idx, times, etc.). The cache group is created here
        # as a placeholder; actual data is written by analysis code.
        self._get_or_create(g, "signal")

    def _write_quotient_cache(self, g, rex, names) -> None:
        quotient_props = {"subcomplex_masks", "quotient_result"}
        if not (names & (quotient_props | {"quotient"})):
            return
        # Quotient results require a subcomplex specification.
        # Create the group; actual data is written by analysis code.
        self._get_or_create(g, "quotient")

    def _write_persistence_cache(self, g, rex, names) -> None:
        persist_props = {"persistence_diagram", "persistence_enrichment"}
        if not (names & (persist_props | {"persistence"})):
            return
        # Persistence requires a filtration. Create group placeholder.
        self._get_or_create(g, "persistence")

    def _write_standard_metrics_cache(self, g, rex, names) -> None:
        if not (names & {"standard_metrics"}):
            return
        try:
            from ._serialization import ZarrAdapter, write_namedtuple
            # StandardMetrics is computed from the adjacency structure
            # Try to get it if it has been cached on the rex
            metrics = None
            if hasattr(rex, "_standard_metrics_cache"):
                metrics = rex._standard_metrics_cache
            if metrics is None:
                # Compute standard metrics on demand
                try:
                    from ..core import _standard
                    src, tgt = rex._ensure_src_tgt()
                    metrics = _standard.build_standard_metrics(
                        rex.nV, rex.nE, src, tgt
                    )
                except Exception:
                    pass
            if metrics is not None:
                adapter = ZarrAdapter(g, compressor=self.compressor,
                                      chunks=self.chunks)
                write_namedtuple(adapter, "standard_metrics", metrics)
        except Exception:
            pass

    # Cache reading

    def read_cache(self, path: str) -> dict:
        """Read cached properties without full RexGraph reconstruction.

        Returns a dict of property name to value.
        """
        root = open_root_group(ensure_zarr_suffix(path), mode="r")
        return self._read_cache_groups(root)

    def _read_cache_groups(self, g) -> dict:
        """Read all cache groups from a Zarr group."""
        result: dict = {}

        # algebra
        if self._has(g, "algebra"):
            ag = g["algebra"]
            for name in ("B1", "B2", "L0", "L1", "L2",
                         "L1_down", "L1_up",
                         "overlap_adjacency", "L_overlap"):
                if self._has(ag, name):
                    result[name] = self._load(ag, name)

        # spectral
        if self._has(g, "spectral"):
            sg = g["spectral"]

            # Full bundle
            if self._has(sg, "bundle"):
                result["spectral_bundle"] = g_load_dict(sg, "bundle")

            for name in ("eigenvalues_L0", "fiedler_vector_L0",
                         "layout", "layout_3d",
                         "evals_L1", "evecs_L1", "evals_L2",
                         "evals_L_O", "evecs_L_O",
                         "diag_L1_down", "diag_L1_up"):
                if self._has(sg, name):
                    result[name] = self._load(sg, name)

            if self._has(sg, "fiedler_overlap"):
                fog = sg["fiedler_overlap"]
                result["fiedler_overlap"] = (
                    float(fog.attrs["value"]),
                    self._load(fog, "vector"),
                )

        # relational
        if self._has(g, "relational"):
            rg = g["relational"]
            for name in ("RL_1", "evals_RL1", "evecs_RL1",
                         "L1_alpha", "evals_L1a", "evecs_L1a",
                         "Lambda", "evals_Lambda", "evecs_Lambda"):
                if self._has(rg, name):
                    result[name] = self._load(rg, name)
            for attr in ("alpha_G", "alpha_T", "alpha_used"):
                if attr in rg.attrs:
                    result[attr] = float(rg.attrs[attr])

        # topology
        if self._has(g, "topology"):
            tg = g["topology"]
            if "betti" in tg.attrs:
                raw = as_str(tg.attrs["betti"])
                result["betti"] = tuple(json.loads(raw))
            if "euler_characteristic" in tg.attrs:
                result["euler_characteristic"] = int(tg.attrs["euler_characteristic"])
            if "chain_valid" in tg.attrs:
                result["chain_valid"] = bool(tg.attrs["chain_valid"])
            if "nF_hodge" in tg.attrs:
                result["nF_hodge"] = int(tg.attrs["nF_hodge"])
            if "self_loop_face_indices" in tg.attrs:
                raw = as_str(tg.attrs["self_loop_face_indices"])
                result["self_loop_face_indices"] = json.loads(raw)
            if self._has(tg, "edge_types"):
                result["edge_types"] = self._load(tg, "edge_types")
            if self._has(tg, "cycle_basis"):
                cbg = tg["cycle_basis"]
                n = int(cbg.attrs.get("n_cycles", 0))
                result["cycle_basis"] = [
                    self._load(cbg, f"c{i}") for i in range(n)
                ]
            if self._has(tg, "harmonic_space"):
                result["harmonic_space"] = self._load(tg, "harmonic_space")
            if self._has(tg, "B2_hodge"):
                result["B2_hodge"] = self._load(tg, "B2_hodge")

        # hodge
        if self._has(g, "hodge"):
            hg = g["hodge"]
            result["hodge"] = {}
            for name in ("gradient", "curl", "harmonic", "rho"):
                if self._has(hg, name):
                    result["hodge"][name] = self._load(hg, name)
            for attr in ("pct_gradient", "pct_curl", "pct_harmonic"):
                if attr in hg.attrs:
                    result["hodge"][attr] = float(hg.attrs[attr])

        # faces
        if self._has(g, "faces"):
            fg = g["faces"]
            raw = fg.attrs.get("face_data")
            if raw:
                result["detected_faces"] = json.loads(as_str(raw))
            if self._has(fg, "metrics"):
                result["face_metrics"] = g_load_dict(fg, "metrics")

        # field
        if self._has(g, "field"):
            fg = g["field"]
            if self._has(fg, "M"):
                result["field_M"] = self._load(fg, "M")
            if "coupling_g" in fg.attrs:
                result["field_coupling_g"] = float(fg.attrs["coupling_g"])
            if "is_psd" in fg.attrs:
                result["field_is_psd"] = bool(fg.attrs["is_psd"])
            for name in ("field_evals", "field_evecs", "field_freqs"):
                if self._has(fg, name):
                    result[name] = self._load(fg, name)
            if self._has(fg, "modes"):
                result["mode_classification"] = g_load_dict(fg, "modes")

        # temporal
        if self._has(g, "temporal"):
            tg = g["temporal"]
            for name in ("first_seen", "last_seen", "duration",
                         "edge_counts", "edge_born", "edge_died",
                         "betti_matrix"):
                if self._has(tg, name):
                    result[name] = self._load(tg, name)
            if self._has(tg, "face_lifecycle"):
                flg = tg["face_lifecycle"]
                n = int(flg.attrs.get("n_arrays", 0))
                result["face_lifecycle"] = tuple(
                    self._load(flg, f"arr_{i}") for i in range(n)
                )
            if self._has(tg, "bioes_result"):
                try:
                    from ._serialization import ZarrAdapter, read_namedtuple
                    adapter = ZarrAdapter(tg, compressor=self.compressor,
                                          chunks=self.chunks)
                    result["bioes_result"] = read_namedtuple(
                        adapter, "bioes_result"
                    )
                except Exception:
                    pass

        # standard_metrics
        if self._has(g, "standard_metrics"):
            try:
                from ._serialization import ZarrAdapter, read_namedtuple
                adapter = ZarrAdapter(g, compressor=self.compressor,
                                      chunks=self.chunks)
                result["standard_metrics"] = read_namedtuple(
                    adapter, "standard_metrics"
                )
            except Exception:
                pass

        return result

    # NamedTuple serialization

    def write_typed(self, path: str, obj: Any) -> None:
        """Write a types.py NamedTuple to a .zarr store."""
        from ._serialization import ZarrAdapter, write_namedtuple

        path = ensure_zarr_suffix(path)
        if os.path.exists(path):
            rm_rf(path)
        root = create_root_group(path)
        root.attrs["format_version"] = _FORMAT_VERSION
        root.attrs["object_type"] = type(obj).__name__

        adapter = ZarrAdapter(root, compressor=self.compressor,
                              chunks=self.chunks)
        write_namedtuple(adapter, "data", obj)

    def read_typed(self, path: str) -> Any:
        """Read a types.py NamedTuple from a .zarr store."""
        from ._serialization import ZarrAdapter, read_namedtuple

        root = open_root_group(ensure_zarr_suffix(path), mode="r")
        type_name = as_str(root.attrs.get("object_type"))
        adapter = ZarrAdapter(root, compressor=self.compressor,
                              chunks=self.chunks)

        # Resolve type class
        from ._serialization import _resolve_type
        type_class = _resolve_type(type_name)

        return read_namedtuple(adapter, "data", type_class)

    # Write/read signal and quotient results directly

    def write_signal_result(self, path: str, result, *, group_name: str = "signal") -> None:
        """Write a PerturbationResult or FieldPerturbationResult into a store."""
        from ._serialization import ZarrAdapter, write_namedtuple

        path = ensure_zarr_suffix(path)
        root = open_root_group(path, mode="a")
        sg = self._get_or_create(root, group_name)
        adapter = ZarrAdapter(sg, compressor=self.compressor,
                              chunks=self.chunks)
        write_namedtuple(adapter, type(result).__name__, result)

    def read_signal_result(self, path: str, type_name: str,
                           *, group_name: str = "signal") -> Any:
        """Read a signal result from a store."""
        from ._serialization import ZarrAdapter, read_namedtuple, _resolve_type

        root = open_root_group(ensure_zarr_suffix(path), mode="r")
        sg = root[group_name]
        adapter = ZarrAdapter(sg, compressor=self.compressor,
                              chunks=self.chunks)
        return read_namedtuple(adapter, type_name, _resolve_type(type_name))

    def write_persistence_result(self, path: str, diagram, enrichment=None) -> None:
        """Write persistence diagram and optional enrichment."""
        from ._serialization import ZarrAdapter, write_namedtuple

        path = ensure_zarr_suffix(path)
        root = open_root_group(path, mode="a")
        pg = self._get_or_create(root, "persistence")
        adapter = ZarrAdapter(pg, compressor=self.compressor,
                              chunks=self.chunks)
        write_namedtuple(adapter, "diagram", diagram)
        if enrichment is not None:
            write_namedtuple(adapter, "enrichment", enrichment)

    def read_persistence_result(self, path: str) -> dict:
        """Read persistence diagram and enrichment if present."""
        from ._serialization import ZarrAdapter, read_namedtuple

        root = open_root_group(ensure_zarr_suffix(path), mode="r")
        pg = root["persistence"]
        adapter = ZarrAdapter(pg, compressor=self.compressor,
                              chunks=self.chunks)
        result = {}
        if adapter.has("diagram"):
            result["diagram"] = read_namedtuple(adapter, "diagram")
        if adapter.has("enrichment"):
            result["enrichment"] = read_namedtuple(adapter, "enrichment")
        return result

    def write_quotient_result(self, path: str, masks, quotient_result) -> None:
        """Write subcomplex masks and quotient result."""
        from ._serialization import ZarrAdapter, write_namedtuple

        path = ensure_zarr_suffix(path)
        root = open_root_group(path, mode="a")
        qg = self._get_or_create(root, "quotient")

        g_store_bool_masks(qg, "masks", {
            "v_mask": masks.v_mask if hasattr(masks, "v_mask") else masks[0],
            "e_mask": masks.e_mask if hasattr(masks, "e_mask") else masks[1],
            "f_mask": masks.f_mask if hasattr(masks, "f_mask") else masks[2],
        }, compressor=self.compressor, chunks=self.chunks)

        adapter = ZarrAdapter(qg, compressor=self.compressor,
                              chunks=self.chunks)
        write_namedtuple(adapter, "result", quotient_result)

    def read_quotient_result(self, path: str) -> dict:
        """Read subcomplex masks and quotient result."""
        from ._serialization import ZarrAdapter, read_namedtuple

        root = open_root_group(ensure_zarr_suffix(path), mode="r")
        qg = root["quotient"]
        result = {}
        if self._has(qg, "masks"):
            result["masks"] = g_load_bool_masks(qg, "masks")
        adapter = ZarrAdapter(qg, compressor=self.compressor,
                              chunks=self.chunks)
        if adapter.has("result"):
            result["quotient"] = read_namedtuple(adapter, "result")
        return result


# Module-level convenience functions

_default_fmt: Optional[RexZarrFormat] = None


def _get_fmt() -> RexZarrFormat:
    global _default_fmt
    if _default_fmt is None:
        _default_fmt = RexZarrFormat()
    return _default_fmt


def save_zarr(
    path: str,
    obj: Any,
    *,
    cache: Union[None, str, List[str]] = None,
    compressor: Any = "default",
) -> None:
    """Save a RexGraph, TemporalRex, or array to Zarr format.

    Parameters
    ----------
    path : str
        Output path (.zarr appended automatically).
    obj : RexGraph, TemporalRex, or ndarray
    cache : None, "all", or list of str
        Precomputed properties to include.
    compressor
        Override default compressor.
    """
    if compressor != "default":
        fmt = RexZarrFormat(compressor=compressor)
    else:
        fmt = _get_fmt()
    fmt.write(path, obj, cache=cache)


def load_zarr(path: str) -> Any:
    """Load a RexGraph, TemporalRex, or array from Zarr format."""
    return _get_fmt().read(path)
