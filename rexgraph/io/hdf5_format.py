# rexgraph/io/hdf5_format.py
"""
HDF5-based storage for the rex framework.

Single-file counterpart to zarr_format. Same serialization surface
(RexGraph, TemporalRex, cache groups, NamedTuples) stored in a
single .h5 file. On-disk layout mirrors the Zarr format.
"""

from __future__ import annotations

from rexgraph.io._cache_layout import (
    _ALL_CACHEABLE,
    _CACHE_GROUPS,
    CacheLayoutMixin,
)

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..graph import TemporalRex

import contextlib

from ._compat import (
    HAS_HDF5,
    as_str,
    dumps,
    h5_load_bool_masks,
    h5_load_complex,
    h5_load_dict,
    h5_load_sparse_csr,
    h5_store_bool_masks,
    h5_store_complex,
    h5_store_dict,
)

if HAS_HDF5:
    import h5py

__all__ = [
    "RexHDF5Format",
    "save_hdf5",
    "load_hdf5",
    "save_hdf5_array",
    "load_hdf5_array",
]

_FORMAT_VERSION = "2.0.0"

# Cache group definitions (same as zarr_format.py)


_ALL_CACHEABLE.update(_CACHE_GROUPS.keys())


# Path helper

def _ensure_h5(path: str) -> str:
    """Append .h5 if not already present."""
    return path if path.endswith(".h5") else path + ".h5"


# Simple array save/load

def save_hdf5_array(arr: NDArray, path: str) -> None:
    """Save a NumPy array into an .h5 file."""
    with h5py.File(_ensure_h5(path), "w") as f:
        f.attrs["object_type"] = "array"
        f.attrs["format_version"] = _FORMAT_VERSION
        h5_store_complex(f, "data", np.asarray(arr))


def load_hdf5_array(path: str) -> np.ndarray:
    """Load a NumPy array from an .h5 file."""
    with h5py.File(_ensure_h5(path), "r") as f:
        return h5_load_complex(f, "data")


# RexHDF5Format

class RexHDF5Format(CacheLayoutMixin):
    """HDF5-based on-disk format for the rex framework.

    Parameters
    ----------
    compression : str or None
        HDF5 compression filter: "lzf" (fast, default), "gzip"
        (portable), or None.
    compression_opts : int or None
        Compression level for gzip (1-9).
    chunks : bool
        Enable dataset chunking (required for compression).
    large_threshold : int
        Edge count above which explicit chunk sizes are used.
        Default: 50000.
    """

    extension = ".h5"

    def __init__(
        self,
        compression: str | None = "lzf",
        compression_opts: int | None = None,
        chunks: bool = True,
        large_threshold: int = 50_000,
    ):
        if not HAS_HDF5:
            raise ImportError("h5py is required: pip install h5py")
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunks = chunks
        self.large_threshold = large_threshold

    # Internal helpers

    def _comp_kw(self) -> dict:
        """Compression kwargs for dataset creation."""
        kw: dict = {}
        if self.compression:
            kw["compression"] = self.compression
        if self.compression_opts is not None:
            kw["compression_opts"] = self.compression_opts
        if self.chunks:
            kw["chunks"] = True
        return kw

    def _store(self, group, name: str, arr: NDArray) -> None:
        """Store a dense or complex array."""
        h5_store_complex(group, name, np.asarray(arr),
                         compression=self.compression or "",
                         chunks=self.chunks)

    def _store_chunked(
        self, group, name: str, arr: NDArray, chunk_rows: int = 10_000
    ) -> None:
        """Store a large array with explicit row-based chunking."""
        arr = np.asarray(arr)
        kw: dict = {}
        if self.compression:
            kw["compression"] = self.compression
        if self.compression_opts is not None:
            kw["compression_opts"] = self.compression_opts

        if arr.ndim == 2:
            cr = min(chunk_rows, arr.shape[0])
            kw["chunks"] = (cr, arr.shape[1])
        elif arr.ndim == 1:
            cr = min(chunk_rows, arr.shape[0])
            kw["chunks"] = (cr,)
        elif self.chunks:
            kw["chunks"] = True

        if np.iscomplexobj(arr):
            sub = group.create_group(name)
            sub.attrs["is_complex"] = True
            sub.attrs["dtype"] = str(arr.dtype)
            sub.attrs["shape"] = list(arr.shape)
            sub.create_dataset("real", data=arr.real, **kw)
            sub.create_dataset("imag", data=arr.imag, **kw)
        else:
            group.create_dataset(name, data=arr, **kw)

    def _load(self, group, name: str) -> np.ndarray:
        """Load an array, handling complex and sparse subgroups."""
        obj = group[name]
        if isinstance(obj, h5py.Group):
            if obj.attrs.get("is_complex", False):
                return h5_load_complex(group, name)
            if obj.attrs.get("is_sparse", False):
                return h5_load_sparse_csr(group, name, dense=True)
        return obj[:]

    def _has(self, group, name: str) -> bool:
        return name in group


    def _get_or_create(self, group, name: str):
        """Get existing subgroup or create a new one."""
        if name in group:
            return group[name]
        return group.create_group(name)

    # Public API

    def write(
        self,
        path: str,
        obj: Any,
        *,
        cache: None | str | list[str] = None,
    ) -> None:
        """Write a RexGraph, TemporalRex, or ndarray to an .h5 file.

        Parameters
        ----------
        path : str
            Output path (.h5 suffix added if missing).
        obj : RexGraph, TemporalRex, or ndarray
        cache : None, "all", or list of str
            Precomputed results to include.
        """
        from ..graph import RexGraph, TemporalRex

        path = _ensure_h5(path)
        with h5py.File(path, "w", libver="latest") as f:
            f.attrs["format_version"] = _FORMAT_VERSION

            if isinstance(obj, TemporalRex):
                f.attrs["object_type"] = "TemporalRex"
                self._write_temporal_rex(f, obj, cache=cache)
            elif isinstance(obj, RexGraph):
                f.attrs["object_type"] = "RexGraph"
                self._write_rex_graph(f, obj, cache=cache)
            elif isinstance(obj, np.ndarray):
                f.attrs["object_type"] = "array"
                self._store(f, "data", obj)
            else:
                raise TypeError(f"Unsupported type: {type(obj).__name__}")

    def read(self, path: str) -> Any:
        """Read a RexGraph, TemporalRex, or ndarray from an .h5 file."""

        path = _ensure_h5(path)
        with h5py.File(path, "r") as f:
            obj_type = as_str(f.attrs.get("object_type"))

            if obj_type == "RexGraph":
                return self._read_rex_graph(f)
            if obj_type == "TemporalRex":
                return self._read_temporal_rex(f)
            if obj_type == "array":
                return self._load(f, "data")
            raise TypeError(f"Unknown object_type: {obj_type}")

    # Container API

    def write_to_group(self, path: str, name: str, obj: Any, **kw) -> None:
        """Write an object into /objects/<n> inside path."""
        from ..graph import RexGraph, TemporalRex

        path = _ensure_h5(path)
        mode = "a" if os.path.exists(path) else "w"
        with h5py.File(path, mode, libver="latest") as f:
            if "format_version" not in f.attrs:
                f.attrs["format_version"] = _FORMAT_VERSION
                f.attrs["object_type"] = "Container"

            objs = f.require_group("objects")
            if name in objs:
                del objs[name]

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
        """Read /objects/<n> from path."""

        path = _ensure_h5(path)
        with h5py.File(path, "r") as f:
            g = f["objects"][name]
            t = as_str(g.attrs.get("object_type"))
            if t == "RexGraph":
                return self._read_rex_graph(g)
            if t == "TemporalRex":
                return self._read_temporal_rex(g)
            if t == "array":
                return self._load(g, "data")
            raise TypeError(f"Unknown object_type in group '{name}': {t}")

    def list_groups(self, path: str) -> list[str]:
        """List sub-object names in a container file."""
        path = _ensure_h5(path)
        if not os.path.exists(path):
            return []
        with h5py.File(path, "r") as f:
            if "objects" not in f:
                return []
            return list(f["objects"].keys())

    # RexGraph serialization



    # TemporalRex serialization


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
                from ._serialization import HDF5Adapter, write_namedtuple
                bioes = trex.bioes_joint()
                adapter = HDF5Adapter(tg, compression=self.compression or "",
                                      chunks=self.chunks)
                write_namedtuple(adapter, "bioes_result", bioes)
            except Exception:
                pass

        if "betti_matrix" in names or "temporal" in names:
            try:
                T_len = trex.T
                betti_rows = []
                for t in range(T_len):
                    snap = trex.at(t)
                    betti_rows.append(list(snap.betti))
                self._store(tg, "betti_matrix",
                            np.array(betti_rows, dtype=np.int64))
            except Exception:
                pass

    def _read_temporal_rex(self, g) -> TemporalRex:
        """Reconstruct a TemporalRex from an HDF5 group."""
        from ..graph import TemporalRex

        T = int(g.attrs["T"])
        directed = bool(g.attrs.get("directed", False))
        general = bool(g.attrs.get("general", False))

        sg = g["snapshots"]
        snapshots = []
        relation_ids = []
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
            relation_ids.append(
                self._load(tg, "relation_ids") if self._has(tg, "relation_ids") else None
            )

        face_snapshots = []
        if self._has(g, "face_snapshots"):
            fg = g["face_snapshots"]
            keys = sorted(fg.keys(), key=int)
            for k in keys:
                ftg = fg[k]
                face_snapshots.append((
                    self._load(ftg, "B2_col_ptr"),
                    self._load(ftg, "B2_row_idx"),
                ))

        return TemporalRex(
            snapshots,
            face_snapshots=face_snapshots or None,
            relation_ids=relation_ids,
            directed=directed,
            general=general,
        )

    # Cache resolution


    # Cache writing



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

        if "spectral_bundle" in names or "spectral" in names:
            try:
                sb = rex.spectral_bundle
                h5_store_dict(sg, "bundle", sb,
                              compression=self.compression or "",
                              chunks=self.chunks)
            except Exception:
                pass

        for prop in ("eigenvalues_L0", "fiedler_vector_L0",
                      "layout", "layout_3d"):
            if prop in names or "spectral" in names:
                with contextlib.suppress(Exception):
                    self._store(sg, prop, getattr(rex, prop))

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

        if "fiedler_overlap" in names or "spectral" in names:
            try:
                val, vec = rex.fiedler_overlap
                fog = self._get_or_create(sg, "fiedler_overlap")
                fog.attrs["value"] = float(val)
                self._store(fog, "vector", vec)
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
                        fg.attrs["face_data"] = dumps(fdata.faces)
                    if hasattr(fdata, "metrics"):
                        h5_store_dict(fg, "metrics", fdata.metrics,
                                      compression=self.compression or "",
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
                h5_store_dict(fg, "modes", modes,
                              compression=self.compression or "",
                              chunks=self.chunks)
            except Exception:
                pass




    def _write_standard_metrics_cache(self, g, rex, names) -> None:
        if not (names & {"standard_metrics"}):
            return
        try:
            from ._serialization import HDF5Adapter, write_namedtuple
            metrics = None
            if hasattr(rex, "_standard_metrics_cache"):
                metrics = rex._standard_metrics_cache
            if metrics is None:
                try:
                    from ..core import _standard

                    # the adjacency bundle, matching graph.py's own call. The previous
                    # form passed (nV, nE, src, tgt), which the kernel rejects: it takes
                    # at least six arguments and the first four are the CSR. So it
                    # raised TypeError on every call, the except below swallowed it, and
                    # cache="standard_metrics" wrote nothing while reporting success.
                    adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
                    e_wt = (np.asarray(rex.w_E, dtype=np.float64)
                            if rex.w_E is not None
                            else np.ones(rex.nE, dtype=np.float64))
                    adj_wt = _standard.build_adj_weights(adj_edge, e_wt)
                    # the kernel hands back a DICT and write_namedtuple wants a
                    # namedtuple, so writing the raw return produced a group holding
                    # nothing but a type marker. StandardMetrics' fields are exactly the
                    # dict's keys, so the wrap is total rather than a selection.
                    from ..rextypes import StandardMetrics
                    metrics = StandardMetrics(**_standard.build_standard_metrics(
                        adj_ptr, adj_idx, adj_edge, adj_wt, rex.nV, rex.nE
                    ))
                except Exception:
                    pass
            if metrics is not None:
                adapter = HDF5Adapter(g, compression=self.compression or "",
                                      chunks=self.chunks)
                write_namedtuple(adapter, "standard_metrics", metrics)
        except Exception:
            pass

    # Cache reading

    def read_cache(self, path: str) -> dict:
        """Read cached properties without full RexGraph reconstruction."""
        path = _ensure_h5(path)
        with h5py.File(path, "r") as f:
            return self._read_cache_groups(f)

    def _read_cache_groups(self, g) -> dict:
        """Read all cache groups from an HDF5 group."""
        result: dict = {}

        # algebra
        if self._has(g, "algebra"):
            ag = g["algebra"]
            for name in ("B1", "B2", "L0", "L1", "L2",
                         "L1_down", "L1_up",
                         "L_overlap"):
                if self._has(ag, name):
                    result[name] = self._load(ag, name)

        # spectral
        if self._has(g, "spectral"):
            sg = g["spectral"]
            if self._has(sg, "bundle"):
                result["spectral_bundle"] = h5_load_dict(sg, "bundle")
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
                result["betti"] = tuple(json.loads(as_str(tg.attrs["betti"])))
            if "euler_characteristic" in tg.attrs:
                result["euler_characteristic"] = int(tg.attrs["euler_characteristic"])
            if "chain_valid" in tg.attrs:
                result["chain_valid"] = bool(tg.attrs["chain_valid"])
            if "nF_hodge" in tg.attrs:
                result["nF_hodge"] = int(tg.attrs["nF_hodge"])
            if "self_loop_face_indices" in tg.attrs:
                result["self_loop_face_indices"] = json.loads(
                    as_str(tg.attrs["self_loop_face_indices"])
                )
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
                result["face_metrics"] = h5_load_dict(fg, "metrics")

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
                result["mode_classification"] = h5_load_dict(fg, "modes")

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
                    from ._serialization import HDF5Adapter, read_namedtuple
                    adapter = HDF5Adapter(tg, compression=self.compression or "",
                                          chunks=self.chunks)
                    result["bioes_result"] = read_namedtuple(
                        adapter, "bioes_result"
                    )
                except Exception:
                    pass

        # standard_metrics
        if self._has(g, "standard_metrics"):
            try:
                from ._serialization import HDF5Adapter, read_namedtuple
                adapter = HDF5Adapter(g, compression=self.compression or "",
                                      chunks=self.chunks)
                result["standard_metrics"] = read_namedtuple(
                    adapter, "standard_metrics"
                )
            except Exception:
                pass

        return result

    # NamedTuple serialization

    def write_typed(self, path: str, obj: Any) -> None:
        """Write a types.py NamedTuple to an .h5 file."""
        from ._serialization import HDF5Adapter, write_namedtuple

        path = _ensure_h5(path)
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = _FORMAT_VERSION
            f.attrs["object_type"] = type(obj).__name__

            adapter = HDF5Adapter(f, compression=self.compression or "",
                                  chunks=self.chunks)
            write_namedtuple(adapter, "data", obj)

    def read_typed(self, path: str) -> Any:
        """Read a types.py NamedTuple from an .h5 file."""
        from ._serialization import HDF5Adapter, _resolve_type, read_namedtuple

        path = _ensure_h5(path)
        with h5py.File(path, "r") as f:
            type_name = as_str(f.attrs.get("object_type"))
            adapter = HDF5Adapter(f, compression=self.compression or "",
                                  chunks=self.chunks)
            type_class = _resolve_type(type_name)
            return read_namedtuple(adapter, "data", type_class)

    # Direct result write/read methods

    def write_signal_result(self, path: str, result, *, group_name: str = "signal") -> None:
        """Write a PerturbationResult or FieldPerturbationResult."""
        from ._serialization import HDF5Adapter, write_namedtuple

        path = _ensure_h5(path)
        with h5py.File(path, "a") as f:
            sg = self._get_or_create(f, group_name)
            adapter = HDF5Adapter(sg, compression=self.compression or "",
                                  chunks=self.chunks)
            write_namedtuple(adapter, type(result).__name__, result)

    def read_signal_result(self, path: str, type_name: str,
                           *, group_name: str = "signal") -> Any:
        """Read a signal result from an .h5 file."""
        from ._serialization import HDF5Adapter, _resolve_type, read_namedtuple

        path = _ensure_h5(path)
        with h5py.File(path, "r") as f:
            sg = f[group_name]
            adapter = HDF5Adapter(sg, compression=self.compression or "",
                                  chunks=self.chunks)
            return read_namedtuple(adapter, type_name, _resolve_type(type_name))

    def write_persistence_result(self, path: str, diagram, enrichment=None) -> None:
        """Write persistence diagram and optional enrichment."""
        from ._serialization import HDF5Adapter, write_namedtuple

        path = _ensure_h5(path)
        with h5py.File(path, "a") as f:
            pg = self._get_or_create(f, "persistence")
            adapter = HDF5Adapter(pg, compression=self.compression or "",
                                  chunks=self.chunks)
            write_namedtuple(adapter, "diagram", diagram)
            if enrichment is not None:
                write_namedtuple(adapter, "enrichment", enrichment)

    def read_persistence_result(self, path: str) -> dict:
        """Read persistence diagram and enrichment if present."""
        from ._serialization import HDF5Adapter, read_namedtuple

        path = _ensure_h5(path)
        with h5py.File(path, "r") as f:
            pg = f["persistence"]
            adapter = HDF5Adapter(pg, compression=self.compression or "",
                                  chunks=self.chunks)
            result = {}
            if adapter.has("diagram"):
                result["diagram"] = read_namedtuple(adapter, "diagram")
            if adapter.has("enrichment"):
                result["enrichment"] = read_namedtuple(adapter, "enrichment")
            return result

    def write_quotient_result(self, path: str, masks, quotient_result) -> None:
        """Write subcomplex masks and quotient result."""
        from ._serialization import HDF5Adapter, write_namedtuple

        path = _ensure_h5(path)
        with h5py.File(path, "a") as f:
            qg = self._get_or_create(f, "quotient")

            h5_store_bool_masks(qg, "masks", {
                "v_mask": masks.v_mask if hasattr(masks, "v_mask") else masks[0],
                "e_mask": masks.e_mask if hasattr(masks, "e_mask") else masks[1],
                "f_mask": masks.f_mask if hasattr(masks, "f_mask") else masks[2],
            }, compression=self.compression or "", chunks=self.chunks)

            adapter = HDF5Adapter(qg, compression=self.compression or "",
                                  chunks=self.chunks)
            write_namedtuple(adapter, "result", quotient_result)

    def read_quotient_result(self, path: str) -> dict:
        """Read subcomplex masks and quotient result."""
        from ._serialization import HDF5Adapter, read_namedtuple

        path = _ensure_h5(path)
        with h5py.File(path, "r") as f:
            qg = f["quotient"]
            result = {}
            if self._has(qg, "masks"):
                result["masks"] = h5_load_bool_masks(qg, "masks")
            adapter = HDF5Adapter(qg, compression=self.compression or "",
                                  chunks=self.chunks)
            if adapter.has("result"):
                result["quotient"] = read_namedtuple(adapter, "result")
            return result


# Module-level convenience functions

_default_fmt: RexHDF5Format | None = None


def _get_fmt() -> RexHDF5Format:
    global _default_fmt
    if _default_fmt is None:
        _default_fmt = RexHDF5Format()
    return _default_fmt


def save_hdf5(
    path: str,
    obj: Any,
    *,
    cache: None | str | list[str] = None,
    compression: str | None = "lzf",
) -> None:
    """Save a RexGraph, TemporalRex, or array to HDF5 format.

    Parameters
    ----------
    path : str
        Output path (.h5 appended automatically).
    obj : RexGraph, TemporalRex, or ndarray
    cache : None, "all", or list of str
        Precomputed properties to include.
    compression : str or None
        Override default compression filter.
    """
    fmt = RexHDF5Format(compression=compression) if compression != "lzf" else _get_fmt()
    fmt.write(path, obj, cache=cache)


def load_hdf5(path: str) -> Any:
    """Load a RexGraph, TemporalRex, or array from HDF5 format."""
    return _get_fmt().read(path)
