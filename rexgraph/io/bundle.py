# rexgraph/io/bundle.py
"""
RexGraph Bundle (.rex) - portable relational complex package.

A bundle is a self-contained directory that stores a RexGraph or
TemporalRex with all data needed for exact reconstruction, plus
optional precomputed results.  It uses only NumPy `.npy` files
and JSON - no Zarr, HDF5, or heavy dependencies required.

On-disk layout::

    my_graph.rex/
    ├── MANIFEST.json           # version, object_type, metadata
    ├── boundary_ptr.npy        # int32 (nE+1,)
    ├── boundary_idx.npy        # int32 (nnz,)
    ├── B2_col_ptr.npy          # int32 (nF+1,)
    ├── B2_row_idx.npy          # int32 (nnz,)
    ├── B2_vals.npy             # float64 (nnz,)
    ├── w_E.npy                 # float64 (nE,) [if weighted]
    └── cache/                  # optional precomputed results
        ├── layout.npy          # float64 (nV, 2)
        ├── eigenvalues_L0.npy  # float64 (nV,)
        ├── B1.npy              # float64 (nV, nE)
        └── ...

For a TemporalRex::

    temporal.rex/
    ├── MANIFEST.json
    ├── snapshots/
    │   ├── 0/
    │   │   ├── sources.npy
    │   │   └── targets.npy
    │   ├── 1/ ...
    └── face_snapshots/         # optional
        ├── 0/
        │   ├── B2_col_ptr.npy
        │   └── B2_row_idx.npy

Design principles:

1. Round-trip fidelity: `RexGraph.from_dict(to_dict())` is the
   contract. Every field that `from_dict` needs is stored as an
   individual `.npy` file, and every field is loaded back.
2. Memory-mappable: Individual `.npy` files can be `mmap`'d
   for lazy/partial reads of large graphs.
3. Zero heavy deps: Only `numpy` and `json`. No zarr, h5py,
   scipy, pandas, or pyarrow.
4. Cache groups: Same `algebra/spectral/topology/hodge/faces`
   structure as :mod:`zarr_format` and :mod:`hdf5_format`, so
   precomputed results transfer across formats.

Usage::

    from rexgraph.io import save_rex, load_rex

    save_rex("graph.rex", rex)
    rex2 = load_rex("graph.rex")
    assert rex2.betti == rex.betti

    # With precomputed cache
    save_rex("graph.rex", rex, cache=["topology", "spectral"])

    # Bundle API for inspection
    bundle = RexBundle.from_graph(rex, cache="all")
    bundle.save("graph.rex")
    bundle = RexBundle.load("graph.rex")
    print(bundle.manifest)
    print(bundle["boundary_ptr"].shape)
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RexBundle",
    "save_rex",
    "load_rex",
]

_FORMAT_VERSION = 1
_MAGIC = "rex-bundle"

# Cache groups - same definitions as zarr_format / hdf5_format.
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


# Helpers


def _ensure_rex(path: str) -> pathlib.Path:
    """Ensure path has `.rex` suffix and return as Path."""
    p = pathlib.Path(path)
    if p.suffix != ".rex":
        p = pathlib.Path(str(p) + ".rex")
    return p


def _resolve_cache(cache) -> Set[str]:
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


def _save_npy(directory: pathlib.Path, name: str, arr: NDArray) -> str:
    """Save an array as `<name>.npy` and return the relative path."""
    arr = np.asarray(arr)
    fname = f"{name}.npy"
    np.save(directory / fname, arr)
    return fname


def _load_npy(
    directory: pathlib.Path,
    name: str,
    *,
    mmap: bool = False,
) -> np.ndarray:
    """Load `<name>.npy` from *directory*."""
    fpath = directory / f"{name}.npy"
    if not fpath.exists():
        raise FileNotFoundError(f"Array not found: {fpath}")
    mode = "r" if mmap else None
    return np.load(fpath, mmap_mode=mode)


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


# RexBundle


class RexBundle:
    """Portable RexGraph bundle.

    Wraps a directory of `.npy` files and a `MANIFEST.json` that
    together encode a RexGraph or TemporalRex with optional
    precomputed results.

    Parameters
    ----------
    root : Path
        Bundle directory.
    manifest : dict
        Parsed MANIFEST.json content.
    """

    def __init__(self, root: pathlib.Path, manifest: dict):
        self._root = root
        self._manifest = manifest

    @property
    def manifest(self) -> dict:
        """The parsed MANIFEST.json."""
        return self._manifest

    @property
    def object_type(self) -> str:
        """`'RexGraph'` or `'TemporalRex'`."""
        return self._manifest.get("object_type", "unknown")

    @property
    def path(self) -> pathlib.Path:
        return self._root


    # Construction


    @classmethod
    def from_graph(
        cls,
        graph,
        *,
        cache: Union[None, str, List[str]] = None,
    ) -> "RexBundle":
        """Create an in-memory bundle spec from a RexGraph.

        Does not write to disk - call `.save()` to persist.  The
        returned bundle stores references to the graph's arrays (not
        copies) until `save()` is called.
        """
        from ..graph import RexGraph, TemporalRex

        if isinstance(graph, TemporalRex):
            manifest = _build_temporal_manifest(graph)
        elif isinstance(graph, RexGraph):
            manifest = _build_rex_manifest(graph, cache)
        else:
            raise TypeError(
                f"Expected RexGraph or TemporalRex, got {type(graph).__name__}"
            )

        # Store references for deferred write
        bundle = cls.__new__(cls)
        bundle._root = None  # not yet saved
        bundle._manifest = manifest
        bundle._source = graph
        bundle._cache_spec = cache
        return bundle

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        *,
        mmap: bool = False,
    ) -> "RexBundle":
        """Load a bundle from a `.rex` directory.

        Parameters
        ----------
        path : str or path-like
            Bundle directory.
        mmap : bool
            If `True`, memory-map arrays for lazy loading.
        """
        root = _ensure_rex(str(path))
        if not root.exists():
            raise FileNotFoundError(f"Bundle not found: {root}")

        mf_path = root / "MANIFEST.json"
        if not mf_path.exists():
            raise FileNotFoundError(f"No MANIFEST.json in {root}")

        manifest = json.loads(mf_path.read_text())
        if manifest.get("magic") != _MAGIC:
            raise ValueError(
                f"Not a rex bundle (magic={manifest.get('magic')!r})"
            )

        bundle = cls(root, manifest)
        bundle._mmap = mmap
        return bundle


    # Persistence


    def save(self, path: Union[str, os.PathLike]) -> None:
        """Write this bundle to a `.rex` directory.

        If the bundle was created via `from_graph()`, arrays are
        written from the source graph.  If it was loaded from disk,
        the existing directory is copied.
        """
        root = _ensure_rex(str(path))

        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        source = getattr(self, "_source", None)
        if source is not None:
            from ..graph import RexGraph, TemporalRex
            if isinstance(source, TemporalRex):
                _write_temporal_bundle(root, source)
            elif isinstance(source, RexGraph):
                cache = getattr(self, "_cache_spec", None)
                _write_rex_bundle(root, source, cache)
            else:
                raise TypeError(f"Unexpected source: {type(source)}")
        elif self._root is not None and self._root.exists():
            # Copy existing bundle
            if root != self._root:
                shutil.copytree(self._root, root, dirs_exist_ok=True)
        else:
            raise RuntimeError("Bundle has no source data and no existing path")

        self._root = root
        self._source = None  # drop reference after write


    # Reconstruction


    def to_graph(self) -> "RexGraph":
        """Reconstruct a RexGraph from this bundle.

        Raises
        ------
        TypeError
            If the bundle contains a TemporalRex.
        """
        if self.object_type != "RexGraph":
            raise TypeError(
                f"Bundle contains {self.object_type}, not RexGraph"
            )
        from ..graph import RexGraph
        return _read_rex_graph(self._root)

    def to_temporal(self) -> "TemporalRex":
        """Reconstruct a TemporalRex from this bundle."""
        if self.object_type != "TemporalRex":
            raise TypeError(
                f"Bundle contains {self.object_type}, not TemporalRex"
            )
        from ..graph import TemporalRex
        return _read_temporal_rex(self._root)

    def to_object(self):
        """Reconstruct the appropriate object (RexGraph or TemporalRex)."""
        if self.object_type == "TemporalRex":
            return self.to_temporal()
        return self.to_graph()


    # Array access


    def __getitem__(self, key: str) -> np.ndarray:
        """Load a single array by name."""
        if self._root is None:
            # In-memory bundle from from_graph()
            source = getattr(self, "_source", None)
            if source is not None and hasattr(source, key):
                val = getattr(source, key)
                if isinstance(val, np.ndarray):
                    return val
            raise KeyError(key)

        mmap = getattr(self, "_mmap", False)

        # Check root level
        npy = self._root / f"{key}.npy"
        if npy.exists():
            return np.load(npy, mmap_mode="r" if mmap else None)

        # Check cache/
        npy = self._root / "cache" / f"{key}.npy"
        if npy.exists():
            return np.load(npy, mmap_mode="r" if mmap else None)

        raise KeyError(f"Array '{key}' not found in bundle")

    def __contains__(self, key: str) -> bool:
        if self._root is None:
            return False
        return (
            (self._root / f"{key}.npy").exists()
            or (self._root / "cache" / f"{key}.npy").exists()
        )

    def list_arrays(self) -> List[str]:
        """List all available array names."""
        if self._root is None:
            return []
        names = []
        for f in self._root.glob("*.npy"):
            names.append(f.stem)
        cache_dir = self._root / "cache"
        if cache_dir.exists():
            for f in cache_dir.glob("*.npy"):
                names.append(f.stem)
        return sorted(names)

    def read_cache(self) -> dict:
        """Read all cached properties as a dict."""
        cache_dir = self._root / "cache"
        if not cache_dir.exists():
            return {}
        result = {}
        for f in cache_dir.glob("*.npy"):
            result[f.stem] = np.load(f)
        # Scalar cache from manifest
        result.update(self._manifest.get("cache_scalars", {}))
        return result

    def __repr__(self) -> str:
        obj = self.object_type
        arrays = self.list_arrays()
        n = len(arrays)
        preview = ", ".join(arrays[:5])
        if n > 5:
            preview += f", ... (+{n - 5})"
        path = self._root or "<unsaved>"
        return f"RexBundle({obj}, {n} arrays: [{preview}], path={path})"


# Internal: manifest builders


def _build_rex_manifest(rex, cache) -> dict:
    """Build MANIFEST.json content for a RexGraph."""
    return {
        "magic": _MAGIC,
        "version": _FORMAT_VERSION,
        "object_type": "RexGraph",
        "nV": int(rex.nV),
        "nE": int(rex.nE),
        "nF": int(rex.nF),
        "directed": bool(rex._directed),
        "dimension": int(rex.dimension),
        "weighted": rex._w_E is not None,
        "has_faces": rex.nF > 0,
        "cache_requested": cache if isinstance(cache, (list, type(None))) else str(cache),
    }


def _build_temporal_manifest(trex) -> dict:
    """Build MANIFEST.json content for a TemporalRex."""
    return {
        "magic": _MAGIC,
        "version": _FORMAT_VERSION,
        "object_type": "TemporalRex",
        "T": trex.T,
        "directed": bool(trex._directed),
        "general": bool(trex._general),
        "has_face_snapshots": bool(trex._face_snapshots),
    }


# Internal: RexGraph write/read


def _write_rex_bundle(root: pathlib.Path, rex, cache) -> None:
    """Write a RexGraph to a .rex directory."""
    manifest = _build_rex_manifest(rex, cache)

    # -- Core arrays (the from_dict contract) --
    core_arrays = []
    _save_npy(root, "boundary_ptr", rex._boundary_ptr)
    core_arrays.append("boundary_ptr")
    _save_npy(root, "boundary_idx", rex._boundary_idx)
    core_arrays.append("boundary_idx")
    _save_npy(root, "B2_col_ptr", rex._B2_col_ptr)
    core_arrays.append("B2_col_ptr")
    _save_npy(root, "B2_row_idx", rex._B2_row_idx)
    core_arrays.append("B2_row_idx")
    _save_npy(root, "B2_vals", rex._B2_vals)
    core_arrays.append("B2_vals")

    if rex._w_E is not None:
        _save_npy(root, "w_E", rex._w_E)
        core_arrays.append("w_E")

    # General boundary weights
    if rex._w_boundary:
        manifest["w_boundary"] = {
            str(k): v for k, v in rex._w_boundary.items()
        }

    # Vertex attribution
    if hasattr(rex, "_attribution") and rex._attribution is not None:
        _save_npy(root, "attribution", rex._attribution)
        core_arrays.append("attribution")

    manifest["core_arrays"] = core_arrays

    # -- Cache --
    names = _resolve_cache(cache)
    if names:
        cache_dir = root / "cache"
        cache_dir.mkdir()
        written_cache, scalar_cache = _write_cache(cache_dir, rex, names)
        manifest["cached_arrays"] = written_cache
        if scalar_cache:
            manifest["cache_scalars"] = scalar_cache

    # -- Write manifest --
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default)
    )


def _read_rex_graph(root: pathlib.Path) -> "RexGraph":
    """Reconstruct a RexGraph from a .rex directory."""
    from ..graph import RexGraph

    manifest = json.loads((root / "MANIFEST.json").read_text())

    kw: dict = {
        "boundary_ptr": _load_npy(root, "boundary_ptr"),
        "boundary_idx": _load_npy(root, "boundary_idx"),
        "directed": manifest.get("directed", False),
    }

    for name in ("B2_col_ptr", "B2_row_idx", "B2_vals"):
        npy = root / f"{name}.npy"
        if npy.exists():
            kw[name] = np.load(npy)

    npy_wE = root / "w_E.npy"
    if npy_wE.exists():
        kw["w_E"] = np.load(npy_wE)

    wb = manifest.get("w_boundary")
    if wb:
        kw["w_boundary"] = {int(k): v for k, v in wb.items()}

    rex = RexGraph(**kw)

    npy_attr = root / "attribution.npy"
    if npy_attr.exists():
        rex.set_vertex_attribution(np.load(npy_attr))

    return rex


# Internal: TemporalRex write/read


def _write_temporal_bundle(root: pathlib.Path, trex) -> None:
    """Write a TemporalRex to a .rex directory."""
    manifest = _build_temporal_manifest(trex)

    snap_dir = root / "snapshots"
    snap_dir.mkdir()

    for t in range(trex.T):
        tdir = snap_dir / str(t)
        tdir.mkdir()
        snap = trex._snapshots[t]
        if trex._general:
            _save_npy(tdir, "boundary_ptr", snap[0])
            _save_npy(tdir, "boundary_idx", snap[1])
        else:
            _save_npy(tdir, "sources", snap[0])
            _save_npy(tdir, "targets", snap[1])

    if trex._face_snapshots:
        fdir = root / "face_snapshots"
        fdir.mkdir()
        for t, fsnap in enumerate(trex._face_snapshots):
            ftdir = fdir / str(t)
            ftdir.mkdir()
            _save_npy(ftdir, "B2_col_ptr", fsnap[0])
            _save_npy(ftdir, "B2_row_idx", fsnap[1])

    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default)
    )


def _read_temporal_rex(root: pathlib.Path) -> "TemporalRex":
    """Reconstruct a TemporalRex from a .rex directory."""
    from ..graph import TemporalRex

    manifest = json.loads((root / "MANIFEST.json").read_text())
    T = manifest["T"]
    directed = manifest.get("directed", False)
    general = manifest.get("general", False)

    snap_dir = root / "snapshots"
    snapshots = []
    for t in range(T):
        tdir = snap_dir / str(t)
        if general:
            snapshots.append((
                _load_npy(tdir, "boundary_ptr"),
                _load_npy(tdir, "boundary_idx"),
            ))
        else:
            snapshots.append((
                _load_npy(tdir, "sources"),
                _load_npy(tdir, "targets"),
            ))

    face_snapshots = []
    fdir = root / "face_snapshots"
    if fdir.exists():
        for t in range(len(list(fdir.iterdir()))):
            ftdir = fdir / str(t)
            if not ftdir.exists():
                break
            face_snapshots.append((
                _load_npy(ftdir, "B2_col_ptr"),
                _load_npy(ftdir, "B2_row_idx"),
            ))

    return TemporalRex(
        snapshots,
        face_snapshots=face_snapshots or None,
        directed=directed,
        general=general,
    )


# Internal: cache writer


def _write_cache(
    cache_dir: pathlib.Path,
    rex,
    names: Set[str],
) -> Tuple[List[str], dict]:
    """Write precomputed properties to cache/ directory.

    Returns (list of array names written, dict of scalar values).
    """
    written: List[str] = []
    scalars: dict = {}

    def _try_array(prop_name: str, cache_name: Optional[str] = None):
        """Attempt to get a property and save it."""
        cn = cache_name or prop_name
        try:
            arr = getattr(rex, prop_name)
            if isinstance(arr, np.ndarray):
                _save_npy(cache_dir, cn, arr)
                written.append(cn)
        except Exception:
            pass

    # --- algebra ---
    if names & {"algebra", "B1"}:
        _try_array("B1")
    if names & {"algebra", "B2"}:
        _try_array("B2")
    if names & {"algebra", "L0"}:
        _try_array("L0")
    if names & {"algebra", "L1"}:
        _try_array("L1")
    if names & {"algebra", "L2"}:
        _try_array("L2")
    if names & {"algebra", "overlap_adjacency"}:
        _try_array("overlap_adjacency")
    if names & {"algebra", "L_overlap"}:
        _try_array("L_overlap")

    # --- spectral ---
    if names & {"spectral", "eigenvalues_L0"}:
        _try_array("eigenvalues_L0")
    if names & {"spectral", "fiedler_vector_L0"}:
        _try_array("fiedler_vector_L0")
    if names & {"spectral", "layout"}:
        _try_array("layout")
    if names & {"spectral", "layout_3d"}:
        _try_array("layout_3d")
    if names & {"spectral", "fiedler_overlap_value", "fiedler_overlap_vector"}:
        try:
            val, vec = rex.fiedler_overlap
            scalars["fiedler_overlap_value"] = float(val)
            _save_npy(cache_dir, "fiedler_overlap_vector", vec)
            written.append("fiedler_overlap_vector")
        except Exception:
            pass

    # --- topology ---
    if names & {"topology", "betti"}:
        try:
            scalars["betti"] = list(rex.betti)
        except Exception:
            pass
    if names & {"topology", "euler_characteristic"}:
        try:
            scalars["euler_characteristic"] = int(rex.euler_characteristic)
        except Exception:
            pass
    if names & {"topology", "chain_valid"}:
        try:
            scalars["chain_valid"] = bool(rex.chain_valid)
        except Exception:
            pass
    if names & {"topology", "edge_types"}:
        _try_array("edge_types")
    if names & {"topology", "harmonic_space"}:
        _try_array("harmonic_space")

    # --- hodge ---
    if names & {"hodge", "hodge_gradient", "hodge_curl", "hodge_harmonic"}:
        try:
            w = rex.w_E if rex.w_E is not None else np.ones(rex.nE)
            grad, curl, harm = rex.hodge(w)
            _save_npy(cache_dir, "hodge_gradient", grad)
            _save_npy(cache_dir, "hodge_curl", curl)
            _save_npy(cache_dir, "hodge_harmonic", harm)
            written.extend(["hodge_gradient", "hodge_curl", "hodge_harmonic"])
            total = np.dot(w, w)
            if total > 0:
                scalars["hodge_pct_gradient"] = float(np.dot(grad, grad) / total)
                scalars["hodge_pct_curl"] = float(np.dot(curl, curl) / total)
                scalars["hodge_pct_harmonic"] = float(np.dot(harm, harm) / total)
        except Exception:
            pass

    return written, scalars


# Module-level convenience functions


def save_rex(
    path: str,
    obj: Any,
    *,
    cache: Union[None, str, List[str]] = None,
) -> None:
    """Save a RexGraph or TemporalRex to a `.rex` bundle.

    Parameters
    ----------
    path : str
        Output directory (`.rex` suffix added if missing).
    obj : RexGraph or TemporalRex
        Object to save.
    cache : None, "all", or list of str
        Precomputed properties to include.

    Examples
    --------
    >>> save_rex("graph.rex", rex)
    >>> save_rex("graph.rex", rex, cache="all")
    >>> save_rex("graph.rex", rex, cache=["topology", "spectral"])
    """
    bundle = RexBundle.from_graph(obj, cache=cache)
    bundle.save(path)


def load_rex(path: str) -> Any:
    """Load a RexGraph or TemporalRex from a `.rex` bundle.

    Parameters
    ----------
    path : str
        Path to `.rex` directory.

    Returns
    -------
    RexGraph or TemporalRex

    Examples
    --------
    >>> rex = load_rex("graph.rex")
    >>> trex = load_rex("temporal.rex")
    """
    bundle = RexBundle.load(path)
    return bundle.to_object()
