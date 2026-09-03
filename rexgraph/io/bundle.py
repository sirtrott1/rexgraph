# rexgraph/io/bundle.py
"""
RexGraph Bundle (.rex) - portable relational complex package.

A bundle is a self-contained directory that stores a RexGraph or
TemporalRex with all data needed for exact reconstruction, plus
optional precomputed results.  It uses only NumPy `.npy` files
and JSON: no Zarr, HDF5, or heavy dependencies required.

On-disk layout:

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

For a TemporalRex:

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
2. Indexed access: Plain `.npy` members can be `mmap`'d. Authenticated
   encrypted members use chunk-selective reads instead of pretending
   ciphertext is memory-mappable typed data.
3. Zero heavy deps: Only `numpy` and `json`. No zarr, h5py,
   scipy, pandas, or pyarrow.
4. Cache groups: Same `algebra/spectral/topology/hodge/faces`
   structure as :mod:`zarr_format` and :mod:`hdf5_format`, so
   precomputed results transfer across formats.

Usage:

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

import contextlib
import hashlib
import json
import os
import pathlib
import secrets
import shutil
import tempfile
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

try:
    import fcntl
except ImportError:  # pragma: no cover, non-POSIX publication is thread-safe only
    fcntl = None

if TYPE_CHECKING:
    from ..graph import RexGraph, TemporalRex

from ._container_crypto import (
    ContainerDecryptionProperties,
    ContainerEncryptionError,
    ContainerEncryptionProperties,
    _predicate_mask,
    _predicate_value,
    _protect_tensor_members,
    _query_protected_indices,
    _read_protected_indices,
    encrypted_metadata,
    open_encrypted_manifest,
    read_protected_tensor,
    validate_storage_inventory,
)
from .rex_state import fname_encode as _fname_encode

__all__ = [
    "RexBundle",
    "save_rex",
    "load_rex",
]

_FORMAT_VERSION = 1
_MAGIC = "rex-bundle"
_ENCRYPTED_STORAGE = "__rex_encrypted_storage__"
_PUBLISH_LOCK = threading.Lock()
_PUBLISH_LOCK_PID = os.getpid()

# Cache groups - same definitions as zarr_format / hdf5_format.
_CACHE_GROUPS: dict[str, list[str]] = {
    "algebra": [
        "B1", "B2", "L0", "L1", "L2",
        "L_overlap",
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
    "harmonic": [
        "harmonic_basis", "harmonic_dim",
        "frustration_per_edge", "coparticipation_per_edge",
        "sigma_asymmetry_per_edge",
    ],
}

_ALL_CACHEABLE: set[str] = set()
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


def _resolve_cache(cache) -> set[str]:
    """Expand cache spec into individual property names."""
    if cache is None:
        return set()
    if isinstance(cache, str):
        if cache == "all":
            return set(_ALL_CACHEABLE)
        if cache in _CACHE_GROUPS:
            return set(_CACHE_GROUPS[cache])
        return {cache}
    out: set[str] = set()
    for c in cache:
        if c == "all":
            return set(_ALL_CACHEABLE)
        if c in _CACHE_GROUPS:
            out.update(_CACHE_GROUPS[c])
        else:
            out.add(c)
    return out


def _channel_diagonals(rex):
    """The four channel diagonals per edge, trace-normalized, shape (nE, 4).

    Exact from the boundary structure when the complex carries a rational
    character. The normalized G channel takes a square root and does not, so
    that case reads the assembled hats instead, which is the only reason the
    fallback exists.
    """
    from rexgraph.rational_trig import exact_channel_diagonals

    nE = int(rex.nE)
    chi = np.zeros((nE, 4), dtype=np.float64)
    diagonals, names = exact_channel_diagonals(rex)
    if diagonals is not None:
        for ci, name in enumerate(names[:4]):
            col = np.array([float(x) for x in diagonals[name]], dtype=np.float64)
            total = col.sum()
            chi[:, ci] = col / total if total else col
        return chi
    bundle = rex._hat_eigen_bundle
    for ci in range(min(4, len(bundle))):
        ev, evec = bundle[ci]
        col = np.einsum("ij,j,ij->i", evec, ev, evec)
        total = col.sum()
        chi[:, ci] = col / total if total else col
    return chi


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


#: the one encoder (rexgraph.io._compat). Re-exported under the local name so the
#: existing call sites keep working; `dumps` is what applies the non-finite policy.
from ._compat import dumps as _dumps


@contextlib.contextmanager
def _bundle_publish_lock(parent: pathlib.Path):
    """Serialize the short directory replacement across threads/processes."""
    global _PUBLISH_LOCK, _PUBLISH_LOCK_PID
    pid = os.getpid()
    if pid != _PUBLISH_LOCK_PID:
        # A child must not inherit a lock another thread held across fork.
        _PUBLISH_LOCK = threading.Lock()
        _PUBLISH_LOCK_PID = pid
    with _PUBLISH_LOCK:
        if fcntl is None:
            yield
            return
        fd = os.open(parent, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)


def _bundle_staging_directory(root: pathlib.Path) -> pathlib.Path:
    """Create a same-parent staging directory with normal mkdir/umask semantics."""
    for _ in range(100):
        candidate = root.with_name(
            f".{root.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        )
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not allocate a staging directory beside {root}")


def _encrypted_member_path(
    root: pathlib.Path,
    member: dict[str, Any],
) -> pathlib.Path:
    suffix = ".rexenc" if member["protected"] else ".npy"
    return root / f"{member['storage_name']}{suffix}"


class _BundleStorageView:
    """Safetensors-like uint8 view over one directory member."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def get_dtype(self) -> str:
        return "U8"

    def get_shape(self) -> list[int]:
        return [int(self._array.size)]

    def __getitem__(self, index):
        return self._array[index]


class _BundleStorage:
    """Adapt authenticated directory members to the common chunk reader."""

    def __init__(self, root: pathlib.Path, manifest: dict[str, Any]):
        self.root = root
        self.manifest = manifest
        self._members = {
            member["storage_name"]: member for member in manifest["members"]
        }
        self._logical_members = {
            member["logical_name"]: member for member in manifest["members"]
        }

    def keys(self) -> list[str]:
        return list(self._members)

    def logical_member(self, logical_name: str) -> dict[str, Any] | None:
        return self._logical_members.get(logical_name)

    def logical_array(
        self,
        member: dict[str, Any],
        *,
        mmap: bool = True,
    ) -> np.ndarray:
        if member["protected"]:
            raise ContainerEncryptionError("protected member is not a native npy array")
        path = _encrypted_member_path(self.root, member)
        try:
            array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise ContainerEncryptionError(
                f"plaintext storage for {member['logical_name']!r} is malformed"
            ) from exc
        if array.dtype.str != member["dtype"] or list(array.shape) != member["shape"]:
            raise ContainerEncryptionError(
                f"plaintext storage spec for {member['logical_name']!r} "
                "differs from the authenticated manifest"
            )
        if not array.flags.c_contiguous:
            raise ContainerEncryptionError("plaintext bundle member is not C-contiguous")
        return array

    def get_slice(self, storage_name: str) -> _BundleStorageView:
        member = self._members[storage_name]
        path = _encrypted_member_path(self.root, member)
        if member["protected"]:
            try:
                array = np.memmap(path, dtype=np.uint8, mode="r")
            except (OSError, ValueError) as exc:
                raise ContainerEncryptionError(
                    f"protected storage for {member['logical_name']!r} is malformed"
                ) from exc
        else:
            logical = self.logical_array(member)
            array = logical.view(np.uint8).reshape(-1)
        return _BundleStorageView(array)


def _bundle_storage_digest(view: _BundleStorageView) -> str:
    digest = hashlib.sha256()
    size = view.get_shape()[0]
    block = 8 << 20
    for start in range(0, size, block):
        digest.update(np.asarray(view[start:min(start + block, size)]).tobytes())
    return digest.hexdigest()


def _validate_encrypted_bundle_files(
    root: pathlib.Path,
    manifest: dict[str, Any],
    storage: _BundleStorage,
) -> None:
    expected_files = {"MANIFEST.json"}
    for member in manifest["members"]:
        expected_files.add(
            _encrypted_member_path(root, member).relative_to(root).as_posix()
        )
    expected_directories = {_ENCRYPTED_STORAGE} if manifest["members"] else set()

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ContainerEncryptionError(
                f"encrypted bundle inventory contains symlink {relative!r}"
            )
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files.add(relative)
        else:
            raise ContainerEncryptionError(
                f"encrypted bundle inventory contains unsupported entry {relative!r}"
            )

    if actual_files != expected_files or actual_directories != expected_directories:
        raise ContainerEncryptionError(
            "encrypted bundle file inventory differs from authenticated manifest; "
            f"missing={sorted(expected_files - actual_files)!r}, "
            f"extra={sorted(actual_files - expected_files)!r}, "
            f"unexpected_directories={sorted(actual_directories - expected_directories)!r}"
        )

    validate_storage_inventory(storage, manifest)
    for member in manifest["members"]:
        expected_digest = member.get("storage_sha256")
        if not isinstance(expected_digest, str):
            raise ContainerEncryptionError(
                f"storage digest is missing for {member['logical_name']!r}"
            )
        actual = _bundle_storage_digest(storage.get_slice(member["storage_name"]))
        if actual != expected_digest:
            raise ContainerEncryptionError(
                f"storage digest failed for {member['logical_name']!r}"
            )


def _write_encrypted_bundle(
    root: pathlib.Path,
    tensors: dict[str, NDArray],
    metadata: dict[str, Any],
    encryption_properties: ContainerEncryptionProperties,
    *,
    kind: str,
) -> dict[str, Any]:
    storage, outer_metadata, manifest = _protect_tensor_members(
        tensors,
        metadata,
        encryption_properties,
        kind=kind,
    )
    if manifest["members"]:
        (root / _ENCRYPTED_STORAGE).mkdir()
    by_name = {member["logical_name"]: member for member in manifest["members"]}
    for logical_name, member in by_name.items():
        path = _encrypted_member_path(root, member)
        if member["protected"]:
            path.write_bytes(np.asarray(storage[member["storage_name"]]).tobytes())
        else:
            np.save(path, np.ascontiguousarray(np.asarray(tensors[logical_name])))

    public_manifest = {
        "encrypted": True,
        "magic": _MAGIC,
        "version": _FORMAT_VERSION,
        **outer_metadata,
    }
    # Publication happens only after this last file exists in the staging directory.
    (root / "MANIFEST.json").write_text(_dumps(public_manifest))
    storage_reader = _BundleStorage(root, manifest)
    _validate_encrypted_bundle_files(root, manifest, storage_reader)
    return manifest


def _open_bundle_manifest(
    root: pathlib.Path,
    decryption_properties: ContainerDecryptionProperties | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, _BundleStorage | None]:
    public = json.loads((root / "MANIFEST.json").read_text())
    if public.get("magic") != _MAGIC:
        raise ValueError(f"Not a rex bundle (magic={public.get('magic')!r})")

    marked_encrypted = public.get("encrypted") is True or public.get("rex_encrypted") == "1"
    is_encrypted = encrypted_metadata(public)
    if marked_encrypted and not is_encrypted:
        raise ContainerEncryptionError("encrypted bundle descriptor is missing")
    if not is_encrypted:
        return public, None, None

    manifest = open_encrypted_manifest(public, decryption_properties)
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict) or metadata.get("magic") != _MAGIC:
        raise ContainerEncryptionError("authenticated bundle metadata is invalid")
    if metadata.get("object_type") != manifest["kind"]:
        raise ContainerEncryptionError(
            "authenticated bundle object type differs from its container kind"
        )
    storage = _BundleStorage(root, manifest)
    _validate_encrypted_bundle_files(root, manifest, storage)
    return metadata, manifest, storage

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
        self._mmap = False
        self._encrypted_manifest = None
        self._encrypted_storage = None
        self._decryption_properties = None
        self._query_chunk_cache: dict[tuple[str, str, int], bytes] = {}
        self._query_statistics_cache: dict[str, list[dict[str, Any]]] = {}

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
        cache: None | str | list[str] = None,
    ) -> RexBundle:
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
        bundle._mmap = False
        bundle._encrypted_manifest = None
        bundle._encrypted_storage = None
        bundle._decryption_properties = None
        bundle._query_chunk_cache = {}
        bundle._query_statistics_cache = {}
        return bundle

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        mmap: bool = False,
        decryption_properties: ContainerDecryptionProperties | None = None,
    ) -> RexBundle:
        """Load a bundle from a `.rex` directory.

        Parameters
        ----------
        path : str or path-like
            Bundle directory.
        mmap : bool
            If `True`, memory-map arrays for lazy loading.
        decryption_properties : opaque property, optional
            Caller-owned authenticated opener for an encrypted bundle. Core
            receives no key bytes and imports no KMS implementation.
        """
        root = _ensure_rex(str(path))
        if not root.exists():
            raise FileNotFoundError(f"Bundle not found: {root}")

        mf_path = root / "MANIFEST.json"
        if not mf_path.exists():
            raise FileNotFoundError(f"No MANIFEST.json in {root}")

        manifest, encrypted_manifest, storage = _open_bundle_manifest(
            root,
            decryption_properties,
        )

        bundle = cls(root, manifest)
        bundle._mmap = mmap
        bundle._encrypted_manifest = encrypted_manifest
        bundle._encrypted_storage = storage
        bundle._decryption_properties = decryption_properties
        return bundle


    # Persistence


    def save(
        self,
        path: str | os.PathLike,
        *,
        encryption_properties: ContainerEncryptionProperties | None = None,
    ) -> None:
        """Write this bundle to a `.rex` directory.

        If the bundle was created via `from_graph()`, arrays are
        written from the source graph.  If it was loaded from disk,
        the existing directory is copied. An opaque encryption property writes
        an authenticated manifest and indexed encrypted members.
        """
        root = _ensure_rex(str(path))
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = _bundle_staging_directory(root)
        encrypted_state: tuple[dict[str, Any], dict[str, Any]] | None = None

        try:
            source = getattr(self, "_source", None)
            if source is not None:
                from ..graph import RexGraph, TemporalRex
                if isinstance(source, TemporalRex):
                    encrypted_state = _write_temporal_bundle(
                        staging,
                        source,
                        encryption_properties=encryption_properties,
                    )
                elif isinstance(source, RexGraph):
                    cache = getattr(self, "_cache_spec", None)
                    encrypted_state = _write_rex_bundle(
                        staging,
                        source,
                        cache,
                        encryption_properties=encryption_properties,
                    )
                else:
                    raise TypeError(f"Unexpected source: {type(source)}")
            elif self._root is not None and self._root.exists():
                if encryption_properties is not None:
                    raise ValueError(
                        "cannot change encryption while copying a loaded bundle; "
                        "reconstruct the object and save it with the new property"
                    )
                shutil.copytree(self._root, staging, dirs_exist_ok=True)
                if self._encrypted_manifest is not None:
                    _open_bundle_manifest(staging, self._decryption_properties)
            else:
                raise RuntimeError("Bundle has no source data and no existing path")

            with _bundle_publish_lock(root.parent):
                if root.is_symlink() or (root.exists() and not root.is_dir()):
                    raise FileExistsError(f"Bundle destination is not a directory: {root}")
                backup = root.with_name(
                    f".{root.name}.old-{os.getpid()}-{secrets.token_hex(8)}"
                )
                had_existing = root.exists()
                if had_existing:
                    os.replace(root, backup)
                try:
                    os.replace(staging, root)
                except Exception:
                    if had_existing and backup.exists() and not root.exists():
                        os.replace(backup, root)
                    raise
                if had_existing:
                    shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        if encrypted_state is not None:
            self._manifest, self._encrypted_manifest = encrypted_state
            self._encrypted_storage = _BundleStorage(root, self._encrypted_manifest)
            self._decryption_properties = encryption_properties
        elif getattr(self, "_encrypted_manifest", None) is not None:
            self._encrypted_storage = _BundleStorage(root, self._encrypted_manifest)
        else:
            self._encrypted_manifest = None
            self._encrypted_storage = None
            self._decryption_properties = None

        self._root = root
        self._source = None  # drop reference after write


    # Reconstruction


    def to_graph(self, *, allow_unsealed: bool = False) -> RexGraph:
        """Reconstruct a RexGraph from this bundle.

        ``allow_unsealed=True`` is an explicit migration path for trusted bundles
        written before content digests existed. The default refuses them because their
        stored tensors cannot be checked for modification or truncation.

        Raises
        ------
        TypeError
            If the bundle contains a TemporalRex.
        """
        if self.object_type != "RexGraph":
            raise TypeError(
                f"Bundle contains {self.object_type}, not RexGraph"
            )
        if self._encrypted_manifest is not None:
            from .rex_state import RexState, from_state

            tensors = {
                name: self._read_encrypted_tensor(name)
                for name in self._manifest.get("tensor_names", [])
            }
            # An authenticated v1 container is never treated as an unsealed legacy
            # state, even when a caller enables the plaintext migration flag.
            return from_state(RexState(tensors, self._manifest), _allow_unsealed=False)
        return _read_rex_graph(self._root, allow_unsealed=allow_unsealed)

    def to_temporal(self) -> TemporalRex:
        """Reconstruct a TemporalRex from this bundle."""
        if self.object_type != "TemporalRex":
            raise TypeError(
                f"Bundle contains {self.object_type}, not TemporalRex"
            )
        if self._encrypted_manifest is not None:
            return _read_temporal_rex(
                self._root,
                tensor_reader=self._read_encrypted_tensor,
                manifest=self._manifest,
            )
        return _read_temporal_rex(self._root)

    def to_object(self, *, allow_unsealed: bool = False):
        """Reconstruct the appropriate object (RexGraph or TemporalRex)."""
        if self.object_type == "TemporalRex":
            return self.to_temporal()
        return self.to_graph(allow_unsealed=allow_unsealed)


    # Array access

    def _encrypted_member(self, key: str) -> dict[str, Any]:
        for candidate in (key, f"cache/{key}"):
            member = self._encrypted_storage.logical_member(candidate)
            if member is not None:
                return member
        raise KeyError(f"Array {key!r} not found in bundle")

    def _read_encrypted_tensor(
        self,
        key: str,
        index: int | slice | None = None,
    ) -> np.ndarray:
        member = self._encrypted_member(key)
        return read_protected_tensor(
            self._encrypted_storage,
            self._encrypted_manifest,
            member["logical_name"],
            self._decryption_properties,
            index=index,
        )

    def read_slice(
        self,
        key: str,
        index: int | slice | None = None,
    ) -> np.ndarray:
        """Read an array or first-axis slice without opening unrelated members."""
        if self._encrypted_manifest is None:
            array = self[key]
            return np.asarray(array if index is None else array[index])
        member = self._encrypted_member(key)
        if not member["protected"]:
            array = self._encrypted_storage.logical_array(
                member,
                mmap=getattr(self, "_mmap", False),
            )
            return np.asarray(array if index is None else array[index])
        return self._read_encrypted_tensor(member["logical_name"], index)

    def where(
        self,
        key: str,
        operator: str,
        value: Any = None,
    ) -> NDArray[np.int64]:
        """Return first-axis positions satisfying a scalar member predicate."""
        if self._encrypted_manifest is not None:
            member = self._encrypted_member(key)
            return _query_protected_indices(
                self._encrypted_storage,
                self._encrypted_manifest,
                member["logical_name"],
                operator,
                value,
                self._decryption_properties,
                chunk_cache=self._query_chunk_cache,
                statistics_cache=self._query_statistics_cache,
            )
        values = np.asarray(self[key])
        if values.ndim != 1:
            raise ValueError("query predicates require a one-dimensional tensor")
        converted = _predicate_value(values.dtype, operator, value)
        return np.flatnonzero(
            _predicate_mask(values, operator, converted)
        ).astype(np.int64, copy=False)

    def select(
        self,
        keys: str | Sequence[str],
        *,
        where: tuple[str, str, Any],
    ) -> dict[str, np.ndarray]:
        """Gather named members at rows matching ``(name, operator, value)``."""
        requested = [keys] if isinstance(keys, str) else list(keys)
        if any(not isinstance(key, str) or not key for key in requested):
            raise ValueError("query result names must be nonempty strings")
        if not isinstance(where, tuple) or len(where) != 3:
            raise TypeError("where must be a (name, operator, value) tuple")
        predicate_key, operator, value = where
        positions = self.where(predicate_key, operator, value)
        predicate = np.asarray(self[predicate_key]) if self._encrypted_manifest is None else None
        result: dict[str, np.ndarray] = {}
        for key in requested:
            if self._encrypted_manifest is not None:
                predicate_member = self._encrypted_member(predicate_key)
                member = self._encrypted_member(key)
                if (
                    not member["shape"]
                    or member["shape"][0] != predicate_member["shape"][0]
                ):
                    raise ValueError(
                        f"query result {key!r} does not share the predicate first axis"
                    )
                result[key] = _read_protected_indices(
                    self._encrypted_storage,
                    self._encrypted_manifest,
                    member["logical_name"],
                    positions,
                    self._decryption_properties,
                    chunk_cache=self._query_chunk_cache,
                )
            else:
                values = np.asarray(self[key])
                if values.ndim == 0 or values.shape[0] != predicate.shape[0]:
                    raise ValueError(
                        f"query result {key!r} does not share the predicate first axis"
                    )
                result[key] = values[positions]
        return result

    def clear_query_cache(self) -> None:
        """Forget decrypted chunks and statistics retained by bundle queries."""
        self._query_chunk_cache.clear()
        self._query_statistics_cache.clear()


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

        if self._encrypted_manifest is not None:
            member = self._encrypted_member(key)
            if not member["protected"]:
                return self._encrypted_storage.logical_array(member, mmap=mmap)
            if mmap:
                raise ValueError(
                    "encrypted bundle members cannot be memory-mapped; "
                    "use bundle.read_slice(name, index) to decrypt selected chunks"
                )
            return self._read_encrypted_tensor(member["logical_name"])

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
        if self._encrypted_manifest is not None:
            try:
                self._encrypted_member(key)
            except KeyError:
                return False
            return True
        return (
            (self._root / f"{key}.npy").exists()
            or (self._root / "cache" / f"{key}.npy").exists()
        )

    def list_arrays(self) -> list[str]:
        """List all available array names."""
        if self._root is None:
            return []
        if self._encrypted_manifest is not None:
            return sorted(
                member["logical_name"].removeprefix("cache/")
                for member in self._encrypted_manifest["members"]
            )
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
        if self._encrypted_manifest is not None:
            result = {
                name: self[name]
                for name in self._manifest.get("cached_arrays", [])
            }
            result.update(self._manifest.get("cache_scalars", {}))
            return result
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
    g_channels = list(getattr(trex, "_g_channels", ()))
    c_channels = list(getattr(trex, "_c_channels", ()))
    return {
        "magic": _MAGIC,
        "version": _FORMAT_VERSION,
        "object_type": "TemporalRex",
        "T": trex.T,
        "directed": bool(trex._directed),
        "general": bool(trex._general),
        "has_face_snapshots": bool(trex._face_snapshots),
        "times": [float(value) for value in trex._times],
        "g_channels": [
            str(g_channels[index]) if index < len(g_channels) else "raw"
            for index in range(trex.T)
        ],
        "c_channels": [
            str(c_channels[index]) if index < len(c_channels) else "share"
            for index in range(trex.T)
        ],
    }


# Internal: RexGraph write/read


def _write_rex_bundle(
    root: pathlib.Path,
    rex,
    cache,
    *,
    encryption_properties: ContainerEncryptionProperties | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Write a RexGraph to a .rex directory."""
    from .rex_state import to_state
    st = to_state(rex)
    names = list(st.tensors.keys())
    manifest = dict(st.header)
    manifest["magic"] = _MAGIC
    manifest["object_type"] = "RexGraph"
    manifest["tensor_names"] = names

    if encryption_properties is not None:
        tensors = {
            name: np.ascontiguousarray(np.asarray(value))
            for name, value in st.tensors.items()
        }
        if cache:
            requested = _resolve_cache(cache)
            if requested:
                with tempfile.TemporaryDirectory(
                    prefix=".rex-cache-",
                    dir=root,
                ) as cache_temp:
                    cache_root = pathlib.Path(cache_temp)
                    written_cache, scalar_cache = _write_cache(
                        cache_root,
                        rex,
                        requested,
                    )
                    for cache_name in written_cache:
                        tensors[f"cache/{cache_name}"] = np.load(
                            cache_root / f"{cache_name}.npy",
                            allow_pickle=False,
                        )
                manifest["cached_arrays"] = written_cache
                if scalar_cache:
                    manifest["cache_scalars"] = scalar_cache
        encrypted_manifest = _write_encrypted_bundle(
            root,
            tensors,
            manifest,
            encryption_properties,
            kind="RexGraph",
        )
        return manifest, encrypted_manifest

    # Filenames use a REVERSIBLE, collision-free percent-encoding of the tensor name (see
    # _fname_encode). A tensor name may contain '/' (nested rexes) or '__' (user metadata keys); the
    # old '/'->'__' substitution was neither filesystem-safe nor invertible and collided. Core arrays
    # (boundary_ptr, signs, ...) have no unsafe chars so their filenames stay their logical names,
    # keeping the RexBundle array-access API working.
    for name in names:
        _save_npy(root, _fname_encode(name), np.asarray(st.tensors[name]))
    (root / "MANIFEST.json").write_text(_dumps(manifest))
    if cache:
        names = _resolve_cache(cache)
        if names:
            cache_dir = root / "cache"
            cache_dir.mkdir()
            written_cache, scalar_cache = _write_cache(cache_dir, rex, names)
            manifest["cached_arrays"] = written_cache
            if scalar_cache:
                manifest["cache_scalars"] = scalar_cache
            (root / "MANIFEST.json").write_text(_dumps(manifest))
    return None


def _read_rex_graph(
    root: pathlib.Path,
    *,
    allow_unsealed: bool = False,
) -> RexGraph:
    """Reconstruct a RexGraph from a .rex directory."""
    from .rex_state import RexState, from_state
    manifest = json.loads((root / "MANIFEST.json").read_text())
    tensors = {}
    for name in manifest.get("tensor_names", []):
        tensors[name] = np.load(root / f"{_fname_encode(name)}.npy")
    return from_state(
        RexState(tensors, manifest),
        _allow_unsealed=allow_unsealed,
    )


# Internal: TemporalRex write/read


def _write_temporal_bundle(
    root: pathlib.Path,
    trex,
    *,
    encryption_properties: ContainerEncryptionProperties | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Write a TemporalRex to a .rex directory."""
    manifest = _build_temporal_manifest(trex)
    relation_id_snapshots = [
        value is not None for value in getattr(trex, "_snapshot_relation_ids", ())
    ]
    if relation_id_snapshots:
        manifest["relation_id_snapshots"] = relation_id_snapshots

    if encryption_properties is not None:
        manifest["face_snapshot_count"] = len(trex._face_snapshots)
        tensors: dict[str, NDArray] = {}
        for t in range(trex.T):
            snap = trex._snapshots[t]
            if trex._general:
                tensors[f"snapshots/{t}/boundary_ptr"] = snap[0]
                tensors[f"snapshots/{t}/boundary_idx"] = snap[1]
            else:
                tensors[f"snapshots/{t}/sources"] = snap[0]
                tensors[f"snapshots/{t}/targets"] = snap[1]
            relation_ids = trex._snapshot_relation_ids[t]
            if relation_ids is not None:
                tensors[f"snapshots/{t}/relation_ids"] = relation_ids
        for t, face_snapshot in enumerate(trex._face_snapshots):
            tensors[f"face_snapshots/{t}/B2_col_ptr"] = face_snapshot[0]
            tensors[f"face_snapshots/{t}/B2_row_idx"] = face_snapshot[1]
            if len(face_snapshot) > 2:
                tensors[f"face_snapshots/{t}/B2_vals"] = face_snapshot[2]
        manifest["face_snapshot_values"] = [
            len(face_snapshot) > 2 for face_snapshot in trex._face_snapshots
        ]
        encrypted_manifest = _write_encrypted_bundle(
            root,
            tensors,
            manifest,
            encryption_properties,
            kind="TemporalRex",
        )
        return manifest, encrypted_manifest

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
        relation_ids = trex._snapshot_relation_ids[t]
        if relation_ids is not None:
            _save_npy(tdir, "relation_ids", relation_ids)

    if trex._face_snapshots:
        fdir = root / "face_snapshots"
        fdir.mkdir()
        for t, fsnap in enumerate(trex._face_snapshots):
            ftdir = fdir / str(t)
            ftdir.mkdir()
            _save_npy(ftdir, "B2_col_ptr", fsnap[0])
            _save_npy(ftdir, "B2_row_idx", fsnap[1])

    (root / "MANIFEST.json").write_text(
        _dumps(manifest, indent=2)
    )
    return None


def _read_temporal_rex(
    root: pathlib.Path,
    *,
    tensor_reader=None,
    manifest: dict[str, Any] | None = None,
) -> TemporalRex:
    """Reconstruct a TemporalRex from a .rex directory."""
    from ..graph import TemporalRex

    if manifest is None:
        manifest = json.loads((root / "MANIFEST.json").read_text())
    T = manifest["T"]
    directed = manifest.get("directed", False)
    general = manifest.get("general", False)

    snap_dir = root / "snapshots"
    snapshots = []
    relation_ids = []
    raw_relation_ids = manifest.get("relation_id_snapshots", [False] * T)
    if (
        not isinstance(raw_relation_ids, list)
        or len(raw_relation_ids) != T
        or any(not isinstance(value, bool) for value in raw_relation_ids)
    ):
        raise ValueError("TemporalRex bundle relation_id_snapshots must be boolean per step")
    for t in range(T):
        tdir = snap_dir / str(t)
        if general:
            snapshots.append((
                tensor_reader(f"snapshots/{t}/boundary_ptr")
                if tensor_reader else _load_npy(tdir, "boundary_ptr"),
                tensor_reader(f"snapshots/{t}/boundary_idx")
                if tensor_reader else _load_npy(tdir, "boundary_idx"),
            ))
        else:
            snapshots.append((
                tensor_reader(f"snapshots/{t}/sources")
                if tensor_reader else _load_npy(tdir, "sources"),
                tensor_reader(f"snapshots/{t}/targets")
                if tensor_reader else _load_npy(tdir, "targets"),
            ))
        if raw_relation_ids[t]:
            relation_ids.append(
                tensor_reader(f"snapshots/{t}/relation_ids")
                if tensor_reader else _load_npy(tdir, "relation_ids")
            )
        else:
            relation_ids.append(None)

    face_snapshots = []
    fdir = root / "face_snapshots"
    if tensor_reader:
        face_count = int(manifest.get("face_snapshot_count", 0))
    else:
        face_count = len(list(fdir.iterdir())) if fdir.exists() else 0
    if face_count:
        face_values = manifest.get("face_snapshot_values", [])
        for t in range(face_count):
            ftdir = fdir / str(t)
            face_snapshot = (
                tensor_reader(f"face_snapshots/{t}/B2_col_ptr")
                if tensor_reader else _load_npy(ftdir, "B2_col_ptr"),
                tensor_reader(f"face_snapshots/{t}/B2_row_idx")
                if tensor_reader else _load_npy(ftdir, "B2_row_idx"),
            )
            if tensor_reader and t < len(face_values) and face_values[t]:
                face_snapshot = (*face_snapshot, tensor_reader(
                    f"face_snapshots/{t}/B2_vals"
                ))
            face_snapshots.append(face_snapshot)

    trex = TemporalRex(
        snapshots,
        face_snapshots=face_snapshots or None,
        relation_ids=relation_ids,
        directed=directed,
        general=general,
    )
    raw_times = manifest.get("times")
    if raw_times is not None:
        if not isinstance(raw_times, list) or len(raw_times) != T:
            raise ValueError("TemporalRex bundle times must contain one value per step")
        trex._times = [float(value) for value in raw_times]
    for field, default, allowed in (
        ("g_channels", "raw", {"raw", "normalized"}),
        ("c_channels", "share", {"share", "count"}),
    ):
        values = manifest.get(field)
        if values is None:
            values = [default] * T
        if (
            not isinstance(values, list)
            or len(values) != T
            or any(not isinstance(value, str) or value not in allowed for value in values)
        ):
            raise ValueError(f"TemporalRex bundle {field} are invalid")
        setattr(trex, f"_{field}", list(values))
    return trex


# Internal: cache writer


def _write_cache(
    cache_dir: pathlib.Path,
    rex,
    names: set[str],
) -> tuple[list[str], dict]:
    """Write precomputed properties to cache/ directory.

    Returns (list of array names written, dict of scalar values).
    """
    written: list[str] = []
    scalars: dict = {}

    def _try_array(prop_name: str, cache_name: str | None = None):
        """Attempt to get a property and save it."""
        cn = cache_name or prop_name
        try:
            arr = getattr(rex, prop_name)
            if isinstance(arr, np.ndarray):
                _save_npy(cache_dir, cn, arr)
                written.append(cn)
        except Exception:
            pass

    # algebra
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
    if names & {"algebra", "L_overlap"}:
        _try_array("L_overlap")

    # spectral
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

    # topology
    if names & {"topology", "betti"}:
        with contextlib.suppress(Exception):
            scalars["betti"] = list(rex.betti)
    if names & {"topology", "euler_characteristic"}:
        with contextlib.suppress(Exception):
            scalars["euler_characteristic"] = int(rex.euler_characteristic)
    if names & {"topology", "chain_valid"}:
        with contextlib.suppress(Exception):
            scalars["chain_valid"] = bool(rex.chain_valid)
    if names & {"topology", "edge_types"}:
        _try_array("edge_types")
    if names & {"topology", "harmonic_space"}:
        _try_array("harmonic_space")

    # hodge
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

    # harmonic
    if names & {"harmonic", "harmonic_basis", "harmonic_dim",
                "frustration_per_edge", "coparticipation_per_edge",
                "sigma_asymmetry_per_edge"}:
        try:
            # the sparse cycle frame, not core._harmonic.harmonic_projectors: that
            # one builds a dense nE x nE L1, eigendecomposes it, then assembles
            # P_harm, P_grad and P_curl at nE x nE each plus a pinv, and only
            # P_harm and the basis are ever read here
            from rexgraph.harmonic_sparse import harmonic_basis, harmonic_projection
            H = harmonic_basis(rex)
            dim_H = int(H.shape[1])
            scalars["harmonic_dim"] = dim_H

            if dim_H > 0:
                import scipy.sparse as _sp
                _save_npy(cache_dir, "harmonic_basis",
                          np.asarray(H.todense()) if _sp.issparse(H) else np.asarray(H))
                written.append("harmonic_basis")

                w = rex.w_E if rex.w_E is not None else np.ones(rex.nE)
                harm_sig = np.asarray(harmonic_projection(H, w), dtype=np.float64)

                # channel diagonals from the boundary structure. The old path
                # rebuilt each hat as evec @ diag(ev) @ evec.T, an nE x nE per
                # channel, to read nE numbers off the diagonal.
                chi = _channel_diagonals(rex)

                # chi columns are (topology, geometry, frustration,
                # coparticipation). frustration read column 0 and coparticipation
                # column 1, so both named a channel and used another; the sigma
                # line below already had the order right.
                frustration = np.abs(harm_sig) * chi[:, 2]
                coparticipation = np.abs(harm_sig) * chi[:, 3]
                _save_npy(cache_dir, "frustration_per_edge", frustration)
                _save_npy(cache_dir, "coparticipation_per_edge", coparticipation)
                written.extend(["frustration_per_edge", "coparticipation_per_edge"])

                # T against F, per edge, where the pair carries anything at all
                T, F = chi[:, 0], chi[:, 2]
                total = T + F
                sigma = np.divide(T - F, total, out=np.zeros_like(total),
                                  where=total > 0)
                _save_npy(cache_dir, "sigma_asymmetry_per_edge", sigma)
                written.append("sigma_asymmetry_per_edge")

                frust_sum = float(np.sum(frustration))
                copart_sum = float(np.sum(coparticipation))
                scalars["frustration_total"] = frust_sum
                scalars["coparticipation_total"] = copart_sum
                if copart_sum != 0.0:
                    scalars["health_ratio"] = frust_sum / copart_sum
        except (ImportError, Exception):
            pass

    return written, scalars


# Module-level convenience functions


def save_rex(
    path: str,
    obj: Any,
    *,
    cache: None | str | list[str] = None,
    encryption_properties: ContainerEncryptionProperties | None = None,
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
    encryption_properties : opaque property, optional
        Authenticated sealing context. ``None`` preserves the native plaintext
        directory layout.

    Examples
    --------
    >>> save_rex("graph.rex", rex)
    >>> save_rex("graph.rex", rex, cache="all")
    >>> save_rex("graph.rex", rex, cache=["topology", "spectral"])
    """
    bundle = RexBundle.from_graph(obj, cache=cache)
    bundle.save(path, encryption_properties=encryption_properties)


def load_rex(
    path: str,
    *,
    allow_unsealed: bool = False,
    decryption_properties: ContainerDecryptionProperties | None = None,
) -> Any:
    """Load a RexGraph or TemporalRex from a `.rex` bundle.

    Parameters
    ----------
    path : str
        Path to `.rex` directory.
    allow_unsealed : bool
        Permit a trusted legacy RexGraph bundle with no content digest. This is false
        by default because an unsealed bundle cannot be checked for modification.
    decryption_properties : opaque property, optional
        Authenticated opening context required by an encrypted bundle.

    Returns
    -------
    RexGraph or TemporalRex

    Examples
    --------
    >>> rex = load_rex("graph.rex")
    >>> trex = load_rex("temporal.rex")
    """
    bundle = RexBundle.load(path, decryption_properties=decryption_properties)
    return bundle.to_object(allow_unsealed=allow_unsealed)
