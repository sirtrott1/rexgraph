# rexgraph/io/_compat.py
"""
Zarr v2/v3 and HDF5 compatibility layer.

Zarr helpers: path normalization, root group open/create, numcodecs
Blosc to v3 BloscCodec bridge, compressor normalization, array
creation (v2 create_dataset / v3 create_array), complex array
storage, sparse CSR storage, ragged UTF-8 strings, dict-of-arrays
group storage.

HDF5 helpers: file open/close, complex arrays, sparse CSR, dict
storage, variable-length UTF-8 strings.

Shared helpers: numpy-to-native conversion, JSON encoding, boolean
mask storage.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Zarr version detection

try:
    import zarr
    from packaging.version import Version

    _ZARR_VER = Version(getattr(zarr, "__version__", "0"))
    ZARR_V3: bool = _ZARR_VER.major >= 3
    HAS_ZARR: bool = True
except ImportError:
    zarr = None  # type: ignore[assignment]
    ZARR_V3 = False
    HAS_ZARR = False

# HDF5 detection

try:
    import h5py
    HAS_HDF5: bool = True
except ImportError:
    h5py = None  # type: ignore[assignment]
    HAS_HDF5 = False

# Scipy detection

try:
    import scipy.sparse as _sp
    HAS_SCIPY: bool = True
except ImportError:
    _sp = None  # type: ignore[assignment]
    HAS_SCIPY = False


__all__ = [
    "ZARR_V3",
    "HAS_ZARR",
    "HAS_HDF5",
    "HAS_SCIPY",
    "to_native",
    "json_default",
    "NumpyJSONEncoder",
    "as_str",
    "ensure_zarr_suffix",
    "rm_rf",
    "open_root_group",
    "create_root_group",
    "default_zarr_compressor",
    "normalize_zarr_compressor",
    "g_create_array",
    "g_store_complex",
    "g_load_complex",
    "g_store_sparse_csr",
    "g_load_sparse_csr",
    "g_store_dict",
    "g_load_dict",
    "g_store_bool_masks",
    "g_load_bool_masks",
    "write_text_array",
    "read_text_array",
    "open_hdf5",
    "h5_store_array",
    "h5_load_array",
    "h5_store_complex",
    "h5_load_complex",
    "h5_store_sparse_csr",
    "h5_load_sparse_csr",
    "h5_store_dict",
    "h5_load_dict",
    "h5_store_bool_masks",
    "h5_load_bool_masks",
    "h5_store_strings",
    "h5_load_strings",
]


# Attrs that are internal metadata, not user data (used by dict loaders)
_INTERNAL_ATTRS = frozenset({
    "is_complex", "is_sparse", "dtype", "shape",
    "format", "nnz", "encoding", "is_bool_masks",
})


# Shared type conversion

def to_native(v: Any) -> Any:
    """Convert numpy scalars to Python natives for JSON and attrs.

    NaN and Inf become 0.0. ndarrays become nested lists.
    """
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        val = float(v)
        if val != val or val == float("inf") or val == float("-inf"):
            return 0.0
        return val
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def json_default(o: Any) -> Any:
    """JSON serializer fallback for numpy types.

    Pass as json.dumps(obj, default=json_default).
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        val = float(o)
        if val != val or val == float("inf") or val == float("-inf"):
            return 0.0
        return val
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types. NaN/Inf become 0."""

    def default(self, obj: Any) -> Any:
        return json_default(obj)


def as_str(x: Any) -> Any:
    """Decode bytes or np.bytes_ to str; pass through everything else."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    if isinstance(x, np.str_):
        return str(x)
    return x


# Path utilities

def ensure_zarr_suffix(path: str | os.PathLike) -> str:
    """Append .zarr to path if not already present."""
    p = os.fspath(path)
    return p if p.endswith(".zarr") else p + ".zarr"


def rm_rf(path: str) -> None:
    """Remove file or directory tree if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


# Zarr root-group helpers

def _require_zarr() -> None:
    if not HAS_ZARR:
        raise ImportError("zarr is required: pip install zarr")


def open_root_group(path: str, mode: str = "r"):
    """Open a Zarr root group. Works for both v2 and v3."""
    _require_zarr()
    return zarr.open_group(store=path, mode=mode)


def create_root_group(path: str, *, overwrite: bool = True):
    """Create a fresh Zarr root group at path."""
    _require_zarr()
    if overwrite and os.path.exists(path):
        rm_rf(path)
    return zarr.open_group(store=path, mode="w" if overwrite else "a")


# Compressor bridging

def _numcodecs_blosc_to_v3_codec(comp: Any):
    """Convert a numcodecs.Blosc to a v3 BloscCodec. Returns None on failure."""
    try:
        from numcodecs import Blosc as _NCBlosc
    except Exception:
        return None

    if not isinstance(comp, _NCBlosc):
        return None

    try:
        from zarr.codecs import BloscCodec, BloscShuffle
    except Exception:
        return None

    shuffle_val = int(getattr(comp, "shuffle", 0) or 0)
    if shuffle_val == 2:
        shuffle = BloscShuffle.bitshuffle
    elif shuffle_val == 1:
        shuffle = BloscShuffle.shuffle
    else:
        shuffle = BloscShuffle.noshuffle

    return BloscCodec(
        cname=str(getattr(comp, "cname", "zstd")),
        clevel=int(getattr(comp, "clevel", 3)),
        shuffle=shuffle,
    )


def _normalize_create_kwargs(kw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize array-creation kwargs across Zarr v2 and v3."""
    out = dict(kw)

    if ZARR_V3:
        if "chunks" in out and isinstance(out["chunks"], (bool, type(None))):
            out.pop("chunks", None)

        if "compressor" in out:
            comp = out.pop("compressor")
            if comp is not None:
                v3c = _numcodecs_blosc_to_v3_codec(comp)
                if v3c is not None:
                    out["compressors"] = (v3c,)

        if "compressors" in out and out["compressors"] is not None:
            c = out["compressors"]
            if isinstance(c, list):
                out["compressors"] = tuple(c)
            elif not isinstance(c, tuple):
                out["compressors"] = (c,)
    else:
        out.pop("compressors", None)

    return out


def default_zarr_compressor() -> Any:
    """Return Blosc(zstd, clevel=3, bitshuffle) or None if unavailable."""
    try:
        from numcodecs import Blosc
        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except Exception:
        return None


def normalize_zarr_compressor(comp: Any) -> Any:
    """Normalize a compressor value.

    Accepts None, a numcodecs compressor, a v3 codec, or string
    shorthands: "blosc", "zstd", "none", "off".
    """
    if comp is None:
        return None
    if isinstance(comp, str):
        key = comp.strip().lower()
        if key in ("none", "null", "no", "off", "false"):
            return None
        if key in ("blosc", "blosc-zstd", "zstd", "default"):
            return default_zarr_compressor()
        return None
    return comp


# Zarr array creation

def g_create_array(
    group,
    name: str,
    data=None,
    *,
    dtype=None,
    shape=None,
    **kw,
):
    """Create an array inside a Zarr group.

    Tries create_array() first, then create_dataset() for older v2.
    If the data= kwarg is rejected (v3), falls back to create then
    assign.
    """
    kw = _normalize_create_kwargs(kw)

    create_fn = (
        getattr(group, "create_array", None)
        or getattr(group, "create_dataset", None)
    )
    if create_fn is None:
        raise AttributeError(
            "Group has neither create_array nor create_dataset."
        )

    if data is not None:
        arr = np.asarray(data)
        shp = arr.shape if shape is None else shape
        dt = arr.dtype if dtype is None else dtype
        try:
            return create_fn(name=name, data=arr, **kw)
        except TypeError:
            z = create_fn(name=name, shape=shp, dtype=dt, **kw)
            z[...] = arr
            return z

    if shape is None:
        raise TypeError("shape is required when creating an empty array.")
    return create_fn(name=name, shape=shape, dtype=dtype, **kw)


# Zarr complex array helpers

def g_store_complex(
    group,
    name: str,
    arr: NDArray,
    *,
    compressor=None,
    chunks=True,
) -> None:
    """Store a possibly-complex ndarray in a Zarr group.

    Complex arrays become a subgroup with real and imag datasets
    plus is_complex/dtype/shape attrs. Real arrays are stored as a
    single dataset.
    """
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        sub = group.create_group(name)
        sub.attrs["is_complex"] = True
        sub.attrs["dtype"] = str(arr.dtype)
        sub.attrs["shape"] = list(arr.shape)
        g_create_array(sub, "real", data=arr.real,
                       compressor=compressor, chunks=chunks)
        g_create_array(sub, "imag", data=arr.imag,
                       compressor=compressor, chunks=chunks)
    else:
        g_create_array(group, name, data=arr,
                       compressor=compressor, chunks=chunks)


def g_load_complex(group, name: str) -> np.ndarray:
    """Load a real or complex array from a Zarr group.

    Checks for the is_complex attr and reassembles real + imag if
    present.
    """
    obj = group[name]
    if hasattr(obj, "attrs") and obj.attrs.get("is_complex", False):
        real = np.asarray(obj["real"])
        imag = np.asarray(obj["imag"])
        dtype = np.dtype(as_str(obj.attrs.get("dtype", "complex128")))
        return (real + 1j * imag).astype(dtype)
    return np.asarray(obj)


# Zarr sparse CSR helpers

def g_store_sparse_csr(
    group,
    name: str,
    matrix,
    *,
    compressor=None,
    chunks=True,
) -> None:
    """Store a scipy sparse matrix as a Zarr subgroup.

    Creates <name>/{data, indices, indptr} with shape/nnz/format
    attrs. Dense ndarrays fall through to g_store_complex.
    """
    if isinstance(matrix, np.ndarray):
        g_store_complex(group, name, matrix,
                        compressor=compressor, chunks=chunks)
        return

    if not HAS_SCIPY:
        raise ImportError("scipy is required for sparse storage: pip install scipy")

    if _sp.issparse(matrix):
        csr = matrix.tocsr()
    else:
        raise TypeError(
            f"Expected ndarray or scipy sparse, got {type(matrix).__name__}"
        )

    sub = group.create_group(name)
    sub.attrs["is_sparse"] = True
    sub.attrs["format"] = "csr"
    sub.attrs["shape"] = list(csr.shape)
    sub.attrs["nnz"] = int(csr.nnz)
    sub.attrs["dtype"] = str(csr.dtype)

    g_create_array(sub, "data", data=csr.data,
                   compressor=compressor, chunks=chunks)
    g_create_array(sub, "indices", data=csr.indices.astype(np.int32),
                   compressor=compressor, chunks=chunks)
    g_create_array(sub, "indptr", data=csr.indptr.astype(np.int32),
                   compressor=compressor, chunks=chunks)


def g_load_sparse_csr(group, name: str, *, dense: bool = False):
    """Load a sparse or dense array from a Zarr group.

    Parameters
    ----------
    group : Zarr group
    name : str
    dense : bool
        If True, return a dense ndarray via .toarray(). Matches the
        graph.py convention where cached properties are dense.
    """
    obj = group[name]

    if hasattr(obj, "attrs") and obj.attrs.get("is_sparse", False):
        if not HAS_SCIPY:
            raise ImportError("scipy is required: pip install scipy")
        data = np.asarray(obj["data"])
        indices = np.asarray(obj["indices"])
        indptr = np.asarray(obj["indptr"])
        shape = tuple(int(s) for s in obj.attrs.get("shape"))
        mat = _sp.csr_matrix((data, indices, indptr), shape=shape)
        return mat.toarray() if dense else mat

    if hasattr(obj, "attrs") and obj.attrs.get("is_complex", False):
        return g_load_complex(group, name)

    return np.asarray(obj)


# Zarr dict-of-arrays helpers

def g_store_dict(
    group,
    name: str,
    data: Dict[str, Any],
    *,
    compressor=None,
    chunks=True,
) -> None:
    """Store a dict of arrays and scalars as a Zarr subgroup.

    Arrays become datasets (complex-aware). Scalars and lists become
    JSON attrs. Nested dicts become sub-subgroups (one level deep).
    """
    sub = group.create_group(name)
    _dict_to_group(sub, data, compressor=compressor, chunks=chunks)


def _dict_to_group(
    group,
    data: Dict[str, Any],
    *,
    compressor=None,
    chunks=True,
) -> None:
    """Write dict contents into a Zarr group (recursive)."""
    for key, val in data.items():
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            g_store_complex(group, key, val,
                            compressor=compressor, chunks=chunks)
        elif isinstance(val, dict):
            sub2 = group.create_group(key)
            _dict_to_group(sub2, val, compressor=compressor, chunks=chunks)
        elif isinstance(val, (list, tuple)):
            try:
                arr = np.asarray(val)
                if arr.dtype.kind in ("f", "i", "u", "c", "b"):
                    g_store_complex(group, key, arr,
                                    compressor=compressor, chunks=chunks)
                    continue
            except (ValueError, TypeError):
                pass
            group.attrs[key] = json.dumps(val, default=json_default)
        elif isinstance(val, (int, float, bool, np.integer, np.floating, np.bool_)):
            group.attrs[key] = to_native(val)
        elif isinstance(val, str):
            group.attrs[key] = val
        else:
            try:
                group.attrs[key] = json.dumps(val, default=json_default)
            except (TypeError, ValueError):
                pass


def g_load_dict(group, name: str) -> Dict[str, Any]:
    """Load a dict of arrays and scalars from a Zarr subgroup."""
    try:
        sub = group[name]
    except KeyError:
        return {}
    return _group_to_dict(sub)


def _group_to_dict(group) -> Dict[str, Any]:
    """Read a Zarr group into a dict (recursive)."""
    result: Dict[str, Any] = {}

    for key in (group.attrs.keys() if hasattr(group.attrs, "keys") else []):
        if key in _INTERNAL_ATTRS:
            continue
        val = group.attrs[key]
        val = as_str(val)
        if isinstance(val, str):
            try:
                result[key] = json.loads(val)
                continue
            except (json.JSONDecodeError, ValueError):
                pass
        result[key] = to_native(val) if isinstance(val, (np.integer, np.floating, np.bool_)) else val

    for key in (group.keys() if hasattr(group, "keys") else []):
        obj = group[key]
        if hasattr(obj, "attrs") and obj.attrs.get("is_sparse", False):
            result[key] = g_load_sparse_csr(group, key, dense=True)
        elif hasattr(obj, "attrs") and obj.attrs.get("is_complex", False):
            result[key] = g_load_complex(group, key)
        elif hasattr(obj, "keys") and hasattr(obj, "attrs"):
            result[key] = _group_to_dict(obj)
        else:
            result[key] = np.asarray(obj)

    return result


# Zarr boolean mask helpers

def g_store_bool_masks(
    group,
    name: str,
    masks: Dict[str, NDArray],
    *,
    compressor=None,
    chunks=True,
) -> None:
    """Store boolean mask arrays as a Zarr subgroup.

    Stored as uint8 because Zarr v3 has issues with bool dtype.
    """
    sub = group.create_group(name)
    sub.attrs["is_bool_masks"] = True
    for key, mask in masks.items():
        g_create_array(sub, key, data=np.asarray(mask).astype(np.uint8),
                       dtype="u1", compressor=compressor, chunks=chunks)


def g_load_bool_masks(group, name: str) -> Dict[str, np.ndarray]:
    """Load boolean masks from a Zarr subgroup."""
    sub = group[name]
    result: Dict[str, np.ndarray] = {}
    for key in sub.keys() if hasattr(sub, "keys") else []:
        result[key] = np.asarray(sub[key]).astype(bool)
    return result


# Zarr ragged UTF-8 string helpers

def write_text_array(
    group,
    name: str,
    seq,
    *,
    compressor=None,
    chunks=True,
) -> None:
    """Store a sequence of strings as ragged UTF-8 byte arrays.

    Creates <name>_vls/values (uint8) and <name>_vls/offsets (int64).
    Avoids fixed-width |S# dtypes which break across Zarr v2/v3 and
    NumPy 2.x.
    """
    encoded: List[bytes] = []
    for s in seq:
        if isinstance(s, (bytes, bytearray, memoryview)):
            encoded.append(bytes(s))
        else:
            encoded.append(("" if s is None else str(s)).encode("utf-8"))

    lengths = np.fromiter(
        (len(b) for b in encoded), count=len(encoded), dtype="int64"
    )
    offsets = np.empty(len(encoded) + 1, dtype="int64")
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    values = np.frombuffer(b"".join(encoded), dtype="uint8")

    vls_name = f"{name}_vls"
    if vls_name in group:
        try:
            del group[vls_name]
        except Exception:
            pass

    sub = group.create_group(vls_name)
    g_create_array(sub, "values", values, dtype="u1",
                   compressor=compressor, chunks=chunks)
    g_create_array(sub, "offsets", offsets, dtype="i8",
                   compressor=compressor, chunks=chunks)
    sub.attrs["encoding"] = "utf-8"


def read_text_array(group, name: str) -> List[bytes]:
    """Read a string array from legacy fixed-width or ragged layout.

    Returns raw bytes. The caller decodes (typically .decode("utf-8")).
    Raises KeyError if neither format is found.
    """
    # Legacy fixed-width bytes dataset
    try:
        obj = group[name]
    except KeyError:
        obj = None

    if obj is not None and hasattr(obj, "dtype"):
        if getattr(obj.dtype, "kind", None) == "S":
            return [bytes(x).rstrip(b"\0") for x in obj[:]]

    # Ragged layout
    vls_name = f"{name}_vls"
    try:
        sub = group[vls_name]
    except KeyError:
        sub = None

    if sub is not None:
        vals = sub["values"][:]
        offs = sub["offsets"][:]
        out: List[bytes] = []
        for i in range(len(offs) - 1):
            lo, hi = int(offs[i]), int(offs[i + 1])
            out.append(vals[lo:hi].tobytes())
        return out

    raise KeyError(
        f"String field '{name}' not found "
        f"(checked '{name}' dataset and '{vls_name}' ragged group)."
    )


# HDF5 helpers

def _require_hdf5() -> None:
    if not HAS_HDF5:
        raise ImportError("h5py is required: pip install h5py")


def open_hdf5(path: str, mode: str = "r"):
    """Open an HDF5 file. Returns an h5py.File."""
    _require_hdf5()
    return h5py.File(path, mode)


# HDF5 array storage

def h5_store_array(
    group,
    name: str,
    arr: np.ndarray,
    *,
    compression: str = "lzf",
    chunks: bool = True,
) -> None:
    """Store an ndarray as an HDF5 dataset.

    Complex arrays become a subgroup with real and imag datasets.
    """
    arr = np.asarray(arr)
    kw: dict = {}
    if compression:
        kw["compression"] = compression
    if chunks:
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


def h5_load_array(group, name: str) -> np.ndarray:
    """Load an ndarray from an HDF5 dataset or complex subgroup."""
    obj = group[name]
    if isinstance(obj, h5py.Group) and obj.attrs.get("is_complex", False):
        real = obj["real"][:]
        imag = obj["imag"][:]
        dtype = np.dtype(as_str(obj.attrs.get("dtype", "complex128")))
        return (real + 1j * imag).astype(dtype)
    return obj[:]


# HDF5 complex array helpers

def h5_store_complex(
    group,
    name: str,
    arr: NDArray,
    *,
    compression: str = "lzf",
    chunks: bool = True,
) -> None:
    """Store a possibly-complex ndarray in HDF5.

    Same behavior as h5_store_array; provided for API symmetry with
    g_store_complex.
    """
    h5_store_array(group, name, np.asarray(arr),
                   compression=compression, chunks=chunks)


def h5_load_complex(group, name: str) -> np.ndarray:
    """Load a possibly-complex ndarray from HDF5."""
    return h5_load_array(group, name)


# HDF5 sparse CSR helpers

def h5_store_sparse_csr(
    group,
    name: str,
    matrix,
    *,
    compression: str = "lzf",
    chunks: bool = True,
) -> None:
    """Store a scipy sparse matrix as an HDF5 subgroup.

    Creates <name>/{data, indices, indptr}. Dense arrays fall through
    to h5_store_complex.
    """
    if isinstance(matrix, np.ndarray):
        h5_store_complex(group, name, matrix,
                         compression=compression, chunks=chunks)
        return

    if not HAS_SCIPY:
        raise ImportError("scipy is required for sparse storage: pip install scipy")

    if _sp.issparse(matrix):
        csr = matrix.tocsr()
    else:
        raise TypeError(
            f"Expected ndarray or scipy sparse, got {type(matrix).__name__}"
        )

    kw: dict = {}
    if compression:
        kw["compression"] = compression
    if chunks:
        kw["chunks"] = True

    sub = group.create_group(name)
    sub.attrs["is_sparse"] = True
    sub.attrs["format"] = "csr"
    sub.attrs["shape"] = list(csr.shape)
    sub.attrs["nnz"] = int(csr.nnz)
    sub.attrs["dtype"] = str(csr.dtype)

    sub.create_dataset("data", data=csr.data, **kw)
    sub.create_dataset("indices", data=csr.indices.astype(np.int32), **kw)
    sub.create_dataset("indptr", data=csr.indptr.astype(np.int32), **kw)


def h5_load_sparse_csr(group, name: str, *, dense: bool = False):
    """Load a sparse or dense array from an HDF5 group.

    Parameters
    ----------
    group : h5py group
    name : str
    dense : bool
        If True, return a dense ndarray via .toarray().
    """
    obj = group[name]

    if isinstance(obj, h5py.Group) and obj.attrs.get("is_sparse", False):
        if not HAS_SCIPY:
            raise ImportError("scipy is required: pip install scipy")
        data = obj["data"][:]
        indices = obj["indices"][:]
        indptr = obj["indptr"][:]
        shape = tuple(int(s) for s in obj.attrs.get("shape"))
        mat = _sp.csr_matrix((data, indices, indptr), shape=shape)
        return mat.toarray() if dense else mat

    return h5_load_array(group, name)


# HDF5 dict-of-arrays helpers

def h5_store_dict(
    group,
    name: str,
    data: Dict[str, Any],
    *,
    compression: str = "lzf",
    chunks: bool = True,
) -> None:
    """Store a dict of arrays and scalars as an HDF5 subgroup.

    Same behavior as g_store_dict but for HDF5.
    """
    sub = group.create_group(name)
    _h5_dict_to_group(sub, data, compression=compression, chunks=chunks)


def _h5_dict_to_group(
    group,
    data: Dict[str, Any],
    *,
    compression: str = "lzf",
    chunks: bool = True,
) -> None:
    """Write dict contents into an HDF5 group (recursive)."""
    for key, val in data.items():
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            h5_store_complex(group, key, val,
                             compression=compression, chunks=chunks)
        elif isinstance(val, dict):
            sub2 = group.create_group(key)
            _h5_dict_to_group(sub2, val,
                              compression=compression, chunks=chunks)
        elif isinstance(val, (list, tuple)):
            try:
                arr = np.asarray(val)
                if arr.dtype.kind in ("f", "i", "u", "c", "b"):
                    h5_store_complex(group, key, arr,
                                     compression=compression, chunks=chunks)
                    continue
            except (ValueError, TypeError):
                pass
            group.attrs[key] = json.dumps(val, default=json_default)
        elif isinstance(val, (int, float, bool, np.integer, np.floating, np.bool_)):
            group.attrs[key] = to_native(val)
        elif isinstance(val, str):
            group.attrs[key] = val
        else:
            try:
                group.attrs[key] = json.dumps(val, default=json_default)
            except (TypeError, ValueError):
                pass


def h5_load_dict(group, name: str) -> Dict[str, Any]:
    """Load a dict of arrays and scalars from an HDF5 subgroup."""
    try:
        sub = group[name]
    except KeyError:
        return {}
    return _h5_group_to_dict(sub)


def _h5_group_to_dict(group) -> Dict[str, Any]:
    """Read an HDF5 group into a dict (recursive)."""
    result: Dict[str, Any] = {}

    for key in group.attrs:
        if key in _INTERNAL_ATTRS:
            continue
        val = group.attrs[key]
        val = as_str(val)
        if isinstance(val, str):
            try:
                result[key] = json.loads(val)
                continue
            except (json.JSONDecodeError, ValueError):
                pass
        result[key] = to_native(val) if isinstance(val, (np.integer, np.floating, np.bool_)) else val

    for key in group:
        obj = group[key]
        if isinstance(obj, h5py.Group):
            if obj.attrs.get("is_sparse", False):
                result[key] = h5_load_sparse_csr(group, key, dense=True)
            elif obj.attrs.get("is_complex", False):
                result[key] = h5_load_complex(group, key)
            else:
                result[key] = _h5_group_to_dict(obj)
        else:
            result[key] = obj[:]

    return result


# HDF5 boolean mask helpers

def h5_store_bool_masks(
    group,
    name: str,
    masks: Dict[str, NDArray],
    *,
    compression: str = "lzf",
    chunks: bool = True,
) -> None:
    """Store boolean mask arrays as an HDF5 subgroup (uint8)."""
    kw: dict = {}
    if compression:
        kw["compression"] = compression
    if chunks:
        kw["chunks"] = True

    sub = group.create_group(name)
    sub.attrs["is_bool_masks"] = True
    for key, mask in masks.items():
        sub.create_dataset(key, data=np.asarray(mask).astype(np.uint8), **kw)


def h5_load_bool_masks(group, name: str) -> Dict[str, np.ndarray]:
    """Load boolean masks from an HDF5 subgroup."""
    sub = group[name]
    result: Dict[str, np.ndarray] = {}
    for key in sub:
        if isinstance(sub[key], h5py.Dataset):
            result[key] = sub[key][:].astype(bool)
    return result


# HDF5 string helpers

def h5_store_strings(
    group,
    name: str,
    strings: List[str],
) -> None:
    """Store a list of strings as a variable-length UTF-8 HDF5 dataset."""
    _require_hdf5()
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def h5_load_strings(group, name: str) -> List[str]:
    """Load a list of strings from an HDF5 dataset."""
    return [
        s.decode("utf-8") if isinstance(s, bytes) else str(s)
        for s in group[name][:]
    ]
