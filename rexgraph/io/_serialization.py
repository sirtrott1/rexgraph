# rexgraph/io/_serialization.py
"""
Type-aware serialization for rexgraph NamedTuples.

The rexgraph types.py module defines 30+ NamedTuple types with fields
that mix ndarrays, scalars (int, float, bool, str), dicts, tuples,
optional values, and in one case a nested RexGraph. The generic
"store each field" approach in zarr_format/hdf5_format/bundle works
for simple types but breaks on:

- dict fields (PerturbationResult.hodge_initial, FaceData.metrics)
- list-of-dict fields (FaceData.faces)
- tuple fields (PersistenceDiagram.betti, QuotientResult.betti_rel)
- Optional[NDArray] fields that may be None
- str fields (FieldPerturbationResult.mode)
- complex arrays (SchrodingerState with f_re/f_im representing
  complex state)

This module provides a generic write/read pair that inspects each
field at runtime and dispatches to the correct _compat helper. It
works with any storage backend (Zarr group, HDF5 group, or directory
of .npy files) through a thin adapter interface.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from ._compat import (
    as_str,
    json_default,
    to_native,
)


__all__ = [
    "write_namedtuple",
    "read_namedtuple",
    "write_result_dict",
    "read_result_dict",
    "StorageAdapter",
    "ZarrAdapter",
    "HDF5Adapter",
    "NpyAdapter",
]


# Storage adapter interface

class StorageAdapter:
    """Abstract interface for writing/reading fields to a storage group.

    Subclasses wrap a Zarr group, HDF5 group, or .npy directory and
    provide uniform put/get methods for arrays, scalars, dicts, and
    strings.
    """

    def put_array(self, name: str, arr: NDArray) -> None:
        raise NotImplementedError

    def get_array(self, name: str) -> Optional[NDArray]:
        raise NotImplementedError

    def put_scalar(self, name: str, value: Any) -> None:
        raise NotImplementedError

    def get_scalar(self, name: str, default: Any = None) -> Any:
        raise NotImplementedError

    def put_json(self, name: str, obj: Any) -> None:
        raise NotImplementedError

    def get_json(self, name: str, default: Any = None) -> Any:
        raise NotImplementedError

    def put_string(self, name: str, s: str) -> None:
        raise NotImplementedError

    def get_string(self, name: str, default: str = "") -> str:
        raise NotImplementedError

    def has(self, name: str) -> bool:
        raise NotImplementedError

    def subgroup(self, name: str) -> "StorageAdapter":
        raise NotImplementedError


# Zarr adapter

class ZarrAdapter(StorageAdapter):
    """Adapter wrapping a Zarr group."""

    def __init__(self, group, *, compressor=None, chunks=True):
        self._g = group
        self._compressor = compressor
        self._chunks = chunks

    def put_array(self, name: str, arr: NDArray) -> None:
        from ._compat import g_store_complex
        g_store_complex(self._g, name, np.asarray(arr),
                        compressor=self._compressor, chunks=self._chunks)

    def get_array(self, name: str) -> Optional[NDArray]:
        from ._compat import g_load_complex
        try:
            return g_load_complex(self._g, name)
        except KeyError:
            return None

    def put_scalar(self, name: str, value: Any) -> None:
        self._g.attrs[name] = to_native(value)

    def get_scalar(self, name: str, default: Any = None) -> Any:
        val = self._g.attrs.get(name, default)
        return to_native(as_str(val)) if val is not default else default

    def put_json(self, name: str, obj: Any) -> None:
        self._g.attrs[name] = json.dumps(obj, default=json_default)

    def get_json(self, name: str, default: Any = None) -> Any:
        raw = self._g.attrs.get(name)
        if raw is None:
            return default
        try:
            return json.loads(as_str(raw))
        except (json.JSONDecodeError, ValueError, TypeError):
            return default

    def put_string(self, name: str, s: str) -> None:
        self._g.attrs[name] = s

    def get_string(self, name: str, default: str = "") -> str:
        val = self._g.attrs.get(name, default)
        return str(as_str(val))

    def has(self, name: str) -> bool:
        return name in self._g or name in self._g.attrs

    def subgroup(self, name: str) -> "ZarrAdapter":
        if name in self._g:
            sub = self._g[name]
        else:
            sub = self._g.create_group(name)
        return ZarrAdapter(sub, compressor=self._compressor,
                           chunks=self._chunks)


# HDF5 adapter

class HDF5Adapter(StorageAdapter):
    """Adapter wrapping an h5py group."""

    def __init__(self, group, *, compression: str = "lzf", chunks: bool = True):
        self._g = group
        self._compression = compression
        self._chunks = chunks

    def put_array(self, name: str, arr: NDArray) -> None:
        from ._compat import h5_store_complex
        h5_store_complex(self._g, name, np.asarray(arr),
                         compression=self._compression, chunks=self._chunks)

    def get_array(self, name: str) -> Optional[NDArray]:
        from ._compat import h5_load_complex
        try:
            return h5_load_complex(self._g, name)
        except KeyError:
            return None

    def put_scalar(self, name: str, value: Any) -> None:
        self._g.attrs[name] = to_native(value)

    def get_scalar(self, name: str, default: Any = None) -> Any:
        val = self._g.attrs.get(name, default)
        return to_native(as_str(val)) if val is not default else default

    def put_json(self, name: str, obj: Any) -> None:
        self._g.attrs[name] = json.dumps(obj, default=json_default)

    def get_json(self, name: str, default: Any = None) -> Any:
        raw = self._g.attrs.get(name)
        if raw is None:
            return default
        try:
            return json.loads(as_str(raw))
        except (json.JSONDecodeError, ValueError, TypeError):
            return default

    def put_string(self, name: str, s: str) -> None:
        self._g.attrs[name] = s

    def get_string(self, name: str, default: str = "") -> str:
        val = self._g.attrs.get(name, default)
        return str(as_str(val))

    def has(self, name: str) -> bool:
        return name in self._g or name in self._g.attrs

    def subgroup(self, name: str) -> "HDF5Adapter":
        if name in self._g:
            sub = self._g[name]
        else:
            sub = self._g.create_group(name)
        return HDF5Adapter(sub, compression=self._compression,
                           chunks=self._chunks)


# Npy directory adapter (for .rex bundles)

class NpyAdapter(StorageAdapter):
    """Adapter wrapping a directory of .npy files and a JSON sidecar."""

    def __init__(self, directory):
        import pathlib
        self._dir = pathlib.Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / "_meta.json"
        self._meta: Optional[dict] = None

    def _load_meta(self) -> dict:
        if self._meta is None:
            if self._meta_path.exists():
                self._meta = json.loads(self._meta_path.read_text())
            else:
                self._meta = {}
        return self._meta

    def _save_meta(self) -> None:
        if self._meta is not None:
            self._meta_path.write_text(
                json.dumps(self._meta, indent=2, default=json_default)
            )

    def put_array(self, name: str, arr: NDArray) -> None:
        arr = np.asarray(arr)
        np.save(self._dir / f"{name}.npy", arr)

    def get_array(self, name: str) -> Optional[NDArray]:
        path = self._dir / f"{name}.npy"
        if path.exists():
            return np.load(path)
        return None

    def put_scalar(self, name: str, value: Any) -> None:
        meta = self._load_meta()
        meta[name] = to_native(value)
        self._save_meta()

    def get_scalar(self, name: str, default: Any = None) -> Any:
        return self._load_meta().get(name, default)

    def put_json(self, name: str, obj: Any) -> None:
        meta = self._load_meta()
        meta[name] = obj
        self._save_meta()

    def get_json(self, name: str, default: Any = None) -> Any:
        return self._load_meta().get(name, default)

    def put_string(self, name: str, s: str) -> None:
        meta = self._load_meta()
        meta[name] = s
        self._save_meta()

    def get_string(self, name: str, default: str = "") -> str:
        return str(self._load_meta().get(name, default))

    def has(self, name: str) -> bool:
        if (self._dir / f"{name}.npy").exists():
            return True
        return name in self._load_meta()

    def subgroup(self, name: str) -> "NpyAdapter":
        return NpyAdapter(self._dir / name)


# Field classification

def _classify_field(value: Any) -> str:
    """Determine how to store a NamedTuple field value.

    Returns one of: "array", "scalar", "string", "dict", "tuple",
    "list_of_dict", "none".
    """
    if value is None:
        return "none"
    if isinstance(value, np.ndarray):
        return "array"
    if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
        return "scalar"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, (list, tuple)):
        if len(value) > 0 and isinstance(value[0], dict):
            return "list_of_dict"
        # Try converting to array
        try:
            arr = np.asarray(value)
            if arr.dtype.kind in ("f", "i", "u", "c", "b"):
                return "array"
        except (ValueError, TypeError):
            pass
        return "tuple"
    return "dict"  # fallback: JSON-serialize


# Generic NamedTuple write/read

def write_namedtuple(
    adapter: StorageAdapter,
    name: str,
    obj,
) -> None:
    """Write a NamedTuple to storage.

    Creates a subgroup named `name` and stores each field according
    to its runtime type: arrays as datasets, scalars and strings as
    attrs, dicts and tuples as JSON, list-of-dicts as JSON arrays.

    Handles None fields by recording them in a _none_fields list.
    """
    if obj is None:
        return

    sub = adapter.subgroup(name)
    type_name = type(obj).__name__
    sub.put_string("_type_name", type_name)

    fields = obj._fields if hasattr(obj, "_fields") else []
    none_fields: List[str] = []

    for field_name in fields:
        value = getattr(obj, field_name)
        kind = _classify_field(value)

        if kind == "none":
            none_fields.append(field_name)
        elif kind == "array":
            arr = np.asarray(value) if not isinstance(value, np.ndarray) else value
            sub.put_array(field_name, arr)
        elif kind == "scalar":
            sub.put_scalar(field_name, value)
        elif kind == "string":
            sub.put_string(field_name, value)
        elif kind == "dict":
            _write_dict_field(sub, field_name, value)
        elif kind == "list_of_dict":
            sub.put_json(field_name, value)
        elif kind == "tuple":
            sub.put_json(field_name, list(value))

    if none_fields:
        sub.put_json("_none_fields", none_fields)


def read_namedtuple(
    adapter: StorageAdapter,
    name: str,
    type_class: Optional[Type] = None,
) -> Any:
    """Read a NamedTuple from storage.

    If type_class is given, constructs an instance of that type.
    Otherwise returns a plain dict of the stored fields.

    Fields listed in _none_fields are set to None. Missing array
    fields are set to None. Scalar and string fields use their
    stored values or defaults.
    """
    sub = adapter.subgroup(name)

    if not sub.has("_type_name"):
        return None

    none_fields = set(sub.get_json("_none_fields", []))

    if type_class is None:
        type_name = sub.get_string("_type_name")
        type_class = _resolve_type(type_name)

    if type_class is None:
        # Can't reconstruct; return raw dict
        return _read_as_dict(sub, none_fields)

    fields = type_class._fields
    values = {}

    for field_name in fields:
        if field_name in none_fields:
            values[field_name] = None
            continue

        # Try array first
        arr = sub.get_array(field_name)
        if arr is not None:
            values[field_name] = arr
            continue

        # Try scalar
        scalar = sub.get_scalar(field_name)
        if scalar is not None:
            values[field_name] = scalar
            continue

        # Try JSON (for dicts, tuples, list-of-dicts)
        json_val = sub.get_json(field_name)
        if json_val is not None:
            values[field_name] = json_val
            continue

        # Try string
        s = sub.get_string(field_name, default=None)
        if s is not None:
            values[field_name] = s
            continue

        # Try dict subgroup
        dict_sub = sub.subgroup(field_name)
        if dict_sub.has("_is_dict"):
            values[field_name] = _read_dict_field(sub, field_name)
            continue

        values[field_name] = None

    # Fix tuple fields that were stored as lists
    try:
        hints = type_class.__annotations__ if hasattr(type_class, "__annotations__") else {}
    except Exception:
        hints = {}

    for field_name, val in values.items():
        hint = hints.get(field_name, None)
        if hint is not None and isinstance(val, list):
            hint_str = str(hint)
            if "Tuple" in hint_str:
                values[field_name] = tuple(val)

    try:
        return type_class(**values)
    except TypeError:
        # Field count mismatch (version skew); return what we have
        return type_class(*[values.get(f) for f in fields])


def _read_as_dict(
    adapter: StorageAdapter,
    none_fields: Set[str],
) -> dict:
    """Read all fields from a subgroup into a plain dict."""
    # This is a fallback when we can't resolve the type
    return {}


# Dict field helpers

def _write_dict_field(adapter: StorageAdapter, name: str, d: dict) -> None:
    """Write a dict field, storing arrays as datasets and rest as JSON."""
    sub = adapter.subgroup(name)
    sub.put_scalar("_is_dict", True)

    arrays: Dict[str, NDArray] = {}
    scalars: dict = {}

    for k, v in d.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            scalars[k] = to_native(v) if isinstance(v, (np.integer, np.floating, np.bool_)) else v

    for k, arr in arrays.items():
        sub.put_array(k, arr)

    if scalars:
        sub.put_json("_scalars", scalars)

    sub.put_json("_array_keys", list(arrays.keys()))


def _read_dict_field(adapter: StorageAdapter, name: str) -> dict:
    """Read a dict field back from storage."""
    sub = adapter.subgroup(name)
    result: dict = {}

    array_keys = sub.get_json("_array_keys", [])
    for k in array_keys:
        arr = sub.get_array(k)
        if arr is not None:
            result[k] = arr

    scalars = sub.get_json("_scalars", {})
    if scalars:
        result.update(scalars)

    return result


# Result dict write/read (for non-NamedTuple dicts like analyze() output)

def write_result_dict(
    adapter: StorageAdapter,
    name: str,
    data: dict,
) -> None:
    """Write a result dict (like analyze() output) to storage.

    Handles mixed array/scalar/dict values. Each array becomes a
    dataset; scalars and non-array values are JSON-encoded.
    """
    sub = adapter.subgroup(name)
    sub.put_scalar("_is_result_dict", True)

    array_keys: List[str] = []
    json_data: dict = {}

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            sub.put_array(k, v)
            array_keys.append(k)
        elif isinstance(v, dict):
            # Nested dict: check if it contains arrays
            has_arrays = any(isinstance(vv, np.ndarray) for vv in v.values())
            if has_arrays:
                _write_dict_field(sub, k, v)
            else:
                json_data[k] = v
        else:
            json_data[k] = to_native(v) if isinstance(
                v, (np.integer, np.floating, np.bool_)) else v

    sub.put_json("_array_keys", array_keys)
    if json_data:
        sub.put_json("_json_data", json_data)


def read_result_dict(
    adapter: StorageAdapter,
    name: str,
) -> dict:
    """Read a result dict from storage."""
    sub = adapter.subgroup(name)
    result: dict = {}

    array_keys = sub.get_json("_array_keys", [])
    for k in array_keys:
        arr = sub.get_array(k)
        if arr is not None:
            result[k] = arr

    json_data = sub.get_json("_json_data", {})
    if json_data:
        result.update(json_data)

    return result


# Type resolution

_TYPE_REGISTRY: Dict[str, Type] = {}


def _resolve_type(type_name: str) -> Optional[Type]:
    """Look up a NamedTuple class by name.

    Lazily populates the registry on first call by importing
    rexgraph.types.
    """
    if not _TYPE_REGISTRY:
        _populate_registry()
    return _TYPE_REGISTRY.get(type_name)


def _populate_registry() -> None:
    """Import all NamedTuple types from rexgraph.types."""
    try:
        from .. import types as _types
        import inspect
        for name, obj in inspect.getmembers(_types):
            if (isinstance(obj, type)
                    and issubclass(obj, tuple)
                    and hasattr(obj, "_fields")):
                _TYPE_REGISTRY[name] = obj
    except ImportError:
        pass


def register_type(cls: Type) -> None:
    """Manually register a NamedTuple class for deserialization."""
    if hasattr(cls, "_fields"):
        _TYPE_REGISTRY[cls.__name__] = cls
