"""Bounded, non executable integrity framing for HDF5 and Zarr trees.

This physical seal covers every dataset, group, name and attribute, including
derived caches and appended results. The native semantic state seal still
defines the format independent identity of a complex. Content digests detect
accidental changes; they do not authenticate an untrusted author.
"""
from __future__ import annotations

import hashlib
import hmac
from itertools import product
import json
import struct

import numpy as np

from rexgraph.value_codec import pack_value

ATTRIBUTE = "rex_container_integrity"
FORMAT_VERSION = "3.0.0"
_BLOCK_BYTES = 8 * 1024 * 1024


def _frame(digest, tag, *values):
    digest.update(tag)
    for value in values:
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)


def _value(value, depth=0):
    if depth > 64:
        raise ValueError("container attributes are nested too deeply")
    # Physical cache attributes may explicitly contain IEEE NaN/Infinity.
    # Preserve their bits rather than applying the semantic finite value rule.
    if type(value) is float:
        return b"float" + struct.pack("<d", value)
    if isinstance(value, np.generic):
        return _value(np.asarray(value), depth+1)
    if isinstance(value, np.ndarray):
        dtype = pack_value(value.dtype.descr if value.dtype.fields else value.dtype.str)
        shape = pack_value(value.shape)
        data = (_value(value.tolist(), depth+1) if value.dtype.hasobject else value.tobytes(order="C"))
        return b"array" + pack_value((dtype, shape, data))
    if isinstance(value, (tuple, list)):
        return b"sequence" + pack_value(tuple(_value(v, depth+1) for v in value))
    if isinstance(value, dict):
        return b"mapping" + pack_value({k: _value(v, depth+1) for k, v in value.items()})
    return b"value" + pack_value(value)


def _blocks(array):
    shape = tuple(array.shape)
    if not shape:
        yield (), np.asarray(array[()])
        return
    if 0 in shape:
        return
    elements = max(1, _BLOCK_BYTES // max(1, np.dtype(array.dtype).itemsize))
    sizes = [1] * len(shape)
    for axis in reversed(range(len(shape))):
        sizes[axis] = min(shape[axis], elements)
        elements = max(1, elements // sizes[axis])
    for starts in product(*(range(0, n, size) for n, size in zip(shape, sizes, strict=True))):
        selection = tuple(slice(start, min(n, start+size)) for start, n, size in zip(starts, shape, sizes, strict=True))
        yield starts, np.asarray(array[selection])


def container_digest(root, *, _prepare=None):
    """Hash the logical stored tree, reading tensors in bounded blocks."""
    digest = hashlib.sha256(b"RexGraph physical container\x01")
    ancestors = set()
    def walk(node, path, depth=0):
        if depth > 64:
            raise ValueError("container group nesting exceeds the supported limit")
        group = hasattr(node, "keys")
        identity = getattr(node, "id", None)
        if identity is not None and identity in ancestors:
            raise ValueError("cyclic container group links are not supported")
        if group and _prepare is not None:
            _prepare(node)
        _frame(digest, b"G" if group else b"D", path.encode("utf-8"))
        attrs = dict(node.attrs)
        if path == "":
            attrs.pop(ATTRIBUTE, None)
        for name in sorted(attrs):
            _frame(digest, b"A", name.encode("utf-8"), _value(attrs[name]))
        if not group:
            dtype = np.dtype(node.dtype)
            _frame(digest, b"S", pack_value(tuple(node.shape)), pack_value(dtype.descr if dtype.fields else dtype.str))
            for starts, block in _blocks(node):
                _frame(digest, b"B", pack_value(starts), _value(block))
            return
        if identity is not None:
            ancestors.add(identity)
        try:
            for name in sorted(node.keys()):
                # Refuse links that read outside this file or depend on a path.
                if identity is not None and type(node.get(name, getlink=True)).__name__ != "HardLink":
                    raise ValueError("external and symbolic container links are not supported")
                walk(node[name], path+"/"+name, depth+1)
        finally:
            if identity is not None:
                ancestors.remove(identity)
    walk(root, "")
    return digest.hexdigest()


def seal_container(root):
    from rexgraph.state import fname_encode
    def inventories(group):
        if "rex_state_header" in group.attrs:
            names = json.loads(group.attrs["tensor_names"])
            expected = {fname_encode(name) for name in names}
            caches = set(json.loads(group.attrs.get("rex_cache_groups", "[]")))
            group.attrs["rex_auxiliary_groups"] = json.dumps(sorted(set(group.keys()) - expected - caches))
    root.attrs["format_version"] = FORMAT_VERSION
    root.attrs[ATTRIBUTE] = json.dumps({"version": 1, "algorithm": "sha256", "digest": container_digest(root, _prepare=inventories)}, sort_keys=True)


def verify_container(root, *, required=False):
    """Verify new containers; allow older graph formats to use their state seal."""
    envelope = root.attrs.get(ATTRIBUTE)
    if envelope is None:
        if required or root.attrs.get("format_version") == FORMAT_VERSION:
            raise ValueError("container has no physical integrity seal; legacy auxiliary reads require allow_unsealed=True")
        return False
    try:
        record = json.loads(envelope)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid container integrity envelope") from exc
    if (not isinstance(record, dict) or set(record) != {"version", "algorithm", "digest"}
            or type(record["version"]) is not int or record["version"] != 1
            or record["algorithm"] != "sha256" or type(record["digest"]) is not str
            or len(record["digest"]) != 64):
        raise ValueError("invalid container integrity envelope")
    if not hmac.compare_digest(record["digest"], container_digest(root)):
        raise ValueError("container integrity digest mismatched (modified or unclaimed entries)")
    return True
