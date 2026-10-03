"""V10 semantic envelope over the canonical graph component encoder.

The binary semantic record owns every interpretation field and component claim.
Container headers are a bounded compatibility projection, never reconstruction
authority. Legacy state decoding is retained separately in :mod:`rexgraph.state`.
"""
from __future__ import annotations

import hmac

import numpy as np

from .components import component_registry
from .identity import tensor_digest
from .value_codec import pack_value, unpack_value

__all__ = ["SEMANTICS_TENSOR", "to_sealed_state", "migrate_state", "state_identity",
           "verify_sealed_state"]
SEMANTICS_TENSOR = "rex_semantics"
_PROJECTION = ("object_type", "nV", "nE", "nF", "directed", "g_channel", "c_channel")
_HEADER_FIELDS = {*_PROJECTION, "format_version", "agent_meta", "cell_meta", "nested", "sectionings", "merkle", "n_graded_duals", "graded_shapes", "empty_face_grade"}
_CONTAINER_FIELDS = {"magic", "tensor_names", "cached_arrays", "cache_scalars", "wire_digest", "meta", "bridge_version"}


def _projection(header):
    return dict({name: header[name] for name in _PROJECTION if name in header}, format_version=10)


def _header_contract(header):
    if not isinstance(header, dict) or header.get("object_type") != "RexGraph":
        raise ValueError("invalid native state header")
    if set(header) - _HEADER_FIELDS:
        raise ValueError("unknown semantic header fields")
    for name in ("nV", "nE", "nF"):
        if type(header.get(name)) is not int or not 0 <= header[name] < 2**31:
            raise ValueError(f"invalid native state dimension {name}")
    if type(header.get("directed")) is not bool or header.get("g_channel") not in ("raw", "normalized") or header.get("c_channel") not in ("share", "count"):
        raise ValueError("invalid native state direction or channel declaration")
    count = header.get("n_graded_duals", 0)
    if type(count) is not int or not 0 <= count <= 64:
        raise ValueError("invalid graded boundary count")
    if count and len(header.get("graded_shapes", [])) != count:
        raise ValueError("invalid graded boundary shapes")
    if "empty_face_grade" in header and (header["empty_face_grade"] is not True or header["nF"] != 0 or count):
        raise ValueError("invalid explicit empty face grade")
    schema = header.get("cell_meta", [])
    if not isinstance(schema, list):
        raise ValueError("invalid attribute schema")
    seen = set()
    for col in schema:
        if (not isinstance(col, dict) or type(col.get("dim")) is not int or col["dim"] < 0
                or type(col.get("key")) is not str or col.get("kind") not in {"num", "str", "structured", "value", "field", "section", "model", "span", "rex"}):
            raise ValueError("invalid attribute schema entry")
        identity = (col["dim"], col["key"])
        if identity in seen:
            raise ValueError("duplicate attribute schema entry")
        seen.add(identity)


def seal_state(tensors, header, tensor_codecs):
    from .state import RexState
    if SEMANTICS_TENSOR in tensors:
        raise ValueError("reserved semantic tensor already present")
    _header_contract(header)
    registry = component_registry()
    descriptors = registry.describe(header, tensors, tensor_codecs)
    semantic_header = dict(header, format_version=10)
    record = {"format_version": 10, "object_type": "RexGraph", "header": semantic_header,
              "components": [d.as_record() for d in descriptors], "tensor_codecs": tensor_codecs}
    payload = pack_value(record)
    result = dict(tensors)
    result[SEMANTICS_TENSOR] = np.frombuffer(payload, np.uint8).copy()
    summary = _projection(semantic_header)
    summary.update(digest_names=sorted(result), digest_algo=2, digest=tensor_digest(result))
    return RexState(result, summary)


def _record(state, *, verify=True):
    summary, tensors = state.header, state.tensors
    if set(summary) - {*_PROJECTION, "format_version", "digest", "digest_algo", "digest_names", *_CONTAINER_FIELDS}:
        raise ValueError("unclaimed container header fields")
    if type(summary.get("format_version")) is not int or summary["format_version"] != 10:
        raise ValueError("unsupported rex state format_version or downgraded semantic state")
    if "digest" not in summary:
        raise ValueError("the stored state carries no content digest; native state cannot be downgraded")
    names = summary.get("digest_names")
    if names != sorted(tensors) or summary.get("digest_algo") != 2:
        raise ValueError("state tensors do not match their semantic seal framing")
    if verify:
        digest = summary.get("digest")
        if not isinstance(digest, str) or len(digest) != 64 or not hmac.compare_digest(digest, tensor_digest(tensors)):
            raise ValueError("state tensors do not match their semantic digest")
    raw = np.asarray(tensors[SEMANTICS_TENSOR])
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("invalid semantic record tensor")
    record = unpack_value(raw.tobytes())
    if (not isinstance(record, dict) or set(record) != {"format_version", "object_type", "header", "components", "tensor_codecs"}
            or type(record["format_version"]) is not int or record["format_version"] != 10 or record["object_type"] != "RexGraph"):
        raise ValueError("unknown or invalid semantic record")
    header = record["header"]
    _header_contract(header)
    if header.get("format_version") != 10:
        raise ValueError("semantic header version mismatch")
    for name, value in _projection(header).items():
        if type(summary.get(name)) is not type(value) or summary[name] != value:
            raise ValueError(f"container header disagrees with sealed field {name!r}")
    payloads = {k: v for k, v in tensors.items() if k != SEMANTICS_TENSOR}
    codecs = record["tensor_codecs"]
    if not isinstance(codecs, dict):
        raise ValueError("invalid tensor codec declarations")
    for name, spec in codecs.items():
        if not isinstance(spec, dict) or spec.get("c") not in {"arange", "delta", "exact-array"}:
            raise ValueError("unknown tensor codec declaration")
        if (spec["c"] == "arange" and name in payloads) or (spec["c"] != "arange" and name not in payloads):
            raise ValueError("tensor codec has ambiguous or missing payload")
        if spec["c"] == "exact-array":
            if set(spec) != {"c"}:
                raise ValueError("unknown exact-array codec fields")
            continue
        dtype = np.dtype(spec.get("dtype"))
        if dtype.kind not in "iu" or dtype.str != dtype.newbyteorder("<").str:
            raise ValueError("invalid structural codec dtype")
        if spec["c"] == "arange":
            if (set(spec) != {"c", "start", "n", "dtype"} or type(spec["start"]) is not int
                    or type(spec["n"]) is not int or spec["n"] < 0 or spec["n"]*dtype.itemsize > 256*1024*1024):
                raise ValueError("invalid or oversized structural arange")
            limits = np.iinfo(dtype)
            if spec["n"] and not limits.min <= spec["start"] <= spec["start"]+spec["n"]-1 <= limits.max:
                raise ValueError("structural arange exceeds its integer dtype")
        else:
            cols = spec.get("cols")
            shape = np.asarray(payloads[name]).shape
            width = 1 if len(shape) == 1 else shape[1] if len(shape) == 2 else 0
            if (set(spec) != {"c", "cols", "dtype"} or not isinstance(cols, list)
                    or any(type(i) is not int or not 0 <= i < width for i in cols) or len(set(cols)) != len(cols)):
                raise ValueError("invalid structural delta columns")
    component_registry().validate(record["components"], header, payloads, codecs)
    from .state import RexState
    for entry in header.get("nested", []):
        prefix = f"nested/{entry['group']}/{entry['j']}/"
        child = {k[len(prefix):]: v for k, v in payloads.items() if k.startswith(prefix)}
        _record(RexState(child, entry["header"]), verify=verify)
    return record, payloads


def verify_sealed_state(state):
    try:
        _record(state)
        return True
    except (KeyError, ValueError, TypeError, OverflowError, UnicodeError):
        return False


def open_state(state, *, verify=True):
    """Checked reconstruction view; interpretation comes only from sealed bytes."""
    from .state import CODEC_TENSOR, RexState
    record, tensors = _record(state, verify=verify)
    header = dict(record["header"])
    # This private logical view retains the compatibility version expected by
    # framing helpers. Native from_state dispatches it to the component registry.
    header["format_version"] = 9
    if record["tensor_codecs"]:
        tensors[CODEC_TENSOR] = np.frombuffer(pack_value(record["tensor_codecs"]), np.uint8).copy()
    return RexState(tensors, header)


def to_sealed_state(rex):
    from .state import to_state
    return to_state(rex, _native=True)


def migrate_state(state):
    """One way decode/re-encode; no patching of legacy bytes or inferred declarations."""
    from .state import from_state
    result = to_sealed_state(from_state(state))
    return result


def state_identity(state):
    _record(state)
    return state.header["digest"]
