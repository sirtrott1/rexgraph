"""Finite model state encoding without executable objects or pickle."""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import json
import numpy as np

from rexgraph.model_state import ModelState, ModelOutput, ModelBatch, ModelTimeline, ModelInput
from rexgraph.tensor_field import FieldSource
from rexgraph.type_accession import CoordinateSpace

MODEL_VALUES = (ModelState, ModelOutput, ModelBatch, ModelTimeline, ModelInput)


def model_payload(value):
    tensors = {}
    def put(v):
        if isinstance(v, ModelInput):
            return ["input", put(v.original), put(v.values), [v.space.name, list(v.space.keys)],
                    [[a.name, list(a.keys)] for a in v.axes], v.source.as_record(),
                    v.origin_digest, v.origin_arithmetic, [d.as_record() for d in v.dependencies]]
        if isinstance(v, np.ndarray):
            name = f"tensor/{len(tensors)}"
            a = np.array(v, copy=True, order="C")
            stored = a.astype(np.uint8) if a.dtype == bool else a
            tensors[name] = stored.reshape(1) if a.ndim == 0 else stored
            return ["array", name, a.dtype.str, list(a.shape)]
        if v is None: return ["none"]
        if isinstance(v, bool): return ["bool", v]
        if isinstance(v, str): return ["str", v]
        if isinstance(v, bytes):
            name = f"tensor/{len(tensors)}"
            tensors[name] = np.frombuffer(v, dtype=np.uint8).copy()
            return ["bytes", name]
        if isinstance(v, int): return ["int", hex(v)]
        if isinstance(v, Fraction): return ["rational", hex(v.numerator), hex(v.denominator)]
        if isinstance(v, float): return ["float", v.hex()]
        if isinstance(v, Mapping):
            items = sorted(v.items(), key=lambda kv: (type(kv[0]).__name__, str(kv[0])))
            return ["map", [[put(k), put(x)] for k, x in items]]
        if isinstance(v, (tuple, list)): return ["tuple", [put(x) for x in v]]
        raise TypeError("unsupported model payload value")
    def space(v): return [v.name, list(v.keys)]
    def source(v): return v.as_record()
    if isinstance(value, ModelState):
        spec = {"kind": "model", "adapter": value.adapter, "adapter_version": value.adapter_version,
                "source": source(value.source), "space": space(value.space),
                "axes": [space(v) for v in value.output_axes], "configuration": put(value.configuration),
                "payload": put(value.payload), "arithmetic": value.arithmetic, "step": value.step,
                "parent": value.parent, "dependencies": [source(v) for v in value.dependencies]}
    elif isinstance(value, ModelOutput):
        spec = {"kind": "output", "source": source(value.source), "space": space(value.space),
                "axes": [space(v) for v in value.axes], "values": put(value.values),
                "model_digest": value.model_digest, "arithmetic": value.arithmetic, "method": value.method,
                "grade": value.grade, "variance": value.variance,
                "dependencies": [source(v) for v in value.dependencies]}
    elif isinstance(value, ModelInput):
        spec = {"kind": "input", "value": put(value)}
    elif isinstance(value, ModelBatch):
        spec = {"kind": "batch", "source": source(value.source), "space": space(value.space),
                "targets": put(value.targets), "observed": put(value.observed), "inputs": put(value.inputs),
                "target_input": put(value.target_input),
                "dependencies": [source(v) for v in value.dependencies]}
    elif isinstance(value, ModelTimeline):
        spec = {"kind": "timeline", "times": put(value.times), "model_digests": list(value.model_digests),
                "source_digests": list(value.source_digests), "time_axis": value.time_axis, "time_unit": value.time_unit,
                "temporal_header": put(value.temporal_header),
                "temporal_tensors": put(value.temporal_tensors)}
    else:
        raise TypeError("unsupported native model object")
    spec["version"] = 1
    tensors["spec"] = np.frombuffer(json.dumps(spec, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":")).encode(), dtype=np.uint8).copy()
    return tensors


def pack_model(value):
    value.check_state()
    tensors = model_payload(value)
    tensors["model_digest"] = np.frombuffer(value.coefficient_digest.encode(), dtype=np.uint8).copy()
    return tensors


def unpack_model(tensors):
    used = {"spec", "model_digest"}
    def take(v):
        if not isinstance(v, list) or not v: raise ValueError("invalid model value")
        tag = v[0]
        if tag == "input" and len(v) == 9:
            return ModelInput(take(v[1]), take(v[2]), space(v[3]), tuple(space(a) for a in v[4]),
                              source(v[5]), v[6], v[7], tuple(source(a) for a in v[8]))
        if (tag == "array" and len(v) == 4) or (tag == "bytes" and len(v) == 2):
            used.add(v[1])
            a = np.asarray(tensors[v[1]])
            if tag == "bytes":
                if a.dtype != np.uint8 or a.ndim != 1: raise ValueError("invalid model bytes")
                return a.tobytes()
            dtype, shape = np.dtype(v[2]), tuple(v[3])
            if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape):
                raise ValueError("invalid model tensor shape")
            if dtype == bool:
                if a.dtype != np.uint8 or not np.isin(a, (0, 1)).all():
                    raise ValueError("invalid model boolean tensor")
                a = a.astype(bool)
            elif a.dtype != dtype:
                raise ValueError("model tensor dtype differs")
            return a.reshape(shape)
        if tag == "none" and len(v) == 1: return None
        if tag == "bool" and len(v) == 2 and isinstance(v[1], bool): return v[1]
        if tag == "str" and len(v) == 2 and isinstance(v[1], str): return v[1]
        if tag == "int" and len(v) == 2: return int(v[1], 16)
        if tag == "rational" and len(v) == 3: return Fraction(int(v[1], 16), int(v[2], 16))
        if tag == "float" and len(v) == 2: return float.fromhex(v[1])
        if tag == "tuple" and len(v) == 2: return tuple(take(x) for x in v[1])
        if tag == "map" and len(v) == 2:
            out = {}
            for key, item in v[1]:
                key = take(key)
                if key in out: raise ValueError("duplicate model mapping key")
                out[key] = take(item)
            return out
        raise ValueError("unsupported model value tag")
    def source(v): return FieldSource(None, v["record_id"], v["version"], v["state_digest"])
    def space(v): return CoordinateSpace(v[0], tuple(v[1]))
    try:
        raw = np.asarray(tensors["spec"])
        if raw.dtype != np.uint8 or raw.ndim != 1: raise ValueError("invalid model specification")
        spec = json.loads(raw.tobytes().decode())
        if spec.get("version") != 1: raise ValueError("unsupported model schema")
        kind = spec["kind"]
        if kind == "model":
            result = ModelState(spec["adapter"], spec["adapter_version"], source(spec["source"]), space(spec["space"]),
                tuple(space(v) for v in spec["axes"]), take(spec["configuration"]), take(spec["payload"]),
                spec["arithmetic"], spec["step"], spec["parent"], tuple(source(v) for v in spec["dependencies"]))
        elif kind == "output":
            result = ModelOutput(take(spec["values"]), space(spec["space"]), tuple(space(v) for v in spec["axes"]),
                source(spec["source"]), spec["model_digest"], spec["arithmetic"], spec["method"],
                tuple(source(v) for v in spec["dependencies"]), spec["grade"], spec["variance"])
        elif kind == "input":
            result = take(spec["value"])
        elif kind == "batch":
            result = ModelBatch(source(spec["source"]), space(spec["space"]), take(spec["targets"]),
                take(spec["observed"]), take(spec["inputs"]), tuple(source(v) for v in spec["dependencies"]), take(spec["target_input"]))
        elif kind == "timeline":
            header = take(spec["temporal_header"])
            def lists(v):
                if isinstance(v, dict): return {k: lists(x) for k, x in v.items()}
                if isinstance(v, tuple): return [lists(x) for x in v]
                return v
            result = ModelTimeline(take(spec["times"]), tuple(spec["model_digests"]), tuple(spec["source_digests"]),
                                   lists(header), take(spec["temporal_tensors"]), spec["time_axis"], spec["time_unit"])
        else: raise ValueError("unknown model object kind")
        encoded_digest = np.asarray(tensors["model_digest"])
        if encoded_digest.dtype != np.uint8 or encoded_digest.ndim != 1: raise ValueError("invalid model seal")
        if set(tensors) != used: raise ValueError("unclaimed model tensor")
        if result.coefficient_digest != encoded_digest.tobytes().decode():
            raise ValueError("model identity mismatch")
        return result
    except (KeyError, UnicodeError, TypeError, IndexError, json.JSONDecodeError, ZeroDivisionError) as exc:
        raise ValueError("invalid model tensor payload") from exc
