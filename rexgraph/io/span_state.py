"""Versioned tensor encoding for native span attachments."""
from __future__ import annotations

import json
import numpy as np

from rexgraph.coordinate_map import CoordinateMap
from rexgraph.span import SpanAttachment, SpanBlock


def pack_attachment(value):
    """Return raw tensors for the canonical RexState codec."""
    if not isinstance(value, SpanAttachment):
        raise TypeError("span encoding requires a SpanAttachment")
    spec = {"version": 1, "annotation_id": value.annotation_id, "owner_id": value.owner_id,
            "role": value.role, "source_id": value.source_id, "qualifiers": value.qualifiers,
            "blocks": {}, "grounding": value.grounding is not None}
    tensors = {}
    for name in ("text", "time"):
        block = getattr(value, name)
        if block is None:
            continue
        spec["blocks"][name] = {"name": block.name, "axis": block.axis, "unit": block.unit,
                                "keys": [key for key, _, _ in block.components],
                                "interpretation": block.interpretation}
        tensors[name + "/endpoints"] = np.asarray([(a, b) for _, a, b in block.components],
                                                   dtype=object).reshape((-1, 2))
    if value.grounding is not None:
        tensors["grounding/indices"] = np.asarray([(i, j) for i, j, _ in value.grounding.entries],
                                                  dtype=np.int64).reshape((-1, 2))
        tensors["grounding/values"] = np.asarray([v for _, _, v in value.grounding.entries], dtype=object)
    tensors["spec"] = np.frombuffer(json.dumps(spec, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8"), dtype=np.uint8).copy()
    return tensors


def unpack_attachment(tensors):
    """Read decoded canonical tensors without evaluating arbitrary objects."""
    try:
        raw = np.asarray(tensors["spec"])
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("invalid span schema tensor")
        spec = json.loads(raw.tobytes().decode("utf-8"))
        if type(spec.get("version")) is not int or spec["version"] != 1:
            raise ValueError("unsupported span attachment version")
        if set(spec["blocks"]) - {"text", "time"} or type(spec["grounding"]) is not bool:
            raise ValueError("invalid span attachment schema")
        blocks = {}
        for name, desc in spec["blocks"].items():
            endpoints = np.asarray(tensors[name + "/endpoints"])
            if endpoints.shape != (len(desc["keys"]), 2):
                raise ValueError("span endpoint count does not match component keys")
            blocks[name] = SpanBlock(desc["name"], desc["axis"], desc["unit"],
                                     tuple((key, a, b) for key, (a, b) in
                                           zip(desc["keys"], endpoints, strict=True)), desc["interpretation"])
        grounding = None
        if spec["grounding"]:
            indices = np.asarray(tensors["grounding/indices"])
            values = np.asarray(tensors["grounding/values"])
            if indices.dtype.kind not in "iu" or indices.shape != (values.size, 2) or values.ndim != 1:
                raise ValueError("invalid grounding coordinate tensors")
            grounding = CoordinateMap(blocks["text"].coordinates, blocks["time"].coordinates,
                                       tuple((int(i), int(j), v) for (i, j), v in
                                             zip(indices, values, strict=True)))
        return SpanAttachment(spec["annotation_id"], spec["owner_id"], spec["role"], spec["source_id"],
                              blocks.get("text"), blocks.get("time"), grounding, tuple(spec["qualifiers"]))
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid span attachment tensors") from exc
