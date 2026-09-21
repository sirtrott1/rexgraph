"""Exact tensor encoding for section recipes and completion certificates."""
from __future__ import annotations

import json
import numpy as np

from rexgraph.section_calculus import SectionRecipe, SectionFamily
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.type_accession import CoordinateSpace
from rexgraph.tensor_field import FieldSource
from rexgraph.io.field_state import pack_field, unpack_field

SECTION_VALUES = (SectionRecipe, SectionFamily)


def _space(value):
    return {"name": value.name, "keys": value.keys}


def pack_section(value):
    """Store finite declarations and exact values, never executable Python."""
    tensors = {}

    def put_map(action, path):
        tensors[path + "indices"] = np.asarray([(i, j) for i, j, _ in action.entries], dtype=np.int64).reshape(-1, 2)
        tensors[path + "values"] = np.asarray([v for _, _, v in action.entries], dtype=object)
        return {"domain": _space(action.domain), "codomain": _space(action.codomain), "path": path}

    def put_field(value, path):
        for key, tensor in pack_field(value).items():
            tensors[path + key] = tensor
        return path

    if isinstance(value, SectionRecipe):
        value = value.detached()
        spec = {"kind": "recipe", "version": 1, "name": value.name, "grade": value.grade,
                "cells": value.cells, "mediators": value.mediators,
                "stalk_spaces": [_space(s) for s in value.stalk_spaces],
                "mediator_spaces": [_space(s) for s in value.mediator_spaces],
                "incidences": value.incidences,
                "source": value.source.as_record(),
                "dependencies": [s.as_record() for s in value.dependencies],
                "stalk_sources": [None if s is None else s.as_record() for s in value.stalk_sources],
                "restrictions": [{"cell": c, "mediator": m, "action": put_map(a, f"restriction/{i}/")}
                                 for i, (c, m, a) in enumerate(value.restrictions)]}
    elif isinstance(value, SectionFamily):
        value.check_state()
        spec = {"kind": "family", "version": 1, "declaration": value.declaration_digest,
                "equations": put_map(value.equations, "equations/"),
                "directions": put_map(value.directions, "directions/"),
                "rhs": put_field(value.rhs, "rhs/"),
                "particular": put_field(value.particular, "particular/")}
    else:
        raise TypeError("unsupported section declaration")
    spec["digest"] = value.coefficient_digest
    tensors["spec"] = np.frombuffer(json.dumps(spec, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode(), dtype=np.uint8).copy()
    return tensors


def unpack_section(tensors):
    """Restore detached equations and verify their mathematical certificate."""
    def space(value):
        return CoordinateSpace(value["name"], tuple(value["keys"]))

    def source(value):
        return FieldSource(None, value["record_id"], value["version"], value["state_digest"])

    def take_map(value):
        path = value["path"]
        indices = np.asarray(tensors[path + "indices"])
        values = np.asarray(tensors[path + "values"], dtype=object)
        if values.ndim != 1 or indices.shape != (len(values), 2) or indices.dtype.kind not in "iu":
            raise ValueError("invalid section map tensors")
        return CoordinateMap(space(value["domain"]), space(value["codomain"]),
                             tuple((int(i), int(j), v) for (i, j), v in zip(indices, values, strict=True)))

    def take_field(path):
        return unpack_field({k[len(path):]: v for k, v in tensors.items() if k.startswith(path)})

    try:
        raw = np.asarray(tensors["spec"])
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("invalid section specification tensor")
        spec = json.loads(raw.tobytes().decode())
        if spec.get("version") != 1:
            raise ValueError("unsupported section schema version")
        if spec["kind"] == "recipe":
            value = SectionRecipe(spec["name"], spec["grade"], tuple(spec["cells"]), tuple(spec["mediators"]),
                    tuple(space(s) for s in spec["stalk_spaces"]), tuple(space(s) for s in spec["mediator_spaces"]),
                    tuple(tuple(ms) for ms in spec["incidences"]),
                    tuple((v["cell"], v["mediator"], take_map(v["action"])) for v in spec["restrictions"]),
                    source(spec["source"]), tuple(source(s) for s in spec["dependencies"]),
                    tuple(None if s is None else source(s) for s in spec["stalk_sources"]))
        elif spec["kind"] == "family":
            value = SectionFamily(take_map(spec["equations"]), take_field(spec["rhs"]),
                                  take_field(spec["particular"]), take_map(spec["directions"]), spec["declaration"])
        else:
            raise ValueError("unknown section schema kind")
        if value.coefficient_digest != spec["digest"]:
            raise ValueError("section declaration digest mismatch")
        return value
    except (KeyError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid section tensor payload") from exc
