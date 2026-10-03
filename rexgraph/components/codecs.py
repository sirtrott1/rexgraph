"""Native component encoders and decoders; legacy framing stays in state.py.

The attribute owner owns cell addresses and the column directory. Retained value
owners consume those checked addresses and own their namespaced payloads. This
keeps the existing v10 tensor ownership and semantic identities compatible.
"""
from __future__ import annotations

from functools import cached_property
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from . import ComponentPayload, DecodedComponent


def _bytes(value):
    from rexgraph.value_codec import pack_value
    return np.frombuffer(pack_value(value), np.uint8).copy()


def _value(tensor):
    from rexgraph.value_codec import unpack_value
    a = np.asarray(tensor)
    if a.dtype != np.uint8 or a.ndim != 1:
        raise ValueError("native value record requires a byte vector")
    return unpack_value(a.tobytes())


class EncodeContext:
    def __init__(self, rex):
        self.rex = rex

    @cached_property
    def columns(self):
        metadata = getattr(self.rex, "_cell_metadata", {})
        if not any(values for cells in metadata.values() for values in cells.values()):
            return []
        from .transport import _attribute_kind
        from rexgraph.tensor_moment import TensorMomentKernel, TensorMoments
        from rexgraph.native_field import NativeFieldCalculus
        from rexgraph.temporal_field import TensorEvolution, ResolvedEvolution, NativeFieldEvolution, SectorTransport
        from rexgraph.section_calculus import SectionSystem, SectionImage
        grouped = {}
        for grade, cells in metadata.items():
            for index, values in cells.items():
                for key, value in values.items():
                    grouped.setdefault((grade, key), []).append((index, value))
        result = []
        for (grade, key), pairs in sorted(grouped.items()):
            pairs.sort(key=lambda pair: pair[0])
            values = [value for _, value in pairs]
            if any(isinstance(v, (TensorMomentKernel, TensorMoments, NativeFieldCalculus,
                                  TensorEvolution, ResolvedEvolution, NativeFieldEvolution, SectorTransport)) for v in values):
                raise TypeError("live field declarations require an explicit retained result before storage")
            if any(isinstance(v, (SectionSystem, SectionImage)) for v in values):
                raise TypeError("store a section recipe or its evaluated completion, not a live action")
            kinds = {_attribute_kind(value) for value in values}
            if len(kinds) != 1:
                raise TypeError("an attribute column cannot mix retained states and generic values")
            kind = kinds.pop()
            result.append((int(grade), key, np.asarray([i for i, _ in pairs], np.int64), values, kind))
        return result


class DecodeContext:
    def __init__(self, header):
        self.header = MappingProxyType(dict(header))
        self.attribute_indices = {}


def _support_encode(rex, context):
    from rexgraph.native_sparse import sparse_arrays
    tensors = {"boundary_ptr": rex._boundary_ptr.copy(), "boundary_idx": rex._boundary_idx.copy()}
    header = {"format_version": 2, "object_type": "RexGraph", "nV": int(rex._nV),
              "nE": int(rex._nE), "nF": int(rex._nF), "directed": bool(rex._directed),
              "g_channel": rex.g_channel, "c_channel": rex.c_channel}
    if rex._nF:
        tensors.update(B2_col_ptr=rex._B2_col_ptr.copy(), B2_row_idx=rex._B2_row_idx.copy(), B2_vals=rex._B2_vals.copy())
    elif rex._face_grade and not rex._graded_duals:
        header["empty_face_grade"] = True
    if rex._graded_duals:
        header.update(n_graded_duals=len(rex._graded_duals), graded_shapes=[])
        for i, matrix in enumerate(rex._graded_duals):
            ptr, indices, data, shape = sparse_arrays(matrix)
            tensors.update({f"gd{i}_indptr": ptr.copy(), f"gd{i}_indices": indices.copy(), f"gd{i}_data": data.copy()})
            header["graded_shapes"].append(list(map(int, shape)))
    return ComponentPayload(tensors, header)


def _support_decode(payload, context):
    from rexgraph.native_sparse import csr_carrier
    t, h = payload.tensors, payload.header
    if not {"boundary_ptr", "boundary_idx"} <= t.keys():
        raise ValueError("native support is missing its primary boundary")
    faces = {"B2_col_ptr", "B2_row_idx", "B2_vals"} & t.keys()
    if faces and len(faces) != 3:
        raise ValueError("native face boundary requires all three carriers")
    arguments = {name: t[name] for name in ("boundary_ptr", "boundary_idx", *sorted(faces))}
    arguments.update(directed=h["directed"], g_channel=h["g_channel"], c_channel=h["c_channel"])
    upper, lower = [], h["nF"]
    for i, shape in enumerate(h.get("graded_shapes", [])):
        if len(shape) != 2 or shape[0] != lower:
            raise ValueError("native upper boundary has incompatible grade axes")
        upper.append(csr_carrier(t[f"gd{i}_indptr"], t[f"gd{i}_indices"], t[f"gd{i}_data"], tuple(shape)))
        lower = shape[1]

    def restore(rex):
        if h["nV"] < rex._nV or h["nE"] != rex._nE or h["nF"] != rex._nF:
            raise ValueError("sealed dimensions disagree with the reconstructed carrier")
        rex._ensure_vertex_count(h["nV"])
        rex._face_grade = bool(h["nF"] or upper or h.get("empty_face_grade", False))
        if upper:
            rex._graded_duals = upper
    return DecodedComponent(arguments, restore)


def _declaration_encode(rex, context):
    declaration = rex._declaration
    return ComponentPayload({} if declaration is None else {
        "column_head": declaration.head_slot.copy(), "column_share_num": declaration.share_num.copy(),
        "column_share_den": declaration.share_den.copy()})


def _declaration_decode(payload, context):
    t = payload.tensors
    if not t:
        return DecodedComponent()
    if set(t) != {"column_head", "column_share_num", "column_share_den"}:
        raise ValueError("native column declaration requires head, numerator and denominator")
    head, num, den = (np.asarray(t[key]) for key in ("column_head", "column_share_num", "column_share_den"))
    if (head.dtype.kind not in "iu" or num.dtype.kind not in "iu" or den.dtype.kind not in "iu"
            or head.shape != (context.header["nE"],) or num.ndim != 1 or num.shape != den.shape
            or np.any(den < 0) or np.any((den == 0) & (num != 0))):
        raise ValueError("invalid native column declaration")
    return DecodedComponent({"head_slot": head, "shares": [None if d == 0 else Fraction(int(n), int(d)) for n, d in zip(num, den, strict=True)]})


def _weights_encode(rex, context):
    t = {}
    if rex._w_E is not None:
        t["w_E"] = np.asarray(rex._w_E).copy()
        presence = rex._weight_presence
        if presence is not None and not presence.all():
            t["weight_presence"] = np.asarray(presence, np.uint8)
    if rex._signs is not None:
        t["signs"] = np.asarray(rex._signs, np.float64).copy()
    if rex._w_boundary:
        t["wb_record"] = _bytes(rex._w_boundary)
    return ComponentPayload(t)


def _weights_decode(payload, context):
    t = payload.tensors
    arguments = {key: t[key] for key in ("w_E", "signs") if key in t}
    if "wb_record" in t:
        if set(t) & {"wb_keys", "wb_offsets", "wb_values", "wb_scalar"}:
            raise ValueError("native boundary attribution has conflicting carriers")
        weights = _value(t["wb_record"])
        if not isinstance(weights, dict):
            raise ValueError("native boundary attribution requires a mapping")
        arguments["w_boundary"] = weights
    elif "wb_keys" in t:
        from rexgraph.state import _unpack_w_boundary
        arguments["w_boundary"] = _unpack_w_boundary(t["wb_keys"], t["wb_offsets"], t["wb_values"], t.get("wb_scalar"))
    elif set(t) & {"wb_offsets", "wb_values", "wb_scalar"}:
        raise ValueError("incomplete native boundary attribution")

    def restore(rex):
        if "weight_presence" not in t:
            return
        presence = np.asarray(t["weight_presence"])
        if (presence.dtype != np.uint8 or presence.shape != (rex._nE,) or not np.isin(presence, (0, 1)).all()
                or presence.all() or not presence.any() or rex._w_E is None):
            raise ValueError("invalid native weight presence carrier")
        if any(value != 1 for value in rex._w_E[presence == 0]):
            raise ValueError("absent metric does not have its declared unit view")
        rex._weight_presence = presence.astype(bool)
    return DecodedComponent(arguments, restore)


def _identity_encode(rex, context):
    from rexgraph.state import _pack_strings
    t, h = {}, {}
    if rex._relation_ids is not None:
        t["relation_ids"] = rex._relation_ids.copy()
    meta = getattr(rex, "_agent_meta", {})
    if meta:
        labels = meta.get("vertex_labels")
        if labels:
            t["label_bytes"], t["label_offsets"] = _pack_strings([str(label) for label in labels])
        from .transport import _clone_value
        h["agent_meta"] = _clone_value({key: value for key, value in meta.items() if key != "vertex_labels"})
    return ComponentPayload(t, h)


def _identity_decode(payload, context):
    from rexgraph.state import _unpack_strings
    from .transport import _clone_value
    t = payload.tensors
    meta = _clone_value(payload.header.get("agent_meta", {}))
    if ("label_bytes" in t) != ("label_offsets" in t):
        raise ValueError("native labels require bytes and offsets")
    if "label_bytes" in t:
        meta["vertex_labels"] = _unpack_strings(t["label_bytes"], t["label_offsets"])
        # Older native writers also stored partial provenance label lists. Keep
        # those bytes readable; public set_provenance validates new basis labels.
    return DecodedComponent({"relation_ids": t["relation_ids"]} if "relation_ids" in t else {},
                            lambda rex: setattr(rex, "_agent_meta", meta) if meta else None)


def _signals_encode(rex, context):
    value = getattr(rex, "_signals", None)
    return ComponentPayload({"signals": value.copy()} if isinstance(value, np.ndarray) else {})


def _signals_decode(payload, context):
    value = payload.tensors.get("signals")
    return DecodedComponent(restore=None if value is None else lambda rex: setattr(rex, "_signals", value.copy()))


def _attributes_encode(rex, context):
    t, schema = {}, []
    for grade, key, indices, values, kind in context.columns:
        prefix = f"cm_{grade}_{key}"
        if kind == "nested":
            schema.append({"dim": grade, "key": key, "kind": "rex", "idx": indices.tolist()})
            continue
        t[prefix+"_idx"] = indices
        if kind == "attributes":
            from rexgraph.value_codec import pack_value
            encoded = [pack_value(value) for value in values]
            t[prefix+"_valbytes"] = np.frombuffer(b"".join(encoded), np.uint8).copy()
            t[prefix+"_valoffs"] = np.asarray([0, *np.cumsum([len(value) for value in encoded])], np.int64)
        schema.append({"dim": grade, "key": key, "kind": "value" if kind == "attributes" else kind})
    return ComponentPayload(t, {"cell_meta": schema})


def _attributes_decode(payload, context):
    from rexgraph.state import _unpack_strings, _attribute_value
    import json
    t, restored = payload.tensors, []
    for column in payload.header.get("cell_meta", []):
        grade, key, kind = column["dim"], column["key"], column["kind"]
        prefix = f"cm_{grade}_{key}"
        indices = np.asarray(column["idx"] if kind == "rex" else t[prefix+"_idx"])
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu") or np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError("invalid native attribute cell indices")
        context.attribute_indices[(grade, key)] = indices
        if kind in {"field", "section", "model", "span", "rex"}:
            continue
        if kind == "value":
            from rexgraph.value_codec import unpack_value
            buf, offsets = np.asarray(t[prefix+"_valbytes"]), np.asarray(t[prefix+"_valoffs"])
            if (buf.dtype != np.uint8 or buf.ndim != 1 or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
                    or len(offsets) != len(indices)+1 or offsets[0] != 0 or offsets[-1] != len(buf) or np.any(offsets[1:] < offsets[:-1])):
                raise ValueError("invalid native attribute column")
            raw = buf.tobytes()
            values = [unpack_value(raw[int(a):int(b)]) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
        elif kind == "num":
            values = list(map(float, t[prefix+"_val"]))
        else:
            values = _unpack_strings(t[prefix+"_valbytes"], t[prefix+"_valoffs"])
            if kind == "structured":
                values = [_attribute_value(json.loads(value)) for value in values]
        if len(values) != len(indices):
            raise ValueError("native attribute values are not aligned with cell indices")
        restored.extend((grade, int(index), key, value) for index, value in zip(indices, values, strict=True))

    def restore(rex):
        for grade, index, key, value in restored:
            rex.attach_metadata(grade, index, key, value)
    return DecodedComponent(restore=restore)


def _retained_codec(kind):
    if kind == "field":
        from rexgraph.field_codec import pack_field, unpack_field
        return "field", pack_field, unpack_field
    if kind == "section":
        from rexgraph.section_codec import pack_section, unpack_section
        return "section", pack_section, unpack_section
    if kind == "model":
        from rexgraph.model_codec import pack_model, unpack_model
        return "model", pack_model, unpack_model
    from rexgraph.span_codec import pack_attachment, unpack_attachment
    return "annotation", pack_attachment, unpack_attachment


def _retained_encode(kind):
    def encode(rex, context):
        columns = [column for column in context.columns if column[-1] == kind]
        if not columns:
            return ComponentPayload()
        directory, pack, _ = _retained_codec(kind)
        t = {}
        for grade, key, indices, values, owner in columns:
            for j, value in enumerate(values):
                prefix = f"{directory}/cm_{grade}_{key}/{j}/"
                t.update({prefix+name: tensor for name, tensor in pack(value, native=True).items()})
        return ComponentPayload(t)
    return encode


def _retained_decode(kind):
    def decode(payload, context):
        columns = [column for column in context.header.get("cell_meta", []) if column["kind"] == kind]
        if not columns and not payload.tensors:
            return DecodedComponent()
        directory, _, unpack = _retained_codec(kind)
        restored, used = [], set()
        for column in columns:
            grade, key = column["dim"], column["key"]
            for j, index in enumerate(context.attribute_indices[(grade, key)]):
                prefix = f"{directory}/cm_{grade}_{key}/{j}/"
                names = {name for name in payload.tensors if name.startswith(prefix)}
                tensors = {name[len(prefix):]: payload.tensors[name] for name in names}
                restored.append((grade, int(index), key, unpack(tensors)))
                used.update(names)
        if used != set(payload.tensors):
            raise ValueError("unclaimed retained component payloads")

        def restore(rex):
            for grade, index, key, value in restored:
                rex.attach_metadata(grade, index, key, value)
        return DecodedComponent(restore=restore)
    return decode


def _nested_encode(rex, context):
    from rexgraph.state import to_state
    t, entries = {}, []
    for grade, key, indices, values, kind in context.columns:
        if kind == "nested":
            group = f"cm_{grade}_{key}"
            for j, value in enumerate(values):
                state = to_state(value)
                t.update({f"nested/{group}/{j}/"+name: tensor for name, tensor in state.tensors.items()})
                entries.append({"group": group, "j": j, "header": state.header})
    return ComponentPayload(t, {"nested": entries})


def _nested_decode(payload, context):
    from rexgraph.state import RexState, from_state
    restored, used, entries = [], set(), {}
    for entry in payload.header.get("nested", []):
        identity = (entry["group"], entry["j"])
        if identity in entries:
            raise ValueError("duplicate nested state entry")
        entries[identity] = entry
    expected = set()
    for column in context.header.get("cell_meta", []):
        if column["kind"] != "rex":
            continue
        grade, key = column["dim"], column["key"]
        group = f"cm_{grade}_{key}"
        for j, index in enumerate(context.attribute_indices[(grade, key)]):
            identity = (group, j)
            expected.add(identity)
            if identity not in entries:
                raise ValueError("missing nested state entry")
            prefix = f"nested/{group}/{j}/"
            names = {name for name in payload.tensors if name.startswith(prefix)}
            child = from_state(RexState({name[len(prefix):]: payload.tensors[name] for name in names}, entries[identity]["header"]))
            restored.append((grade, int(index), key, child))
            used.update(names)
    if expected != set(entries) or used != set(payload.tensors):
        raise ValueError("unclaimed nested state entries or tensors")

    def restore(rex):
        for grade, index, key, child in restored:
            rex.attach_metadata(grade, index, key, child)
    return DecodedComponent(restore=restore)


def _sectioning_encode(rex, context):
    from rexgraph.sectioning import pack_sectionings
    from rexgraph.merkle import pack_merkle
    t, h = {}, {}
    sections = pack_sectionings(rex, t, h)
    if sections:
        h["sectionings"] = sections
        merkle = pack_merkle(rex, t, h)
        if merkle:
            h["merkle"] = merkle
    return ComponentPayload({name: tensor.copy() for name, tensor in t.items()}, h)


def _sectioning_decode(payload, context):
    from rexgraph.sectioning import unpack_sectionings
    from rexgraph.merkle import unpack_merkle

    def restore(rex):
        tensors = {name: tensor.copy() for name, tensor in payload.tensors.items()}
        if payload.header.get("sectionings"):
            unpack_sectionings(rex, tensors, payload.header)
        if payload.header.get("merkle"):
            unpack_merkle(rex, tensors, payload.header)
    return DecodedComponent(restore=restore)


def _relations_encode(rex, context):
    return ComponentPayload({"relation_record": _bytes(rex.relations.as_state_record(native_ids=rex.relation_ids))})


def _relations_decode(payload, context):
    from rexgraph.relations import Relations
    from rexgraph.value_codec import pack_value
    if "relation_record" not in payload.tensors:
        return DecodedComponent()
    record = _value(payload.tensors["relation_record"])
    if not isinstance(record, dict):
        raise ValueError("invalid relation declaration record")

    def restore(rex):
        declared = Relations.from_record(record, attributes=getattr(rex, "_cell_metadata", {})) if record.get("version") == 1 else Relations.from_state_record(record, rex)
        if declared.n_relations != rex._nE or len(declared.vertices) != rex._nV:
            raise ValueError("relation declaration dimensions disagree with the reconstructed carrier")
        rex._relation_source = declared
        if record["version"] in {1, 2}:
            if rex._weight_presence is not None and not rex._weight_presence.all() and not np.array_equal(rex._weight_presence, declared.weight.presence):
                raise ValueError("weight presence disagrees with retained relation declarations")
            rex._weight_presence = None if rex._w_E is None else declared.weight.presence.copy()
            if rex._weight_presence is not None and not rex._weight_presence.any():
                rex._w_E = rex._weight_presence = None
        if declared.embedding is not None:
            rex.set_embedding(declared.embedding)
        canonical = rex.relations.as_record() if record["version"] == 1 else rex.relations.as_state_record(native_ids=rex.relation_ids, version=record["version"])
        if pack_value(canonical) != pack_value(record):
            raise ValueError("relation declaration disagrees with the reconstructed carrier")
    return DecodedComponent(restore=restore)


def codec_hooks(name):
    if name in {"field", "section", "model", "span"}:
        return _retained_encode(name), _retained_decode(name), ()
    headers = {"support": ("format_version", "object_type", "nV", "nE", "nF", "directed", "g_channel", "c_channel", "n_graded_duals", "graded_shapes", "empty_face_grade"),
               "identity": ("agent_meta",), "attributes": ("cell_meta",), "nested": ("nested",), "sectioning": ("sectionings", "merkle")}
    return globals()[f"_{name}_encode"], globals()[f"_{name}_decode"], headers.get(name, ())
