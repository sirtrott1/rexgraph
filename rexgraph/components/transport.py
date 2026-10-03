"""Cell basis transport owned by the same components as sealed native state.

All maps run from old indices to new indices, with -1 meaning removed. The
native edit owns the new primary support; these hooks carry its declarations
and attachments without deriving replacement values from mathematical views.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

import numpy as np


class CellMaps(Mapping):
    """Checked injective old to new maps. Upper grade owners may extend the tower."""

    def __init__(self, maps, *, required_grades=(0, 1, 2)):
        self._maps = {}
        for grade, values in maps.items():
            self.add(grade, values)
        if not set(required_grades) <= self._maps.keys():
            raise ValueError("cell transport requires maps for every declared grade")

    def add(self, grade, values):
        a = np.asarray(values)
        if type(grade) is not int or grade < 0 or a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
            raise ValueError("cell maps require nonnegative grades and integral vectors")
        if a.size and (int(a.min()) < -1 or int(a.max()) >= 2**31):
            raise ValueError("cell map is outside the native index domain")
        a = np.asarray(a, dtype=np.int64)
        surviving = a[a >= 0]
        if len(np.unique(surviving)) != len(surviving):
            raise ValueError("cell map must be injective on surviving cells")
        self._maps[grade] = np.frombuffer(a.tobytes(), dtype=np.int64)

    def __getitem__(self, grade):
        return self._maps[grade]

    def __iter__(self):
        return iter(self._maps)

    def __len__(self):
        return len(self._maps)

    def take(self, grade, size):
        mapping = self[grade]
        alive = np.flatnonzero(mapping >= 0)
        if alive.size and int(mapping[alive].max()) >= size:
            raise ValueError("cell map exceeds its target basis")
        if len(alive) != size or (size and not np.array_equal(np.sort(mapping[alive]), np.arange(size))):
            raise ValueError("restriction map must cover its target cell basis")
        return alive[np.argsort(mapping[alive], kind="stable")]


def _support_capture(rex):
    return rex._graded_duals, bool(rex._face_grade or rex._nF)


def _support_remap(value, rex, maps, mode):
    from rexgraph.native_sparse import csr_carrier, sparse_arrays
    upper, face_grade = value
    rex._face_grade = bool(face_grade or rex._nF)
    if mode == "append":
        output = []
        for grade, matrix in enumerate(upper or (), 3):
            ptr, columns, data, shape = sparse_arrays(matrix)
            if shape[0] != len(maps[grade-1]):
                raise ValueError("upper boundary shape does not match its source basis")
            new_lower = rex._nF if grade == 3 else shape[0]
            if new_lower < shape[0]:
                raise ValueError("append cannot shrink the upper boundary row domain")
            rows = np.pad(ptr, (0, new_lower-shape[0]), constant_values=int(ptr[-1]))
            output.append(csr_carrier(rows, columns, data, (new_lower, shape[1])))
            maps.add(grade, np.arange(shape[1], dtype=np.int64))
        rex._graded_duals = None if upper is None else output
        return
    output = []
    for grade, matrix in enumerate(upper or (), 3):
        ptr, columns, data, shape = sparse_arrays(matrix)
        lower = maps[grade-1]
        if shape[0] != len(lower):
            raise ValueError("upper boundary shape does not match its lower cell map")
        alive = np.ones(shape[1], dtype=bool)
        for row in np.flatnonzero(lower < 0):
            start, stop = int(ptr[row]), int(ptr[row+1])
            alive[columns[start:stop][data[start:stop] != 0]] = False
        if grade in maps:
            current = maps[grade]
            if len(current) != shape[1] or np.any((current >= 0) & ~alive):
                raise ValueError("selected upper cells do not retain their full boundary")
            maps.take(grade, int(np.count_nonzero(current >= 0)))
            alive = current >= 0
        else:
            current = np.full(shape[1], -1, dtype=np.int64)
            current[alive] = np.arange(np.count_nonzero(alive))
            maps.add(grade, current)
        kept = np.flatnonzero(lower >= 0)
        kept = kept[np.argsort(lower[kept], kind="stable")]
        if kept.size and not np.array_equal(np.sort(lower[kept]), np.arange(len(kept))):
            raise ValueError("upper boundary transport requires a contiguous lower basis")
        out_ptr, out_idx, out_data = [0], [], []
        for row in kept:
            start, stop = int(ptr[row]), int(ptr[row+1])
            cols, vals = columns[start:stop], data[start:stop]
            keep = alive[cols]
            out_idx.extend(current[cols[keep]])
            out_data.extend(vals[keep])
            out_ptr.append(len(out_idx))
        output.append(csr_carrier(np.asarray(out_ptr, dtype=ptr.dtype), np.asarray(out_idx, dtype=columns.dtype),
                                  np.asarray(out_data, dtype=data.dtype), (len(kept), int(alive.sum()))))
    rex._graded_duals = None if upper is None else output


def _declaration_capture(rex):
    return rex._boundary_ptr, rex._boundary_idx, rex._declaration


def _declaration_remap(value, rex, maps, mode):
    from rexgraph.column import ColumnDeclaration
    ptr, old_idx, declaration = value
    alive = np.flatnonzero(maps[1] >= 0)
    if len(maps[1]) != len(ptr)-1:
        raise ValueError("relation map is not aligned with its source support")
    order = alive[np.argsort(maps[1][alive], kind="stable")]
    new = maps[1][order]
    if not np.array_equal(np.diff(ptr)[order], np.diff(rex._boundary_ptr)[new]):
        raise ValueError("transport changed a declared relation's arity")
    if np.array_equal(order, alive):
        slots = old_idx[np.repeat(maps[1] >= 0, np.diff(ptr))]
    else:
        chunks = [old_idx[ptr[i]:ptr[i+1]] for i in order]
        slots = np.concatenate(chunks) if chunks else np.empty(0, np.int32)
    if (slots.size and (slots.min() < 0 or slots.max() >= len(maps[0]))) or not np.array_equal(
            maps[0][slots], rex._boundary_idx[:len(slots)]):
        raise ValueError("cell maps disagree with the target relation support")
    if declaration is None:
        rex._declaration = None
        return
    if mode == "append" and len(declaration.head_slot) == rex._nE:
        return
    heads = np.zeros(rex._nE, np.int32)
    num = np.zeros(len(rex._boundary_idx), np.int64)
    den = np.zeros(len(rex._boundary_idx), np.int64)
    for old, new in enumerate(maps[1]):
        if new < 0:
            continue
        a, b = map(int, ptr[old:old+2])
        start, stop = map(int, rex._boundary_ptr[new:new+2])
        if b-a != stop-start:
            raise ValueError("transport changed a declared relation's arity")
        heads[new] = declaration.head_slot[old]
        num[start:stop] = declaration.share_num[a:b]
        den[start:stop] = declaration.share_den[a:b]
    rex._declaration = None if not heads.any() and not den.any() else ColumnDeclaration(heads, num, den)


def _weights_capture(rex):
    presence = rex._weight_presence
    if presence is None and rex._w_E is not None:
        presence = np.ones(len(rex._w_E), bool)
    return rex._w_E, presence, rex._signs, rex._w_boundary


def _weights_remap(value, rex, maps, mode):
    if mode == "append":
        return  # primary append batches already own their weights/presence/signs
    weights, presence, signs, boundary = value
    take = maps.take(1, rex._nE)
    rex._w_E = None if weights is None else np.asarray(weights)[take].copy()
    rex._weight_presence = None if presence is None else np.asarray(presence)[take].copy()
    if rex._weight_presence is not None and not rex._weight_presence.any():
        rex._w_E = rex._weight_presence = None
    rex._signs = None if signs is None else np.asarray(signs)[take].copy()
    output = {}
    for key, feature in boundary.items():
        edge = key[0] if isinstance(key, tuple) else key
        if not 0 <= edge < len(maps[1]):
            raise ValueError("boundary attribution is outside its relation basis")
        new = int(maps[1][edge])
        if new < 0:
            continue
        if isinstance(key, tuple):
            vertex = key[1]
            if not 0 <= vertex < len(maps[0]):
                raise ValueError("boundary attribution is outside its vertex basis")
            vertex = int(maps[0][vertex])
            if vertex < 0:
                continue
            key = (new, vertex)
        else:
            key = new
        output[key] = _clone_value(feature)
    rex._w_boundary = output


def _identity_capture(rex):
    return rex._relation_ids, getattr(rex, "_agent_meta", {}), getattr(rex, "_embedding", None)


def _identity_remap(value, rex, maps, mode):
    ids, metadata, embedding = value
    if mode != "append":
        take = maps.take(1, rex._nE)
        rex._relation_ids = None if ids is None else np.asarray(ids)[take].copy()
    metadata = {} if mode == "structural" else _clone_value(metadata)
    if mode == "structural":
        embedding = None
    if "vertex_labels" in metadata:
        old = metadata["vertex_labels"]
        if len(old) != len(maps[0]):
            raise ValueError("vertex labels are not aligned with their source basis")
        labels = [""] * rex._nV
        for i, j in enumerate(maps[0]):
            if j >= 0:
                labels[j] = old[i]
        metadata["vertex_labels"] = labels
    rex._agent_meta = metadata
    if embedding is not None:
        from rexgraph.value import Absent
        if len(embedding) != len(maps[0]):
            raise ValueError("embedding is not aligned with its source vertex basis")
        points = [Absent] * rex._nV
        for i, j in enumerate(maps[0]):
            if j >= 0:
                points[j] = _clone_value(embedding[i])
        rex._embedding = points


def _clone_value(value, *, _checked=False):
    # Only closed values are copied. No deepcopy/pickle hooks on caller objects.
    from rexgraph.value_codec import pack_value
    if not _checked:
        pack_value(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_value(item, _checked=True) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [_clone_value(item, _checked=True) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _attribute_kind(value):
    from rexgraph.graph import RexGraph
    from rexgraph.field_codec import FIELD_VALUES
    from rexgraph.section_codec import SECTION_VALUES
    from rexgraph.model_codec import MODEL_VALUES
    from rexgraph.span import SpanAttachment
    if isinstance(value, RexGraph):
        return "nested"
    for kind, classes in (("field", FIELD_VALUES), ("section", SECTION_VALUES), ("model", MODEL_VALUES), ("span", (SpanAttachment,))):
        if isinstance(value, classes):
            return kind
    return "attributes"


def _attribute_capture(kind):
    def capture(rex):
        output = {}
        for grade, cells in getattr(rex, "_cell_metadata", {}).items():
            for index, attributes in cells.items():
                for key, value in attributes.items():
                    if _attribute_kind(value) == kind:
                        output.setdefault(grade, {}).setdefault(index, {})[key] = _clone_attachment(kind, value)
        return output
    return capture


def _clone_attachment(kind, value):
    if kind == "attributes":
        return _clone_value(value)
    if kind == "nested":
        from rexgraph.state import to_state, from_state
        return from_state(to_state(value))
    if kind == "model":
        value.check_state()
        return value  # model trees are immutable retained values
    if kind == "field":
        from rexgraph.field_codec import pack_field, unpack_field
        return unpack_field(pack_field(value, native=True))
    if kind == "section":
        from rexgraph.section_codec import pack_section, unpack_section
        return unpack_section(pack_section(value, native=True))
    from rexgraph.span_codec import pack_attachment, unpack_attachment
    return unpack_attachment(pack_attachment(value, native=True))


def _attribute_remap(kind):
    def remap(value, rex, maps, mode):
        store = getattr(rex, "_cell_metadata", None)
        if store is None:
            store = rex._cell_metadata = {}
        for grade, cells in value.items():
            if grade not in maps:
                # Append changes only lower bases; higher attachments stay put.
                if mode == "append":
                    for index, attributes in cells.items():
                        store.setdefault(grade, {}).setdefault(index, {}).update(attributes)
                    continue
                raise ValueError(f"no carried cell map for metadata grade {grade}")
            for old, attributes in cells.items():
                if not 0 <= old < len(maps[grade]):
                    raise ValueError("attribute cell index is outside its source basis")
                new = int(maps[grade][old])
                if new >= 0:
                    store.setdefault(grade, {}).setdefault(new, {}).update(attributes)
    return remap


def _signals_capture(rex):
    return getattr(rex, "_signals", None)


def _signals_remap(value, rex, maps, mode):
    if value is None:
        return
    if not isinstance(value, np.ndarray) or not value.ndim or value.shape[0] != len(maps[1]):
        raise ValueError("user signals must be aligned with their relation basis")
    if mode == "append" and rex._nE > value.shape[0]:
        from rexgraph.exact_array import ExactArray
        from rexgraph.value import Absent, NumberRule
        old = ExactArray.from_values(value, rule=NumberRule.BINARY_EXACT).values()
        output = np.full((rex._nE, *value.shape[1:]), Absent, dtype=object)
        output[maps[1]] = old
    else:
        output = value[maps.take(1, rex._nE)].copy()
    rex._signals = output


def _sectioning_capture(rex):
    return getattr(rex, "_sectionings", {})


def _sectioning_remap(value, rex, maps, mode):
    output = {}
    for name, section in value.items():
        if section.grade not in maps:
            if mode == "append":
                output[name] = deepcopy(section)
                continue
            raise ValueError("no carried map for sectioning grade")
        mapping = maps[section.grade]
        section = deepcopy(section)
        size = {0: rex._nV, 1: rex._nE, 2: rex._nF}.get(section.grade, int(np.count_nonzero(mapping >= 0)))
        section.n_cells = size
        if not section.is_derived:
            ptr, indices = [0], []
            for i in range(section.n_sections):
                old = section.cells(i)
                if old.size and (old.min() < 0 or old.max() >= len(mapping)):
                    raise ValueError("section membership exceeds its source basis")
                members = mapping[old]
                indices.extend(members[members >= 0])
                ptr.append(len(indices))
            section.indptr = np.asarray(ptr, np.int64)
            section.indices = np.asarray(indices, np.int64)
        output[name] = section
    rex._sectionings = output


def _relations_capture(rex):
    return getattr(rex, "_relation_source", None)


def _relations_remap(value, rex, maps, mode):
    if value is not None and mode != "append":
        rex._relation_source = value.remap_to(rex, maps)


def transport_hooks(name):
    if name in {"attributes", "field", "section", "model", "span", "nested"}:
        return _attribute_capture(name), _attribute_remap(name)
    return {"support": (_support_capture, _support_remap),
            "declaration": (_declaration_capture, _declaration_remap),
            "weights": (_weights_capture, _weights_remap),
            "identity": (_identity_capture, _identity_remap),
            "relations": (_relations_capture, _relations_remap),
            "signals": (_signals_capture, _signals_remap),
            "sectioning": (_sectioning_capture, _sectioning_remap)}[name]
