"""Declared sparse relations, independent of file formats and Agent policy."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral

import numpy as np

from .exact_array import ExactArray
from .value import Absent, NumberRule

__all__ = ["VertexTable", "Relations", "RelationSpec"]


def _copy_attributes(attributes):
    # Model states own frozen trees (including mappingproxy). They are immutable
    # retained values, not pickleable mutable dictionaries to deep copy.
    from .model_codec import MODEL_VALUES
    memo = {}
    for cells in attributes.values():
        for values in cells.values():
            for value in values.values():
                if isinstance(value, MODEL_VALUES):
                    value.check_state()
                    memo[id(value)] = value
    return deepcopy(attributes, memo)


def _integers(values, *, name, dtype="<i8"):
    a = np.asarray(values)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"{name} requires a one-dimensional integral array")
    limits = np.iinfo(dtype)
    if a.size and (int(a.min()) < limits.min or int(a.max()) > limits.max):
        raise ValueError(f"{name} exceeds {dtype}")
    return np.frombuffer(np.asarray(a, dtype=dtype).tobytes(), dtype=dtype)


def _identities(values, size, name):
    result = tuple(Absent for _ in range(size)) if values is None else tuple(values)
    if len(result) != size:
        raise ValueError(f"{name} must have one value per cell")
    seen = set()
    for value in result:
        if value is Absent:
            continue
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} cannot use boolean identity")
        try:
            if value in seen:
                raise ValueError(f"duplicate {name} {value!r}; IDs must be unique")
            seen.add(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be hashable declared values") from exc
    return result


@dataclass(frozen=True)
class VertexTable:
    ids: tuple
    labels: tuple[str, ...] = ()
    aliases: tuple[tuple[str, ...], ...] = ()
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        ids = _identities(self.ids, len(self.ids), "vertex ID")
        labels = tuple(self.labels)
        if not labels:
            labels = tuple("" for _ in ids)
        if len(labels) != len(ids) or any(type(label) is not str for label in labels):
            raise ValueError("vertex labels must be aligned strings")
        aliases = tuple(tuple(a) for a in self.aliases) if self.aliases else tuple(() for _ in ids)
        if len(aliases) != len(ids) or any(type(a) is not str for row in aliases for a in row):
            raise ValueError("vertex aliases must be aligned strings")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "attributes", deepcopy(self.attributes))

    @classmethod
    def for_count(cls, count, *, labels=None):
        if type(count) is not int or not 0 <= count < 2**31:
            raise ValueError("vertex count must fit the native cell domain")
        return cls(tuple(Absent for _ in range(count)), tuple(labels) if labels is not None else ())

    def __len__(self):
        return len(self.ids)

    def as_record(self):
        return {"ids": self.ids, "labels": self.labels, "aliases": self.aliases}


@dataclass(frozen=True, eq=False)
class Relations:
    support_ptr: np.ndarray
    support_idx: np.ndarray
    vertices: VertexTable
    head_slot: np.ndarray | None = None
    share: ExactArray | None = None
    weight: ExactArray | None = None
    sign: np.ndarray | None = None
    relation_id: tuple | None = None
    relation_type: tuple | None = None
    attributes: dict = field(default_factory=dict)
    embedding: object = None
    provenance: dict = field(default_factory=dict)
    share_rule: str = "canonical_for_absent"
    weight_rule: str = "unit_for_absent"
    weight_source: NumberRule = NumberRule.EXACT

    def __post_init__(self):
        ptr = _integers(self.support_ptr, name="relation support pointers")
        idx = _integers(self.support_idx, name="relation support indices")
        if not isinstance(self.vertices, VertexTable):
            raise TypeError("relations require a declared VertexTable")
        if not len(ptr) or ptr[0] != 0:
            raise ValueError("relation support pointers must start at zero")
        if np.any(ptr[1:] < ptr[:-1]):
            raise ValueError("relation support pointers must be nondecreasing")
        if ptr[-1] != len(idx):
            raise ValueError("relation support terminal pointer must match the index count")
        if np.any(ptr[1:] == ptr[:-1]):
            raise ValueError("empty C1 relations are not supported")
        if np.any(idx < 0):
            raise ValueError("relation support contains a negative vertex index")
        if idx.size and int(idx.max()) >= len(self.vertices):
            raise ValueError("relation support exceeds the declared vertex domain")
        if ptr[-1] >= 2**31 or len(ptr) >= 2**31:
            raise ValueError("relation support exceeds the native int32 domain")
        count = len(ptr)-1
        heads = _integers(np.zeros(count, np.int32) if self.head_slot is None else self.head_slot,
                          name="head slots", dtype="<i4")
        signs = _integers(np.ones(count, np.int8) if self.sign is None else self.sign,
                          name="relation signs", dtype="i1")
        if heads.shape != (count,) or signs.shape != (count,) or not np.isin(signs, (-1, 1)).all():
            raise ValueError("head slots and signs must be aligned; signs are -1 or +1")
        share = self.share if self.share is not None else ExactArray.absent(idx.shape)
        weight = self.weight if self.weight is not None else ExactArray.absent((count,))
        if not isinstance(share, ExactArray) or share.shape != idx.shape or not isinstance(weight, ExactArray) or weight.shape != (count,):
            raise ValueError("shares and weights must be aligned exact arrays")
        if self.share_rule not in {"canonical_for_absent", "declared"} or self.weight_rule not in {"unit_for_absent", "declared"}:
            raise ValueError("unknown share or metric absence rule")
        if self.weight_rule == "declared" and not weight.presence.all():
            raise ValueError("declared metric requires every relation weight")
        values = share.values() if share.presence.any() else None
        arities = np.diff(ptr)
        if np.any(heads < 0) or np.any(heads >= arities):
            raise ValueError("head slot is outside its relation")
        loops = np.zeros(count, bool)
        pairs = np.flatnonzero(arities == 2)
        loops[pairs] = idx[ptr[pairs]] == idx[ptr[pairs]+1]
        special = (arities == 1) | loops
        present_groups = (np.logical_or.reduceat(share.presence, ptr[:-1])
                          if values is not None else np.zeros(count, bool))
        if np.any(special & ((heads != 0) | present_groups)):
            raise ValueError("witnesses and self-loops declare no head/share")
        if self.share_rule == "declared" and np.any(~special & ~present_groups):
            raise ValueError("declared share rule requires all ordinary relation shares")
        # A distinct pair is already a valid support, and an absent share uses
        # the canonical rule. Only branching or explicitly shared columns need
        # per column rational validation.
        for e in np.flatnonzero((arities > 2) | present_groups):
            start, end = ptr[e:e+2]
            row = idx[int(start):int(end)]
            arity = len(row)
            if len(np.unique(row)) != arity:
                raise ValueError("relation repeats a C0 participant; only an exact [v, v] self-loop may repeat a participant")
            present = share.presence[int(start):int(end)]
            if present_groups[e]:
                if not present.all():
                    raise ValueError("a declared share must cover its whole relation")
                shares = values[int(start):int(end)]
                if shares[int(heads[e])] != 0 or any(v <= 0 for i, v in enumerate(shares) if i != heads[e]) or sum(shares) != 1:
                    raise ValueError("shares require a zero head and positive tails summing to one")
        object.__setattr__(self, "support_ptr", ptr)
        object.__setattr__(self, "support_idx", idx)
        object.__setattr__(self, "head_slot", heads)
        object.__setattr__(self, "sign", signs)
        object.__setattr__(self, "share", share)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "relation_id", _identities(self.relation_id, count, "relation ID"))
        types = tuple(Absent for _ in range(count)) if self.relation_type is None else tuple(self.relation_type)
        if len(types) != count:
            raise ValueError("relation types must be aligned")
        object.__setattr__(self, "relation_type", types)
        object.__setattr__(self, "attributes", _copy_attributes(self.attributes))
        object.__setattr__(self, "provenance", deepcopy(self.provenance))
        object.__setattr__(self, "embedding", deepcopy(self.embedding))
        object.__setattr__(self, "weight_source", NumberRule(self.weight_source))

    @property
    def n_relations(self):
        return len(self.support_ptr)-1

    @classmethod
    def from_supports(cls, supports, *, vertices=None, n_vertices=None, weights=None, signs=None,
                      heads=None, shares=None, relation_ids=None, relation_types=None, number_rule=NumberRule.EXACT, **kwargs):
        ptr, indices = [0], []
        for support in supports:
            row = _integers(support, name="relation participants")
            indices.extend(row)
            ptr.append(len(indices))
        if vertices is None:
            count = (int(max(indices))+1 if indices else 0) if n_vertices is None else n_vertices
            vertices = VertexTable.for_count(count)
        return cls(np.asarray(ptr, np.int64), np.asarray(indices, np.int64), vertices, heads,
                   None if shares is None else ExactArray.from_values(shares),
                   None if weights is None else ExactArray.from_values(weights, rule=number_rule), signs,
                   None if relation_ids is None else tuple(relation_ids), None if relation_types is None else tuple(relation_types),
                   weight_source=number_rule, **kwargs)

    @classmethod
    def from_arrays(cls, support_ptr, support_idx, *, n_vertices=None, weights=None, signs=None, relation_ids=None):
        """Compatibility normalizer for existing sparse, binary numeric callers."""
        idx = _integers(support_idx, name="relation participants")
        nv = int(idx.max())+1 if idx.size else 0
        if n_vertices is not None:
            nv = n_vertices
        if relation_ids is not None:
            from .graph import _as_relation_ids
            relation_ids = _as_relation_ids(relation_ids, len(support_ptr)-1)
        if signs is not None:
            values = np.asarray(signs)
            if values.ndim != 1 or values.dtype.kind not in "iuf" or not np.isin(values, (-1, 1)).all():
                raise ValueError("relation signs must be -1 or +1")
            signs = np.asarray(values, np.int8)
        return cls(support_ptr, idx, VertexTable.for_count(nv),
                   weight=None if weights is None else ExactArray.from_values(weights, rule=NumberRule.BINARY_EXACT),
                   sign=signs, relation_id=None if relation_ids is None else tuple(relation_ids),
                   weight_source=NumberRule.BINARY_EXACT)

    def graph_arguments(self):
        weights = self.weight.values()
        weights = [Fraction(1) if v is Absent else v for v in weights]
        # Preserve absence in this carrier; the native mathematical view explicitly
        # uses the declared unit for absent rule. No arity dependent defaults.
        w_E = None if not self.weight.presence.any() else np.asarray(weights, dtype=object)
        if w_E is not None and not self.weight.kind.any() and not self.weight.bigint:
            w_E = np.asarray(weights, dtype=np.int64)
        elif w_E is not None and self.weight_source in {NumberRule.BINARY_EXACT, NumberRule.TYPED_COLUMNAR}:
            from .exact_value import binary_fraction
            try:
                view = np.asarray(weights, dtype=np.float64)
                if all(binary_fraction(v) == q for v, q in zip(view, weights, strict=True)):
                    w_E = view
            except (OverflowError, ValueError):
                pass
        native_ids = None
        if self.relation_id and all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) and -(2**63) <= int(v) < 2**63 for v in self.relation_id):
            native_ids = np.asarray(self.relation_id, np.int64)
        shares = self.share.values()
        return {"boundary_ptr": self.support_ptr, "boundary_idx": self.support_idx,
                "head_slot": self.head_slot, "shares": [None if v is Absent else v for v in shares],
                "w_E": w_E, "signs": None if np.all(self.sign == 1) else self.sign,
                "relation_ids": native_ids}

    def as_record(self):
        """Declaration record; cell attributes are encoded by their registered owners."""
        return {"version": 1, "support_ptr": self.support_ptr, "support_idx": self.support_idx,
                "vertices": self.vertices.as_record(), "head_slot": self.head_slot, "share": self.share,
                "weight": self.weight, "sign": self.sign, "relation_id": self.relation_id,
                "relation_type": self.relation_type, "embedding": self.embedding, "provenance": self.provenance,
                "share_rule": self.share_rule, "weight_rule": self.weight_rule, "weight_source": self.weight_source.value}

    def as_state_record(self, *, native_ids=None, version=3):
        """Retained declarations only; primary tensors have one canonical owner.

        Support, mathematical weights/signs, labels and provenance are already
        sealed by their components. This record preserves only information those
        components cannot express, avoiding a second full copy of every relation.
        """
        pair_shares = []
        if self.share.presence.any():
            shares = self.share.values()
            for e in np.flatnonzero(np.diff(self.support_ptr) == 2):
                a, b = map(int, self.support_ptr[e:e+2])
                if self.support_idx[a] != self.support_idx[a+1] and self.share.presence[a:b].all():
                    pair_shares.append((int(e), ExactArray.from_values(shares[a:b])))
        vertices = None
        if any(v is not Absent for v in self.vertices.ids) or any(self.vertices.aliases):
            vertices = {"ids": self.vertices.ids, "aliases": self.vertices.aliases}
        ids = None if native_ids is not None or all(v is Absent for v in self.relation_id) else self.relation_id
        types = None if all(v is Absent for v in self.relation_type) else self.relation_type
        # All absent metrics already have no primary weight tensor. Mixed absence
        # needs only its indices, not another numerator/denominator array.
        if type(version) is not int or version not in {2, 3}:
            raise ValueError("unknown retained relation declaration version")
        result = {"version": version, "vertices": vertices, "relation_id": ids, "relation_type": types,
                "pair_shares": tuple(pair_shares),
                "embedding": self.embedding, "share_rule": self.share_rule,
                "weight_rule": self.weight_rule, "weight_source": self.weight_source.value}
        if version == 2:
            result["weight_absent"] = np.flatnonzero(~self.weight.presence).astype("<i8") if self.weight.presence.any() else None
        return result

    @classmethod
    def from_state_record(cls, record, graph):
        fields = {"version", "vertices", "relation_id", "relation_type", "pair_shares", "embedding", "share_rule", "weight_rule", "weight_source"}
        if isinstance(record, dict) and record.get("version") == 2:
            fields.add("weight_absent")
        if not isinstance(record, dict) or set(record) != fields or type(record["version"]) is not int or record["version"] not in {2, 3}:
            raise ValueError("unknown retained relation declaration")
        base = cls.from_graph(graph)
        vertex = base.vertices
        if record["vertices"] is not None:
            values = record["vertices"]
            if not isinstance(values, dict) or set(values) != {"ids", "aliases"}:
                raise ValueError("unknown retained vertex fields")
            vertex = VertexTable(values["ids"], vertex.labels, values["aliases"])
        weights = base.weight.values()
        if record.get("weight_absent") is not None:
            indices = _integers(record["weight_absent"], name="absent weight indices")
            if (indices.size and (indices[0] < 0 or indices[-1] >= base.n_relations)) or np.any(indices[1:] <= indices[:-1]):
                raise ValueError("invalid absent weight indices")
            for i in indices:
                if weights[i] is not Absent and weights[i] != 1:
                    raise ValueError("absent metric does not have its declared unit view")
                weights[i] = Absent
        shares = base.share.values()
        prior = -1
        for e, exact in record["pair_shares"]:
            if type(e) is not int or not prior < e < base.n_relations or not isinstance(exact, ExactArray) or exact.shape != (2,):
                raise ValueError("invalid retained pair share")
            a, b = map(int, base.support_ptr[e:e+2])
            if b-a != 2 or base.support_idx[a] == base.support_idx[a+1]:
                raise ValueError("retained pair share is outside the pair carrier")
            shares[a:b] = exact.values()
            prior = e
        return cls(base.support_ptr, base.support_idx, vertex, base.head_slot, ExactArray.from_values(shares),
                   ExactArray.from_values(weights), base.sign,
                   base.relation_id if record["relation_id"] is None else record["relation_id"],
                   base.relation_type if record["relation_type"] is None else record["relation_type"],
                   attributes=base.attributes, embedding=record["embedding"], provenance=base.provenance,
                   share_rule=record["share_rule"], weight_rule=record["weight_rule"], weight_source=record["weight_source"])

    def remap_to(self, graph, maps):
        """Transport declared tables through the core's actual cell compaction.

        Read the already compacted support directly: calling a graph property
        while the edit is being materialized would recursively flush that edit.
        """
        from .column import declaration_of
        nv, ne = graph._nV, graph._nE
        ids, aliases = [Absent]*nv, [()]*nv
        for old, new in enumerate(maps[0]):
            if new >= 0 and old < len(self.vertices):
                ids[int(new)] = self.vertices.ids[old]
                aliases[int(new)] = self.vertices.aliases[old]
        labels = tuple(graph.provenance.get("vertex_labels", ("",)*nv))
        vertex = VertexTable(tuple(ids), labels, tuple(aliases))
        declaration = declaration_of(graph)
        heads = np.zeros(ne, np.int32) if declaration is None else declaration.head_slot
        shares = [Absent]*len(graph._boundary_idx)
        if declaration is not None:
            shares = [Absent if d == 0 else Fraction(int(n), int(d))
                      for n, d in zip(declaration.share_num, declaration.share_den, strict=True)]
        weights = [Absent]*ne if graph._w_E is None else list(graph._w_E)
        presence = getattr(graph, "_weight_presence", None)
        if presence is not None:
            weights = [value if presence[i] else Absent for i, value in enumerate(weights)]
        relation_ids, types = [Absent]*ne, [Absent]*ne
        original_shares = self.share.values()
        for old, new in enumerate(maps[1]):
            if new < 0 or old >= self.n_relations:
                continue
            new = int(new)
            relation_ids[new], types[new] = self.relation_id[old], self.relation_type[old]
            a, b = map(int, self.support_ptr[old:old+2])
            start, end = map(int, graph._boundary_ptr[new:new+2])
            if self.share.presence[a:b].all() and not any(v is not Absent for v in shares[start:end]):
                positions = {int(maps[0][v]): a+i for i,v in enumerate(self.support_idx[a:b])}
                if len(positions) == end-start and set(positions) == set(map(int, graph._boundary_idx[start:end])):
                    if end-start == 2:
                        head = a+int(self.head_slot[old])
                        shares[start:end] = [original_shares[head], original_shares[a+b-1-head]]
                    else:
                        shares[start:end] = [original_shares[positions[int(v)]] for v in graph._boundary_idx[start:end]]
        if graph._relation_ids is not None:
            relation_ids = list(graph._relation_ids)
        signs = np.ones(ne, np.int8) if graph._signs is None else np.asarray(graph._signs, np.int8)
        return Relations(graph._boundary_ptr, graph._boundary_idx, vertex, heads,
                         ExactArray.from_values(shares), ExactArray.from_values(weights, rule=NumberRule.BINARY_EXACT),
                         signs, tuple(relation_ids), tuple(types), attributes=getattr(graph, "_cell_metadata", {}),
                         embedding=graph.embedding, provenance=graph.provenance,
                         share_rule=self.share_rule, weight_rule=self.weight_rule, weight_source=self.weight_source)

    @classmethod
    def from_record(cls, record, *, attributes=None):
        fields = {"version", "support_ptr", "support_idx", "vertices", "head_slot", "share", "weight", "sign", "relation_id", "relation_type", "embedding", "provenance", "share_rule", "weight_rule", "weight_source"}
        if not isinstance(record, dict) or set(record) != fields or record["version"] != 1:
            raise ValueError("unknown relation declaration record")
        vertices = record["vertices"]
        if set(vertices) != {"ids", "labels", "aliases"}:
            raise ValueError("unknown vertex declaration fields")
        return cls(**{k: v for k, v in record.items() if k not in {"version", "vertices"}},
                   vertices=VertexTable(**vertices), attributes=attributes or {})

    @classmethod
    def from_graph(cls, graph):
        """Read the current mathematical carrier and retained declaration tables."""
        from .column import declaration_of
        graph._ensure_clean()
        source = getattr(graph, "_relation_source", None)
        count, nv = graph.nE, graph.nV
        if source is not None and (source.n_relations > count or len(source.vertices) > nv):
            raise ValueError("relation declaration was not transported through cell removal")
        meta = graph.provenance
        labels = list(meta.get("vertex_labels", ()))
        labels.extend("" for _ in range(nv-len(labels)))
        ids = list(source.vertices.ids) if source is not None else []
        ids.extend(Absent for _ in range(nv-len(ids)))
        aliases = list(source.vertices.aliases) if source is not None else []
        aliases.extend(() for _ in range(nv-len(aliases)))
        vertex = VertexTable(tuple(ids), tuple(labels), tuple(aliases))
        declaration = declaration_of(graph)
        heads = np.zeros(count, np.int32) if declaration is None else declaration.head_slot
        shares = [Absent]*len(graph.boundary_idx)
        if declaration is not None:
            shares = [Absent if d == 0 else Fraction(int(n), int(d)) for n,d in zip(declaration.share_num, declaration.share_den, strict=True)]
        # Pair shares may be canonicalized away by the native boundary, but their
        # explicit presence remains a source declaration, not an inferred value.
        if source is not None and source.share.presence.any():
            original = source.share.values()
            for e in range(source.n_relations):
                start, end = map(int, graph.boundary_ptr[e:e+2])
                a, b = map(int, source.support_ptr[e:e+2])
                if not any(v is not Absent for v in shares[start:end]) and source.share.presence[a:b].all() and end-start == b-a:
                    row = graph.boundary_idx[start:end]
                    old = source.support_idx[a:b]
                    positions = {int(v): i for i,v in enumerate(old)}
                    if set(positions) == set(map(int,row)):
                        if end-start == 2:
                            # The native pair stores its current head at slot zero.
                            # Preserve the declared role/kind when an explicit
                            # orientation update swaps the participants.
                            head = a+int(source.head_slot[e])
                            shares[start:end] = [original[head], original[a+b-1-head]]
                        else:
                            shares[start:end] = [original[a+positions[int(v)]] for v in row]
        weights = [Absent]*count
        if graph.w_E is not None:
            weights = list(graph.w_E)
            presence = getattr(graph, "_weight_presence", None)
            if presence is not None:
                if np.asarray(presence).shape != (count,):
                    raise ValueError("weight presence is not aligned with its relation basis")
                if any(value != 1 for value, present in zip(weights, presence, strict=True) if not present):
                    raise ValueError("absent metric does not have its declared unit view; use set_cell_attrs")
                weights = [value if presence[i] else Absent for i, value in enumerate(weights)]
        if graph.relation_ids is not None:
            relation_ids = tuple(graph.relation_ids)
        else:
            relation_ids = tuple(source.relation_id) if source is not None else ()
            relation_ids += tuple(Absent for _ in range(count-len(relation_ids)))
        types = tuple(source.relation_type) if source is not None else ()
        types += tuple(Absent for _ in range(count-len(types)))
        sign = np.ones(count, np.int8) if graph._signs is None else graph._signs
        # Legacy floating sign arrays are admitted only when exactly +/-1.
        if np.asarray(sign).dtype.kind == "f":
            if not np.isin(sign, (-1, 1)).all():
                raise ValueError("relation signs must be -1 or +1")
            sign = np.asarray(sign, np.int8)
        share_array = (ExactArray.from_values(shares) if any(v is not Absent for v in shares)
                       else ExactArray.absent((len(shares),)))
        weight_array = (ExactArray.from_values(weights, rule=NumberRule.BINARY_EXACT) if graph.w_E is not None
                        else ExactArray.absent((count,)))
        return cls(graph.boundary_ptr, graph.boundary_idx, vertex, heads, share_array,
                   weight_array, sign, relation_ids, types,
                   attributes=getattr(graph, "_cell_metadata", {}), embedding=graph.embedding,
                   provenance=meta, share_rule=source.share_rule if source is not None else "canonical_for_absent",
                   weight_rule=source.weight_rule if source is not None else "unit_for_absent",
                   weight_source=source.weight_source if source is not None else NumberRule.BINARY_EXACT)


@dataclass(frozen=True)
class RelationSpec:
    """Reader independent mapping from typed records to primary relations."""
    shape: str
    participants: tuple[str, ...]
    head: str | None = None
    share: str | None = None
    weight: str | None = None
    sign: str | None = None
    identity: str | None = None
    type: str | None = None
    attributes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.shape not in {"pair", "row", "long"}:
            raise ValueError("unsupported relation shape")
        participants = tuple(self.participants)
        if not participants or any(type(v) is not str or not v for v in participants):
            raise ValueError("participants must name declared fields")
        if self.shape == "pair" and len(participants) != 2:
            raise ValueError("a pair declaration requires two participant fields")
        if self.shape in {"row", "long"} and len(participants) != 1:
            raise ValueError("row and long declarations require one participant field")
        if self.shape == "long" and self.identity is None:
            raise ValueError("long relations require an identity field")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "attributes", tuple(self.attributes))
