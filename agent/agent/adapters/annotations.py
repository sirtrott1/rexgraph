"""Registered annotation readers with exact occurrence and attachment coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from rexgraph.span import SpanAttachment, SpanBlock
from rexgraph.coordinate_map import CoordinateMap
from . import EdgeConstruction

__all__ = ["AnnotationRecord", "AnnotationBundle", "read_brat", "load_brat",
           "read_conll", "load_conll", "annotation_records", "register"]


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _read(path):
    path = Path(path)
    raw = path.read_bytes()
    decoded = gzip.decompress(raw) if path.suffix.lower() == ".gz" else raw
    text = decoded.decode("utf-8", errors="strict")
    return raw, decoded, text


def _source(path, role, raw, decoded):
    return {"role": role, "name": Path(path).name, "sha256": _hash(raw),
            "decoded_sha256": _hash(decoded), "bytes": len(raw)}


@dataclass(frozen=True)
class AnnotationRecord:
    """One source assertion with its references and original line retained."""
    identifier: str
    kind: str
    label: str
    components: tuple = ()
    references: tuple = ()
    value: str = ""
    raw_line: str = ""
    line_number: int = 0
    byte_components: tuple = ()
    identity_origin: str = "source"

    def __post_init__(self):
        for name in ("identifier", "kind", "label", "identity_origin"):
            _name(getattr(self, name), name)
        if self.kind not in {"text", "event", "relation", "attribute", "normalization", "note", "equivalence"}:
            raise ValueError("unknown annotation record kind")
        for name in ("components", "byte_components"):
            values = tuple(tuple(v) for v in getattr(self, name))
            if any(len(v) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in v)
                   or not 0 <= v[0] < v[1] for v in values):
                raise ValueError("annotation coordinates require nonempty integer intervals")
            object.__setattr__(self, name, values)
        refs = tuple(tuple(v) for v in self.references)
        if any(len(v) != 2 or any(not isinstance(x, str) or not x for x in v) for v in refs):
            raise ValueError("annotation references require a role and identifier")
        object.__setattr__(self, "references", refs)
        if self.kind == "text" and not self.components:
            raise ValueError("a text annotation requires at least one component")
        if self.byte_components and len(self.byte_components) != len(self.components):
            raise ValueError("byte and source component counts differ")
        if any(not isinstance(v, str) for v in (self.value, self.raw_line)):
            raise TypeError("annotation text must be strings")
        if isinstance(self.line_number, bool) or not isinstance(self.line_number, int) or self.line_number < 0:
            raise ValueError("line number must be a nonnegative integer")

    def as_record(self):
        return {"schema": "annotation_record/1", "id": self.identifier,
                "kind": self.kind, "label": self.label, "components": self.components,
                "references": self.references, "value": self.value,
                "raw_line": self.raw_line, "line_number": self.line_number,
                "byte_components": self.byte_components, "identity_origin": self.identity_origin}


@dataclass(frozen=True)
class AnnotationBundle:
    """Parsed assertions and exact attachments before native construction."""
    document_id: str
    format: str
    axis: str
    unit: str
    records: tuple
    attachments: tuple
    manifest_json: str

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    @property
    def digest(self):
        return self.manifest["bundle_digest"]

    def construction(self):
        """Build one primary relation per recorded assertion without pair expansion."""
        records = tuple(self.records)
        by_id = {r.identifier: r for r in records}
        if len(by_id) != len(records):
            raise ValueError("annotation identifiers must be unique")
        vertices, index, supports, attrs = [], {}, [], {}
        types = sorted({r.label for r in records})
        type_index = {v: i for i, v in enumerate(types)}
        numeric_ids = []
        for i, record in enumerate(records):
            candidates = []
            if record.kind == "text":
                occurrence = {}
                for a, b in record.components:
                    j = occurrence.get((a, b), 0)
                    occurrence[(a, b)] = j + 1
                    candidates.append(("fragment", self.axis, a, b, j))
            elif record.references:
                occurrence, candidates = {}, []
                for role, target in record.references:
                    j = occurrence.get(role, 0)
                    occurrence[role] = j + 1
                    candidates.append(("reference", self.document_id, role, target, j))
                candidates.sort(key=lambda key: (key[2] != "Trigger", key[2], key[4]))
            else:
                candidates = [("document", self.document_id)]
            support = []
            for key in candidates:
                key = _json(key)
                if key not in index:
                    index[key] = len(vertices)
                    vertices.append(key)
                support.append(index[key])
            supports.append(support)
            identity = _json((self.document_id, record.identifier))
            numeric_ids.append(int(_hash(identity.encode("utf-8"))[:16], 16) & ((1 << 63) - 1))
            attrs[i] = {"annotation_id": record.identifier, "annotation_kind": record.kind,
                        "annotation_type": record.label, "annotation_record": _json(record.as_record()),
                        "annotation_namespace": self.document_id}
        if len(set(numeric_ids)) != len(numeric_ids):
            raise ValueError("primary numeric identity collision; choose another document namespace")
        positions = {r.identifier: i for i, r in enumerate(records)}
        for owner, attachment in self.attachments:
            if owner not in positions:
                raise ValueError("attachment owner is not a recorded assertion")
            attrs[positions[owner]]["span:" + attachment.annotation_id] = attachment
        return EdgeConstruction(
            sources=np.array([], dtype=np.int32), targets=np.array([], dtype=np.int32),
            weights=np.array([], dtype=np.float64), signs=np.array([], dtype=np.float64),
            type_labels=np.array([type_index[r.label] for r in records], dtype=np.int32),
            vertex_labels=vertices, n_types=len(types), type_names=types, branching=supports,
            attributes={1: attrs}, origin=self.document_id,
            relation_ids=np.asarray(numeric_ids, dtype=np.int64), source_manifest=self.manifest)

    def to_rex(self):
        from agent.auto import build_rex_from_edges
        return build_rex_from_edges(self.construction(), face_selection="none",
                                    input_type="annotation_bundle")


def _bundle(document_id, format, axis, unit, records, files, profile, *, time_supports=None, groundings=None):
    records = tuple(records)
    by_id = {r.identifier: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate annotation identifier")
    for record in records:
        for _, target in record.references:
            if target not in by_id:
                raise ValueError(f"{record.identifier}: unknown annotation reference {target}")
    times = dict(time_supports or {})
    maps = dict(groundings or {})
    used, attachments = set(), []
    qualifiers_by_target = {}
    for record in records:
        if record.kind in {"attribute", "normalization", "note"}:
            qualifiers_by_target.setdefault(record.references[0][1], []).append(
                (record.kind + ":" + record.identifier, _json((record.label, record.value))))
    def attach(cell, identity, target, role):
        record = by_id[target]
        if record.kind != "text":
            return
        text = SpanBlock(target, axis, unit,
                         tuple((str(j), a, b) for j, (a, b) in enumerate(record.components)))
        temporal = times.get(identity)
        grounding = maps.get(identity)
        if temporal is not None and not isinstance(temporal, SpanBlock):
            raise TypeError("supplied temporal supports must be SpanBlocks")
        if grounding is not None and not isinstance(grounding, CoordinateMap):
            raise TypeError("supplied grounding must be a CoordinateMap")
        qualifiers = [("format", format), ("referent", target), ("annotation_type", record.label)]
        qualifiers.extend(qualifiers_by_target.get(cell, ()))
        value = SpanAttachment(identity, cell, role, document_id, text, temporal, grounding,
                               tuple(qualifiers))
        attachments.append((cell, value))
        used.add(identity)
    for record in records:
        if record.kind == "text":
            attach(record.identifier, record.identifier, record.identifier, "mention")
        elif record.kind in {"event", "relation"}:
            occurrence = {}
            for role, target in record.references:
                j = occurrence.get(role, 0)
                occurrence[role] = j + 1
                attach(record.identifier, f"{record.identifier}/{role}/{j}", target, role)
    unknown = (set(times) | set(maps)) - used
    if unknown:
        raise ValueError("temporal declarations refer to unknown textual attachments: " + ", ".join(sorted(unknown)))
    manifest = {"schema": "annotation_bundle/1", "document_id": document_id, "format": format,
                "axis": axis, "unit": unit, "files": files, "profile": profile,
                "annotation_count": len(records), "attachment_count": len(attachments),
                "primary_identity": "namespace_and_declared_assertion_identity",
                "numeric_identity": "sha256_63_bits_with_collision_check",
                "boundary_policy": "trigger_then_named_roles_or_source_components",
                "attachments_digest": _hash(_json([a.coefficient_digest for _, a in attachments]).encode("utf-8"))}
    manifest["bundle_digest"] = _hash(_json(manifest).encode("utf-8"))
    return AnnotationBundle(document_id, format, axis, unit, records, tuple(attachments), _json(manifest))


def _argument(token, identifier):
    pieces = token.split(":", 1)
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f"{identifier}: expected ROLE:ID")
    return tuple(pieces)


def read_brat(path, *, document_id, text_path=None, check_text=True, time_supports=None, groundings=None):
    """Read BRAT standoff records without changing bytes or inferring event times."""
    document_id = _name(document_id, "document_id")
    if not isinstance(check_text, bool):
        raise TypeError("check_text must be boolean")
    path = Path(path)
    if text_path is None:
        base = path.with_suffix("") if path.suffix.lower() == ".gz" else path
        text_path = base.with_suffix(".txt")
        if not Path(text_path).exists() and Path(str(text_path) + ".gz").exists():
            text_path = Path(str(text_path) + ".gz")
    ann_raw, ann_decoded, ann = _read(path)
    text_raw, text_decoded, text = _read(text_path)
    axis = "text/" + _hash(_json((document_id, _hash(text_decoded), "unicode_codepoint")).encode("utf-8"))
    from rexgraph.document import _byte_starts
    starts = _byte_starts(text, "utf-8")
    records, seen, equivalences = [], set(), 0
    for line_number, raw_line in enumerate(ann.splitlines(keepends=True), 1):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) < 2:
            raise ValueError(f"line {line_number}: annotation requires a tab separated identifier")
        identifier, spec = columns[:2]
        if identifier == "*":
            equivalences += 1
            identifier = "*" + str(equivalences)
        elif not re.fullmatch(r"[TERAMN#][0-9]+", identifier):
            raise ValueError(f"line {line_number}: unsupported annotation identifier {identifier!r}")
        if identifier in seen:
            raise ValueError(f"line {line_number}: duplicate annotation identifier {identifier}")
        seen.add(identifier)
        tokens = spec.split()
        if not tokens:
            raise ValueError(f"{identifier}: missing annotation type")
        prefix, label = identifier[0], tokens[0]
        components, refs, value = (), (), ""
        if prefix == "T":
            if len(columns) != 3:
                raise ValueError(f"{identifier}: text annotation requires reference text")
            parts = spec.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{identifier}: missing span coordinates")
            parsed = []
            for segment in parts[1].split(";"):
                bounds = segment.split()
                if len(bounds) != 2 or any(not re.fullmatch(r"[0-9]+", v) for v in bounds):
                    raise ValueError(f"{identifier}: invalid character interval")
                start, stop = map(int, bounds)
                if not 0 <= start < stop <= len(text):
                    raise ValueError(f"{identifier}: span lies outside the document or is empty")
                parsed.append((start, stop))
            components = tuple(parsed)
            value = columns[2]
            expected = " ".join(text[a:b] for a, b in components)
            if check_text and expected != value:
                raise ValueError(f"{identifier}: reference text does not match the original character spans")
            kind = "text"
        elif prefix == "E":
            if len(columns) != 2:
                raise ValueError(f"{identifier}: unexpected tab in event record")
            label, trigger = _argument(tokens[0], identifier)
            refs = (("Trigger", trigger),) + tuple(_argument(t, identifier) for t in tokens[1:])
            kind = "event"
        elif prefix == "R":
            if len(columns) != 2 or len(tokens) != 3:
                raise ValueError(f"{identifier}: binary relation requires two arguments")
            refs = tuple(_argument(t, identifier) for t in tokens[1:])
            kind = "relation"
        elif prefix == "*":
            if len(columns) != 2 or label != "Equiv" or len(tokens) < 3:
                raise ValueError(f"{identifier}: invalid equivalence record")
            refs = tuple(("Member", target) for target in tokens[1:])
            if len(set(tokens[1:])) != len(tokens[1:]):
                raise ValueError(f"{identifier}: repeated equivalence member")
            kind = "equivalence"
        elif prefix in {"A", "M"}:
            if len(columns) != 2 or not 2 <= len(tokens) <= 3:
                raise ValueError(f"{identifier}: invalid attribute record")
            refs = (("Target", tokens[1]),)
            value = tokens[2] if len(tokens) == 3 else "true"
            kind = "attribute"
        elif prefix == "N":
            if len(columns) != 3 or len(tokens) != 3 or label != "Reference" or ":" not in tokens[2]:
                raise ValueError(f"{identifier}: invalid normalization record")
            refs = (("Target", tokens[1]),)
            value = _json((tokens[2], columns[2]))
            kind = "normalization"
        else:
            if len(columns) < 3 or len(tokens) != 2:
                raise ValueError(f"{identifier}: invalid note record")
            refs = (("Target", tokens[1]),)
            value = "\t".join(columns[2:])
            kind = "note"
        records.append(AnnotationRecord(identifier, kind, label, components, refs, value, raw_line, line_number,
                       tuple((int(starts[a]), int(starts[b])) for a, b in components),
                       "positional_equivalence" if prefix == "*" else "source"))
    by_id = {r.identifier: r for r in records}
    for record in records:
        if record.kind == "event":
            trigger = by_id.get(record.references[0][1])
            if trigger is None or trigger.kind != "text" or trigger.label != record.label:
                raise ValueError(f"{record.identifier}: event requires a matching text trigger")
            if any(by_id.get(t) is None or by_id[t].kind not in {"text", "event"} for _, t in record.references[1:]):
                raise ValueError(f"{record.identifier}: event arguments must reference entities or events")
    return _bundle(document_id, "brat", axis, "unicode_codepoint", records,
                   [_source(text_path, "text", text_raw, text_decoded),
                    _source(path, "annotations", ann_raw, ann_decoded)],
                   {"dialect": "brat_standoff_1.3", "check_text": check_text,
                    "encoding": "utf8", "line_endings": "preserved", "temporal_normalization": "explicit_only"},
                   time_supports=time_supports, groundings=groundings)


def load_brat(path, *, document_id, text_path=None, check_text=True, time_supports=None, groundings=None):
    """Return BRAT data through the standard relation construction interface."""
    return read_brat(path, document_id=document_id, text_path=text_path, check_text=check_text,
                     time_supports=time_supports, groundings=groundings).construction()


def read_conll(path, *, document_id, token_column=0, tag_column=-1, scheme="IOB2"):
    """Read a declared NER tag column on token occurrence coordinates."""
    document_id = _name(document_id, "document_id")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (token_column, tag_column)):
        raise TypeError("column selectors must be integers")
    if scheme not in {"IOB1", "IOB2", "BIOES"}:
        raise ValueError("scheme must be IOB1, IOB2 or BIOES")
    raw, decoded, text = _read(path)
    tokens, tags, breaks, source_lines = [], [], set(), []
    sentence, document = 0, 0
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), 1):
        parts = raw_line.split()
        if not parts or parts[0] == "-DOCSTART-":
            breaks.add(len(tokens))
            sentence += 1
            if parts and parts[0] == "-DOCSTART-":
                document += 1
            continue
        try:
            token, tag = parts[token_column], parts[tag_column]
        except IndexError as exc:
            raise ValueError(f"line {line_number}: selected column is absent") from exc
        tokens.append((document, sentence, token))
        tags.append(tag)
        source_lines.append((line_number, raw_line))
    breaks.add(len(tokens))
    axis = "tokens/" + _hash(_json((document_id, tokens)).encode("utf-8"))
    entities, start, label = [], None, None
    def finish(stop):
        nonlocal start, label
        if start is not None:
            entities.append((start, stop, label))
        start, label = None, None
    for i in range(len(tokens) + 1):
        if i in breaks:
            if scheme == "BIOES" and start is not None:
                raise ValueError("BIOES entity is not closed before a boundary")
            finish(i)
        if i == len(tokens):
            break
        tag = tags[i]
        if tag == "O":
            if scheme == "BIOES" and start is not None:
                raise ValueError("BIOES entity is not closed before O")
            finish(i)
            continue
        parts = tag.split("-", 1)
        if len(parts) != 2 or not parts[1] or parts[0] not in ({"B", "I", "E", "S"} if scheme == "BIOES" else {"B", "I"}):
            raise ValueError(f"line {source_lines[i][0]}: invalid {scheme} tag {tag!r}")
        prefix, kind = parts
        if prefix in {"B", "S"}:
            if scheme == "BIOES" and start is not None:
                raise ValueError("BIOES entity is not closed before a new entity")
            finish(i)
            start, label = i, kind
            if prefix == "S":
                finish(i + 1)
        elif start is None or label != kind:
            if scheme != "IOB1":
                raise ValueError(f"line {source_lines[i][0]}: tag lacks a compatible entity start")
            finish(i)
            start, label = i, kind
        elif prefix == "E":
            finish(i + 1)
    records = []
    for a, b, kind in entities:
        key = "T" + _hash(_json((axis, a, b, kind)).encode("utf-8"))
        records.append(AnnotationRecord(key, "text", kind, ((a, b),), (),
                                        " ".join(v[2] for v in tokens[a:b]),
                                        "".join(v[1] for v in source_lines[a:b]), source_lines[a][0], (), "span_content"))
    return _bundle(document_id, "conll_ner", axis, "token", records,
                   [_source(path, "token_annotations", raw, decoded)],
                   {"dialect": "declared_ner_column", "scheme": scheme, "token_column": token_column,
                    "tag_column": tag_column, "encoding": "utf8", "offset_origin": "token_occurrence",
                    "identity_origin": "span_content_not_inferred_lineage", "token_count": len(tokens)})


def load_conll(path, *, document_id, token_column=0, tag_column=-1, scheme="IOB2"):
    """Return the declared NER column as primary mention relations."""
    return read_conll(path, document_id=document_id, token_column=token_column,
                      tag_column=tag_column, scheme=scheme).construction()


def annotation_records(rex):
    """Read the declared source records from a persisted annotation construction."""
    manifest = getattr(rex, "_agent_meta", {}).get("source_manifest", {})
    if manifest.get("schema") != "annotation_bundle/1":
        raise ValueError("source is not a declared annotation bundle")
    records = []
    for _, values in sorted(getattr(rex, "_cell_metadata", {}).get(1, {}).items()):
        raw = values.get("annotation_record")
        if raw is None:
            continue
        value = json.loads(raw)
        if value.pop("schema", None) != "annotation_record/1":
            raise ValueError("unsupported annotation record schema")
        identifier = value.pop("id")
        records.append(AnnotationRecord(identifier=identifier, **value))
    if len(records) != manifest["annotation_count"] or len({r.identifier for r in records}) != len(records):
        raise ValueError("annotation records do not match the bundle declaration")
    return tuple(records)


def register(register_reader):
    register_reader("brat", load_brat, extensions=(".ann",))
    register_reader("conll_ner", load_conll, extensions=(".conll", ".iob", ".bio"))
