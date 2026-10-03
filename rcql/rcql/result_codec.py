"""Portable exact results with a closed value inventory and sealed graph bases.

Decoding reconstructs values and syntax records only; it never executes a query,
imports a class named by the payload, or materializes an operator. Live handles
without a portable codec are refused explicitly.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
import hashlib
import hmac

from rexgraph.value_codec import pack_value, unpack_value

from . import ast
from .types import Exactness, PredicateResult

__all__ = ["pack_result", "unpack_result", "register_result_storage_codec"]


def register_result_storage_codec():
    """Explicitly install the existing portable result codec into native RCDB.

    RCDB never imports this package from a persisted provider name. Importing
    RCQL alone also does not add an optional RCDB dependency.
    """
    from rcdb import QUERY_RESULT_CODEC, RecordCodec, register_record_codec
    return register_record_codec(RecordCodec(QUERY_RESULT_CODEC, "RCQLResult", pack_result, unpack_result,
                                            admission=_record_cell_counts))
_MAGIC = b"RGQR\x01"
_LIMIT = 64*1024*1024
_SYNTAX = {cls.__name__: cls for cls in (
    ast.Literal, ast.Parameter, ast.Reference, ast.Call, ast.ListExpr,
    ast.Member, ast.Alias, ast.StructuralEdit, ast.Comparison,
)}


@lru_cache(maxsize=1)
def _component_codecs():
    from rexgraph.field_codec import FIELD_VALUES, pack_field, unpack_field
    from rexgraph.section_codec import SECTION_VALUES, pack_section, unpack_section
    from rexgraph.model_codec import MODEL_VALUES, pack_model, unpack_model
    from rexgraph.span import SpanAttachment
    from rexgraph.span_codec import pack_attachment, unpack_attachment
    return {"field": (FIELD_VALUES, pack_field, unpack_field),
            "section": (SECTION_VALUES, pack_section, unpack_section),
            "model": (MODEL_VALUES, pack_model, unpack_model),
            "span": ((SpanAttachment,), pack_attachment, unpack_attachment)}


@lru_cache(maxsize=1)
def _program_types():
    from .program import Program
    from .name_relation import NameRelation
    from .program_transformation import ProgramTransformation
    from .recursive_program import RecursiveProgram
    return {cls.__name__: cls for cls in (Program, NameRelation, ProgramTransformation, RecursiveProgram)}


def bind_sources(value, references, *, strict=False):
    """Rebind checked retained source declarations, preserving their identities."""
    from rexgraph.tensor_field import FieldSource
    from rexgraph.selection import Selection
    if isinstance(value, Selection):
        for (state, _record_id, _version), source in references.items():
            if state == value.source_state:
                return value.bind(source)
        if strict:
            raise ValueError("portable selection has an unbound source dependency")
        value.check_state()
        return value
    if isinstance(value, FieldSource):
        identity = (value.state_digest, value.record_id, value.version)
        if identity in references:
            return value.bind(references[identity])
        if strict:
            raise ValueError("portable result has an unbound source dependency")
        return value
    if isinstance(value, (tuple, list)):
        values = [bind_sources(v, references, strict=strict) for v in value]
        return tuple(values) if isinstance(value, tuple) else values
    if isinstance(value, dict):
        return {k: bind_sources(v, references, strict=strict) for k, v in value.items()}
    if is_dataclass(value) and type(value).__module__.startswith(("rexgraph.", "rcql.")):
        return replace(value, **{f.name: bind_sources(getattr(value, f.name), references, strict=strict)
                                 for f in fields(value) if f.init})
    return value


def _bindings(records, graphs, value, depth):
    from rexgraph.tensor_field import FieldSource
    from .source_context import field_references
    if not isinstance(records, tuple):
        raise ValueError("invalid portable result source bindings")
    declared = {(ref.state_digest, ref.record_id, ref.version) for ref in field_references(value)}
    bindings = {}
    for record in records:
        if not isinstance(record, tuple) or len(record) != 2 or not isinstance(record[0], dict) or set(record[0]) != {"state_digest", "record_id", "version"}:
            raise ValueError("invalid portable source binding record")
        source = _node(record[1], graphs, depth+1)
        from rexgraph.graph import RexGraph
        if not isinstance(source, RexGraph):
            raise ValueError("portable source binding requires a sealed graph")
        ref = FieldSource(source, **record[0])
        key = (ref.state_digest, ref.record_id, ref.version)
        if key not in declared or key in bindings:
            raise ValueError("unclaimed or duplicate portable source binding")
        bindings[key] = source
    return bind_sources(value, bindings)


class _Encoder:
    def __init__(self):
        self.graphs, self.positions = [], {}

    def node(self, value, depth=0):
        if depth > 64:
            raise ValueError("result value nesting is too deep or cyclic")
        from rexgraph.graph import RexGraph
        from rexgraph.cochain import Chain, Cochain
        from rexgraph.tensor_field import FieldSource
        from rexgraph.partition_state import RexPartition
        from rexgraph.graph import TemporalRex
        from rexgraph.relations import Relations
        from rexgraph.selection import Selection, Lineage
        if isinstance(value, RexGraph):
            identity = id(value)
            if identity not in self.positions:
                from rexgraph.state import to_state
                state = to_state(value)
                self.positions[identity] = len(self.graphs)
                self.graphs.append({"header": state.header, "tensors": state.tensors})
            return ("graph", self.positions[identity])
        if isinstance(value, FieldSource):
            value.check()
            return ("source", value.as_record(), self.node(value.source, depth+1))
        if isinstance(value, RexPartition):
            value.check_state()
            record = ("partition", self.node(value.rex, depth+1), value.state.manifest(), value.cell_maps)
            return (*record, value.lineage.as_record()) if value.source_sizes else record
        if isinstance(value, Selection):
            return ("selection", self.node(value.source, depth+1), value.as_record())
        if isinstance(value, Lineage):
            return ("lineage", value.as_record())
        if isinstance(value, Relations):
            return ("relations", self.node(RexGraph.from_relations(value), depth+1))
        if isinstance(value, TemporalRex):
            from rexgraph.temporal_state import to_temporal_state
            state = to_temporal_state(value)
            return ("temporal", state.header, state.tensors)
        for kind, (types, encode, _) in _component_codecs().items():
            if isinstance(value, types):
                return ("component", kind, self.node(encode(value, native=True), depth+1), self.bindings(value, depth))
        if type(value) in _program_types().values():
            return ("program", type(value).__name__, value.to_bytes())
        from .recursive_program import RecursionResult
        if isinstance(value, RecursionResult):
            return ("recursion", self.node(value.to_record(), depth+1), self.bindings(value, depth))
        if type(value) in (Chain, Cochain):
            return (type(value).__name__, value.grade, self.node(value.values, depth+1),
                    self.node(value.cell_keys, depth+1), self.node(value.source, depth+1))
        if type(value) in _SYNTAX.values():
            return ("syntax", type(value).__name__,
                    {field.name: self.node(getattr(value, field.name), depth+1) for field in fields(value)})
        if isinstance(value, PredicateResult):
            return ("predicate", value.name, value.status, value.evidence)
        if isinstance(value, tuple):
            return ("tuple", tuple(self.node(v, depth+1) for v in value))
        if isinstance(value, list):
            return ("list", tuple(self.node(v, depth+1) for v in value))
        if isinstance(value, dict):
            return ("map", tuple(sorted((pack_value(k), self.node(v, depth+1)) for k, v in value.items())))
        try:
            return ("value", pack_value(value))
        except TypeError as exc:
            raise TypeError(f"result type {type(value).__name__} requires a portable codec") from exc

    def bindings(self, value, depth):
        from .source_context import field_references
        references = {}
        for ref in field_references(value):
            if ref.source is not None:
                key = (ref.state_digest, ref.record_id, ref.version)
                if key not in references:
                    references[key] = (ref.as_record(), self.node(ref.source, depth+1))
        return tuple(references.values())


def _node(record, graphs, depth=0):
    if depth > 64 or not isinstance(record, tuple) or not record or type(record[0]) is not str:
        raise ValueError("invalid or oversized result value record")
    tag = record[0]
    child = lambda value: _node(value, graphs, depth+1)
    if tag == "value" and len(record) == 2:
        return unpack_value(record[1])
    if tag == "graph" and len(record) == 2 and type(record[1]) is int and 0 <= record[1] < len(graphs):
        return graphs[record[1]]
    if tag == "source" and len(record) == 3 and isinstance(record[1], dict) and set(record[1]) == {"state_digest", "record_id", "version"}:
        from rexgraph.tensor_field import FieldSource
        from rexgraph.graph import RexGraph
        source = child(record[2])
        if source is not None and not isinstance(source, RexGraph):
            raise ValueError("invalid portable field source")
        return FieldSource(source, **record[1])
    if tag == "component" and len(record) == 4 and type(record[1]) is str and record[1] in _component_codecs():
        tensors = child(record[2])
        if not isinstance(tensors, dict):
            raise ValueError("invalid portable component tensors")
        value = _component_codecs()[record[1]][2](tensors)
        return _bindings(record[3], graphs, value, depth)
    if tag == "selection" and len(record) == 3:
        from rexgraph.selection import Selection
        return Selection.from_record(record[2], child(record[1]))
    if tag == "lineage" and len(record) == 2:
        from rexgraph.selection import Lineage
        return Lineage.from_record(record[1])
    if tag == "partition" and len(record) in (4, 5):
        from rexgraph.partition_state import RexPartition, PartitionState, partition_tower
        from rexgraph.graph import RexGraph
        rex, state, maps = child(record[1]), record[2], record[3]
        if not isinstance(rex, RexGraph) or not isinstance(state, dict) or set(state) != {"version", "source_state", "result_state", "selection_digest", "policy_digest", "closure"} or type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("invalid portable partition lineage")
        if any(type(state[k]) is not str for k in state if k != "version"):
            raise ValueError("invalid portable partition lineage values")
        for key in ("source_state", "result_state", "selection_digest"):
            if len(state[key]) != 64 or any(c not in "0123456789abcdef" for c in state[key]):
                raise ValueError("invalid partition state digest")
        boundaries, _ = partition_tower(rex)
        sizes = (rex.nV, *(b.shape[1] for b in boundaries))
        mapped_sizes = sizes
        if len(record) == 4 and isinstance(maps, tuple) and len(maps) > len(sizes):
            # Older writers could retain empty upper maps while dropping the
            # corresponding empty grade from their graph. Keep their coarse
            # legacy lineage; a complete source basis must not be invented.
            mapped_sizes = (*sizes, *(0 for _ in range(len(maps)-len(sizes))))
        if not isinstance(maps, tuple) or (maps and (len(maps) != len(mapped_sizes) or any(
                not isinstance(m, tuple) or len(m) != n or any(type(i) is not int or i < 0 for i in m)
                or tuple(sorted(set(m))) != m for m, n in zip(maps, mapped_sizes, strict=True)))):
            raise ValueError("invalid portable partition cell maps")
        values = ()
        if len(record) == 5:
            from rexgraph.selection import Lineage
            lineage = Lineage.from_record(record[4])
            if (lineage.result_state != state["result_state"] or lineage.source_state != state["source_state"]
                    or lineage.policy_digest != state["policy_digest"] or lineage.cell_maps != maps
                    or lineage.result_sizes != sizes or state["closure"] != "subcomplex"
                    or lineage.selection_digest != state["selection_digest"]):
                raise ValueError("portable partition lineage does not match its declared state")
            values = (lineage.source_sizes, lineage.requested, lineage.carried_state, lineage.parents)
        value = RexPartition(rex, PartitionState(**{k: v for k, v in state.items() if k != "version"}), maps, *values)
        value.check_state()
        return value
    if tag == "relations" and len(record) == 2:
        from rexgraph.graph import RexGraph
        graph = child(record[1])
        if not isinstance(graph, RexGraph):
            raise ValueError("invalid portable relation basis")
        return graph.relations
    if tag == "temporal" and len(record) == 3:
        from rexgraph.temporal_state import TemporalState, from_temporal_state
        return from_temporal_state(TemporalState(record[2], record[1]))
    if tag == "program" and len(record) == 3 and type(record[1]) is str and record[1] in _program_types():
        return _program_types()[record[1]].from_bytes(record[2])
    if tag == "recursion" and len(record) == 3:
        from .recursive_program import RecursionResult
        value = RecursionResult.from_record(child(record[1]))
        return _bindings(record[2], graphs, value, depth)
    if tag in ("Chain", "Cochain") and len(record) == 5:
        from rexgraph.cochain import Chain, Cochain
        if type(record[1]) is not int or record[1] < 0:
            raise ValueError("invalid result grade")
        values, keys, source = map(child, record[2:])
        if source is not None:
            from rexgraph.graph import RexGraph
            from rexgraph.cells import cell_count
            if not isinstance(source, RexGraph) or not hasattr(values, "shape") or len(values.shape) == 0 or values.shape[0] != cell_count(source, record[1]):
                raise ValueError("result coefficients do not match their source grade")
        return (Chain if tag == "Chain" else Cochain)(record[1], values, keys, source)
    if tag == "syntax" and len(record) == 3 and type(record[1]) is str:
        cls = _SYNTAX.get(record[1])
        if cls is None or not isinstance(record[2], dict) or set(record[2]) != {f.name for f in fields(cls)}:
            raise ValueError("unknown result syntax record")
        return cls(**{name: child(v) for name, v in record[2].items()})
    if tag == "predicate" and len(record) == 4 and all(type(v) is str for v in record[1:]):
        return PredicateResult(*record[1:])
    if tag in ("tuple", "list") and len(record) == 2 and isinstance(record[1], tuple):
        values = [child(v) for v in record[1]]
        return tuple(values) if tag == "tuple" else values
    if tag == "map" and len(record) == 2 and isinstance(record[1], tuple):
        result, prior = {}, None
        for entry in record[1]:
            if not isinstance(entry, tuple) or len(entry) != 2 or type(entry[0]) is not bytes:
                raise ValueError("invalid result map entry")
            key, value = entry
            if prior is not None and key <= prior:
                raise ValueError("noncanonical result map keys")
            decoded = unpack_value(key)
            try:
                if decoded in result:
                    raise ValueError("duplicate result map key")
                result[decoded] = child(value)
            except TypeError as exc:
                raise ValueError("invalid result map key") from exc
            prior = key
        return result
    raise ValueError(f"unknown or malformed result value tag {tag!r}")


def _metadata(record, count):
    if not isinstance(record["plan"], tuple) or any(type(v) is not str for v in record["plan"]):
        raise ValueError("result plan must contain text")
    for name in ("exactness", "aliases"):
        if not isinstance(record[name], tuple) or len(record[name]) not in (0, count):
            raise ValueError(f"result {name} must be aligned with its values")
    if any(v is not None and type(v) is not str for v in record["aliases"]):
        raise ValueError("invalid result alias")
    if any(v is not None and v not in {e.value for e in Exactness} for v in record["exactness"]):
        raise ValueError("unknown result arithmetic declaration")


def pack_result(result, *, max_bytes=_LIMIT):
    from .executor import Result
    if not isinstance(result, Result):
        raise TypeError("pack_result requires an RCQL Result")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("result byte limit must be a positive integer")
    encoder = _Encoder()
    record = {"version": 1, "values": encoder.node(result.values),
              "rewrites": tuple({"before": encoder.node(r.before), "after": encoder.node(r.after),
                                  "reason": r.reason, "predicates": encoder.node(r.predicates)} for r in result.rewrites),
              "plan": result.plan, "exactness": tuple(None if v is None else v.value for v in result.exactness),
              "aliases": result.aliases, "native_plan": encoder.node(result.native_plan),
              "provenance": encoder.node(result.provenance), "execution": encoder.node(result.execution),
              "graphs": encoder.graphs}
    _metadata(record, len(result.values))
    raw = pack_value(record)
    payload = _MAGIC + hashlib.sha256(raw).digest() + raw
    if len(payload) > max_bytes:
        raise ValueError("result exceeds the declared byte limit")
    return payload


def _unpack_record(payload, *, max_bytes=_LIMIT):
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("result byte limit must be a positive integer")
    if type(payload) is not bytes or len(payload) > max_bytes or not payload.startswith(_MAGIC) or len(payload) < 37:
        raise ValueError("invalid or oversized result framing")
    raw = payload[37:]
    if not hmac.compare_digest(payload[5:37], hashlib.sha256(raw).digest()):
        raise ValueError("result does not match its content digest")
    record = unpack_value(raw)
    names = {"version", "values", "rewrites", "plan", "exactness", "aliases", "native_plan", "provenance", "execution", "graphs"}
    if not isinstance(record, dict) or set(record) != names or type(record["version"]) is not int or record["version"] != 1:
        raise ValueError("unknown result record version or fields")
    if not isinstance(record["graphs"], list):
        raise ValueError("invalid result graph table")
    return record


def _record_cell_counts(payload):
    """Installed RCDB admission capability; inspect bases before rebuilding any."""
    from types import SimpleNamespace
    from rcdb.packet import _counts
    record = _unpack_record(payload)
    counts = {}
    def add(header, tensors, kind):
        if not isinstance(header, dict) or header.get("object_type") != kind or not isinstance(tensors, dict):
            raise ValueError("invalid result state for cell admission")
        for key, value in _counts(SimpleNamespace(header=header, tensors=tensors)).items():
            if type(value) is not int or value < 0:
                raise ValueError("invalid result cell counts")
            counts[key] = counts.get(key, 0)+value
    for graph in record["graphs"]:
        if not isinstance(graph, dict) or set(graph) != {"header", "tensors"}:
            raise ValueError("invalid result graph state")
        add(graph["header"], graph["tensors"], "RexGraph")
    # Temporal states are inline nodes; shared static bases are in the table.
    pending = [record]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            if len(value) == 3 and type(value[0]) is str and value[0] == "temporal":
                add(value[1], value[2], "TemporalRex")
            else:
                pending.extend(value)
    return counts


def unpack_result(payload, *, max_bytes=_LIMIT):
    from .executor import Result
    from .optimizer import Rewrite
    record = _unpack_record(payload, max_bytes=max_bytes)
    from rexgraph.state import RexState, from_state
    graphs = []
    for graph in record["graphs"]:
        if not isinstance(graph, dict) or set(graph) != {"header", "tensors"}:
            raise ValueError("invalid result graph state")
        graphs.append(from_state(RexState(graph["tensors"], graph["header"])))
    values = _node(record["values"], graphs)
    if not isinstance(values, tuple):
        raise ValueError("result values must be an ordered tuple")
    _metadata(record, len(values))
    rewrites = []
    if not isinstance(record["rewrites"], tuple):
        raise ValueError("invalid result rewrite table")
    for rewrite in record["rewrites"]:
        if not isinstance(rewrite, dict) or set(rewrite) != {"before", "after", "reason", "predicates"} or type(rewrite["reason"]) is not str:
            raise ValueError("invalid result rewrite record")
        before, after = (_node(rewrite[k], graphs) for k in ("before", "after"))
        predicates = _node(rewrite["predicates"], graphs)
        if not isinstance(before, ast.Expr) or not isinstance(after, ast.Expr) or not isinstance(predicates, tuple) or any(not isinstance(p, PredicateResult) for p in predicates):
            raise ValueError("invalid result rewrite syntax or predicates")
        rewrites.append(Rewrite(before, after, rewrite["reason"], predicates))
    native_plan, provenance, execution = (_node(record[k], graphs) for k in ("native_plan", "provenance", "execution"))
    if (native_plan is not None and not isinstance(native_plan, dict)) or any(
        not isinstance(items, tuple) or any(not isinstance(v, dict) for v in items) for items in (provenance, execution)
    ):
        raise ValueError("invalid result plan/provenance records")
    return Result(values, tuple(rewrites), record["plan"],
                  tuple(None if e is None else Exactness(e) for e in record["exactness"]),
                  native_plan, provenance, execution, record["aliases"])
