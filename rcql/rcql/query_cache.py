"""Explicit native result reuse with checked sources and finite stored values."""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping
from threading import RLock

from .program_codec import dumps, loads, digest, implementation_identity
from .capabilities import BoundSource
from .source_context import field_references


def _fingerprint(value, *, seen=None, depth=0):
    if depth > 64:
        raise ValueError("cache parameters are nested too deeply or cyclic")
    seen = frozenset() if seen is None else seen
    if isinstance(value, (tuple, list, Mapping)):
        if id(value) in seen:
            raise ValueError("cache parameters are cyclic")
        seen = seen | {id(value)}
    from rexgraph.tensor_field import TensorField, FieldSource
    from rexgraph.coordinate_map import _is_action
    from rexgraph.tensor_moment import CoordinatePairing
    from rexgraph.coordinate_map import CoordinateMetric
    from rexgraph.section_calculus import SectionRecipe, SectionFamily, SectionImage
    from rexgraph.tensor_field import TensorChannels
    from rexgraph.tensor_moment import MomentSpan
    from rexgraph.model_state import ModelState, ModelOutput, ModelInput, ModelBatch
    from fractions import Fraction
    from .name_relation import NameRelation
    from .program import Program
    from .program_transformation import ProgramTransformation
    from .recursive_program import RecursiveProgram, RecursionResult
    from rexgraph.affine_feedback import AffineFeedback
    from rexgraph.selection import Selection, Lineage
    from rexgraph.partition_state import RexPartition
    if isinstance(value, (Selection, Lineage)):
        return (type(value).__name__, value.digest)
    if isinstance(value, RexPartition):
        value.check_state()
        return ("partition", value.digest, value.cell_maps,
                None if not value.source_sizes else value.lineage.digest)
    if isinstance(value, (NameRelation, RecursiveProgram, ProgramTransformation, Program)):
        return (type(value).__name__, value.coefficient_digest)
    if isinstance(value, (RecursionResult, AffineFeedback)):
        return (type(value).__name__, value.coefficient_digest,
                tuple(sorted(r.coefficient_digest for r in field_references(value))))
    if value is None or type(value) in (bool, int, float, str, bytes) or isinstance(value, Fraction):
        return ("literal", dumps(value))
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_fingerprint(v, seen=seen, depth=depth+1) for v in value))
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        return ("mapping", tuple((k, _fingerprint(value[k], seen=seen, depth=depth+1)) for k in sorted(value)))
    if isinstance(value, TensorChannels):
        return ("channels", value.names, value.declaration_digest,
                tuple(_fingerprint(v, seen=seen, depth=depth+1) for v in value.fields),
                tuple(_fingerprint(v, seen=seen, depth=depth+1) for v in value.endpoint_sources))
    if isinstance(value, (TensorField, FieldSource, CoordinatePairing, CoordinateMetric,
                          SectionRecipe, SectionFamily, SectionImage, TensorChannels, MomentSpan,
                          ModelState, ModelOutput, ModelInput, ModelBatch)) or _is_action(value):
        check = getattr(value, "check_state", None)
        if callable(check):
            check()
        refs = field_references(value)
        return (type(value).__module__, type(value).__name__, value.coefficient_digest,
                tuple(sorted(ref.coefficient_digest for ref in refs)))
    from rexgraph.value_codec import pack_value
    import hashlib
    try:
        return ("native-value", hashlib.sha256(pack_value(value)).hexdigest())
    except TypeError as exc:
        raise TypeError("cache parameters require a certified finite identity") from exc



def _legacy_restore_result(record, key, references):
    from rexgraph.tensor_field import FieldSource
    from .executor import Result
    from .optimizer import Rewrite
    from .types import Exactness, PredicateResult
    meta = loads(record.get_metadata(1, 0, "rcql_cache_manifest"))
    if meta.get("schema") != "rcql.cached-result" or meta.get("version") != 1 or meta.get("key") != key:
        raise ValueError("cache result identity does not match the requested calculation")
    def bind_value(value):
        if isinstance(value, FieldSource):
            identity = (value.state_digest, value.record_id, value.version)
            if identity not in references:
                raise ValueError("cached result has an unbound source dependency")
            return value.bind(references[identity])
        if isinstance(value, tuple):
            return tuple(bind_value(v) for v in value)
        if is_dataclass(value) and type(value).__module__.startswith("rexgraph."):
            return replace(value, **{f.name: bind_value(getattr(value, f.name))
                                      for f in fields(value) if f.init})
        return value
    def take(node):
        if set(node) == {"recursion"}:
            from .recursive_program import RecursionResult
            nested = record.get_metadata(1, 0, node["recursion"])
            detached = RecursionResult.from_record(nested)
            refs = []
            for ref in detached.dependencies:
                identity = (ref.state_digest, ref.record_id, ref.version)
                if identity not in references:
                    raise ValueError("cached recursion has an unbound dependency")
                refs.append(ref.bind(references[identity]))
            return RecursionResult.from_record(nested, refs)
        if set(node) == {"operation_name"}:
            from .name_relation import NameRelation
            return NameRelation.from_bytes(node["operation_name"])
        if set(node) == {"finite_program"}:
            from .program import Program
            return Program.from_bytes(node["finite_program"])
        if set(node) == {"program_transformation"}:
            from .program_transformation import ProgramTransformation
            return ProgramTransformation.from_bytes(node["program_transformation"])
        if set(node) == {"native"}:
            return bind_value(record.get_metadata(1, 0, node["native"]))
        if set(node) == {"literal"}:
            return node["literal"]
        if set(node) == {"mapping"}:
            return {k: take(v) for k, v in node["mapping"]}
        if set(node) == {"sequence", "values"}:
            values = tuple(take(v) for v in node["values"])
            return values if node["sequence"] == "tuple" else list(values)
        raise ValueError("invalid cached value declaration")
    event = {"node": "cache", "operator": "RESULT_REUSE", "method_status": "observed",
             "methods": [{"method": "validated-native-result-reuse", "cache_key": key}],
             "exactness": "structural"}
    return Result(tuple(take(v) for v in meta["values"]),
                  tuple(Rewrite(a, b, reason, tuple(PredicateResult(*p) for p in predicates))
                        for a, b, reason, predicates in meta["rewrites"]),
                  tuple(meta["plan"]), tuple(Exactness(e) for e in meta["exactness"]),
                  dict(meta["native_plan"] or {}, cache={"key": key, "hit": True}),
                  tuple(dict(p, reuse={"key": key, "original_execution": p.get("execution")})
                        for p in meta["provenance"]), (event,), tuple(meta["aliases"]))


def _result_record(result, key, references):
    from rexgraph.graph import RexGraph
    from .result_codec import pack_result
    record = RexGraph.from_cells([1, [[0]]])
    record.set_provenance({"rcql_evidence": [ref.as_record() for ref in references]})
    record.attach_metadata(1, 0, "rcql_cache_key", key)
    record.attach_metadata(1, 0, "rcql_cache_payload", pack_result(result))
    return record


def _restore_result(record, key, references):
    payload = record.get_metadata(1, 0, "rcql_cache_payload")
    if payload is None:
        return _legacy_restore_result(record, key, references)
    if record.get_metadata(1, 0, "rcql_cache_key") != key:
        raise ValueError("cache result identity does not match the requested calculation")
    from .result_codec import unpack_result, bind_sources
    result = unpack_result(payload)
    for ref in field_references(result.values):
        if (ref.state_digest, ref.record_id, ref.version) not in references:
            raise ValueError("cached result has an unbound source dependency")
    values = bind_sources(result.values, references, strict=True)
    event = {"node": "cache", "operator": "RESULT_REUSE", "method_status": "observed",
             "methods": [{"method": "validated-native-result-reuse", "cache_key": key}],
             "exactness": "structural"}
    return replace(result, values=values,
                   native_plan=dict(result.native_plan or {}, cache={"key": key, "hit": True}),
                   provenance=tuple(dict(p, reuse={"key": key, "original_execution": p.get("execution")})
                                    for p in result.provenance), execution=(event,))


class QueryCache:
    """A separate RCDB result store, never the authority for source state."""

    def __init__(self, store, *, namespace="rcql_result"):
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("cache namespace must be nonempty")
        if isinstance(store, BoundSource):
            for permission in ("read", "identity", "mutate"):
                store.require(permission)
            store = store.value
        if not callable(getattr(store, "read_record", None)) or not callable(getattr(store, "commit_mutation", None)):
            raise TypeError("query cache requires an explicit native RCDB result store")
        self.store = store
        self.namespace = namespace
        self._lock = RLock()

    def execute(self, executor, query):
        from .executor import Executor
        from .ast import Query, StructuralEdit, Parameter
        from .planning import plan_query
        from .types import Effect, ValueKind
        from rexgraph.tensor_field import FieldSource
        from .capabilities import BoundSource
        if not isinstance(executor, Executor) or not isinstance(query, Query) or query.explain or query.matches:
            raise TypeError("persistent reuse requires a finite ordinary query without MATCH")
        source = executor._eval_source(query.source)
        binding = executor._planning_binding(query.source, source)
        if binding.schema.kind is not ValueKind.REX:
            raise ValueError("result reuse requires a selected native snapshot")
        plan = plan_query(binding, query, parameters=executor.params)
        for node in plan.dag().nodes:
            call = node.expression.call
            if isinstance(node.expression.expr, StructuralEdit) or (call is not None and
                (not node.reusable or not call.signature.memoizable or
                 call.signature.effects - {Effect.READ} or "train" in call.signature.requires)):
                raise ValueError("result reuse refuses observable reads and effectful operations")
        ref = FieldSource(binding.value, binding.ref.record_id, binding.ref.record_version, binding.ref.state_digest)
        all_refs = (ref, *field_references(executor.params))
        references = {(r.state_digest, r.record_id, r.version): r.source for r in all_refs if r.source is not None}
        if executor.evidence is not None:
            executor.evidence.validate_binding(binding)
            executor.evidence.validate_values(executor.params)
        code = implementation_identity()
        def identity():
            ref.check()
            for r in all_refs:
                r.check()
            account = ("rcql.result-cache.v2", query, binding.ref.name, ref.as_record(),
                       binding.source.policy.digest, code, _fingerprint(executor.params),
                       None if executor.evidence is None else executor.evidence.digest)
            if executor.exactness_policy != "declared":
                account += (("evaluation-policy", executor.exactness_policy),)
            return digest(account)
        key = identity()
        record_id = self.namespace+"/"+key
        with self._lock:
            snapshot = self.store.read_record(record_id)
            if snapshot is not None:
                result = _restore_result(snapshot.value, key, references)
                if executor.exactness_policy == "exact":
                    from .evaluation_policy import require_exact
                    for value in result.values:
                        require_exact(value)
                if identity() != key:
                    raise ValueError("inputs changed while reading the cached result")
                if executor.evidence is not None:
                    executor.evidence.validate_values(result.values)
                return result
            pinned = BoundSource(binding.value, binding.source.policy, ref=binding.ref, temporal=binding.temporal)
            child = Executor(sources={"cache_source": pinned}, params=executor.params,
                             artifacts=executor.artifacts, scheduler=executor.scheduler, evidence=executor.evidence,
                             exactness=executor.exactness_policy)
            result = child.execute(replace(query, source=Parameter("cache_source")))
            if identity() != key or implementation_identity() != code:
                raise ValueError("calculation dependencies changed before cache publication")
            for result_ref in field_references(result.values):
                if (result_ref.state_digest, result_ref.record_id, result_ref.version) not in references:
                    raise ValueError("cache output contains a source not declared by its inputs")
            record = _result_record(result, key, all_refs)
            self.store.commit_mutation(record_id, record, expected_version=0)
            return replace(result, native_plan=dict(result.native_plan or {}, cache={"key": key, "hit": False}))
