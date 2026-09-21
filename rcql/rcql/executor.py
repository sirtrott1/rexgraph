"""RCQL execution."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any

from .ast import (
    Alias,
    Call,
    Expr,
    ListExpr,
    Literal,
    Member,
    MutationQuery,
    Parameter,
    Query,
    Reference,
    StructuralEdit,
    Comparison,
)
from .operators import get_operator
from .optimizer import Rewrite, optimize
from .types import Exactness


@dataclass(frozen=True)
class Result:
    values: tuple[Any, ...]
    rewrites: tuple[Rewrite, ...] = ()
    plan: tuple[str, ...] = ()
    exactness: tuple[Exactness, ...] = ()
    native_plan: dict | None = None
    provenance: tuple[dict, ...] = ()
    execution: tuple[dict, ...] = ()
    aliases: tuple[str | None, ...] = ()

    @property
    def named_values(self):
        return {name: value for name, value in zip(self.aliases, self.values, strict=False) if name is not None}


# A direct C1 temporal field is carried on the reconstructed current Rex, not on the
# TemporalRex history object that supplied it.  These operators consume that carrier's
# actual source basis.  This is source routing, not a projection: the C1 field itself is
# passed unchanged and the temporal source remains in its provenance contract.
_CARRIER_SOURCE_OPERATORS = frozenset({
    "ACCUMULATE", "HODGE", "HARMONIC", "HODGE_COORDS", "METRIC_CURVATURE", "QUADRANCE",
    "WINDING",
})


class Executor:
    """Evaluate RCQL against explicit source and parameter bindings."""

    def __init__(self, *, sources=None, params=None, artifacts=None, scheduler=None, evidence=None):
        from .artifact_services import ArtifactServices
        if artifacts is not None and not isinstance(artifacts, ArtifactServices):
            raise TypeError("artifacts must be ArtifactServices")
        self.sources = dict(sources or {})
        self.params = dict(params or {})
        self.artifacts = artifacts
        if scheduler is not None:
            from .scheduling import PlanScheduler
            if not isinstance(scheduler, PlanScheduler):
                raise TypeError("scheduler must be PlanScheduler")
        if evidence is not None:
            from .source_context import SnapshotContext
            if not isinstance(evidence, SnapshotContext):
                raise TypeError("evidence must be SnapshotContext")
        self.scheduler = scheduler
        self.evidence = evidence

    def execute_cached(self, query, cache):
        from .query_cache import QueryCache
        if not isinstance(cache, QueryCache):
            raise TypeError("execute_cached requires an explicit QueryCache")
        return cache.execute(self, query)

    def execute_program(self, program, *, explain=False):
        from .program import Program
        if not isinstance(program, Program):
            raise TypeError("execute_program requires a finite Program")
        return program.execute(self, explain=explain)

    def execute_recursive(self, program, entry, arguments, *, source, limits=None,
                          history=True, memoize=True, explain=False):
        """Run an explicit recursive group through the ordinary query machinery."""
        from .recursive_program import RecursiveProgram
        from .ast import Query, Parameter, Call, Literal
        if not isinstance(program, RecursiveProgram):
            raise TypeError("recursive execution requires a declared RecursiveProgram")
        source_expr = Parameter(source) if isinstance(source, str) else source
        operator = "RECURSIVE_EXPLAIN" if explain else "RECURSIVE_RUN"
        arguments_expr = (Parameter("recursive_definition"), Literal(entry), Parameter("recursive_arguments"))
        if not explain:
            arguments_expr += (Literal(limits), Literal(history), Literal(memoize))
        child = Executor(sources=self.sources,
                         params={"recursive_definition": program, "recursive_arguments": arguments},
                         artifacts=self.artifacts, scheduler=self.scheduler, evidence=self.evidence)
        return child.execute(Query(source_expr, (Call(operator, arguments_expr),)))

    def topology(self, query):
        """Inspect operation ports without evaluating expression actions."""
        from .planning import plan_query
        from .plan_topology import PlanTopology
        if not isinstance(query, Query) or query.matches:
            raise TypeError("plan topology requires a finite query without dependent MATCH")
        source = self._eval_source(query.source)
        binding = self._planning_binding(query.source, source)
        if self.evidence is not None:
            self.evidence.validate_binding(binding)
            self.evidence.validate_values(self.params)
        plan = plan_query(binding, query, parameters=self.params)
        from .source_context import field_references
        return PlanTopology.from_plan(plan.dag(), field_references(self.params))

    @staticmethod
    def _unwrap(source, permission):
        from .capabilities import BoundSource
        if isinstance(source, BoundSource):
            permissions = (permission,) if isinstance(permission, str) else tuple(permission)
            for item in permissions:
                source.require(item)
            return source.value, source.policy
        return source, None

    @staticmethod
    def _carrier_source(source, args: tuple[object, ...], operator: str):
        """Use one carried field's current Rex for a temporal field operation.

        A whole temporal history is not a Rex snapshot, so handing it to a C1 action
        would lose the field's basis.  Routing is allowed only for the named single field
        operations and only when every carried source candidate agrees by identity;
        multi field alignment remains a static plan responsibility rather than a guess.
        """
        if operator not in _CARRIER_SOURCE_OPERATORS:
            return source
        if not hasattr(source, "reconstruct_at") or not hasattr(source, "T"):
            return source
        candidates = []
        for value in args:
            value = getattr(value, "cochain", value)
            candidate = getattr(value, "source", None)
            if candidate is not None:
                candidates.append(candidate)
        if candidates and all(candidate is candidates[0] for candidate in candidates):
            return candidates[0]
        return source

    def _eval_source(self, expr: Expr):
        if isinstance(expr, Parameter):
            if expr.name not in self.sources:
                raise KeyError(f"unknown source ${expr.name}")
            return self.sources[expr.name]
        if isinstance(expr, Call) and expr.name in {"REX", "CATALOG", "RCDB"} and len(expr.args) == 1:
            name = self._eval(expr.args[0], None)
            if not isinstance(name, str):
                raise TypeError(f"{expr.name} expects a bound source name, not a URI or object")
            if name not in self.sources:
                raise KeyError(f"unknown source {name!r}")
            if expr.name == "RCDB":
                from .binding import classify
                from .capabilities import BoundSource
                from .types import ValueKind
                source = self.sources[name]
                raw = source.value if isinstance(source, BoundSource) else source
                if classify(raw) is not ValueKind.RCDB_STORE:
                    raise TypeError(f"source {name!r} is not an RCDB store")
            return self.sources[name]
        if isinstance(expr, Call) and expr.name == "FILE" and len(expr.args) == 2:
            catalog_name = self._eval(expr.args[0], None)
            entry_name = self._eval(expr.args[1], None)
            if catalog_name not in self.sources:
                raise KeyError(f"unknown catalog {catalog_name!r}")
            catalog = self.sources[catalog_name]
            raw, policy = self._unwrap(catalog, "file_read")
            from rexgraph.io.catalog import FileCatalog
            if not isinstance(raw, FileCatalog):
                raise TypeError(f"source {catalog_name!r} is not a file catalog")
            value = raw.load(str(entry_name))
            if policy is not None:
                from .capabilities import BoundSource
                return BoundSource(value, policy)
            return value
        if isinstance(expr, Call) and expr.name == "PHRASE":
            if len(expr.args) != 1:
                raise TypeError(
                    f"PHRASE in FROM takes one local-section argument, got {len(expr.args)}"
                )
            section = self._eval(expr.args[0], None)
            from .phrase import PhraseSheaf

            if not isinstance(section, PhraseSheaf):
                raise TypeError("PHRASE in FROM expects a policy-aware PhraseSheaf")
            return section.as_bound_source()
        if isinstance(expr, Call) and expr.name in {"VALID_AT", "TRANSACTION_AT"}:
            selector = "RCDB_VALID_AT" if expr.name == "VALID_AT" else "RCDB_AS_OF"
            return self._eval_rcdb_source(Call(selector, expr.args))
        if isinstance(expr, Call) and expr.name in {
            "RCDB_GET", "RCDB_VERSION", "RCDB_AS_OF", "RCDB_VALID_AT",
        }:
            return self._eval_rcdb_source(expr)
        if isinstance(expr, Call) and expr.name == "AT" and len(expr.args) == 2:
            parent = self._eval_source(expr.args[0])
            from .capabilities import BoundSource

            temporal = parent.value if isinstance(parent, BoundSource) else parent
            version = self._eval(expr.args[1], None)
            if isinstance(version, bool) or not isinstance(version, int):
                raise TypeError("AT expects an integer TemporalRex snapshot version")
            if not hasattr(temporal, "reconstruct_at") or not hasattr(temporal, "T"):
                raise TypeError("AT expects a TemporalRex source")
            if version < 0 or version >= int(temporal.T):
                raise ValueError(
                    f"AT version must lie in [0, {int(temporal.T) - 1}], got {version}"
                )
            snapshot = temporal.reconstruct_at(version)
            if isinstance(parent, BoundSource):
                from .types import TemporalRef
                return BoundSource(snapshot, parent.policy, ref=parent.ref,
                                   temporal=TemporalRef(version=version))
            return snapshot
        if isinstance(expr, Call) and expr.name == "AT_TIME" and len(expr.args) == 2:
            parent = self._eval_source(expr.args[0])
            from .capabilities import BoundSource

            temporal = parent.value if isinstance(parent, BoundSource) else parent
            when = self._eval(expr.args[1], None)
            if isinstance(when, bool) or not isinstance(when, (int, float)):
                raise TypeError("AT_TIME expects a numeric TemporalRex clock time")
            if not isfinite(float(when)):
                raise ValueError("AT_TIME expects a finite TemporalRex clock time")
            if not hasattr(temporal, "reconstruct_at_time") or not hasattr(temporal, "T"):
                raise TypeError("AT_TIME expects a TemporalRex source")
            snapshot = temporal.reconstruct_at_time(float(when))
            if snapshot is None:
                raise ValueError(f"AT_TIME has no declared TemporalRex state at {when}")
            if isinstance(parent, BoundSource):
                from .types import TemporalRef
                return BoundSource(snapshot, parent.policy, ref=parent.ref,
                                   temporal=TemporalRef(as_of=float(when)))
            return snapshot
        raise TypeError("FROM expects a source parameter, REX(name), CATALOG(name), RCDB(name), "
                        "FILE(catalog, name), RCDB_GET(store, id), RCDB_VERSION(store, id, "
                        "version), RCDB_AS_OF(store, id, time), RCDB_VALID_AT(store, id, time), "
                        "TRANSACTION_AT(store, id, time), VALID_AT(store, id, time), "
                        "PHRASE(section), AT(temporal_source, version), or "
                        "AT_TIME(temporal_source, time)")

    def _eval_rcdb_source(self, expr: Call):
        """Resolve one selected RCDB state before planning its structural phrase.

        ``RCDB_GET`` keeps its existing one argument return expression. In FROM position
        it is instead a source transform: it consumes an explicit store source and an
        identity, then carries the decoded Rex and the same capability policy onward.
        ``RCDB_VERSION`` makes a persisted version selection explicit rather than
        overloading a scalar whose meaning could be a clock time. ``RCDB_AS_OF`` and
        ``RCDB_VALID_AT`` expose the store's bitemporal selectors with equally explicit
        transaction time and valid time meanings.
        """
        from .binding import bind, classify, resolve
        from .capabilities import BoundSource, SourcePolicy
        from .types import SourceRef, ValueKind

        expected = 2 if expr.name == "RCDB_GET" else 3
        if len(expr.args) != expected:
            raise TypeError(f"{expr.name} in FROM takes {expected} arguments, got {len(expr.args)}")
        parent_expr, record_expr = expr.args[:2]
        parent = self._eval_source(parent_expr)
        record_id = self._eval(record_expr, None)
        if not isinstance(record_id, str):
            raise TypeError(f"{expr.name} expects a string RCDB record id")

        parent_policy = parent.policy if isinstance(parent, BoundSource) else SourcePolicy.allow("*")
        parent_ref = parent.ref if isinstance(parent, BoundSource) else None
        parent_binding = bind(self._source_label(parent_expr),
                              parent.value if isinstance(parent, BoundSource) else parent,
                              parent_policy, source_ref=parent_ref)
        # Use the declared return form contract to check store kind and identity before a
        # storage adapter is reached. The source form has its own arity, but the surface
        # it reads is exactly RCDB_GET's one record identity lookup.
        resolve(parent_binding, "RCDB_GET", (record_id,))
        raw, _policy = self._unwrap(parent, "identity")
        version = None
        as_of = valid_at = None
        if expr.name == "RCDB_VERSION":
            version = self._eval(expr.args[2], None)
            if isinstance(version, bool) or not isinstance(version, int):
                raise TypeError("RCDB_VERSION expects an exact integer record version")
        elif expr.name in {"RCDB_AS_OF", "RCDB_VALID_AT"}:
            when = self._eval(expr.args[2], None)
            if isinstance(when, bool) or not isinstance(when, (int, float, Fraction)):
                raise TypeError(f"{expr.name} expects a numeric RCDB time")
            if not isfinite(float(when)):
                raise ValueError(f"{expr.name} expects a finite RCDB time")
            if expr.name == "RCDB_AS_OF":
                as_of = float(when)
            else:
                valid_at = float(when)
        snapshot = raw.read_record(record_id, version=version, as_of=as_of, valid_at=valid_at)
        if snapshot is None:
            state = f" version {version}" if version is not None else ""
            raise KeyError(f"RCDB record {record_id!r}{state} is not present")
        value = snapshot.value
        record_id, version = snapshot.record.id, snapshot.record.version
        kind = classify(value)
        if kind not in {ValueKind.REX, ValueKind.TEMPORAL_REX}:
            raise TypeError(f"{expr.name} record {record_id!r} is not a Rex or TemporalRex source")
        # The record id and version name the selected store entry. The canonical object
        # digest names the exact decoded state that structural operators will read, so an
        # EXPLAIN account cannot claim one state while the store returned another.
        ref = SourceRef(
            name=f"{parent_binding.ref.name}/{record_id}@{version}",
            state_digest=snapshot.state_digest,
            policy_digest=parent_policy.digest,
            record_id=record_id,
            record_version=version,
            record_as_of=as_of,
            record_valid_at=valid_at,
        )
        return BoundSource(value, parent_policy, ref=ref)

    @staticmethod
    def _source_label(expr: Expr) -> str:
        """Name a bound query source without evaluating an expression under it."""
        if isinstance(expr, Parameter):
            return expr.name
        if isinstance(expr, Call) and expr.name in {"REX", "CATALOG", "RCDB"}:
            if len(expr.args) == 1 and isinstance(expr.args[0], Literal):
                return str(expr.args[0].value)
            return expr.name.lower()
        if isinstance(expr, Call) and expr.name in {"AT", "AT_TIME"} and expr.args:
            return Executor._source_label(expr.args[0])
        if isinstance(expr, Call) and expr.name in {
            "RCDB_GET", "RCDB_VERSION", "RCDB_AS_OF", "RCDB_VALID_AT",
            "VALID_AT", "TRANSACTION_AT",
        } and expr.args:
            return Executor._source_label(expr.args[0])
        if isinstance(expr, Call) and expr.name == "PHRASE":
            return "phrase"
        if isinstance(expr, Call) and expr.name == "FILE":
            return "file"
        return "source"

    def _source_temporal(self, expr: Expr):
        """Attach a snapshot version to the static phrase binding when declared."""
        if not (isinstance(expr, Call) and expr.name in {"AT", "AT_TIME"} and len(expr.args) == 2):
            return None
        from .types import TemporalRef

        value = self._eval(expr.args[1], None)
        if expr.name == "AT":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("AT expects an integer TemporalRex snapshot version")
            return TemporalRef(version=value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("AT_TIME expects a numeric TemporalRex clock time")
        if not isfinite(float(value)):
            raise ValueError("AT_TIME expects a finite TemporalRex clock time")
        return TemporalRef(as_of=float(value))

    def _planning_binding(self, source_expr: Expr, source):
        """Make the same policy aware binding used by static phrase planning."""
        from .binding import bind
        from .capabilities import BoundSource, SourcePolicy

        if isinstance(source, BoundSource):
            binding = bind(self._source_label(source_expr), source.value, source.policy,
                        temporal=source.temporal or self._source_temporal(source_expr),
                        source_ref=source.ref)
        else:
            binding = bind(self._source_label(source_expr), source, SourcePolicy.allow("*"),
                           temporal=self._source_temporal(source_expr))
        return binding

    def _eval(self, expr: Expr, source, *, memo: dict[Call, object] | None = None):
        """Resolve a FROM argument only; expressions execute exclusively as typed DAGs."""
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, Parameter):
            if expr.name not in self.params:
                raise KeyError(f"unknown parameter ${expr.name}")
            return self.params[expr.name]
        raise TypeError("FROM arguments must be literals or bound parameters; operator calls require a typed query")

    def _validate_runtime_call(self, expression, args):
        """Validate actual derived values under the original source binding."""
        from .ast import Literal
        from .inference import infer
        from .planning import PlannedExpression, _carrier_literal
        from .validation import ValidationContext
        binding = expression.call.binding
        carried = self._carrier_source(binding.value, args, expression.call.operator)
        if carried is not binding.value:
            from .binding import bind
            binding = bind(binding.ref.name, carried, binding.source.policy, temporal=binding.temporal)
        children = tuple(PlannedExpression(Literal(value),
                         _carrier_literal(binding, value) or value) for value in args)
        infer(binding, expression.call.operator, tuple(c.result for c in children),
              context=ValidationContext(binding), children=children)

    def _execute_dag(self, dag, source, *, computed=None):
        if self.evidence is not None:
            self.evidence.validate_binding(dag.binding)
            self.evidence.validate_values(self.params)
        if self.scheduler is not None:
            return self.scheduler.evaluate(self, dag, source, computed)
        return self._execute_dag_serial(dag, source, computed=computed)

    def _execute_dag_serial(self, dag, source, *, computed=None):
        from .execution_trace import capture_methods
        from .native_plan import plain
        computed = {} if computed is None else computed
        observations = []
        for node in dag.nodes:
            expression = node.expression
            if node.operator is None:
                if isinstance(expression.expr, Comparison):
                    from .comparison import evaluate
                    value = evaluate(expression.expr.operation, *(computed[i] for i in node.inputs))
                elif isinstance(expression.expr, StructuralEdit):
                    from rexgraph.structural_edit import edit_relations
                    self._unwrap(source, "mutate")
                    value = edit_relations(computed[node.inputs[0]], expression.expr.operation,
                                           computed[node.inputs[1]])
                    observations.append({"node": node.id, "operator": expression.expr.operation,
                        "adapter": "rexgraph.structural_edit.edit_relations", "method_status": "observed",
                        "methods": [{"method": "native-owned-structural-edit"}], "exactness": "structural"})
                elif isinstance(expression.expr, ListExpr):
                    value = [computed[i] for i in node.inputs]
                elif isinstance(expression.expr, Member):
                    from .members import project
                    value = project(computed[node.inputs[0]], expression.expr.name)
                else:
                    value = (expression.expr.value if isinstance(expression.expr, Literal)
                             else self.params[expression.expr.name])
                computed[node.id] = value
                continue
            args = tuple(computed[i] for i in node.inputs)
            raw, policy = self._unwrap(source, expression.call.signature.requires)
            self._validate_runtime_call(expression, args)
            with capture_methods(policy_digest=expression.call.binding.ref.policy_digest,
                                 policy=policy, artifact_services=self.artifacts, binding=expression.call.binding) as methods:
                value = get_operator(node.operator).fn(self._carrier_source(raw, args, node.operator), *args)
            if policy is not None and node.operator in {"RCDB_LIST", "RCDB_SEARCH", "RCDB_HISTORY"}:
                value = policy.project_record(value)
            if self.evidence is not None:
                self.evidence.validate_values((args, value))
            computed[node.id] = value
            observations.append(plain({
                "node": node.id, "operator": node.operator,
                "adapter": expression.call.signature.implementation_key,
                "method_status": "observed" if methods else "unreported",
                "methods": methods, "exactness": value_exactness(value).value,
            }))
        return computed, observations

    def _execute_mutation(self, query, source):
        from rexgraph.io.catalog import object_digest

        from .mutation_plan import plan_mutation, validate_arguments
        from .native_plan import plain
        binding = self._planning_binding(query.source, source)
        planned = plan_mutation(binding, query, parameters=self.params)
        serialized = planned.explain()
        if query.explain:
            return Result((serialized,), plan=("COMMIT",), exactness=(Exactness.STRUCTURAL,),
                          native_plan=serialized)
        dag = planned.inputs.dag()
        computed, observations = self._execute_dag(dag, source)
        args = tuple(computed[i] for i in dag.outputs)
        from .types import ValueKind
        file = binding.schema.kind is ValueKind.CATALOG_ENTRY_SET
        validate_arguments(args, file=file)
        record_id, resulting, actor, valid_from, valid_to, expected, expected_hash = args
        raw, _policy = self._unwrap(source, ("mutate", "identity", "file_write") if file else ("mutate", "identity"))
        options = {"actor": actor, "valid_from": valid_from, "valid_to": valid_to}
        if expected is not None:
            options["expected_version"] = expected
        if file:
            options["expected_hash"] = expected_hash
        digest = object_digest(resulting)
        rec = raw.commit_mutation(record_id, resulting, **options)
        terminal = serialized["outputs"][0]
        event = plain({
            "node": terminal, "operator": "COMMIT", "adapter": "catalog.commit_mutation" if file else "rcdb.commit_mutation",
            "method_status": "observed", "exactness": "structural",
            "methods": [{"method": "recoverable-native-file-replacement" if file else "rcdb-temporal-mutation-commit", "record_id": rec.id,
                         "record_version": rec.version, "state_digest": digest,
                         "expected_version": expected, **({"backup": rec.backup, "expected_hash": expected_hash} if file else {})}],
        })
        observations.append(event)
        provenance = plain({"node": terminal, "logical_operator": "COMMIT",
                            "record_id": rec.id, "record_version": rec.version,
                            "state_digest": digest, "policy_digest": binding.ref.policy_digest,
                            "execution": event})
        return Result((rec,), plan=(f"COMMIT({record_id!r})",), exactness=(Exactness.STRUCTURAL,),
                      native_plan=serialized, provenance=(provenance,), execution=tuple(observations))

    def execute(self, query: Query | MutationQuery) -> Result:
        from .execution_trace import evidence_scope
        with evidence_scope(self.evidence):
            result = self._execute(query)
            if self.evidence is not None:
                from dataclasses import replace
                self.evidence.validate_values(result.values)
                evidence = self.evidence.as_record()
                result = replace(result, native_plan=dict(result.native_plan or {}, evidence=evidence),
                                 provenance=tuple(dict(item, evidence=evidence) for item in result.provenance))
            return result

    def _execute(self, query: Query | MutationQuery) -> Result:
        if not isinstance(query, (Query, MutationQuery)):
            raise TypeError("Executor.execute requires a typed Query or MutationQuery")
        if isinstance(query, MutationQuery) and (self.scheduler is not None or self.evidence is not None):
            raise ValueError("snapshot and parallel execution are read only; commit separately")
        source = self._eval_source(query.source)
        if isinstance(query, MutationQuery):
            return self._execute_mutation(query, source)
        from .planning import plan_query
        binding = self._planning_binding(query.source, source)
        if self.evidence is not None:
            self.evidence.validate_binding(binding)
            self.evidence.validate_values(self.params)
        if query.matches and self.scheduler is not None:
            raise ValueError("parallel plans do not infer a schedule for dependent MATCH iteration")
        if query.matches:
            from .matching import execute_match, plan_match
            return execute_match(self, source, plan_match(binding, query, parameters=self.params))
        original = plan_query(binding, query, parameters=self.params)
        planned, rewrites = optimize(query, plan=original, parameters=self.params)
        phrase = plan_query(binding, planned, parameters=self.params) if rewrites else original
        dag = phrase.dag()
        serialized = dag.explain()
        from dataclasses import asdict

        from .native_plan import plain
        rewrite_trace = [{"reason": item.reason, "before": format_expr(item.before),
                          "after": format_expr(item.after),
                          "predicates": [asdict(p) for p in item.predicates]} for item in rewrites]
        serialized["rewrites"] = rewrite_trace
        if planned.explain:
            # No expression operator is evaluated in this branch.  The source is bound
            # once so type/capability/provenance checks have a real contract, then the
            # whole AST is typed recursively and returned as a plain structural value.
            explanation = phrase.explain()
            explanation["native_plan"] = serialized
            return Result(
                (explanation,), tuple(rewrites),
                tuple(format_expr(expr) for expr in planned.returns),
                (Exactness.STRUCTURAL,), native_plan=serialized,
            )
        # A normal phrase receives the same contract check as EXPLAIN before even its
        # first adapter runs.  The returned plan is intentionally not discarded work:
        # it is the source/grade/basis/time proof for this execution, while runtime
        # remains responsible for data dependent bounds and numerical residuals.
        from .planning import _plain_source, _plain_type
        from .types import RCType
        computed, observations = self._execute_dag(dag, source)
        values = tuple(computed[i] for i in dag.outputs)
        plan = tuple(format_expr(expr) for expr in planned.returns)
        exactness = tuple(value_exactness(value) for value in values)
        logical_returns = tuple(expr.children[0] if isinstance(expr.expr, Alias) else expr
                                for expr in original.returns)
        provenance = tuple(plain({
            "node": node_id, "logical_operator": (logical.call.operator if logical.call else None),
            "reference": logical.expr.name if isinstance(logical.expr, Reference) else None,
            "source_state": _plain_source(binding.ref), "policy_digest": binding.ref.policy_digest,
            "result_type": _plain_type(expression.result) if isinstance(expression.result, RCType) else None,
            "exactness": arithmetic.value, "rewrites": rewrite_trace,
            "execution": next((item for item in observations if item["node"] == node_id), None),
            **({"alias": alias} if alias is not None else {}),
        }) for node_id, expression, logical, arithmetic, alias in zip(
            dag.outputs, phrase.returns, logical_returns, exactness, dag.aliases, strict=True))
        return Result(values, tuple(rewrites), plan, exactness, serialized, provenance, tuple(observations), dag.aliases)



def format_expr(expr: Expr) -> str:
    """Return a compact RCQL expression string."""
    if isinstance(expr, Comparison):
        return f"({format_expr(expr.left)} {expr.operation} {format_expr(expr.right)})"
    if isinstance(expr, StructuralEdit):
        return f"{format_expr(expr.state)} {expr.operation} {format_expr(expr.value)}"
    if isinstance(expr, Alias):
        return f"{format_expr(expr.value)} AS {expr.name}"
    if isinstance(expr, Member):
        return f"{format_expr(expr.value)}.{expr.name}"
    if isinstance(expr, ListExpr):
        return "[" + ", ".join(format_expr(item) for item in expr.items) + "]"
    if isinstance(expr, Literal):
        if isinstance(expr.value, Fraction):
            return f"{expr.value.numerator}/{expr.value.denominator}"
        from .planning import _plain_literal
        return repr(_plain_literal(expr.value))
    if isinstance(expr, Parameter):
        return "$" + expr.name
    if isinstance(expr, Reference):
        return expr.name
    if isinstance(expr, Call):
        return f"{expr.name}({', '.join(format_expr(a) for a in expr.args)})"
    return repr(expr)


def value_exactness(value: Any) -> Exactness:
    """Classify the numeric representation returned by one expression.

    This reads a finished value, which is not type inference: it cannot answer before the
    work happens and it cannot distinguish a rational that was rendered to a float from a
    float that was never exact. Signature driven inference replaces it wherever a
    signature exists; this remains the fallback for expressions that have none yet.

    Deliberately written without importing numpy. An array is recognised by carrying a
    ``dtype`` whose ``kind`` is a single character from the array protocol, so this
    classifies numpy values correctly without RCQL depending on numpy to do it. The
    exact tensor carriers come from the core library, and the only numpy in this stack
    belongs to the binary bundles beneath it.
    """
    from .program_transformation import ProgramTransformation
    if isinstance(value, ProgramTransformation):
        return Exactness.STRUCTURAL
    from rexgraph.model_state import ModelState, ModelOutput, ModelBatch, ModelTimeline, ModelInput
    if isinstance(value, (ModelState, ModelBatch, ModelTimeline, ModelInput)):
        return Exactness.STRUCTURAL
    if isinstance(value, ModelOutput):
        return Exactness.RATIONAL if value.arithmetic == "rational" else Exactness.APPROXIMATE
    from rexgraph.cells import CellBoundary, CellCoboundary, CompositeBinary
    from rexgraph.chain_map import ChainHomotopy, ChainMap, GradedMap, SymmetryGroup
    if isinstance(value, SymmetryGroup):
        return Exactness.STRUCTURAL
    from rexgraph.cell_neighborhood import Hyperslice
    from rexgraph.column_expansion import ColumnExpansion, ColumnLegs, PrimaryColumnLift
    from rexgraph.cochain import Chain, Cochain, Field
    from rexgraph.graded_metric import DiagonalMetric
    from rexgraph.hodge_coords import HodgeCoords
    from rexgraph.linear_operator import RexOperator
    from rexgraph.metric_field import MetricCurvature
    from rexgraph.operator_bracket import GradedOperatorBracket
    from rexgraph.sheaf import ExactGlueResult
    from rexgraph.temporal_signal import TemporalSignal, TemporalSignalFlow
    from rexgraph.type_accession import (
        AccessionFamily,
        CrossMetric,
        FamilyMetric,
        TypeAccession,
        TypedFamily,
        TypedMomentTensor,
        TypeView,
    )
    from rexgraph.weighted_dirac import GradedChain, WeightedDiracOperator

    from .operators import CharacterResult

    if isinstance(value, (WeightedDiracOperator, GradedOperatorBracket, Hyperslice, ColumnExpansion, ColumnLegs, PrimaryColumnLift)):
        return Exactness.STRUCTURAL
    if isinstance(value, GradedChain):
        return Exactness.RATIONAL if value.exact else Exactness.APPROXIMATE

    from rexgraph.tensor_field import TensorField, TensorChannels
    from rexgraph.tensor_moment import MomentSpan, TensorMoments, TensorMomentKernel, CoordinatePairing, RealizedPairing
    from rexgraph.temporal_field import MomentChange, TensorEvolution, ResolvedEvolution, NativeFieldEvolution, SectorTransport
    from rexgraph.attachment_field import AttachmentField, AttachmentObservation
    from rexgraph.native_field import NativeFieldCalculus
    if isinstance(value, (TensorField, TensorChannels, MomentSpan, TensorMoments, MomentChange)):
        return Exactness.RATIONAL
    if isinstance(value, (TensorMomentKernel, CoordinatePairing, RealizedPairing, TensorEvolution, ResolvedEvolution,
                          NativeFieldEvolution, SectorTransport, AttachmentField, AttachmentObservation, NativeFieldCalculus)):
        return Exactness.STRUCTURAL
    from rexgraph.coordinate_map import CoordinateMap, CoordinateWord, CoordinateDifference, CoordinateMetric
    from rexgraph.temporal_calculus import TemporalOperation, TemporalWord, TemporalMetrics, MomentKernel
    from rexgraph.type_accession import CoordinateField
    if isinstance(value, (CoordinateMap, CoordinateWord, CoordinateDifference, CoordinateMetric,
                          TemporalOperation, TemporalWord, TemporalMetrics, MomentKernel)):
        return Exactness.STRUCTURAL
    if isinstance(value, CoordinateField):
        return value_exactness(value.values)

    if isinstance(value, (TypeAccession, AccessionFamily, CrossMetric, FamilyMetric, GradedMap, ChainMap, ChainHomotopy)):
        return Exactness.STRUCTURAL
    if isinstance(value, (TypeView, TypedMomentTensor)):
        return value_exactness(value.values)
    if isinstance(value, TypedFamily):
        kinds = {value_exactness(v) for v in value.views}
        if kinds <= {Exactness.INTEGER, Exactness.RATIONAL}:
            return Exactness.RATIONAL if Exactness.RATIONAL in kinds else Exactness.INTEGER
        return Exactness.APPROXIMATE if kinds <= {Exactness.INTEGER, Exactness.RATIONAL, Exactness.APPROXIMATE} else Exactness.STRUCTURAL

    if isinstance(value, CharacterResult):
        coefficients = value['values']
        if all(isinstance(entry, Fraction) for entry in coefficients.flat):
            return Exactness.RATIONAL
        return value_exactness(coefficients)

    if isinstance(value, (bool, RexOperator, ExactGlueResult, DiagonalMetric)):
        return Exactness.STRUCTURAL
    if isinstance(value, int):
        return Exactness.INTEGER
    if isinstance(value, Fraction):
        return Exactness.RATIONAL
    if isinstance(value, (Chain, Cochain, Field)):
        return value_exactness(value.values)
    if isinstance(value, CellBoundary):
        return Exactness.STRUCTURAL if value.chain is None else value_exactness(value.chain)
    if isinstance(value, CellCoboundary):
        return value_exactness(value.cochain)
    if isinstance(value, CompositeBinary):
        return value_exactness(value.boundary)
    if isinstance(value, TemporalSignal):
        # The carrier is structural: individual channels retain their own
        # exactness, notably a numerical amplitude field beside exact topology.
        return Exactness.STRUCTURAL
    if isinstance(value, HodgeCoords):
        return Exactness.APPROXIMATE
    if isinstance(value, TemporalSignalFlow):
        return value_exactness(value.returned_boundary)
    if isinstance(value, MetricCurvature):
        # Integer relation metrics can produce rational local means and strain
        # through declared C1 share coefficients.  Classify the actual returned
        # field, not merely its input dtype.
        return value_exactness(value.local_mean)
    if isinstance(value, Mapping):
        # Compound field results must not hide the contract of their members.
        # In particular, SIGNAL_HODGE returns a named C1 split whose numerical
        # solver components are approximate even though their carrier/basis is
        # structurally well defined.  Preserve a uniform contract; a mixed
        # compound has no single arithmetic contract to promise.
        contracts = tuple(value_exactness(item) for item in value.values())
        if contracts and all(contract is contracts[0] for contract in contracts):
            return contracts[0]
        return Exactness.STRUCTURAL

    if isinstance(value, (tuple, list)):
        contracts = {value_exactness(item) for item in value}
        if not contracts:
            return Exactness.STRUCTURAL
        if contracts <= {Exactness.INTEGER, Exactness.RATIONAL}:
            return Exactness.RATIONAL if Exactness.RATIONAL in contracts else Exactness.INTEGER
        return contracts.pop() if len(contracts) == 1 else Exactness.STRUCTURAL

    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if kind is not None:
        if kind == "b":
            return Exactness.STRUCTURAL
        if kind in "iu":
            return Exactness.INTEGER
        if kind == "O":
            flat = getattr(value, "flat", None)
            if flat is not None:
                from numbers import Integral
                seen, integer, rational = False, True, True
                for entry in flat:
                    seen = True
                    integer &= isinstance(entry, Integral) and not isinstance(entry, bool)
                    rational &= isinstance(entry, (Integral, Fraction)) and not isinstance(entry, bool)
                if seen and integer:
                    return Exactness.INTEGER
                if rational:
                    return Exactness.RATIONAL
            return Exactness.STRUCTURAL
        if kind in "fc":
            return Exactness.APPROXIMATE
        return Exactness.STRUCTURAL

    if isinstance(value, (float, complex)):
        return Exactness.APPROXIMATE
    return Exactness.STRUCTURAL
