"""Native expression DAG: deterministic topological IDs, not a table/join plan.

The serialized form is an inspection protocol, not executable deserialization.
Live values stay in the bound plan. Reuse is query local and transitive: a pure
parent of an observable read must not suppress that read on its next occurrence.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

from .ast import Alias, Call, ListExpr, Literal, Member, Parameter, Reference, StructuralEdit, Comparison
from .planning import _plain_literal, _plain_source
from .types import RCType


def plain(value):
    """JSON safe metadata without repr of arbitrary live objects."""
    if isinstance(value, Fraction):
        return _plain_literal(value)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"python_type": type(value).__name__}


def method_plan(expression):
    if isinstance(expression.expr, Comparison):
        return {"status": "selected", "method": "scalar-coefficient-comparison"}
    if isinstance(expression.expr, StructuralEdit):
        return {"status": "selected", "method": "native-owned-structural-edit",
                "adapter": "rexgraph.structural_edit.edit_relations"}
    if isinstance(expression.expr, ListExpr):
        return {"status": "selected", "method": "construct-list"}
    if isinstance(expression.expr, Member):
        return {"status": "selected", "method": "checked-record-projection"}
    if expression.call is None:
        return {"status": "input", "method": None}
    name = expression.call.operator
    result = expression.result
    args = expression.call.args
    exact = isinstance(result, RCType) and result.exactness is not None and result.exactness.value in {"integer", "rational"}
    from .model_contracts import ARGUMENTS as MODEL_ARGUMENTS
    if name in MODEL_ARGUMENTS:
        return {"status": "selected", "method": "native-model-lifecycle", "adapter": expression.call.signature.implementation_key}
    from .section_contracts import ARGUMENTS as SECTION_ARGUMENTS
    if name in SECTION_ARGUMENTS:
        return {"status": "selected", "method": "exact-section-calculus",
                "adapter": "rexgraph.section_calculus." + name.lower()}
    from .tensor_contracts import ARGUMENTS as TENSOR_ARGUMENTS
    if name in TENSOR_ARGUMENTS:
        return {"status": "selected", "method": "rational-retained-tensor-action",
                "adapter": "rexgraph.retained_tensor." + name.lower()}
    method = None
    if name in {"COORDINATE_APPLY", "OPERATION_DELTA", "INJECTION_DELTA", "WORD_DELTA", "KERNEL_MOMENTS"}:
        method = {"COORDINATE_APPLY": "core-exact-coordinate-action", "OPERATION_DELTA": "core-exact-temporal-operation",
                  "INJECTION_DELTA": "core-exact-temporal-operation", "WORD_DELTA": "core-exact-temporal-word",
                  "KERNEL_MOMENTS": "core-exact-moment-kernel"}[name]
    elif name in {"COUNT", "SUM", "MEAN"}:
        method = "sequence-count" if name == "COUNT" else f"scalar-sequence-{name.lower()}"
    elif name == "DIFF":
        method = "core-exact-boundary-difference"
    elif name in {"CAYLEY", "COMPLEX_STRUCTURE", "RATIONAL_ROTATION"}:
        method = "core-factored-rational-transform"
    elif name == "PAGERANK_EXACT":
        method = "rational-sparse-ranking-solve"
    elif name in {"MARKOV_VIEW", "PAGERANK"}:
        method = "core-tensor-markov" if name == "MARKOV_VIEW" else "core-compiled-pagerank"
    elif name == "TEXT_OVERLAP_VIEW":
        method = "core-primary-text-overlap"
    elif name in {"DOCUMENT_FIELD", "SECTION_RESPONSE"}:
        method = "core-sparse-rational-response"
    elif name in {"CLOSURE", "SEMANTIC_CLOSURE"}:
        method = "native-induced-structural-closure"
    elif name in {"FIELD_DELTA", "FIELD_DELTA_MOMENT", "ORIENTED_FIELD_DELTA_MOMENT"}:
        method = "core-exact-correspondence-defects"
    elif name in {"BOUNDARY", "COBOUNDARY"} and isinstance(args[0], int):
        method = ("sparse-operator-handle" if len(args) == 1 else
                  "exact-incidence-action" if exact else "sparse-boundary-action")
    elif name in {"RANK", "NULLITY"}:
        # Branch selection depends on actual columns. This is a strategy, not
        # a claim that EXPLAIN has executed elimination or found a memo hit.
        method = "certified-exact-rank"
    elif name == "BETTI":
        method = "native-exact-betti"
    elif name == "CHAIN_MAP":
        method = "exact-sparse-chain-map-verification"
    elif name == "GLUE":
        method = "exact-all-pair-section-gluing"
    elif name == "SECTION_CHECK":
        method = "exact-incidence-section-check"
    elif name == "HODGE_OPERATOR":
        method = "factored-hodge-handle"
    elif name == "ADJOINT":
        method = "factored-metric-adjoint-handle"
    elif name in {"HODGE_DOWN", "HODGE_UP", "HODGE_SUM", "HODGE_DIFFERENCE"}:
        method = "factored-weighted-hodge-handle"
    elif name in {"DIRAC", "ANTI_DIRAC"}:
        method = "factored-weighted-dirac-handle"
    elif name in {"COMMUTATOR", "ANTICOMMUTATOR"}:
        method = "factored-operator-bracket-handle"
    elif name == "GRADED_CHAIN":
        method = "graded-chain-construction"
    elif name == "GRADE_COMPONENT":
        method = "graded-chain-component"
    elif name == "METRIC":
        method = "identity-metric" if len(args) == 1 or args[1] is None else "positive-diagonal-metric"
    elif name == "RESOLVENT":
        method = "identity-plus-operator-resolvent"
    elif name in {"ACCESS", "ACCESS_TYPES"}:
        method = "rational-sparse-accession" if exact else "csr-accession"
    elif name in {"CO_RELATE", "MOMENT_TENSOR"} and result.family_metric is not None:
        method = "rational-factored-family-contraction" if exact else "csr-hermitian-family-contraction"
    elif name == "CO_RELATE" and result.cross_metric is not None:
        method = "rational-sparse-cross-contraction" if exact else "csr-sesquilinear-cross-contraction"
    elif name in {"CO_RELATE", "MOMENT_TENSOR"}:
        method = "rational-type-contraction" if exact else "hermitian-type-contraction"
    elif name == "INTEGRATE":
        method = "rational-dual-pairing" if exact else "numeric-dual-pairing"
    elif name == "MOMENT" or (name in {"QUADRANCE", "SPREAD"} and result.metric is not None):
        method = "rational-metric-contraction" if exact else "hermitian-metric-contraction"
    elif name == "CHANNEL":
        method = "factored-channel-handle"
    elif name == "APPLY" and ((args[0].operator is not None and args[0].operator.construction == "operator-bracket") or (
            args[0].graded_operator is not None and args[0].graded_operator.construction == "graded-operator-bracket")):
        method = "rational-operator-bracket-action" if exact else "factored-operator-bracket-action"
    elif name == "APPLY" and args[0].graded_operator is not None:
        method = "rational-weighted-dirac-action" if exact else "factored-weighted-dirac-action"
    elif name == "APPLY" and args[0].operator.construction == "channel":
        method = "rational-channel-action" if exact else "factored-channel-action"
    elif name == "APPLY" and args[0].operator.construction == "metric-adjoint":
        method = "rational-metric-adjoint-action" if exact else "factored-metric-adjoint-action"
    elif name == "APPLY" and args[0].operator.construction == "weighted-hodge":
        method = "rational-weighted-hodge-action" if exact else "factored-weighted-hodge-action"
    elif name == "STAR_CHARACTER":
        method = "exact-star-mean" if exact else "incidence-star-mean"
    elif name in {"SCALE_MOMENT", "CHARACTER_ENERGY"}:
        order = args[1] if name == "SCALE_MOMENT" else 2
        method = ("identity-moment" if order == 0 else
                  "exact-channel-diagonal" if exact else
                  "channel-diagonal" if order == 1 and args[0].operator.construction == "channel" else
                  "sparse-row-quadrance" if name == "CHARACTER_ENERGY" else "sparse-halved-moment")
    elif name == "APPLY" and args[0].operator.construction == "hodge_operator":
        method = "factored-hodge-action"
    elif name == "CHARACTER" and exact:
        method = "exact-incidence-diagonal-character"
    elif name in {"QUADRANCE", "SPREAD"}:
        method = "rational-contraction" if exact else "hermitian-contraction"
    elif name == "ZERO":
        method = "integer-zero"
    return {
        "status": "selected" if method else "deferred",
        "method": method,
        "adapter": expression.call.signature.implementation_key,
        "arithmetic": None if not isinstance(result, RCType) or result.exactness is None else result.exactness.value,
        "note": None if method else "adapter dispatch retained; runtime kernel selection is not inferred from a name or result dtype",
    }


@dataclass(frozen=True)
class NativeNode:
    id: str
    inputs: tuple[str, ...]
    expression: object
    reusable: bool

    @property
    def operator(self):
        return None if self.expression.call is None else self.expression.call.operator

    def explain(self):
        expression = self.expression
        if isinstance(expression.expr, Comparison):
            payload = expression.explain()
            payload['kind'] = 'comparison'
        elif isinstance(expression.expr, StructuralEdit):
            payload = expression.explain()
            payload['kind'] = 'structural-edit'
        elif isinstance(expression.expr, (ListExpr, Member)):
            payload = expression.explain()
            payload['kind'] = 'list' if isinstance(expression.expr, ListExpr) else 'member'
        elif expression.call is not None:
            payload = expression.call.explain()
            payload["kind"] = "call"
            payload["predicates"] = [vars(item) for item in expression.predicates]
        else:
            payload = {"kind": "parameter" if isinstance(expression.expr, Parameter) else "literal",
                       "value": _plain_literal(expression.result)}
            if isinstance(expression.expr, Parameter):
                payload["parameter"] = expression.expr.name
        return plain(dict(payload, id=self.id, inputs=self.inputs, source="s0",
                          reusable=self.reusable, physical=method_plan(expression)))


@dataclass(frozen=True)
class NativePlan:
    binding: object
    nodes: tuple[NativeNode, ...]
    outputs: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...] = ()
    aliases: tuple[str | None, ...] = ()
    source_alias: str | None = None

    def explain(self):
        result = {
            "schema": "rcql.native-plan", "version": 1,
            "source": {"id": "s0", "kind": self.binding.schema.kind.value,
                       "state": _plain_source(self.binding.ref), "policy_digest": self.binding.ref.policy_digest},
            "nodes": [node.explain() for node in self.nodes],
            "outputs": self.outputs,
            "bindings": [{"name": name, "node": node} for name, node in self.bindings],
        }
        if any(self.aliases):
            result['return_aliases'] = self.aliases
        if self.source_alias is not None:
            result['source_alias'] = self.source_alias
        return plain(result)


def _literal_key(value):
    from rexgraph.model_state import ModelState, ModelOutput, ModelInput, ModelBatch, ModelTimeline
    if isinstance(value, (ModelState, ModelOutput, ModelInput, ModelBatch, ModelTimeline)):
        value.check_state()
        return type(value), value.coefficient_digest
    if value is None or type(value) in (bool, int, float, str, bytes, Fraction):
        return type(value), value
    if isinstance(value, tuple):
        parts = tuple(_literal_key(item) for item in value)
        if all(item is not None for item in parts):
            return tuple, parts
    return None


def lower(plan):
    nodes, interned = [], {}
    bound = {}

    def visit(expression):
        if isinstance(expression.expr, Alias):
            return visit(expression.children[0])
        if isinstance(expression.expr, Reference):
            # A LET captures one evaluation, even for an observable read. Reusing
            # its value must not re run the adapter; this is not implicit CSE.
            if expression.expr.name not in bound:
                bound[expression.expr.name] = visit(expression.children[0])[0]
            return bound[expression.expr.name], True
        inputs = tuple(visit(child) for child in expression.children)
        expr = expression.expr
        key = None
        if isinstance(expr, Parameter):
            key = ("parameter", expr.name)
        elif isinstance(expr, Literal):
            key = _literal_key(expr.value)
        elif isinstance(expr, (ListExpr, Member)) and all(reuse for _, reuse in inputs):
            key = (type(expr), expr.name if isinstance(expr, Member) else None,
                   tuple(node_id for node_id, _ in inputs))
        elif (isinstance(expr, Call) and expression.call.signature.memoizable
              and all(reuse for _, reuse in inputs)):
            key = ("call", expression.call.operator, tuple(node_id for node_id, _ in inputs))
        if key is not None and key in interned:
            return interned[key], True
        node_id = f"n{len(nodes)}"
        nodes.append(NativeNode(node_id, tuple(i for i, _ in inputs), expression, key is not None))
        if key is not None:
            interned[key] = node_id
        return node_id, key is not None

    for name, expression in plan.bindings:
        bound[name] = visit(expression)[0]
    outputs = tuple(visit(expression)[0] for expression in plan.returns)
    return NativePlan(plan.binding, tuple(nodes), outputs, tuple(bound.items()),
                      tuple(expr.expr.name if isinstance(expr.expr, Alias) else None for expr in plan.returns),
                      plan.query.source_alias)
