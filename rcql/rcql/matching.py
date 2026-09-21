"""Typed dependent iteration over native cell selections and catalog records.

This selects values, not an implicitly closed subcomplex. The existing DAG
executes every expression. Iteration introduces scoped input slots only; it
does not introduce a second graph representation or a SQL join engine.
"""
from dataclasses import dataclass, replace
from itertools import islice

from .ast import Alias, LetBinding, Literal, Parameter, Query
from .types import RCType, ValueKind, Exactness


@dataclass(frozen=True)
class MatchPlan:
    inputs: object
    query: Query
    collections: tuple[str, ...]
    slots: tuple[str, ...]

    @property
    def binding(self):
        return self.inputs.binding

    @property
    def effects(self):
        return self.inputs.effects

    def dag(self):
        return self.inputs.dag()

    def explain(self):
        plan = self.dag().explain()
        plan["match"] = {"bindings": [m.name for m in self.query.matches],
                         "collections": self.collections, "slots": self.slots,
                         "where": self.query.where is not None,
                         "order": ["DESC" if reverse else "ASC" for _, reverse in self.query.order],
                         "limit": self.query.limit, "offset": self.query.offset,
                         "closure": "none; selected cells retain their source basis",
                         "execution": "dependent native iteration; stable sort only when requested"}
        return plan


def plan_match(binding, query, *, parameters=None):
    from .planning import plan_query
    supplied = dict(parameters or {})
    prefix = "_rcql_match_"
    if any(name.startswith(prefix) for name in supplied):
        raise ValueError("reserved MATCH input parameter prefix")
    bindings = list(query.bindings)
    names = {item.name for item in bindings} | {query.source_alias}
    if any(name is not None and name.startswith(prefix) for name in names):
        raise ValueError("reserved MATCH local name prefix")
    collections, slots = [], []
    for i, match in enumerate(query.matches):
        if match.name in names or match.name.startswith(prefix):
            raise ValueError("MATCH requires distinct local names")
        names.add(match.name)
        candidate = plan_query(binding, Query(query.source, (match.value,), bindings=tuple(bindings),
                                               source_alias=query.source_alias), parameters=supplied)
        value = candidate.returns[0].result
        if isinstance(value, RCType) and value.kind is ValueKind.CELL_SET:
            item = value.with_(name="Cell", kind=ValueKind.CELL)
        elif isinstance(value, RCType) and value.kind in {ValueKind.CATALOG_ENTRY_SET, ValueKind.RECORD_SET}:
            item = value.with_(name="CatalogEntry" if value.kind is ValueKind.CATALOG_ENTRY_SET else "Record",
                               kind=ValueKind.CATALOG_ENTRY if value.kind is ValueKind.CATALOG_ENTRY_SET else ValueKind.RECORD)
        else:
            raise TypeError("MATCH IN requires a native CellSet, CatalogEntrySet or RecordSet")
        collection, slot = f"{prefix}collection_{i}", f"{prefix}value_{i}"
        if collection in names or slot in names:
            raise ValueError("reserved MATCH local name")
        collections.append(collection)
        slots.append(slot)
        bindings.extend((LetBinding(collection, match.value), LetBinding(match.name, Parameter(slot))))
        supplied[slot] = item
    for value in (query.limit, query.offset):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("MATCH limit/offset require nonnegative integers")
    predicate = query.where or Literal(True)
    returns = (predicate, *(v for v, _ in query.order), *query.returns)
    inputs = plan_query(binding, Query(query.source, returns, bindings=tuple(bindings),
                                      source_alias=query.source_alias), parameters=supplied)
    condition = inputs.returns[0].result
    if not isinstance(condition, bool) and not (isinstance(condition, RCType) and condition.kind is ValueKind.BOOLEAN):
        raise TypeError("WHERE requires a scalar boolean predicate")
    from .comparison import scalar_kind
    for item in inputs.returns[1:1 + len(query.order)]:
        scalar_kind(item.result)
    return MatchPlan(inputs, query, tuple(collections), tuple(slots))


def execute_match(executor, source, plan):
    from .executor import Result
    from .native_plan import plain
    from .planning import _plain_source
    query, dag = plan.query, plan.dag()
    explained = plan.explain()
    if query.explain:
        return Result((explained,), exactness=(Exactness.STRUCTURAL,), native_plan=explained)
    nodes = {node.id: node for node in dag.nodes}
    bindings = dict(dag.bindings)
    collection_ids = [bindings[name] for name in plan.collections]
    slot_ids = [next(n.id for n in dag.nodes if isinstance(n.expression.expr, Parameter)
                     and n.expression.expr.name == slot) for slot in plan.slots]
    observations = []
    shared = {}
    dependencies = {}
    for node in dag.nodes:
        dependencies[node.id] = node.id in slot_ids or any(dependencies[c] for c in node.inputs)
    invariant = {n.id for n in dag.nodes if n.reusable and not dependencies[n.id]}

    def evaluate(outputs, cache):
        cache.update(shared)
        needed = set()
        def visit(key):
            if key in cache or key in needed:
                return
            needed.add(key)
            for child in nodes[key].inputs:
                visit(child)
        for key in outputs:
            visit(key)
        fragment = replace(dag, nodes=tuple(n for n in dag.nodes if n.id in needed))
        _, readings = executor._execute_dag(fragment, source, computed=cache)
        observations.extend(readings)
        shared.update((key, cache[key]) for key in needed if key in invariant)
        return tuple(cache[key] for key in outputs)

    seed = {}
    # LET is eager, even if the selected collection is empty or LIMIT is zero.
    evaluate([bindings[item.name] for item in query.bindings], seed)
    n_order = len(query.order)
    def rows(depth, cache):
        if depth == len(slot_ids):
            if evaluate(dag.outputs[:1], cache)[0]:
                values = evaluate(dag.outputs[1:], cache)
                yield values[:n_order], values[n_order:]
            return
        collection, = evaluate((collection_ids[depth],), cache)
        from rexgraph.cells import CellSet, Cell
        iterable = (Cell(collection.source, collection.grade, i) for i in collection.indices) if isinstance(
            collection, CellSet) else iter(collection)
        for value in iterable:
            child = cache.copy()
            child[slot_ids[depth]] = value
            yield from rows(depth + 1, child)
    selected = rows(0, seed)
    if query.order:
        from .comparison import order_key, scalar_kind
        selected = list(selected)
        for i in reversed(range(n_order)):
            kinds = {scalar_kind(row[0][i]) for row in selected} - {"none"}
            if len(kinds) > 1:
                raise TypeError("ORDER BY requires compatible scalar readings in each column")
            selected.sort(key=lambda row, i=i: order_key(row[0][i]), reverse=query.order[i][1])
    stop = None if query.limit is None else query.offset + query.limit
    result = tuple(row for _, row in islice(selected, query.offset, stop))
    provenance = plain({"source_state": _plain_source(plan.inputs.binding.ref),
                        "policy_digest": plan.inputs.binding.ref.policy_digest,
                        "logical_operator": "MATCH", "row_count": len(result), "closure": "none"})
    explained["match"]["return_aliases"] = [e.name if isinstance(e, Alias) else None for e in query.returns]
    return Result((result,), plan=("MATCH",), exactness=(Exactness.STRUCTURAL,), native_plan=explained,
                  provenance=(provenance,), execution=tuple(observations))
