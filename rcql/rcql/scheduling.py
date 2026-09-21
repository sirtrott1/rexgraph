"""Explicit parallel evaluation through the existing Coordinator lanes."""
from __future__ import annotations

from dataclasses import replace
from contextvars import copy_context
from threading import Event


class QueryCancelledError(RuntimeError):
    pass


class PlanScheduler:
    """Evaluate ready pure nodes on declared thread lanes with ordered results."""

    def __init__(self, coordinator, *, width=4, cancel=None):
        from rexgraph.coordinator import Coordinator, LanePools
        if not isinstance(coordinator, Coordinator) or not isinstance(coordinator.pools, LanePools):
            raise TypeError("query scheduling requires a Coordinator with managed LanePools")
        if type(width) is not int or width < 1:
            raise ValueError("scheduler width must be positive")
        if cancel is not None and not isinstance(cancel, Event):
            raise TypeError("cancellation must use a threading Event")
        self.coordinator = coordinator
        self.width = width
        self.cancel = cancel or Event()
        self.last_waves = ()

    def evaluate(self, executor, dag, source, computed=None):
        from .ast import StructuralEdit
        from .types import Effect, ValueKind
        from rexgraph.tensor_field import FieldSource
        if dag.binding.schema.kind is not ValueKind.REX:
            raise ValueError("parallel plans require a fixed native source snapshot")
        for node in dag.nodes:
            call = node.expression.call
            if isinstance(node.expression.expr, StructuralEdit) or (call is not None and
                (not node.reusable or not call.signature.memoizable or
                 call.signature.effects - {Effect.READ} or "train" in call.signature.requires)):
                raise ValueError("parallel execution requires only declared pure reusable actions")
        reference = FieldSource(dag.binding.value, dag.binding.ref.record_id,
                                dag.binding.ref.record_version, dag.binding.ref.state_digest)
        reference.check()
        computed = {} if computed is None else computed
        pending = [n for n in dag.nodes if n.id not in computed]
        records, waves = {}, []
        while pending:
            if self.cancel.is_set():
                raise QueryCancelledError("query cancelled before the next ready wave")
            reference.check()
            ready = [n for n in pending if all(i in computed for i in n.inputs)][:self.width]
            if not ready:
                raise ValueError("query plan has an unsatisfied dependency")
            units = []
            for node in ready:
                fragment = replace(dag, nodes=(node,))
                inputs = dict(computed)
                context = copy_context()
                def run(fragment=fragment, inputs=inputs, node=node, context=context):
                    try:
                        values, observations = context.run(
                            executor._execute_dag_serial, fragment, source, computed=inputs)
                        return True, values[node.id], observations
                    except Exception as exc:
                        return False, exc, ()
                units.append({"id": node.id, "type": "cpu_coordination", "fn": run})
            # These tasks capture live native state and must remain in this process.
            returned = self.coordinator.pools.run(units, {n.id: "thread" for n in ready}, cost=self.coordinator.cost)
            waves.append(tuple(n.id for n in ready))
            self.last_waves = tuple(waves)
            reference.check()
            if self.cancel.is_set():
                raise QueryCancelledError("query cancelled after a completed wave")
            for node in ready:
                if node.id not in returned:
                    raise RuntimeError("Coordinator did not return a query task result")
                success, value, observations = returned[node.id]
                if not success:
                    note = getattr(value, "add_note", None)
                    if callable(note):
                        note("RCQL plan node " + node.id + ": " + str(node.operator))
                    raise value
                computed[node.id] = value
                records[node.id] = observations
            selected = {n.id for n in ready}
            pending = [n for n in pending if n.id not in selected]
        observations = [entry for n in dag.nodes for entry in records.get(n.id, ())]
        return computed, observations
