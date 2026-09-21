"""Recursive relation definitions evaluated through the native RCQL planner."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace, field as data_field

from .ast import Expr, Literal, Parameter, Call, ListExpr, Query
from .program import ProgramInput, _name
from .program_codec import dumps, loads, digest
from .name_relation import validate_expression, parameters, bound_arguments, walk
from .relation_runtime import RecursiveCycleError, runtime_scope, fingerprint


def recur(name, *arguments, **named):
    from .builder import expr
    _name(name)
    positional = ListExpr(tuple(expr(v) for v in arguments))
    if not named:
        return Call("RECURSE", (Literal(name), positional))
    for key in named:
        _name(key)
    keywords = ListExpr(tuple(ListExpr((Literal(k), expr(v))) for k, v in named.items()))
    return Call("RECURSE", (Literal(name), positional, keywords))


def recursive_arguments(node, target):
    positional = node.args[1]
    if not isinstance(positional, ListExpr):
        raise TypeError("recursive arguments require explicit occurrence expressions")
    names = tuple(p.name for p in target.inputs)
    if len(positional.items) > len(names):
        raise TypeError("too many recursive arguments")
    result = dict(zip(names, positional.items, strict=False))
    if len(node.args) == 3:
        keywords = node.args[2]
        if not isinstance(keywords, ListExpr):
            raise TypeError("recursive named arguments require explicit pairs")
        for pair in keywords.items:
            if (not isinstance(pair, ListExpr) or len(pair.items) != 2
                    or not isinstance(pair.items[0], Literal) or not isinstance(pair.items[0].value, str)):
                raise TypeError("recursive named argument has no declared port")
            key = pair.items[0].value
            if key not in names or key in result:
                raise TypeError("unknown or duplicate recursive argument: " + key)
            result[key] = pair.items[1]
    if set(result) != set(names):
        raise TypeError("recursive reference must bind every argument port")
    return tuple(result[name] for name in names)


@dataclass(frozen=True)
class RecursiveDefinition:
    """One relation with an explicit guard and two independently typed branches."""
    name: str
    inputs: tuple[ProgramInput, ...]
    guard: Expr
    base: Expr
    step: Expr
    result: ProgramInput
    result_like: str | None = None
    decreases: str | None = None

    def __post_init__(self):
        _name(self.name)
        inputs = tuple(self.inputs)
        if any(not isinstance(v, ProgramInput) for v in inputs) or len({v.name for v in inputs}) != len(inputs):
            raise ValueError("recursive inputs require distinct declared ports")
        if not isinstance(self.result, ProgramInput) or self.result.kind is None:
            raise TypeError("recursive results require an explicit value kind")
        names = {v.name for v in inputs}
        if self.result_like is not None and self.result_like not in names:
            raise ValueError("the result template must name an input")
        if self.decreases is not None and self.decreases not in names:
            raise ValueError("the decreasing coordinate must name an input")
        for expr in (self.guard, self.base, self.step):
            validate_expression(expr, recursive=True)
            if set(parameters(expr)) - names:
                raise ValueError("recursive body contains an undeclared parameter")
        if any(isinstance(v, Call) and v.name == "RECURSE" for v in walk(self.guard)):
            raise ValueError("the base guard must be decidable without a recursive call")
        object.__setattr__(self, "inputs", inputs)

    def declaration(self):
        return {"name": self.name,
                "inputs": tuple((i.name, i.kind, i.grade, i.variance) for i in self.inputs),
                "guard": self.guard, "base": self.base, "step": self.step,
                "result": (self.result.name, self.result.kind, self.result.grade, self.result.variance),
                "result_like": self.result_like, "decreases": self.decreases}

    @classmethod
    def from_declaration(cls, data):
        if not isinstance(data, dict) or set(data) != {"name", "inputs", "guard", "base", "step", "result", "result_like", "decreases"}:
            raise ValueError("invalid recursive definition fields")
        return cls(data["name"], tuple(ProgramInput(*v) for v in data["inputs"]), data["guard"],
                   data["base"], data["step"], ProgramInput(*data["result"]), data["result_like"], data["decreases"])

    def arguments(self, arguments):
        return bound_arguments(tuple(v.name for v in self.inputs), (), arguments)

    def result_type(self, binding, arguments):
        from .types import RCType, ValueKind, Variance
        from .program_contracts import result_types
        if self.result_like is not None:
            return result_types(binding, (arguments[self.result_like],))[0]
        return RCType(self.result.kind, kind=ValueKind(self.result.kind), grade=self.result.grade,
                      variance=None if self.result.variance is None else Variance(self.result.variance))


def _same_contract(expected, actual):
    from .types import RCType, ValueKind
    if not isinstance(expected, RCType) or not isinstance(actual, RCType):
        raise TypeError("a recursive result has no native type declaration")
    for attr in ("kind", "grade", "variance", "coordinates", "tensor_axes", "basis", "source", "domain", "exactness"):
        value = getattr(expected, attr)
        if value is not None and value is not ValueKind.UNKNOWN and value != getattr(actual, attr):
            raise TypeError("recursive result differs at its declared " + attr)


@dataclass(frozen=True)
class RecursionResult:
    """A completed value with retained invocation history and source dependencies."""
    value: object
    definition_digest: str
    entry: str
    calls: int
    evaluations: int
    invocations: tuple
    history: tuple
    dependencies: tuple = ()
    arguments: tuple = ()

    def __post_init__(self):
        from rexgraph.tensor_field import FieldSource
        _name(self.entry)
        if (not isinstance(self.definition_digest, str) or len(self.definition_digest) != 64
                or any(c not in "0123456789abcdef" for c in self.definition_digest)):
            raise ValueError("a recursion result requires its complete definition identity")
        if any(type(n) is not int or n < 0 for n in (self.calls, self.evaluations)):
            raise ValueError("completed operation counts must be nonnegative integers")
        events, history, refs = tuple(self.invocations), tuple(self.history), tuple(self.dependencies)
        seen = set()
        for row in events:
            if (not isinstance(row, dict) or set(row) != {"id", "name", "parent", "branch", "arguments"}
                    or type(row["id"]) is not int or row["id"] < 0 or row["id"] in seen
                    or (row["parent"] is not None and row["parent"] not in seen)
                    or row["branch"] not in {"base", "step", "reuse", "initial", "iterate"}):
                raise ValueError("invalid recursive invocation declaration")
            _name(row["name"])
            seen.add(row["id"])
        if len({v[0] for v in history}) != len(history) or any(i not in seen for i, _, _ in history):
            raise ValueError("retained results require distinct actual invocations")
        if (any(not isinstance(r, FieldSource) for r in refs)
                or len({r.coefficient_digest for r in refs}) != len(refs)):
            raise ValueError("recursive dependencies require distinct source references")
        object.__setattr__(self, "invocations", events)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "dependencies", refs)
        arguments = tuple(self.arguments)
        if (len({row[0] for row in arguments}) != len(arguments)
                or any(i not in seen or len(dict(values)) != len(values) for i, _, values in arguments)):
            raise ValueError("invocation arguments require distinct actual ports")
        object.__setattr__(self, "arguments", arguments)

    @property
    def coefficient_digest(self):
        return digest(("rcql.recursion-result", self.definition_digest, self.entry,
                       fingerprint(self.value), self.calls, self.evaluations, self.invocations,
                       tuple((i, n, fingerprint(v)) for i, n, v in self.history),
                       tuple(r.coefficient_digest for r in self.dependencies), fingerprint(self.arguments)))

    @property
    def completed(self):
        return True

    def topology(self):
        from .relation_topology import RelationTopology
        return RelationTopology.from_invocations(self)

    def fields(self, definition=None, order="completion", port=None):
        from rexgraph.tensor_field import TensorField, TensorChannels
        if order not in {"completion", "invocation"}:
            raise ValueError("history order is completion or invocation, not an inferred time axis")
        if port is None:
            rows = tuple(row for row in self.history if definition is None or row[1] == definition)
        else:
            rows = tuple((i, name+"/"+port, dict(values)[port]) for i, name, values in self.arguments
                         if (definition is None or name == definition) and port in dict(values))
            if order == "completion":
                completed = {i: index for index, (i, _, _) in enumerate(self.history)}
                rows = tuple(sorted(rows, key=lambda row: completed.get(row[0], len(completed)+row[0])))
        if order == "invocation":
            rows = tuple(sorted(rows, key=lambda row: row[0]))
        if not rows or any(not isinstance(value, TensorField) for _, _, value in rows):
            raise TypeError("selected history must contain retained tensor fields")
        return TensorChannels(tuple(name+"/"+str(i) for i, name, _ in rows),
                              tuple(value for _, _, value in rows), self.coefficient_digest, self.dependencies)

    def to_record(self):
        from .recursive_state import result_record
        return result_record(self)

    @classmethod
    def from_record(cls, record, references=()):
        from .recursive_state import restore_result
        return restore_result(record, references)


@dataclass(frozen=True)
class RecursiveProgram:
    """A finite set of mutually referential operation definitions."""
    name: str
    definitions: tuple[RecursiveDefinition, ...]
    _seal: bytes = data_field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _name(self.name)
        values = tuple(self.definitions)
        if not values or len(values) > 1000 or any(not isinstance(v, RecursiveDefinition) for v in values):
            raise ValueError("a recursive program requires a finite definition family")
        known = {v.name: v for v in values}
        if len(known) != len(values):
            raise ValueError("recursive definition names must be distinct")
        for definition in values:
            for expression in (definition.base, definition.step):
                for node in walk(expression):
                    if isinstance(node, Call) and node.name == "RECURSE":
                        target = node.args[0].value
                        if target not in known:
                            raise ValueError("recursive reference has no declared target: " + target)
                        recursive_arguments(node, known[target])
        object.__setattr__(self, "definitions", values)
        encoded = self.to_bytes()
        if len(encoded) > 4*1024*1024:
            raise ValueError("recursive definition group exceeds the finite size limit")
        object.__setattr__(self, "_seal", encoded)

    def declaration(self):
        return {"schema": "rcql.recursive-relations", "version": 1, "name": self.name,
                "definitions": tuple(v.declaration() for v in self.definitions)}

    @property
    def coefficient_digest(self):
        if self.to_bytes() != self._seal:
            raise ValueError("recursive definition changed outside a new declaration")
        return digest(self.declaration())

    def to_bytes(self):
        return dumps(self.declaration())

    @classmethod
    def from_bytes(cls, raw):
        data = loads(raw)
        if not isinstance(data, dict) or set(data) != {"schema", "version", "name", "definitions"}:
            raise ValueError("invalid recursive program declaration")
        if data["schema"] != "rcql.recursive-relations" or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported recursive program schema")
        return cls(data["name"], tuple(RecursiveDefinition.from_declaration(v) for v in data["definitions"]))

    def to_record(self):
        _ = self.coefficient_digest
        from rexgraph.graph import RexGraph
        record = RexGraph.from_cells([1, [[0]]])
        record.attach_metadata(1, 0, "rcql_recursive_program", self.to_bytes().decode("utf8"))
        return record

    @classmethod
    def from_record(cls, record):
        raw = record.get_metadata(1, 0, "rcql_recursive_program")
        if not isinstance(raw, str):
            raise ValueError("record has no recursive program declaration")
        return cls.from_bytes(raw)

    def definition(self, name):
        for value in self.definitions:
            if value.name == name:
                return value
        raise ValueError("unknown recursive definition: " + str(name))

    def _plan_expression(self, binding, expression, supplied):
        from .planning import plan_query
        slots = dict(supplied)
        count = 0
        def visit(expr):
            nonlocal count
            if isinstance(expr, Call) and expr.name == "RECURSE":
                target = self.definition(expr.args[0].value)
                args = []
                for argument in recursive_arguments(expr, target):
                    rewritten = visit(argument)
                    plan = plan_query(binding, Query(Parameter("recursive_source"), (rewritten,)), parameters=slots)
                    if isinstance(rewritten, Parameter) and rewritten.name in slots:
                        args.append(slots[rewritten.name])
                    elif isinstance(rewritten, Literal):
                        args.append(rewritten.value)
                    else:
                        args.append(plan.returns[0].result)
                bound = target.arguments(args)
                for port in target.inputs:
                    port.validate(binding, bound[port.name])
                key = "__recursive_result_" + str(count)
                count += 1
                while key in slots:
                    key += "_"
                slots[key] = target.result_type(binding, bound)
                return Parameter(key)
            if isinstance(expr, Expr):
                return replace(expr, **{f.name: visit_value(getattr(expr, f.name)) for f in fields(expr)})
            return expr
        def visit_value(value):
            if isinstance(value, Expr):
                return visit(value)
            if isinstance(value, tuple):
                return tuple(visit_value(v) for v in value)
            return value
        rewritten = visit(expression)
        return plan_query(binding, Query(Parameter("recursive_source"), (rewritten,)), parameters=slots)

    def explain(self, binding, entry, arguments):
        from .types import RCType, ValueKind, Variance, BOOLEAN
        from .program_contracts import result_types
        from .relation_runtime import runtime_scope
        definition = self.definition(entry)
        supplied = definition.arguments(arguments)
        with runtime_scope(binding) as runtime:
            runtime.check(supplied)
            for port in definition.inputs:
                port.validate(binding, supplied[port.name])
            plans = []
            for item in self.definitions:
                local = supplied if item.name == entry else {p.name: RCType(p.kind or "Unknown",
                    kind=ValueKind(p.kind) if p.kind else ValueKind.UNKNOWN, grade=p.grade,
                    variance=None if p.variance is None else Variance(p.variance)) for p in item.inputs}
                guard = self._plan_expression(binding, item.guard, local)
                if guard.returns[0].result != BOOLEAN and getattr(guard.returns[0].result, "kind", None) is not ValueKind.BOOLEAN:
                    if not isinstance(item.guard, Literal) or type(item.guard.value) is not bool:
                        raise TypeError("a recursive guard must return one boolean")
                base = self._plan_expression(binding, item.base, local)
                step = self._plan_expression(binding, item.step, local)
                expected = item.result_type(binding, local)
                for result in (base.returns[0].result, step.returns[0].result):
                    _same_contract(expected, result_types(binding, (result,))[0])
                plans.append({"name": item.name, "guard": guard.explain(), "base": base.explain(),
                              "step": step.explain(), "decreases": item.decreases})
            return {"schema": "rcql.recursive-explanation", "definition_digest": self.coefficient_digest,
                    "entry": entry, "definitions": tuple(plans),
                    "result_type": definition.result_type(binding, supplied), "evaluation": "none"}

    def execute(self, binding, entry, arguments, *, limits=None, cancellation=None, history=True, memoize=True):
        from .executor import Executor
        from .capabilities import BoundSource
        from .execution_trace import current_artifact_services, record_method
        from .program_contracts import result_types
        from .source_context import field_references
        from rexgraph.tensor_field import FieldSource
        if type(history) is not bool or type(memoize) is not bool:
            raise TypeError("history and memoization selections must be boolean")
        definition = self.definition(entry)
        initial = definition.arguments(arguments)
        definition_identity = self.coefficient_digest
        self.explain(binding, entry, initial)
        with runtime_scope(binding, limits, cancellation) as runtime:
            source = BoundSource(binding.value, binding.source.policy, ref=binding.ref, temporal=binding.temporal)
            memo, active, invocations, retained, retained_inputs = {}, set(), [], [], []
            serial = 0
            initial_identity = fingerprint(initial)
            def ordinary(expression, supplied):
                runtime.tick(supplied)
                executor = Executor(sources={"recursive_source": source}, params=supplied,
                    artifacts=current_artifact_services(required=False), evidence=runtime.evidence)
                result = executor.execute(Query(Parameter("recursive_source"), (expression,)))
                runtime.check(supplied, result.values)
                record_method("recursive-expression", execution=result.execution)
                return result.values[0]
            def evaluate(expression, supplied):
                if not any(isinstance(n, Call) and n.name == "RECURSE" for n in walk(expression)):
                    return ordinary(expression, supplied)
                if isinstance(expression, Call) and expression.name == "RECURSE":
                    values = []
                    target = self.definition(expression.args[0].value)
                    for item in recursive_arguments(expression, target):
                        values.append((yield from evaluate(item, supplied)))
                    return (yield (expression.args[0].value, tuple(values)))
                replacements = {}
                def children(value):
                    if isinstance(value, Expr):
                        result = yield from evaluate(value, supplied)
                        key = "__recursive_value_" + str(len(replacements))
                        while key in supplied or key in replacements:
                            key += "_"
                        replacements[key] = result
                        return Parameter(key)
                    if isinstance(value, tuple):
                        result = []
                        for item in value:
                            result.append((yield from children(item)))
                        return tuple(result)
                    return value
                changes = {}
                for field in fields(expression):
                    changes[field.name] = yield from children(getattr(expression, field.name))
                return ordinary(replace(expression, **changes), {**supplied, **replacements})
            def invocation(item, supplied, number, parent):
                if history:
                    retained_inputs.append((number, item.name, tuple(supplied.items())))
                for port in item.inputs:
                    port.validate(binding, supplied[port.name])
                if item.decreases is not None:
                    value = supplied[item.decreases]
                    if type(value) is not int or value < 0:
                        raise ValueError("a decreasing coordinate must be a nonnegative integer")
                guard = ordinary(item.guard, supplied)
                if type(guard) is not bool:
                    raise TypeError("recursive guard did not return a boolean")
                invocations.append({"id": number, "name": item.name, "parent": parent,
                                    "branch": "base" if guard else "step", "arguments": digest(fingerprint(supplied))})
                value = yield from evaluate(item.base if guard else item.step, supplied)
                _same_contract(item.result_type(binding, supplied), result_types(binding, (value,))[0])
                runtime.check(value)
                if history:
                    retained.append((number, item.name, value))
                return value
            stack = []
            def push(name, values, parent):
                nonlocal serial
                item = self.definition(name)
                supplied = item.arguments(values)
                runtime.call(len(stack)+1)
                key = digest((self.coefficient_digest, name, fingerprint(supplied)))
                if key in active:
                    raise RecursiveCycleError("an active call repeats without progress; declare affine feedback to solve a fixed point")
                if stack and item.decreases is not None:
                    prior = stack[-1]
                    old_item = prior["definition"]
                    if old_item.decreases is not None:
                        before = prior["arguments"][old_item.decreases]
                        after = supplied[item.decreases]
                        if type(after) is not int or not 0 <= after < before:
                            raise ValueError("recursive call does not decrease its declared coordinate")
                if memoize and key in memo:
                    if history:
                        retained_inputs.append((serial, item.name, tuple(supplied.items())))
                    invocations.append({"id": serial, "name": name, "parent": parent, "branch": "reuse", "arguments": key})
                    serial += 1
                    return False, memo[key]
                number = serial
                serial += 1
                active.add(key)
                stack.append({"generator": invocation(item, supplied, number, parent), "key": key,
                              "id": number, "definition": item, "arguments": supplied,
                              "identity": fingerprint(supplied), "started": False, "send": None})
                return True, None
            push(entry, initial, None)
            completed = None
            while stack:
                frame = stack[-1]
                try:
                    runtime.check(frame["arguments"])
                    request = (next(frame["generator"]) if not frame["started"]
                               else frame["generator"].send(frame["send"]))
                    frame["started"] = True
                    frame["send"] = None
                    entered, value = push(request[0], request[1], frame["id"])
                    if not entered:
                        frame["send"] = value
                except StopIteration as stop:
                    if fingerprint(frame["arguments"]) != frame["identity"]:
                        raise ValueError("recursive arguments changed during evaluation") from None
                    completed = stop.value
                    active.remove(frame["key"])
                    if memoize:
                        memo[frame["key"]] = completed
                    stack.pop()
                    if stack:
                        stack[-1]["send"] = completed
                except Exception as exc:
                    if callable(getattr(exc, "add_note", None)):
                        exc.add_note("RCQL recursive relation " + self.name + ": " + frame["definition"].name)
                    raise
            if fingerprint(initial) != initial_identity or self.coefficient_digest != definition_identity:
                raise ValueError("recursive input or definition changed during evaluation")
            refs = field_references((initial, completed, tuple(v for _, _, v in retained)))
            native = binding.value if hasattr(binding.value, "_boundary_ptr") else binding.value.rex
            own = FieldSource(native, binding.ref.record_id, binding.ref.record_version)
            dependencies = tuple({v.coefficient_digest: v for v in (own, *refs)}.values())
            return RecursionResult(completed, self.coefficient_digest, entry, runtime.calls,
                                   runtime.evaluations, tuple(invocations), tuple(retained), dependencies, tuple(retained_inputs))

    def topology(self):
        from .relation_topology import RelationTopology
        return RelationTopology.from_recursive(self)
