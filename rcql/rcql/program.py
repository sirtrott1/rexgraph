"""Reusable finite RCQL programs over explicitly bound native sources."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace, field, is_dataclass
from collections.abc import Mapping
import re

from .ast import Query, Parameter, Literal, Call
from .program_codec import dumps, loads, digest, operators


def _name(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value) is None:
        raise ValueError("program names must be ASCII identifiers")
    return value


def _parameters(value):
    if isinstance(value, Parameter):
        yield value.name
    elif hasattr(value, "__dataclass_fields__"):
        for field in fields(value):
            yield from _parameters(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _parameters(child)


def _rename_parameter(value, old, new):
    return _map_parameters(value, {old: new})


def _map_parameters(value, names):
    if isinstance(value, Parameter):
        return Parameter(names.get(value.name, value.name))
    if isinstance(value, Literal):
        return value
    if is_dataclass(value):
        return replace(value, **{f.name: _map_parameters(getattr(value, f.name), names) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(_map_parameters(v, names) for v in value)
    return value


def _source_names(expr):
    if isinstance(expr, Parameter):
        return {expr.name}
    if isinstance(expr, Call):
        if expr.name in {"REX", "RCDB", "CATALOG"} and len(expr.args) == 1 and isinstance(expr.args[0], Literal):
            return {expr.args[0].value}
        if expr.name in {"AT", "AT_TIME", "RCDB_GET", "RCDB_VERSION", "RCDB_AS_OF",
                         "RCDB_VALID_AT", "VALID_AT", "TRANSACTION_AT"} and expr.args:
            return _source_names(expr.args[0])
        if expr.name == "FILE" and expr.args and isinstance(expr.args[0], Literal):
            return {expr.args[0].value}
        if expr.name == "PHRASE":
            return set()
    raise ValueError("program source must have an explicit resolvable source declaration")


@dataclass(frozen=True)
class ProgramInput:
    name: str
    kind: str | None = None
    grade: int | None = None
    variance: str | None = None

    def __post_init__(self):
        _name(self.name)
        from .types import ValueKind, Variance
        if self.kind is not None:
            ValueKind(self.kind)
        if self.variance is not None:
            Variance(self.variance)
        if self.grade is not None and (type(self.grade) is not int or self.grade < 0):
            raise ValueError("input grade must be a nonnegative integer")

    def validate(self, binding, value):
        from .types import RCType
        from .program_contracts import result_types
        typed = result_types(binding, (value,))[0]
        if any(v is not None for v in (self.kind, self.grade, self.variance)):
            if not isinstance(typed, RCType):
                raise TypeError("typed program input requires a declared native value")
            actual = (typed.kind.value, typed.grade, None if typed.variance is None else typed.variance.value)
            for expected, got in zip((self.kind, self.grade, self.variance), actual, strict=True):
                if expected is not None and expected != got:
                    raise TypeError("program input does not satisfy its declared type")


@dataclass(frozen=True)
class OutputRef:
    step: str
    output: int = 0

    def __post_init__(self):
        _name(self.step)
        if type(self.output) is not int or self.output < 0:
            raise ValueError("program output index must be nonnegative")


@dataclass(frozen=True)
class ProgramStep:
    name: str
    query: Query
    inputs: tuple[tuple[str, OutputRef], ...] = ()
    contracts: tuple[ProgramInput, ...] = ()

    def __post_init__(self):
        _name(self.name)
        if not isinstance(self.query, Query) or self.query.explain:
            raise TypeError("program steps require ordinary typed queries")
        dumps(self.query)
        pairs = tuple(self.inputs)
        if any(not isinstance(v, tuple) or len(v) != 2 or not isinstance(v[1], OutputRef) for v in pairs):
            raise TypeError("program links require parameter and output reference pairs")
        for key, _ in pairs:
            _name(key)
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("program step parameters must be distinct")
        object.__setattr__(self, "inputs", pairs)
        contracts = tuple(self.contracts)
        if (any(not isinstance(v, ProgramInput) for v in contracts)
                or len({v.name for v in contracts}) != len(contracts)
                or any(v.name not in set(_parameters(self.query)) for v in contracts)):
            raise ValueError("step contracts require distinct used parameters")
        object.__setattr__(self, "contracts", contracts)


    @property
    def output_count(self):
        return 1 if self.query.matches else len(self.query.returns)


@dataclass(frozen=True)
class ProgramResult:
    values: tuple
    steps: tuple
    program_digest: str
    sources: tuple
    aliases: tuple[str, ...]
    dependencies: tuple = ()

    @property
    def named_values(self):
        return dict(zip(self.aliases, self.values, strict=True))

    def step(self, name):
        return dict(self.steps)[name]


@dataclass(frozen=True)
class Program:
    """A finite query program with explicit input and output bindings."""
    name: str
    steps: tuple[ProgramStep, ...]
    inputs: tuple[ProgramInput, ...] = ()
    outputs: tuple[tuple[str, OutputRef], ...] = ()
    captures: tuple[tuple[ProgramInput, object], ...] = ()
    dependencies: tuple = ()
    _seal: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _name(self.name)
        steps, inputs = tuple(self.steps), tuple(self.inputs)
        if not steps or len(steps) > 10000 or any(not isinstance(s, ProgramStep) for s in steps):
            raise ValueError("program requires a finite nonempty list of steps")
        if any(not isinstance(v, ProgramInput) for v in inputs) or len({v.name for v in inputs}) != len(inputs):
            raise ValueError("program inputs require distinct declarations")
        captures = tuple(self.captures)
        if any(type(p) is not tuple or len(p) != 2 or not isinstance(p[0], ProgramInput) for p in captures):
            raise TypeError("program captures require input contracts and finite values")
        captured = {spec.name for spec, _ in captures}
        if len(captured) != len(captures) or captured & {v.name for v in inputs}:
            raise ValueError("program captures must be distinct from open inputs")
        captures = tuple((spec, loads(dumps(value))) for spec, value in captures)
        known, external = {}, {v.name for v in inputs} | captured
        for step in steps:
            if step.name in known:
                raise ValueError("program step names must be distinct")
            linked = {k for k, _ in step.inputs}
            if linked & external:
                raise ValueError("a step output may not shadow an external program input")
            for _, reference in step.inputs:
                if reference.step not in known or reference.output >= known[reference.step].output_count:
                    raise ValueError("program links must refer to preceding declared outputs")
            source_slots = _source_names(step.query.source)
            if captured & set(_parameters(step.query.source)):
                raise ValueError("program captures cannot change source selection")
            ordinary = set(_parameters(step.query)) - source_slots
            if ordinary != (ordinary & external) | linked:
                raise ValueError("program has an undeclared input or an unused link")
            if set(_parameters(step.query.source)) & linked:
                raise ValueError("source selection cannot depend on a computed program output")
            from .signatures import lookup
            from .types import Effect
            for name in operators(step.query):
                if name in {"PROGRAM_RUN", "PROGRAM_EXPLAIN", "PROGRAM_TOPOLOGY"}:
                    raise ValueError("a finite program composes explicit steps, not recursive program calls")
                # Source constructors are not traversed by operators().
                candidates = lookup(name)
                if not isinstance(candidates, tuple):
                    candidates = (candidates,)
                for signature in candidates:
                    if signature.effects - {Effect.READ} or "train" in signature.requires:
                        raise ValueError("reusable programs are read calculations; publish changes explicitly")
            known[step.name] = step
        outputs = tuple(self.outputs) or tuple((str(i), OutputRef(steps[-1].name, i))
                                                for i in range(steps[-1].output_count))
        if len({k for k, _ in outputs}) != len(outputs):
            raise ValueError("program output names must be distinct")
        for alias, ref in outputs:
            if not isinstance(alias, str) or not alias or not isinstance(ref, OutputRef):
                raise TypeError("program outputs require names and output references")
            if ref.step not in known or ref.output >= known[ref.step].output_count:
                raise ValueError("program export refers to an unknown result")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "captures", captures)
        from rexgraph.tensor_field import FieldSource
        dependencies = tuple(self.dependencies)
        if any(not isinstance(v, FieldSource) for v in dependencies):
            raise TypeError("program dependencies require native source references")
        dependencies = tuple({v.coefficient_digest: FieldSource(None, v.record_id, v.version,
                              v.state_digest) for v in dependencies}.values())
        object.__setattr__(self, "dependencies", dependencies)
        encoded = self.to_bytes()
        if len(encoded) > 4*1024*1024:
            raise ValueError("program declaration exceeds the size limit")
        object.__setattr__(self, "_seal", encoded)

    def rename_input(self, old, new):
        _ = self.coefficient_digest
        _name(new)
        reserved = {v.name for v in self.inputs} | {s.name for s, _ in self.captures}
        for step in self.steps:
            reserved.update(k for k, _ in step.inputs)
            reserved.update(_source_names(step.query.source))
            reserved.update(_parameters(step.query.source))
            if old in set(_parameters(step.query.source)):
                raise ValueError("input renaming cannot change source selection")
        if old not in {v.name for v in self.inputs} or new in reserved:
            raise ValueError("input renaming requires an open input and an unused name")
        return replace(self, inputs=tuple(replace(v, name=new) if v.name == old else v for v in self.inputs),
            steps=tuple(replace(s, query=_rename_parameter(s.query, old, new),
                contracts=tuple(replace(v, name=new) if v.name == old else v for v in s.contracts)) for s in self.steps))

    def rename_step(self, old, new):
        _ = self.coefficient_digest
        _name(new)
        names = {s.name for s in self.steps}
        if old not in names or new in names:
            raise ValueError("step renaming requires an existing and an unused name")
        def ref(r):
            return replace(r, step=new) if r.step == old else r
        return replace(self, steps=tuple(replace(s, name=new if s.name == old else s.name,
            inputs=tuple((k, ref(r)) for k, r in s.inputs)) for s in self.steps),
            outputs=tuple((k, ref(r)) for k, r in self.outputs))

    def specialize(self, bindings):
        _ = self.coefficient_digest
        if not isinstance(bindings, Mapping):
            raise TypeError("program specialization requires an explicit mapping")
        if not bindings or any(k not in {v.name for v in self.inputs} for k in bindings):
            raise ValueError("program specialization requires nonempty open input bindings")
        return replace(self, inputs=tuple(v for v in self.inputs if v.name not in bindings),
            captures=(*self.captures, *((v, bindings[v.name]) for v in self.inputs if v.name in bindings)))

    def declaration(self):
        data = {"schema": "rcql.program", "version": 1, "name": self.name,
                "inputs": tuple((v.name, v.kind, v.grade, v.variance) for v in self.inputs),
                "steps": tuple((s.name, s.query, tuple((k, r.step, r.output) for k, r in s.inputs)) for s in self.steps),
                "outputs": tuple((k, r.step, r.output) for k, r in self.outputs)}
        if self.captures:
            data.update(version=2, captures=tuple(((v.name, v.kind, v.grade, v.variance), value)
                                                 for v, value in self.captures))
        if self.dependencies or any(s.contracts for s in self.steps):
            data.update(version=3, captures=data.get("captures", ()),
                dependencies=tuple(v.as_record() for v in self.dependencies),
                step_contracts=tuple((s.name, tuple((v.name, v.kind, v.grade, v.variance)
                    for v in s.contracts)) for s in self.steps))
        return data

    @property
    def coefficient_digest(self):
        if self.to_bytes() != self._seal:
            raise ValueError("program declaration changed outside an explicit transformation")
        return digest(self.declaration())

    def to_bytes(self):
        return dumps(self.declaration())

    @classmethod
    def from_bytes(cls, raw):
        data = loads(raw)
        expected = {"schema", "version", "name", "inputs", "steps", "outputs"}
        if isinstance(data, dict) and data.get("version") in (2, 3):
            expected.add("captures")
            if data["version"] == 3:
                expected.update(("dependencies", "step_contracts"))
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("invalid program declaration fields")
        if data["schema"] != "rcql.program" or type(data["version"]) is not int or data["version"] not in (1, 2, 3):
            raise ValueError("unsupported program declaration version")
        if data["version"] == 2 and not data["captures"]:
            raise ValueError("program schema two requires explicit captures")
        contracts = dict(data.get("step_contracts", ()))
        if "step_contracts" in data and (len(contracts) != len(data["step_contracts"])
                or tuple(contracts) != tuple(s[0] for s in data["steps"])):
            raise ValueError("step contracts must follow the declared steps")
        from rexgraph.tensor_field import FieldSource
        result = cls(data["name"], tuple(ProgramStep(n, q, tuple((k, OutputRef(s, i)) for k, s, i in links),
                                      tuple(ProgramInput(*v) for v in contracts.get(n, ())))
                                      for n, q, links in data["steps"]),
                   tuple(ProgramInput(*v) for v in data["inputs"]),
                   tuple((k, OutputRef(s, i)) for k, s, i in data["outputs"]),
                   tuple((ProgramInput(*v), value) for v, value in data.get("captures", ())),
                   tuple(FieldSource(None, **v) for v in data.get("dependencies", ())))
        if result.to_bytes() != dumps(data):
            raise ValueError("program declaration differs from its canonical contract")
        return result

    def to_record(self):
        _ = self.coefficient_digest
        from rexgraph.graph import RexGraph
        result = RexGraph.from_cells([1, [[0]]])
        result.attach_metadata(1, 0, "rcql_program", self.to_bytes().decode("utf8"))
        if self.dependencies:
            result._agent_meta = {"rcql_evidence": [v.as_record() for v in self.dependencies]}
        return result

    @classmethod
    def from_record(cls, record):
        values = [record.get_metadata(1, 0, "rcql_program")]
        if len(values) != 1 or not isinstance(values[0], str):
            raise ValueError("record must contain one declared RCQL program")
        return cls.from_bytes(values[0])

    def execute(self, executor, *, explain=False):
        from .executor import Executor
        from .capabilities import BoundSource
        from .types import RCType
        from .native_plan import plain
        if not isinstance(executor, Executor):
            raise TypeError("a program requires the native Executor")
        declaration = self.coefficient_digest
        parameters = dict(executor.params)
        required = {v.name for v in self.inputs}
        if set(parameters) != required:
            raise ValueError("program parameters must match the declared inputs exactly")
        parameters.update((v.name, value) for v, value in self.captures)
        selected, result_by_name, explanations = {}, {}, []
        dependency_bindings = []
        if self.dependencies:
            from rexgraph.tensor_field import FieldSource
            available = []
            for name in executor.sources:
                expr = Parameter(name)
                value = executor._eval_source(expr)
                binding = executor._planning_binding(expr, value)
                if binding.ref.state_digest is not None:
                    available.append(binding)
            for expected in self.dependencies:
                found = next((b for b in available if expected.matches(FieldSource(
                    b.value, b.ref.record_id, b.ref.record_version, b.ref.state_digest))), None)
                if found is None:
                    raise ValueError("bind every exact program dependency before execution")
                found.source.require("read")
                if executor.evidence is not None:
                    executor.evidence.validate_binding(found)
                dependency_bindings.append(found)
        # Resolve each distinct source before evaluating the first program step.
        for step in self.steps:
            key = digest(step.query.source)
            if key not in selected:
                source = executor._eval_source(step.query.source)
                binding = executor._planning_binding(step.query.source, source)
                if executor.evidence is not None:
                    executor.evidence.validate_binding(binding)
                    executor.evidence.validate_values(executor.params)
                selected[key] = BoundSource(binding.value, binding.source.policy,
                                            ref=binding.ref, temporal=binding.temporal)
        for step in self.steps:
            supplied = dict(parameters)
            for key, ref in step.inputs:
                supplied[key] = result_by_name[ref.step].values[ref.output]
            source = selected[digest(step.query.source)]
            if dependency_bindings:
                from .capabilities import SourcePolicy
                if SourcePolicy.intersection(source.policy, *(b.source.policy for b in dependency_bindings)).digest != source.policy.digest:
                    raise PermissionError("bind program sources under the dependency intersection policy")
            child = Executor(sources={"program_source": source}, params=supplied,
                             artifacts=executor.artifacts, scheduler=executor.scheduler,
                             evidence=executor.evidence)
            query = replace(step.query, source=Parameter("program_source"), explain=explain)
            binding = child._planning_binding(query.source, source)
            for spec in self.inputs:
                if spec.name in set(_parameters(step.query)):
                    spec.validate(binding, supplied[spec.name])
            for spec, value in self.captures:
                spec.validate(binding, value)
            for spec in step.contracts:
                spec.validate(binding, supplied[spec.name])
            if explain:
                from .planning import plan_query
                plan = plan_query(binding, query, parameters=supplied)
                detail = plan.explain()
                detail["effects"] = sorted(effect.value for effect in plan.effects)
                explanations.append((step.name, detail))
                from .executor import Result
                if step.query.matches:
                    from .types import ValueKind, Exactness
                    result_by_name[step.name] = Result((RCType("QueryTable", kind=ValueKind.QUERY_TABLE,
                                                              exactness=Exactness.STRUCTURAL),))
                else:
                    result_by_name[step.name] = Result(tuple(e.result for e in plan.returns))
            else:
                try:
                    result_by_name[step.name] = child.execute(query)
                except Exception as exc:
                    note = getattr(exc, "add_note", None)
                    if callable(note):
                        note("RCQL program " + self.name + ": step " + step.name)
                    raise
        if self.coefficient_digest != declaration:
            raise ValueError("program declaration changed during evaluation")
        if explain:
            from .program_contracts import result_types
            return {"schema": "rcql.program.explanation", "program_digest": self.coefficient_digest,
                    "steps": tuple(explanations),
                    "output_types": result_types(
                        binding, tuple(result_by_name[r.step].values[r.output] for _, r in self.outputs)),
                    "evaluation": "none; dependent values remain typed declarations"}
        values = tuple(result_by_name[r.step].values[r.output] for _, r in self.outputs)
        refs = tuple(plain({"state_digest": s.ref.state_digest, "record_id": s.ref.record_id,
                           "record_version": s.ref.record_version, "policy_digest": s.policy.digest})
                     for s in selected.values())
        from rexgraph.tensor_field import FieldSource
        from .source_context import field_references
        dependencies = tuple(FieldSource(s.value, s.ref.record_id, s.ref.record_version, s.ref.state_digest)
                             for s in selected.values())
        dependencies += tuple(FieldSource(b.value, b.ref.record_id, b.ref.record_version,
                                          b.ref.state_digest) for b in dependency_bindings)
        dependencies += field_references(tuple(result.values for result in result_by_name.values()))
        dependencies = tuple({(ref.state_digest, ref.record_id, ref.version): ref for ref in dependencies}.values())
        return ProgramResult(values, tuple(result_by_name.items()), self.coefficient_digest,
                             refs, tuple(k for k, _ in self.outputs), dependencies)


    def topology(self, executor):
        from .plan_topology import PlanTopology
        from .source_context import field_references
        explanation = self.execute(executor, explain=True)
        nodes, sources, exports, stages = [], [], {}, []
        for step, (_, declared) in zip(self.steps, explanation["steps"], strict=True):
            if step.query.matches:
                raise ValueError("plan topology requires fixed operation instances without MATCH iteration")
            plan = declared["native_plan"]
            prefix = step.name + "/"
            source_key = "source/" + step.name
            sources.append(dict(plan["source"], id=source_key))
            links = dict(step.inputs)
            for old in plan["nodes"]:
                node = dict(old)
                node["id"] = prefix + old["id"]
                node["source"] = source_key
                node["inputs"] = [prefix + key for key in old["inputs"]]
                if old["kind"] == "parameter" and old.get("parameter") in links:
                    ref = links[old["parameter"]]
                    node["inputs"] = [exports[ref.step][ref.output]]
                    node["program_binding"] = ref.step
                nodes.append(node)
            exports[step.name] = tuple(prefix + key for key in plan["outputs"])
            stages.append({"name": step.name, "outputs": exports[step.name], "bindings": plan["bindings"]})
        outputs = tuple(exports[ref.step][ref.output] for _, ref in self.outputs)
        return PlanTopology({"schema": "rcql.program-plan", "version": 1,
                             "sources": sources, "nodes": nodes, "outputs": outputs,
                             "stages": stages, "program": self.to_bytes().decode("utf8"),
                             "program_digest": self.coefficient_digest,
                             "dependencies": tuple(ref.as_record() for ref in
                                                    (*self.dependencies, *field_references(executor.params)))})
