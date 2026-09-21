"""Scoped operation declarations with explicit argument and modifier relations."""
from __future__ import annotations

from dataclasses import dataclass, replace, fields, field as data_field
from collections.abc import Mapping

from .ast import Expr, Call, Parameter, Literal, ListExpr, Comparison, Member, Alias, Query
from .program_codec import dumps, loads, digest
from .program import _name


_ALLOWED = (Call, Parameter, Literal, ListExpr, Comparison, Member, Alias)
_CONTROL = {"PROGRAM_RUN", "PROGRAM_EXPLAIN", "PROGRAM_TOPOLOGY", "RECURSIVE_RUN",
            "RECURSIVE_EXPLAIN", "NAME_ITERATE", "RECURSE"}


def walk(value):
    if isinstance(value, Expr):
        yield value
        for field in fields(value):
            yield from walk(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from walk(item)


def substitute(value, replacements):
    if isinstance(value, Parameter) and value.name in replacements:
        return replacements[value.name]
    if isinstance(value, Expr):
        return replace(value, **{f.name: substitute(getattr(value, f.name), replacements) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(substitute(v, replacements) for v in value)
    return value


def validate_expression(expression, *, recursive=False):
    from .signatures import lookup
    from .arguments import EXPRESSION_ARGUMENTS
    from .types import Effect
    if not isinstance(expression, Expr):
        raise TypeError("an operation body requires an RCQL expression")
    dumps(expression)
    for node in walk(expression):
        if not isinstance(node, _ALLOWED):
            raise TypeError("operation bodies use declared RCQL expressions")
        if not isinstance(node, Call):
            continue
        if node.name == "RECURSE" and recursive:
            if len(node.args) not in (2, 3) or not isinstance(node.args[0], Literal) or not isinstance(node.args[0].value, str):
                raise TypeError("a recursive reference requires a literal definition name and explicit arguments")
            continue
        if node.name in _CONTROL:
            raise ValueError("use an explicit recursive definition rather than a nested program control call")
        labels, defaults = EXPRESSION_ARGUMENTS[node.name]
        if not len(labels)-len(defaults) <= len(node.args) <= len(labels):
            raise TypeError(f"{node.name} has incompatible argument arity")
        signatures = lookup(node.name)
        if not isinstance(signatures, tuple):
            signatures = (signatures,)
        for signature in signatures:
            if signature.effects - {Effect.READ} or "train" in signature.requires or not signature.memoizable:
                raise ValueError("named operation bodies require declared pure read operations")


def parameters(expression):
    return tuple(dict.fromkeys(n.name for n in walk(expression) if isinstance(n, Parameter)))


def bound_arguments(inputs, defaults, arguments):
    inputs = tuple(inputs)
    if isinstance(arguments, Mapping):
        supplied = dict(arguments)
        if set(supplied) - set(inputs):
            raise TypeError("unknown or previously captured operation argument")
    elif isinstance(arguments, (tuple, list)):
        if len(arguments) > len(inputs):
            raise TypeError("too many operation arguments")
        supplied = dict(zip(inputs, arguments, strict=False))
    else:
        raise TypeError("arguments must be a positional sequence or a named mapping")
    values = dict(defaults)
    values.update(supplied)
    missing = set(inputs)-set(values)
    if missing:
        raise TypeError("missing operation arguments: " + ", ".join(sorted(missing)))
    return {key: values[key] for key in inputs}


@dataclass(frozen=True)
class NameRelation:
    """One immutable operation name with retained argument occurrences."""
    name: str
    inputs: tuple[str, ...]
    body: Expr
    defaults: tuple[tuple[str, object], ...] = ()
    derivation: tuple = ()
    bound_values: tuple[tuple[str, object], ...] = ()
    stages: tuple[tuple[str, Expr], ...] = ()
    _seal: bytes = data_field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _name(self.name)
        inputs = tuple(self.inputs)
        for value in inputs:
            _name(value)
        stages = tuple(self.stages)
        stage_names = tuple(k for k, _ in stages)
        external = set(inputs) | {k for k, _ in self.bound_values}
        if (len(set(inputs)) != len(inputs) or len(set(stage_names)) != len(stage_names)
                or external & set(stage_names)):
            raise ValueError("operation inputs and intermediate ports must be distinct")
        available = set(external)
        used = set(parameters(self.body))
        for key, expression in stages:
            _name(key)
            validate_expression(expression)
            if set(parameters(expression)) - available:
                raise ValueError("an intermediate relation requires preceding input ports")
            used.update(parameters(expression))
            available.add(key)
        if set(parameters(self.body)) - available or (used - set(stage_names)) != external:
            raise ValueError("operation inputs must identify exactly the free argument names")
        object.__setattr__(self, "stages", stages)
        captured = tuple(self.bound_values)
        if len({k for k, _ in captured}) != len(captured) or set(inputs) & {k for k, _ in captured}:
            raise ValueError("captured ports must be distinct from open input ports")
        for key, value in captured:
            _name(key)
            dumps(value)
        object.__setattr__(self, "bound_values", captured)
        defaults = tuple(self.defaults)
        if len({key for key, _ in defaults}) != len(defaults) or any(key not in inputs for key, _ in defaults):
            raise ValueError("defaults require distinct open argument names")
        validate_expression(self.body)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "defaults", defaults)
        object.__setattr__(self, "derivation", tuple(self.derivation))
        encoded = self.to_bytes()
        if len(encoded) > 4*1024*1024:
            raise ValueError("operation declaration exceeds the finite size limit")
        object.__setattr__(self, "_seal", encoded)

    @classmethod
    def operator(cls, operator, *, name=None):
        from .arguments import EXPRESSION_ARGUMENTS
        from .names import canonical_name
        operator = canonical_name(operator)
        labels, defaults = EXPRESSION_ARGUMENTS[operator]
        return cls(name or operator, labels, Call(operator, tuple(Parameter(k) for k in labels)),
                   tuple(zip(labels[len(labels)-len(defaults):], defaults, strict=False)) if defaults else (),
                   (("operator", operator),))

    @property
    def coefficient_digest(self):
        if self.to_bytes() != self._seal:
            raise ValueError("named declaration changed outside an explicit modifier")
        return digest(self.declaration())

    def declaration(self):
        return {"schema": "rcql.name-relation", "version": 1, "name": self.name,
                "inputs": self.inputs, "body": self.body, "defaults": self.defaults,
                "derivation": self.derivation, "bound_values": self.bound_values, "stages": self.stages}

    def to_bytes(self):
        return dumps(self.declaration())

    @classmethod
    def from_bytes(cls, raw):
        data = loads(raw)
        if not isinstance(data, dict) or set(data) != {"schema", "version", "name", "inputs", "body", "defaults", "derivation", "bound_values", "stages"}:
            raise ValueError("invalid named operation declaration")
        if data["schema"] != "rcql.name-relation" or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported named operation schema")
        return cls(data["name"], data["inputs"], data["body"], data["defaults"], data["derivation"], data["bound_values"], data["stages"])

    def named(self, name):
        return replace(self, name=name, derivation=(*self.derivation, ("name", self.name, name)))

    def bind(self, argument, value):
        if argument not in self.inputs:
            raise ValueError("binding requires an open argument; modify a captured value explicitly")
        dumps(value)
        return NameRelation(self.name, tuple(k for k in self.inputs if k != argument), self.body,
            tuple((k, v) for k, v in self.defaults if k != argument),
            (*self.derivation, ("bind", argument, value)), (*self.bound_values, (argument, value)), self.stages)

    def rebind(self, argument, value):
        if argument not in dict(self.bound_values):
            raise ValueError("replacement requires a captured argument")
        dumps(value)
        return replace(self, bound_values=tuple((k, value if k == argument else v) for k, v in self.bound_values),
                       derivation=(*self.derivation, ("replace", argument, value)))

    def rename(self, old, new):
        if old not in self.inputs or new in self.inputs:
            raise ValueError("port renaming requires one existing and one unused name")
        _name(new)
        return NameRelation(self.name, tuple(new if k == old else k for k in self.inputs),
            substitute(self.body, {old: Parameter(new)}),
            tuple((new if k == old else k, v) for k, v in self.defaults),
            (*self.derivation, ("port", old, new)), self.bound_values,
            tuple((k, substitute(e, {old: Parameter(new)})) for k, e in self.stages))

    def then(self, following, argument, *, name=None, modifier=False):
        if not isinstance(following, NameRelation):
            raise TypeError("a following name must declare its operation")
        ports = (argument,) if isinstance(argument, str) else tuple(argument)
        if not ports or len(set(ports)) != len(ports) or any(k not in following.inputs for k in ports):
            raise ValueError("composition requires distinct explicit destination ports")
        remaining = tuple(k for k in following.inputs if k not in ports)
        if set(remaining) & set(self.inputs):
            raise ValueError("unconnected argument names collide; rename those ports explicitly")
        open_inputs = (*self.inputs, *remaining)
        captured, renamed, stages = [], {}, []
        used = set(open_inputs)
        def fresh(stem):
            key = stem
            while key in used:
                key += "_"
            used.add(key)
            return key
        bodies = []
        output_key = None
        for prefix, relation in (("left", self), ("right", following)):
            names = {}
            for key, value in relation.bound_values:
                new = fresh(prefix+"__"+key)
                names[key] = Parameter(new)
                captured.append((new, value))
            renamed[prefix] = tuple((k, v.name) for k, v in names.items())
            if prefix == "right":
                names.update({k: Parameter(output_key) for k in ports})
            for key, expression in relation.stages:
                new = fresh("relation_stage_"+str(len(stages)))
                stages.append((new, substitute(expression, names)))
                names[key] = Parameter(new)
            body = substitute(relation.body, names)
            if prefix == "left":
                output_key = fresh("relation_stage_"+str(len(stages)))
                stages.append((output_key, body))
            else:
                bodies.append(body)
        return NameRelation(name or (self.name if modifier else following.name), open_inputs, bodies[0],
            (*self.defaults, *((k, v) for k, v in following.defaults if k in remaining)),
            (*self.derivation, *following.derivation,
             ("modify" if modifier else "compose", self.coefficient_digest, following.coefficient_digest, ports, renamed)),
            tuple(captured), tuple(stages))

    def query(self):
        """Compile retained intermediate relations into existing local bindings."""
        from .ast import LetBinding, Reference
        names = {k: Reference(k) for k, _ in self.stages}
        bindings = tuple(LetBinding(k, substitute(expression, names)) for k, expression in self.stages)
        return Query(Parameter("name_source"), (substitute(self.body, names),), bindings=bindings)

    def call(self, *args, **keywords):
        """Build a call using declared ports without changing the global registry."""
        from .builder import expr
        if len(args) > len(self.inputs):
            raise TypeError("too many positional arguments")
        values = {k: expr(v) for k, v in zip(self.inputs, args, strict=False)}
        for key, value in keywords.items():
            if key in values:
                raise TypeError("an operation argument was supplied twice")
            values[key] = expr(value)
        bound = bound_arguments(self.inputs, tuple((k, Literal(v)) for k, v in self.defaults), values)
        return Call("NAME_APPLY", (Literal(self), ListExpr(tuple(bound[k] for k in self.inputs))))

    def explain(self, binding, arguments):
        from .planning import plan_query
        _ = self.coefficient_digest
        params = {**dict(self.bound_values), **bound_arguments(self.inputs, self.defaults, arguments)}
        query = self.query()
        plan = plan_query(binding, query, parameters=params)
        return plan

    def apply(self, binding, arguments):
        from .relation_runtime import runtime_scope
        from .executor import Executor
        from .capabilities import BoundSource
        from .execution_trace import current_artifact_services, record_method
        params = {**dict(self.bound_values), **bound_arguments(self.inputs, self.defaults, arguments)}
        self.explain(binding, arguments)
        declaration = self.coefficient_digest
        with runtime_scope(binding) as runtime:
            token = runtime.enter_name(self, params)
            try:
                child = Executor(sources={"name_source": BoundSource(binding.value, binding.source.policy,
                    ref=binding.ref, temporal=binding.temporal)}, params=params,
                    artifacts=current_artifact_services(required=False), evidence=runtime.evidence)
                result = child.execute(self.query())
                runtime.check(params, result.values)
                if self.coefficient_digest != declaration:
                    raise ValueError("named operation changed during its invocation")
                record_method("typed-named-operation", declaration=self.coefficient_digest,
                              name=self.name, execution=result.execution)
                return result.values[0]
            finally:
                runtime.leave_name(token)

    def to_record(self):
        _ = self.coefficient_digest
        from rexgraph.graph import RexGraph
        record = RexGraph.from_cells([1, [[0]]])
        record.attach_metadata(1, 0, "rcql_name_relation", self.to_bytes().decode("utf8"))
        return record

    @classmethod
    def from_record(cls, record):
        raw = record.get_metadata(1, 0, "rcql_name_relation")
        if not isinstance(raw, str):
            raise ValueError("record has no named operation declaration")
        return cls.from_bytes(raw)

    def topology(self):
        from .relation_topology import RelationTopology
        return RelationTopology.from_name(self)
