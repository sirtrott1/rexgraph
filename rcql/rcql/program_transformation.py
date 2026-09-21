"""Finite name transformations with exact interface and boundary certificates."""
from dataclasses import dataclass

from .program_codec import dumps, loads, digest, operators
from .name_relation import NameRelation, bound_arguments, substitute
from .ast import Parameter, Call, Literal


def _section_name(claim, side):
    return NameRelation("section_readout", ("family", "left", "right"),
        Call("SECTION_CERTIFIED_OBSERVE", (Parameter("family"), Parameter("left"),
             Parameter("right"), Literal(claim), Literal(side))))


def _contract(operation):
    return _query_contract(operation.query())


def _query_contract(query):
    from .signatures import lookup
    rows = []
    for name in operators(query):
        signatures = lookup(name)
        if not isinstance(signatures, tuple):
            signatures = (signatures,)
        rows.append((name, tuple((s.implementation_key, tuple(sorted(s.requires)),
                                 tuple(sorted(e.value for e in s.effects))) for s in signatures)))
    return tuple(rows)


def _program_certificate(source, target, rule, arguments):
    from .program import _rename_parameter, _parameters, _source_names
    expected = source.declaration()
    ports = tuple((v.name, v.name) for v in target.inputs)
    scope = "finite program under explicit interface renaming"
    if rule in ("input", "step"):
        if type(arguments) is not tuple or len(arguments) != 2 or any(type(v) is not str for v in arguments):
            raise ValueError("program renaming requires two explicit names")
        old, new = arguments
        if rule == "input":
            reserved = {v.name for v in source.inputs} | {v.name for v, _ in source.captures}
            for step in source.steps:
                reserved.update(k for k, _ in step.inputs)
                reserved.update(_source_names(step.query.source))
                reserved.update(_parameters(step.query.source))
            if old not in {v.name for v in source.inputs} or new in reserved:
                raise ValueError("program input rename captures an existing name")
            expected["inputs"] = tuple((new if n == old else n, k, g, v) for n, k, g, v in expected["inputs"])
            expected["steps"] = tuple((n, _rename_parameter(q, old, new), links) for n, q, links in expected["steps"])
            if "step_contracts" in expected:
                expected["step_contracts"] = tuple((s, tuple((new if n == old else n, k, g, v)
                    for n, k, g, v in contracts)) for s, contracts in expected["step_contracts"])
            ports = tuple((v.name, new if v.name == old else v.name) for v in source.inputs)
        else:
            names = {s.name for s in source.steps}
            if old not in names or new in names:
                raise ValueError("program step rename requires a distinct replacement")
            expected["steps"] = tuple((new if n == old else n, q,
                tuple((k, new if s == old else s, i) for k, s, i in links)) for n, q, links in expected["steps"])
            expected["outputs"] = tuple((k, new if s == old else s, i) for k, s, i in expected["outputs"])
            if "step_contracts" in expected:
                expected["step_contracts"] = tuple((new if s == old else s, contracts)
                    for s, contracts in expected["step_contracts"])
    elif rule == "specialize":
        if (type(arguments) is not tuple or not arguments or any(type(p) is not tuple or len(p) != 2
                or type(p[0]) is not str for p in arguments)):
            raise ValueError("program specialization requires named finite values")
        keys = tuple(k for k, _ in arguments)
        if keys != tuple(v.name for v in source.inputs if v.name in keys):
            raise ValueError("program specialization inputs must be distinct and ordered")
        specs = {v[0]: v for v in expected["inputs"]}
        expected.update(version=max(2, expected["version"]), inputs=tuple(v for v in expected["inputs"] if v[0] not in keys),
            captures=(*expected.get("captures", ()), *((specs[k], value) for k, value in arguments)))
        scope = "finite program under exactly the declared typed input captures"
    else:
        raise ValueError("unsupported finite program transformation")
    if dumps(expected) != target.to_bytes():
        raise ValueError("candidate program differs from the declared transformation")
    sources = tuple(digest(s.query.source) for s in source.steps)
    if sources != tuple(digest(s.query.source) for s in target.steps):
        raise ValueError("program transformation changed source selection")
    contracts = tuple(_query_contract(s.query) for s in source.steps)
    if contracts != tuple(_query_contract(s.query) for s in target.steps):
        raise ValueError("program transformation changed operation requirements")
    return {"rule_version": "rcql.finite-program-transform.v1", "scope": scope,
        "source_digest": source.coefficient_digest, "target_digest": target.coefficient_digest,
        "ports": ports, "source_selectors": sources,
        "steps": tuple((a.name, b.name) for a, b in zip(source.steps, target.steps, strict=True)),
        "source_links": tuple(row[2] for row in source.declaration()["steps"]),
        "target_links": tuple(row[2] for row in target.declaration()["steps"]),
        "exports": tuple(k for k, _ in target.outputs),
        "captures": tuple((v.name, v.kind, v.grade, v.variance, digest(value)) for v, value in target.captures),
        "operator_contracts": contracts,
        "topology_status": "bound static plan required; no unbound operation tower asserted",
        "excluded_claims": ("dynamic execution topology", "trace label equality", "arbitrary program equivalence")}


def _check_interface(source, target, rule, arguments):
    """Check substitution independently of the candidate construction method."""
    if rule == "section_readout":
        from .readout_equivalence import ReadoutEquivalence
        if type(arguments) is not tuple or len(arguments) != 1:
            raise ValueError("section transformation requires one explicit readout claim")
        ReadoutEquivalence.claim(arguments[0])
        if (source.to_bytes() != _section_name(arguments[0], "left").to_bytes() or
                target.to_bytes() != _section_name(arguments[0], "right").to_bytes()):
            raise ValueError("section transformation must retain both guarded readout declarations")
        return tuple((key, key) for key in source.inputs)
    if type(rule) is not str or type(arguments) is not tuple or any(type(v) is not str for v in arguments):
        raise TypeError("transformation rules require a name and a finite tuple of names")
    expected = source.declaration()
    if rule == "port":
        if len(arguments) != 2:
            raise ValueError("port transformation requires old and new names")
        old, new = arguments
        if old not in source.inputs or new in source.inputs:
            raise ValueError("port transformation requires an existing and an unused name")
        replacements = {old: Parameter(new)}
        expected.update(inputs=tuple(new if p == old else p for p in source.inputs),
                        body=substitute(source.body, replacements),
                        stages=tuple((k, substitute(v, replacements)) for k, v in source.stages),
                        defaults=tuple((new if k == old else k, v) for k, v in source.defaults),
                        derivation=(*source.derivation, ("port", old, new)))
    elif rule == "alias":
        if len(arguments) != 1:
            raise ValueError("alias transformation requires one name")
        expected.update(name=arguments[0], derivation=(*source.derivation, ("name", source.name, arguments[0])))
    else:
        raise ValueError("unsupported program transformation rule")
    if dumps(expected) != target.to_bytes():
        raise ValueError("candidate does not satisfy the declared interface substitution")
    return tuple(zip(source.inputs, target.inputs, strict=True))


def _topology_payload(topology):
    from .relation_topology import _complex_payload
    return {"declaration": topology.declaration, "tower": _complex_payload(topology.boundary_tower())}


def _restore_topology(payload, operation):
    from .relation_topology import RelationTopology, _complex_from_payload
    if not isinstance(payload, dict) or set(payload) != {"declaration", "tower"}:
        raise ValueError("invalid transformation topology")
    topology = RelationTopology(payload["declaration"], _tower=_complex_from_payload(payload["tower"]))
    expected = operation.topology()
    if (topology.coefficient_digest != expected.coefficient_digest or
            topology.boundary_tower().coefficient_digest != expected.boundary_tower().coefficient_digest):
        raise ValueError("transformation topology differs from the operation declaration")
    return topology


def _check_specialization(source, target, arguments):
    if (type(arguments) is not tuple or not arguments or any(type(p) is not tuple or len(p) != 2
            or type(p[0]) is not str for p in arguments)):
        raise ValueError("specialization requires explicit named values")
    keys = tuple(k for k, _ in arguments)
    if keys != tuple(k for k in source.inputs if k in keys):
        raise ValueError("specialization ports must be distinct open inputs in declaration order")
    expected = source.declaration()
    expected.update(inputs=tuple(k for k in source.inputs if k not in keys),
        defaults=tuple((k, v) for k, v in source.defaults if k not in keys),
        bound_values=(*source.bound_values, *arguments),
        derivation=(*source.derivation, *(("bind", k, v) for k, v in arguments)))
    if dumps(expected) != target.to_bytes():
        raise ValueError("candidate does not retain the declared specialization")
    return tuple((k, k) for k in target.inputs)


def _check_composition(source, target, arguments):
    """Check each retained stage without expanding or executing the expression DAG."""
    if type(arguments) is not tuple or len(arguments) != 2 or type(arguments[0]) is not bytes:
        raise ValueError("composition requires a following declaration and explicit ports")
    following, ports = NameRelation.from_bytes(arguments[0]), arguments[1]
    if (type(ports) is not tuple or not ports or any(type(k) is not str for k in ports)
            or len(set(ports)) != len(ports) or any(k not in following.inputs for k in ports)):
        raise ValueError("composition requires distinct following input ports")
    remaining = tuple(k for k in following.inputs if k not in ports)
    if set(remaining) & set(source.inputs) or target.inputs != (*source.inputs, *remaining):
        raise ValueError("composition open ports disagree")
    expected_defaults = (*source.defaults, *((k, v) for k, v in following.defaults if k in remaining))
    if dumps(target.defaults) != dumps(expected_defaults) or target.name != following.name:
        raise ValueError("composition name or defaults disagree")
    if len(target.stages) != len(source.stages) + 1 + len(following.stages):
        raise ValueError("composition must retain each stage and one shared intermediate")
    if len(target.bound_values) != len(source.bound_values) + len(following.bound_values):
        raise ValueError("composition must retain both capture sets")
    index, stage_index, renamed, stage_map = 0, 0, {}, []
    intermediate = None
    for prefix, operation in (("left", source), ("right", following)):
        names = {}
        for key, value in operation.bound_values:
            new, captured = target.bound_values[index]
            index += 1
            if dumps(captured) != dumps(value):
                raise ValueError("composition changed a captured value")
            names[key] = Parameter(new)
        renamed[prefix] = tuple((k, v.name) for k, v in names.items())
        if prefix == "right":
            names.update({k: Parameter(intermediate) for k in ports})
        for key, expression in operation.stages:
            new, retained = target.stages[stage_index]
            stage_index += 1
            if dumps(retained) != dumps(substitute(expression, names)):
                raise ValueError("composition changed a retained stage")
            names[key] = Parameter(new)
            stage_map.append((prefix, key, new))
        body = substitute(operation.body, names)
        if prefix == "left":
            intermediate, retained = target.stages[stage_index]
            stage_index += 1
            if dumps(retained) != dumps(body):
                raise ValueError("composition changed the shared input operation")
        elif dumps(target.body) != dumps(body):
            raise ValueError("composition changed the following operation")
    expected_derivation = (*source.derivation, *following.derivation,
        ("compose", source.coefficient_digest, following.coefficient_digest, ports, renamed))
    if dumps(target.derivation) != dumps(expected_derivation):
        raise ValueError("composition derivation disagrees with its retained occurrences")
    return following, ports, renamed, tuple(stage_map), intermediate


def _construction_certificate(source, target, rule, arguments, before, after):
    from rexgraph.chain_map import GradedMap
    following = None
    if rule == "specialize":
        ports = _check_specialization(source, target, arguments)
        detail = {"fixed_inputs": tuple((k, digest(v)) for k, v in arguments)}
        scope = "original operation with exactly the declared fixed inputs"
    else:
        following, connected, captures, stages, intermediate = _check_composition(source, target, arguments)
        ports = tuple((k, k) for k in source.inputs)
        detail = {"following_digest": following.coefficient_digest, "connected_ports": connected,
                  "capture_mapping": captures, "stage_mapping": stages, "shared_intermediate": intermediate}
        scope = "declared value chaining with one shared intermediate result"
    topologies = (before, after) if following is None else (before, after, following.topology())
    residuals = []
    for topology in topologies:
        tower = topology.boundary_tower()
        identities = tuple(tuple((i, i, 1) for i in range(n)) for n in tower.sizes)
        closed = GradedMap(tower, tower, identities).verify()
        residuals.append((closed.source_residual, closed.target_residual, *closed.commutation_residuals))
    expected_contract = dict(_contract(source))
    if following is not None:
        expected_contract.update(_contract(following))
    if tuple(sorted(expected_contract.items())) != _contract(target):
        raise ValueError("construction changed the declared operator requirements")
    return {"rule_version": "rcql.name-construction.v1", "scope": scope,
        "source_digest": source.coefficient_digest, "target_digest": target.coefficient_digest,
        "ports": ports, **detail,
        "source_operations": tuple(c.identity for c in before.relations(1)),
        "target_operations": tuple(c.identity for c in after.relations(1)),
        "tower_digests": tuple(t.boundary_tower().coefficient_digest for t in topologies),
        "tower_closure_residuals": tuple(residuals),
        "target_captures": tuple((k, digest(v)) for k, v in target.bound_values),
        "operator_contracts": _contract(target),
        "dependencies": () if following is None else (following.coefficient_digest,),
        "source_requirements": "same selected source and explicit arguments at compilation and application",
        "excluded_claims": ("unconditional source equivalence", "cross tower chain map", "metric equivalence")}


def _certificate(source, target, rule, arguments, before, after):
    if rule in ("specialize", "compose"):
        return _construction_certificate(source, target, rule, arguments, before, after)
    from rexgraph.chain_map import GradedMap
    ports = _check_interface(source, target, rule, arguments)
    a, b = before.boundary_tower(), after.boundary_tower()
    if a.sizes != b.sizes:
        raise ValueError("interface renaming changed operation occurrences")
    # Traversal order is retained by the checked substitution, not inferred from shape.
    components = tuple(tuple((i, i, 1) for i in range(n)) for n in a.sizes)
    chain = GradedMap(a, b, components).verify()
    left, right = before.port_realization(), after.port_realization()
    if left.entries != right.entries or left.domain != right.domain or left.codomain != right.codomain:
        raise ValueError("interface transformation changed shared value realization")
    if _contract(source) != _contract(target):
        raise ValueError("interface transformation changed operator requirements")
    proof = {"rule_version": "rcql.name-interface.v1", "scope": "declaration under explicit interface renaming",
            "source_digest": source.coefficient_digest, "target_digest": target.coefficient_digest,
            "ports": ports, "captures": tuple((k, k, digest(v)) for k, v in source.bound_values),
            "operation_mapping": tuple(zip(a.spaces[1].keys, b.spaces[1].keys, strict=True)),
            "grade_mapping": tuple(tuple(zip(x.keys, y.keys, strict=True)) for x, y in zip(a.spaces, b.spaces, strict=True)),
            "components": components, "chain_map_digest": chain.declaration.coefficient_digest,
            "chain_residuals": (chain.source_residual, chain.target_residual, *chain.commutation_residuals),
            "operator_contracts": _contract(source),
            "source_requirements": "same selected source, arguments and evidence at compilation and application",
            "dependencies": (), "dependency_scope": "finite declaration; native inputs are bound at execution",
            "changed_labels": (source.name, target.name),
            "excluded_claims": ("trace label equality", "arbitrary program equivalence", "metric equivalence")}
    if rule == "section_readout":
        from .readout_equivalence import ReadoutEquivalence
        claim = ReadoutEquivalence.claim(arguments[0])
        proof.update(rule_version="rcql.section-readout.v1",
            scope="guarded linear readout values on the declared affine family",
            readout_claim=claim, readout_status="requires exact live inputs at compilation and application",
            dependencies=(claim["family"], claim["left"], claim["right"]),
            dependency_scope="exact family and both readout declarations")
    return proof


@dataclass(frozen=True)
class ProgramTransformation:
    """A name or program transformation with structural and execution checks."""
    data: bytes

    def __post_init__(self):
        if type(self.data) is not bytes:
            raise TypeError("transformation storage requires immutable bytes")
        self.verify()

    @classmethod
    def name(cls, operation, rule, arguments):
        if not isinstance(operation, NameRelation):
            raise TypeError("name transformation requires a NameRelation")
        _ = operation.coefficient_digest
        if type(arguments) not in (tuple, list) or any(type(v) is not str for v in arguments):
            raise TypeError("transformation arguments require an explicit sequence of names")
        arguments = tuple(arguments)
        if rule == "port" and len(arguments) == 2:
            target = operation.rename(*arguments)
        elif rule == "alias" and len(arguments) == 1:
            target = operation.named(*arguments)
        else:
            raise ValueError("unsupported transformation rule or argument count")
        return cls._create(operation, target, rule, arguments)

    @classmethod
    def section(cls, family, left, right):
        """Propose a guarded replacement over the entire declared section family."""
        from .readout_equivalence import ReadoutEquivalence
        certificate = ReadoutEquivalence.check(family, left, right)
        if not certificate.equivalent:
            raise ValueError("readouts differ on the retained section family")
        claim = certificate.to_bytes()
        return cls._create(_section_name(claim, "left"), _section_name(claim, "right"),
                           "section_readout", (claim,))

    @classmethod
    def specialize(cls, operation, bindings):
        from collections.abc import Mapping
        if not isinstance(operation, NameRelation) or not isinstance(bindings, Mapping):
            raise TypeError("specialization requires a name and an explicit binding mapping")
        if not bindings or any(type(k) is not str or k not in operation.inputs for k in bindings):
            raise ValueError("specialization requires nonempty open input bindings")
        arguments = tuple((k, bindings[k]) for k in operation.inputs if k in bindings)
        target = operation
        for key, value in arguments:
            target = target.bind(key, value)
        return cls._create(operation, target, "specialize", arguments)

    @classmethod
    def compose(cls, operation, following, ports):
        if not isinstance(operation, NameRelation) or not isinstance(following, NameRelation):
            raise TypeError("composition requires two explicit name declarations")
        if not isinstance(ports, (str, tuple, list)):
            raise TypeError("composition requires explicit following input ports")
        ports = (ports,) if isinstance(ports, str) else tuple(ports)
        target = operation.then(following, ports)
        return cls._create(operation, target, "compose", (following.to_bytes(), ports))

    @classmethod
    def _create(cls, operation, target, rule, arguments):
        before, after = operation.topology(), target.topology()
        proof = _certificate(operation, target, rule, arguments, before, after)
        return cls(dumps({"schema": "rcql.program-transformation", "version": 1,
                          "source": operation.to_bytes(), "target": target.to_bytes(),
                          "rule": rule, "arguments": arguments, "certificate": proof,
                          "before": _topology_payload(before), "after": _topology_payload(after)}))

    @classmethod
    def program(cls, program, rule, arguments):
        from .program import Program
        from collections.abc import Mapping
        if not isinstance(program, Program):
            raise TypeError("finite program transformation requires a Program")
        _ = program.coefficient_digest
        if rule == "specialize":
            if not isinstance(arguments, Mapping):
                raise TypeError("program specialization requires a binding mapping")
            target = program.specialize(arguments)
            arguments = tuple((v.name, arguments[v.name]) for v in program.inputs if v.name in arguments)
        elif rule in ("input", "step") and type(arguments) in (tuple, list) and len(arguments) == 2:
            target = (program.rename_input if rule == "input" else program.rename_step)(*arguments)
            arguments = tuple(arguments)
        else:
            raise ValueError("unsupported finite program transformation or arguments")
        proof = _program_certificate(program, target, rule, arguments)
        return cls(dumps({"schema": "rcql.program-transformation", "version": 2,
            "source": program.to_bytes(), "target": target.to_bytes(), "rule": rule,
            "arguments": arguments, "certificate": proof}))

    @property
    def is_program(self):
        return loads(self.data)["version"] == 2

    def original(self):
        self.verify()
        from .program import Program
        return (Program if self.is_program else NameRelation).from_bytes(loads(self.data)["source"])

    @property
    def coefficient_digest(self):
        return digest(("rcql.program-transformation.v1", self.data))

    def to_bytes(self):
        return self.data

    @classmethod
    def from_bytes(cls, raw):
        if isinstance(raw, str):
            raw = raw.encode("utf8")
        return cls(raw)

    def verify(self, operation=None):
        data = loads(self.data)
        if isinstance(data, dict) and type(data.get("version")) is int and data["version"] == 2:
            from .program import Program
            if set(data) != {"schema", "version", "source", "target", "rule", "arguments", "certificate"} or data["schema"] != "rcql.program-transformation":
                raise ValueError("unsupported finite program transformation schema")
            if type(data["source"]) is not bytes or type(data["target"]) is not bytes:
                raise TypeError("program endpoints require finite declaration bytes")
            source, target = Program.from_bytes(data["source"]), Program.from_bytes(data["target"])
            if operation is not None and (not isinstance(operation, Program) or operation.coefficient_digest != source.coefficient_digest):
                raise ValueError("transformation source program mismatch")
            proof = _program_certificate(source, target, data["rule"], data["arguments"])
            if dumps(proof) != dumps(data["certificate"]):
                raise ValueError("program transformation certificate mismatch")
            return proof
        fields = {"schema", "version", "source", "target", "rule", "arguments", "certificate", "before", "after"}
        if (not isinstance(data, dict) or set(data) != fields or data["schema"] != "rcql.program-transformation"
                or type(data["version"]) is not int or data["version"] != 1):
            raise ValueError("unsupported program transformation schema")
        if type(data["source"]) is not bytes or type(data["target"]) is not bytes:
            raise TypeError("transformation endpoints require finite declaration bytes")
        source, target = NameRelation.from_bytes(data["source"]), NameRelation.from_bytes(data["target"])
        if operation is not None:
            if not isinstance(operation, NameRelation) or operation.coefficient_digest != source.coefficient_digest:
                raise ValueError("transformation source declaration mismatch")
        before = _restore_topology(data["before"], source)
        after = _restore_topology(data["after"], target)
        proof = _certificate(source, target, data["rule"], data["arguments"], before, after)
        if dumps(proof) != dumps(data["certificate"]):
            raise ValueError("transformation certificate does not match its declarations")
        return proof

    def topology(self):
        self.verify()
        if self.is_program:
            raise ValueError("use TRANSFORM_PROGRAM_TOPOLOGY with explicit sources and parameters")
        data = loads(self.data)
        return _restore_topology(data["after"], NameRelation.from_bytes(data["target"]))

    def compile(self, binding, operation, arguments):
        """Validate current bindings and return the candidate without evaluating it."""
        from .execution_trace import current_evidence
        proof = self.verify(operation)
        if self.is_program:
            raise ValueError("use TRANSFORM_PROGRAM_COMPILE with explicit sources and parameters")
        data = loads(self.data)
        candidate = NameRelation.from_bytes(data["target"])
        supplied = bound_arguments(candidate.inputs, candidate.defaults, arguments)
        original = {old: supplied[new] for old, new in proof["ports"]}
        if data["rule"] == "specialize":
            original.update(data["arguments"])
        binding.source.require("read")
        evidence = current_evidence()
        if evidence is not None:
            evidence.validate_binding(binding)
            evidence.validate_values((arguments, operation))
        if data["rule"] == "section_readout":
            from .transformation_operators import validate_readout
            from .types import RCType
            if not any(isinstance(v, RCType) for v in supplied.values()):
                validate_readout(binding, supplied["family"], supplied["left"], supplied["right"],
                                 data["arguments"][0], "right")
        before, after = operation.explain(binding, original), candidate.explain(binding, supplied)
        effects = before.effects
        if data["rule"] == "compose":
            following = NameRelation.from_bytes(data["arguments"][0])
            connected = data["arguments"][1]
            inputs = {k: before.returns[0].result if k in connected else supplied[k] for k in following.inputs}
            before = following.explain(binding, inputs)
            effects = effects | before.effects
        if before.returns[0].result != after.returns[0].result or effects != after.effects:
            raise ValueError("transformed operation has a different bound result or effect contract")
        return candidate

    def compile_program(self, program, sources, parameters):
        from .program import Program
        from .program_operators import _program_executor
        from collections.abc import Mapping
        proof = self.verify(program)
        if not self.is_program or not isinstance(parameters, Mapping):
            raise TypeError("program compilation requires a program transformation and explicit parameters")
        data = loads(self.data)
        candidate = Program.from_bytes(data["target"])
        if set(parameters) != {v.name for v in candidate.inputs}:
            raise ValueError("candidate parameters must match the declared open inputs")
        original = {old: parameters[new] for old, new in proof["ports"]}
        if data["rule"] == "specialize":
            original.update(data["arguments"])
        before = program.execute(_program_executor(sources, original), explain=True)
        after = candidate.execute(_program_executor(sources, parameters), explain=True)
        if (before["output_types"] != after["output_types"] or
                tuple(p["effects"] for _, p in before["steps"]) != tuple(p["effects"] for _, p in after["steps"])):
            raise ValueError("program transformation changed the bound output or effect contract")
        return candidate

    def to_record(self):
        """Retain the candidate operation carrier and the full transformation evidence."""
        if self.is_program:
            from .program import Program
            self.verify()
            record = Program.from_bytes(loads(self.data)["target"]).to_record()
        else:
            record = self.topology().to_record()
        record.attach_metadata(1, 0, "rcql_program_transformation", self.data.decode("utf8"))
        return record

    @classmethod
    def from_record(cls, record):
        from rexgraph.io.catalog import object_digest
        raw = record.get_metadata(1, 0, "rcql_program_transformation")
        if not isinstance(raw, str):
            raise ValueError("record has no program transformation")
        result = cls.from_bytes(raw)
        if object_digest(result.to_record()) != object_digest(record):
            raise ValueError("stored transformation carrier differs from its declaration")
        return result
