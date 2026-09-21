"""Finite program fragments assembled through exact section observations."""
from dataclasses import dataclass, replace, field

from .program import Program, ProgramStep, OutputRef, _name, _parameters, _map_parameters, _source_names
from .program_codec import dumps, loads, digest


@dataclass(frozen=True)
class ProgramAssembly:
    """Fixed local programs with explicit ports, links and coefficient stalks."""
    data: bytes
    dependencies: tuple = field(init=False)

    def __post_init__(self):
        if type(self.data) is not bytes:
            raise TypeError("program assembly requires immutable declaration bytes")
        object.__setattr__(self, "dependencies", self.template().dependencies)

    @classmethod
    def create(cls, name, fragments, coefficients, inputs=(), links=(), outputs=()):
        fragments = tuple(fragments)
        for _, program in fragments:
            if not isinstance(program, Program):
                raise TypeError("program stalks require finite Program declarations")
            _ = program.coefficient_digest
        return cls(dumps(dict(schema="rcql.program-assembly", version=1, name=name,
            fragments=tuple((cell, program.to_bytes()) for cell, program in fragments),
            coefficients=tuple(tuple(v) for v in coefficients),
            inputs=tuple(tuple(v) for v in inputs), links=tuple(tuple(v) for v in links),
            outputs=tuple(tuple(v) for v in outputs))))

    @property
    def coefficient_digest(self):
        return digest(("rcql.program-assembly.v1", self.data))

    def declaration(self):
        value = loads(self.data)
        if (not isinstance(value, dict) or set(value) != {"schema", "version", "name", "fragments",
                "coefficients", "inputs", "links", "outputs"}
                or value["schema"] != "rcql.program-assembly" or type(value["version"]) is not int
                or value["version"] != 1):
            raise ValueError("unsupported program assembly schema")
        return value

    def template(self):
        data = self.declaration()
        _name(data["name"])
        fragments = tuple((cell, Program.from_bytes(raw)) for cell, raw in data["fragments"])
        if not fragments or len({c for c, _ in fragments}) != len(fragments):
            raise ValueError("assembly requires distinct local program stalks")
        for cell, _ in fragments:
            if not isinstance(cell, str) or not cell:
                raise ValueError("program stalks require explicit cell identities")
        specs = {(c, v.name): v for c, p in fragments for v in p.inputs}
        assigned, external, captured, steps, available = {}, {}, [], [], {}
        for kind, rows in (("coefficient", data["coefficients"]), ("input", data["inputs"]), ("link", data["links"])):
            for row in rows:
                if type(row) is not tuple or len(row) != (4 if kind == "link" else 3):
                    raise ValueError("assembly port declarations have the wrong arity")
                cell, port, *target = row
                key = (cell, port)
                if key not in specs or key in assigned:
                    raise ValueError("each assembly input needs one explicit binding")
                if any(type(v) is not str or not v for v in row):
                    raise TypeError("assembly ports require named coordinates")
                if kind == "input":
                    _name(target[0])
                if kind == "coefficient" and specs[key].kind not in (None, "ExactInteger", "ExactRational"):
                    raise TypeError("section coefficients require exact scalar program inputs")
                assigned[key] = (kind, tuple(target))
        if set(assigned) != set(specs):
            raise ValueError("every local input needs an external, section or linked binding")
        reserved = set()
        for _, program in fragments:
            for step in program.steps:
                reserved.update(_source_names(step.query.source))
                if set(_parameters(step.query.source)) & {v.name for v in program.inputs}:
                    raise ValueError("assembly cannot remap source selection parameters")
        generated = set()
        for index, (cell, program) in enumerate(fragments):
            names = {v.name: f"p{index}_{v.name}" for v in program.inputs}
            names.update((v.name, f"p{index}_{v.name}") for v, _ in program.captures)
            linked = {k for step in program.steps for k, _ in step.inputs}
            names.update((k, f"p{index}_{k}") for k in linked)
            for spec in program.inputs:
                kind, target = assigned[cell, spec.name]
                if kind == "link" and target not in available:
                    raise ValueError("assembly links require a preceding fragment export")
                if kind == "input":
                    names[spec.name] = target[0]
                new = names[spec.name]
                if new in reserved:
                    raise ValueError("assembly input collides with a source name")
                contract = replace(spec, name=new)
                if kind in ("input", "coefficient"):
                    if new in external and external[new] != contract:
                        raise ValueError("shared assembly inputs have different type contracts")
                    external[new] = contract
            local_generated = {new for old, new in names.items()
                if (cell, old) not in assigned or assigned[cell, old][0] != "input"}
            if local_generated & (generated | reserved) or local_generated & set(external) - {
                    names[v.name] for v in program.inputs if assigned[cell, v.name][0] == "coefficient"}:
                raise ValueError("assembly generated names collide with declared inputs")
            generated.update(local_generated)
            def reference(ref):
                return OutputRef(f"f{index}_{ref.step}", ref.output)
            for spec, value in program.captures:
                captured.append((replace(spec, name=names[spec.name]), value))
            for step in program.steps:
                query = _map_parameters(step.query, names)
                bindings = [(names[k], reference(ref)) for k, ref in step.inputs]
                for spec in program.inputs:
                    kind, target = assigned[cell, spec.name]
                    if kind == "link" and spec.name in set(_parameters(step.query)):
                        if target not in available:
                            raise ValueError("assembly links require a preceding fragment export")
                        bindings.append((names[spec.name], available[target]))
                contracts = {names[v.name]: replace(v, name=names[v.name])
                    for v in program.inputs if v.name in set(_parameters(step.query))}
                for v in step.contracts:
                    contract = replace(v, name=names.get(v.name, v.name))
                    if contract.name in contracts and contracts[contract.name] != contract:
                        raise ValueError("local program contracts disagree")
                    contracts[contract.name] = contract
                steps.append(ProgramStep(f"f{index}_{step.name}", query, tuple(bindings), tuple(contracts.values())))
            available.update(((cell, alias), reference(ref)) for alias, ref in program.outputs)
        if any((cell, export) not in available for _, _, cell, export in data["links"]):
            raise ValueError("assembly links require declared fragment exports")
        if generated & {row[2] for row in data["inputs"]}:
            raise ValueError("external input collides with a local generated name")
        outputs = data["outputs"] or tuple((alias, fragments[-1][0], alias) for alias, _ in fragments[-1][1].outputs)
        exports = []
        for alias, cell, export in outputs:
            if (cell, export) not in available:
                raise ValueError("assembly export does not name a local output")
            exports.append((alias, available[cell, export]))
        return Program(data["name"], tuple(steps), tuple(external.values()), tuple(exports), tuple(captured),
                       tuple(ref for _, p in fragments for ref in p.dependencies))

    def coefficient_map(self, system):
        from rexgraph.coordinate_map import CoordinateMap
        from rexgraph.type_accession import CoordinateSpace
        data = self.declaration()
        indices = {cell: i for i, (cell, _) in enumerate(data["fragments"])}
        if any(cell not in system.recipe.cells for cell in indices):
            raise ValueError("every program fragment requires its declared section stalk")
        keys, entries = [], []
        for cell, port, coordinate in data["coefficients"]:
            action = system.selection(cell)
            local = action.codomain.keys.index(coordinate)
            keys.append(f"p{indices[cell]}_{port}")
            entries.extend((len(keys)-1, j, v) for i, j, v in action.entries if i == local)
        return CoordinateMap(system.space, CoordinateSpace("program_coefficients/"+self.coefficient_digest,
                                                         tuple(keys)), tuple(entries))

    def complete(self, system, observation=None, observed=None):
        self.coefficient_map(system)
        family = system.complete(observation, observed)
        object.__setattr__(family, "_query_contributors", getattr(system, "_query_contributors", ()))
        return ProgramFamily(self, system.recipe, family)

    def to_record(self):
        record = self.template().to_record()
        record.attach_metadata(1, 0, "rcql_program_assembly", self.data.decode())
        return record

    @classmethod
    def from_record(cls, record):
        from rexgraph.io.catalog import object_digest
        raw = record.get_metadata(1, 0, "rcql_program_assembly")
        if not isinstance(raw, str):
            raise ValueError("record has no program assembly")
        result = cls(raw.encode())
        if object_digest(result.to_record()) != object_digest(record):
            raise ValueError("program assembly carrier differs from its declaration")
        return result


@dataclass(frozen=True)
class ProgramFamily:
    """Fixed executable structure over an exact compatible coefficient family."""
    assembly: ProgramAssembly
    recipe: object
    family: object

    def __post_init__(self):
        from rexgraph.section_calculus import SectionSystem, SectionFamily
        if not isinstance(self.assembly, ProgramAssembly) or not isinstance(self.family, SectionFamily):
            raise TypeError("program families require an assembly and exact sections")
        system = SectionSystem(self.recipe)
        self.assembly.coefficient_map(system)
        if self.family.declaration_digest != system.coefficient_digest or self.family.particular.space != system.space:
            raise ValueError("program family belongs to another section declaration")
        if self.family.particular.axes:
            raise ValueError("program coefficients require an explicit scalar section observation")
        if not self.family.source.matches(system.source):
            raise ValueError("program family and recipe source identities disagree")
        if any(not any(a.matches(b) for b in self.family.dependencies) for a in system.recipe.dependencies):
            raise ValueError("program family omitted a section contributor")
        if (any(system.action.apply(self.family.particular.values))
                or self.family.directions.compose(system.action).entries):
            raise ValueError("program family does not satisfy the declared restrictions")

    @property
    def coefficient_digest(self):
        return digest(("rcql.program-family.v1", self.assembly.coefficient_digest,
                       self.recipe.coefficient_digest, self.family.coefficient_digest))

    def observation(self):
        from rexgraph.section_calculus import SectionSystem
        return self.family.observe(self.assembly.coefficient_map(SectionSystem(self.recipe)))

    def compile(self, sources, parameters):
        from .program_operators import _program_executor
        image = self.observation()
        values = image.value()
        template = self.assembly.template()
        specs = {v.name: v for v in template.inputs}
        captures = {}
        for key, value in zip(image.action.codomain.keys, values.values, strict=True):
            if specs[key].kind == "ExactInteger":
                if value.denominator != 1:
                    raise TypeError("section coefficient violates its integer program contract")
                value = value.numerator
            captures[key] = value
        candidate = template.specialize(captures) if captures else template
        candidate = replace(candidate, dependencies=(*candidate.dependencies, self.family.source,
                                                      *self.family.dependencies))
        candidate.execute(_program_executor(sources, parameters), explain=True)
        return candidate

    def to_record(self):
        self.family.check_state()
        record = self.assembly.to_record()
        record.attach_metadata(1, 0, "rcql_program_recipe", self.recipe)
        record.attach_metadata(1, 0, "rcql_program_sections", self.family)
        record.attach_metadata(1, 0, "rcql_program_family", self.coefficient_digest)
        return record

    @classmethod
    def from_record(cls, record):
        raw = record.get_metadata(1, 0, "rcql_program_assembly")
        if not isinstance(raw, str):
            raise ValueError("record has no program family")
        result = cls(ProgramAssembly(raw.encode()), record.get_metadata(1, 0, "rcql_program_recipe"),
                     record.get_metadata(1, 0, "rcql_program_sections"))
        if result.coefficient_digest != record.get_metadata(1, 0, "rcql_program_family"):
            raise ValueError("program family certificate mismatch")
        from rexgraph.io.catalog import object_digest
        if object_digest(result.to_record()) != object_digest(record):
            raise ValueError("program family carrier differs from its declaration")
        return result
