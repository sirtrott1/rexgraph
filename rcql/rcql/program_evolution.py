"""Explicit correspondence between retained operation topology versions."""
from dataclasses import dataclass, field

from .program_codec import dumps, loads, digest
from .plan_topology import PlanTopology
from .relation_topology import RelationTopology, _complex_payload, _complex_from_payload


def _tower(topology):
    if isinstance(topology, RelationTopology):
        return topology.boundary_tower()
    if not isinstance(topology, PlanTopology):
        raise TypeError("program evolution requires declared operation topology")
    from rexgraph.chain_map import CoordinateComplex
    from rexgraph.type_accession import CoordinateSpace
    base = CoordinateComplex.from_rex(topology.to_record())
    ports = topology.port_realization().domain.keys
    nodes = tuple(n["id"] for n in topology.declaration["nodes"])
    keys = (ports, nodes, *(s.keys for s in base.spaces[2:]))
    return CoordinateComplex(tuple(CoordinateSpace("plan/"+topology.coefficient_digest+"/"+str(k), row)
                                   for k, row in enumerate(keys)), base.boundaries)


def _payload(topology):
    return dict(kind="relations" if isinstance(topology, RelationTopology) else "plan",
                declaration=topology.declaration, tower=_complex_payload(_tower(topology)))


def _restore(payload):
    if not isinstance(payload, dict) or set(payload) != {"kind", "declaration", "tower"}:
        raise ValueError("invalid program evolution endpoint")
    tower = _complex_from_payload(payload["tower"])
    if payload["kind"] == "relations":
        return RelationTopology(payload["declaration"], _tower=tower)
    if payload["kind"] != "plan":
        raise ValueError("unknown program topology kind")
    result = PlanTopology(payload["declaration"])
    if _tower(result).coefficient_digest != tower.coefficient_digest:
        raise ValueError("program plan tower differs from its declared occurrences")
    return result


@dataclass(frozen=True)
class ProgramEvolution:
    """Structural change under declared occurrence matches, not program equivalence."""
    data: bytes
    dependencies: tuple = field(init=False)

    def __post_init__(self):
        if type(self.data) is not bytes:
            raise TypeError("program evolution requires immutable declaration bytes")
        old, new = self.endpoints()
        from .source_context import field_references
        object.__setattr__(self, "dependencies", field_references((old.to_record(), new.to_record())))

    @classmethod
    def compare(cls, old, new, matches):
        if type(old) is not type(new) or not isinstance(old, (PlanTopology, RelationTopology)):
            raise TypeError("program comparison requires the same declared topology kind")
        return cls(dumps(dict(schema="rcql.program-evolution", version=1,
            old=_payload(old), new=_payload(new), matches=tuple(tuple(tuple(p) for p in grade) for grade in matches))))

    def endpoints(self):
        data = loads(self.data)
        if (not isinstance(data, dict) or set(data) != {"schema", "version", "old", "new", "matches"}
                or data["schema"] != "rcql.program-evolution" or type(data["version"]) is not int
                or data["version"] != 1):
            raise ValueError("unsupported program evolution declaration")
        old, new = _restore(data["old"]), _restore(data["new"])
        if type(old) is not type(new):
            raise ValueError("program topology kinds differ")
        a, b = _tower(old), _tower(new)
        if len(a.spaces) != len(b.spaces) or len(data["matches"]) != len(a.spaces):
            raise ValueError("declare occurrence matches at every retained grade")
        for left, right, pairs in zip(a.spaces, b.spaces, data["matches"], strict=True):
            seen_a, seen_b = set(), set()
            for pair in pairs:
                if type(pair) is not tuple or len(pair) != 2:
                    raise ValueError("occurrence matches require source and target keys")
                x, y = pair
                if x not in left.keys or y not in right.keys or x in seen_a or y in seen_b:
                    raise ValueError("occurrence matches must be explicit partial bijections")
                seen_a.add(x); seen_b.add(y)
        return old, new

    @property
    def coefficient_digest(self):
        return digest(("rcql.program-evolution.v1", self.data))

    def correspondence(self, grade):
        from rexgraph.coordinate_map import CoordinateMap
        old, new = self.endpoints()
        a, b = _tower(old), _tower(new)
        if type(grade) is not int or not 0 <= grade < len(a.spaces):
            raise ValueError("program correspondence grade is outside the tower")
        left, right = a.spaces[grade], b.spaces[grade]
        pairs = loads(self.data)["matches"][grade]
        return CoordinateMap(left, right, tuple((right.keys.index(y), left.keys.index(x), 1) for x, y in pairs))

    def boundary_change(self, grade):
        from rexgraph.coordinate_map import CoordinateMap
        from rexgraph.temporal_calculus import TemporalOperation
        old, new = self.endpoints()
        a, b = _tower(old), _tower(new)
        if type(grade) is not int or not 1 <= grade < len(a.spaces):
            raise ValueError("boundary change requires an adjacent retained grade")
        left = CoordinateMap(a.spaces[grade], a.spaces[grade-1], a.boundaries[grade-1])
        right = CoordinateMap(b.spaces[grade], b.spaces[grade-1], b.boundaries[grade-1])
        return TemporalOperation(left, right, self.correspondence(grade), self.correspondence(grade-1)).defect

    def explain(self):
        old, new = self.endpoints()
        left, right = old.declaration, new.declaration
        matches = loads(self.data)["matches"]
        a, b = _tower(old), _tower(new)
        grades = []
        for k, pairs in enumerate(matches):
            old_keys, new_keys = dict(pairs), {y for _, y in pairs}
            grades.append(dict(grade=k, matched=pairs,
                removed=tuple(x for x in a.spaces[k].keys if x not in old_keys),
                added=tuple(y for y in b.spaces[k].keys if y not in new_keys)))
        nodes_a = {v["id"]: v for v in left["nodes"]}
        nodes_b = {v["id"]: v for v in right["nodes"]}
        changes = []
        for x, y in matches[1]:
            if x not in nodes_a or y not in nodes_b:
                continue
            before, after = nodes_a[x], nodes_b[y]
            fields = tuple((key, before.get(key), after.get(key)) for key in sorted(set(before) | set(after))
                           if key != "id" and dumps(before.get(key)) != dumps(after.get(key)))
            if fields or x != y:
                changes.append(dict(old=x, new=y, fields=fields))
        # Keep complete endpoint declarations: equal boundaries can hide changed captures.
        declarations = tuple((key, left.get(key), right.get(key)) for key in
            sorted((set(left) | set(right)) - {"nodes"}) if dumps(left.get(key)) != dumps(right.get(key)))
        return dict(schema="rcql.program-evolution-report", version=1,
            digest=self.coefficient_digest, old=old.coefficient_digest, new=new.coefficient_digest,
            grades=tuple(grades), operations=tuple(changes), declarations=declarations,
            claim="structural change under supplied occurrence correspondences",
            excluded=("program equivalence", "causality", "implicit metric", "observed output equality"))

    def to_record(self):
        from rexgraph.graph import RexGraph
        old, new = self.endpoints()
        record = RexGraph.from_cells([1, [[0]]])
        record.attach_metadata(1, 0, "rcql_program_evolution", self.data.decode())
        from .source_context import field_references
        refs = field_references((old.to_record(), new.to_record()))
        record._agent_meta = {"rcql_evidence": [v.as_record() for v in refs]}
        return record

    @classmethod
    def from_record(cls, record):
        from rexgraph.io.catalog import object_digest
        raw = record.get_metadata(1, 0, "rcql_program_evolution")
        if not isinstance(raw, str):
            raise ValueError("record has no program evolution")
        result = cls(raw.encode())
        if object_digest(result.to_record()) != object_digest(record):
            raise ValueError("program evolution carrier differs from its declaration")
        return result
