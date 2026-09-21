"""A retained relational observation of a validated native query plan."""
from __future__ import annotations

from .program_codec import dumps, loads, digest


class PlanTopology:
    """Operation occurrences with ordered ports and a shared value realization."""

    def __init__(self, declaration):
        declaration = loads(dumps(declaration))
        if not isinstance(declaration, dict) or declaration.get("schema") not in {"rcql.native-plan", "rcql.program-plan"}:
            raise ValueError("plan topology requires a native plan declaration")
        nodes = declaration.get("nodes", ())
        known = set()
        for node in nodes:
            key = node.get("id")
            if not isinstance(key, str) or not key or key in known:
                raise ValueError("plan nodes need distinct identities")
            if any(parent not in known for parent in node.get("inputs", ())):
                raise ValueError("plan input must refer to a preceding result")
            known.add(key)
        if not nodes or any(key not in known for key in declaration.get("outputs", ())):
            raise ValueError("plan outputs require known operation results")
        self._data = dumps(declaration)
        self._rex = None

    @classmethod
    def from_plan(cls, plan, dependencies=()):
        from .native_plan import NativePlan
        if not isinstance(plan, NativePlan):
            raise TypeError("plan topology requires a bound NativePlan")
        declaration = plan.explain()
        declaration["dependencies"] = tuple(ref.as_record() for ref in dependencies)
        return cls(declaration)

    @property
    def declaration(self):
        return loads(self._data)

    @property
    def coefficient_digest(self):
        return digest(("rcql.plan-topology.v1", self.declaration))

    def to_record(self):
        import numpy as np
        from rexgraph.graph import RexGraph
        declaration = self.declaration
        ports, groups, roles = [], [], []
        for node in declaration["nodes"]:
            group = [len(ports)]
            ports.append(node["id"]+"/result")
            roles.append((node["id"], "result", 0, node["id"]))
            for i, parent in enumerate(node.get("inputs", ())):
                group.append(len(ports))
                ports.append(node["id"]+"/argument/"+str(i))
                roles.append((node["id"], "argument", i, parent))
            groups.append(group)
        ptr, indices = [0], []
        for group in groups:
            indices.extend(group); ptr.append(len(indices))
        rex = RexGraph.from_hypergraph(np.array(ptr, dtype=np.int64), np.array(indices, dtype=np.int64))
        states = [declaration["source"]["state"]] if "source" in declaration else [s["state"] for s in declaration.get("sources", ())]
        states += list(declaration.get("dependencies", ()))
        evidence = {}
        for state in states:
            if state.get("state_digest") is not None:
                key = (state["state_digest"], state.get("record_id"), state.get("record_version", state.get("version")))
                evidence[key] = {"state_digest": key[0], "record_id": key[1], "version": key[2]}
        rex._agent_meta = {"rcql_evidence": list(evidence.values()), "vertex_labels": ports, "rcql_plan_schema": 1,
                           "rcql_plan": self._data.decode("utf8"),
                           "rcql_plan_digest": self.coefficient_digest,
                           "rcql_plan_ports": [list(role) for role in roles]}
        return rex

    @classmethod
    def from_record(cls, record):
        from rexgraph.io.catalog import object_digest
        meta = getattr(record, "_agent_meta", {})
        if meta.get("rcql_plan_schema") != 1:
            raise ValueError("record does not carry the plan topology schema")
        result = cls(loads(meta["rcql_plan"]))
        if result.coefficient_digest != meta.get("rcql_plan_digest"):
            raise ValueError("stored plan declaration identity differs")
        if object_digest(result.to_record()) != object_digest(record):
            raise ValueError("stored plan cells differ from their declaration")
        return result

    def port_realization(self):
        from rexgraph.coordinate_map import CoordinateMap
        from rexgraph.type_accession import CoordinateSpace
        declaration = self.declaration
        keys = tuple(node["id"] for node in declaration["nodes"])
        positions = {key: i for i, key in enumerate(keys)}
        ports, entries = [], []
        for node in declaration["nodes"]:
            ports.append(node["id"]+"/result")
            entries.append((positions[node["id"]], len(ports)-1, 1))
            for i, parent in enumerate(node.get("inputs", ())):
                ports.append(node["id"]+"/argument/"+str(i))
                entries.append((positions[parent], len(ports)-1, 1))
        domain = CoordinateSpace("query_ports/"+self.coefficient_digest, tuple(ports))
        codomain = CoordinateSpace("query_values/"+self.coefficient_digest, keys)
        return CoordinateMap(domain, codomain, tuple(entries))
