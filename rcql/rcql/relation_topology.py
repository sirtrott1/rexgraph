"""Primary operation occurrences and checked relations between their boundaries."""
from __future__ import annotations

from fractions import Fraction
from dataclasses import dataclass

from .ast import Parameter, Literal, Call, ListExpr, Comparison, Member, Alias
from .program_codec import dumps, loads, digest


@dataclass(frozen=True)
class OperationRelationCell:
    """One exact retained relation at a declared operation grade."""
    grade: int
    identity: str
    boundary: tuple


def _complex_payload(complex_):
    return {
        "schema": "rcql.operation-coordinate-complex",
        "version": 1,
        "spaces": tuple((space.name, tuple(space.keys)) for space in complex_.spaces),
        "boundaries": tuple(complex_.boundaries),
        "digest": complex_.coefficient_digest,
    }


def _complex_from_payload(data):
    from rexgraph.chain_map import CoordinateComplex, _chain_residual
    from rexgraph.type_accession import CoordinateSpace
    if not isinstance(data, dict) or set(data) != {"schema", "version", "spaces", "boundaries", "digest"}:
        raise ValueError("invalid persisted operation complex")
    if data["schema"] != "rcql.operation-coordinate-complex" or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("unsupported persisted operation complex")
    spaces = tuple(CoordinateSpace(name, tuple(keys)) for name, keys in data["spaces"])
    complex_ = CoordinateComplex(spaces, tuple(data["boundaries"]))
    if _chain_residual(complex_):
        raise ValueError("persisted operation complex violates the chain condition")
    if complex_.coefficient_digest != data["digest"]:
        raise ValueError("persisted operation complex digest does not match its exact tower")
    return complex_


class RelationTopology:
    """A retained declaration with explicit shared ports and certified fillings."""
    def __init__(self, declaration, *, _tower=None):
        data = loads(dumps(declaration))
        if not isinstance(data, dict) or set(data) != {"schema", "version", "name", "nodes", "outputs", "composites", "fillings", "origin"}:
            raise ValueError("invalid operation relation topology")
        if data["schema"] != "rcql.operation-relations" or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported operation relation topology")
        nodes = data["nodes"]
        if not isinstance(nodes, (tuple, list)) or not nodes:
            raise ValueError("operation topology requires primary occurrences")
        known = set()
        for node in nodes:
            if set(node) != {"id", "name", "kind", "inputs", "roles"} or not isinstance(node["id"], str) or node["id"] in known:
                raise ValueError("operation occurrences require distinct identities")
            if len(node["inputs"]) != len(node["roles"]):
                raise ValueError("every argument occurrence requires a role")
            known.add(node["id"])
        if any(parent not in known for node in nodes for parent in node["inputs"]) or any(v not in known for v in data["outputs"]):
            raise ValueError("every operation port requires a declared value identity")
        self._data = dumps(data)
        tower = self._build_boundary_tower() if _tower is None else _tower
        self._validate_persisted_tower(tower)
        self._tower = tower

    @property
    def declaration(self):
        return loads(self._data)

    @property
    def coefficient_digest(self):
        return digest(self.declaration)

    @staticmethod
    def _expression(expression, prefix, nodes, parameter_ids, references=None):
        def add(name, kind, inputs=(), roles=()):
            key = prefix + "/" + str(len(nodes))
            nodes.append({"id": key, "name": name, "kind": kind, "inputs": tuple(inputs), "roles": tuple(roles)})
            return key
        if isinstance(expression, Parameter):
            key = (prefix.split("/")[0], expression.name)
            if key not in parameter_ids:
                parameter_ids[key] = add(expression.name, "parameter")
            return parameter_ids[key]
        if isinstance(expression, Literal):
            return add("literal", "literal")
        if isinstance(expression, Call):
            if expression.name == "RECURSE":
                target = expression.args[0].value
                from .recursive_program import recursive_arguments
                definition = references.definition(target)
                inputs = tuple(RelationTopology._expression(v, prefix, nodes, parameter_ids, references)
                               for v in recursive_arguments(expression, definition))
                return add(target, "recursive_reference", (*inputs, "definition/"+target),
                           (*(p.name for p in definition.inputs), "definition"))
            from .arguments import EXPRESSION_ARGUMENTS
            inputs = tuple(RelationTopology._expression(v, prefix, nodes, parameter_ids, references) for v in expression.args)
            labels = EXPRESSION_ARGUMENTS[expression.name][0][:len(inputs)]
            return add(expression.name, "operation", inputs, labels)
        if isinstance(expression, ListExpr):
            inputs = tuple(RelationTopology._expression(v, prefix, nodes, parameter_ids, references) for v in expression.items)
            return add("list", "tuple", inputs, tuple(str(i) for i in range(len(inputs))))
        if isinstance(expression, Comparison):
            inputs = tuple(RelationTopology._expression(v, prefix, nodes, parameter_ids, references)
                           for v in (expression.left, expression.right))
            return add(expression.operation, "predicate", inputs, ("left", "right"))
        if isinstance(expression, (Member, Alias)):
            value = RelationTopology._expression(expression.value, prefix, nodes, parameter_ids, references)
            return add(expression.name, type(expression).__name__, (value,), ("value",))
        raise TypeError("unsupported operation expression")

    @classmethod
    def from_name(cls, relation):
        nodes = []
        from .name_relation import substitute
        replacements = {k: Literal(v) for k, v in relation.bound_values}
        params = {}
        for key, expression in relation.stages:
            value = cls._expression(substitute(expression, replacements), "body", nodes, params)
            params[("body", key)] = value
        body = substitute(relation.body, replacements)
        output = cls._expression(body, "body", nodes, params)
        data = {"schema": "rcql.operation-relations", "version": 1, "name": relation.name,
                "nodes": tuple(nodes), "outputs": (output,),
                "composites": (("name/"+relation.name, output),), "fillings": (),
                "origin": relation.declaration()}
        return cls(data)

    @classmethod
    def from_recursive(cls, program):
        nodes, outputs, params = [], [], {}
        for definition in program.definitions:
            arguments = tuple(cls._expression(v, definition.name+"/"+role, nodes, params, program)
                              for role, v in (("guard", definition.guard), ("base", definition.base), ("step", definition.step)))
            key = "definition/"+definition.name
            nodes.append({"id": key, "name": definition.name, "kind": "conditional_definition",
                          "inputs": arguments, "roles": ("guard", "base", "step")})
            outputs.append(key)
        return cls({"schema": "rcql.operation-relations", "version": 1, "name": program.name,
                    "nodes": tuple(nodes), "outputs": tuple(outputs), "composites": (),
                    "fillings": (), "origin": program.declaration()})

    @classmethod
    def from_invocations(cls, result):
        nodes = []
        children = {}
        for item in result.invocations:
            if item["parent"] is not None:
                children.setdefault(item["parent"], []).append("call/"+str(item["id"]))
        for item in result.invocations:
            inputs = (("call/"+str(item["parent"]),) if item["branch"] == "iterate"
                      else () if item["branch"] == "initial" else tuple(children.get(item["id"], ())))
            nodes.append({"id": "call/"+str(item["id"]), "name": item["name"], "kind": item["branch"],
                          "inputs": inputs, "roles": tuple(str(i) for i in range(len(inputs)))})
        return cls({"schema": "rcql.operation-relations", "version": 1, "name": result.entry,
                    "nodes": tuple(nodes), "outputs": ("call/"+str(result.invocations[-1]["id"])
                    if result.invocations[0]["branch"] == "initial" else "call/0",), "composites": (), "fillings": (),
                    "origin": {"definition_digest": result.definition_digest, "invocations": result.invocations}})

    def _build_boundary_tower(self):
        from rexgraph.chain_map import CoordinateComplex, _chain_residual
        from rexgraph.type_accession import CoordinateSpace
        data = self.declaration
        nodes = data["nodes"]
        indices = {node["id"]: i for i, node in enumerate(nodes)}
        columns = []
        for i, node in enumerate(nodes):
            column = {i: Fraction(-1 if node["inputs"] else 1)}
            for parent in node["inputs"]:
                row = indices[parent]
                column[row] = column.get(row, 0) + Fraction(1, len(node["inputs"]))
            columns.append({i: v for i, v in column.items() if v})
        relation_keys = [node["id"] for node in nodes]
        fillings, filling_names = [], []
        for name, root in data["composites"]:
            if root not in indices or name in relation_keys:
                raise ValueError("a composite requires a declared root and new identity")
            # Propagate shares on the retained graph, not on enumerated paths.
            order, visited, active = [], set(), set()
            pending = [(root, False)]
            while pending:
                key, finishing = pending.pop()
                if finishing:
                    active.remove(key)
                    visited.add(key)
                    order.append(key)
                    continue
                if key in visited:
                    continue
                if key in active:
                    raise ValueError("a cyclic definition requires feedback semantics, not finite expansion")
                active.add(key)
                pending.append((key, True))
                for parent in reversed(nodes[indices[key]]["inputs"]):
                    if nodes[indices[parent]]["inputs"]:
                        pending.append((parent, False))
            coefficients = {indices[root]: Fraction(1)}
            for key in reversed(order):
                i = indices[key]
                weight = coefficients.get(i, Fraction(0))
                node = nodes[i]
                for parent in node["inputs"]:
                    if nodes[indices[parent]]["inputs"]:
                        j = indices[parent]
                        coefficients[j] = coefficients.get(j, 0) + weight / len(node["inputs"])
            aggregate = {}
            for col, coefficient in coefficients.items():
                for row, value in columns[col].items():
                    aggregate[row] = aggregate.get(row, 0) + coefficient*value
            index = len(columns)
            columns.append({i: v for i, v in aggregate.items() if v})
            relation_keys.append(name)
            filling = {i: -v for i, v in coefficients.items() if v}
            filling[index] = Fraction(1)
            fillings.append(filling)
            filling_names.append("expansion/"+name)
        for name, declaration in data["fillings"]:
            if name in filling_names:
                raise ValueError("filling identities must be distinct")
            filling = {}
            for key, coefficient in declaration:
                if key not in relation_keys:
                    raise ValueError("filling names an absent primary relation")
                from rexgraph.graded_metric import _fraction
                index = relation_keys.index(key)
                filling[index] = filling.get(index, 0) + _fraction(coefficient)
            filling = {i: v for i, v in filling.items() if v}
            if not filling:
                raise ValueError("an asserted filling must be nonzero")
            fillings.append(filling)
            filling_names.append(name)
        spaces = (CoordinateSpace("operation_values", tuple(indices)),
                  CoordinateSpace("operation_relations", tuple(relation_keys)),
                  CoordinateSpace("operation_fillings", tuple(filling_names)))
        boundaries = tuple(tuple((i, j, v) for j, col in enumerate(matrix) for i, v in sorted(col.items()))
                           for matrix in (columns, fillings))
        tower = CoordinateComplex(spaces, boundaries)
        if _chain_residual(tower):
            raise ValueError("the declared relation between operations violates the chain condition")
        return tower

    def _validate_persisted_tower(self, tower):
        from rexgraph.chain_map import CoordinateComplex, _chain_residual
        if not isinstance(tower, CoordinateComplex) or len(tower.spaces) != 3:
            raise TypeError("operation topology requires an exact three grade coordinate complex")
        if _chain_residual(tower):
            raise ValueError("the declared relation between operations violates the chain condition")
        data = self.declaration
        node_keys = tuple(node["id"] for node in data["nodes"])
        composite_keys = tuple(name for name, _ in data["composites"])
        filling_keys = tuple("expansion/" + name for name in composite_keys) + tuple(name for name, _ in data["fillings"])
        expected = (
            ("operation_values", node_keys),
            ("operation_relations", node_keys + composite_keys),
            ("operation_fillings", filling_keys),
        )
        actual = tuple((space.name, tuple(space.keys)) for space in tower.spaces)
        if actual != expected:
            raise ValueError("persisted operation complex has incompatible grade identities")
        columns = [{} for _ in tower.spaces[1].keys]
        for row, column, value in tower.boundaries[0]:
            columns[column][row] = value
        indices = {key: i for i, key in enumerate(node_keys)}
        for i, node in enumerate(data["nodes"]):
            expected_column = {i: Fraction(-1 if node["inputs"] else 1)}
            for parent in node["inputs"]:
                row = indices[parent]
                expected_column[row] = expected_column.get(row, 0) + Fraction(1, len(node["inputs"]))
            expected_column = {row: value for row, value in expected_column.items() if value}
            if columns[i] != expected_column:
                raise ValueError("persisted operation complex changed a primary operation boundary")
        declared = dict(data["fillings"])
        c2_columns = [{} for _ in tower.spaces[2].keys]
        for row, column, value in tower.boundaries[1]:
            c2_columns[column][row] = value
        relation_index = {key: i for i, key in enumerate(tower.spaces[1].keys)}
        offset = len(composite_keys)
        from rexgraph.graded_metric import _fraction
        for j, name in enumerate(tuple(declared)):
            expected_column = {}
            for key, coefficient in declared[name]:
                row = relation_index[key]
                expected_column[row] = expected_column.get(row, 0) + _fraction(coefficient)
            expected_column = {row: value for row, value in expected_column.items() if value}
            if c2_columns[offset + j] != expected_column:
                raise ValueError("persisted operation complex changed an asserted co relation")

    def boundary_tower(self):
        """Return the exact retained operation complex without rebuilding its co relations."""
        return self._tower

    def relations(self, grade):
        """Return exact retained relation cells at one operation grade."""
        if type(grade) is not int or grade < 1 or grade >= len(self._tower.spaces):
            raise ValueError("operation relation grade is outside the retained tower")
        keys = self._tower.spaces[grade].keys
        lower = self._tower.spaces[grade - 1].keys
        columns = [[] for _ in keys]
        for row, column, value in self._tower.boundaries[grade - 1]:
            columns[column].append((lower[row], value))
        return tuple(OperationRelationCell(grade, key, tuple(column))
                     for key, column in zip(keys, columns, strict=True))

    def corelations(self):
        """Return the retained grade two operation co relations."""
        return self.relations(2)

    def declare_filling(self, name, coefficients):
        data = self.declaration
        data["fillings"] = (*data["fillings"], (name, tuple(coefficients)))
        return type(self)(data)

    def port_realization(self):
        from rexgraph.coordinate_map import CoordinateMap
        from rexgraph.type_accession import CoordinateSpace
        nodes = self.declaration["nodes"]
        indices = {node["id"]: i for i, node in enumerate(nodes)}
        ports, entries = [], []
        for node in nodes:
            ports.append(node["id"]+"/result")
            entries.append((indices[node["id"]], len(ports)-1, 1))
            for i, parent in enumerate(node["inputs"]):
                ports.append(node["id"]+"/argument/"+str(i))
                entries.append((indices[parent], len(ports)-1, 1))
        return CoordinateMap(CoordinateSpace("operation_ports", tuple(ports)),
                             CoordinateSpace("operation_values", tuple(indices)), tuple(entries))

    def _record(self, schema):
        from rexgraph.graph import RexGraph
        import numpy as np
        ptr, indices, labels = [0], [], []
        for node in self.declaration["nodes"]:
            size = 1 + len(node["inputs"])
            labels.extend((node["id"] + "/result", *(node["id"] + "/argument/" + str(i) for i in range(size - 1))))
            indices.extend(range(ptr[-1], ptr[-1] + size))
            ptr.append(ptr[-1] + size)
        result = RexGraph.from_hypergraph(np.asarray(ptr, dtype=np.int64), np.asarray(indices, dtype=np.int64))
        meta = {
            "vertex_labels": labels,
            "rcql_relation_schema": schema,
            "rcql_relation_declaration": self._data.decode("utf8"),
            "rcql_relation_digest": self.coefficient_digest,
        }
        if schema == 2:
            payload = _complex_payload(self._tower)
            meta.update({
                "rcql_operation_complex": dumps(payload).decode("utf8"),
                "rcql_operation_complex_digest": self._tower.coefficient_digest,
                "rcql_operation_top_grade": len(self._tower.spaces) - 1,
            })
        result._agent_meta = meta
        return result

    def to_record(self):
        """Persist the primary carrier and the exact higher operation relations."""
        return self._record(2)

    @classmethod
    def from_record(cls, record):
        from rexgraph.io.catalog import object_digest
        meta = getattr(record, "_agent_meta", {})
        schema = meta.get("rcql_relation_schema")
        if schema not in (1, 2):
            raise ValueError("record has no operation relation topology")
        declaration = loads(meta["rcql_relation_declaration"])
        if schema == 1:
            result = cls(declaration)
            expected = result._record(1)
        else:
            raw = meta.get("rcql_operation_complex")
            if not isinstance(raw, str):
                raise ValueError("record has no persisted exact operation complex")
            tower = _complex_from_payload(loads(raw))
            if tower.coefficient_digest != meta.get("rcql_operation_complex_digest"):
                raise ValueError("stored operation complex identity does not match its exact tower")
            if meta.get("rcql_operation_top_grade") != len(tower.spaces) - 1:
                raise ValueError("stored operation complex grade does not match its exact tower")
            result = cls(declaration, _tower=tower)
            expected = result._record(2)
        if result.coefficient_digest != meta.get("rcql_relation_digest") or object_digest(expected) != object_digest(record):
            raise ValueError("stored operation topology differs from its primary declaration")
        return result
