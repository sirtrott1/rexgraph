"""Finite recursive declarations and evaluated native values in ordinary records."""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

from .program_codec import dumps, loads


def result_record(result):
    from rexgraph.graph import RexGraph
    from rexgraph.io.field_state import FIELD_VALUES
    from rexgraph.io.section_state import SECTION_VALUES
    from rexgraph.io.model_state import MODEL_VALUES
    from .name_relation import NameRelation
    from .program import Program
    from .program_transformation import ProgramTransformation
    record = RexGraph.from_cells([1, [[0]]])
    count = 0
    def put(value):
        nonlocal count
        if isinstance(value, (*FIELD_VALUES, *SECTION_VALUES, *MODEL_VALUES)):
            key = "recursive_value_" + str(count)
            count += 1
            record.attach_metadata(1, 0, key, value)
            return {"native": key}
        if isinstance(value, NameRelation):
            return {"name": value.to_bytes()}
        if isinstance(value, Program):
            _ = value.coefficient_digest
            return {"program": value.to_bytes()}
        if isinstance(value, ProgramTransformation):
            return {"transformation": value.to_bytes()}
        if type(value) in (tuple, list):
            return {"sequence": type(value).__name__, "values": tuple(put(v) for v in value)}
        if type(value) is dict and all(isinstance(k, str) for k in value):
            return {"mapping": tuple((k, put(v)) for k, v in sorted(value.items()))}
        dumps(value)
        return {"literal": value}
    data = {"schema": "rcql.recursion-result", "version": 1, "value": put(result.value),
            "definition_digest": result.definition_digest, "entry": result.entry,
            "calls": result.calls, "evaluations": result.evaluations,
            "invocations": result.invocations,
            "history": tuple((i, name, put(value)) for i, name, value in result.history),
            "arguments": tuple((i, name, tuple((port, put(value)) for port, value in values))
                               for i, name, values in result.arguments),
            "dependencies": tuple(v.as_record() for v in result.dependencies),
            "digest": result.coefficient_digest}
    record.attach_metadata(1, 0, "rcql_recursive_result", dumps(data).decode("utf8"))
    record._agent_meta = {"rcql_evidence": list(data["dependencies"])}
    return record


def restore_result(record, references=()):
    from rexgraph.tensor_field import FieldSource
    from rexgraph.io.field_state import FIELD_VALUES
    from rexgraph.io.section_state import SECTION_VALUES
    from rexgraph.io.model_state import MODEL_VALUES
    from .recursive_program import RecursionResult
    from .name_relation import NameRelation
    from .program_transformation import ProgramTransformation
    raw = record.get_metadata(1, 0, "rcql_recursive_result")
    if not isinstance(raw, str):
        raise ValueError("record has no completed recursive result")
    data = loads(raw)
    expected = {"schema", "version", "value", "definition_digest", "entry", "calls", "evaluations",
                "invocations", "history", "dependencies", "arguments", "digest"}
    if not isinstance(data, dict) or set(data) != expected or data["schema"] != "rcql.recursion-result" or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("invalid recursive result schema")
    references = tuple(references)
    refs = {r.coefficient_digest: r for r in references if isinstance(r, FieldSource)}
    if len(refs) != len(tuple(references)):
        raise ValueError("result rebinding requires distinct native source references")
    dependencies = tuple(FieldSource(None, row["record_id"], row["version"], row["state_digest"]) for row in data["dependencies"])
    if refs and set(refs) != {v.coefficient_digest for v in dependencies}:
        raise ValueError("every result dependency must be rebound explicitly")
    def bind(value):
        if isinstance(value, FieldSource):
            if not refs:
                return value
            result = refs.get(value.coefficient_digest)
            if result is None or result.source is None:
                raise ValueError("native result has an unbound source")
            result.check()
            return result
        if isinstance(value, tuple):
            return tuple(bind(v) for v in value)
        if is_dataclass(value) and type(value).__module__.startswith("rexgraph."):
            return replace(value, **{f.name: bind(getattr(value, f.name)) for f in fields(value) if f.init})
        return value
    def take(node):
        if not isinstance(node, dict):
            raise ValueError("invalid recursive value tree")
        if set(node) == {"native"}:
            value = record.get_metadata(1, 0, node["native"])
            if not isinstance(value, (*FIELD_VALUES, *SECTION_VALUES, *MODEL_VALUES)):
                raise ValueError("stored recursive value is not its declared native type")
            return bind(value)
        if set(node) == {"name"}:
            return NameRelation.from_bytes(node["name"])
        if set(node) == {"program"}:
            from .program import Program
            return Program.from_bytes(node["program"])
        if set(node) == {"transformation"}:
            return ProgramTransformation.from_bytes(node["transformation"])
        if set(node) == {"literal"}:
            return node["literal"]
        if set(node) == {"mapping"}:
            if len({k for k, _ in node["mapping"]}) != len(node["mapping"]):
                raise ValueError("duplicate recursive record key")
            return {k: take(v) for k, v in node["mapping"]}
        if set(node) == {"sequence", "values"} and node["sequence"] in {"tuple", "list"}:
            values = tuple(take(v) for v in node["values"])
            return values if node["sequence"] == "tuple" else list(values)
        raise ValueError("unknown recursive value encoding")
    result = RecursionResult(take(data["value"]), data["definition_digest"], data["entry"],
        data["calls"], data["evaluations"], data["invocations"],
        tuple((i, name, take(v)) for i, name, v in data["history"]), tuple(bind(v) for v in dependencies),
        tuple((i, name, tuple((port, take(value)) for port, value in values)) for i, name, values in data["arguments"]))
    if result.coefficient_digest != data["digest"]:
        raise ValueError("recursive result differs from its recorded identity")
    return result


def feedback_record(system):
    """Retain one primary relation for each finite local feedback equation."""
    from rexgraph.affine_feedback import AffineFeedback
    from rexgraph.graph import RexGraph
    if not isinstance(system, AffineFeedback):
        raise TypeError("feedback persistence requires its native equation system")
    system.check_state()
    groups, ports, descriptions = [], [], []
    for index, equation in enumerate(system.equations):
        occurrences = (("output", equation.output), *((str(i), key) for i, (key, _) in enumerate(equation.terms)))
        group = []
        for role, variable in occurrences:
            group.append(len(ports))
            ports.append((equation.name, role, variable))
        groups.append(group)
        descriptions.append({"name": equation.name, "output": equation.output,
            "terms": tuple((key, action.entries) for key, action in equation.terms),
            "forcing": "feedback_forcing_"+str(index)})
    record = RexGraph.from_cells([len(ports), groups])
    for index, equation in enumerate(system.equations):
        record.attach_metadata(1, index, descriptions[index]["forcing"], equation.forcing)
    source_refs = (system.source, *system.dependencies)
    refs = tuple({v.coefficient_digest: v for v in source_refs}.values())
    declaration = {"schema": "rcql.affine-feedback", "version": 1, "name": system.name,
        "spaces": tuple((key, space.name, space.keys) for key, space in system.spaces),
        "equations": tuple(descriptions), "ports": tuple(ports),
        "source": system.source.as_record(), "dependencies": tuple(v.as_record() for v in refs),
        "digest": system.coefficient_digest}
    encoded = dumps(declaration)
    if len(encoded) > 4*1024*1024:
        raise ValueError("feedback declaration exceeds the finite record limit")
    record._agent_meta = {"rcql_feedback": encoded.decode("utf8"),
                          "rcql_evidence": [v.as_record() for v in refs]}
    return record


def restore_feedback(record, references):
    """Rebind all recorded sources before exposing a stored feedback system."""
    from rexgraph.affine_feedback import AffineFeedback, FeedbackEquation
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.type_accession import CoordinateSpace
    from rexgraph.tensor_field import FieldSource, TensorField
    from rexgraph.io.catalog import object_digest
    raw = getattr(record, "_agent_meta", {}).get("rcql_feedback")
    if not isinstance(raw, str):
        raise ValueError("record has no affine feedback declaration")
    data = loads(raw)
    if (not isinstance(data, dict) or set(data) != {"schema", "version", "name", "spaces", "equations", "ports", "source", "dependencies", "digest"}
            or data["schema"] != "rcql.affine-feedback" or type(data["version"]) is not int or data["version"] != 1):
        raise ValueError("unsupported feedback schema")
    references = tuple(references)
    by_key = {r.coefficient_digest: r for r in references if isinstance(r, FieldSource)}
    def source(row):
        ref = FieldSource(None, row["record_id"], row["version"], row["state_digest"])
        if ref.coefficient_digest not in by_key:
            raise ValueError("feedback requires every recorded source binding")
        value = by_key[ref.coefficient_digest]
        if value.source is None:
            raise ValueError("feedback sources must be live bindings")
        value.check()
        return value
    if (len(by_key) != len(references)
            or {source(v).coefficient_digest for v in data["dependencies"]} != set(by_key)):
        raise ValueError("feedback source bindings differ from its recorded dependencies")
    spaces = tuple((key, CoordinateSpace(name, tuple(keys))) for key, name, keys in data["spaces"])
    known = dict(spaces)
    equations = []
    for index, row in enumerate(data["equations"]):
        if set(row) != {"name", "output", "terms", "forcing"}:
            raise ValueError("invalid feedback equation fields")
        forcing = record.get_metadata(1, index, row["forcing"])
        if not isinstance(forcing, TensorField) or forcing.source is None:
            raise ValueError("feedback forcing field is missing")
        forcing = replace(forcing, source=source(forcing.source.as_record()),
                          dependencies=tuple(source(r.as_record()) for r in forcing.dependencies))
        terms = tuple((key, CoordinateMap(known[key], known[row["output"]], entries)) for key, entries in row["terms"])
        equations.append(FeedbackEquation(row["name"], row["output"], terms, forcing))
    result = AffineFeedback(data["name"], spaces, tuple(equations), source(data["source"]))
    if result.coefficient_digest != data["digest"] or object_digest(feedback_record(result)) != object_digest(record):
        raise ValueError("feedback record differs from its primary equation declaration")
    return result
