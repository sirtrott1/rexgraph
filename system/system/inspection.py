"""Source metadata and query forms for the System observatory.

RCQL classifies and reads sources. System presents those contracts and never
opens a path, reconstructs a store or selects a reader on behalf of query text.
"""
from __future__ import annotations

import json

from rcql import BoundSource, SourcePolicy, call, dataset
from rcql.binding import classify
from rcql.describe import describe_rex
from rcql.executor import format_expr
from rcql.types import ValueKind


def readable(value):
    return value.require("read") if isinstance(value, BoundSource) else value


def available_views(value):
    """Advertise executable views without reading store data or materializing input."""
    raw = readable(value)
    kind = classify(raw)
    views = ["Overview"]
    if kind in (ValueKind.REX, ValueKind.DATASET):
        views.extend(("Structure", "Hodge", "Character", "Flow", "State"))
    elif kind is ValueKind.RCDB_STORE:
        from rcql.binding import bind, resolve
        policy = value.policy if isinstance(value, BoundSource) else SourcePolicy.allow("*")
        try:
            resolve(bind("System panel", raw, policy), "RCDB_STATS")
        except (TypeError, PermissionError):
            pass
        else:
            views.append("RCDB")
    elif kind is ValueKind.CATALOG_ENTRY_SET:
        if not isinstance(value, BoundSource) or all(value.policy.permits(p) for p in ("files", "file_read")):
            views.append("Files")
    views.append("Queries")
    return views


def source_expression(name, value):
    """Use the declared dataset transform instead of a direct Rex parameter."""
    kind = classify(readable(value))
    if kind is ValueKind.DATASET:
        return dataset(name)
    form = {ValueKind.REX: "REX", ValueKind.RCDB_STORE: "RCDB",
            ValueKind.CATALOG_ENTRY_SET: "CATALOG"}.get(kind)
    if form is None:
        raise TypeError(f"System has no default query form for {kind.value!r}")
    return call(form, name)


def default_query(name, value):
    raw = readable(value)
    kind = classify(raw)
    if kind in (ValueKind.REX, ValueKind.DATASET):
        returns = (call("DESCRIBE"), call("BETTI", 0))
    elif kind is ValueKind.RCDB_STORE:
        if "RCDB" not in available_views(value):
            return None
        returns = (call("RCDB_STATS"),)
    elif kind is ValueKind.CATALOG_ENTRY_SET:
        if isinstance(value, BoundSource) and not all(value.policy.permits(p) for p in ("files", "file_read")):
            return None
        returns = (call("FILES", 100, 0),)
    else:
        return None
    # RCQL text strings use JSON double quotes. Its diagnostic formatter uses
    # Python repr, so render this single registered name argument with JSON.
    # Names containing quotes, whitespace or a slash stay literal data.
    form = source_expression(name, value).name
    return "FROM " + form + "(" + json.dumps(name) + ") RETURN " + ", ".join(
        format_expr(item) for item in returns)


def source_description(value, *, detail=False):
    raw = readable(value)
    kind = classify(raw)
    if kind is ValueKind.DATASET:
        record = raw.as_record()
        out = {"kind": "Dataset", "digest": record["digest"],
               "declaration_digest": record["declaration_digest"],
               "state_digest": record["state_digest"],
               "reader": {key: record["reader"][key] for key in ("name", "version")}}
        if detail:
            out = {**describe_rex(raw.materialize()), **out}
        return out
    if kind is ValueKind.REX:
        return describe_rex(raw)
    return {"kind": kind.value}


def source_row(name, value):
    raw = value.value if isinstance(value, BoundSource) else value
    row = {"name": name, "type": type(raw).__name__, "scoped": isinstance(value, BoundSource),
           "accessible": False, "default_query": None, "panels": []}
    try:
        raw = readable(value)
        row.update(accessible=True, kind=classify(raw).value,
                   description=source_description(value), default_query=default_query(name, value),
                   panels=available_views(value))
    except PermissionError:
        pass
    except (TypeError, ValueError) as exc:
        row["error"] = str(exc)
    return row


__all__ = ["available_views", "source_expression", "source_description", "source_row"]
