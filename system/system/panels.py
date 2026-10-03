"""Typed RCQL queries used by System views."""
from __future__ import annotations

from rcql import call, query
from rcql.binding import classify
from rcql.describe import describe_rex
from rcql.types import ValueKind
from .inspection import available_views, readable, source_expression


def panel_query(name: str, source_name: str, value):
    """Build the query for one System panel."""
    key = str(name).strip().lower()
    raw = readable(value)
    src = source_expression(source_name, value)
    kind = classify(raw)
    if key == "rcdb" and kind is ValueKind.RCDB_STORE:
        if "RCDB" not in available_views(value):
            raise TypeError("System RCDB panel requires the RCQL statistics capability")
        return query(src, call("RCDB_STATS"))
    if kind not in (ValueKind.REX, ValueKind.DATASET):
        raise TypeError("System structural panels require a Rex or declared dataset source")
    if key not in ("overview", "state", "structure", "hodge", "character", "flow"):
        raise KeyError(f"System panel {name!r} has no query")
    if key in ("overview", "state"):
        return query(src, call("DESCRIBE"))

    info = describe_rex(raw.materialize() if kind is ValueKind.DATASET else raw)
    dimension = int(info.get("dimension", 0))

    if key == "structure":
        items = [call("DESCRIBE")]
        for grade in range(1, dimension + 1):
            items.extend((call("RANK", grade), call("NULLITY", grade)))
        return query(src, *items)

    if key == "hodge":
        items = [call("BETTI", grade) for grade in range(dimension + 1)]
        items.extend(call("HODGE_OPERATOR", grade) for grade in range(dimension + 1))
        return query(src, *items)

    if key == "character":
        return query(src, call("CHARACTER"))

    if key == "flow":
        items = [call("HODGE_OPERATOR", grade) for grade in range(dimension + 1)]
        return query(src, *items)

    raise KeyError(f"System panel {name!r} has no query")
