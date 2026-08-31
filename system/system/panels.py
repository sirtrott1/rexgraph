"""Typed RCQL queries used by System views."""
from __future__ import annotations

from rcql import call, query, source
from rcql.describe import describe_rex


def panel_query(name: str, source_name: str, value):
    """Build the query for one System panel."""
    key = str(name).strip().lower()
    src = source(source_name)
    if key in ("overview", "state"):
        return query(src, call("DESCRIBE"))

    info = describe_rex(value)
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
