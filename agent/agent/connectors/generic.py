"""agent.connectors.generic - the long-tail edge/table connector.

For anything already shaped as relationships: an in-memory edge list, a
dataframe of pairs, a 2-column CSV, a dump. Point it at ``(src, dst[, weight])``
rows and it emits the complex directly.

    read(source) -> (rex, meta)

``source`` is an iterable of ``(src_label, dst_label)`` or
``(src_label, dst_label, weight)`` tuples.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from . import BaseConnector, Capabilities, ConnectorError


class GenericConnector(BaseConnector):
    CAPABILITIES = Capabilities(weights=True, schemes=("edges", "table"))

    def read(self, source: Any) -> Tuple[Any, Dict[str, Any]]:
        rows = list(source or [])
        if not rows:
            raise ConnectorError("empty edge list - nothing to form a complex")
        labels: List[str] = []
        for row in rows:
            for name in (row[0], row[1]):
                if name not in labels:
                    labels.append(name)
        idx = {n: i for i, n in enumerate(labels)}
        edges = [(r[0], r[1]) for r in rows]
        srcs = [idx[r[0]] for r in rows]
        tgts = [idx[r[1]] for r in rows]
        weighted = all(len(r) >= 3 for r in rows)
        weights = [float(r[2]) for r in rows] if weighted else None
        return self.result((srcs, tgts), vertex_labels=labels, edges=edges,
                           weights=weights, source="edges://in-memory")

    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES
