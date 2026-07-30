"""agent.connectors.graph: property-graph databases (Neo4j shape).

Nodes -> vertices, relationships -> edges. Harness-validatable here against an
in-memory ``{"nodes": [...], "relationships": [(src, dst), ...]}`` stand-in;
the live path (a ``neo4j://`` URI over the bolt driver) is a host-environment
task - the driver isn't present in-sandbox.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from . import BaseConnector, Capabilities


class GraphConnector(BaseConnector):
    CAPABILITIES = Capabilities(schemes=("neo4j", "bolt"))

    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    def read(self, source: Any) -> Tuple[Any, Dict[str, Any]]:
        if not isinstance(source, dict):
            raise NotImplementedError(
                "live property-graph reads need the host's bolt driver; pass an "
                "in-memory {'nodes': [...], 'relationships': [(src,dst),...]} "
                "structure to validate the shape in-sandbox")
        nodes: List[str] = list(source.get("nodes") or [])
        rels: List[Tuple[str, str]] = [tuple(r) for r in source.get("relationships") or []]
        for a, b in rels:
            for n in (a, b):
                if n not in nodes:
                    nodes.append(n)
        idx = {n: i for i, n in enumerate(nodes)}
        srcs = [idx[a] for a, _ in rels]
        tgts = [idx[b] for _, b in rels]
        return self.result((srcs, tgts), vertex_labels=nodes, edges=rels,
                           source="neo4j://in-memory")
