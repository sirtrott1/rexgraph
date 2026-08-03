"""agent.connectors.stream: streaming platforms (Kafka / Pulsar shape).

Structure-only: topics -> vertices, schema-registry references between topics
-> edges. Harness-validatable here against an in-memory
``{"topics": [...], "references": [(from, to), ...]}`` stand-in; consuming a
live cluster is a host-environment task (no broker in-sandbox).
"""
from __future__ import annotations

from typing import Any

from . import BaseConnector, Capabilities


class StreamConnector(BaseConnector):
    CAPABILITIES = Capabilities(schemes=("kafka", "pulsar"))

    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    def read(self, source: Any) -> tuple[Any, dict[str, Any]]:
        if not isinstance(source, dict):
            raise NotImplementedError(
                "live stream reads need the host's Kafka/Pulsar client; pass an "
                "in-memory {'topics': [...], 'references': [(from,to),...]} "
                "structure to validate the shape in-sandbox")
        topics: list[str] = list(source.get("topics") or [])
        refs: list[tuple[str, str]] = [tuple(r) for r in source.get("references") or []]
        for a, b in refs:
            for n in (a, b):
                if n not in topics:
                    topics.append(n)
        idx = {n: i for i, n in enumerate(topics)}
        srcs = [idx[a] for a, _ in refs]
        tgts = [idx[b] for _, b in refs]
        return self.result((srcs, tgts), vertex_labels=topics, edges=refs,
                           source="kafka://in-memory")
