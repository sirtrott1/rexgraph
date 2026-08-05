"""
agent.connectors.template: copy this file to start a new connector.

A connector teaches the engine one system. The whole job is: read your source
read-only, and emit its relationships as edges. Everything else (the math, the
store, the analyses) follows from that. Fill in the two TODOs below.

    cp agent/connectors/template.py agent/connectors/my_system.py

Then validate it turns your integration into a pass/fail known quantity:

    from agent.connectors.validate import validate_connector
    from agent.connectors.my_system import MyConnector
    print(validate_connector(MyConnector(), my_source))

Run this file directly to see the worked example execute end-to-end:

    python -m agent.connectors.template
"""

from __future__ import annotations

from typing import Any

from . import BaseConnector, Capabilities

# the skeleton to copy

class MyConnector(BaseConnector):
    """Rename me. Read a source read-only and return ``(rex, meta)``."""

    # Advertise what you can supply. topology is always True; flip the others
    # on only if you actually emit them below. ``schemes`` are the URI schemes
    # the registry should route to you (e.g. ("mysystem",)).
    CAPABILITIES = Capabilities(
        weights=False,      # set True if you emit meta["weights"]
        modality=False,     # set True if you emit meta["modality"]
        faces=False,        # set True if you emit meta["faces"]
        schemes=(),         # e.g. ("mysystem",)
    )

    def read(self, source: Any) -> tuple[Any, dict[str, Any]]:
        # TODO 1: READ YOUR SOURCE (read-only)
        # Pull the *structure* only - the entities and how they relate. Never
        # read cell/row values; the engine persists structure, not data.
        # Produce, from your source:
        #   labels : list[str]                 one per entity (vertex)
        #   links  : list[(src_label, dst_label)]   one per relationship (edge)
        labels: list[str] = []          # e.g. table / node / class names
        links: list[tuple[str, str]] = []   # e.g. child->parent FK pairs

        # TODO 2: EMIT EDGES
        # Map each relationship to an edge between two vertices. B₁ is built for
        # you from these index pairs (source = -1, target = +1). Optionally:
        #   weights  : list[float]   per edge - cardinality/magnitude -> strain
        #   modality : list[dict]    per edge - {"nullable":..,"identifying":..}
        #   faces    : np.ndarray    dense B₂ (nE×nF) if some edges co-close
        idx = {name: i for i, name in enumerate(labels)}
        sources = [idx[a] for a, _ in links]
        targets = [idx[b] for _, b in links]

        return self.result(
            (sources, targets),
            vertex_labels=labels,
            edges=links,
            source="mysystem://<identifier>",
            # weights=[...],           # if CAPABILITIES.weights
            # modality=[...],          # if CAPABILITIES.modality
            # faces=b2_dense,          # if CAPABILITIES.faces
        )


# a worked example that runs immediately

class ExampleEdgesConnector(BaseConnector):
    """A trivial connector over an in-memory edge list, a stand-in for "any
    system that can enumerate its relationships." Weighted so strain is
    available. Runs with no external service."""

    CAPABILITIES = Capabilities(weights=True, schemes=("example",))

    def read(self, source: Any) -> tuple[Any, dict[str, Any]]:
        # ``source`` here is a list of (parent, child, cardinality) triples -
        # e.g. a hand-written schema. A real connector reads this from the
        # system instead of taking it as an argument.
        rows = source or [
            ("orders", "customers", 1000),
            ("order_items", "orders", 5000),
            ("order_items", "products", 5000),
            ("payments", "orders", 1200),
        ]
        labels: list[str] = []
        for parent, child, _ in rows:
            for name in (parent, child):
                if name not in labels:
                    labels.append(name)
        idx = {n: i for i, n in enumerate(labels)}
        links = [(a, b) for a, b, _ in rows]
        sources = [idx[a] for a, b, _ in rows]
        targets = [idx[b] for a, b, _ in rows]
        weights = [float(w) for _, _, w in rows]
        return self.result(
            (sources, targets),
            vertex_labels=labels,
            edges=links,
            weights=weights,
            source="example://mini-schema",
        )


def _demo() -> None:
    rex, meta = ExampleEdgesConnector().read(None)
    print("capabilities:", ExampleEdgesConnector().capabilities().summary())
    print("nV=%d nE=%d source=%s" % (meta["nV"], meta["nE"], meta["source"]))
    print("labels:", meta["vertex_labels"])
    print("edges :", meta["edges"])
    print("weights:", meta["weights"])
    from . import to_rexgraph
    g = to_rexgraph(rex, meta)
    print("built complex: nV=%d nE=%d betti=%s chain_valid=%s"
          % (g.nV, g.nE, tuple(g.betti), g.chain_valid))


if __name__ == "__main__":
    _demo()
