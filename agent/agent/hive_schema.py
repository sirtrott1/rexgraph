"""agent.hive_schema - the hive's own structure as a versioned relational complex.

A hive is not a fixed set of workers; it is a living schema. Which workers and
models it holds, what capabilities they provide, which databases and stores it is
attached to, which datasets are loaded - all of that is structure, and it changes
in response to events: a new task deploys a worker, new data attaches a store, an
issue adds a guard or reroutes, another hive federates in.

HiveSchema captures that full structure as ONE relational complex (the same kind
of object as a database schema, a query, or the coordination complex) and versions
it in the RCDB on every change - only when the topology actually changes, tagged
with the cause. The hive's evolution becomes a tracked lineage: a starting schema
that mutates in response to queries, data, issues, and deployments, queryable by
topology like everything else.

This reuses the substrate that already exists: hive.type_complex's worker-type
ontology, rcdb.version_if_changed's change-only lineage, and the RCDB store.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import rcdb
from .ontology_complex import parse_rdf, ontology_to_rex


class HiveSchema:
    """The hive's self-structure, snapshotted as a versioned complex in the RCDB."""

    def __init__(self, hive, *, store: Optional[rcdb.RCStore] = None, lineage_id: str = "hive"):
        self.hive = hive
        self.store = store or rcdb.default_store()
        self.lineage_id = lineage_id
        # resources the hive is attached to but that are not bees: databases, stores, datasets.
        # name -> {"kind": str, "links": [(bee_name, relation)]}
        self.resources: Dict[str, Dict[str, Any]] = {}

    # -- the self-schema as triples -> a complex -------------------------------

    def triples(self) -> List[Tuple[str, str, str]]:
        """(subject, predicate, object) triples describing the hive's whole structure."""
        t: List[Tuple[str, str, str]] = []
        for b in self.hive.bees():
            wt = b.worker_type or f"role:{b.role}"
            parts = wt.split(":")
            for i in range(1, len(parts)):                      # worker-type subsumption chain
                t.append((":".join(parts[:i + 1]), "rdfs:subClassOf", ":".join(parts[:i])))
            t.append((b.name, "rdf:type", wt))                  # the worker is an instance of its type
            t.append((b.name, "provides", f"cap:{b.capability}"))
            t.append(("hive", "has_member", b.name))
        for name, r in self.resources.items():
            t.append((name, "rdf:type", f"resource:{r['kind']}"))
            for bee, rel in r.get("links", []):
                t.append((bee, rel, name))                      # e.g. bee 'reads' a database
        return t

    def complex(self):
        """Build the hive's structure complex. Returns (rex_or_None, meta)."""
        return ontology_to_rex(parse_rdf(self.triples()))

    # -- versioned lifecycle ---------------------------------------------------

    def snapshot(self, cause: str = "") -> dict:
        """Version the current structure in the RCDB - only if the topology changed since the
        last snapshot. `cause` records WHY it changed (a query, new data, an issue, a deploy)."""
        rex, meta = self.complex()
        if rex is None:
            return {"unchanged": True, "empty": True}
        meta = dict(meta, cause=cause)
        return rcdb.version_if_changed(self.store, self.lineage_id, rex, meta=meta,
                                       tags=["hive-schema"])

    def attach_resource(self, name: str, kind: str, *, links=None, cause: str = "") -> dict:
        """Register a database/store/dataset the hive is now attached to, then version the schema.
        `links` is a list of (bee_name, relation) e.g. [("db.search", "reads")]."""
        self.resources[name] = {"kind": kind, "links": list(links or [])}
        return self.snapshot(cause=cause or f"attached {kind}:{name}")

    def detach_resource(self, name: str, *, cause: str = "") -> dict:
        self.resources.pop(name, None)
        return self.snapshot(cause=cause or f"detached {name}")

    # -- history ---------------------------------------------------------------

    def lineage(self) -> List[dict]:
        return rcdb.lineage(self.store, self.lineage_id)

    def evolution(self) -> List[dict]:
        """The tracked life history: each version, why it happened, and its size/topology.

        Versions now live on one native version chain under ``self.lineage_id``
        (rcdb.lineage no longer mints a separate id per version), so each
        historical record is addressed by its own ``created`` (tx_from), not
        by ``v["id"]`` (that field is a display string, not a store id)."""
        out = []
        for v in self.lineage():
            rec = self.store.get_record(self.lineage_id, as_of=v["created"])
            sig = rec.signature if rec else {}
            out.append({"version": v["version"],
                        "cause": (rec.meta or {}).get("cause", "") if rec else "",
                        "n_nodes": sig.get("nV"), "betti": sig.get("betti")})
        return out
