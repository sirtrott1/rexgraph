"""
agent.lineage_adapters - represent the platform's own work as relational
complexes so it becomes queryable structure in the RCDB.

A pipeline run is a DAG of stages (stages = vertices, data-flow / ordering =
edges, fan-in/out = co-participation). A conversation is a chain/branch of
turns. Once each is a complex, the same three primitives apply as to schemas:
signature (identity), cross-complex coherence (similarity), and drift (change).
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def sequence_to_rex(labels: List[str], edges: Optional[List[Tuple]] = None,
                    kind: str = "sequence"):
    """Build a complex from a labeled node list.

    ``edges`` may be (from, to) pairs given by label or index; if omitted the
    nodes are chained consecutively (a linear run/conversation). Returns
    ``(rex_or_None, meta)``.
    """
    import numpy as np
    names = [str(x) for x in labels]
    idx = {n: i for i, n in enumerate(names)}
    if edges is None:
        pairs = [(i, i + 1) for i in range(len(names) - 1)]
    else:
        pairs = []
        for a, b in edges:
            ia = idx[a] if a in idx else int(a)
            ib = idx[b] if b in idx else int(b)
            if 0 <= ia < len(names) and 0 <= ib < len(names) and ia != ib:
                pairs.append((ia, ib))
    meta = {"vertex_labels": names, "input_type": kind,
            "source": kind, "n_nodes": len(names), "n_edges": len(pairs)}
    if not pairs:
        return None, meta
    from rexgraph.graph import RexGraph
    rex = RexGraph(sources=np.array([p[0] for p in pairs], dtype=np.int32),
                   targets=np.array([p[1] for p in pairs], dtype=np.int32))
    rex._agent_meta = meta
    return rex, meta


def run_to_rex(stages: List[str], edges: Optional[List[Tuple]] = None):
    """A pipeline run -> a complex over its stages (linear unless the caller
    supplies branch/dependency edges)."""
    return sequence_to_rex(stages, edges=edges, kind="pipeline-run")


def conversation_to_rex(turns: List[str], edges: Optional[List[Tuple]] = None):
    """A conversation -> a complex over its turns (a chain, or a branch tree if
    edges are supplied)."""
    return sequence_to_rex(turns, edges=edges, kind="conversation")
