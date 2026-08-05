"""agent.data_complex: records as a relational complex (cluster/relate the DATA, not the schema).

A set of rows becomes a relational complex: each record is a vertex, and two records are joined by an
edge when they share a value in a `link_on` column (a co-participation). The topology then reads the
data itself: connected components are clusters of related records, per-record coherence is
structural centrality (a hub record vs a peripheral one), and a record that shares no link value is
an isolated outlier. This is the row-level companion to schema_complex (which is the schema as a
complex): here the returned data is the complex.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _row_id(row, i, id_col):
    return str(row.get(id_col, i)) if id_col else str(i)


def _edges(rows, link_cols):
    """Undirected edges (as sorted index pairs) between rows sharing a link value; a star per group
    keeps it sparse while preserving connectivity."""
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        for col in link_cols:
            v = r.get(col)
            if v is not None:
                groups[(col, v)].append(i)
    edges = set()
    for members in groups.values():
        anchor = members[0]
        for m in members[1:]:
            edges.add((anchor, m) if anchor < m else (m, anchor))
    return edges


def rows_to_complex(rows: list[dict], *, link_on, id_col: str | None = None):
    """Build the record complex. Returns (rex_or_None, meta). Vertices are the records that share at
    least one link value (isolated records carry no edge); labels are the id_col (or the index)."""
    link_cols = [link_on] if isinstance(link_on, str) else list(link_on)
    row_labels = [_row_id(r, i, id_col) for i, r in enumerate(rows)]
    edges = _edges(rows, link_cols)
    verts = sorted({v for e in edges for v in e})
    remap = {v: i for i, v in enumerate(verts)}
    labels = [row_labels[v] for v in verts]
    rex = None
    if edges:
        from rexgraph.graph import RexGraph
        src = np.array([remap[a] for a, b in edges], np.int32)
        tgt = np.array([remap[b] for a, b in edges], np.int32)
        rex = RexGraph.from_graph(src, tgt)
        rex._agent_meta = {"vertex_labels": labels, "source": "data"}
    meta = {"vertex_labels": labels, "n_rows": len(rows), "link_on": link_cols,
            "row_labels": row_labels, "edges": sorted(edges)}
    return rex, meta


def _components(n: int, edges) -> list[list[int]]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    return list(comp.values())


def analyze_rows(rows: list[dict], *, link_on, id_col: str | None = None,
                 top: int = 5) -> dict[str, Any]:
    """Topological read of a record set: clusters (connected components of the shared-value graph),
    isolated outliers (records that share no link value), and structural centrality (coherence) -
    the hub records vs the peripheral ones. All exact-structural."""
    link_cols = [link_on] if isinstance(link_on, str) else list(link_on)
    row_labels = [_row_id(r, i, id_col) for i, r in enumerate(rows)]
    edges = _edges(rows, link_cols)
    clusters = _components(len(rows), edges)
    out: dict[str, Any] = {
        "n_rows": len(rows), "link_on": link_cols,
        "n_clusters": len(clusters),
        "clusters": [sorted(row_labels[i] for i in c) for c in sorted(clusters, key=len, reverse=True)],
        "outliers": [row_labels[c[0]] for c in clusters if len(c) == 1],
    }
    # structural centrality: per-record coherence kappa (hub vs peripheral), most-central first
    central = []
    if edges:
        rex, meta = rows_to_complex(rows, link_on=link_on, id_col=id_col)
        try:
            kap = np.asarray(rex.coherence, dtype=np.float64)
            labels = meta["vertex_labels"]
            central = sorted(({"row": labels[i], "kappa": round(float(kap[i]), 4)}
                             for i in range(min(len(kap), len(labels)))),
                            key=lambda d: -d["kappa"])[:top]
        except Exception:
            pass
    out["central"] = central
    return out
