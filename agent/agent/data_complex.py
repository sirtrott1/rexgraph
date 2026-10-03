"""agent.data_complex: records as a relational complex (cluster/relate the DATA, not the schema).

A set of rows becomes a relational complex: each record is a vertex, and two records are joined by an
edge when they share a value in a `link_on` column (a co participation). The topology then reads the
data itself: connected components are clusters of related records, per record coherence is
structural centrality (a hub record vs a peripheral one), and a record that shares no link value is
an isolated outlier. This is the row level companion to schema_complex (which is the schema as a
complex): here the returned data is the complex.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent.metrics import coherence_kappa


def _row_id(row, i, id_col):
    return str(row.get(id_col, i)) if id_col else str(i)


def _link_groups(rows, link_cols):
    """Return observed link groups as ordered participant lists.

    One value shared by k records gives one k-ary relation. Participants use the
    lowest row index as a deterministic head. The returned head_is_canonical flag
    marks this orientation as a construction convention, not a causal assertion.
    """
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        for col in link_cols:
            v = r.get(col)
            if v is not None:
                groups[(col, v)].append(i)
    # sorted for determinism; the first entry becomes the head
    return {key: sorted(set(members)) for key, members in groups.items()}


def _support_components(n: int, groups) -> list[list[int]]:
    """Connected components of the participation support.

    This is a projection of the relations onto co participation, and it answers a
    different question from H0. Two records are in the same support component when a chain
    of shared link values connects them. The algebraic reading is beta_0 of the complex,
    which counts differently once relations are k-ary, and both are reported.
    """
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for members in groups.values():
        anchor = members[0]
        for m in members[1:]:
            ra, rb = find(anchor), find(m)
            if ra != rb:
                parent[ra] = rb
    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    return list(comp.values())


def rows_to_complex(rows: list[dict], *, link_on, id_col: str | None = None):
    """Build the record complex. Returns (rex_or_None, meta).

    Each observed link value becomes one relation over the records carrying it, at its
    actual arity. Records are the grade zero participants those relations are declared
    over, and every record is present whether or not it participates in one.

    A record sharing no link value is a grade zero participant with no relation. It is
    NOT given an arity one relation of its own: that would assert it was observed as a
    standalone fact, when all that happened is it linked to nothing. The distinction is
    visible in the boundary. An arity one column carries a single +1 and does not sum to
    zero, so manufacturing one for every unlinked record would break the zero sum law
    across the complex to represent an absence. Leaving the participant unattached keeps
    every column zero sum and still counts the record in H0.
    """
    link_cols = [link_on] if isinstance(link_on, str) else list(link_on)
    row_labels = [_row_id(r, i, id_col) for i, r in enumerate(rows)]
    groups = _link_groups(rows, link_cols)
    relations = [members for members in groups.values() if len(members) >= 2]

    rex = None
    if rows:
        from rexgraph.graph import RexGraph

        # every record is a declared participant, so the complex is faithful to the
        # record set rather than only to the linked part of it
        rex = RexGraph.from_cells([len(rows), [list(members) for members in relations]])
        rex._agent_meta = {"vertex_labels": row_labels, "source": "data"}

    unattached = sorted(set(range(len(rows))) - {v for m in relations for v in m})
    meta = {
        "vertex_labels": row_labels, "n_rows": len(rows), "link_on": link_cols,
        "row_labels": row_labels,
        "relations": [sorted(row_labels[v] for v in members) for members in relations],
        "relation_arities": [len(members) for members in relations],
        "unattached_participants": [row_labels[i] for i in unattached],
        # the orientation of a link group is not given by the record source; the head is
        # the lowest row index purely so the construction is reproducible
        "head_is_canonical": True,
    }
    return rex, meta


def analyze_rows(rows: list[dict], *, link_on, id_col: str | None = None,
                 top: int = 5) -> dict[str, Any]:
    """Return support components, homology and structural centrality.

    n_support_components counts components in the shared link projection.
    h0_dimension is beta_0 of the complex. Four records sharing one value form
    one support component and one rank 1 relation, with beta_0 equal to 3.
    n_clusters is a compatibility alias for n_support_components.
    """
    link_cols = [link_on] if isinstance(link_on, str) else list(link_on)
    row_labels = [_row_id(r, i, id_col) for i, r in enumerate(rows)]
    groups = _link_groups(rows, link_cols)
    clusters = _support_components(len(rows), groups)
    out: dict[str, Any] = {
        "n_rows": len(rows), "link_on": link_cols,
        "n_support_components": len(clusters),
        "n_clusters": len(clusters),          # alias, kept for existing callers
        "clusters": [sorted(row_labels[i] for i in c) for c in sorted(clusters, key=len, reverse=True)],
        "outliers": [row_labels[c[0]] for c in clusters if len(c) == 1],
        "relation_arities": sorted((len(m) for m in groups.values() if len(m) >= 2), reverse=True),
    }
    # structural centrality: per record coherence kappa (hub vs peripheral), most central first
    central = []
    if rows:
        rex, meta = rows_to_complex(rows, link_on=link_on, id_col=id_col)
        out["h0_dimension"] = int(rex.betti[0])
        out["unattached_participants"] = meta["unattached_participants"]
        try:
            kap = coherence_kappa(rex)
            labels = meta["vertex_labels"]
            central = sorted(({"row": labels[i], "kappa": round(float(kap[i]), 4)}
                             for i in range(min(len(kap), len(labels)))),
                            key=lambda d: -d["kappa"])[:top]
        except Exception:
            pass
    out["central"] = central
    return out
