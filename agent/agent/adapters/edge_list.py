"""
Edge list adapter: pre-structured edge data -> typed relational complex.

Thin wrapper around rexgraph.io.csv_loader and json_loader. Handles
CSV and JSON files that already contain explicit edges with optional
type and sign columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import DomainAdapter, EdgeConstruction


class EdgeListAdapter(DomainAdapter):
    """Construct a typed relational complex from an explicit edge list.

    Wraps the existing rexgraph IO layer which already handles column
    classification (source/target detection, type/polarity/weight
    columns, ordinal scaling).
    """

    name = "edge_list"

    def build(
        self,
        path: str,
        roles: dict | None = None,
        face_selection: str = "auto",
        **kwargs,
    ) -> EdgeConstruction:
        """Build typed edges from a CSV or JSON edge list.

        Parameters
        ----------
        path : str
            File path to CSV or JSON.
        roles : dict, optional
            Manual column role overrides. Passed to load_edge_csv.
            E.g., {"effect": "polarity", "score": "numeric"}
        face_selection : str
            'typed': use detected type column for face selection.
            'promote': ignore types, promote all cycles.
            'none': no face selection.

        Returns
        -------
        EdgeConstruction
        """
        from rexgraph.io.csv_loader import load_edge_csv
        from rexgraph.io.json_loader import load_json

        p = Path(path)

        if p.suffix.lower() in (".csv", ".tsv", ".txt"):
            gd = load_edge_csv(str(p), roles=roles)
            sources = gd.src_idx
            targets = gd.tgt_idx
            vertex_labels = gd.vertices
            weights = np.abs(gd.w_E)
            signs = np.sign(gd.w_E)
            signs[signs == 0] = 1.0

            # Extract type labels from the classified columns
            type_labels, type_names = self._extract_types(gd)

        elif p.suffix.lower() == ".json":
            # Use rexgraph's auto-detecting JSON loader
            rex = load_json(str(p))
            sources = rex.sources
            targets = rex.targets
            n_edges = rex.nE
            vertex_labels = [str(i) for i in range(rex.nV)]
            weights = rex.w_E if rex.w_E is not None else np.ones(n_edges)
            signs = np.ones(n_edges, dtype=np.float64)
            type_labels = np.zeros(n_edges, dtype=np.int32)
            type_names = ["edge"]

        else:
            raise ValueError(f"Unsupported file format: {p.suffix}")

        return EdgeConstruction(
            sources=sources,
            targets=targets,
            weights=weights,
            signs=signs,
            type_labels=type_labels,
            vertex_labels=vertex_labels,
            n_types=len(type_names),
            type_names=type_names,
        )

    def _extract_types(self, gd) -> tuple:
        """Extract edge types from the classified GraphData."""
        from rexgraph.io.csv_loader import get_type_column

        type_col = get_type_column(gd.profiles)
        if type_col is not None and type_col.values:
            unique_types = sorted(set(type_col.values))
            type_map = {t: i for i, t in enumerate(unique_types)}
            labels = np.array(
                [type_map.get(v, 0) for v in type_col.values],
                dtype=np.int32,
            )
            return labels, unique_types
        else:
            return (
                np.zeros(gd.nE, dtype=np.int32),
                ["edge"],
            )
