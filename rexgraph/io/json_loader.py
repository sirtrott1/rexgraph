# rexgraph/io/json_loader.py
"""
JSON loaders for relational complex data.

Supports common graph interchange formats used in bioinformatics,
clinical research, and network science:

- Edge-list JSON (API export format)
- RexGraph native JSON (from to_json())
- Cytoscape.js JSON (standard bioinformatics interchange)
- NetworkX node-link JSON (Python network science standard)
- Adjacency matrix JSON (correlation matrices, gene expression)

Each loader returns either a RexGraph directly or a GraphData
compatible with the csv_loader pipeline for column classification.

Usage::

    from rexgraph.io.json_loader import load_json

    rex = load_json("graph.json")  # auto-detects format
    rex = load_json("cytoscape_export.json", format="cytoscape")

Interop: all loaders produce RexGraph via from_graph() or from_dict(),
using the same contract as bundle.py and arrow_bridge.py.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray


def load_json(
    path: Union[str, os.PathLike],
    *,
    format: Optional[str] = None,
    threshold: float = 0.0,
    directed: bool = False,
) -> Any:
    """Load a JSON file and return a RexGraph.

    Auto-detects the format from the JSON structure if format is None.

    Parameters
    ----------
    path : str or path-like
    format : str, optional
        One of: "edge_list", "rexgraph", "cytoscape", "networkx",
        "adjacency". Auto-detected if None.
    threshold : float
        For adjacency matrices, edges with |weight| <= threshold
        are excluded. Default 0.0 (keep all nonzero).
    directed : bool
        Whether to treat edges as directed.

    Returns
    -------
    RexGraph
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if format is None:
        format = _detect_format(data)

    if format == "rexgraph":
        return _load_rexgraph_json(data)
    elif format == "edge_list":
        return _load_edge_list_json(data, directed=directed)
    elif format == "cytoscape":
        return _load_cytoscape_json(data, directed=directed)
    elif format == "networkx":
        return _load_networkx_json(data, directed=directed)
    elif format == "adjacency":
        return _load_adjacency_json(data, threshold=threshold, directed=directed)
    else:
        raise ValueError(f"Unknown JSON format: {format!r}")


def _detect_format(data: Any) -> str:
    """Detect JSON format from structure."""
    if isinstance(data, dict):
        # RexGraph native: has boundary_ptr
        if "boundary_ptr" in data:
            return "rexgraph"
        # Cytoscape: has elements.nodes/elements.edges
        if "elements" in data:
            return "cytoscape"
        # NetworkX node-link: has nodes + links
        if "nodes" in data and "links" in data:
            return "networkx"
        # Edge list: has edges key
        if "edges" in data:
            return "edge_list"
        # Adjacency matrix: has matrix key or is a nested list
        if "matrix" in data or "adjacency" in data:
            return "adjacency"
    elif isinstance(data, list):
        # List of lists = adjacency matrix
        if data and isinstance(data[0], list):
            return "adjacency"
        # List of dicts = edge list
        if data and isinstance(data[0], dict):
            return "edge_list"
    raise ValueError(
        "Cannot detect JSON format. Provide format= explicitly."
    )


# RexGraph native JSON (from to_json())

def _load_rexgraph_json(data: dict) -> Any:
    """Load from RexGraph.to_json() output."""
    from ..graph import RexGraph

    kw = {
        "boundary_ptr": np.array(data["boundary_ptr"], dtype=np.int32),
        "boundary_idx": np.array(data["boundary_idx"], dtype=np.int32),
        "directed": data.get("directed", False),
    }
    if "B2_col_ptr" in data:
        kw["B2_col_ptr"] = np.array(data["B2_col_ptr"], dtype=np.int32)
        kw["B2_row_idx"] = np.array(data["B2_row_idx"], dtype=np.int32)
        # to_json doesn't store B2_vals; reconstruct as ones
        nz = len(kw["B2_row_idx"])
        kw["B2_vals"] = np.ones(nz, dtype=np.float64)
    if "w_E" in data:
        kw["w_E"] = np.array(data["w_E"], dtype=np.float64)
    return RexGraph(**kw)


def load_rexgraph_json(path: Union[str, os.PathLike]) -> Any:
    """Load a RexGraph from its native JSON format."""
    return load_json(path, format="rexgraph")


# Edge-list JSON

def _load_edge_list_json(data: Any, directed: bool = False) -> Any:
    """Load from edge-list JSON.

    Accepts:
        {"edges": [{"source": "A", "target": "B", ...}, ...]}
    or:
        [{"source": "A", "target": "B", ...}, ...]

    String vertex names are mapped to integer indices.
    Metadata columns are classified using the csv_loader pipeline.
    """
    from ..graph import RexGraph
    from .csv_loader import classify_columns, build_weights

    if isinstance(data, dict):
        edges = data.get("edges", data.get("links", []))
    elif isinstance(data, list):
        edges = data
    else:
        raise ValueError("Edge list JSON must be a dict with 'edges' key or a list of dicts.")

    if not edges:
        raise ValueError("Empty edge list.")

    # Detect source/target keys
    first = edges[0]
    src_key = _find_key(first, ["source", "src", "from", "head"])
    tgt_key = _find_key(first, ["target", "tgt", "to", "tail", "dest"])
    meta_keys = [k for k in first.keys() if k not in (src_key, tgt_key)]

    sources, targets = [], []
    meta = {k: [] for k in meta_keys}
    signs = []

    for edge in edges:
        s = str(edge[src_key])
        t = str(edge[tgt_key])
        sources.append(s)
        targets.append(t)
        for k in meta_keys:
            meta[k].append(str(edge.get(k, "")))

    # Vertex index mapping
    vertex_set = set()
    for s, t in zip(sources, targets):
        vertex_set.add(s)
        vertex_set.add(t)
    vertices = sorted(vertex_set)
    vi = {v: i for i, v in enumerate(vertices)}
    src_idx = np.array([vi[s] for s in sources], dtype=np.int32)
    tgt_idx = np.array([vi[t] for t in targets], dtype=np.int32)

    # Classify metadata and build weights
    nE = len(sources)
    w_E = None
    edge_signs = None

    if meta_keys:
        profiles = classify_columns(meta)
        w_E_arr, negative_types = build_weights(profiles, nE)
        if not np.allclose(w_E_arr, 1.0):
            w_E = np.abs(w_E_arr)
        # Extract signs from polarity
        if negative_types:
            edge_signs = np.sign(w_E_arr).astype(np.float64)

    return RexGraph(
        sources=src_idx, targets=tgt_idx,
        directed=directed, w_E=w_E, signs=edge_signs,
    )


def load_edge_list_json(path: Union[str, os.PathLike], **kwargs) -> Any:
    """Load a RexGraph from edge-list JSON."""
    return load_json(path, format="edge_list", **kwargs)


# Cytoscape.js JSON

def _load_cytoscape_json(data: dict, directed: bool = False) -> Any:
    """Load from Cytoscape.js JSON format.

    Expects:
        {"elements": {"nodes": [...], "edges": [...]}}
    or:
        {"elements": [{"group": "nodes", ...}, {"group": "edges", ...}]}
    """
    from ..graph import RexGraph

    elements = data.get("elements", {})

    if isinstance(elements, dict):
        nodes = elements.get("nodes", [])
        edges = elements.get("edges", [])
    elif isinstance(elements, list):
        nodes = [e for e in elements if e.get("group") == "nodes"]
        edges = [e for e in elements if e.get("group") == "edges"]
    else:
        raise ValueError("Unexpected Cytoscape elements format.")

    # Build vertex index from node IDs
    node_ids = []
    for n in nodes:
        ndata = n.get("data", n)
        node_ids.append(str(ndata.get("id", ndata.get("name", ""))))
    vi = {nid: i for i, nid in enumerate(node_ids)}

    # Extract edges
    src_list, tgt_list = [], []
    w_list = []
    sign_list = []

    for e in edges:
        edata = e.get("data", e)
        s = str(edata.get("source", ""))
        t = str(edata.get("target", ""))
        if s not in vi or t not in vi:
            continue
        src_list.append(vi[s])
        tgt_list.append(vi[t])
        # Weight from 'weight' or 'score' attribute
        w = edata.get("weight", edata.get("score", 1.0))
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 1.0
        w_list.append(abs(w))
        sign_list.append(-1.0 if w < 0 else 1.0)

    nE = len(src_list)
    src_arr = np.array(src_list, dtype=np.int32)
    tgt_arr = np.array(tgt_list, dtype=np.int32)
    w_arr = np.array(w_list, dtype=np.float64)
    sign_arr = np.array(sign_list, dtype=np.float64)

    w_E = w_arr if not np.allclose(w_arr, 1.0) else None
    signs = sign_arr if np.any(sign_arr < 0) else None

    return RexGraph(
        sources=src_arr, targets=tgt_arr,
        directed=directed, w_E=w_E, signs=signs,
    )


def load_cytoscape_json(path: Union[str, os.PathLike], **kwargs) -> Any:
    """Load a RexGraph from Cytoscape.js JSON."""
    return load_json(path, format="cytoscape", **kwargs)


# NetworkX node-link JSON

def _load_networkx_json(data: dict, directed: bool = False) -> Any:
    """Load from NetworkX node-link JSON format.

    Expects:
        {"nodes": [{"id": ...}, ...], "links": [{"source": ..., "target": ...}, ...]}
    """
    from ..graph import RexGraph

    nodes = data.get("nodes", [])
    links = data.get("links", data.get("edges", []))

    # Build vertex index
    node_ids = [n.get("id", i) for i, n in enumerate(nodes)]
    vi = {nid: i for i, nid in enumerate(node_ids)}

    src_list, tgt_list, w_list, sign_list = [], [], [], []

    for link in links:
        s = link.get("source")
        t = link.get("target")
        if s not in vi or t not in vi:
            continue
        src_list.append(vi[s])
        tgt_list.append(vi[t])
        w = link.get("weight", link.get("value", 1.0))
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 1.0
        w_list.append(abs(w))
        sign_list.append(-1.0 if w < 0 else 1.0)

    nE = len(src_list)
    src_arr = np.array(src_list, dtype=np.int32)
    tgt_arr = np.array(tgt_list, dtype=np.int32)
    w_arr = np.array(w_list, dtype=np.float64)
    sign_arr = np.array(sign_list, dtype=np.float64)

    w_E = w_arr if not np.allclose(w_arr, 1.0) else None
    signs = sign_arr if np.any(sign_arr < 0) else None

    return RexGraph(
        sources=src_arr, targets=tgt_arr,
        directed=directed, w_E=w_E, signs=signs,
    )


def load_networkx_json(path: Union[str, os.PathLike], **kwargs) -> Any:
    """Load a RexGraph from NetworkX node-link JSON."""
    return load_json(path, format="networkx", **kwargs)


# Adjacency matrix JSON

def _load_adjacency_json(
    data: Any, threshold: float = 0.0, directed: bool = False,
) -> Any:
    """Load from adjacency matrix JSON.

    Accepts:
        {"matrix": [[...], ...]}
    or:
        {"adjacency": [[...], ...], "labels": ["gene1", ...]}
    or:
        [[...], ...]  (bare list of lists)
    """
    from ..graph import RexGraph

    if isinstance(data, list):
        matrix = np.array(data, dtype=np.float64)
        labels = None
    elif isinstance(data, dict):
        matrix = np.array(
            data.get("matrix", data.get("adjacency", [])),
            dtype=np.float64,
        )
        labels = data.get("labels", data.get("node_names", None))
    else:
        raise ValueError("Adjacency JSON must be a matrix or dict with 'matrix' key.")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got {matrix.shape}")

    return RexGraph.from_adjacency(matrix, directed=directed)


def load_adjacency_json(path: Union[str, os.PathLike], **kwargs) -> Any:
    """Load a RexGraph from adjacency matrix JSON."""
    return load_json(path, format="adjacency", **kwargs)


# Matrix CSV loading (correlation matrices, gene expression)

def load_matrix_csv(
    path: Union[str, os.PathLike],
    *,
    threshold: float = 0.0,
    absolute: bool = True,
    directed: bool = False,
) -> Any:
    """Load a square matrix CSV as a RexGraph.

    For correlation matrices, gene expression similarity matrices,
    and other symmetric matrices where entries represent edge weights.

    Parameters
    ----------
    path : str or path-like
    threshold : float
        Edges with |weight| <= threshold are excluded.
    absolute : bool
        If True, use |weight| as edge weight and sign(weight) as edge sign.
        If False, use raw weight (negative weights become zero-weight edges).
    directed : bool
        Whether the matrix is asymmetric (directed edges).

    Returns
    -------
    RexGraph
    """
    import csv

    from ..graph import RexGraph

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    # Detect if first row/column are labels
    has_header = not _is_numeric(rows[0][0]) if rows[0] else False
    has_row_labels = not _is_numeric(rows[1][0]) if len(rows) > 1 and rows[1] else False

    if has_header:
        col_labels = rows[0][1:] if has_row_labels else rows[0]
        data_rows = rows[1:]
    else:
        col_labels = None
        data_rows = rows

    if has_row_labels:
        matrix = np.array(
            [[float(x) for x in row[1:]] for row in data_rows],
            dtype=np.float64,
        )
    else:
        matrix = np.array(
            [[float(x) for x in row] for row in data_rows],
            dtype=np.float64,
        )

    n = matrix.shape[0]
    if matrix.shape[1] != n:
        raise ValueError(f"Matrix must be square, got {matrix.shape}")

    # Threshold
    if threshold > 0:
        mask = np.abs(matrix) <= threshold
        matrix[mask] = 0.0

    # Extract edges
    if directed:
        rows_idx, cols_idx = np.nonzero(matrix)
    else:
        rows_idx, cols_idx = np.nonzero(np.triu(matrix, k=1))

    weights = matrix[rows_idx, cols_idx]
    nE = len(rows_idx)

    if nE == 0:
        return RexGraph(
            sources=np.zeros(0, dtype=np.int32),
            targets=np.zeros(0, dtype=np.int32),
        )

    src = rows_idx.astype(np.int32)
    tgt = cols_idx.astype(np.int32)

    if absolute:
        w_E = np.abs(weights)
        signs = np.sign(weights).astype(np.float64)
        w_E_arg = w_E if not np.allclose(w_E, 1.0) else None
        signs_arg = signs if np.any(signs < 0) else None
    else:
        w_E_arg = weights if not np.allclose(weights, 1.0) else None
        signs_arg = None

    return RexGraph(
        sources=src, targets=tgt,
        directed=directed, w_E=w_E_arg, signs=signs_arg,
    )


# Helpers

def _find_key(d: dict, candidates: list) -> str:
    """Find the first matching key from candidates."""
    for c in candidates:
        if c in d:
            return c
        for k in d:
            if k.lower() == c.lower():
                return k
    raise ValueError(
        f"Could not find source/target key. Available: {list(d.keys())}"
    )


def _is_numeric(s: str) -> bool:
    """Check if a string is numeric."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
