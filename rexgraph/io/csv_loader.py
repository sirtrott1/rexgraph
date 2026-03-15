# rexgraph/io/csv_loader.py
"""
CSV edge list loader with automatic column role classification.

Detects the semantic role of each metadata column - type, polarity,
grouping, ordinal, evidence, description, reference - and maps them
to the appropriate rexgraph concepts (edge coloring, flow sign,
tooltip text, confidence scaling, etc.).

The classification uses a heuristic cascade:

1. Column name pattern matching (e.g. "effect" -> polarity)
2. Value-set statistics (cardinality, average token length,
   delimiter frequency, numeric fraction)
3. Value content scanning (positive/negative stems, ordinal
   terms, identifier patterns)

The result is a `ColumnProfile` per column with an assigned `role`
that downstream code can dispatch on without brittle "first categorical
wins" logic.

Usage::

    from rexgraph.io.csv_loader import load_edge_csv

    graph_data = load_edge_csv("pathway_curated.csv")
    # graph_data.sources, .targets, .vertices, .src_idx, .tgt_idx
    # graph_data.edge_attrs    - dict ready for analyze()
    # graph_data.w_E           - signed weight vector (polarity applied)
    # graph_data.negative_types
    # graph_data.profiles      - per-column ColumnProfile for inspection
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Column role taxonomy


class ColumnRole:
    """Semantic roles a metadata column can take."""
    TYPE      = "type"       # Edge coloring, legend, partition (e.g. binding, phosphorylation)
    POLARITY  = "polarity"   # Binary/ternary sentiment -> flow sign (e.g. stimulation/inhibition)
    GROUPING  = "grouping"   # Pathway, module, cluster membership (may overlap)
    ORDINAL   = "ordinal"    # Ordered categorical (e.g. high/medium/low -> opacity/width)
    NUMERIC   = "numeric"    # Continuous value (e.g. score, weight)
    EVIDENCE  = "evidence"   # Semicolon-delimited source lists (e.g. SIGNOR;KEGG)
    REFERENCE = "reference"  # Identifiers (PMIDs, accessions)
    DESCRIPTION = "description"  # Free text (mechanism detail, notes)
    UNKNOWN   = "unknown"    # Could not classify


# Column name patterns (checked first - strong signal)


_NAME_PATTERNS = [
    # (regex, role)
    (r'(?i)^(interaction[_\s]?type|edge[_\s]?type|rel(ation)?[_\s]?type|type|kind|class|category)$',
     ColumnRole.TYPE),
    (r'(?i)^(effect|polarity|direction|sign|sentiment|regulation|mode)$',
     ColumnRole.POLARITY),
    (r'(?i)^(pathway|module|group|cluster|compartment|subsystem|category)$',
     ColumnRole.GROUPING),
    (r'(?i)^(confidence|quality|evidence[_\s]?level|reliability|strength|tier|grade)$',
     ColumnRole.ORDINAL),
    (r'(?i)^(weight|score|probability|p[_\s]?value|fold[_\s]?change|log[_\s]?fc|magnitude)$',
     ColumnRole.NUMERIC),
    (r'(?i)^(database[_\s]?support|sources?|databases?|evidence|provenance|dbs?)$',
     ColumnRole.EVIDENCE),
    (r'(?i)^(pmid|pubmed|doi|accession|reference|ref|citation|pmid[_\s]?examples?)$',
     ColumnRole.REFERENCE),
    (r'(?i)^(mechanism|description|detail|notes?|comment|annotation|summary|text|info)$',
     ColumnRole.DESCRIPTION),
]


# Value-level heuristics


_POSITIVE_STEMS = [
    "stimulat", "activat", "promot", "enhanc", "upregulat", "up_regulat",
    "induc", "positiv", "increas", "augment", "facilitat",
]

_NEGATIVE_STEMS = [
    "inhibit", "block", "suppress", "degrad", "ubiquitin", "cleav",
    "deactiv", "repress", "negat", "down", "decreas", "reduc",
    "remov", "destroy", "kill", "delet", "attenu", "antagoni",
    "downregulat", "down_regulat", "dephosphorylat",
]

_ORDINAL_SETS = [
    {"high", "medium", "low"},
    {"high", "med", "low"},
    {"strong", "moderate", "weak"},
    {"a", "b", "c"},
    {"1", "2", "3"},
    {"yes", "no"},
    {"true", "false"},
    {"confirmed", "predicted"},
    {"validated", "putative"},
]

_IDENTIFIER_PATTERN = re.compile(r'^\d{5,}$')  # PMIDs, accessions


# Column profiling


@dataclass
class ColumnProfile:
    """Statistical and semantic profile of a metadata column."""
    name: str
    role: str = ColumnRole.UNKNOWN
    n_values: int = 0
    n_unique: int = 0
    avg_length: float = 0.0
    is_numeric: bool = False
    has_delimiter: bool = False        # semicolons, pipes
    unique_ratio: float = 0.0         # n_unique / n_values
    values: List[str] = field(default_factory=list)
    unique_values: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    # Populated for numeric columns
    numeric_min: float = 0.0
    numeric_max: float = 0.0
    numeric_mean: float = 0.0
    # Populated for polarity columns
    positive_values: List[str] = field(default_factory=list)
    negative_values: List[str] = field(default_factory=list)
    # Name-match confidence (1.0 = matched by name, 0.0 = heuristic only)
    name_matched: bool = False

    @property
    def is_categorical(self) -> bool:
        return not self.is_numeric and self.n_unique <= 30

    @property
    def is_freetext(self) -> bool:
        return self.unique_ratio > 0.8 and self.avg_length > 30

    @property
    def is_delimited_list(self) -> bool:
        return self.has_delimiter

    @property
    def is_binary(self) -> bool:
        return self.n_unique == 2

    def to_attribute_dict(self) -> dict:
        """Serialize to the dashboard attribute metadata format."""
        if self.is_numeric:
            return {
                "name": self.name, "kind": "numeric", "role": self.role,
                "min": round(self.numeric_min, 4),
                "max": round(self.numeric_max, 4),
                "mean": round(self.numeric_mean, 4),
                "count": self.n_values,
            }
        return {
            "name": self.name, "kind": "categorical", "role": self.role,
            "values": self.unique_values,
            "counts": self.counts,
            "count": self.n_values,
        }


def _profile_column(name: str, values: List[str]) -> ColumnProfile:
    """Build a statistical profile of a single metadata column."""
    non_empty = [v for v in values if v]
    p = ColumnProfile(name=name, values=values)
    p.n_values = len(non_empty)
    if not non_empty:
        return p

    unique = sorted(set(non_empty))
    p.n_unique = len(unique)
    p.unique_values = unique
    p.counts = {k: v for k, v in Counter(non_empty).most_common()}
    p.avg_length = sum(len(v) for v in non_empty) / len(non_empty)
    p.unique_ratio = p.n_unique / p.n_values if p.n_values > 0 else 0.0
    p.has_delimiter = any(";" in v or "|" in v for v in non_empty)

    # Numeric test
    try:
        floats = [float(v) for v in non_empty]
        p.is_numeric = True
        p.numeric_min = min(floats)
        p.numeric_max = max(floats)
        p.numeric_mean = sum(floats) / len(floats)
    except ValueError:
        p.is_numeric = False

    return p


def _classify_column(profile: ColumnProfile) -> str:
    """Assign a semantic role to a profiled column.

    Priority: name match > value heuristics > statistical fallback.
    """
    name = profile.name

    # 1. Name pattern matching
    for pattern, role in _NAME_PATTERNS:
        if re.match(pattern, name):
            profile.name_matched = True
            # Cross-validate: if name says "weight" but values are
            # categorical, still trust the name but flag it
            if role == ColumnRole.NUMERIC and not profile.is_numeric:
                return ColumnRole.ORDINAL  # "weight" with categorical values
            return role

    # 2. Value-based heuristics

    # Empty or single-value columns are uninformative
    if profile.n_values == 0 or profile.n_unique <= 1:
        return ColumnRole.UNKNOWN

    # Numeric columns
    if profile.is_numeric:
        return ColumnRole.NUMERIC

    # Free text (high uniqueness, long strings)
    if profile.is_freetext:
        return ColumnRole.DESCRIPTION

    # Delimited lists (semicolons/pipes) with moderate cardinality
    if profile.has_delimiter and profile.n_unique > 3:
        # Check if values look like identifiers (PMIDs)
        flat_tokens = []
        for v in profile.unique_values[:10]:
            flat_tokens.extend(v.replace("|", ";").split(";"))
        id_frac = sum(1 for t in flat_tokens if _IDENTIFIER_PATTERN.match(t.strip())) / max(len(flat_tokens), 1)
        if id_frac > 0.5:
            return ColumnRole.REFERENCE
        return ColumnRole.EVIDENCE

    # Binary columns with positive/negative sentiment -> polarity
    lower_unique = {v.lower() for v in profile.unique_values}
    if profile.n_unique <= 3:
        has_pos = any(any(stem in v for stem in _POSITIVE_STEMS) for v in lower_unique)
        has_neg = any(any(stem in v for stem in _NEGATIVE_STEMS) for v in lower_unique)
        if has_pos and has_neg:
            return ColumnRole.POLARITY
        # Ordinal check
        for oset in _ORDINAL_SETS:
            if lower_unique <= oset:
                return ColumnRole.ORDINAL

    # Check for ordinal pattern (small set with ordered semantics)
    if profile.n_unique <= 5:
        for oset in _ORDINAL_SETS:
            if lower_unique <= oset:
                return ColumnRole.ORDINAL

    # Moderate cardinality categorical (3-20 unique values) -> type
    if 2 <= profile.n_unique <= 20:
        return ColumnRole.TYPE

    # Higher cardinality categorical (20-50) -> grouping
    if 20 < profile.n_unique <= 50:
        return ColumnRole.GROUPING

    return ColumnRole.UNKNOWN


def _detect_polarity(profile: ColumnProfile) -> None:
    """Populate positive_values and negative_values on a polarity column."""
    for val in profile.unique_values:
        low = val.lower()
        if any(stem in low for stem in _NEGATIVE_STEMS):
            profile.negative_values.append(val)
        elif any(stem in low for stem in _POSITIVE_STEMS):
            profile.positive_values.append(val)
        else:
            # Neutral - treat as positive by default
            profile.positive_values.append(val)


def classify_columns(
    meta: Dict[str, List[str]],
) -> Dict[str, ColumnProfile]:
    """Profile and classify all metadata columns.

    Returns a dict of column_name -> ColumnProfile with assigned roles.
    """
    profiles = {}
    for name, values in meta.items():
        p = _profile_column(name, values)
        p.role = _classify_column(p)
        if p.role == ColumnRole.POLARITY:
            _detect_polarity(p)
        profiles[name] = p
    return profiles


# Column role accessors (convenience)


def find_by_role(
    profiles: Dict[str, ColumnProfile], role: str,
) -> List[ColumnProfile]:
    """Return all columns with the given role, name-matched first."""
    matches = [p for p in profiles.values() if p.role == role]
    matches.sort(key=lambda p: (not p.name_matched, p.name))
    return matches


def get_type_column(profiles: Dict[str, ColumnProfile]) -> Optional[ColumnProfile]:
    """Return the primary type column (for edge coloring)."""
    candidates = find_by_role(profiles, ColumnRole.TYPE)
    return candidates[0] if candidates else None


def get_polarity_column(profiles: Dict[str, ColumnProfile]) -> Optional[ColumnProfile]:
    """Return the polarity column (for flow sign)."""
    candidates = find_by_role(profiles, ColumnRole.POLARITY)
    return candidates[0] if candidates else None


def get_grouping_column(profiles: Dict[str, ColumnProfile]) -> Optional[ColumnProfile]:
    """Return the primary grouping column (pathway membership)."""
    candidates = find_by_role(profiles, ColumnRole.GROUPING)
    return candidates[0] if candidates else None


def get_ordinal_column(profiles: Dict[str, ColumnProfile]) -> Optional[ColumnProfile]:
    """Return the primary ordinal column (confidence, quality)."""
    candidates = find_by_role(profiles, ColumnRole.ORDINAL)
    return candidates[0] if candidates else None


# Weight construction from classified columns


_ORDINAL_MAPS = {
    "high": 1.0, "medium": 0.6, "med": 0.6, "low": 0.3,
    "strong": 1.0, "moderate": 0.6, "weak": 0.3,
    "confirmed": 1.0, "validated": 1.0, "predicted": 0.5, "putative": 0.5,
    "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
}


def build_weights(
    profiles: Dict[str, ColumnProfile],
    nE: int,
) -> Tuple[np.ndarray, List[str]]:
    """Construct signed edge weight vector from classified columns.

    Weight magnitude comes from the first numeric column (default 1.0).
    Sign comes from the polarity column (negative types get -|w|).
    Ordinal columns scale the magnitude (e.g. high=1.0, medium=0.6).

    Returns
    -------
    w_E : ndarray of float64
        Signed edge weights.
    negative_types : list of str
        Values from the polarity column classified as negative.
    """
    w = np.ones(nE, dtype=np.float64)

    # Numeric weight
    numeric_cols = find_by_role(profiles, ColumnRole.NUMERIC)
    if numeric_cols:
        p = numeric_cols[0]
        for j, v in enumerate(p.values):
            if j >= nE:
                break
            try:
                w[j] = float(v) if v else 1.0
            except ValueError:
                pass

    # Ordinal scaling (multiplicative)
    ordinal_col = get_ordinal_column(profiles)
    if ordinal_col:
        for j, v in enumerate(ordinal_col.values):
            if j >= nE:
                break
            scale = _ORDINAL_MAPS.get(v.lower().strip(), 1.0) if v else 1.0
            w[j] *= scale

    # Polarity sign
    negative_types = []
    polarity_col = get_polarity_column(profiles)
    if polarity_col:
        negative_types = polarity_col.negative_values[:]
        neg_set = set(negative_types)
        for j, v in enumerate(polarity_col.values):
            if j >= nE:
                break
            if v in neg_set:
                w[j] = -abs(w[j])

    return w, negative_types


# Edge attrs assembly for analyze()


def build_edge_attrs(
    profiles: Dict[str, ColumnProfile],
) -> Dict[str, list]:
    """Assemble the edge_attrs dict for analyze().

    Includes all columns, but ensures the type column is keyed as
    "type" (what the dashboard template reads) if one exists.
    """
    edge_attrs = {}

    # All columns go in as-is
    for name, p in profiles.items():
        edge_attrs[name] = p.values

    # Ensure "type" key maps to the type column
    type_col = get_type_column(profiles)
    if type_col and type_col.name != "type":
        edge_attrs["type"] = type_col.values

    return edge_attrs


# High-level loader


@dataclass
class GraphData:
    """Parsed and classified CSV data ready for RexGraph construction."""
    sources: List[str]
    targets: List[str]
    vertices: List[str]
    src_idx: np.ndarray
    tgt_idx: np.ndarray
    meta: Dict[str, List[str]]
    profiles: Dict[str, ColumnProfile]
    edge_attrs: Dict[str, list]
    w_E: np.ndarray
    negative_types: List[str]
    nV: int = 0
    nE: int = 0

    def __post_init__(self):
        self.nV = len(self.vertices)
        self.nE = len(self.sources)

    def summary(self) -> str:
        """Human-readable summary of column classification."""
        lines = [f"{self.nV} vertices, {self.nE} edges, "
                 f"{len(self.profiles)} metadata columns\n"]
        lines.append(f"  {'Column':<22} {'Role':<14} {'Unique':>6} "
                     f"{'Values':>6}  Notes")
        lines.append(f"  {'─'*22} {'─'*14} {'─'*6} {'─'*6}  {'─'*30}")
        for name, p in self.profiles.items():
            notes = []
            if p.name_matched:
                notes.append("name-matched")
            if p.role == ColumnRole.POLARITY and p.negative_values:
                notes.append(f"neg: {', '.join(p.negative_values)}")
            if p.role == ColumnRole.ORDINAL:
                notes.append(f"levels: {', '.join(p.unique_values)}")
            if p.has_delimiter:
                notes.append("delimited")
            lines.append(
                f"  {name:<22} {p.role:<14} {p.n_unique:>6} "
                f"{p.n_values:>6}  {'; '.join(notes)}")
        if self.negative_types:
            lines.append(f"\n  Negative flow types: {', '.join(self.negative_types)}")
        return "\n".join(lines)


def load_edge_csv(path: str) -> GraphData:
    """Load a CSV edge list with full column role classification.

    Parameters
    ----------
    path : str
        Path to the CSV file.  Expects at least two columns
        interpretable as source/target vertex names.

    Returns
    -------
    GraphData
        Fully classified data ready for RexGraph construction and
        visualization.  Access `.edge_attrs` for analyze(),
        `.w_E` for signed weights, `.negative_types` for the
        polarity-derived negative type list.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(8192)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        rows = list(csv.DictReader(f, dialect=dialect))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    cols = list(rows[0].keys())
    src_col = next(
        (c for c in cols if c.lower() in ("source", "src", "from", "head")),
        cols[0],
    )
    tgt_col = next(
        (c for c in cols if c.lower() in ("target", "tgt", "to", "tail", "dest")),
        cols[1],
    )
    meta_cols = [c for c in cols if c not in (src_col, tgt_col)]

    sources, targets = [], []
    meta = {c: [] for c in meta_cols}

    for row in rows:
        s = row[src_col].strip()
        t = row[tgt_col].strip()
        if not s or not t:
            continue
        sources.append(s)
        targets.append(t)
        for c in meta_cols:
            meta[c].append(row.get(c, "").strip())

    # Vertex index mapping
    vertex_set = set()
    for s, t in zip(sources, targets):
        vertex_set.add(s)
        vertex_set.add(t)
    vertices = sorted(vertex_set)
    vi = {v: i for i, v in enumerate(vertices)}
    src_idx = np.array([vi[s] for s in sources], dtype=np.int32)
    tgt_idx = np.array([vi[t] for t in targets], dtype=np.int32)

    # Column classification
    profiles = classify_columns(meta)
    edge_attrs = build_edge_attrs(profiles)
    w_E, negative_types = build_weights(profiles, len(sources))

    return GraphData(
        sources=sources, targets=targets,
        vertices=vertices, src_idx=src_idx, tgt_idx=tgt_idx,
        meta=meta, profiles=profiles, edge_attrs=edge_attrs,
        w_E=w_E, negative_types=negative_types,
    )
