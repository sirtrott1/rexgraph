"""
Base adapter defining the contract for domain-specific edge construction.

Every adapter takes raw data and produces an EdgeConstruction - the typed
edges, signs, and labels that feed into RexGraph.from_graph() and
typed_face_selection().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class EdgeSpan:
    """Maps an edge back to its source text position."""
    edge_idx: int
    source_label: str
    target_label: str
    char_start: int
    char_end: int
    sentence_idx: int


@dataclass
class SentenceSpan:
    """Character offsets for a sentence in the source text."""
    idx: int
    char_start: int
    char_end: int
    text: str


@dataclass
class EdgeConstruction:
    """Complete edge specification ready for RexGraph construction.

    All arrays are aligned by edge index: sources[k], targets[k],
    weights[k], signs[k], type_labels[k] all describe edge k.
    """

    sources: NDArray          # int32, source vertex per edge
    targets: NDArray          # int32, target vertex per edge
    weights: NDArray          # float64, magnitude per edge (>= 0)
    signs: NDArray            # float64, +1 or -1 per edge
    type_labels: NDArray      # int32, type index per edge
    vertex_labels: List[str]  # human-readable vertex names
    n_types: int              # number of distinct edge types
    type_names: List[str]     # human-readable name per type index

    # Text-position mapping (populated by TextAdapter and OCRAdapter)
    edge_spans: List[EdgeSpan] = field(default_factory=list)
    sentence_spans: List[SentenceSpan] = field(default_factory=list)
    source_text: str = ""

    @property
    def nV(self) -> int:
        return len(self.vertex_labels)

    @property
    def nE(self) -> int:
        return len(self.sources)

    @property
    def w_E(self) -> NDArray:
        """Signed edge weights (magnitude * sign)."""
        return self.weights * self.signs

    def summary(self) -> str:
        lines = [
            f"{self.nV} vertices, {self.nE} edges, {self.n_types} types",
        ]
        for t in range(self.n_types):
            mask = self.type_labels == t
            n = int(mask.sum())
            n_neg = int((self.signs[mask] < 0).sum())
            lines.append(
                f"  {self.type_names[t]}: {n} edges"
                f" ({n_neg} negative)" if n_neg else
                f"  {self.type_names[t]}: {n} edges"
            )
        return "\n".join(lines)


class DomainAdapter:
    """Base class for domain-specific edge construction.

    Subclasses implement build() to turn raw data into edges.
    Optionally override interpret() to add domain-specific meaning
    to analysis results.
    """

    name: str = "base"

    def build(self, data, **kwargs) -> EdgeConstruction:
        """Construct typed edges from domain data.

        Parameters
        ----------
        data : any
            Domain-specific input (array, DataFrame, file path, etc.)
        **kwargs
            Adapter-specific options.

        Returns
        -------
        EdgeConstruction
        """
        raise NotImplementedError

    def interpret(self, results: dict) -> dict:
        """Add domain-specific interpretation to analysis results.

        Default: pass through unchanged. Override in subclasses to add
        domain-meaningful labels, clinical mappings, etc.
        """
        return results
