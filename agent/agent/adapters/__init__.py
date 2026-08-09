"""
Base adapter defining the contract for domain-specific edge construction.

Every adapter takes raw data and produces an EdgeConstruction: the typed
edges, signs, and labels that feed into RexGraph.from_graph() and
typed_face_selection().
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    vertex_labels: list[str]  # human-readable vertex names
    n_types: int              # number of distinct edge types
    type_names: list[str]     # human-readable name per type index

    #: relations of arity above two, as vertex lists, one per relation.
    #:
    #: `sources`/`targets` hold two vertices per relation and cannot express a wider
    #: one. Where a source genuinely names a k-way relation - a delocalised ring, a
    #: coordination centre, a reaction with several reagents, a group over its members -
    #: splitting it into pairs invents edges and dissolves the relation's identity, which
    #: is the same loss clique expansion makes. Adapters that have such a relation put it
    #: here and it survives into the complex as ONE cell with a k-ary boundary column.
    #:
    #: Empty for every adapter that does not, so nothing changes for them.
    branching: list[list[int]] = field(default_factory=list)

    #: one position per vertex, when the source carries one.
    #:
    #: Geometry emerges from an EMBEDDING, not from the complex: the complex fixes which
    #: cells exist and how they meet, and where they sit is a further fact a file can
    #: carry. A coordinate file carries it exactly (an SDF writes four decimal places, so
    #: every coordinate is a Fraction over 10^4), so the lengths and angles taken against
    #: it stay on the exact tower rather than being reconstructed from a layout.
    #:
    #: Empty for a source that has no coordinates, where the character embedding is the
    #: only position there is and structural equivalence is what the picture shows.
    embedding: list = field(default_factory=list)

    #: per-cell attributes, `{grade: {cell_index: {key: value}}}`.
    #:
    #: The same shape as `RexGraph._cell_metadata`, so `build_rex_from_edges` hands it
    #: straight to `attach_metadata` and it serialises columnar through `rex_state`,
    #: sparse and typed, indexed by cell index into the boundary tensors.
    #:
    #: Every reader parses more than it can say in a label. A PDB line carries a chain and
    #: a residue sequence number; a GFF line carries a whole `key=value;key=value` column;
    #: an SDF atom carries an element and a formal charge. Flattening those into a label
    #: string means the only way back is to parse the name, and the name is not a schema.
    #: An attribute put here can be queried, filtered and drawn.
    attributes: dict = field(default_factory=dict)

    # Text-position mapping (populated by TextAdapter and OCRAdapter)
    edge_spans: list[EdgeSpan] = field(default_factory=list)
    sentence_spans: list[SentenceSpan] = field(default_factory=list)
    source_text: str = ""

    #: vertex label -> the other identifiers that name the same thing.
    #:
    #: One entity is named differently by every file that mentions it. A GTF exon
    #: row carries `gene_id`, `gene_name` and `transcript_id` at once; a GAF row
    #: carries an accession, a symbol and a synonym list; an OBO term carries its id,
    #: its name and its `alt_id`s. A reader that keeps only the identifier it chose
    #: to label with throws away every key by which its file could be joined to
    #: another, which is why the identifiers have to travel with the vertex.
    #:
    #: Generic on purpose: this is "an entity is known by several names", not
    #: anything about biology.
    vertex_aliases: dict[str, list[str]] = field(default_factory=dict)

    #: where this construction came from, for provenance after a join
    origin: str = ""

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
