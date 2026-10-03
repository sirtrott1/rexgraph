"""
Base adapter defining the contract for domain specific edge construction.

Every adapter takes raw data and produces an EdgeConstruction: the typed
edges, signs, and labels that feed into RexGraph.from_graph() and
typed_face_selection().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
    vertex_labels: list[str]  # human readable vertex names
    n_types: int              # number of distinct edge types
    type_names: list[str]     # human readable name per type index

    #: relations of arity above two, as vertex lists, one per relation.
    #:
    #: `sources`/`targets` hold two vertices per relation and cannot express a wider
    #: one. Where a source names a k-way relation: a delocalised ring, a
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

    #: per cell attributes, `{grade: {cell_index: {key: value}}}`.
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

    # Text position mapping (populated by TextAdapter and OCRAdapter)
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

    #: Optional primary identities in the same order as the constructed relations.
    relation_ids: NDArray | None = None

    #: Declared source files and interpretation used by a registered reader.
    source_manifest: dict = field(default_factory=dict)

    # A legacy adapter may supply pair only arrays. Wider relations carry their
    # own declarations here, or the original arrays may cover all nE relations.
    branching_weights: object = None
    branching_signs: object = None
    branching_type_labels: object = None
    head_slots: object = None
    shares: object = None
    vertex_ids: object = None

    def to_relations(self):
        """Normalize source declarations into the public core construction input.

        Missing wider relation weights stay absent. The core's explicit
        unit for absent rule supplies the mathematical view without changing
        the declaration. Weights, orientation, shares and identity stay separate.
        """
        from rexgraph import Absent, NumberRule, Relations, VertexTable
        from rexgraph.relations import _integers

        sources = _integers(self.sources, name="source vertices")
        targets = _integers(self.targets, name="target vertices")
        if len(sources) != len(targets):
            raise ValueError("source and target vertices must be aligned")
        supports = [list(pair) for pair in zip(sources, targets, strict=True)]
        supports.extend(self.branching or ())
        n_pairs, n_branch = len(sources), len(self.branching or ())

        def aligned(values, wider, name, missing):
            values = list(values) if values is not None else []
            if len(values) == self.nE:
                if wider is not None:
                    raise ValueError(f"{name} has conflicting full and branching declarations")
                return values
            if len(values) != n_pairs:
                raise ValueError(f"{name} must cover the pair or full relation basis")
            extra = [missing]*n_branch if wider is None else list(wider)
            if len(extra) != n_branch:
                raise ValueError(f"branching {name} must cover every wider relation")
            return values + extra

        weights = aligned(self.weights, self.branching_weights, "weights", Absent)
        signs = aligned(self.signs, self.branching_signs, "signs", 1)
        if any(isinstance(s, (bool, np.bool_)) or s not in (-1, 1) for s in signs):
            raise ValueError("relation signs must be -1 or +1")
        types = aligned(self.type_labels, self.branching_type_labels, "types", Absent)
        if self.n_types != len(self.type_names):
            raise ValueError("type names must match the declared type domain")
        names = []
        for value in types:
            if value is Absent:
                names.append(Absent)
            elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and 0 <= value < self.n_types:
                names.append(self.type_names[int(value)])
            else:
                raise ValueError("relation type index is outside its declared domain")
        ids = tuple(Absent for _ in range(self.nV)) if self.vertex_ids is None else tuple(self.vertex_ids)
        aliases = tuple(tuple(self.vertex_aliases.get(label, ())) for label in self.vertex_labels)
        vertices = VertexTable(ids, tuple(self.vertex_labels), aliases)
        provenance = {"type_names": list(self.type_names), "n_types": self.n_types}
        if self.origin:
            provenance["origin"] = self.origin
        if self.source_manifest:
            provenance["source_manifest"] = self.source_manifest
        if self.source_text:
            provenance["source_text"] = self.source_text
        if self.edge_spans:
            provenance["edge_spans"] = [asdict(span) for span in self.edge_spans]
        if self.sentence_spans:
            provenance["sentence_spans"] = [asdict(span) for span in self.sentence_spans]
        return Relations.from_supports(
            supports, vertices=vertices, weights=weights, signs=np.asarray(signs, np.int8),
            heads=self.head_slots, shares=self.shares, relation_ids=self.relation_ids,
            relation_types=names, number_rule=NumberRule.BINARY_EXACT,
            attributes=self.attributes, provenance=provenance,
            embedding=self.embedding if self.embedding is not None and len(self.embedding) else None,
        )

    @property
    def nV(self) -> int:
        return len(self.vertex_labels)

    @property
    def nE(self) -> int:
        """Relations, at ANY arity: the 2 ary ones plus the branching ones.

        `len(self.sources)` alone counts only what (sources, targets) can hold, so a
        construction carrying its relations in `branching` reported zero and every
        caller reading nE as "is there anything here" concluded the text was empty.
        """
        return len(self.sources) + len(self.branching or ())

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
    """Base class for domain specific edge construction.

    Subclasses implement build() to turn raw data into edges.
    Optionally override interpret() to add domain specific meaning
    to analysis results.
    """

    name: str = "base"

    def build(self, data, **kwargs) -> EdgeConstruction:
        """Construct typed edges from domain data.

        Parameters

        data : any
            Domain-specific input (array, DataFrame, file path, etc.)
        **kwargs
            Adapter-specific options.

        Returns

        EdgeConstruction
        """
        raise NotImplementedError

    def interpret(self, results: dict) -> dict:
        """Add domain specific interpretation to analysis results.

        Default: pass through unchanged. Override in subclasses to add
        domain meaningful labels, clinical mappings, etc.
        """
        return results
