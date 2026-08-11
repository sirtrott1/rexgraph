"""
agent.knowledge: one complex out of files that name the same things differently.

A schema, an ontology and a genome annotation are three descriptions of overlapping
subject matter, and the usual reason they cannot be used together is prosaic: each
names an entity its own way. A GTF exon row carries `gene_id ENSG00000012048` and
`gene_name BRCA1`; a GAF row carries `UniProtKB:P38398`, the symbol `BRCA1` and a
synonym list; a GO term carries `GO:0006281` with `alt_id`s. Nothing is ambiguous
about any of it, and nothing joins, because the join key is spelled differently in
each file.

Every one of those files states its own cross-references. This module reads only what
the files declare, unions the identifier sets transitively, and emits a single
relational complex whose vertices are entities and whose edges keep the relation and
the file they came from.

    k = join("genes.gtf", "goa_human.gaf", "go.obo")
    k.rex()                  # one complex spanning genome, annotation and ontology
    k.report["joined"]       # what actually connected, and through which identifier
    k.triples()              # the same thing for an agent or TrustGraph
    k.store(store, "study")  # into the RCDB, searchable by entity and by source

The join is transitive and stated-only: an entity reaches another identifier because
some file says the two name one thing, never because the strings resemble each other.
An identifier claimed by two entities within a file is reported as a collision and
declined as a key, since a wrong join produces a complex that looks richer than the
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import EdgeConstruction

#: identifier prefixes that name a namespace rather than an entity, so `GO:0006281`
#: and `UniProtKB:P38398` keep their prefix but `HGNC:HGNC:1100` collapses.
_REPEATED_PREFIX = ":"


@dataclass
class Part:
    """One file's contribution, in the form the join works on."""

    origin: str
    kind: str
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def entities(self) -> set:
        out = set()
        for s, _r, o in self.edges:
            out.add(s)
            out.add(o)
        return out


def _normalise_identifier(x: str) -> str:
    """An identifier reduced to the form two files would agree on.

    Case is not meaningful in gene symbols across databases, and a doubled namespace
    prefix (`HGNC:HGNC:1100`) is a formatting habit rather than a different id.
    """
    v = str(x).strip()
    if not v:
        return v
    parts = v.split(_REPEATED_PREFIX)
    if len(parts) >= 3 and parts[0] and parts[0] == parts[1]:
        v = _REPEATED_PREFIX.join(parts[1:])
    return v.casefold()


class _Union:
    """Union-find over identifiers, so a join through a chain of files works.

    `gene_name BRCA1` in a GTF and `UniProtKB:P38398` in a GAF are one entity only
    because a GAF row lists both. Following that one step at a time is what makes the
    third file, which knows only the accession, land on the same vertex.
    """

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:            # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _part_from_edge_construction(ec: EdgeConstruction, origin: str,
                                 kind: str) -> Part:
    labels = list(ec.vertex_labels)
    edges = []
    for k in range(ec.nE):
        t = int(ec.type_labels[k])
        rel = ec.type_names[t] if t < len(ec.type_names) else "related"
        edges.append((labels[int(ec.sources[k])], rel, labels[int(ec.targets[k])]))
    return Part(origin=origin or ec.origin or kind, kind=kind, edges=edges,
                aliases=dict(ec.vertex_aliases), meta={"nE": ec.nE, "nV": ec.nV})


def _part_from_ontology(parsed, origin: str) -> Part:
    return Part(origin=origin or parsed.meta.get("file") or parsed.meta.get("format", "ontology"),
                kind=parsed.meta.get("format", "ontology"),
                edges=[(s, r, o) for s, r, o in parsed.triples],
                aliases=dict(parsed.aliases),
                labels=dict(parsed.labels),
                meta=dict(parsed.meta))


def _part_from_rex(rex, origin: str, kind: str) -> Part:
    """A built complex as a part, so a stored record can join a new file.

    Vertex labels come from the complex's own meta where it has them; a complex with
    none is indexed by position, which joins to nothing and says so rather than
    inventing names that would collide with a real identifier.
    """
    meta = getattr(rex, "_agent_meta", {}) or {}
    labels = list(meta.get("vertex_labels") or [])
    if len(labels) < int(rex.nV):
        labels += [f"{origin}#{i}" for i in range(len(labels), int(rex.nV))]
    types = meta.get("type_names") or []
    tl = getattr(rex, "type_labels", None)
    edges = []
    S = np.asarray(rex.sources).astype(int)
    T = np.asarray(rex.targets).astype(int)
    for k in range(int(rex.nE)):
        rel = "related"
        if tl is not None and k < len(tl):
            t = int(tl[k])
            if t < len(types):
                rel = types[t]
        edges.append((labels[S[k]], rel, labels[T[k]]))
    return Part(origin=origin or kind, kind=kind, edges=edges,
                labels={x: x for x in labels}, meta={"nV": int(rex.nV),
                                                     "nE": int(rex.nE)})


def _part_from_schema(model, origin: str) -> Part:
    """A database schema as a part: tables are entities, foreign keys relate them.

    A schema joins on the names it uses. That is the alignment question a schema and
    an ontology are usually asked together: does the table this database calls
    `Specimen` correspond to the class the ontology calls `Specimen`, and does the
    foreign key between two tables agree with the subsumption between two classes.
    Nothing is matched on column contents, which a schema does not carry.
    """
    edges = []
    for fk in model.foreign_keys:
        if fk.from_table and fk.to_table and fk.from_table != fk.to_table:
            rel = "identifying_reference" if getattr(fk, "identifying", False) \
                else "references"
            edges.append((fk.from_table, rel, fk.to_table))
    names = model.table_names()
    return Part(origin=origin or "schema", kind="schema", edges=edges,
                labels={n: n for n in names},
                meta={"n_tables": len(names),
                      "n_foreign_keys": len(model.foreign_keys)})


def as_part(source: Any, origin: str = "") -> Part:
    """Accept whatever a caller has: a path, an EdgeConstruction, a ParsedOntology.

    A path is read by whichever reader claims it, so the same call takes a `.gtf`, a
    `.obo`, a `.gaf`, a `.pdb` or a `.csv` without the caller sorting them first. A
    `SchemaModel` and a built complex are accepted directly, since neither arrives as
    a file: one is reflected from a database, the other read back from the store.
    """
    from .adapters.ontology_formats import ParsedOntology
    from .schema_complex import SchemaModel

    if isinstance(source, Part):
        return source
    if isinstance(source, SchemaModel):
        return _part_from_schema(source, origin)
    if isinstance(source, ParsedOntology):
        return _part_from_ontology(source, origin)
    if isinstance(source, EdgeConstruction):
        return _part_from_edge_construction(source, origin, "edges")
    if hasattr(source, "sources") and hasattr(source, "nE"):
        return _part_from_rex(source, origin or "complex", "rex")
    if isinstance(source, (str, Path)):
        return _part_from_path(str(source), origin)
    raise TypeError(
        f"cannot join a {type(source).__name__}. Pass a path, an EdgeConstruction, "
        "or a ParsedOntology.")


def _part_from_path(path: str, origin: str = "") -> Part:
    """Read a file as whatever it is."""
    from .adapters.formats import reader_for
    from .adapters.ontology_formats import format_for_extension, read_ontology

    name = origin or Path(path).name
    fmt = format_for_extension(path)
    if fmt:
        p = read_ontology(path)
        p.meta.setdefault("file", name)
        return _part_from_ontology(p, name)
    reader = reader_for(path)
    if reader:
        from .adapters.formats import read
        return _part_from_edge_construction(read(path), name, reader)
    # anything else auto_rex can open: csv, json, text, a stored complex
    from .auto import auto_rex
    return _part_from_rex(auto_rex(path, face_selection="none"), name, "auto")


@dataclass
class Knowledge:
    """The joined result: entities, their relations, and what came from where."""

    entities: dict[str, list[str]]                 # canonical -> identifiers
    edges: list[tuple[str, str, str, str]]         # (subject, relation, object, origin)
    labels: dict[str, str]                         # canonical -> display name
    report: dict
    parts: list[Part] = field(default_factory=list)

    @property
    def nV(self) -> int:
        return len(self.entities)

    @property
    def nE(self) -> int:
        return len(self.edges)

    def triples(self, *, with_origin: bool = False):
        """The relations as triples, for an agent, TrustGraph, or an export."""
        if with_origin:
            return [(self.display(s), r, self.display(o), src)
                    for s, r, o, src in self.edges]
        return [(self.display(s), r, self.display(o)) for s, r, o, _ in self.edges]

    def display(self, canonical: str) -> str:
        return self.labels.get(canonical, canonical)

    def edge_construction(self, *, namespace_types: bool = True) -> EdgeConstruction:
        """One EdgeConstruction over the joined entities.

        Edge types are `origin:relation` by default, so the type channel separates a
        subsumption asserted by the ontology from an annotation asserted by the GAF
        from an overlap computed off the GTF. Selecting a sub-complex is then a type
        filter rather than a rebuild.
        """
        from .adapters.formats import _ec

        idx: dict[str, int] = {}
        labels: list[str] = []

        def vid(c):
            if c not in idx:
                idx[c] = len(labels)
                labels.append(self.display(c))
            return idx[c]

        types: dict[str, int] = {}
        src, tgt, tl = [], [], []
        for s, r, o, origin in self.edges:
            if s == o:
                continue
            name = f"{origin}:{r}" if namespace_types else r
            if name not in types:
                types[name] = len(types)
            src.append(vid(s))
            tgt.append(vid(o))
            tl.append(types[name])
        if not src:
            raise ValueError("the joined sources produced no relations")
        return _ec(np.asarray(src, np.int32), np.asarray(tgt, np.int32), labels,
                   types=np.asarray(tl, np.int32),
                   type_names=[n for n, _ in sorted(types.items(),
                                                    key=lambda kv: kv[1])],
                   aliases={self.display(c): v for c, v in self.entities.items()
                            if len(v) > 1},
                   origin="+".join(p.origin for p in self.parts))

    def rex(self, face_selection: str | None = None):
        """The complex. Faces are asked for, not assumed."""
        from .auto import FACE_RULE, build_rex_from_edges
        rex = build_rex_from_edges(
            self.edge_construction(),
            face_selection=FACE_RULE if face_selection is None else face_selection)
        rex._agent_meta = self.meta()
        return rex

    def meta(self) -> dict:
        return {
            "input_type": "knowledge",
            "source": "+".join(p.kind for p in self.parts),
            "origins": [p.origin for p in self.parts],
            "vertex_labels": [self.display(c) for c in self.entities],
            "n_entities": self.nV,
            "n_relations": self.nE,
            "join": self.report,
        }

    def unresolved(self) -> dict[str, list[str]]:
        """Entities still reading as a bare accession, grouped by namespace.

        An entity is resolved when something gave it a name: `GO:0006281` reads as
        "DNA repair" once the ontology defining it is present. One that still reads
        as its own identifier is a reference to a file that was not loaded, and it is
        the only signal needed to say which file that is.
        """
        out: dict[str, list[str]] = {}
        for canon in self.entities:
            shown = self.labels.get(canon, canon)
            if shown != canon:
                continue                          # something named it
            ns = namespace_of(canon)
            if ns:
                out.setdefault(ns, []).append(canon)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def recommendations(self) -> list[dict]:
        """What would make this join say more, based on what is already here.

        Every entry is derived from an exact property of the result: a namespace that
        appears with nothing defining it, an annotation set with no ontology beside
        it, sources that share no identifier at all. Nothing is fetched and nothing is
        guessed; each one names a file and says where it is published.
        """
        recs: list[dict] = []
        kinds = {p.kind for p in self.parts}
        unresolved = self.unresolved()

        for ns in sorted(unresolved):
            spec = NAMESPACES[ns]
            if kinds & spec.get("defined_by", frozenset()):
                # the file that defines this namespace is already here. Those
                # identifiers have no prettier name, which is a fact about the
                # format rather than something to load a second file for.
                continue
            ids = unresolved[ns]
            recs.append({
                "kind": "unresolved_namespace",
                "namespace": ns,
                "n_affected": len(ids),
                "examples": ids[:5],
                "detail": (f"{len(ids)} {spec['label']}(s) are referenced but nothing "
                           f"here names them, so they read as bare identifiers."),
                "action": (f"Load {' or '.join(spec['files'])} alongside these files "
                           f"to name them and pull in their structure."),
                "published": spec["published"],
            })

        if kinds & ANNOTATION_KINDS and not (kinds & ONTOLOGY_KINDS):
            recs.append({
                "kind": "annotation_without_ontology",
                "detail": ("An annotation set points at ontology terms, and the "
                           "ontology defining them is not loaded. The annotations "
                           "connect to term identifiers rather than to a hierarchy."),
                "action": "Add the ontology the annotation set was made against.",
                "published": "OBO Foundry, obofoundry.org",
            })

        if kinds & ONTOLOGY_KINDS and len(self.parts) == 1:
            recs.append({
                "kind": "ontology_without_data",
                "detail": ("An ontology on its own describes terms and nothing that "
                           "uses them."),
                "action": ("Add an annotation set (.gaf/.gpad) or a structure file "
                           "so entities in your data reach these terms."),
            })

        if len(self.parts) > 1 and self.report.get("n_joined", 0) == 0:
            recs.append({
                "kind": "no_shared_entities",
                "detail": ("No entity is named by more than one of these files, so "
                           "they form separate components and nothing crosses between "
                           "them."),
                "action": ("Check that the files describe overlapping subject matter, "
                           "and that at least one of them declares cross-references "
                           "(Dbxref, xref, a GAF synonym column) to the identifiers "
                           "the others use."),
            })

        if self.report.get("n_collisions", 0):
            recs.append({
                "kind": "collisions",
                "n_affected": self.report["n_collisions"],
                "detail": (f"{self.report['n_collisions']} identifier(s) name more "
                           "than one entity within a single file. These were NOT "
                           "merged, so those entities stay separate."),
                "action": ("Check whether the file means them as one thing. A gene "
                           "symbol repeated across feature levels is the usual "
                           "cause."),
            })
        return recs

    def features(self, *, rex=None, signal=None, t_scales=(0.5, 2.0)):
        """Per-relation features read off the complex's own tensor fields.

        The training signal for a joined complex is structural, not textual: an
        ontology and a genome annotation carry no prose to chunk, and the thing worth
        learning from them is where a relation sits in the field. Each relation reads
        its slice of the structural character, the RCFE curvature, the Hodge energies
        of the edge signal and that signal diffused at several scales.

        This is `warehouse.edge_features` on this complex, not a second copy of it.

        Returns
        -------
        (X, names, y, classes)
            `X` is (n_relations, n_features); `y` is the relation's type index and
            `classes` names them, so "which kind of relation is this, and which file
            asserted it" is a supervised target that needs no external labels.
        """
        from .warehouse.source import edge_features

        rex = rex if rex is not None else self.rex()
        ec = self.edge_construction()
        flow = np.asarray(signal if signal is not None else ec.w_E, dtype=np.float64)
        if flow.shape[0] != int(rex.nE):
            raise ValueError(
                f"signal has {flow.shape[0]} values for {rex.nE} relations")
        mask = np.arange(int(rex.nE), dtype=np.int64)
        X, names = edge_features(rex, flow, mask, t_scales=t_scales)
        y = np.asarray(ec.type_labels, dtype=np.int64)
        return X, names, y, list(ec.type_names)

    def health(self, *, flow=None) -> dict:
        """Whether load can drain through this structure or gets trapped circulating.

        The Hodge reading of a coordination graph, applied to a knowledge complex: a
        relation set with no cycle drains, one whose cycles carry harmonic content
        holds load in them. For an ontology that reads as definitional circularity;
        for a joined complex it also finds the entities every path runs through.

        `flow` is a per-relation load, defaulting to uniform, which reads the
        structure alone.
        """
        from rexgraph.mesh_health import mesh_health

        edges = [(self.display(s), self.display(o)) for s, _r, o, _origin in self.edges]
        if not edges:
            return {"n_nodes": 0, "n_edges": 0, "status": "empty"}
        return mesh_health(edges, flow=flow)

    def propagate(self, seed, *, t: float = 1.0, rex=None):
        """Diffuse a signal across grades through the coupled field operator.

        A seed on some relations spreads to the relations near them and to the faces
        above, so "what does this set reach" is answered by the complex rather than by
        a hop count. `seed` is a per-relation vector or a set of entity names, in
        which case every relation touching one of them starts at 1.

        Returns the propagated field over the relations, in the same order.
        """
        from rexgraph.field_propagator import field_heat

        rex = rex if rex is not None else self.rex(face_selection="none")
        n = int(rex.nE)
        if not isinstance(seed, np.ndarray) and not isinstance(seed, (list, tuple)):
            seed = [seed]
        if len(seed) and isinstance(next(iter(seed)), str):
            wanted = {str(x) for x in seed}
            vector = np.zeros(n, dtype=np.float64)
            for i, (a, _r, b, _o) in enumerate(self.edges[:n]):
                if self.display(a) in wanted or self.display(b) in wanted:
                    vector[i] = 1.0
        else:
            vector = np.asarray(seed, dtype=np.float64).ravel()
            if vector.shape[0] != n:
                raise ValueError(
                    f"seed has {vector.shape[0]} values for {n} relations")
        if not vector.any():
            raise ValueError("the seed touches no relation in this complex")
        out = np.asarray(field_heat(rex, vector, float(t))).ravel()
        return out[:n]

    def store(self, store, record_id: str, *, tags=None, face_selection=None,
              rex=None):
        """Put the joined complex in the RCDB, tagged by every source that fed it.

        `rex` accepts a complex already built from this join, so a caller that needed
        it for something else does not pay to build it twice.
        """
        rex = rex if rex is not None else self.rex(face_selection=face_selection)
        meta = self.meta()
        every = ["knowledge"] + sorted({p.kind for p in self.parts}) + list(tags or [])
        store.put(record_id, rex, meta=meta, tags=sorted(set(every)))
        return record_id

    def summary(self) -> str:
        r = self.report
        lines = [f"{self.nV} entities, {self.nE} relations, "
                 f"{len(self.parts)} source(s)"]
        for p in self.parts:
            lines.append(f"  {p.origin} ({p.kind}): {len(p.edges)} relations")
        lines.append(f"  joined across sources: {r['n_joined']} entities")
        if r["collisions"]:
            lines.append(f"  identifier collisions: {len(r['collisions'])}")
        return "\n".join(lines)


def join(*sources, origins: list[str] | None = None) -> Knowledge:
    """Join files into one complex on the identifiers they declare.

    Parameters
    ----------
    *sources
        Paths, EdgeConstructions or ParsedOntologies, in any mix.
    origins
        Names for the sources, one per source. Defaults to filenames.

    Returns
    -------
    Knowledge
        The entities, their relations with provenance, and a report of what joined.
    """
    names = list(origins or [])
    parts = [as_part(s, names[i] if i < len(names) else "")
             for i, s in enumerate(sources)]
    if not parts:
        return Knowledge({}, [], {}, _empty_report(), [])

    # 1. every identifier a part offers for an entity, including the entity itself.
    #    Collected first so the ambiguous ones can be found before anything is unioned.
    id_sets: list[tuple[Part, str, list[str]]] = []
    for part in parts:
        # A term a file names but never relates is still an identity that file
        # declares. GO's `nucleus` has a label and, in a fragment, no is_a; without
        # it in the index a GAF's `located_in GO:0005634` cannot reach the name the
        # ontology gives it, and the entity reads as a bare accession.
        declared = part.entities | set(part.labels) | set(part.aliases)
        for ent in sorted(declared):
            ids = [ent, *part.aliases.get(ent, [])]
            keys = [_normalise_identifier(i) for i in ids if str(i).strip()]
            keys = [k for k in keys if k]
            if not keys:
                continue
            id_sets.append((part, ent, keys))

    # 2. an identifier claimed by two entities of the SAME part is a collision: one
    #    file calling two things by one name is a fact about the file, not a join.
    claims: dict[tuple[str, str], set] = {}
    for part, ent, keys in id_sets:
        for k in keys:
            claims.setdefault((part.origin, k), set()).add(ent)
    collisions = [{"origin": o, "identifier": k, "entities": sorted(v)[:6],
                   "n_entities": len(v)}
                  for (o, k), v in sorted(claims.items()) if len(v) > 1]
    # An identifier one file gives to several entities cannot be a join key: using it
    # merges things the file itself distinguishes. An annotation file whose accession
    # column repeats a value would otherwise collapse every row sharing it into one
    # entity, which is the wrong join rather than a missing one.
    ambiguous = {k for (_o, k), v in claims.items() if len(v) > 1}

    # 3. union on the identifiers that are unambiguous within their own file
    uf = _Union()
    for _part, _ent, keys in id_sets:
        usable = [k for k in keys if k not in ambiguous] or keys[:1]
        for k in usable[1:]:
            uf.union(usable[0], k)

    # 4. group by root; name each group and record which parts reached it
    groups: dict[str, dict] = {}
    for part, ent, keys in id_sets:
        root = uf.find(_root_key(keys, ambiguous))
        g = groups.setdefault(root, {"ids": set(), "names": [], "origins": set(),
                                     "labels": []})
        g["ids"].update(keys)
        g["names"].append(ent)
        g["origins"].add(part.origin)
        lab = part.labels.get(ent)
        if lab:
            g["labels"].append(lab)

    canonical_of: dict[tuple[str, str], str] = {}
    entities: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    # name each group once, then attribute every identifier set to its group in a
    # single pass. Scanning all identifier sets per group instead is quadratic, which
    # at ontology scale is the whole cost of the join.
    canon_of_root: dict[str, str] = {}
    for root, g in groups.items():
        canon = _canonical_name(g)
        canon_of_root[root] = canon
        entities[canon] = sorted(g["ids"])
        labels[canon] = g["labels"][0] if g["labels"] else canon
    for part, ent, keys in id_sets:
        canonical_of[(part.origin, ent)] = canon_of_root[
            uf.find(_root_key(keys, ambiguous))]

    # 4. rewrite every edge onto its canonical endpoints
    edges = []
    for part in parts:
        for s, r, o in part.edges:
            cs = canonical_of.get((part.origin, s))
            co = canonical_of.get((part.origin, o))
            if cs is None or co is None:
                continue
            edges.append((cs, r, co, part.origin))

    referenced = {c for e in edges for c in (e[0], e[2])}
    n_ambiguous_skipped = len(ambiguous)
    # invert canon_of_root rather than re-deriving the root from an identifier.
    # `groups` is keyed by uf.find(_root_key(...)), which SKIPS ambiguous keys, so an
    # entity whose alphabetically-first id happens to be an ambiguous one finds its own
    # singleton root instead, and that root is not in `groups`. Real data hits this:
    # joining go-basic.obo with goa_human.gaf raised KeyError: 'aqp9'.
    root_of_canon = {canon: root for root, canon in canon_of_root.items()}
    joined = {c: sorted(groups[root_of_canon[c]]["origins"])
              for c in entities
              if len(groups[root_of_canon[c]]["origins"]) > 1}
    declared_only = sorted(c for c in entities if c not in referenced)
    report = {
        "n_sources": len(parts),
        "sources": [{"origin": p.origin, "kind": p.kind,
                     "n_relations": len(p.edges),
                     "n_entities": len(p.entities)} for p in parts],
        "n_entities": len(entities),
        "n_relations": len(edges),
        "n_joined": len(joined),
        "n_referenced": len(referenced),
        "n_declared_unreferenced": len(declared_only),
        "declared_unreferenced": [labels.get(c, c) for c in declared_only][:50],
        "joined": [{"entity": labels.get(c, c), "sources": v}
                   for c, v in sorted(joined.items())][:200],
        "collisions": collisions[:50],
        "n_collisions": len(collisions),
        "n_identifiers_declined_as_keys": n_ambiguous_skipped,
    }
    k = Knowledge(entities, edges, labels, report, parts)
    report["recommendations"] = k.recommendations()
    return k


#: parts whose relations point INTO an ontology rather than defining one
ANNOTATION_KINDS = frozenset({"gaf", "gpad"})
ONTOLOGY_KINDS = frozenset({"obo", "obograph", "rdfxml", "turtle", "ntriples",
                            "triples"})

#: identifier namespaces, and the file that defines each one.
#:
#: Reference data, not a download list. Nothing here is fetched: a recommendation
#: names the file and says where it is published, and getting it is the user's call.
#: Extend by adding an entry; nothing else reads these by name.
NAMESPACES: dict[str, dict] = {
    "GO": {
        "label": "Gene Ontology term",
        "prefixes": ("go:",),
        "files": ("go.obo", "go.json"),
        "published": "OBO Foundry, purl.obolibrary.org/obo/go.obo",
            "defined_by": ONTOLOGY_KINDS,
    },
    "CHEBI": {
        "label": "ChEBI chemical entity",
        "prefixes": ("chebi:",),
        "files": ("chebi.obo", "chebi.owl"),
        "published": "OBO Foundry, purl.obolibrary.org/obo/chebi.obo",
            "defined_by": ONTOLOGY_KINDS,
    },
    "HP": {
        "label": "Human Phenotype Ontology term",
        "prefixes": ("hp:",),
        "files": ("hp.obo", "hp.json"),
        "published": "OBO Foundry, purl.obolibrary.org/obo/hp.obo",
            "defined_by": ONTOLOGY_KINDS,
    },
    "MONDO": {
        "label": "Mondo disease term",
        "prefixes": ("mondo:",),
        "files": ("mondo.obo", "mondo.json"),
        "published": "OBO Foundry, purl.obolibrary.org/obo/mondo.obo",
            "defined_by": ONTOLOGY_KINDS,
    },
    "UBERON": {
        "label": "Uberon anatomy term",
        "prefixes": ("uberon:",),
        "files": ("uberon.obo",),
        "published": "OBO Foundry, purl.obolibrary.org/obo/uberon.obo",
            "defined_by": ONTOLOGY_KINDS,
    },
    "SO": {
        "label": "Sequence Ontology term",
        "prefixes": ("so:",),
        "files": ("so.obo",),
        "published": "OBO Foundry, purl.obolibrary.org/obo/so.obo",
            "defined_by": ONTOLOGY_KINDS,
    },
    "PR": {
        "label": "Protein Ontology term",
        "prefixes": ("pr:",),
        "files": ("pr.obo",),
        "published": "OBO Foundry, purl.obolibrary.org/obo/pr.obo",
            "defined_by": ONTOLOGY_KINDS,
    },
    "UniProtKB": {
        "label": "UniProt accession",
        "prefixes": ("uniprotkb:", "uniprot:"),
        "files": ("a GAF annotation set",),
        "published": "UniProt/GOA, current.geneontology.org/annotations",
            "defined_by": ANNOTATION_KINDS,
    },
    "Ensembl": {
        "label": "Ensembl identifier",
        "prefixes": ("ensg", "enst", "ensp", "ensembl:"),
        "files": ("the matching GTF/GFF3 annotation",),
        "published": "Ensembl, ftp.ensembl.org",
            "defined_by": frozenset({"gff", "bed"}),
    },
    "NCBIGene": {
        "label": "NCBI Gene identifier",
        "prefixes": ("geneid:", "ncbigene:"),
        "files": ("a GAF annotation set", "the matching GFF3"),
        "published": "NCBI, ftp.ncbi.nlm.nih.gov/gene",
            "defined_by": ANNOTATION_KINDS,
    },
    "HGNC": {
        "label": "HGNC gene symbol record",
        "prefixes": ("hgnc:",),
        "files": ("a GAF annotation set",),
        "published": "HGNC, genenames.org",
            "defined_by": ANNOTATION_KINDS,
    },
}

def namespace_of(identifier: str) -> str | None:
    """Which catalogued namespace an identifier belongs to, if any."""
    v = str(identifier).strip().casefold()
    for name, spec in NAMESPACES.items():
        if v.startswith(spec["prefixes"]):
            return name
    return None


def _root_key(keys: list[str], ambiguous: set) -> str:
    """The identifier a group is looked up by: the first unambiguous one.

    Falls back to the entity's own name when every identifier it offers is ambiguous,
    which keeps it a group of one rather than merging it into the collision.
    """
    for k in keys:
        if k not in ambiguous:
            return k
    return keys[0]


def _canonical_name(group: dict) -> str:
    """The identifier a group is keyed by.

    A readable name beats an accession, and among identifiers the shortest stable one
    wins, so an entity known as `BRCA1`, `ENSG00000012048` and `UniProtKB:P38398`
    keys on `BRCA1` rather than on whichever file happened to be read first.
    """
    names = sorted(set(group["names"]))
    if not names:
        return sorted(group["ids"])[0]
    return min(names, key=lambda n: (len(n), n))


def _empty_report() -> dict:
    return {"n_sources": 0, "sources": [], "n_entities": 0, "n_relations": 0,
            "n_joined": 0, "joined": [], "collisions": [], "n_collisions": 0}
