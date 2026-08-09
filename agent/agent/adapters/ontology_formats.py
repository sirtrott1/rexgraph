"""
agent.adapters.ontology_formats: readers for the files ontologies actually ship as.

An ontology is distributed as a file, not as a box of typed triples. GO, HPO, MONDO,
ChEBI and UBERON all publish `.obo` and `.owl`; the modern GO distribution is OBO
Graphs JSON; and what relates an ontology to biology is a `.gaf`, which is what an
annotation set is.

Every parser here returns the same thing: `ParsedOntology`, carrying

    triples   (subject, predicate, object) in the ontology's own vocabulary
    labels    id -> human name, so a complex reads "apoptotic process", not GO:0006915
    meta      what the file said about itself

`agent.ontology_complex.parse_rdf` consumes the triples unchanged, so the diagnosis
is the same one the typed-triple path already gets. Nothing here interprets: `is_a`
stays `is_a` and the mapping to gradient/definition/object happens where it already
happened.

Parsers use the standard library. `owlready2`, `rdflib` and `goatools` are not
required. These layouts are specified and stable, and the subset an ontology's
skeleton needs (subsumption, equivalence, typed relations) is the well-behaved part
of each of them.

The RDF/XML and Turtle readers cover that skeleton, not the whole specification. A
file using a construct they do not read is reported through `meta['unparsed']`
rather than silently dropped, because an ontology that quietly loses half its axioms
diagnoses as a clean hierarchy.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

#: predicate written for a GAF/GPAD annotation whose column said nothing more precise
DEFAULT_ANNOTATION = "annotated_with"

#: GAF column 9 (Aspect) -> the GO relation it stands for
GAF_ASPECT = {"P": "involved_in", "F": "enables", "C": "located_in"}


@dataclass
class ParsedOntology:
    """What a parsed ontology file yields, before any interpretation."""

    triples: list[tuple[str, str, str]] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    declarations: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    #: term -> the other identifiers naming it. An OBO term declares `alt_id` and
    #: `xref`; a GAF row names one gene product by accession, symbol and synonym at
    #: once. Those are join keys stated by the file, and they are how an annotation
    #: reaches a genome annotation that spells the same gene differently.
    aliases: dict[str, list[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.triples)

    def named_triples(self) -> list[tuple[str, str, str]]:
        """The triples with ids replaced by names where a name is known.

        This is what a reader wants on screen. It is not what an identity-preserving
        pipeline wants, so it is a method rather than the default.
        """
        def nm(x):
            return self.labels.get(x, x)
        return [(nm(s), p, nm(o)) for s, p, o in self.triples]


def _count(d: dict) -> dict:
    """value -> how many entities declared it. What `kinds` reports."""
    out: dict[str, int] = {}
    for v in d.values():
        out[v] = out.get(v, 0) + 1
    return out


def _open_text(path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")


def _text_of(source) -> str:
    """Accept a path or the text itself."""
    s = str(source)
    if "\n" not in s and len(s) < 4096:
        p = Path(s)
        if p.exists():
            with _open_text(p) as fh:
                return fh.read()
    return s


#### OBO


#: OBO tags that are relations between terms rather than facts about one term
_OBO_REL_TAGS = {"is_a", "relationship", "intersection_of", "union_of",
                 "equivalent_to", "disjoint_from", "inverse_of"}

#: OBO tags naming the SAME term rather than a related one. `alt_id` is a merged-in
#: id and `xref` is the term in another vocabulary; both resolve to this term, so
#: they are aliases and emitting them as edges would relate a term to itself.
_OBO_ALIAS_TAGS = {"alt_id", "xref"}


def parse_obo(source) -> ParsedOntology:
    """OBO 1.4 flat file: the format GO, HPO, MONDO, ChEBI and UBERON ship.

    Stanzas are `[Term]` / `[Typedef]` blocks of `tag: value` lines. A trailing
    ``! comment`` is the human name of the referenced id and is dropped from the
    value, since the id is the identity.

    Obsolete terms are parsed and marked rather than skipped: an ontology's obsolete
    set is part of what it says, and dropping it silently changes the term count.
    """
    text = _text_of(source)
    triples: list[tuple[str, str, str]] = []
    labels: dict[str, str] = {}
    obsolete: set[str] = set()
    namespaces: dict[str, int] = {}
    aliases: dict[str, set] = {}
    stanza, current = None, {}
    n_terms = n_typedefs = 0

    def flush():
        nonlocal n_terms, n_typedefs
        if not current or stanza not in ("Term", "Typedef"):
            return
        tid = current.get("id")
        if not tid:
            return
        if stanza == "Term":
            n_terms += 1
        else:
            n_typedefs += 1
        if "name" in current:
            labels[tid] = current["name"]
        if current.get("namespace"):
            ns = current["namespace"]
            namespaces[ns] = namespaces.get(ns, 0) + 1
        if current.get("is_obsolete") == "true":
            obsolete.add(tid)
        for rel in current.get("_rels", []):
            triples.append((tid, rel[0], rel[1]))
        for a in current.get("_aliases", []):
            if a and a != tid:
                aliases.setdefault(tid, set()).add(a)

    with_lines = text.splitlines()
    for raw in with_lines:
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("[") and line.endswith("]"):
            flush()
            stanza, current = line[1:-1], {}
            continue
        if ":" not in line:
            continue
        tag, _, value = line.partition(":")
        tag = tag.strip()
        value = value.split("!")[0].strip()
        if not value:
            continue
        if tag in _OBO_ALIAS_TAGS:
            current.setdefault("_aliases", []).append(value.split()[0])
        elif tag in _OBO_REL_TAGS:
            parts = value.split()
            if tag == "relationship" and len(parts) >= 2:
                current.setdefault("_rels", []).append((parts[0], parts[1]))
            elif tag == "intersection_of" and len(parts) >= 2:
                # `intersection_of: part_of GO:x` is ONE conjunct of the definition,
                # stated through a relation. Emitting the relation and the
                # membership both would put two parallel edges on one axiom, which
                # reads as a 2-cycle the term does not have.
                current.setdefault("_rels", []).append((parts[0], parts[1]))
            elif parts:
                current.setdefault("_rels", []).append((tag, parts[0]))
        else:
            current.setdefault(tag, value)
    flush()

    return ParsedOntology(triples, labels, aliases={
        k: sorted(v) for k, v in aliases.items()}, meta={
        "format": "obo",
        "n_terms": n_terms,
        "n_typedefs": n_typedefs,
        "n_obsolete": len(obsolete),
        "obsolete": sorted(obsolete)[:200],
        "namespaces": namespaces,
    })


#### OBO Graphs JSON


def parse_obograph(source) -> ParsedOntology:
    """OBO Graphs JSON: the form GO and the OBO Foundry publish alongside `.obo`.

    Shape is `{"graphs": [{"nodes": [...], "edges": [{"sub", "pred", "obj"}]}]}`.
    A `pred` of `is_a` stays `is_a`; anything else is the relation's own IRI, whose
    local name is what the ontology calls it.
    """
    text = _text_of(source)
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not OBO Graphs JSON: {e}") from e

    graphs = doc.get("graphs")
    if graphs is None:
        raise ValueError("not OBO Graphs JSON: no 'graphs' key")

    triples: list[tuple[str, str, str]] = []
    labels: dict[str, str] = {}
    n_nodes = n_deprecated = 0
    for g in graphs:
        for node in g.get("nodes", []) or []:
            n_nodes += 1
            nid = node.get("id")
            if not nid:
                continue
            if node.get("lbl"):
                labels[nid] = node["lbl"]
            if (node.get("meta") or {}).get("deprecated"):
                n_deprecated += 1
        for e in g.get("edges", []) or []:
            sub, pred, obj = e.get("sub"), e.get("pred"), e.get("obj")
            if sub and pred and obj:
                triples.append((sub, _local_name(pred), obj))
        # equivalence axioms live outside `edges`
        for ax in g.get("equivalentNodesSets", []) or []:
            members = ax.get("nodeIds") or []
            for i in range(len(members) - 1):
                triples.append((members[i], "equivalentClass", members[i + 1]))

    return ParsedOntology(triples, labels, meta={
        "format": "obograph",
        "n_terms": n_nodes,
        "n_obsolete": n_deprecated,
        "n_graphs": len(graphs),
    })


#### RDF serialisations


def _local_name(uri: str) -> str:
    """The name an IRI ends in, which is what the ontology calls the thing.

    ElementTree hands back tags as `{namespace}local`, so the brace form is stripped
    before the IRI separators. Splitting on `#` first would otherwise leave the
    closing brace on the front of every RDF/XML predicate.
    """
    u = uri.strip().strip("<>")
    if u.startswith("{") and "}" in u:
        u = u.split("}", 1)[1]
    for sep in ("#", "/"):
        if sep in u:
            u = u.rsplit(sep, 1)[-1]
    return u or uri


_NT_TERM = re.compile(r'<([^>]*)>|_:(\S+)|"((?:[^"\\]|\\.)*)"(?:\^\^\S+|@\S+)?')


def parse_ntriples(source) -> ParsedOntology:
    """N-Triples: one `<s> <p> <o> .` per line. The unambiguous RDF serialisation."""
    text = _text_of(source)
    triples: list[tuple[str, str, str]] = []
    labels: dict[str, str] = {}
    declarations: dict[str, str] = {}
    unparsed = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        terms = _NT_TERM.findall(line)
        if len(terms) < 3:
            unparsed += 1
            continue
        vals = [(a or b or c) for a, b, c in terms[:3]]
        s, p, o = vals
        pl = _local_name(p)
        if pl in ("label", "prefLabel", "title"):
            labels[_local_name(s)] = o
            continue
        if pl == "type":
            declarations[_local_name(s)] = _local_name(o)
            continue
        triples.append((_local_name(s), pl, _local_name(o)))
    return ParsedOntology(triples, labels, declarations, meta={
        "format": "ntriples", "unparsed": unparsed,
        "kinds": _count(declarations)})


_TTL_PREFIX = re.compile(r'@prefix\s+(\S*):\s*<([^>]*)>\s*\.', re.I)


def parse_turtle(source) -> ParsedOntology:
    """Turtle, over the subset an ontology's skeleton uses.

    Handles `@prefix`, the `a` keyword, `;` (same subject) and `,` (same predicate)
    continuations, and `.` termination. Blank-node collections and nested brackets
    are counted in `meta['unparsed']` rather than guessed at.
    """
    text = _text_of(source)
    prefixes = dict(_TTL_PREFIX.findall(text))
    body = _TTL_PREFIX.sub("", text)
    body = re.sub(r'@base\s+<[^>]*>\s*\.', "", body, flags=re.I)
    body = re.sub(r'#.*', "", body)

    def expand(tok: str) -> str:
        t = tok.strip()
        if t.startswith("<") and t.endswith(">"):
            return _local_name(t)
        if t == "a":
            return "type"
        if ":" in t:
            pre, _, rest = t.partition(":")
            if pre in prefixes or pre == "":
                return rest
        return t.strip('"')

    triples: list[tuple[str, str, str]] = []
    labels: dict[str, str] = {}
    declarations: dict[str, str] = {}
    unparsed = 0
    for stmt in body.split("."):
        stmt = stmt.strip()
        if not stmt:
            continue
        if "[" in stmt or "]" in stmt or "(" in stmt:
            unparsed += 1
            continue
        parts = stmt.split(";")
        head = parts[0].split(None, 2)
        if len(head) < 3:
            unparsed += 1
            continue
        subj = expand(head[0])
        rest = [" ".join(head[1:])] + parts[1:]
        for clause in rest:
            c = clause.strip()
            if not c:
                continue
            bits = c.split(None, 1)
            if len(bits) < 2:
                unparsed += 1
                continue
            pred = expand(bits[0])
            for obj in bits[1].split(","):
                o = obj.strip()
                if not o:
                    continue
                if pred in ("label", "prefLabel", "title"):
                    labels[subj] = o.strip('"')
                elif pred == "type":
                    declarations[subj] = expand(o)
                else:
                    triples.append((subj, pred, expand(o)))
    return ParsedOntology(triples, labels, declarations, meta={
        "format": "turtle", "unparsed": unparsed,
        "prefixes": sorted(prefixes), "kinds": _count(declarations)})


def parse_rdfxml(source) -> ParsedOntology:
    """RDF/XML, which is what a `.owl` file usually is.

    Reads the class skeleton: every `owl:Class` / `rdf:Description` with an
    `rdf:about`, and its object-valued children (`rdfs:subClassOf`,
    `owl:equivalentClass`, `owl:disjointWith` and any other `rdf:resource`).
    Restriction bodies are counted in `meta['unparsed']`.
    """
    import xml.etree.ElementTree as ET

    text = _text_of(source)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ValueError(f"not RDF/XML: {e}") from e

    RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
    triples: list[tuple[str, str, str]] = []
    labels: dict[str, str] = {}
    declarations: dict[str, str] = {}
    unparsed = 0

    for node in root.iter():
        about = node.get(f"{RDF}about") or node.get(f"{RDF}ID")
        if not about:
            continue
        subj = _local_name(about)
        tag = _local_name(node.tag)
        if tag not in ("Description",):
            # the element name is the entity's kind: owl:Class, owl:ObjectProperty
            declarations[subj] = tag
        for child in node:
            ctag = _local_name(child.tag)
            res = child.get(f"{RDF}resource")
            if ctag in ("label", "prefLabel", "title"):
                if (child.text or "").strip():
                    labels[subj] = child.text.strip()
                continue
            if ctag == "type":
                if res:
                    declarations[subj] = _local_name(res)
                continue
            if res:
                triples.append((subj, ctag, _local_name(res)))
            elif len(child):
                unparsed += 1
    kinds = _count(declarations)
    return ParsedOntology(triples, labels, declarations, meta={
        "format": "rdfxml", "kinds": kinds,
        "n_classes": kinds.get("Class", 0),
        "n_properties": sum(v for k, v in kinds.items() if k.endswith("Property")),
        "unparsed": unparsed})


#### annotations: where an ontology meets the biology


def parse_gaf(source) -> ParsedOntology:
    """GO Annotation File: what relates gene products to ontology terms.

    17 tab-separated columns. The triple is (gene product, relation, term), where the
    relation is the Qualifier column when it names one (`involved_in`, `enables`,
    `located_in`, `part_of`) and otherwise the relation the Aspect column stands for.

    A `NOT` qualifier is a negative annotation and is recorded in `meta` rather than
    emitted, because asserting the relation would state the opposite of the file.
    """
    text = _text_of(source)
    triples: list[tuple[str, str, str]] = []
    labels: dict[str, str] = {}
    negatives = 0
    taxa: dict[str, int] = {}
    evidence: dict[str, int] = {}
    aliases: dict[str, set] = {}
    version = ""

    for raw in text.splitlines():
        if raw.startswith("!"):
            if "gaf-version" in raw:
                version = raw.strip("! ").strip()
            continue
        cols = raw.rstrip("\n").split("\t")
        if len(cols) < 9:
            continue
        symbol = cols[2].strip() or cols[1].strip()
        qualifier = cols[3].strip()
        term = cols[4].strip()
        ev = cols[6].strip()
        aspect = cols[8].strip()
        if not symbol or not term:
            continue
        if "NOT" in qualifier.split("|"):
            negatives += 1
            continue
        rel = next((q for q in qualifier.split("|") if q and q != "NOT"), "")
        pred = rel or GAF_ASPECT.get(aspect, DEFAULT_ANNOTATION)
        triples.append((symbol, pred, term))
        # the row names this product several ways: DB:accession, the symbol, and a
        # pipe-separated synonym list. Whichever a genome annotation used, it
        # reaches this product through one of them.
        other = set()
        if cols[1].strip():
            other.add(cols[1].strip())
            if cols[0].strip():
                other.add(f"{cols[0].strip()}:{cols[1].strip()}")
        if len(cols) > 10:
            other.update(x.strip() for x in cols[10].split("|") if x.strip())
        other.discard(symbol)
        if other:
            aliases.setdefault(symbol, set()).update(other)
        if not cols[2].strip() and len(cols) > 9 and cols[9].strip():
            # only when column 3 gave no symbol, so an accession still reads as
            # something. A gene symbol is already the name and must not be replaced
            # by the long product description.
            labels[symbol] = cols[9].strip()
        if ev:
            evidence[ev] = evidence.get(ev, 0) + 1
        if len(cols) > 12 and cols[12].strip():
            taxa[cols[12].strip()] = taxa.get(cols[12].strip(), 0) + 1

    return ParsedOntology(triples, labels, aliases={
        k: sorted(v) for k, v in aliases.items()}, meta={
        "format": "gaf", "version": version,
        "n_annotations": len(triples), "n_negative": negatives,
        "evidence_codes": evidence, "taxa": taxa})


def parse_gpad(source) -> ParsedOntology:
    """GPAD: the same association, split so the gene product lives in a GPI file.

    12 columns; the relation is column 3 and is a real relation IRI rather than an
    aspect letter, so nothing has to be inferred.
    """
    text = _text_of(source)
    triples: list[tuple[str, str, str]] = []
    negatives = 0
    evidence: dict[str, int] = {}
    for raw in text.splitlines():
        if raw.startswith("!"):
            continue
        cols = raw.rstrip("\n").split("\t")
        if len(cols) < 6:
            continue
        subject = f"{cols[0].strip()}:{cols[1].strip()}".strip(":")
        qualifier = cols[2].strip()
        term = cols[3].strip()
        ev = cols[5].strip()
        if not subject or not term:
            continue
        if "NOT" in qualifier.split("|"):
            negatives += 1
            continue
        rel = next((q for q in qualifier.split("|") if q and q != "NOT"), "")
        triples.append((subject, _local_name(rel) or DEFAULT_ANNOTATION, term))
        if ev:
            evidence[ev] = evidence.get(ev, 0) + 1
    return ParsedOntology(triples, {}, meta={
        "format": "gpad", "n_annotations": len(triples),
        "n_negative": negatives, "evidence_codes": evidence})


#### plain triples, which is what the box on screen was


def parse_triple_lines(source) -> ParsedOntology:
    """One `subject predicate object` per line, the format the text box accepted.

    Kept as a reader so pasted text and an uploaded file take the same path.
    """
    text = _text_of(source)
    triples = []
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) >= 3:
            triples.append((parts[0], parts[1], " ".join(parts[2:])))
    return ParsedOntology(triples, {}, meta={"format": "triples"})


#### detection and dispatch


#: reader name -> (parser, extensions)
PARSERS = {
    "obo": (parse_obo, (".obo",)),
    "obograph": (parse_obograph, (".obojson",)),
    "rdfxml": (parse_rdfxml, (".owl", ".rdf", ".owx")),
    "turtle": (parse_turtle, (".ttl",)),
    "ntriples": (parse_ntriples, (".nt", ".ntriples")),
    "gaf": (parse_gaf, (".gaf",)),
    "gpad": (parse_gpad, (".gpad", ".gpa")),
    "triples": (parse_triple_lines, (".triples",)),
}


def available_formats() -> dict[str, list[str]]:
    """format name -> the extensions it claims. What the screen offers."""
    return {name: list(exts) for name, (_fn, exts) in PARSERS.items()}


def format_for_extension(path) -> str | None:
    p = Path(str(path))
    ext = p.suffix.lower()
    if ext == ".gz":
        ext = Path(p.stem).suffix.lower()
    for name, (_fn, exts) in PARSERS.items():
        if ext in exts:
            return name
    return None


def sniff_format(text: str) -> str | None:
    """Name the format from the content.

    Needed because an ontology's extension is often `.json` or absent: GO ships OBO
    Graphs as `go.json`, and a pasted block has no filename at all.
    """
    head = text.lstrip()[:4000]
    if not head:
        return None
    if head.startswith("{"):
        return "obograph" if '"graphs"' in head or '"nodes"' in head else None
    if head.startswith("<?xml") or head.startswith("<rdf:RDF") or "<owl:" in head:
        return "rdfxml"
    if head.startswith("!gaf-version") or "\tGO:" in head[:2000]:
        return "gaf"
    if head.startswith("!gpa-version") or head.startswith("!gpad-version"):
        return "gpad"
    if "format-version:" in head or head.startswith("[Term]") or "\n[Term]" in head:
        return "obo"
    if "@prefix" in head or "@base" in head:
        return "turtle"
    if re.search(r'^\s*<[^>]+>\s+<[^>]+>\s+', head, re.M):
        return "ntriples"
    if any(len(ln.split()) >= 3 for ln in head.splitlines() if ln.strip()):
        return "triples"
    return None


def parse(source, fmt: str | None = None) -> ParsedOntology:
    """Parse `source` (a path or the text itself) as `fmt`, or detect the format.

    Detection prefers the extension and falls back to the content, so `go.json`
    parses as OBO Graphs even though `.json` claims nothing here.
    """
    if fmt and fmt not in PARSERS:
        raise ValueError(
            f"unknown ontology format {fmt!r}. Available: "
            f"{', '.join(sorted(PARSERS))}")
    text = _text_of(source)
    if not fmt:
        fmt = format_for_extension(source) or sniff_format(text)
    if not fmt:
        raise ValueError(
            "could not tell what ontology format this is. Supported: "
            f"{', '.join(sorted(PARSERS))}")
    out = PARSERS[fmt][0](text)
    out.meta.setdefault("format", fmt)
    return out


def read_ontology(path, fmt: str | None = None) -> ParsedOntology:
    """Parse an ontology file, gzip transparent."""
    with _open_text(path) as fh:
        text = fh.read()
    return parse(text, fmt or format_for_extension(path))


def combine(*parsed: ParsedOntology) -> ParsedOntology:
    """Merge parsed files into one complex.

    This is the point of reading annotations: a `.gaf` relates gene products to term
    ids and says nothing about what those ids mean, while the `.obo` holds the
    hierarchy and the names. Loaded together, an annotation reaches the ontology it
    annotates, and a gene inherits the subsumption path above every term it is
    annotated with.

    Labels merge across all inputs, so `GO:0006915` coming from the annotation file
    reads as "apoptotic process" from the ontology. Duplicate triples are dropped:
    the same assertion made by two files is one relation.
    """
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    labels: dict[str, str] = {}
    declarations: dict[str, str] = {}
    merged_aliases: dict[str, set] = {}
    formats, sources = [], []
    for p in parsed:
        if p is None:
            continue
        formats.append(p.meta.get("format", "?"))
        sources.append({k: v for k, v in p.meta.items() if k != "obsolete"})
        labels.update(p.labels)
        declarations.update(p.declarations)
        for k, v in p.aliases.items():
            merged_aliases.setdefault(k, set()).update(v)
        for t in p.triples:
            if t not in seen:
                seen.add(t)
                triples.append(t)
    resolved = sum(1 for s, _p, o in triples
                   if s in labels or o in labels)
    return ParsedOntology(triples, labels, declarations, aliases={
        k: sorted(v) for k, v in merged_aliases.items()}, meta={
        "format": "+".join(formats) if formats else "empty",
        "n_inputs": len(formats), "sources": sources,
        "n_triples": len(triples), "n_labelled": len(labels),
        "n_triples_touching_a_named_term": resolved,
    })


#### bridge into the ordinary document path


def to_edge_construction(parsed: ParsedOntology, *, named: bool = True):
    """An `EdgeConstruction` over the parsed triples, typed by predicate.

    This is what makes an ontology file work everywhere a document works: uploaded,
    analysed, chunked, stored. The predicate becomes the edge type, so the type
    channel carries `is_a` apart from `part_of` apart from `enables`.
    """
    import numpy as np

    from .formats import _ec

    triples = parsed.named_triples() if named else parsed.triples
    idx: dict[str, int] = {}
    labels: list[str] = []

    def vid(name):
        if name not in idx:
            idx[name] = len(labels)
            labels.append(name)
        return idx[name]

    preds: dict[str, int] = {}
    src, tgt, types = [], [], []
    for s, p, o in triples:
        if s == o:
            continue
        if p not in preds:
            preds[p] = len(preds)
        src.append(vid(s))
        tgt.append(vid(o))
        types.append(preds[p])
    if not src:
        raise ValueError("the ontology produced no relations between distinct terms")
    return _ec(np.asarray(src, np.int32), np.asarray(tgt, np.int32), labels,
               types=np.asarray(types, np.int32),
               type_names=[p for p, _ in sorted(preds.items(), key=lambda kv: kv[1])])


def load_ontology_file(path, *, fmt: str | None = None, named: bool = True, **_kw):
    """Reader entry point: an ontology file as an EdgeConstruction."""
    return to_edge_construction(read_ontology(path, fmt), named=named)


def register(register_reader=None) -> None:
    """Register every ontology format with the file-reader registry.

    Called at import of `agent.adapters.formats` so an ontology file is openable by
    the same `read(path)` that opens a `.pdb`.
    """
    if register_reader is None:
        from .formats import register_reader
    for name, (_fn, exts) in PARSERS.items():
        if not exts:
            continue
        register_reader(f"ontology_{name}", load_ontology_file, extensions=exts)
