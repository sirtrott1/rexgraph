"""Lexical resources, read into the shape the complex actually wants.

The mapping is not a convention, it follows from what each resource IS:

    WordNet synset     a set of synonymous lemmas. That is a GROUP, so it is one
                       branching relation over its members, and the synset relations
                       (hypernym, antonym, ...) are ordinary oriented 2-ary relations
                       between synsets.
    Roget category     the same shape one level coarser: a numbered heading over the
                       terms filed under it.
    NRC VAD / EmoLex   NOT relations. They are one value per word, so they are
                       0-COCHAINS on the word vertices. Reading them as relations would
                       invent structure the lexicon does not assert.

So two of the three feed `rexgraph.construct.from_groups` and the third feeds whatever
consumes a vertex-indexed field. Nothing here builds a complex; these return the data and
the caller chooses the construction, because the same synset inventory supports several
and picking one here would hide that choice.

Parsers are stdlib only, matching `ontology_formats`: no rdflib, no nltk, no `wn`.
"""
from __future__ import annotations

import gzip
import os
import re
import xml.etree.ElementTree as ET

__all__ = ["load_wordnet", "load_roget", "load_nrc_vad", "load_nrc_emolex",
           "wordnet_groups", "frame_slots"]

#: the role fillers a subcategorisation frame can name. Recorded vocabulary, not a
#: guess: these are the only fillers WordNet's 39 frames use.
_FRAME_SLOTS = ("somebody's (body part)", "somebody", "something", "adjective/noun",
                "adjective", "infinitive", "clause", "verb-ing", "v-ing", "pp", "it")


def _open(path):
    """Text handle for a plain or gzipped file, decided by content not by name."""
    p = os.path.expanduser(str(path))
    with open(p, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")


#### WordNet ###################################################################

def load_wordnet(path, *, with_examples=False):
    """Open English WordNet / any WN-LMF lexicon.

    Streams with `iterparse` and clears as it goes, so a 120k-synset lexicon is read in
    bounded memory rather than held as a tree.

    The GRAMMAR comes out of here too, and it is recorded fact rather than anything
    inferred. `SyntacticBehaviour` declares the subcategorisation frames (39 in English
    WordNet 2024) and each `Sense` names the ones it admits via `subcat` (25,100 of
    them). A frame IS an oriented k-ary relation over argument roles with the verb
    heading it: "Somebody ----s somebody something" is arity 4, the verb carries the
    -1 and agent, recipient and theme share `1/(k-1)`. That is the recipient structure
    a dependency parser approximates, stated exactly, so it belongs in the database
    beside the lexicon and the semantics rather than being tagged back on afterwards.

    Returns a dict::

        lemma_of     {entry_id: written form}
        pos_of       {entry_id: part of speech}
        synsets      {synset_id: [entry_id, ...]}      the GROUPS
        relations    [(src_synset, relType, tgt_synset)]
        sense_of     {sense_id: (entry_id, synset_id)}
        definitions  {synset_id: str}
        examples     {synset_id: [str]}                only when asked
        frames       {frame_id: subcategorization string}          the GRAMMAR
        frames_of    {sense_id: [frame_id, ...]}       which each sense admits
        forms_of     {entry_id: [written form, ...]}   recorded inflections
    """
    lemma_of, pos_of, synsets, relations = {}, {}, {}, []
    sense_of, definitions, examples = {}, {}, {}
    frames, frames_of, forms_of = {}, {}, {}
    with _open(path) as fh:
        for _ev, el in ET.iterparse(fh, events=("end",)):
            tag = el.tag.split("}")[-1]
            if tag == "LexicalEntry":
                eid = el.get("id")
                lem = el.find("{*}Lemma") if el.find("{*}Lemma") is not None \
                    else el.find("Lemma")
                if eid and lem is not None:
                    lemma_of[eid] = lem.get("writtenForm")
                    pos_of[eid] = lem.get("partOfSpeech")
                # `Form` records an inflected written form of this entry. A document
                # has inflected text and the lexicon is keyed on lemmas, so without
                # these a frame lookup misses every "chased" and "gave". 4,474 are
                # recorded, which is the irregulars; a regular inflection not listed
                # simply makes no claim, which is the honest outcome rather than a
                # stemming rule guessing one.
                for f in el.iter():
                    if f.tag.split("}")[-1] == "Form" and f.get("writtenForm"):
                        forms_of.setdefault(eid, []).append(f.get("writtenForm"))
                for s in el.iter():
                    if s.tag.split("}")[-1] == "Sense" and s.get("id"):
                        sense_of[s.get("id")] = (eid, s.get("synset"))
                        # `subcat` is a space-separated list of frame ids: the frames
                        # this sense admits. Several is not ambiguity to resolve here,
                        # it is the sense genuinely taking more than one argument shape.
                        sub = (s.get("subcat") or "").split()
                        if sub:
                            frames_of[s.get("id")] = sub
                el.clear()
            elif tag == "SyntacticBehaviour":
                fid = el.get("id")
                if fid:
                    frames[fid] = el.get("subcategorizationFrame") or ""
                el.clear()
            elif tag == "Synset":
                sid = el.get("id")
                if sid:
                    members = (el.get("members") or "").split()
                    synsets[sid] = members
                    for c in el:
                        ct = c.tag.split("}")[-1]
                        if ct == "SynsetRelation" and c.get("target"):
                            relations.append((sid, c.get("relType"), c.get("target")))
                        elif ct == "Definition" and c.text:
                            definitions.setdefault(sid, c.text.strip())
                        elif with_examples and ct == "Example" and c.text:
                            examples.setdefault(sid, []).append(c.text.strip())
                el.clear()
    return {"lemma_of": lemma_of, "pos_of": pos_of, "synsets": synsets,
            "relations": relations, "sense_of": sense_of,
            "definitions": definitions, "examples": examples,
            "frames": frames, "frames_of": frames_of, "forms_of": forms_of}


def wordnet_groups(wn, *, by="lemma", min_size=2, include_relations=True):
    """WordNet as groups, ready for `rexgraph.construct.from_groups`.

    `by="lemma"` names members by their written form, so two entries spelled the same
    are ONE vertex and the synsets that share a spelling share a boundary. `by="entry"`
    keeps them apart. Which is right depends on whether the reading is about words or
    about senses, so it is a parameter and not a default buried in a parser.

    `min_size=2` drops singleton synsets, which are a group of one and bound nothing.
    `include_relations` adds each synset relation as a 2-ary group over the two synsets'
    members, which is what carries hypernymy into the same complex.

    Returns `(groups, labels)` where `labels[i]` says what group `i` came from.
    """
    name = (lambda e: wn["lemma_of"].get(e, e)) if by == "lemma" else (lambda e: e)
    groups, labels = [], []
    for sid, members in wn["synsets"].items():
        g = list(dict.fromkeys(name(m) for m in members if m))
        if len(g) >= int(min_size):
            groups.append(g); labels.append(("synset", sid))
    if include_relations:
        for src, rel, tgt in wn["relations"]:
            a = wn["synsets"].get(src) or []
            b = wn["synsets"].get(tgt) or []
            if not a or not b:
                continue
            # the relation is between SYNSETS, so it is carried by one representative
            # member of each: adding every cross pair would be clique expansion
            g = list(dict.fromkeys([name(a[0]), name(b[0])]))
            if len(g) >= 2:
                groups.append(g); labels.append((rel or "related", f"{src}->{tgt}"))
    return groups, labels


#### Roget ######################################################################

#: a Roget heading: "#123. Name.—N. term, term; ..." with the number and name up front.
_ROGET_HEAD = re.compile(r"^#\s*(\d+)\.\s*(.+?)\.?\s*(?:—|--|—)", re.UNICODE)
_ROGET_ANY = re.compile(r"^#\s*(\d+)\.\s*(.*)$")


def load_roget(path, *, min_terms=2):
    """Roget's Thesaurus from the Project Gutenberg plain text (#22).

    The file is prose with numbered headings, so this is a segmentation and not a schema
    read: a heading opens a category and everything up to the next heading is its body.
    Terms are split on the separators Roget actually uses, and the cross-reference tails
    ("&c. 494") are dropped because they point at a category rather than naming a term.

    Returns `{number: {"name": str, "terms": [str, ...]}}`.
    """
    cats, num, name, body = {}, None, None, []

    def _flush():
        if num is None:
            return
        text = " ".join(body)
        # "&c. 494" points at a category and "&c. adj." names a part of speech; neither
        # is a term, and leaving the tail on produced entries like "positiveness adj"
        text = re.sub(r"&c\.?\s*(?:adj|adv|n|v|int|phr)?\.?\s*\d*", " ", text,
                      flags=re.IGNORECASE)
        text = re.sub(r"_[^_]*_", " ", text)              # italic gloss markers
        text = re.sub(r"\[[^\]]*\]", " ", text)
        # a sentence period ends a term list as surely as a semicolon does; without it
        # "subsistence.  reality" came back as one term
        parts = re.split(r"[;,.]", text)
        terms = []
        for t in parts:
            # a leading "N." / "V." / "Adj." is the part-of-speech heading Roget puts in
            # front of each run, not the first word of the first term
            t = re.sub(r"^\s*(?:N|V|Adj|Adv|Int|Phr)\.\s*", " ", t, flags=re.IGNORECASE)
            t = re.sub(r"[^A-Za-z' -]", " ", t).strip().lower()
            t = re.sub(r"\s+", " ", t)
            if t and 1 <= len(t.split()) <= 4 and len(t) > 1:
                terms.append(t)
        terms = list(dict.fromkeys(terms))
        if len(terms) >= int(min_terms):
            cats[num] = {"name": name, "terms": terms}

    with _open(path) as fh:
        for line in fh:
            m = _ROGET_HEAD.match(line) or _ROGET_ANY.match(line)
            if m:
                _flush()
                num, name = int(m.group(1)), (m.group(2) or "").strip(" .")
                body = [line[m.end():]]
            elif num is not None:
                body.append(line)
    _flush()
    return cats


#### NRC: values on words, not relations ########################################

def load_nrc_vad(path):
    """NRC VAD: `word \\t valence \\t arousal \\t dominance`.

    A 0-cochain on the word vertices, three channels wide. Returns `{word: (v, a, d)}`
    with floats in [0, 1].
    """
    out = {}
    with _open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            try:
                out[p[0].strip().lower()] = (float(p[1]), float(p[2]), float(p[3]))
            except ValueError:
                continue                                   # the header row, if present
    return out


def load_nrc_emolex(path):
    """NRC EmoLex word level: `word \\t emotion \\t 0|1`, one row per (word, emotion).

    Returns `{word: {emotion: bool}}`, keeping only the emotions marked present, so a
    word with no association is absent rather than carrying ten zeros.
    """
    out = {}
    with _open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            w, emo, val = p[0].strip().lower(), p[1].strip(), p[2].strip()
            if val == "1":
                out.setdefault(w, {})[emo] = True
            else:
                out.setdefault(w, {})
    return out


def frame_slots(frame: str):
    """The argument slots a subcategorisation frame names, in order.

    A frame IS an oriented relation and this is what fixes its arity: the verb heads it
    and each slot is a participant, so `k = 1 + len(slots)` and the share is `1/(k-1)`
    over the slots. "Somebody ----s somebody something" gives three slots, k=4, and each
    argument carries 1/3: agent, recipient and theme share equally because the boundary
    column says so, not because a weighting was chosen.

    Matching is by TOKEN, not by substring. Scanning for fillers inside the string found
    "it" inside "with" and read `vtaa-with` as arity 5, which would have understated every
    share in the frame. A filler the recorded vocabulary does not name is returned in
    `unknown` rather than dropped, because an unseen slot understates the arity and
    therefore overstates every share.

    Returns `(slots, unknown)`.
    """
    text = str(frame or "")
    if "----" not in text:
        return [], []
    #: grammatical glue, which marks a slot without being one
    glue = {"to", "of", "on", "at", "for", "with", "from", "into", "out", "whether",
            "is", "that", "a", "an", "the"}
    low = re.sub(r"----\w*", " \u0000 ", text.lower())          # the verb, marked
    low = low.replace("somebody's (body part)", "somebody's_body_part")
    slots, unknown = [], []
    for tok in low.split():
        if tok == "\u0000" or tok in glue:
            continue
        if tok in _FRAME_SLOTS or tok == "somebody's_body_part":
            slots.append(tok.replace("somebody's_body_part", "somebody's (body part)"))
        else:
            unknown.append(tok)
    return slots, unknown
