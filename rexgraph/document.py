"""A document as one field with its layers attached.

The sequence this runs is not new: it is the one the layer design settled on, written
down once instead of being rebuilt at each call site:

    raw text  ->  document_layers        spans per layer, and the method each matched
              ->  from_text              ONE field, branching sentence relations
              ->  add_sectioning         the sentence PARTITION, carrying byte spans
              ->  add_coarsening         paragraph, as a parent map over sentences
              ->  add_coarsening         chapter, as a parent map over paragraphs

Three choices are made here rather than left to the caller, because measurement settled
them and leaving them open would let a caller build something that does not close:

    the PARTITION, not the cover     a pair recurring in two sentences belongs to both,
                                     but only the partition closes exactly (total mass
                                     2150.0000 against rank 2150, where the cover reads
                                     2824.98) and it is 2.4x smaller. The Merkle tree
                                     also REQUIRES it: a leaf needs exactly one parent.
    pair_mode="none"                 a sentence IS one relation over its words. Nothing
                                     is enumerated from it, because the pairs are not in
                                     the text: measured on one book, 1,469 sentences give
                                     1,469 relations at rank 1,437, and adding spanning
                                     pairs gives 12,890 at rank 2,566: 11,421 invented
                                     columns manufacturing 1,129 dimensions of rank and
                                     10,292 of the 10,324 cycles. Nor is it a forest of
                                     stars: shared vocabulary leaves 32 real cycles.
                                     Connectivity and scale come from the field and its
                                     layers, not from pairwise enumeration.
    coarsenings as parent maps       a paragraph owns SENTENCES, not cells. Re-listing
                                     the memberships stored the same 121,877 entries the
                                     sentence layer already had, and could disagree
                                     with it.
    NO owner vertex per sentence     `owner_vertex=True` gives each group a vertex of its
                                     own, which is a hub the text does not contain: star
                                     expansion, and the one thing the model is built not
                                     to need. The sentence's identity is already the
                                     sectioning's label and span, so the vertex names it
                                     twice. It also costs: an owner appears in no pair,
                                     so it is isolated in the pairwise part, the spanned
                                     -branching rank REFUSES, and betti falls back to the
                                     exact reduction: 44.0s a document against 0.0s,
                                     with 1,469 extra vertices and 1,470 components.

`min_terms=1` keeps the WITNESS, which is a cell class and not a failure. A span of one
term has column `(+1)`, sums to one rather than zero, and satisfies `L0 u = u`: it
exists and bounds nothing, which is exactly what a vocative is. "Take away your mother,
Jerry." and "Take away your mother Jerry." differ precisely in whether Jerry is a witness
or the fifth member of a branching relation, so filtering arity 1 turns the first
sentence silently into the second. Two was a filter standing where a class belongs, and
three before it was arbitrary.

Spans address the ORIGINAL bytes throughout, so the text stays an addressable heap and a
section is recovered by one seek rather than a re-parse.
"""
from __future__ import annotations

import unicodedata

import numpy as np

__all__ = ["build_document", "document_sections", "section_text", "read_document"]


def read_document(path, encoding="utf-8"):
    """A file's text exactly as its BYTES decode, and whether spans can address it.

    Never use text-mode `open` for this. Python translates CRLF to LF on read, so the
    decoded string is shorter than the file and every byte offset past the first line
    ending is wrong. Measured on one book: 3,762 CRLF pairs, 174,311 bytes decoding to
    163,950 characters, and the first bad offset at byte 63. Every span after that point
    addressed another sentence, which no ASCII test fixture can catch because it has no
    line endings to translate.

    Returns `(text, exact)`. `exact` is True when the text re-encodes to the file
    byte-for-byte, which is exactly the condition under which a span computed from the
    text addresses the file. A caller that gets False must not publish a heap pointer.
    """
    with open(path, "rb") as fh:
        b = fh.read()
    text = b.decode(encoding, errors="replace")
    return text, bool(text.encode(encoding) == b)


def _byte_starts(text, encoding="utf-8"):
    """Character index -> byte offset, for the whole text at once.

    Spans have to be BYTE offsets or they cannot be seeked to. `document_layers` works on
    the decoded string, which is right for the text logic and wrong for the heap: a text
    handle's `seek` takes an opaque cookie, not a character count, so a character span
    lands mid-codepoint on any file with a multi-byte character and returns a fragment.
    (Measured: 'ng! No, it'll never' out of a book whose only oddity was curly quotes.)

    utf-8 byte length is a function of the codepoint, so the map is vectorised through
    the utf-32 view rather than encoding a slice per span.
    """
    if encoding.lower().replace("_", "-") in ("utf-8", "utf8"):
        cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        blen = (1 + (cp >= 0x80).astype(np.int64) + (cp >= 0x800).astype(np.int64)
                + (cp >= 0x10000).astype(np.int64))
    else:
        blen = np.fromiter((len(c.encode(encoding, errors="replace")) for c in text),
                           dtype=np.int64, count=len(text))
    out = np.zeros(len(text) + 1, dtype=np.int64)
    np.cumsum(blen, out=out[1:])
    return out


def _to_byte_spans(spans, starts):
    """(char_start, char_len) -> (byte_start, byte_len)."""
    return [(int(starts[a]), int(starts[a + n] - starts[a])) for a, n in spans]


def _parent_map(child_spans, parent_spans):
    """Which parent span each child starts inside, by offset. -1 when none precedes it.

    Offsets are used rather than containment because the layers come from ONE
    segmentation of one text: a sentence starts inside exactly the paragraph that most
    recently began, and asking about containment would re-derive that from geometry.
    """
    if not parent_spans:
        return np.full(len(child_spans), -1, dtype=np.int64)
    starts = np.asarray([a for a, _n in parent_spans], dtype=np.int64)
    child = np.asarray([a for a, _n in child_spans], dtype=np.int64)
    return np.searchsorted(starts, child, "right") - 1


def _partition_of(sections, nE):
    """Each cell to its FIRST owning section: the cover collapsed to a partition.

    First-occurrence is not arbitrary. The sections are in document order, so a shared
    pair is charged to where it first appeared, which is the only assignment that does
    not depend on how the rest of the document turned out.
    """
    seen = np.zeros(int(nE), dtype=bool)
    out = {}
    for key, cells in sections.items():
        fresh = [int(c) for c in cells if not seen[c]]
        if not fresh:
            continue
        seen[fresh] = True
        out[key] = fresh
    return out, int((~seen).sum())


def build_document(raw, *, profile=None, encoding=None, min_terms=1, grammar=None,
                   min_pair_count=1, pair_mode="none", stopwords=None,
                   document_vertex=False, layers=("paragraph", "chapter"),
                   verify=False):
    """One document -> one complex carrying its layers. Returns `(rex, info)`.

    `info` records what was built and what was NOT: `methods` names the
    convention each layer matched, and a layer the document does not support is absent
    from it rather than present with a guess.

    `profile` carries what the CORPUS contributes: its encoding, the markers wrapping
    its content, the heading conventions worth trying, the abbreviation veto and the
    grammar that orients. Without one the reading claims nothing: no markers are
    stripped, no chapter convention is tried, and orientation stays positional. That is
    the honest default for an unknown corpus.

    `min_terms` defaults to 1, which keeps witnesses. Raising it deletes a cell class
    rather than filtering noise: see the module docstring.
    """
    from rexgraph.sectioning import add_coarsening, add_sectioning
    from rexgraph.segment import document_layers

    text = str(raw)
    if profile is not None:
        encoding = encoding or getattr(profile, "encoding", "utf-8")
        grammar = grammar if grammar is not None else getattr(profile, "grammar", None)
    encoding = encoding or "utf-8"
    lay = document_layers(text, encoding=encoding, profile=profile)
    sent = lay.get("sentence", {}).get("spans") or []
    if not sent:
        raise ValueError("no sentence span survived segmentation; nothing to construct")
    # every stored span becomes a BYTE range here, once. Segmentation reasons in
    # characters (correct); the heap is addressed in bytes (required).
    bstart = _byte_starts(text, encoding)
    frames_seen = []

    # SPANS are the base relation when the profile gates, and sentences when it does
    # not. A frame governs a CLAUSE: measured, sentence relations here have median arity
    # 12 against a maximum frame arity of 5, so 76% of them could not be oriented at all
    #, not because the rule was wrong but because it was applied a layer too coarse.
    # Gating inside the sentence puts relations in frame range.
    #
    # With an empty gate every sentence yields exactly one span, so this is one code
    # path and the degenerate case IS the previous behaviour. The finer layer is only
    # NAMED "span" when it actually divides something; a layer identical to the one
    # above it is that layer under another name.
    from rexgraph.construct import _orient, from_groups, spans_of
    from rexgraph.corpus_profile import TEXT, tokenize

    prof = profile if profile is not None else TEXT
    gate = set(getattr(prof, "gate", ()) or ())

    def _gates(tok):
        """Punctuation ALWAYS gates, whether or not a profile lists it.

        A mark is the archetypal delimiter: it says where a relation ends and is not
        one of its participants. Letting it through put "." in a relation's support and
        made `Dog chase cat.` arity 4, so a frame of arity 3 could not match it: a
        punctuation vertex is both a false participant and a false argument slot.
        """
        return tok in gate or all(unicodedata.category(c)[0] in ("P", "S")
                                  for c in tok)
    groups, group_sent, group_char = [], [], []
    for si, (a, n) in enumerate(sent):
        toks = tokenize(text[a:a + n], prof)
        if not toks:
            continue
        stream = [t for t, _x, _y in toks]
        gates_here = {t for t in stream if _gates(t)}
        pieces = (spans_of(stream, gates_here) if gates_here else [stream])
        at = 0
        for piece in pieces:
            if not piece:
                continue
            # find this piece's token positions by walking forward, so the span's byte
            # range is the tokens' own extent rather than the sentence's
            idxs = []
            for tok in piece:
                while at < len(toks) and toks[at][0] != tok:
                    at += 1
                if at < len(toks):
                    idxs.append(at); at += 1
            if not idxs:
                continue
            lo = a + toks[idxs[0]][1]
            hi = a + toks[idxs[-1]][2]
            terms = list(dict.fromkeys(piece))
            if len(terms) < int(min_terms):
                continue
            terms, fid = _orient(terms, grammar)
            groups.append(terms); group_sent.append(si); group_char.append((lo, hi - lo))
            frames_seen.append(fid)
    if not groups:
        raise ValueError("no span survived the gate and the filters; nothing to construct")

    rex, cinfo = from_groups(groups, min_pair_count=min_pair_count,
                             owner_vertex=document_vertex, pair_mode=pair_mode,
                             verify=verify)
    doc_frames = list(frames_seen)
    for g, fid in enumerate(doc_frames):
        if fid:
            # the frame is the contextual boundary AROUND the span, not a cell filling
            # it, so it rides the grade-1 column that already exists
            rex.attach_metadata(1, g, "frame", str(fid))

    part, orphans = _partition_of(cinfo["sections"], int(rex.nE))
    order = sorted(part)
    # the layer is named for what it DOES: `span` when it genuinely divides a
    # sentence, `sentence` when it does not. Testing whether an explicit gate exists was
    # the wrong question: punctuation gates with no profile gate at all, so a document
    # divided 3-ways from 1 sentence was still being called the sentence layer.
    base = "span" if len(groups) > len(sent) else "sentence"
    labels = [f"{base[0]}{g}" for g in order]
    base_b = _to_byte_spans([group_char[g] for g in order], bstart)
    add_sectioning(rex, base, {labels[j]: part[g] for j, g in enumerate(order)},
                   grade=1, spans={labels[j]: base_b[j] for j in range(len(order))},
                   method=(f"gate:{len(gate)}+punct" if base == "span"
                           else lay["sentence"]["method"]))

    info = {
        "n_sentences": len(sent), "n_spans": len(order), "n_dropped": 0,
        "orphan_cells": orphans, "pair_mode": pair_mode, "base_layer": base,
        "methods": {base: (f"gate:{len(gate)}+punct" if base == "span"
                           else lay["sentence"]["method"]),
                    "document": lay["document"]["method"]},
        "vocab": cinfo.get("vocab") or cinfo.get("members") or [],
        "n_wide": int(cinfo.get("n_wide", 0)), "n_pairs": int(cinfo.get("n_pairs", 0)),
        "layers": [base], "encoding": encoding, "span_units": "bytes",
        "frames": doc_frames,
        "n_oriented_by_frame": sum(1 for x in doc_frames if x),
        "spans": {base: list(base_b)},
    }

    # the sentence layer, when spans divide it, is a coarsening over them
    finer, finer_spans = base, list(base_b)
    if base == "span":
        par = np.asarray([group_sent[g] for g in order], dtype=np.int64)
        used, par = np.unique(par, return_inverse=True)
        sent_b_used = _to_byte_spans([sent[u] for u in used], bstart)
        add_coarsening(rex, "sentence", "span", par, [f"s{u}" for u in used],
                       spans=sent_b_used, method=lay["sentence"]["method"])
        info["layers"].append("sentence")
        info["methods"]["sentence"] = lay["sentence"]["method"]
        info["spans"]["sentence"] = sent_b_used
        finer, finer_spans = "sentence", sent_b_used

    # each coarser layer is a parent map over the one below it, never over the cells
    for name in layers:
        got = lay.get(name)
        if not got or not got.get("spans"):
            continue
        got_b = _to_byte_spans(got["spans"], bstart)
        par = _parent_map(finer_spans, got_b)
        if (par < 0).any():
            # front matter precedes the first heading. Charging it to a chapter that has
            # not started would be an invention; charging it to the first one is a
            # convention, and it is stated here rather than hidden.
            par = np.maximum(par, 0)
        used, par = np.unique(par, return_inverse=True)
        add_coarsening(rex, name, finer, par, [f"{name[0]}{u}" for u in used],
                       spans=[got_b[u] for u in used],
                       method=got["method"])
        info["layers"].append(name)
        info["methods"][name] = got["method"]
        info["spans"][name] = [got_b[u] for u in used]
        finer, finer_spans = name, [got_b[u] for u in used]
    return rex, info


def document_sections(rex, layer="sentence"):
    """`{label: [cell ids]}` for a layer, resolving a coarsening against its finer one."""
    from rexgraph.sectioning import sectionings_of
    store = sectionings_of(rex)
    if layer not in store:
        raise ValueError(f"{layer!r} is not a layer of this document; have "
                         f"{sorted(store)}")
    return store[layer].as_sections(store)


def section_text(rex, layer, index, raw=None, *, path=None, encoding="utf-8"):
    """The prose of one section. No re-parse, no stored copy.

    Spans are BYTE ranges into the original file, so `path` is opened in binary and the
    read is one seek. A text handle cannot be used: its `seek` takes an opaque cookie
    rather than an offset, and a character count lands mid-codepoint.

    `raw` may be `bytes` (sliced directly) or `str` (encoded first, which costs one pass
    over the document, so pass bytes or a path in a loop).
    """
    from rexgraph.sectioning import sectionings_of
    s = sectionings_of(rex).get(str(layer))
    if s is None or s.spans is None:
        raise ValueError(f"{layer!r} carries no spans, so its text is not addressable")
    a, n = (int(x) for x in s.spans[int(index)])
    if raw is not None:
        buf = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode(encoding)
        return bytes(buf[a:a + n]).decode(encoding, errors="replace")
    if path is None:
        raise ValueError("give raw= or path=: a span addresses bytes that live somewhere")
    with open(path, "rb") as fh:
        fh.seek(a)
        return fh.read(n).decode(encoding, errors="replace")
