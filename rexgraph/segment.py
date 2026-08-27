"""Document layers: chapters, paragraphs and sentences, decided by agreement.

A period is not a boundary. It ends sentences, but it also ends "Mr.", "vol. II", "e.g.",
an initial, a version number and a typo, and in slang or a transcribed dialect it may not
appear where a sentence plainly ends. So the terminator GATES and the lexicon VETOES: a
period that belongs to "Dr." or an initial is part of the token, which is a recorded
fact. Nothing is voted on. An earlier version summed five channels and cut where two
agreed, which put a threshold on a count inside an existence decision, and the channels
were not independent evidence anyway, since spacing and case are consequences of a
boundary rather than witnesses to one.

Byte energy is one of those channels, and it is read through the ENCODING, exactly. In
utf-8 an ASCII letter costs 1 byte, Greek and Cyrillic 2, CJK 3, so mathematical
notation set into English prose is a run of width 2 inside a run of width 1, and a symbol
run is visible from the encoding alone with no lexicon. The channel fires where the width
CHANGES, which is a discrete fact about the encoding rather than a level anything is
compared against. An earlier version divided the energy by the text's own median and fired
above 1.0; that put a statistic and a cutoff where an exact integer was already available.

What a layer cannot resolve, it does not invent. A book with no chapter headings gets no
chapter sectioning rather than a guessed one, and every layer records the method it
matched, so a reader can tell a chapter split that found real headings from one that fell
back. That distinction matters downstream: the two support different claims, and a
retrieval that reads more context than it needed is a cost, while one that trusts a
fabricated boundary is an error.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from rexgraph.corpus_profile import as_text as _as_text

__all__ = ["encoding_energy", "encoding_width", "boundary_signals",
           "segment_sentences", "segment_paragraphs", "segment_chapters",
           "strip_markers", "document_layers"]

# The English-and-Gutenberg constants belong to the corpus: the inserted markers,
# the chapter conventions and the abbreviation veto: are now on
# `rexgraph.corpus_profile.ENGLISH_GUTENBERG`. They are facts about one corpus in one
# language, and a segmenter that carries them cannot serve a Chinese corpus or a source
# tree without being edited.


def strip_markers(raw, markers=(), *, encoding="utf-8"):
    """The body between a corpus's inserted markers, and which were found.

    `markers` is `(start_pattern, end_pattern)` from the profile. Project Gutenberg's
    wrapper is a GUTENBERG fact, not a property of text, so it lives on a profile and
    this function knows only that some corpora wrap their content.

    Returns `(body, offset, method)` with `offset` the body's start in `raw`, so every
    span computed downstream stays addressable against the ORIGINAL file. That is the
    pointer layer's contract: a section's byte range has to mean something in the file on
    disk, not in a cleaned copy that no longer exists.
    """
    text = _as_text(raw, encoding)
    start, end, how = 0, len(text), []
    pats = tuple(markers or ())
    if pats:
        m = re.search(pats[0], text)
        if m:
            start = (text.index("\n", m.end()) + 1 if "\n" in text[m.end():]
                     else m.end())
            how.append("start_marker")
        if len(pats) > 1:
            m = re.search(pats[1], text, )
            if m and m.start() >= start:
                end = m.start()
                how.append("end_marker")
    return text[start:end], start, "+".join(how) or "no_markers"


def encoding_energy(text, *, encoding="utf-8"):
    """Per-character energy under `encoding`: `sum (byte * position)^2` over its bytes.

    Position is 1-based within the character's own byte run, matching
    `partition.byte_energy`, so a character's energy is a property of how the ENCODING
    spends bytes on it. That is the whole point: the same character has a different
    energy in utf-8 than in a CJK encoding, and comparing across encodings is meaningless
    without saying which one.
    """
    s = _as_text(text)
    out = np.zeros(len(s), dtype=np.float64)
    for i, ch in enumerate(s):
        try:
            bs = ch.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            bs = ch.encode("utf-8", errors="replace")
        out[i] = sum((b * (j + 1)) ** 2 for j, b in enumerate(bs))
    return out


def encoding_width(text, *, encoding="utf-8"):
    """Bytes per character under `encoding`: an exact integer, one per position.

    This is the discrete carrier the energy rides on, and it is what a script change
    actually IS. In utf-8 an ASCII letter is 1 byte, Greek and Cyrillic 2, CJK 3, an
    astral codepoint 4, so mathematical notation set into English prose is exactly a
    run of width 2 inside a run of width 1, and its boundary is exactly where the width
    CHANGES. Nothing is compared against a level.

    This replaces a median-normalised energy ratio. That version divided by the text's
    own median and fired above 1.0, which put a statistic and a cutoff in a decision
    path: the median is not a property of the encoding or of the text's structure, and
    "above the middle" is a level rather than a fact. Byte width is exact, per-encoding,
    and needs neither.
    """
    s = _as_text(text)
    if encoding.lower().replace("_", "-") in ("utf-8", "utf8"):
        cp = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
        return (1 + (cp >= 0x80).astype(np.int64) + (cp >= 0x800).astype(np.int64)
                + (cp >= 0x10000).astype(np.int64))
    return np.fromiter((len(c.encode(encoding, errors="replace")) for c in s),
                       dtype=np.int64, count=len(s))


def boundary_signals(text, *, encoding="utf-8", abbreviations=None):
    """The gate, and the readings that sit beside it, at each character position.

    ONE of these decides: `terminator`, which is the delimiter, gating where a relation
    ends. `suppress` can veto it from the lexicon. The rest are READINGS reported at the
    same positions and are not weighed into the decision: spacing and case are
    consequences of a boundary rather than witnesses to one, and `energy` is a script
    change, which is a boundary of a different kind and does not end a sentence.

        terminator   `.!?` (and the CJK/Arabic equivalents) followed by space
        spacing      a newline, or a run of two or more spaces
        case         the next word-initial character is upper case or a digit
        quote        a closing quote or bracket sits between the mark and the space
        energy       the relative energy drops back to baseline after being above it,
                     which is where a symbol or foreign-script run ENDS

    and one suppressor:

        abbrev       the token before the mark has an abbreviation shape, so the period
                     is part of the token. This can only veto.

    Returns `(signals, suppress)` where `signals` is `{name: bool array}` over character
    positions and `suppress` is a bool array. Nothing is combined here on purpose: the
    caller decides how much agreement it wants, and gets to see which channels fired.
    """
    s = _as_text(text)
    n = len(s)
    term = np.zeros(n, dtype=bool)
    spacing = np.zeros(n, dtype=bool)
    case = np.zeros(n, dtype=bool)
    quote = np.zeros(n, dtype=bool)
    energy = np.zeros(n, dtype=bool)
    suppress = np.zeros(n, dtype=bool)
    if n == 0:
        return ({"terminator": term, "spacing": spacing, "case": case,
                 "quote": quote, "energy": energy}, suppress)

    width = encoding_width(s, encoding=encoding)

    closers = "\"'”’)]}»"
    terminators = ".!?。！？؟۔।"

    # EVERY channel is indexed by the same candidate position, which is the position the
    # cut would happen at. Letting each fire where its own evidence sits (the mark, the
    # following space, the next capital) puts them at different indices, and channels at
    # different indices can never agree, so the agreement rule silently accepted nothing.
    for i, ch in enumerate(s):
        # a change in byte WIDTH is a script change, exactly. Notation set into prose,
        # a transliteration or a formula begins and ends where the encoding starts and
        # stops spending more bytes, and that is a discrete fact rather than a level.
        if i and width[i] != width[i - 1] and not s[i].isspace():
            energy[i - 1] = True
        if ch not in terminators:
            continue
        j = i + 1
        while j < n and s[j] in closers:
            j += 1
        if j < n and not s[j].isspace():
            continue                       # a decimal point, a version number, a URL
        term[i] = True
        if j > i + 1:
            quote[i] = True
        k = j
        while k < n and s[k].isspace():
            k += 1
        gap = s[j:k]
        if "\n" in gap or len(gap) >= 2:
            spacing[i] = True
        if k < n:
            cat = unicodedata.category(s[k])
            if cat in ("Lu", "Lt", "Nd") or s[k] in "\"'“‘":
                case[i] = True
        else:
            case[i] = True                 # end of text closes whatever is open
        # the abbreviation veto reads the token that owns the mark. It can only VETO.
        if abbreviations is not None and abbreviations.search(s[max(0, i - 12):i + 1]):
            suppress[i] = True
    return ({"terminator": term, "spacing": spacing, "case": case,
             "quote": quote, "energy": energy}, suppress)


def segment_sentences(text, *, encoding="utf-8", offset=0, abbreviations=None):
    """Sentence spans, GATED by the terminator rather than voted on.

    A delimiter gates existence: it says where a relation ends. That is a fact about the
    text, not evidence to be weighed, so nothing here counts signals or requires several
    to agree. The previous version summed five channels and cut at `min_agreement=2`,
    which put a threshold on a count in the middle of an existence decision, and the
    channels it was weighing were not independent evidence about the same thing:
    spacing and case are consequences of a boundary, not witnesses to one, and byte
    width is a SCRIPT change, which is a different kind of boundary altogether.

    What the gate does need is the lexicon, because a period is part of the token in
    "Dr.", "vol." and an initial. That is a recorded lexical fact and it can only VETO:
    it never creates a boundary. `abbreviations` is the pattern, so a corpus or a
    language can supply its own without changing what a boundary means.

    Returns `(spans, method)` as `(start, length)` CHARACTER ranges shifted by `offset`.
    """
    s = _as_text(text, encoding)
    if not s.strip():
        return [], "empty"
    sig, suppress = boundary_signals(s, encoding=encoding,
                                     abbreviations=abbreviations)
    cut = sig["terminator"] & ~suppress
    idx = np.flatnonzero(cut)

    spans, start = [], 0
    for i in idx.tolist():
        end = i + 1
        while end < len(s) and s[end] in "\"'\u201d\u2019)]}\u00bb":
            end += 1
        if s[start:end].strip():
            spans.append((start + offset, end - start))
        start = end
        while start < len(s) and s[start].isspace():
            start += 1
    if s[start:].strip():
        spans.append((start + offset, len(s) - start))
    return spans, "terminator_gate"


def segment_paragraphs(text, *, offset=0):
    """Paragraph spans on blank lines, which IS reliable in plain text.

    Unlike a sentence, a paragraph break is written down by the author as a blank line
    and does not have to be inferred, so this needs no agreement rule. Where a file has
    no blank lines at all the whole body is one paragraph, reported as such.
    """
    s = _as_text(text)
    # `\r?\n` on BOTH sides: a CRLF file separates paragraphs with "\r\n\r\n",
    # which an LF-only pattern cannot match, so every such document read as one
    # block. `read_document` deliberately does not translate newlines (that is what
    # shifts byte spans), so the pattern has to.
    parts = re.split(r"\r?\n[ \t]*\r?\n", s)
    if len(parts) <= 1:
        body = s.strip()
        if not body:
            return [], "empty"
        st = s.index(body[0]) if body else 0
        return [(st + offset, len(body))], "single_block"
    spans, method, pos = [], "blank_line", 0
    for part in parts:
        st = s.index(part, pos) if part else pos
        if part.strip():
            spans.append((st + offset, len(part)))
        pos = st + len(part)
    return spans, method


def segment_chapters(text, *, offset=0, min_sections=2, headings=()):
    """Chapter spans, or NOTHING when no convention matches.

    `min_sections=2` is not a tuned number: a layer that yields ONE section is not a
    partition of anything, it is the document again under another name. Two is where a
    division starts to exist.

    Returns `(spans, titles, method)`. A book without headings returns `([], [], "none")`
    and gets no chapter layer, which is the honest outcome: inventing chapter boundaries
    from length or blank-line counts would produce divisions no reader could confirm and
    that no two books would agree on. Retrieval then reads a paragraph where it could
    have read a chapter, which costs context and does not cost correctness.
    """
    s = _as_text(text)
    for name, pat in (headings or ()):
        rx = pat if hasattr(pat, "finditer") else re.compile(pat, re.M)
        marks = [(m.start(), m.group(0).strip()) for m in rx.finditer(s)]
        if len(marks) < int(min_sections):
            continue
        # a convention that matches must also PARTITION: headings in the running text
        # rather than at breaks are a false match, so require they sit at line starts
        # and are separated by real content
        spans, titles = [], []
        for k, (st, title) in enumerate(marks):
            end = marks[k + 1][0] if k + 1 < len(marks) else len(s)
            if s[st:end].strip():
                spans.append((st + offset, end - st))
                titles.append(title)
        if len(spans) >= int(min_sections):
            return spans, titles, name
    return [], [], "none"


def document_layers(raw, *, encoding="utf-8", profile=None):
    """Every layer a document actually supports, as spans into the ORIGINAL text.

    Returns `{layer: {"spans": [(start, length)], "method": str, "titles": [...]}}`,
    with a layer ABSENT when the document does not support it. `method` is carried
    throughout so a reader can tell what was matched from what was assumed.
    """
    markers = getattr(profile, "markers", ()) if profile is not None else ()
    headings = getattr(profile, "headings", ()) if profile is not None else ()
    veto = getattr(profile, "veto", None) if profile is not None else None
    body, off, pg = strip_markers(raw, markers, encoding=encoding)
    out = {"document": {"spans": [(off, len(body))], "method": pg}}
    ch, titles, how = segment_chapters(body, offset=off, headings=headings)
    if ch:
        out["chapter"] = {"spans": ch, "method": how, "titles": titles}
    pa, how = segment_paragraphs(body, offset=off)
    if pa:
        out["paragraph"] = {"spans": pa, "method": how}
    se, how = segment_sentences(body, offset=off, encoding=encoding,
                                abbreviations=veto)
    if se:
        out["sentence"] = {"spans": se, "method": how}
    return out
