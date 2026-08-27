"""What a corpus contributes, separated from what the math does.

(The module is `corpus_profile`, not `profile`: the stdlib has a `profile` module and
`cProfile` imports it by name, so a top-level `profile.py` anywhere on `sys.path`
shadows it. torch._dynamo imports cProfile, so 17 torch tests failed with
`module 'profile' has no attribute 'run'`, and only in the full suite, because that is
when pytest puts the package directory on the path.)

The relational construction is corpus-independent and always was: delimiters gate
existence, arity classes the result into witness / pairwise / branching, the share is
`1/(k-1)`, the head carries the `-1`, layers partition one field, and the lookup is
diffusion over that partition. None of that changes for Chinese, for Rust, or for an
archive whose files disagree about their encoding.

What DOES change is a small set of recorded facts, and they were sitting in the library
as constants. `_TOKEN = r"[a-z']+"` is the one that mattered: measured, it returns SIX
tokens for English and ZERO for Greek, Russian, Chinese and Arabic, so four corpora out
of seven built an empty complex and nothing raised. It also shreds `build_document` into
`build` and `document`, which dissolves an identifier's identity for the same reason
clique expansion dissolves a relation's.

A profile holds those facts. Every field is something a corpus RECORDS rather than
something tuned:

    encoding    per file, not per corpus, because a mixed archive is mixed file by file.
                `document.read_document` already reports whether the text re-encodes to
                the bytes, which is the condition for a span to address the heap.
    token_rule  what a vertex is. "script" for natural language, "identifier" for code.
    casefold    whether case is noise. It is in prose and is not in source.
    gate        which tokens gate existence. Function-POS classes for a language, the
                language's own punctuation for code.
    veto        what stops a gate firing: abbreviations, or a string literal in code.
    layers      the coarsening chain, in order.
    markers     wrapper the corpus inserts around its own content, if any.
    headings    the section conventions to try, each named so a match can be reported.

Programming languages are the EASY case here, not a second implementation. English needs
frames because its grammar is only partly recorded (25,100 WordNet senses carry one and
the rest do not) while a programming language's grammar is complete and executable, so
every profile field is exact and no convention has to be guessed at.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

__all__ = [
    "as_text", "CorpusProfile", "tokenize", "TEXT", "ENGLISH_GUTENBERG", "PYTHON_SOURCE",
           "is_scriptio_continua"]

#: Scripts written without word separators, so their natural unit is the CHARACTER
#: rather than the space-delimited run. This is a property of the script recorded by
#: Unicode, not a judgement: a `\\w+` rule returns one token for a whole Chinese
#: sentence, which is a single vertex where there should be many.
_CONTINUA = (
    (0x2E80, 0x2EFF), (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF),
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),      # CJK, kana
    (0x0E00, 0x0E7F), (0x0E80, 0x0EFF),                        # Thai, Lao
    (0x0F00, 0x0FFF), (0x1000, 0x109F), (0x1780, 0x17FF),      # Tibetan, Myanmar, Khmer
    (0x20000, 0x2A6DF), (0x2A700, 0x2EBEF),                    # CJK extensions
)


def is_scriptio_continua(ch: str) -> bool:
    """Whether `ch` belongs to a script written without word separators."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CONTINUA)


#: a word in a space-separated script: letters and marks of ANY script, plus the
#: joiners that sit inside words. `\\w` is Unicode-aware in Python 3, which is the whole
#: fix: the old `[a-z']+` was ASCII-only and silently produced nothing elsewhere.
_WORD = re.compile(r"[^\W\d_][\w'’­-]*", re.UNICODE)
#: characters that sit INSIDE a word without being letters: the apostrophes English
#: needs, the hyphen, and the joiners Indic and Arabic shaping use.
_JOINERS = frozenset(["'", "\u2019", "\u02bc", "-", "\u200d", "\u200c"])


def _is_word_char(ch):
    """Letter, MARK, or joiner: read from the Unicode CATEGORY, not from a class.

    A `\\w`-based rule excludes combining marks, and Indic scripts write their vowels as
    marks (Mn/Mc). Measured: it broke Hindi at every matra and returned eight fragments
    for four words. Arabic points, Hebrew niqqud and Thai tone marks fail the same way,
    so the category is what has to be read rather than a class that happens to fit Latin.
    """
    if ch in _JOINERS:
        return True
    return unicodedata.category(ch)[0] in ("L", "M")


#: an identifier in source: kept WHOLE. Splitting `build_document` into two vertices
#: dissolves the name the way clique expansion dissolves a relation.
_IDENT = re.compile(r"[^\W\d]\w*", re.UNICODE)
_NUMBER = re.compile(r"\d[\w.]*", re.UNICODE)
#: source punctuation is a token in its own right, because in code it GATES
_OPERATOR = re.compile(r"[(){}\[\];,:.=+\-*/%<>!&|^~@#?]+")


@dataclass(frozen=True)
class CorpusProfile:
    """The recorded facts one corpus contributes. Nothing here is tuned."""

    name: str
    encoding: str = "utf-8"
    #: "script" (natural language, script-aware) or "identifier" (source code)
    token_rule: str = "script"
    #: fold case when tokenising. TRUE for prose, where a sentence-initial capital is
    #: noise and `The` and `the` are the same word; FALSE for source, where `Foo` and
    #: `foo` are different identifiers and folding them merges two vertices that the
    #: language keeps apart.
    casefold: bool = True
    #: tokens that GATE existence. Blind to orientation by construction.
    gate: frozenset = frozenset()
    #: a pattern whose match stops a gate firing. It can only VETO, never create.
    veto: object = None
    layers: tuple = ("sentence", "paragraph", "chapter")
    #: (start, end) patterns for wrapper the corpus inserts around its content
    markers: tuple = ()
    #: ((name, pattern), ...) section conventions, tried in order
    headings: tuple = ()
    #: which participant HEADS a relation. Anything with `head_of(tokens) -> (index,
    #: label) | None` implements it: WordNet's subcategorisation frames for English, a
    #: parser for a programming language. None means orientation stays positional, which
    #: is the approximation the grammar exists to replace.
    grammar: object = None
    #: free-form, for anything a reader needs and the math does not
    extra: dict = field(default_factory=dict)


def as_text(value, encoding="utf-8"):
    """Text from either bytes or str.

    `str(b"...")` is the repr, so a document handed in as bytes is read as its own
    escape sequences: `\r` and `\n` become the characters `r` and `n`, and the first
    token of a file becomes `b'the`. Every entry point that takes text can be handed a
    file's bytes, so each one decodes through here.
    """
    if isinstance(value, (bytes, bytearray)):
        return value.decode(encoding, "replace")
    return str(value)


def tokenize(text, profile: CorpusProfile):
    """`[(token, start, end)]` with CHARACTER offsets, so spans stay addressable.

    Positions are returned because every layer above this addresses the source by span;
    a tokenizer that returns only strings forces the caller to re-find them, which is the
    re-parse the heap design exists to avoid.

    Under `"script"`, a run of a space-separated script is one token and a character of a
    scriptio-continua script is one token. That is a property of the script rather than a
    segmentation decision: Chinese has no spaces, so a `\\w+` rule would make one vertex
    of a whole sentence.

    Under `"identifier"`, an identifier stays whole and each operator run is its own
    token, because in source the punctuation is what gates.
    """
    s = as_text(text, getattr(profile, "encoding", "utf-8"))
    fold = bool(getattr(profile, "casefold", True))
    veto = getattr(profile, "veto", None)
    out = []
    if profile.token_rule == "identifier":
        pos = 0
        while pos < len(s):
            for pat in (_IDENT, _NUMBER, _OPERATOR):
                m = pat.match(s, pos)
                if m:
                    tok = m.group(0)
                    out.append((tok.lower() if fold else tok, m.start(), m.end()))
                    pos = m.end()
                    break
            else:
                pos += 1
        return out

    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if is_scriptio_continua(ch):
            # the unit is a GRAPHEME CLUSTER, not a codepoint: a base character plus the
            # marks that belong to it. Thai and Khmer write vowels and tones as combining
            # marks, so a per-codepoint rule tore `นั่ง` into four vertices where there
            # are two graphemes. CJK carries no marks, so this is a no-op there.
            j = i + 1
            while j < n and unicodedata.category(s[j])[0] == "M":
                j += 1
            tok = s[i:j]
            out.append((tok.lower() if fold else tok, i, j))
            i = j
            continue
        if _is_word_char(ch) or ch.isdigit():
            j = i
            while j < n and (_is_word_char(s[j]) or s[j].isdigit()) \
                    and not is_scriptio_continua(s[j]):
                j += 1
            # a joiner may not END a word: "cat's" keeps the apostrophe, "cat," does not
            while j > i and s[j - 1] in _JOINERS:
                j -= 1
            if j > i:
                # a period the VETO claims belongs to this token is part of it, not a
                # mark after it. "Mr." is one token at every layer or it is one token at
                # none: the sentence gate honoured the veto while the span gate did not,
                # so the same period was a suffix upstairs and a delimiter downstairs.
                if (veto is not None and j < n and s[j] == "."
                        and veto.search(s[max(0, j - 12):j + 1])):
                    j += 1
                tok = s[i:j]
                out.append((tok.lower() if fold else tok, i, j))
                i = j
                continue
        # a RUN of punctuation is one mark, not one token per character. An ellipsis is
        # a single delimiter and `?!` is a single one; emitting three periods for `...`
        # misreports the text even where consecutive gates happen to collapse to the
        # same split. Source already read operator runs whole; prose now does too.
        j = i
        while j < n and not s[j].isspace() and not _is_word_char(s[j]) \
                and not s[j].isdigit() and not is_scriptio_continua(s[j]):
            j += 1
        # PROGRESS is not optional. A token of joiners only ("--", a bare apostrophe)
        # enters the word branch, is trimmed back to zero length because a joiner may
        # not end a word, and arrives here on a character the run loop calls a word,
        # so j never advances and the tokenizer hangs. Emit one character and move.
        if j == i:
            j = i + 1
        out.append((s[i:j], i, j))
        i = j
    return out


#: natural language, no corpus wrapper and no headings claimed
TEXT = CorpusProfile(name="text")

#: English prose as Project Gutenberg distributes it. The abbreviations, the heading
#: conventions and the inserted markers are ENGLISH-AND-GUTENBERG facts, which is why
#: they belong here and not in the segmenter.
ENGLISH_GUTENBERG = CorpusProfile(
    name="english-gutenberg",
    markers=(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",
             r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*"),
    headings=(
        ("numbered_chapter",
         r"^[ \t]*(?:CHAPTER|Chapter)[ \t]+([IVXLCDM]+|\d+)[.\s]*(.*)$"),
        ("numbered_book",
         r"^[ \t]*(?:BOOK|PART|Book|Part)[ \t]+([IVXLCDM]+|\d+)[.\s]*(.*)$"),
        ("roman_alone", r"^[ \t]*([IVXLCDM]{1,7})\.?[ \t]*$"),
        ("arabic_alone", r"^[ \t]*(\d{1,3})\.?[ \t]*$"),
    ),
    veto=re.compile(
        r"(?:^|\s)(?:[A-Z]"
        r"|[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Pp]rof|[Rr]ev|[Hh]on|[Ss]t|[Jj]r|[Ss]r|[Ee]sq"
        r"|[Vv]ol|[Cc]h|[Cc]hap|[Ss]ec|[Ss]ect|[Pp]t|[Pp]p|[Ff]ig|[Nn]o|[Ee]d|[Ee]ds"
        r"|[Tt]rans|[Cc]f|[Vv]iz|[Aa]pprox|[Dd]ept|[Uu]niv|[Ii]nc|[Ll]td|[Cc]o|[Ee]sp"
        r"|vs|etc|ca|al|op|cit|ibid"
        r"|[Ee]\.g|[Ii]\.e"
        r"|[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Ss]ept|[Oo]ct"
        r"|[Nn]ov|[Dd]ec)\.$"),
)

#: Python source. The gate is the language's own punctuation, which is EXACT rather
#: than conventional, and the layers are declared by the grammar rather than inferred.
PYTHON_SOURCE = CorpusProfile(
    name="python-source",
    token_rule="identifier",
    casefold=False,                      # `Foo` and `foo` are different identifiers
    gate=frozenset("( ) [ ] { } ; , : = . @".split()),
    layers=("statement", "block", "function", "module"),
)
