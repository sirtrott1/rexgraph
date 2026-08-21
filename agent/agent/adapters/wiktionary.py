"""Wiktionary: a binary index over the JSONL, with the JSONL kept only as a heap.

3.21 GB and 1,487,639 entries of JSONL is not a format to query. Parsing it per lookup
costs a full scan, and re-encoding the whole thing into records would duplicate 3 GB of
prose that nothing reads until someone asks for one entry.

So the split is the one the owner's storage already makes everywhere else: STRUCTURE goes
binary and RAW stays addressable.

    binary index (safetensors)   the packed string tables, the codes, and the CSR of
                                 which word links to which. This is what every reading
                                 touches, and it is tensors, not documents.
    the JSONL                    an addressable heap. Each entry carries its byte offset
                                 and length, so `raw_entry` is one seek and one readline.
                                 No JSON is parsed unless a caller asks for a specific
                                 entry's raw form.

That is the same trichotomy as `rcdb_index`: existence is the sparsity pattern of the
linkage CSR, orientation is the head word at position 0 of its own span, and share is
`1/(k-1)` from the span width. Nothing stores a weight and nothing stores a sign.

The linkage kinds are the relational content. `synonyms` and the rest name OTHER words, so
`[word] + [linked words]` is a group in exactly the sense a WordNet synset is one, and it
feeds `rexgraph.construct.from_groups` unchanged. Glosses and etymology are prose and stay
in the heap; they feed `construct.from_text` if a caller wants them as a complex, which is
a different construction and therefore a different call.
"""
from __future__ import annotations

import json
import os

import numpy as np

from agent.rcdb_index import _fit, _pack_strings, _unpack_strings

__all__ = ["build_index", "write_index", "read_index", "raw_entry",
           "wiktionary_groups", "function_words", "LINK_KINDS", "FUNCTION_POS"]

#: the parts of speech that GATE rather than participate. These are recorded classes,
#: not a stopword list: a determiner, a preposition or a conjunction is structure the
#: lexicon names, and which words belong is a lexical fact rather than a frequency cut.
FUNCTION_POS = ("article", "conj", "det", "particle", "postp", "prep", "pron")

FORMAT_VERSION = 1

#: entry-level fields naming other words. These are the relations; everything else in an
#: entry is prose, pronunciation or template residue and stays in the heap.
LINK_KINDS = ("synonyms", "antonyms", "hypernyms", "hyponyms", "meronyms",
              "holonyms", "coordinate_terms", "derived", "related", "troponyms")


def _words_of(value):
    """The word strings in a linkage field, which is a list of dicts with a `word` key."""
    out = []
    for item in value or ():
        w = item.get("word") if isinstance(item, dict) else item
        if isinstance(w, str) and w:
            out.append(w)
    return out


def build_index(jsonl_path, *, lang_code="en", limit=None, link_kinds=LINK_KINDS):
    """One streaming pass over the JSONL, returning the index as arrays.

    Byte offsets are recorded from the file position BEFORE each line is read, so an
    offset plus a `readline` recovers the exact bytes without re-parsing anything ahead
    of it. The file is opened in binary for that reason: a text handle's `tell` is opaque
    and cannot be seeked to reliably mid-iteration.
    """
    path = os.path.expanduser(str(jsonl_path))
    kinds = tuple(link_kinds)
    kind_code = {k: i for i, k in enumerate(kinds)}

    words, word_idx = [], {}
    pos_tab, pos_idx = [], {}

    def wcode(w):
        c = word_idx.get(w)
        if c is None:
            c = len(words); word_idx[w] = c; words.append(w)
        return c

    e_word, e_pos, e_off, e_len = [], [], [], []
    l_src, l_dst, l_kind = [], [], []
    n_seen = n_kept = 0

    with open(path, "rb") as fh:
        while True:
            off = fh.tell()
            line = fh.readline()
            if not line:
                break
            n_seen += 1
            if limit is not None and n_kept >= int(limit):
                break
            try:
                d = json.loads(line)
            except Exception:                       # a truncated tail, not a reason to stop
                continue
            if lang_code and d.get("lang_code") != lang_code:
                continue
            w = d.get("word")
            if not isinstance(w, str) or not w:
                continue
            src = wcode(w)
            p = d.get("pos") or ""
            pc = pos_idx.get(p)
            if pc is None:
                pc = len(pos_tab); pos_idx[p] = pc; pos_tab.append(p)

            e_word.append(src); e_pos.append(pc)
            e_off.append(off); e_len.append(len(line))
            n_kept += 1

            for k in kinds:
                for other in _words_of(d.get(k)):
                    if other == w:
                        continue
                    l_src.append(src); l_dst.append(wcode(other))
                    l_kind.append(kind_code[k])
            # sense-level linkages carry the same relations at a finer grain
            for s in d.get("senses") or ():
                if not isinstance(s, dict):
                    continue
                for k in kinds:
                    for other in _words_of(s.get(k)):
                        if other == w:
                            continue
                        l_src.append(src); l_dst.append(wcode(other))
                        l_kind.append(kind_code[k])

    nW = len(words)
    return {
        "format": FORMAT_VERSION, "lang_code": lang_code,
        "jsonl": os.path.abspath(path),
        "n_entries": n_kept, "n_lines_seen": n_seen, "n_words": nW,
        "words": words, "pos": pos_tab, "kinds": list(kinds),
        "entry_word": _fit(e_word, max(nW, 1)),
        "entry_pos": _fit(e_pos, max(len(pos_tab), 1)),
        # the POINTER layer: where each entry's raw bytes live in the heap
        "entry_offset": np.asarray(e_off, dtype=np.int64),
        "entry_length": _fit(e_len, max(e_len) if e_len else 1),
        "link_src": _fit(l_src, max(nW, 1)),
        "link_dst": _fit(l_dst, max(nW, 1)),
        "link_kind": _fit(l_kind, max(len(kinds), 1)),
    }


def write_index(path, index) -> str:
    """The index as safetensors, with a digest over the payload.

    String tables go as one utf-8 blob plus offsets, the same packing `rcdb_index` uses,
    so a read is a slice and nothing re-encodes. The header is metadata only: counts,
    the language, the heap path, and the digest.
    """
    from safetensors.numpy import save_file

    from rexgraph.io.rex_state import DIGEST_ALGO, state_digest

    t = {}
    for key in ("entry_word", "entry_pos", "entry_offset", "entry_length",
                "link_src", "link_dst", "link_kind"):
        t[key] = np.ascontiguousarray(index[key])
    for name in ("words", "pos", "kinds"):
        blob, offs = _pack_strings([str(x) for x in index[name]])
        t[f"{name}/table"] = np.frombuffer(blob, dtype=np.uint8).copy()
        t[f"{name}/offsets"] = np.ascontiguousarray(offs)
    meta = {
        "format": str(FORMAT_VERSION),
        "lang_code": str(index.get("lang_code") or ""),
        "jsonl": str(index.get("jsonl") or ""),
        "n_entries": str(index["n_entries"]),
        "n_words": str(index["n_words"]),
        # the FRAMING this digest was written under. `state_digest` gained
        # length-prefixed fields (algo 2) because the unframed form collided; a file
        # written before that verifies only under algo 1, and without this stamp every
        # index already on disk reads as corrupt.
        "digest_algo": str(DIGEST_ALGO),
        "digest": state_digest(t),
    }
    save_file(t, str(path), metadata=meta)
    return str(path)


def read_index(path, *, verify: bool = True):
    """The inverse of `write_index`. A digest that does not match raises."""
    from safetensors import safe_open
    from safetensors.numpy import load_file

    from rexgraph.io.rex_state import state_digest

    with safe_open(str(path), "numpy") as fh:
        meta = fh.metadata() or {}
    if int(meta.get("format", 0)) != FORMAT_VERSION:
        raise ValueError(f"wiktionary index format {meta.get('format')} "
                         f"!= {FORMAT_VERSION}")
    t = load_file(str(path))
    algo = int(meta.get("digest_algo", 1))
    if verify and meta.get("digest") and state_digest(t, algo=algo) != meta["digest"]:
        raise ValueError("wiktionary index digest mismatch: the file is not what was "
                         "written")

    def table(name):
        return _unpack_strings(t[f"{name}/table"].tobytes(), t[f"{name}/offsets"])

    out = {k: t[k] for k in ("entry_word", "entry_pos", "entry_offset", "entry_length",
                             "link_src", "link_dst", "link_kind")}
    out.update({"words": table("words"), "pos": table("pos"), "kinds": table("kinds"),
                "lang_code": meta.get("lang_code"), "jsonl": meta.get("jsonl"),
                "n_entries": int(meta.get("n_entries", 0)),
                "n_words": int(meta.get("n_words", 0)), "format": FORMAT_VERSION})
    return out


def raw_entry(index, i: int, *, jsonl_path=None):
    """The raw entry for row `i`: one seek and one read, no scan and no ambient parse.

    This is the whole reason the JSONL stays on disk. The index answers every structural
    question without touching it; a caller who wants the prose for ONE entry pays for
    that one entry.
    """
    path = os.path.expanduser(str(jsonl_path or index.get("jsonl") or ""))
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"the heap is not where the index says it is ({path!r}). Pass jsonl_path= if "
            f"it moved; the index stores an absolute path at build time.")
    off = int(np.asarray(index["entry_offset"])[i])
    ln = int(np.asarray(index["entry_length"])[i])
    with open(path, "rb") as fh:
        fh.seek(off)
        return json.loads(fh.read(ln))


def wiktionary_groups(index, *, kinds=None, min_size=2):
    """Linkages as groups for `rexgraph.construct.from_groups`.

    One group per (head word, kind): the head at position 0 and everything that kind
    links it to. The head is FIRST because that is the orientation, and "synonyms of x"
    is a statement about x.

    A kind is kept whole rather than merged with the others, because `synonyms` and
    `antonyms` are different relations and unioning them would assert something neither
    says. Pass `kinds=` to select.

    Returns `(groups, labels)` with `labels[i] = (kind, head_word)`.
    """
    words = list(index["words"])
    kind_names = list(index["kinds"])
    want = set(kind_names if kinds is None else kinds)
    src = np.asarray(index["link_src"]); dst = np.asarray(index["link_dst"])
    kd = np.asarray(index["link_kind"])

    buckets: dict = {}
    for s, d, k in zip(src.tolist(), dst.tolist(), kd.tolist(), strict=True):
        name = kind_names[k] if k < len(kind_names) else str(k)
        if name not in want:
            continue
        buckets.setdefault((name, s), []).append(d)

    groups, labels = [], []
    for (name, s), ds in buckets.items():
        g = [words[s]] + [words[d] for d in dict.fromkeys(ds)]
        g = list(dict.fromkeys(g))
        if len(g) >= int(min_size):
            groups.append(g); labels.append((name, words[s]))
    return groups, labels


def function_words(index, classes=FUNCTION_POS, *, exclusive=True):
    """The words whose recorded parts of speech are function classes.

    `exclusive=True` keeps only words EVERY reading of which is a function class, so
    "the" and "of" qualify while "that" does not if the lexicon also records it as a
    noun. That is the conservative direction and the right one for a gate: gating on an
    ambiguous word would cut a span in a reading where the word is content.

    A gate set is a lexical fact under this definition, not a frequency threshold and not
    a curated list. Returns a frozenset of lowercase words.
    """
    import numpy as np

    want = {str(c) for c in classes}
    words = list(index["words"])
    pos = list(index["pos"])
    ew = np.asarray(index["entry_word"])
    ep = np.asarray(index["entry_pos"])

    seen_fn, seen_other = set(), set()
    for w_i, p_i in zip(ew.tolist(), ep.tolist(), strict=True):
        name = pos[p_i] if p_i < len(pos) else ""
        (seen_fn if name in want else seen_other).add(w_i)
    keep = seen_fn - seen_other if exclusive else seen_fn
    return frozenset(str(words[i]).lower() for i in keep if i < len(words))
