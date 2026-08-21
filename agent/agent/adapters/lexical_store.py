"""Load the lexical stack into an RCDB, in the order the layers depend on each other.

The order is not arbitrary. WordNet and Roget establish the WORD vertices; the NRC
lexicons are 0-cochains ON those vertices and mean nothing before they exist; Wiktionary
is the widest layer and reuses the same vocabulary. Building them in that order lets each
later layer be read against the earlier ones rather than beside them.

Each resource becomes ONE record in the store, so the store holds complexes rather than
documents, and each carries:

    the complex     serialized through the canonical layered binary state, which is what
                    `RCStore.put` does: `to_state` -> safetensors bytes -> blob.
    a signature     `structural_signature`, which is what the index's 0-cochains read
    a digest        `state_digest` over the tensor payload, carried in `meta` so a
                    caller can check a blob against what was written without loading it

The connotation lexicons are NOT complexes and are not stored as one. They are values on
words, so they go in as a cochain aligned to a named complex's vertex order, which is the
only form in which "the valence of this vertex" is a well-posed statement.
"""
from __future__ import annotations

import os
import time

import numpy as np

__all__ = ["build_lexical_store", "attach_connotation"]

DEFAULT_ROOT = "~/projects/rexgraph/data/lexical"


def _paths(root=DEFAULT_ROOT):
    r = os.path.expanduser(root)
    return {
        "wordnet": os.path.join(r, "wordnet", "english-wordnet-2024.xml.gz"),
        "roget": os.path.join(r, "thesaurus", "rogets-pg22.txt"),
        "vad": os.path.join(r, "connotation", "NRC-VAD-Lexicon",
                            "NRC-VAD-Lexicon.txt"),
        "emolex": os.path.join(r, "connotation", "NRC-Emotion-Lexicon",
                               "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
        "wiktionary": os.path.join(r, "wiktionary", "wiktionary-en.rexidx"),
        "store": os.path.join(r, "store"),
    }


def _digest_of(rex):
    from rexgraph.io.rex_state import to_state
    st = to_state(rex)
    return st.header.get("digest", "")


def _edge_types(group_labels):
    """The relation type of each group, as codes into a name table."""
    if not group_labels:
        return None, None, None
    kinds, names, code = [], [], {}
    for lab in group_labels:
        k = str(lab[0]) if isinstance(lab, (tuple, list)) and lab else str(lab)
        if k not in code:
            code[k] = len(names); names.append(k)
        kinds.append(code[k])
    # a group name only where it is not recoverable from the column itself
    gnames = [str(lab[1]) if isinstance(lab, (tuple, list)) and len(lab) > 1 else ""
              for lab in group_labels]
    return kinds, names, (gnames if set(names) == {"category"} else None)


def _put(store, rid, rex, info, *, source, kind, extra=None, group_labels=None,
         log=print):
    """One record: the complex, its vertex labels, its EDGE TYPES, its digest, and what
    built it.

    The edge types were the gap: this stored `vertex_labels` and dropped the per-group
    label, so every stored column was anonymous and `bank`/`money` sharing one said
    nothing about whether that column was `synonyms` or `antonyms`. An untyped edge is
    not a predication, which forced the answerers to re-read the source files.
    """
    labels = list(info.get("members") or info.get("vocab") or [])
    etypes, tnames, gnames = _edge_types(group_labels)
    meta = {
        "source": source, "object_type": kind,
        "vertex_labels": labels,
        "n_wide": int(info.get("n_wide", 0)),
        "n_pairs": int(info.get("n_pairs", 0)),
        "pair_mode": info.get("pair_mode", "spanning"),
        "state_digest": _digest_of(rex),
        "built": time.time(),
    }
    if etypes is not None:
        meta["edge_types"] = etypes            # one code per column
        meta["type_names"] = tnames            # what the codes index
        if gnames is not None:
            meta["group_names"] = gnames       # only where the column cannot say it
    if extra:
        meta.update(extra)
    t = time.perf_counter()
    store.put(rid, rex, meta=meta, tags=[source, kind])
    log(f"    stored {rid!r}: nV {int(rex.nV):,} nE {int(rex.nE):,} "
        f"in {time.perf_counter() - t:.1f}s  digest {meta['state_digest'][:12]}")
    return meta["state_digest"]


def build_lexical_store(root=DEFAULT_ROOT, *, pair_mode="none",
                        include=("wordnet", "roget", "wiktionary"), log=print):
    """Load the lexical sources into a record store, in dependency order."""
    from agent.adapters import lexical as L
    from agent.adapters import wiktionary as WK
    from agent.rcdb import FileStore
    from rexgraph.construct import from_groups

    p = _paths(root)
    os.makedirs(p["store"], exist_ok=True)
    store = FileStore(p["store"])
    written = {}

    if "wordnet" in include:
        log("  WordNet")
        t = time.perf_counter()
        wn = L.load_wordnet(p["wordnet"])
        groups, labels = L.wordnet_groups(wn)
        log(f"    {len(wn['synsets']):,} synsets -> {len(groups):,} groups "
            f"({time.perf_counter() - t:.1f}s)")
        rex, info = from_groups(groups, pair_mode=pair_mode, verify=False)
        info["pair_mode"] = pair_mode
        written["wordnet"] = _put(store, "lex:wordnet", rex, info,
                                  group_labels=labels,
                                  source="wordnet", kind="sense_inventory",
                                  extra={"n_synsets": len(wn["synsets"]),
                                         "n_relations": len(wn["relations"])}, log=log)

    if "roget" in include:
        log("  Roget")
        cats = L.load_roget(p["roget"])
        kept = [v for v in cats.values() if len(v["terms"]) >= 2]
        groups = [v["terms"] for v in kept]
        # the category NAME is the whole content of a Roget column and is nowhere in the
        # boundary, so it is carried as the group label rather than dropped.
        labels = [("category", v["name"]) for v in kept]
        rex, info = from_groups(groups, pair_mode=pair_mode, verify=False)
        info["pair_mode"] = pair_mode
        written["roget"] = _put(store, "lex:roget", rex, info,
                                group_labels=labels,
                                source="roget", kind="thesaurus",
                                extra={"n_categories": len(cats)}, log=log)

    if "wiktionary" in include:
        log("  Wiktionary")
        t = time.perf_counter()
        idx = WK.read_index(p["wiktionary"], verify=True)
        groups, wlabels = WK.wiktionary_groups(idx)
        log(f"    {idx['n_entries']:,} entries -> {len(groups):,} linkage groups "
            f"({time.perf_counter() - t:.1f}s)")
        rex, info = from_groups(groups, pair_mode=pair_mode, verify=False)
        info["pair_mode"] = pair_mode
        written["wiktionary"] = _put(store, "lex:wiktionary", rex, info,
                                     group_labels=wlabels,
                                     source="wiktionary", kind="linkage_graph",
                                     extra={"n_entries": idx["n_entries"],
                                            "heap": idx.get("jsonl", "")}, log=log)
    return store, written


def attach_connotation(store, rid="lex:wordnet", root=DEFAULT_ROOT, log=print):
    """Align NRC VAD and EmoLex onto a stored complex's vertex order.

    A lexicon says nothing about pairs, so it is a 0-cochain and is only meaningful
    against a fixed vertex ORDER. That order is the stored record's, so this reads the
    record back and aligns to it rather than inventing an order of its own.

    Returns `(vad, emo, coverage)` with `vad` shaped (nV, 3) and `emo` (nV, n_emotions),
    both NaN/0 where the lexicon is silent, which is a different statement from zero.
    """
    from agent.adapters import lexical as L

    p = _paths(root)
    rec = store.get_record(rid)
    labels = [str(x) for x in (rec.meta or {}).get("vertex_labels", [])]
    if not labels:
        raise ValueError(f"{rid!r} carries no vertex_labels, so a cochain cannot align")

    vad_tab = L.load_nrc_vad(p["vad"])
    emo_tab = L.load_nrc_emolex(p["emolex"])
    emotions = sorted({e for d in emo_tab.values() for e in d})

    nV = len(labels)
    vad = np.full((nV, 3), np.nan)
    emo = np.zeros((nV, len(emotions)), dtype=np.int8)
    e_at = {e: i for i, e in enumerate(emotions)}
    hit_v = hit_e = 0
    for i, w in enumerate(labels):
        key = w.lower()
        v = vad_tab.get(key)
        if v is not None:
            vad[i] = v; hit_v += 1
        d = emo_tab.get(key)
        if d:
            for e in d:
                emo[i, e_at[e]] = 1
            hit_e += 1
    cov = {"n_vertices": nV, "vad_covered": hit_v, "emolex_covered": hit_e,
           "emotions": emotions,
           "vad_fraction": hit_v / max(nV, 1), "emolex_fraction": hit_e / max(nV, 1)}
    log(f"    VAD covers {hit_v:,}/{nV:,} ({cov['vad_fraction']*100:.1f}%), "
        f"EmoLex {hit_e:,}/{nV:,} ({cov['emolex_fraction']*100:.1f}%)")
    return vad, emo, cov
