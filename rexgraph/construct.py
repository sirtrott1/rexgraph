"""Constructions that carry more than one grade, because one grade carries nothing.

A set of groups handed straight to `from_hypergraph` is a forest of stars. Every group is
one branching relation, no two relations share a cycle, `rank(B_1) = nE`, and the cycle
space is empty: `beta_1 = 0`, every signal is gradient, and every reading above the
existence layer reads zero. That is not a property of the data, it is the construction
throwing away what the data said.

The fix is to carry BOTH readings of the same fact. A group over `k` members asserts the
group, and it also asserts each pairwise contact inside it. Carried together the pairwise
relations span the branching relation's boundary, so the group CLOSES against them, and
that closure is what `faces.auto_hyperface` then solves for. Only now is there a cycle
space to read, a curl/harmonic split to take, and a face to attach.

    wide      one branching relation per group, arity k (optionally with the group's own
              vertex distinguished at position 0, which is what "accessioned BY these
              members" means and where the -1 goes)
    pairs     one 2-ary relation per co-occurring member pair, optionally thresholded by
              how often the pair is seen

`sections` comes back with the complex and is not an afterthought: every reading in
`rexgraph.partition` is taken over a SECTION, and a group's section is its own wide
relation together with its own pairs. Recomputing that from a pair index at each call
site was how the construction stayed a paste instead of becoming a function.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

__all__ = ["from_groups", "group_sections", "from_text", "precedence_field",
           "first_occurrences", "spans_of", "from_spans", "mixed_rank"]


def from_groups(groups, *, min_pair_count=1, owner_vertex=False,
                pair_mode="clique", verify=True):
    """The mixed construction over `groups`, an iterable of member-id iterables.

    Members are any hashable labels; they are mapped to vertices in first-seen order and
    the mapping comes back so a caller can read a relation's support in its own terms.

    `min_pair_count` drops pairs seen fewer than that many times, which is how a text
    corpus keeps its co-occurrence graph from being noise. A dropped pair simply is not
    asserted; nothing is reweighted.

    `owner_vertex` prepends one vertex per group to its own wide relation, so the group
    itself is the distinguished vertex. Use it when the group is a THING (a document, a
    record) rather than only a set (a protein complex read as its members).

    `pair_mode="none"` carries the groups ALONE: a k-ary relation is one column and
    nothing is enumerated from it. The other two modes each invent pairwise facts the
    source never stated, and on prose the invention dominates. Measured on one book,
    1,469 sentences give 1,469 relations at rank 1,437 under "none" against 12,890 at
    rank 2,566 under "spanning", so 11,421 pairs manufactured 1,129 dimensions of rank
    and 10,292 of the 10,324 cycles. It is not a forest of stars either: shared
    vocabulary leaves 32 real cycles. Use "none" when the relation IS the fact and the
    readings come from the field; use the others when a caller genuinely wants the
    pairwise contacts as data.

    `pair_mode` decides HOW MANY pairs a group contributes, and on a wide group it is the
    difference between a usable complex and an unusable one::

        "clique"    every C(k,2) contact. This is what closes a group most redundantly,
                    and on a group of arity 1232 it is 758,296 relations from ONE fact.
        "spanning"  k-1 contacts along the sorted members, which is all that is needed:
                    a connected set's zero-sum space has dimension k-1, so a spanning
                    subset already spans the group's column and the group still closes.

    The RANK is the same either way (Corollary 14.2 depends on the group being connected,
    not on how), and so is closure. What differs is the CYCLE count, and the difference is
    not information: the data said "these k members are one group", one fact, while the
    clique asserts C(k,2) separate pairwise facts it never stated. Measured on a fixture
    with one arity-40 group: rank 59 both ways, group still closes both ways, cycles 1303
    against 221. On Wiktionary the whole corpus is 84,162,599 clique pairs against
    1,882,554 spanning ones, 45x, with 82% of the clique coming from the 0.57% of groups
    at arity >= 100. Clique expansion is what the model exists not to need; "clique" stays
    the default only because it is what every earlier measurement in this repo used.

    Returns `(rex, info)` where `info` carries `sections` (group index -> relation ids),
    `vertex_of` (member label -> vertex), `pair_index` ((u,v) -> relation id),
    `n_wide` and `n_pairs`.
    """
    from rexgraph.graph import RexGraph

    gs = [list(dict.fromkeys(g)) for g in groups]      # de-dup, keep order
    if not gs:
        raise ValueError("no groups given: there is nothing to construct")

    vertex_of, members = {}, []
    n_owner = len(gs) if owner_vertex else 0
    for g in gs:
        for m in g:
            if m not in vertex_of:
                vertex_of[m] = n_owner + len(members)
                members.append(m)

    if pair_mode not in ("clique", "spanning", "none"):
        raise ValueError(
            f"pair_mode must be 'clique', 'spanning' or 'none', not {pair_mode!r}")
    co = Counter()
    if pair_mode != "none":
        for g in gs:
            ids = sorted(vertex_of[m] for m in g)
            if pair_mode == "clique":
                co.update(combinations(ids, 2))
            else:
                co.update((ids[i], ids[i + 1]) for i in range(len(ids) - 1))
    pairs = [p for p, c in sorted(co.items()) if c >= int(min_pair_count)]

    wide = []
    for i, g in enumerate(gs):
        span = [vertex_of[m] for m in g]
        wide.append(([i] + span) if owner_vertex else span)

    pair_index = {p: len(wide) + j for j, p in enumerate(pairs)}
    ptr, idx = [0], []
    for r in wide:
        idx += list(r); ptr.append(len(idx))
    for a, b in pairs:
        idx += [a, b]; ptr.append(len(idx))

    rex = RexGraph.from_hypergraph(np.asarray(ptr, np.int64), np.asarray(idx, np.int64))
    sections = group_sections(gs, vertex_of, pair_index, owner_vertex=owner_vertex)
    info = {"sections": sections, "vertex_of": vertex_of, "pair_index": pair_index,
            "n_wide": len(wide), "n_pairs": len(pairs), "members": members}

    if verify:
        # the whole point of carrying both grades is that a cycle space exists. A group
        # of arity >= 3 whose pairs survived MUST close, so an empty cycle space here
        # means the construction collapsed and every downstream reading would be zero.
        # check whenever a group could close, NOT only when pairs survived: the
        # collapse being guarded against is precisely the case where they did not.
        r1 = (int(rex.rank_tower()["ranks"][0])
              if any(len(g) >= 3 for g in gs) else None)
        if r1 is not None and int(rex.nE) - r1 <= 0:
                raise ValueError(
                    f"the construction closed no cycle: nE {int(rex.nE)} against "
                    f"rank(B1) {r1}. Carrying groups alone gives a forest of stars, so a "
                    f"zero cycle space means the pairs did not survive min_pair_count "
                    f"({min_pair_count}) and there is nothing above the existence layer "
                    f"to read.")
    return rex, info


def group_sections(groups, vertex_of, pair_index, *, owner_vertex=False):
    """Group index -> its own relation ids: its wide relation plus its own pairs.

    Separated because a caller who has already built the complex (from a store, say)
    still needs the sectioning, and it is a lookup rather than a rebuild.
    """
    out = {}
    for i, g in enumerate(groups):
        ids = sorted({vertex_of[m] for m in g})
        rels = [i]
        rels += [pair_index[p] for p in combinations(ids, 2) if p in pair_index]
        out[i] = sorted(set(rels))
    return out


#### text ######################################################################
#
# A sentence is a group over the words it uses, so `from_groups` already does the work.
# What text adds is that the sentence is a THING (it gets its own vertex, distinguished
# at position 0) and that the token stream has an ORDER the group has thrown away.
#
# The two readings a corpus supports want DIFFERENT tokenisations, and the library must
# not pick:
#
#   topical / semantic   drop function words. They co-occur with everything, so left in
#                        they swamp the co-occurrence graph.
#   syntactic / order    KEEP function words. They are most of what order is about, and
#                        removing them is what makes an adjacency measurement meaningless.
#
# So `stopwords` defaults to None (keep everything) and is the caller's decision, stated.

_SENTENCE_SPLIT = r"(?<=[.!?])\s+|\n\n+"
_TOKEN = r"[a-z']+"


def _orient(group, grammar):
    """Reorder a group so the head is first, and say which frame decided it.

    Orientation is carried by POSITION: the participant at index 0 takes the `-1`,
    so orienting by grammar is reordering, not a second mechanism. The frame states
    which participant heads: "Somebody ----s somebody something" puts the verb at the
    head and gives agent, recipient and theme `1/3` each.

    Returns `(group, frame_or_None)`. None means no frame governed and the order stands
    as the text gave it, which is a DIFFERENT statement from a frame choosing the first
    token and has to stay distinguishable in the attribute.
    """
    if grammar is None:
        return group, None
    try:
        hit = grammar.head_of(group)
    except Exception:
        # `grammar` is caller-supplied and optional, so a source that cannot answer
        # for this group makes NO CLAIM, which is what None already means here. It is
        # not "the first token heads it", and nothing downstream reads it as that.
        return group, None
    if not hit:
        return group, None
    i, frame = hit
    if i == 0:
        return group, frame
    return [group[i]] + group[:i] + group[i + 1:], frame


def from_text(text, *, sentences=None, stopwords=None, token_pattern=_TOKEN,
              sentence_pattern=_SENTENCE_SPLIT, lowercase=True, min_token_len=1,
              min_terms=3, max_terms=None, min_pair_count=1, document_vertex=True,
              pair_mode="clique", grammar=None, verify=True):
    """The mixed construction over text: each sentence is a group over its words.

    Pass `text` (split by `sentence_pattern`) or `sentences` (already split).

    `pair_mode` matters more on prose than anywhere else, because a sentence is a wide
    group and clique expansion asserts that every word in it touched every other. On one
    61 KB book, 136 sentences gave 17,408 relations under "clique" against 1,861 under
    "spanning" (9.4x), and betti_1 16,439 against 892. rank(B1) was 969 BOTH ways: the
    extra 15,547 cycles are the expansion's own artifact, not the text's. "clique" stays
    the default only because it is what the existing callers already got.

    `stopwords` is None by default, meaning nothing is dropped. Drop function words for a
    topical reading; keep them for anything about order, because they carry most of it.
    The library does not choose, because the two choices answer different questions and
    the difference is large.

    `document_vertex` gives each sentence its own vertex at position 0 of its own
    relation, which is what makes the sentence a cell rather than only a set.

    Returns `(rex, info)`; `info` carries everything `from_groups` returns plus
    `sequences` (the token stream per sentence, ORDER PRESERVED, which the group has
    thrown away and `precedence_field` needs) and `vocab`.
    """
    import re

    if sentences is None:
        if text is None:
            raise ValueError("give either text or sentences")
        sentences = re.split(sentence_pattern, text)
    stop = set(stopwords or ())
    pat = re.compile(token_pattern)

    seqs, groups, kept, frames = [], [], [], []
    for i, s in enumerate(sentences):
        toks = pat.findall(s.lower() if lowercase else s)
        toks = [t for t in toks if len(t) >= int(min_token_len) and t not in stop]
        if len(set(toks)) < int(min_terms):
            continue
        if max_terms is not None and len(set(toks)) > int(max_terms):
            continue
        seqs.append(toks)
        g = list(dict.fromkeys(toks))                  # the group is the SET, ordered
        # ORIENTATION. Position carries it, so a grammar reorders rather than adding a
        # second mechanism: the frame says which participant heads, and the head takes
        # the -1. Without a grammar the text's own order stands, which is the
        # approximation: right for a verb-initial clause, wrong for a noun phrase.
        g, fid = _orient(g, grammar)
        frames.append(fid)
        groups.append(g)
        # WHICH sentence each group came from. The filters above compact the list, so
        # group j is not sentence j, and a caller aligning spans or labels to sections
        # by position silently misaligns everything after the first drop.
        kept.append(i)
    if not groups:
        raise ValueError(
            f"no sentence survived the filters (min_terms={min_terms}, "
            f"max_terms={max_terms}, min_token_len={min_token_len}, "
            f"{len(stop)} stopwords). Nothing to construct.")

    rex, info = from_groups(groups, min_pair_count=min_pair_count,
                            owner_vertex=document_vertex, pair_mode=pair_mode,
                            verify=verify)
    info["sequences"] = seqs
    info["kept"] = kept                    # group index -> index in `sentences`
    #: which frame oriented each group, or None where none governed. None is NOT "the
    #: first token heads it". It is "no frame claimed this", and the two have to stay
    #: distinguishable in the record.
    info["frames"] = frames
    info["vocab"] = info["members"]
    info["n_sentences"] = len(groups)
    return rex, info


def first_occurrences(sequences):
    """Each sequence reduced to the first occurrence of each token, order preserved.

    This is a step on the SEQUENCES and deliberately not an option inside
    `precedence_field`, because where it happens decides whether a control is a control.
    An all-pairs precedence reading uses each token's position, so a token appearing
    several times holds the earliest of several positions; that is a multiplicity channel
    riding alongside the order channel. Permuting a sequence that still has repeats
    leaves that channel intact, so the "order destroyed" control still carries it and the
    comparison reports the OPPOSITE conclusion. This was measured, twice.

    So reduce FIRST, then read and shuffle the reduced sequences. Do NOT reduce for an
    adjacency reading: removing a repeat makes two non-neighbours adjacent.
    """
    out = []
    for toks in sequences:
        seen, keep = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t); keep.append(t)
        out.append(keep)
    return out


def precedence_field(info, sequences=None, *, adjacent_only=False):
    """Order as a 1-cochain: net precedence on each pair relation of the complex.

    Which vertex sits at position 0 IS the orientation, so "a precedes b" is already an
    oriented relation and needs no new mechanism. The field is zero on the wide relations
    and carries `(# a before b) - (# b before a)` on each pair, signed against the pair's
    own stored orientation.

    `adjacent_only` counts only consecutive tokens, which is the syntactic reading;
    otherwise every ordered pair within a sentence counts, which is the topical one.

    `sequences` defaults to `info["sequences"]` and is taken AS GIVEN: nothing is
    deduplicated here. For an all-pairs reading pass `first_occurrences(...)` and shuffle
    THOSE for a control; see that function for why doing it in the other order inverts
    the answer.
    """
    from collections import defaultdict
    from itertools import combinations

    vof, pidx = info["vertex_of"], info["pair_index"]
    n_rel = info["n_wide"] + info["n_pairs"]
    net = defaultdict(int)

    for toks in (info["sequences"] if sequences is None else sequences):
        ids = [vof[t] for t in toks if t in vof]
        if adjacent_only:
            steps = zip(ids, ids[1:], strict=False)
        else:
            steps = ((ids[i], ids[j]) for i, j in combinations(range(len(ids)), 2))
        for x, y in steps:
            if x == y:
                continue
            key = (min(x, y), max(x, y))
            e = pidx.get(key)
            if e is not None:
                net[e] += 1 if x < y else -1

    f = np.zeros(n_rel)
    for e, v in net.items():
        f[e] = v
    return f


#### spans #####################################################################
#
# `from_text` above is VERTEX-PRIMARY and that is its defect: a sentence becomes one wide
# relation over its word SET, so multiplicity has to be deduplicated away and function
# words have to be dropped or kept by policy. Both problems are the same problem, and it
# is that the token stream's structure was collapsed into a bag before anything read it.
#
# Edge-primary says the SPAN is the relation and the tokens are its boundary. Then:
#
#   function words are not noise to filter, they DELIMIT. "the cat" is a span with `the`
#   distinguished at position 0. Dropping it destroys the boundary; leaving it in a bag
#   drowns the co-occurrence graph. As a span head it is structural and the policy
#   question does not arise.
#
#   multiplicity is not a confound. Two occurrences of `the` sit in two different spans,
#   which are two different cells sharing a boundary vertex. There is no first-occurrence
#   rule, so there is nothing for a shuffle control to get wrong.
#
# Segmenting at function words is NOT a syntactic parse and is not claimed to be one. It
# is a parser-free segmentation whose boundaries are exactly the tokens a bag-of-words
# reading throws away.


def spans_of(tokens, delimiters, *, with_gates=False):
    """Segment a token stream into spans. Delimiters GATE, they do not participate.

    A delimiter says where a relation ends. It is not one of the relation's members and
    it does not decide which member heads it: existence and orientation are separate
    operators, and a gate is blind to the second. So the delimiter is excluded from the
    support and the head is the FIRST CONTENT TOKEN by position, which is the model's
    ordinary rule (`precedence_field`: the vertex at position 0 IS the orientation).

    This is a correction. The delimiters used to be kept in the span at the front and the
    docstring called the first one "the distinguished vertex", which made the comma head
    "your mother, Jerry": a punctuation mark carrying the -1 of a semantic relation.

    A span of ONE token is a witness (arity 1, column `(+1)`, `L0 u = u`), which is a
    real cell class and not a failure: "Take away your mother, Jerry." sets Jerry off as
    a vocative and a vocative is exactly a participant that exists and bounds nothing.
    It is returned as such rather than filtered.

    `with_gates=True` also returns the delimiter run that closed each span, so a caller
    can attribute the gate to the boundary without putting it in the support.

    Returns `[[token, ...]]`, or `([[token, ...]], [[gate, ...]])`.
    """
    delims = set(delimiters)
    spans, gates, body = [], [], []
    for t in tokens:
        if t in delims:
            if body:                      # this gate is what CLOSED the span
                spans.append(body); gates.append([t]); body = []
            # a gate with no content before it opens rather than closes: it already
            # closed the previous span, or it leads the stream. Recording it here would
            # attribute an opening to the span it precedes and read as its boundary.
        else:
            body.append(t)
    if body:
        spans.append(body); gates.append([])   # the stream ended, nothing closed it
    return (spans, gates) if with_gates else spans


def from_spans(spans, *, min_pair_count=1, sentence_of=None, pair_mode="clique",
               verify=True):
    """Spans as relations over their tokens: the edge-primary reading of a token stream.

    `spans` is an iterable of token lists (see `spans_of`). Each becomes one relation
    with its first token distinguished, and the pairwise contacts inside it are carried
    too, so the span CLOSES against them exactly as a group does in `from_groups`.

    `sentence_of` optionally maps span index -> sentence id. When given, `info` gains
    `sentence_sections` (sentence id -> the relation ids of its spans), which is the
    grade-2 candidate for text: a sentence is a cell over its spans rather than a bag
    over its words. Whether that grade-2 reading says anything is untested.

    Returns `(rex, info)` with the same shape `from_groups` returns, plus `spans`.
    """
    # a one-token span is a WITNESS (column `(+1)`, sum one, `L0 u = u`) which is a
    # cell class, not a failure. Filtering it here deleted the vocative reading: "Take
    # away your mother, Jerry." and "...mother Jerry." differ in exactly whether Jerry is
    # a witness or the fifth member of a branching relation.
    sp = [list(s) for s in spans if s]
    if not sp:
        raise ValueError("no span has any token, so there is nothing to build")
    rex, info = from_groups(sp, min_pair_count=min_pair_count, owner_vertex=False,
                            pair_mode=pair_mode, verify=verify)
    info["spans"] = sp
    if sentence_of is not None:
        # the SAME filter the construction uses, or the map is off by every witness
        kept = [i for i, s in enumerate(spans) if s]
        by_sent = {}
        for new_i, old_i in enumerate(kept):
            by_sent.setdefault(sentence_of[old_i], []).append(new_i)
        info["sentence_sections"] = {
            s: sorted({r for i in ix for r in info["sections"][i]})
            for s, ix in by_sent.items()}
    return rex, info


def mixed_rank(rex, info):
    """`rank(B_1)` of a mixed construction in near-linear time, or None if not applicable.

    The exact integer elimination is quadratic in the fill it creates, which is 127s at
    350k relations. For THIS construction it is avoidable, and the reason is the same one
    that makes the construction worth having.

    The pair relations inside a group span the whole zero-sum space on that group's
    vertices, of dimension `k-1`, so the group's own column, which is zero-sum on exactly
    those vertices, is already in their span and contributes NO rank. The pairs are then a
    pairwise boundary map, where `dim ker(L_0)` really is the component count (Theorem 14
    holds at arity two and only there), so::

        rank(B_1) = nV - components(pairs)

    by union-find.

    THE GUARD IS THE POINT. This needs every group CONNECTED in the surviving pair graph.
    `min_pair_count > 1` drops pairs and can fragment a group, and a fragmented group's
    column is no longer spanned, so it does add rank: measured, the shortcut read 35
    against a true 49. It also does not generalise beyond this construction, since two
    arity-3 relations over the same three vertices both add rank while being one
    component. So this returns None unless the construction it was built for is the
    construction in hand, and the caller falls back to the exact path.
    """
    from rexgraph.graded_boundary import _beta0_components

    n_wide, n_pairs = info.get("n_wide"), info.get("n_pairs")
    if not n_pairs or n_wide is None:
        return None
    Bint = rex._integer_B1().tocsc()
    if Bint.shape[1] != n_wide + n_pairs:
        return None                                  # not the complex this info describes
    pairs = Bint[:, n_wide:]

    # union-find over the PAIR graph only, then every group must land in one component
    nV = int(rex.nV)
    parent = np.arange(nV, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    pc = pairs.tocsc()
    for j in range(pc.shape[1]):
        rows = pc.indices[pc.indptr[j]:pc.indptr[j + 1]]
        if rows.size < 2:
            continue
        a = find(int(rows[0]))
        for r in rows[1:]:
            b = find(int(r))
            if a != b:
                parent[b] = a
    for sec in info["sections"].values():
        wide = [e for e in sec if e < n_wide]
        if not wide:
            continue
        span = Bint[:, wide[0]].tocoo().row
        if span.size and len({find(int(v)) for v in span}) > 1:
            return None                              # a fragmented group: not spanned
    return nV - _beta0_components(pairs)
