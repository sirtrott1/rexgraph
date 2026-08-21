"""Groups alone are a forest of stars; the mixed construction is what makes them readable.

The failure this guards against is silent: hand `from_hypergraph` a set of groups and it
builds a perfectly valid complex in which every reading above the existence layer is zero,
because no two relations share a cycle. These pin that the construction closes, that the
sections it hands back are the ones the readings consume, and that the degenerate case is
still reachable so the difference is demonstrable rather than asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.construct import from_groups, group_sections
from rexgraph.graph import RexGraph


def _bare(groups):
    """Groups handed straight to from_hypergraph: the construction being warned about."""
    ptr, idx = [0], []
    for g in groups:
        idx += list(g); ptr.append(len(idx))
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int64), np.asarray(idx, np.int64))


def test_groups_alone_carry_no_cycle_and_the_mixed_construction_does():
    """The whole reason this module exists, as a measurement rather than a claim."""
    groups = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    bare = _bare(groups)
    rb = int(bare.rank_tower()["ranks"][0])
    assert int(bare.nE) - rb == 0, "groups alone are a forest of stars"

    rex, _info = from_groups([["a", "b", "c"], ["b", "c", "d"], ["c", "d", "e"]])
    r1 = int(rex.rank_tower()["ranks"][0])
    assert int(rex.nE) - r1 > 0, "carrying both grades opens a cycle space"


def test_the_sections_are_the_group_plus_its_own_pairs():
    rex, info = from_groups([["a", "b", "c"], ["b", "c", "d"]])
    secs = info["sections"]
    assert set(secs) == {0, 1}
    # each section holds its own wide relation
    assert 0 in secs[0] and 1 in secs[1]
    # and the pairs inside that group, and no others
    vof, pidx = info["vertex_of"], info["pair_index"]
    from itertools import combinations
    for gi, g in enumerate((["a", "b", "c"], ["b", "c", "d"])):
        ids = sorted(vof[m] for m in g)
        want = {gi} | {pidx[p] for p in combinations(ids, 2) if p in pidx}
        assert set(secs[gi]) == want


def test_the_sections_feed_the_readings_directly():
    """A section is what partition consumes, so it has to be usable without translation.

    Group sections OVERLAP: a pair relation belongs to every group holding both of its
    members. So this also pins that `section_readings` does not apply its partition
    closure identity to a cover, which it did until this construction exercised it.
    """
    from rexgraph.partition import section_readings
    rex, info = from_groups([["a", "b", "c"], ["b", "c", "d"], ["c", "d", "e"]])
    out = section_readings(rex, {f"g{k}": v for k, v in info["sections"].items()},
                           verify=True)
    assert len(out) == 3
    for r in out.values():
        assert r["mass"] <= r["own_rank"] + 1e-9
        assert r["own_cycles"] <= r["share"] + 1e-9


def test_owner_vertex_makes_the_group_the_distinguished_vertex():
    """For a document or a record the group is a THING, and it goes at position 0."""
    rex, info = from_groups([["a", "b"], ["b", "c"]], owner_vertex=True)
    spans = [list(map(int, s)) for s in rex.relation_supports()]
    assert spans[0][0] == 0 and spans[1][0] == 1, "each group owns its own vertex, first"
    assert int(rex.nV) == 2 + 3, "two group vertices plus three members"
    plain, _ = from_groups([["a", "b"], ["b", "c"]])
    assert int(plain.nV) == 3


def test_min_pair_count_drops_pairs_and_says_so_when_nothing_closes():
    # every pair is seen once, so a threshold of 2 removes all of them and the
    # construction collapses back to the forest of stars it exists to avoid
    with pytest.raises(ValueError, match="closed no cycle"):
        from_groups([["a", "b", "c"], ["d", "e", "f"]], min_pair_count=2)
    # the same call with verify off still builds, so a caller who means it can proceed
    rex, info = from_groups([["a", "b", "c"], ["d", "e", "f"]], min_pair_count=2,
                            verify=False)
    assert info["n_pairs"] == 0


def test_a_repeated_pair_survives_a_threshold():
    # the two groups share a,b,c so those three pairs are each seen twice and survive;
    # the pairs reaching d and e are seen once and do not
    rex, info = from_groups([["a", "b", "c", "d"], ["a", "b", "c", "e"]],
                            min_pair_count=2)
    vof = info["vertex_of"]
    kept = set(info["pair_index"])
    assert (min(vof["a"], vof["b"]), max(vof["a"], vof["b"])) in kept
    assert info["n_pairs"] == 3, "only the three pairs seen twice survive"
    assert not any(vof["d"] in p or vof["e"] in p for p in kept)


def test_members_are_labels_and_the_mapping_comes_back():
    rex, info = from_groups([["x", "y"], ["y", "z"]])
    assert set(info["vertex_of"]) == {"x", "y", "z"}
    assert info["members"] == ["x", "y", "z"], "first-seen order"
    assert int(rex.nV) == 3


def test_duplicate_members_within_a_group_are_one_vertex():
    rex, info = from_groups([["a", "a", "b"], ["b", "c"]])
    assert int(rex.nV) == 3
    spans = [list(map(int, s)) for s in rex.relation_supports()]
    assert len(spans[0]) == 2, "the repeat is the same vertex, not a second one"


def test_empty_input_refuses():
    with pytest.raises(ValueError, match="no groups"):
        from_groups([])


def test_group_sections_is_a_lookup_not_a_rebuild():
    """A caller holding a complex from a store still needs the sectioning."""
    groups = [["a", "b", "c"], ["b", "c", "d"]]
    _rex, info = from_groups(groups)
    again = group_sections(groups, info["vertex_of"], info["pair_index"])
    assert again == info["sections"]


def test_the_construction_closes_under_auto_hyperface():
    """The point of carrying the pairs is that the group can then BOUND something."""
    from rexgraph.faces import auto_hyperface
    rex, _info = from_groups([["a", "b", "c", "d"], ["b", "c", "d", "e"],
                              ["c", "d", "e", "f"]])
    n = auto_hyperface(rex)
    assert n > 0, "a group whose pairs span its boundary must close"
    assert len(rex.graded_boundaries()) >= 2


#### text ######################################################################

_SENTS = ["the cat sat on the mat",
          "the dog sat on the rug",
          "a cat and a dog met",
          "the mat and the rug were red",
          "the cat met the dog on the mat"]


def test_from_text_makes_each_sentence_a_group_over_its_words():
    from rexgraph.construct import from_text
    rex, info = from_text(None, sentences=_SENTS, min_terms=3, min_pair_count=1)
    assert info["n_sentences"] == len(_SENTS)
    assert "cat" in info["vertex_of"] and "the" in info["vertex_of"]
    r1 = int(rex.rank_tower()["ranks"][0])
    assert int(rex.nE) - r1 > 0, "the mixed construction must open a cycle space"
    # a sentence is a THING: it gets its own vertex, distinguished
    spans = [list(map(int, s)) for s in rex.relation_supports()]
    assert spans[0][0] == 0, "sentence 0 owns vertex 0, at position 0"


def test_stopwords_are_the_callers_decision_and_change_the_object():
    """The two readings a corpus supports want different tokenisations."""
    from rexgraph.construct import from_text
    keep, _i = from_text(None, sentences=_SENTS, min_terms=3, min_pair_count=1)
    drop, _j = from_text(None, sentences=_SENTS, min_terms=2, min_pair_count=1,
                         stopwords={"the", "a", "and", "on", "were"})
    assert int(keep.nV) > int(drop.nV), "function words are most of the vocabulary"


def test_sequences_keep_the_order_the_group_threw_away():
    from rexgraph.construct import from_text
    _rex, info = from_text(None, sentences=["the cat sat on the mat"],
                           min_terms=3, min_pair_count=1)
    assert info["sequences"][0] == ["the", "cat", "sat", "on", "the", "mat"]
    # the GROUP is the set, deduplicated and order-preserving
    assert info["sequences"][0].count("the") == 2


def test_precedence_is_signed_against_the_pair_orientation():
    from rexgraph.construct import from_text, precedence_field
    rex, info = from_text(None, sentences=["alpha beta gamma"] * 3,
                          min_terms=3, min_pair_count=1)
    f = precedence_field(info)
    assert f.shape == (int(rex.nE),)
    assert np.abs(f).sum() > 0
    # the wide relations carry no precedence; only the pairs do
    assert np.all(f[:info["n_wide"]] == 0)


def test_adjacent_only_counts_fewer_pairs_than_all_pairs():
    from rexgraph.construct import from_text, precedence_field
    _rex, info = from_text(None, sentences=_SENTS, min_terms=3, min_pair_count=1)
    a = precedence_field(info, adjacent_only=True)
    b = precedence_field(info, adjacent_only=False)
    assert int((a != 0).sum()) < int((b != 0).sum())


def test_first_occurrences_reduces_and_preserves_order():
    from rexgraph.construct import first_occurrences
    assert first_occurrences([["a", "b", "a", "c"]]) == [["a", "b", "c"]]
    assert first_occurrences([[]]) == [[]]


def test_reducing_after_shuffling_inverts_the_control():
    """The confound, pinned. It cost an opposite conclusion twice.

    An all-pairs precedence reading uses each token's position, so a repeated token holds
    the earliest of several positions. Permuting a sequence that still carries repeats
    leaves that multiplicity channel intact, so a control built that way is not a control.
    Reduce FIRST, then shuffle the reduced sequences.
    """
    from rexgraph.construct import first_occurrences, from_text, precedence_field
    rng = np.random.default_rng(3)
    sents = [" ".join(rng.choice(list("abcdefgh"), 9)) for _ in range(120)]
    rex, info = from_text(None, sentences=sents, min_terms=3, min_pair_count=1)
    red = first_occurrences(info["sequences"])

    def grad(seqs):
        from rexgraph.partition import hodge_share
        f = precedence_field(info, seqs)
        return hodge_share(rex, f)["share"]["gradient"] if f.any() else 0.0

    right = [grad([list(rng.permutation(t)) for t in red]) for _ in range(5)]
    wrong = [grad(first_occurrences([list(rng.permutation(t))
                                     for t in info["sequences"]])) for _ in range(5)]
    assert abs(float(np.mean(right)) - float(np.mean(wrong))) > 1e-6, (
        "reducing before vs after shuffling must give different answers, or the "
        "confound this test exists for has silently gone away")


def test_from_text_refuses_when_the_filters_leave_nothing():
    from rexgraph.construct import from_text
    with pytest.raises(ValueError, match="no sentence survived"):
        from_text(None, sentences=_SENTS, min_terms=99)


def test_from_text_needs_text_or_sentences():
    from rexgraph.construct import from_text
    with pytest.raises(ValueError, match="either text or sentences"):
        from_text(None)


#### spans #####################################################################

_D = {"the", "a", "and", "on", "of", "to", "in"}


def test_a_delimiter_gates_the_span_without_joining_it():
    """Existence and orientation are separate operators and a gate is blind to the
    second, so the delimiter is not in the support and does not head anything. It used to
    be kept at the front and called the distinguished vertex, which made a comma carry
    the -1 of a semantic relation."""
    from rexgraph.construct import spans_of
    assert spans_of(["the", "cat", "sat", "on", "the", "mat"], _D) == [
        ["cat", "sat"], ["mat"]]
    # a span of one content token is a WITNESS, returned as such rather than filtered
    assert spans_of(["on", "the", "mat"], _D) == [["mat"]]
    # content before any gate is its own span
    assert spans_of(["cats", "sat"], _D) == [["cats", "sat"]]
    # gates with no content between them produce no span: they bound nothing
    assert spans_of(["the", "the"], _D) == []
    assert spans_of([], _D) == []


def test_the_gate_is_reported_beside_the_span_not_inside_it():
    """The gate is real information (it is what an attributed boundary records) so it
    comes back separately rather than being lost or smuggled into the support."""
    from rexgraph.construct import spans_of
    spans, gates = spans_of(["the", "cat", "sat", "on", "the", "mat"], _D,
                            with_gates=True)
    assert spans == [["cat", "sat"], ["mat"]]
    assert gates == [["on"], []]


def test_the_head_is_the_first_content_token_not_the_gate():
    """Orientation is position within the span, and the gate is not in the span."""
    from rexgraph.construct import from_spans, spans_of
    spans = spans_of(["the", "cat", "sat", "on", "the", "mat"], _D)
    rex, info = from_spans(spans, min_pair_count=1)
    assert "the" not in info["vertex_of"], "a gate is not a participant"
    heads = [list(map(int, s))[0] for s in rex.relation_supports()[:len(info["spans"])]]
    assert heads[0] == info["vertex_of"]["cat"], "the first CONTENT token heads it"


def test_multiplicity_is_separate_cells_so_nothing_needs_deduplicating():
    """The other half of the reframe.

    A repeated token sits in different spans, which are different relations sharing a
    boundary vertex. There is no first-occurrence rule here, so there is nothing for a
    shuffle control to get wrong.
    """
    from rexgraph.construct import from_spans, spans_of
    toks = ["the", "cat", "and", "the", "dog", "and", "cat", "bird"]
    spans = spans_of(toks, _D)
    rex, info = from_spans(spans, min_pair_count=1)
    assert len(info["spans"]) >= 3
    v_cat = info["vertex_of"]["cat"]
    touching = [i for i, s in enumerate(rex.relation_supports())
                if v_cat in [int(x) for x in s]]
    assert len(touching) > 1, "one token type, many cells"


def test_sentence_sections_are_the_grade_two_candidate():
    from rexgraph.construct import from_spans, spans_of
    sents = ["the cat sat on the mat", "a dog and a bird"]
    spans, sent_of = [], {}
    for si, s in enumerate(sents):
        for sp in spans_of(s.split(), _D):
            sent_of[len(spans)] = si; spans.append(sp)
    rex, info = from_spans(spans, min_pair_count=1, sentence_of=sent_of)
    assert set(info["sentence_sections"]) == {0, 1}
    for rels in info["sentence_sections"].values():
        assert rels and max(rels) < int(rex.nE)


def test_a_one_token_span_builds_a_witness_rather_than_being_refused():
    """It used to raise, on the belief that a one-token span "is not a relation". It is:
    a witness, `(+1)`, which exists and bounds nothing."""
    from rexgraph.construct import from_spans
    rex, info = from_spans([["solo"], ["x"], ["y", "y"]], verify=False)
    assert int(rex.nE) == 3
    assert sorted(map(int, rex.edge_types)) == [3, 3, 3], "three witnesses"


def test_from_spans_refuses_only_when_there_is_no_token_at_all():
    from rexgraph.construct import from_spans
    with pytest.raises(ValueError, match="nothing to build"):
        from_spans([[], []])


def test_spans_feed_the_readings_like_any_other_section():
    from rexgraph.construct import from_spans, spans_of
    from rexgraph.partition import section_readings
    toks = ["the", "cat", "sat", "on", "the", "mat", "and", "a", "dog", "ran", "to", "the", "park"]
    rex, info = from_spans(spans_of(toks, _D), min_pair_count=1)
    out = section_readings(rex, {f"s{k}": v for k, v in info["sections"].items()},
                           verify=True)
    assert len(out) == len(info["spans"])


#### the near-linear rank #######################################################

def test_mixed_rank_is_exact_when_it_answers():
    from rexgraph.construct import from_groups, mixed_rank
    from rexgraph.graded_boundary import _sparse_rank
    rng = np.random.default_rng(1)
    for groups in ([["a", "b", "c"], ["b", "c", "d"]],
                   [["a", "b", "c"], ["d", "e", "f"]],
                   [sorted(rng.choice(40, 5, replace=False).tolist()) for _ in range(30)]):
        rex, info = from_groups(groups, verify=False)
        fast = mixed_rank(rex, info)
        assert fast is not None
        assert fast == _sparse_rank(rex._integer_B1().tocsc())


def test_mixed_rank_refuses_a_fragmented_group():
    """The guard is the point: a dropped pair can disconnect a group, and then its
    wide column is no longer spanned and DOES add rank. Measured, the unguarded
    shortcut read 35 against a true 49."""
    from rexgraph.construct import from_groups, mixed_rank
    rng = np.random.default_rng(1)
    groups = [sorted(rng.choice(50, 5, replace=False).tolist()) for _ in range(40)]
    rex, info = from_groups(groups, min_pair_count=2, verify=False)
    assert mixed_rank(rex, info) is None, "a fragmented group must refuse, not guess"


def test_mixed_rank_refuses_a_complex_it_does_not_describe():
    from rexgraph.construct import from_groups, mixed_rank
    rex, info = from_groups([["a", "b", "c"], ["b", "c", "d"]])
    other, _ = from_groups([["x", "y", "z"], ["y", "z", "w"], ["z", "w", "v"]])
    bad = dict(info); bad["n_wide"] = 99
    assert mixed_rank(other, bad) is None


def test_the_identity_it_rests_on_is_arity_two_only():
    """dim ker(L0) == components holds at arity 2 and fails above it.

    A lone arity-4 relation has rank 1, so dim ker(L0) is 3, while the support is one
    connected component. That is why `mixed_rank` leans on the PAIR graph rather than on
    the whole complex, and why `_pairwise_rank` guards on column arity.
    """
    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.graded_boundary import _beta0_components
    for rels, arity_two in (([[0, 1], [1, 2], [2, 3]], True), ([[0, 1, 2, 3]], False)):
        ptr, idx = [0], []
        for r in rels:
            idx += r; ptr.append(len(idx))
        rex = RexGraph.from_hypergraph(np.asarray(ptr, np.int64), np.asarray(idx, np.int64))
        B = to_scipy_csr(rex._B1_dual).tocsc()
        ker = int(rex.nV) - np.linalg.matrix_rank((B @ B.T).toarray())
        comp = _beta0_components(B)
        assert bool(ker == comp) is arity_two, f"arity_two={arity_two}: ker {ker} comp {comp}"


def test_pair_mode_spanning_keeps_the_rank_and_drops_the_invented_cycles():
    """A group is ONE fact; the clique asserts C(k,2) pairwise facts it never stated.

    A connected set's zero-sum space has dimension k-1, so a spanning subset already
    spans the group's own column: the rank is unchanged and the group still closes. What
    goes away is cycles the data never asserted.
    """
    from rexgraph.construct import from_groups
    from rexgraph.graded_boundary import _sparse_rank
    rng = np.random.default_rng(0)
    groups = [sorted(rng.choice(60, int(rng.integers(3, 14)), replace=False).tolist())
              for _ in range(40)]
    groups.append(sorted(rng.choice(60, 40, replace=False).tolist()))   # one wide group

    cl, ci = from_groups(groups, pair_mode="clique", verify=False)
    sp, si = from_groups(groups, pair_mode="spanning", verify=False)
    r_cl = _sparse_rank(cl._integer_B1().tocsc())
    r_sp = _sparse_rank(sp._integer_B1().tocsc())
    assert r_cl == r_sp, "the rank does not depend on how a group is connected"
    assert si["n_pairs"] < ci["n_pairs"] / 4, "the wide group is where the saving is"
    assert int(sp.nE) - r_sp < int(cl.nE) - r_cl, "the clique invents cycles"


def test_pair_mode_spanning_still_lets_a_group_close():
    from rexgraph.construct import from_groups
    from rexgraph.faces import face_reading
    groups = [["a", "b", "c", "d", "e"], ["c", "d", "e", "f"]]
    rex, info = from_groups(groups, pair_mode="spanning", verify=False)
    st = face_reading(rex, info["sections"][0])["state"]
    assert st in ("bounds", "degenerate"), f"a spanned group must still close, got {st}"


def test_pair_mode_is_checked():
    from rexgraph.construct import from_groups
    with pytest.raises(ValueError, match="pair_mode must be"):
        from_groups([["a", "b", "c"]], pair_mode="star")
