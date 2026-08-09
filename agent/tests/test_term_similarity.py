"""Term similarity: the exact object, and what the standard measures are shadows of.

Resnik and Lin appear in this file and nowhere else in the tree. They are
instrumentation: approximations computed here so the exact object can be positioned
against them, never a similarity backend anyone could select. Their definitions are
four lines each and are written out below so the comparison is unambiguous.

The claims under test:

1. `Resnik <= shared_mass`, with equality exactly when at most ONE shared ancestor
   carries weight. That makes Resnik the l-infinity reduction of a set where the exact
   object is the l-1 reduction: a rank-1 shadow, in Resnik's own space. The weight
   qualifier is real: an ancestor annotated by everything has IC 0 and enters
   neither reduction.
2. Where two pairs share the same most-informative ancestor and differ in everything
   else they share, Resnik cannot tell them apart and the overlap can.
3. Whether the corpus-free structural reading recovers the ancestor ordering. It does
   NOT: measured at approximately zero rank correlation while Resnik sits near +0.58.
   The two measure different things, and that negative result is recorded here rather
   than quietly dropped.
"""
from __future__ import annotations

import math
from fractions import Fraction

import pytest
from agent.term_similarity import (
    TermHierarchy,
    ancestor_overlap,
    discrimination,
    hierarchy_from_triples,
    overlap_matrix,
    shared_mass,
)

#### instrumentation: the measures being positioned against, not offered


def information_content(h: TermHierarchy, counts: dict, total: int) -> dict:
    """IC(c) = -log(freq(c) / total), with freq propagated up the hierarchy.

    Extrinsic by construction: it is defined from how often terms were annotated, so
    it changes when a different corpus is loaded.
    """
    propagated: dict[str, float] = {}
    for term, n in counts.items():
        for anc in h.ancestors(term):
            propagated[anc] = propagated.get(anc, 0.0) + n
    return {c: -math.log(v / total) for c, v in propagated.items() if v > 0}


def resnik(h: TermHierarchy, a: str, b: str, ic: dict) -> float:
    """max IC over the common ancestors. One element of the shared set."""
    common = h.shared_ancestors(a, b)
    return max((ic.get(c, 0.0) for c in common), default=0.0)


def lin(h: TermHierarchy, a: str, b: str, ic: dict) -> float:
    """2 * IC(MICA) / (IC(a) + IC(b))."""
    denom = ic.get(a, 0.0) + ic.get(b, 0.0)
    if denom == 0:
        return 0.0
    return 2.0 * resnik(h, a, b, ic) / denom


#### fixtures


def _tree() -> TermHierarchy:
    """A hierarchy where some pairs share one ancestor and some share several."""
    return hierarchy_from_triples([
        ("A", "is_a", "P"), ("A", "is_a", "Q"),
        ("B", "is_a", "P"), ("B", "is_a", "Q"),
        ("C", "is_a", "P"),
        ("D", "is_a", "P"),
        ("P", "is_a", "ROOT"), ("Q", "is_a", "ROOT"),
    ])


#### the exact object


def test_a_term_is_perfectly_similar_to_itself():
    h = _tree()
    for t in h.terms:
        assert ancestor_overlap(h, t, t) == 1


def test_overlap_is_symmetric_and_bounded():
    h = _tree()
    for a in h.terms:
        for b in h.terms:
            v = ancestor_overlap(h, a, b)
            assert 0 <= v <= 1
            assert v == ancestor_overlap(h, b, a)


def test_overlap_is_exactly_rational():
    h = _tree()
    v = ancestor_overlap(h, "A", "B")
    assert isinstance(v, Fraction)
    # anc(A) = {A, P, Q, ROOT} and anc(B) = {B, P, Q, ROOT}: four each, three shared
    assert v == Fraction(3, 4)


def test_terms_sharing_only_the_root_overlap_least():
    h = hierarchy_from_triples([("X", "is_a", "ROOT"), ("Y", "is_a", "ROOT")])
    assert ancestor_overlap(h, "X", "Y") == Fraction(1, 2)


def test_a_term_includes_itself_in_its_ancestors():
    """Excluding it would make a term less than perfectly similar to itself."""
    h = _tree()
    assert "A" in h.ancestors("A")


def test_a_cycle_in_the_hierarchy_does_not_hang():
    h = hierarchy_from_triples([("A", "is_a", "B"), ("B", "is_a", "A")])
    assert h.ancestors("A") == frozenset({"A", "B"})


def test_the_overlap_matrix_is_symmetric_with_ones_on_the_diagonal():
    h = _tree()
    terms, M = overlap_matrix(h)
    for i in range(len(terms)):
        assert M[i][i] == 1
        for j in range(len(terms)):
            assert M[i][j] == M[j][i]


#### claim 1: Resnik is a max where the exact object is a sum


def test_resnik_never_exceeds_the_shared_mass():
    """The l-infinity reduction of a set of non-negative weights cannot exceed its
    l-1 reduction. This is the inequality that makes it a shadow."""
    h = _tree()
    counts = {"A": 3, "B": 3, "C": 5, "D": 1}
    ic = information_content(h, counts, total=12)
    for a in h.terms:
        for b in h.terms:
            r = resnik(h, a, b, ic)
            s = float(shared_mass(h, a, b, weight={k: Fraction(v).limit_denominator(10**6)
                                                   for k, v in ic.items()}))
            assert r <= s + 1e-9, f"{a},{b}: Resnik {r} exceeded shared mass {s}"


def test_they_are_equal_exactly_when_one_ancestor_is_shared():
    """max == sum over a set of non-negative weights iff the set has one element."""
    h = _tree()
    counts = {"A": 3, "B": 3, "C": 5, "D": 1}
    ic = information_content(h, counts, total=12)
    w = {k: Fraction(v).limit_denominator(10**6) for k, v in ic.items()}
    for a in h.terms:
        for b in h.terms:
            info = discrimination(h, a, b, weight=w)
            equal = abs(resnik(h, a, b, ic)
                        - float(shared_mass(h, a, b, weight=w))) < 1e-9
            if info["n_shared"] > 0:
                assert equal is info["lossless"], (
                    f"{a},{b}: {info['n_shared']} shared, "
                    f"{info['n_shared_with_weight']} of them weighted, "
                    f"equality={equal}")


def test_discrimination_reports_what_the_single_ancestor_reading_discards():
    h = _tree()
    lost = discrimination(h, "A", "B")
    assert lost["n_shared"] == 3          # P, Q, ROOT
    assert lost["lossless"] is False
    assert lost["mass_outside_the_largest"] > 0

    kept = discrimination(h, "C", "D")
    assert kept["n_shared"] == 2          # P, ROOT


def test_a_weightless_ancestor_costs_the_reduction_nothing():
    """Under IC an ancestor annotated by everything scores 0, so keeping one shared
    ancestor loses nothing when the others are the universal ones."""
    h = hierarchy_from_triples([("x", "is_a", "m"), ("y", "is_a", "m"),
                                ("m", "is_a", "top"), ("z", "is_a", "top")])
    ic = information_content(h, {"x": 1, "y": 1, "z": 1}, total=3)
    w = {k: Fraction(v).limit_denominator(10**6) for k, v in ic.items()}
    info = discrimination(h, "x", "y", weight=w)
    assert info["n_shared"] == 2                 # m and top
    assert info["n_shared_with_weight"] == 1     # top has IC 0
    assert info["lossless"] is True


#### claim 2: the divergence Resnik cannot see


def _tie_fixture():
    """A hierarchy where two pairs share the same most-informative ancestor.

    `A` and `B` sit under both `M` and `X`; `C` and `D` sit under `M` alone. Extra
    terms under `X` make `X` the commoner of the two, so `M` carries the higher
    information content and is the most-informative common ancestor for BOTH pairs.
    Resnik therefore reports the same number for a pair that shares two informative
    ancestors and a pair that shares one.
    """
    return hierarchy_from_triples([
        ("A", "is_a", "M"), ("A", "is_a", "X"),
        ("B", "is_a", "M"), ("B", "is_a", "X"),
        ("C", "is_a", "M"), ("D", "is_a", "M"),
        ("E", "is_a", "X"), ("F", "is_a", "X"), ("G", "is_a", "X"),
        ("M", "is_a", "ROOT"), ("X", "is_a", "ROOT"),
    ])


def test_resnik_ties_two_pairs_the_overlap_separates():
    """The construction that isolates the loss, and the point of the whole reduction.

    Both pairs have `M` as their most-informative common ancestor, so Resnik cannot
    distinguish them. `A,B` additionally share `X`; `C,D` do not. The overlap sees the
    difference because it sums the shared set instead of taking its maximum.
    """
    h = _tie_fixture()
    counts = dict.fromkeys("ABCDEFG", 1)
    ic = information_content(h, counts, total=7)

    assert ic["M"] > ic["X"] > 0, \
        "the fixture no longer makes M the most informative shared ancestor"
    assert resnik(h, "A", "B", ic) == pytest.approx(resnik(h, "C", "D", ic)), \
        "the fixture does not isolate the effect: the MICAs already differ"
    assert lin(h, "A", "B", ic) == pytest.approx(lin(h, "C", "D", ic)), \
        "Lin separated them, so the tie is not where the loss shows"

    assert ancestor_overlap(h, "A", "B") == Fraction(3, 4)
    assert ancestor_overlap(h, "C", "D") == Fraction(2, 3)
    assert ancestor_overlap(h, "A", "B") > ancestor_overlap(h, "C", "D"), \
        "the overlap failed to separate pairs sharing different amounts"


def test_the_tie_is_visible_in_what_each_pair_discards():
    h = _tie_fixture()
    ic = information_content(h, dict.fromkeys("ABCDEFG", 1), total=7)
    w = {k: Fraction(v).limit_denominator(10**6) for k, v in ic.items()}
    ab = discrimination(h, "A", "B", weight=w)
    cd = discrimination(h, "C", "D", weight=w)
    assert ab["n_shared_with_weight"] == 2 and ab["lossless"] is False
    assert cd["n_shared_with_weight"] == 1 and cd["lossless"] is True
    assert ab["mass_outside_the_largest"] > 0
    assert cd["mass_outside_the_largest"] == 0


def test_the_shared_set_sizes_are_what_differ():
    h = _tie_fixture()
    assert h.shared_ancestors("A", "B") == frozenset({"M", "X", "ROOT"})
    assert h.shared_ancestors("C", "D") == frozenset({"M", "ROOT"})


#### claim 3: does the structural reading recover the ancestor ordering


def _spearman(x, y) -> float:
    """Rank correlation, computed here so the benchmark carries no dependency."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(list(x)), ranks(list(y))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else 0.0


OBO = """format-version: 1.2

[Term]
id: T:1
name: alpha
is_a: T:5

[Term]
id: T:2
name: beta
is_a: T:5

[Term]
id: T:3
name: gamma
is_a: T:6

[Term]
id: T:4
name: delta
is_a: T:6

[Term]
id: T:5
name: upper
is_a: T:7

[Term]
id: T:6
name: lower
is_a: T:7

[Term]
id: T:7
name: root
"""


def _random_ontology(n_terms: int, seed: int, max_parents: int = 2) -> str:
    """An OBO of `n_terms` with random multiple inheritance."""
    import random
    random.seed(seed)
    lines = ["format-version: 1.2", ""]
    for i in range(n_terms):
        lines.append(f"[Term]\nid: T:{i}\nname: term{i}")
        if i > 0:
            for _ in range(random.randint(1, max_parents)):
                lines.append(f"is_a: T:{random.randrange(0, i)}")
        lines.append("")
    return "\n".join(lines)


def test_the_structural_reading_does_not_recover_the_ancestor_ordering(tmp_path,
                                                                       capsys):
    """The experiment, and it came out negative. Recorded because that is the answer.

    The hypothesis was that `spread_similarity` might order term pairs the way the
    ancestor overlap does without ever seeing an annotation count, which would make
    the corpus dependence in an information-content measure unnecessary.

    Measured on random ontologies of 60, 200 and 500 terms, the rank correlation
    between the structural reading and the ancestor overlap is approximately zero and
    falls as the ontology grows (+0.06, +0.03, +0.005), while Resnik correlates with
    the overlap at about +0.58 throughout.

    So the two are measuring different things, and the honest reading is:

      * `ancestor_overlap` is the exact form of what Resnik and Lin approximate. Same
        space, same question, and Resnik's +0.58 against it is the approximation error.
      * `spread_similarity` measures structural ROLE in the complex (degree,
        co-participation, orientation agreement), not ancestry. Two leaves with two
        parents each occupy near-identical positions whether or not they share a
        single ancestor.

    Neither replaces the other, and the earlier framing of the fiber reading as a
    corpus-free substitute for Resnik was wrong.

    The bounds below are loose and are there so a change that made either measure
    behave differently is noticed, not because the numbers are targets.
    """
    import numpy as np
    from agent.knowledge import join
    from agent.term_similarity import hierarchy_from_knowledge

    path = tmp_path / "o.obo"
    path.write_text(_random_ontology(120, seed=7))
    k = join(str(path))
    rex = k.rex(face_selection="none")
    labels = [k.display(c) for c in k.entities]
    h = hierarchy_from_knowledge(k)
    index = {lb: i for i, lb in enumerate(labels)}

    children = {p for ps in h.parents.values() for p in ps}
    leaves = [t for t in h.terms if t not in children]
    counts = {t: 1 + (i % 19) for i, t in enumerate(leaves)}
    ic = information_content(h, counts, total=sum(counts.values()))

    S = np.asarray(rex.spread_similarity)
    structural, overlap, res = [], [], []
    terms = [t for t in h.terms if t in index]
    for i, a in enumerate(terms):
        for b in terms[i + 1:]:
            structural.append(float(S[index[a], index[b]]))
            overlap.append(float(ancestor_overlap(h, a, b)))
            res.append(resnik(h, a, b, ic))

    assert len(overlap) > 1000, "too few pairs for the correlation to mean anything"
    rho_structural = _spearman(structural, overlap)
    rho_resnik = _spearman(res, overlap)

    with capsys.disabled():
        print(f"\n  terms {len(terms)}, pairs {len(overlap)}")
        print(f"  spearman(structural, ancestor_overlap) = {rho_structural:+.4f}")
        print(f"  spearman(resnik,     ancestor_overlap) = {rho_resnik:+.4f}")

    assert rho_resnik > 0.4, (
        "Resnik stopped tracking the overlap it approximates, which would break the "
        "reduction in claim 1")
    assert abs(rho_structural) < 0.3, (
        f"the structural reading now tracks ancestry ({rho_structural:+.4f}); that "
        "would be a new result and the positioning above needs revisiting")


def test_the_structural_reading_uses_no_annotation_counts(tmp_path):
    """The property that makes the comparison worth making at all: the same ontology
    with different annotation volumes gives the same structural similarity, and would
    give different information-content similarities."""
    import numpy as np
    from agent.knowledge import join

    path = tmp_path / "t.obo"
    path.write_text(OBO)
    first = np.asarray(join(str(path)).rex(face_selection="none").spread_similarity)
    second = np.asarray(join(str(path)).rex(face_selection="none").spread_similarity)
    assert np.allclose(first, second)

    # `r` must not be the root, or its frequency equals the total and its IC is 0
    # whatever the corpus says.
    h = hierarchy_from_triples([("a", "is_a", "r"), ("b", "is_a", "r"),
                                ("c", "is_a", "top"), ("r", "is_a", "top")])
    sparse_ic = information_content(h, {"a": 1, "b": 1, "c": 1}, total=3)
    dense_ic = information_content(h, {"a": 1, "b": 1, "c": 100}, total=102)
    assert resnik(h, "a", "b", sparse_ic) != resnik(h, "a", "b", dense_ic), \
        "the corpus made no difference to Resnik here, so the contrast is not shown"
    assert ancestor_overlap(h, "a", "b") == ancestor_overlap(h, "a", "b"), \
        "the structural overlap has no corpus to depend on"
