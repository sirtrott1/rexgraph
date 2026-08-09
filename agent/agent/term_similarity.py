"""
agent.term_similarity: how much two terms share, over all of it.

Two terms in a hierarchy share their common ancestors. How much they share is a
question about that whole set, and the standard measures answer it with one element
of the set: Resnik takes the most informative common ancestor and reports its
information content, and Lin normalises that same single number by the two terms'
own. Everything else the terms have in common is discarded.

The exact object is the shared mass over the whole set. Give each term the
distribution its ancestors induce and the shared mass is the overlap coefficient::

    overlap(a, b) = sum_c min(p_a(c), p_b(c))     over every common ancestor
                  = 1 - TV(p_a, p_b)

which is a proper measure of agreement between distributions, is exactly rational when
the weights are, and equals 1 exactly when the two terms have the same up-set.

The relationship to the standard measures is a reduction, not a rivalry. Over the same
weighted set:

    Resnik  = max over the shared ancestors      (the l-infinity reduction)
    shared  = sum over the shared ancestors      (the l-1 reduction)

so `Resnik <= shared`, with equality exactly when at most one shared ancestor carries
weight. Every further thing two terms share that carries weight widens the gap, and
the gap is what Resnik cannot see. Two pairs with the same most-informative ancestor
and different amounts of shared structure get the same Resnik score and different
overlaps.

The "carries weight" qualifier is not a technicality. Under information content an
ancestor annotated by everything has `IC = -log(1) = 0`, so it enters neither the
maximum nor the sum, and a pair sharing three ancestors two of which are the universal
ones loses nothing by keeping one. The gap opens where the shared ancestors are
genuinely informative, which is where it matters.

The weight is a parameter, and that matters. Information content is defined from
annotation frequency, so a Resnik or Lin score between two terms changes when a
different corpus is loaded: the same two terms are more or less similar depending on
how much they happen to have been studied. The default here is uniform, which depends
on the hierarchy alone. Pass `weight` to reproduce an information-content reading, and
the reduction above holds for whatever weight is passed.

Nothing in this module implements Resnik or Lin. They are approximations of the object
computed here, and the comparison against them lives in the benchmark, not in the
library surface.

**This is not the fiber similarity, and the two do not substitute for each other.**
`RexGraph.spread_similarity` reads a term's position in the complex: its degree, its
co-participation, its orientation agreement. It does not track ancestry and does not
substitute for this: use this for "how much do these two terms share", and the fiber
similarity for "do these two terms sit in the same kind of structural position".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

#: relation names that place a term under another
HIERARCHY = frozenset({"is_a", "isa", "subclassof", "part_of", "partof"})


@dataclass
class TermHierarchy:
    """A term hierarchy, and the up-set each term induces."""

    parents: dict[str, set] = field(default_factory=dict)
    _ancestors: dict[str, frozenset] = field(default_factory=dict, repr=False)

    @property
    def terms(self) -> list[str]:
        out = set(self.parents)
        for ps in self.parents.values():
            out |= ps
        return sorted(out)

    def ancestors(self, term: str) -> frozenset:
        """`term` and everything above it.

        Includes the term itself, because a term is its own most specific common
        ancestor with itself and excluding it would make a term less than perfectly
        similar to itself. A cycle in the hierarchy stops the walk rather than
        hanging it.
        """
        if term in self._ancestors:
            return self._ancestors[term]
        seen, stack = {term}, [term]
        while stack:
            t = stack.pop()
            for p in self.parents.get(t, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        out = frozenset(seen)
        self._ancestors[term] = out
        return out

    def shared_ancestors(self, a: str, b: str) -> frozenset:
        return self.ancestors(a) & self.ancestors(b)


def hierarchy_from_triples(triples) -> TermHierarchy:
    """Build the hierarchy from (subject, predicate, object) triples."""
    parents: dict[str, set] = {}
    for s, p, o in triples:
        if str(p).split(":")[-1].strip().lower() in HIERARCHY:
            parents.setdefault(str(s), set()).add(str(o))
    return TermHierarchy(parents)


def hierarchy_from_knowledge(knowledge) -> TermHierarchy:
    """Build the hierarchy from a joined complex, using its relation types."""
    parents: dict[str, set] = {}
    for s, rel, o, _origin in knowledge.edges:
        if str(rel).split(":")[-1].strip().lower() in HIERARCHY:
            parents.setdefault(knowledge.display(s), set()).add(knowledge.display(o))
    return TermHierarchy(parents)


def _weights(h: TermHierarchy, term: str, weight) -> dict:
    """The weight each ancestor of `term` carries, as exact Fractions."""
    anc = h.ancestors(term)
    if weight is None:
        return {c: Fraction(1) for c in anc}
    out = {}
    for c in anc:
        w = weight.get(c, 0)
        out[c] = w if isinstance(w, Fraction) else Fraction(w).limit_denominator(10**9)
    return out


def shared_mass(h: TermHierarchy, a: str, b: str, *, weight=None) -> Fraction:
    """Total weight of everything two terms have in common.

    The l-1 reduction of the shared ancestor set. Resnik reports the maximum over the
    same set, so this is never smaller, and equal exactly when at most one of the
    shared ancestors carries weight.
    """
    wa = _weights(h, a, weight)
    wb = _weights(h, b, weight)
    return sum((wa[c] for c in (set(wa) & set(wb))), Fraction(0))


def ancestor_overlap(h: TermHierarchy, a: str, b: str, *, weight=None) -> Fraction:
    """`sum_c min(p_a(c), p_b(c))`: the shared mass, normalised as distributions.

    Exactly rational. 1 when the two terms have the same up-set, 0 when they share
    nothing, and in between it is the fraction of each term's own mass that the other
    accounts for. This is the overlap coefficient, equivalently `1 - TV`.
    """
    wa, wb = _weights(h, a, weight), _weights(h, b, weight)
    ta = sum(wa.values(), Fraction(0))
    tb = sum(wb.values(), Fraction(0))
    if ta == 0 or tb == 0:
        return Fraction(0)
    pa = {c: w / ta for c, w in wa.items()}
    pb = {c: w / tb for c, w in wb.items()}
    return sum((min(pa[c], pb[c]) for c in (set(pa) & set(pb))), Fraction(0))


def overlap_matrix(h: TermHierarchy, terms=None, *, weight=None):
    """Pairwise overlaps, exactly. Returns `(terms, matrix_of_Fractions)`."""
    ts = list(terms) if terms is not None else h.terms
    n = len(ts)
    M = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = Fraction(1)
        for j in range(i + 1, n):
            v = ancestor_overlap(h, ts[i], ts[j], weight=weight)
            M[i][j] = M[j][i] = v
    return ts, M


def discrimination(h: TermHierarchy, a: str, b: str, *, weight=None) -> dict:
    """What the single-ancestor reading loses on this pair.

    `n_shared` is how many common ancestors there are; a reading that keeps one of
    them discards `n_shared - 1`. `mass_outside_the_largest` is the weight it
    discards, which is 0 exactly when the reduction is lossless.

    Lossless is about WEIGHT, not count. Under information content an ancestor
    annotated by everything has IC 0, so it contributes to neither the maximum nor the
    sum: a pair sharing three ancestors two of which are weightless loses nothing by
    keeping one. `n_shared_with_weight` is the number that actually matters.
    """
    wa, wb = _weights(h, a, weight), _weights(h, b, weight)
    common = set(wa) & set(wb)
    if not common:
        return {"n_shared": 0, "n_shared_with_weight": 0,
                "shared_mass": Fraction(0), "largest_single": Fraction(0),
                "mass_outside_the_largest": Fraction(0), "lossless": True}
    masses = sorted((wa[c] for c in common), reverse=True)
    total = sum(masses, Fraction(0))
    largest = masses[0]
    n_weighted = sum(1 for m in masses if m != 0)
    return {
        "n_shared": len(common),
        "n_shared_with_weight": n_weighted,
        "shared_mass": total,
        "largest_single": largest,
        "mass_outside_the_largest": total - largest,
        "lossless": n_weighted <= 1,
    }
