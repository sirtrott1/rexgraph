"""Sense model against WordNet."""
import os
import random

import pytest

from agent.answerers.lexical import DEFAULT_WORDNET
from agent.senses import SenseModel

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_WORDNET),
                                reason="WordNet not on disk")


@pytest.fixture(scope="module")
def wn():
    from agent.adapters.lexical import load_wordnet
    raw = load_wordnet(DEFAULT_WORDNET)
    lemma = {e: str(w).lower() for e, w in (raw.get("lemma_of") or {}).items()}
    synsets = {}
    for sid, members in (raw.get("synsets") or {}).items():
        got = sorted({lemma[m] for m in members if m in lemma})
        if got:
            synsets[sid] = got
    by_lemma = {}
    for sid, words in synsets.items():
        for w in words:
            by_lemma.setdefault(w, []).append(sid)
    rel = {}
    for src, _kind, tgt in (raw.get("relations") or ()):
        if src in synsets and tgt in synsets:
            rel.setdefault(src, []).append(tgt)
    return synsets, by_lemma, rel


def sample(wn, n, *, hops=1, seed=1):
    synsets, by_lemma, rel = wn
    rng = random.Random(seed)
    cands = [w for w, ss in by_lemma.items() if len(ss) >= 2]
    ok = tied = wrong = decided = 0
    for w in rng.sample(cands, min(n * 4, len(cands))):
        if decided >= n:
            break
        m = SenseModel.for_word(w, synsets, by_lemma, rel, hops=hops)
        if m.d < 2:
            continue
        with_rel = [s for s in m.senses if rel.get(s)]
        if not with_rel:
            continue
        target = with_rel[rng.randrange(len(with_rel))]
        nbr = rel[target][rng.randrange(len(rel[target]))]
        probe = [x for x in synsets[nbr] if x != w and x not in synsets[target]]
        if not probe:
            continue
        r = m.disambiguate(probe)
        if r["abstain"]:
            continue
        decided += 1
        if r["senses"] == [target]:
            ok += 1
        elif target in r["senses"]:
            tied += 1
        else:
            wrong += 1
    return decided, ok, tied, wrong


def test_the_extent_beats_chance_by_a_wide_margin(wn):
    decided, ok, _tied, _wrong = sample(wn, 120)
    assert decided >= 60
    assert ok / decided > 0.70            # chance is ~0.41


def test_failures_are_TIES_and_not_errors(wn):
    decided, ok, tied, wrong = sample(wn, 200)
    assert ok + tied + wrong == decided
    assert wrong / decided < 0.02


def test_membership_only_cannot_decide(wn):
    decided, _ok, _tied, _wrong = sample(wn, 120, hops=0)
    assert decided < 20


def test_widening_the_scale_dilutes(wn):
    d1, ok1, tie1, _ = sample(wn, 150, hops=1)
    d2, ok2, tie2, _ = sample(wn, 150, hops=2)
    assert ok1 / d1 > ok2 / d2
    assert tie2 / d2 > tie1 / d1
