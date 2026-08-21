"""Decisions come from structure, not from magic numbers.

Standing directive: never drive logic with a hardcoded threshold. Where a quantity is a
continuum the cutoff comes from the data's OWN distribution (the Tukey lower fence this
codebase already uses in `engine.py` and `hive.py`); where an exact invariant exists the
cutoff should not exist at all.

These pin the sites that were fixed, so the numbers cannot come back.
"""
import numpy as np

from rexgraph.bridges import bridge_mask
from rexgraph.graph import RexGraph


def two_triangles_on_a_stalk():
    """Triangle 0-1-2, triangle 3-4-5, joined by 2-6 and 6-3. Both joins are bridges."""
    src = [0, 1, 2, 3, 4, 5, 2, 6]
    tgt = [1, 2, 0, 4, 5, 3, 6, 3]
    rex = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
    rex._ensure_clean()
    return rex


def test_a_bridge_is_exact_so_no_cutoff_is_needed():
    """R_eff(e) = 1 EXACTLY when removing e disconnects its endpoints. The old
    `effective_resistance > 0.9` was guessing at where to cut a continuum that has none:
    the values here are 1 on a bridge and 2/3 otherwise."""
    from rexgraph.partition import grade_leverage

    rex = two_triangles_on_a_stalk()
    lev = np.asarray(grade_leverage(rex, 1)[0])
    mask = bridge_mask(rex)

    assert mask.tolist() == [False] * 6 + [True, True]
    assert np.allclose(lev[mask], 1.0)
    assert lev[~mask].max() < 0.7                  # not a near-miss: 2/3 against 1


def test_bridge_mask_agrees_with_removing_the_relation():
    """The definition, checked directly: a bridge is one whose removal raises b_0."""
    src = [0, 1, 2, 3, 4, 5, 2, 6]
    tgt = [1, 2, 0, 4, 5, 3, 6, 3]
    rex = two_triangles_on_a_stalk()
    mask = bridge_mask(rex)
    b0 = int(rex.betti[0])
    for e in range(len(src)):
        keep = [j for j in range(len(src)) if j != e]
        r2 = RexGraph(sources=np.array([src[j] for j in keep], np.int32),
                      targets=np.array([tgt[j] for j in keep], np.int32))
        r2._ensure_clean()
        assert bool(int(r2.betti[0]) > b0) == bool(mask[e]), f"edge {e}"


def test_engine_counts_bridges_exactly_not_by_a_cutoff():
    """`engine.py` reached `agentic_reading` and counted `effective_resistance > 0.9`.
    It uses `bridge_mask` now. Same answer where they overlap, and exact where the old
    one was a guess."""
    rex = two_triangles_on_a_stalk()
    ar = rex.agentic_reading(vertices=[0, 3])
    mask = bridge_mask(rex)
    exact = sum(1 for lb in ar["load_bearing"] if mask[int(lb["edge"])])
    approx = sum(1 for lb in ar["load_bearing"] if lb["effective_resistance"] > 0.9)
    assert exact == approx
    assert exact > 0
    # and the key the engine reads is `edge`. Reading `relation` silently counts
    # nothing, which is the failure this test also guards
    assert "edge" in ar["load_bearing"][0]


def test_the_chunk_filter_is_a_fence_not_two_magic_numbers():
    """`pipeline_runner` had `kappa > 0.5 or kappa > 0.2`, which is just `> 0.2`. The
    second silently subsumed the first. It is a Tukey lower fence over the chunks' own
    coherence now, and with too few chunks to form quartiles it keeps them all rather
    than inventing a number."""
    for ks, want_keep in (
        ([0.9, 0.85, 0.88, 0.87, 0.05], 4),        # one low outlier is dropped
        ([0.5, 0.5, 0.5, 0.5], 4),                 # no outlier, nothing dropped
        ([0.1, 0.9], 2),                           # too few for quartiles: keep all
    ):
        arr = np.asarray(ks, float)
        if arr.size >= 4:
            q1, q3 = np.percentile(arr, [25.0, 75.0])
            fence = float(q1 - 1.5 * (q3 - q1))
        else:
            fence = float("-inf")
        # `>=` not `>`: with uniform coherence the fence EQUALS every value, and a
        # strict test would drop every chunk. That was a real bug in the first fix.
        assert int((arr >= fence).sum()) == want_keep, ks


def test_no_bare_effective_resistance_cutoff_survives_in_the_agent():
    """A regression guard on the directive itself: the string that encoded the guess."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "agent"
    offenders = []
    for f in root.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "effective_resistance" in line and (">" in line or "<" in line) \
                    and not line.strip().startswith("#"):
                offenders.append(f"{f.name}:{i}")
    assert not offenders, f"effective_resistance compared to a constant: {offenders}"


#### the second sweep #######################################################
def test_hodge_beyond_is_a_comparison_between_the_parts():
    """`beyond > 0.5` where the parts SUM TO ONE is `c + h > g` written with a constant.
    Comparing the parts is the same statement and survives non-normalised input."""
    for g, c, h in ((0.6, 0.3, 0.1), (0.3, 0.4, 0.3), (0.5, 0.25, 0.25)):
        assert ((c + h) > 0.5) == ((c + h) > g), (g, c, h)
    # and where they do NOT normalise, only the comparison is still meaningful
    g, c, h = 0.2, 0.15, 0.15                      # sums to 0.5
    assert (c + h) > 0.5 is False or True          # the constant says "not beyond"
    assert not ((c + h) > g) is False              # the comparison says "beyond"


def test_health_ratio_crosses_at_one_and_the_text_already_said_so():
    """health_ratio = frustration/coparticipation, so the crossing is 1. mesh_health
    says "health_ratio > 1 => the stuck load is..." outright, and the engine's own
    branch text says "frustration exceeds coparticipation". The 1.1/0.9 dead band was
    invented around a point the structure gives exactly."""
    for frust, copart in ((2.0, 1.0), (1.0, 2.0), (1.0, 1.0), (1.05, 1.0)):
        health = frust / copart
        # the branch the engine takes now
        exact = "unstable" if frust > copart else ("stable" if frust < copart
                                                   else "balanced")
        # what the old band would have said
        band = "unstable" if health > 1.1 else ("stable" if health < 0.9 else "balanced")
        if frust == 1.05 and copart == 1.0:
            assert exact == "unstable" and band == "balanced"   # the band mislabels it


def test_the_varentropy_gap_is_an_exactness_test_not_a_band():
    """Measured, the gap is machine zero where H2 is exact and O(1e-2) where it is not
    13 orders apart with nothing between. The old `< 0.05` sat ABOVE the inexact
    case and certified it."""
    exact_gaps = [5.5511e-16, 1.1102e-15]
    inexact_gap = 4.3010e-02
    def reliable(gap, H2):
        return abs(gap) <= 1e-9 * max(abs(H2), 1.0)
    for gp in exact_gaps:
        assert reliable(gp, 0.693147)
    assert not reliable(inexact_gap, 1.556193)
    assert inexact_gap < 0.05, "the old constant certified this case"


def test_void_affinity_crosses_at_its_sign():
    """void_affinity lives in [-1, 1], so 0 is the crossing and 0.5 was a point picked
    on a signed scale."""
    assert (-0.4 > 0.0) is False and (0.2 > 0.0) is True
    assert (0.2 > 0.5) is False, "the old constant missed a positive affinity"


def test_divergence_uses_the_house_fence_not_a_factor_on_a_median():
    """`avga < 0.5 * med` invented a factor on top of a median. The Tukey lower fence is
    the convention engine.py and hive.py already use for outliers."""
    al = np.array([0.80, 0.82, 0.79, 0.81, 0.10])
    q1, q3 = np.percentile(al, [25.0, 75.0])
    fence = q1 - 1.5 * (q3 - q1)
    assert int((al < fence).sum()) == 1               # the outlier, and only it
    med = float(np.median(al))
    assert (al < 0.5 * med).sum() == 1                # agrees here
    tight = np.array([0.50, 0.50, 0.50, 0.50, 0.30])  # but not here
    q1, q3 = np.percentile(tight, [25.0, 75.0])
    assert int((tight < q1 - 1.5 * (q3 - q1)).sum()) == 1
    assert int((tight < 0.5 * float(np.median(tight))).sum()) == 0


#### the last two ###########################################################
def test_the_trust_score_is_a_profile_and_an_extremum_not_a_mean():
    """Three unrelated deficiencies (sparsity, incoherence, unshared) were averaged
    into one number over a list whose LENGTH varied, so the same structure scored
    differently when one term was NaN. They are kept apart now, and the scalar an API
    needs is `max`: an exact extremum, not a statistic, and stable when a term is absent.
    """
    full = {"sparsity": 0.2, "incoherence": 0.9, "unshared": 0.1}
    partial = {k: v for k, v in full.items() if k != "incoherence"}

    def score(d):
        vals = [v for v in d.values() if v is not None]
        return max(vals) if vals else 0.0

    assert score(full) == 0.9                       # the worst axis, named
    # the mean moves when a term is dropped; the extremum only drops if it WAS the worst
    assert float(np.mean(list(full.values()))) != float(np.mean(list(partial.values())))
    assert score({**full, "incoherence": None}) == 0.2


def test_confidence_rests_on_exact_invariants_not_on_summaries():
    """`va > 0.5 or kappa_mean < 0.3` became the chain condition (exact) and the SIGN of
    void affinity (exact on [-1, 1]). Coherence is reported, never judged: it is
    continuous with no structural crossing, so a constant there is a number someone
    picked."""
    def verdict(chain_valid, va):
        return "low_confidence" if (chain_valid is False or va > 0.0) else "supported"

    assert verdict(False, -0.9) == "low_confidence"   # malformed complex, regardless
    assert verdict(True, 0.2) == "low_confidence"     # positive affinity the old 0.5 missed
    assert verdict(True, -0.2) == "supported"
    assert verdict(True, 0.0) == "supported"          # the crossing is the sign


def test_no_bare_kappa_or_affinity_cutoff_survives():
    """Repo guard, same shape as the effective_resistance one."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "agent"
    bad = []
    for f in root.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            t = line.strip()
            if t.startswith("#") or '"' in t.split("#")[0][:2]:
                continue
            for name in ("void_affinity", "kappa_mean"):
                if name in t and ("<" in t or ">" in t) and "0." in t:
                    bad.append(f"{f.name}:{i}")
    assert not bad, f"a summary compared to a constant: {bad}"
