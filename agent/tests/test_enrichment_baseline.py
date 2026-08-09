"""The enrichment statistic, against an independent implementation.

What this checks and what it does not. `goatools` is not installed here and adding it
as a dependency to run a test would be the wrong trade, so this is NOT an end-to-end
comparison of two tools. It is a comparison of the STATISTIC, which is where a
numerical disagreement would live: goatools reports the hypergeometric survival
function, and so does `scipy.stats.hypergeom`, so agreeing with scipy to machine
precision is agreeing with goatools' number.

What remains unbenchmarked, stated so the gap is not mistaken for coverage: goatools'
OBO parsing, its handling of the GO relation set beyond is_a/part_of, and its
propagation edge cases. Those are tool differences, not statistical ones, and settling
them needs the tool.

The multiple-testing correction is checked the same way: against an independent
Benjamini-Hochberg written from the definition rather than from the implementation
under test.
"""
from __future__ import annotations

import pytest
from agent.enrichment import benjamini_hochberg, enrich, hypergeometric_sf
from agent.knowledge import join
from tests.test_ontology_reasoning import IMMUNE, OBO, REPAIR, _gaf

hypergeom = pytest.importorskip("scipy.stats").hypergeom


#### the statistic


@pytest.mark.parametrize("k,N,K,n", [
    (4, 12, 6, 4),
    (1, 100, 10, 5),
    (3, 50, 20, 10),
    (0, 20, 5, 5),
    (5, 5, 5, 5),
    (2, 1000, 3, 900),
    (7, 200, 40, 25),
    (1, 10, 1, 1),
])
def test_the_tail_matches_scipy_exactly(k, N, K, n):
    """`P(X >= k)` is `hypergeom.sf(k-1, N, K, n)`. This is the number goatools
    reports, so agreement here is agreement with it."""
    mine = hypergeometric_sf(k, N, K, n)
    theirs = float(hypergeom.sf(k - 1, N, K, n)) if k > 0 else 1.0
    assert mine == pytest.approx(theirs, abs=1e-12, rel=1e-9)


def test_the_tail_matches_scipy_across_a_sweep():
    """A sweep rather than a handful, because a log-space sum can be right on small
    inputs and lose precision on large ones."""
    worst = 0.0
    for N in (20, 200, 2000):
        for K in (1, N // 7, N // 2):
            for n in (1, N // 5, N // 2):
                for k in range(0, min(K, n) + 1):
                    mine = hypergeometric_sf(k, N, K, n)
                    theirs = float(hypergeom.sf(k - 1, N, K, n)) if k > 0 else 1.0
                    worst = max(worst, abs(mine - theirs))
    assert worst < 1e-10, f"largest disagreement with scipy: {worst:.2e}"


def test_an_impossible_overlap_is_zero_in_both():
    assert hypergeometric_sf(6, 12, 5, 4) == 0.0
    assert float(hypergeom.sf(5, 12, 5, 4)) == pytest.approx(0.0)


def test_the_whole_tail_is_one_in_both():
    assert hypergeometric_sf(0, 30, 10, 5) == 1.0


#### the correction


def _bh_reference(pvalues):
    """Benjamini-Hochberg written from the definition, independent of the code under
    test: sort, scale by m/rank, enforce monotonicity from the largest down."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [0.0] * m
    running = 1.0
    for position in range(m - 1, -1, -1):
        idx = order[position]
        running = min(running, pvalues[idx] * m / (position + 1))
        out[idx] = running
    return out


@pytest.mark.parametrize("ps", [
    [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216],
    [0.5],
    [1.0, 1.0, 1.0],
    [0.0001, 0.9, 0.5, 0.02],
])
def test_the_correction_matches_an_independent_implementation(ps):
    mine = benjamini_hochberg(ps)
    theirs = _bh_reference(ps)
    assert mine == pytest.approx(theirs, abs=1e-12)


def test_the_correction_never_decreases_with_p():
    ps = [0.001, 0.01, 0.02, 0.5, 0.9]
    qs = benjamini_hochberg(ps)
    assert qs == sorted(qs)
    assert all(q >= p for q, p in zip(qs, ps, strict=True))


#### the whole pipeline, checked against the statistic run by hand


@pytest.fixture
def study(tmp_path):
    obo = tmp_path / "go.obo"
    obo.write_text(OBO)
    gaf = tmp_path / "goa.gaf"
    gaf.write_text(_gaf([(g, "GO:0006281") for g in REPAIR]
                        + [(g, "GO:0006955") for g in IMMUNE]))
    return join(str(obo), str(gaf))


def test_every_reported_p_matches_scipy_on_its_own_counts(study):
    """The end-to-end check: whatever counts the pipeline arrived at, the p it reports
    for them is the hypergeometric tail for those counts."""
    out = enrich(study, ["BRCA1", "BRCA2", "ATM", "RAD51"])
    N, n = out["n_universe"], out["n_study"]
    assert out["terms"], "nothing was enriched, so there is nothing to check"
    for row in out["terms"]:
        expected = float(hypergeom.sf(row["n_study"] - 1, N, row["n_term"], n))
        assert row["p_value"] == pytest.approx(expected, abs=1e-12), (
            f"{row['term']}: reported {row['p_value']}, hypergeometric {expected}")


def test_the_counts_are_what_the_true_path_rule_gives(study):
    """The other half of a p-value is the counts, and propagation is where an
    enrichment implementation usually differs from another."""
    out = enrich(study, ["BRCA1", "BRCA2", "ATM", "RAD51"])
    by_term = {r["term"]: r for r in out["terms"]}
    assert by_term["DNA repair"]["n_term"] == 6
    assert by_term["response to DNA damage"]["n_term"] == 6, \
        "the parent did not inherit its child's annotations"
    assert by_term["biological_process"]["n_term"] == 12
    assert out["n_universe"] == 12


def test_the_ranking_is_by_significance(study):
    out = enrich(study, REPAIR[:4])
    ps = [r["p_value"] for r in out["terms"]]
    assert ps == sorted(ps)
