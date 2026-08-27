"""L_gb, the graded boundary delta, and its live surface.

Where the mass tower reads each grade alone, L_gb reads the coupling between
adjacent grades. It is `a a^T/|a|^2 - b b^T/|b|^2` on the two grades' normalized
coherence spectra: a difference of two rank-1 ORTHOGONAL projectors, which the
reference states as "rank-2 by construction, one positive eigenvalue, one
negative". For that shape the spectrum is closed-form and needs no eigensolver.
"""

import itertools

import numpy as np

from rexgraph import channel_delta, graded_delta
from rexgraph.core._l_gb import l_gb_scalar
from rexgraph.graded_boundary import build_graded_boundaries, truncated_icosahedron_3rex
from rexgraph.graph import RexGraph


def _c60_solid():
    return build_graded_boundaries(truncated_icosahedron_3rex())


def _complete(n):
    e = list(itertools.combinations(range(n), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    r._ensure_clean()
    return r


#### the closed form
def test_the_spectrum_is_plus_minus_root_spread():
    """A difference of two rank-1 orthogonal projectors has eigenvalues +-sin(theta)
    and zeros, so one dot product settles the whole thing."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 60))
        a = np.abs(rng.normal(size=n))
        b = np.abs(rng.normal(size=n))
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        c = float(a @ b) / (na * nb)
        spread = max(0.0, 1.0 - c * c)
        # the kernel reports the Frobenius norm, which is sqrt(2 * spread)
        assert np.isclose(l_gb_scalar(a, b), np.sqrt(2.0 * spread), atol=1e-12)


def test_the_closed_form_matches_forming_the_operator():
    """Against the outer products it replaces."""
    rng = np.random.default_rng(1)
    for _ in range(30):
        n = int(rng.integers(2, 40))
        a = np.abs(rng.normal(size=n))
        b = np.abs(rng.normal(size=n))
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        L = np.outer(a, a) / (na * na) - np.outer(b, b) / (nb * nb)
        assert np.isclose(l_gb_scalar(a, b), np.linalg.norm(L, "fro"), atol=1e-12)
        ev = np.linalg.eigvalsh(0.5 * (L + L.T))
        assert np.isclose(ev[-1], -ev[0], atol=1e-12), "rank 2, one up one down"


def test_identical_spectra_couple_at_zero():
    a = np.array([1.0, 0.5, 0.25])
    assert np.isclose(l_gb_scalar(a, a), 0.0, atol=1e-12)
    assert np.isclose(l_gb_scalar(a, 7.0 * a), 0.0, atol=1e-12), "scale is normalized out"


#### the tower
def test_the_tower_reads_every_adjacent_pair():
    """C60 as a solid has three boundary operators, so three couplings."""
    from rexgraph.core._l_gb import l_gb_tower
    tower = l_gb_tower([np.asarray(b.todense(), float) for b in _c60_solid()])
    assert [d["pair"] for d in tower] == [(0, 1), (1, 2), (2, 3)]


def test_the_tower_entries_obey_the_rank_two_structure():
    from rexgraph.core._l_gb import l_gb_tower
    tower = l_gb_tower([np.asarray(b.todense(), float) for b in _c60_solid()])
    for d in tower:
        assert np.isclose(d["top_eig"], -d["bot_eig"], atol=1e-12)
        assert np.isclose(d["spread"], 2.0 * d["top_eig"], atol=1e-12)
        assert np.isclose(d["frob"], np.sqrt(2.0) * d["top_eig"], atol=1e-12)
        assert -1.0 <= d["localization"] <= 1.0


def test_graded_delta_is_reachable_from_the_package():
    """The point of this file: L_gb had a compiled kernel, a reference and tests,
    and no live caller."""
    import rexgraph
    assert hasattr(rexgraph, "graded_delta") and hasattr(rexgraph, "channel_delta")
    r = _complete(5)
    out = graded_delta(r)
    assert isinstance(out, list)
    for d in out:
        assert {"pair", "top_eig", "bot_eig", "frob", "localization"} <= set(d)


#### the within-grade channel tensor
def test_channel_delta_is_a_symmetric_four_by_four_with_a_zero_diagonal():
    T = channel_delta(_complete(5))
    assert T.shape == (4, 4)
    assert np.allclose(np.diag(T), 0.0), "a channel matches itself"
    assert np.allclose(T, T.T, atol=1e-12)
    assert (T >= 0).all()


def test_channel_delta_matches_the_reference_implementation():
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "l_gb_reference", "rexgraph/tests/reference/l_gb_reference.py")
    ref = iu.module_from_spec(spec)
    spec.loader.exec_module(ref)
    for n in (4, 5, 6):
        r = _complete(n)
        hats = list(r._rcf_bundle.get("hats", []) or [])
        assert hats, "fixture has no channel hats"
        assert np.allclose(channel_delta(r), np.asarray(ref.l_gb_channel_tensor(hats)),
                           atol=1e-10)


#### the degenerate case, which is where the reference convention lives
def _ref():
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "l_gb_reference", "rexgraph/tests/reference/l_gb_reference.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_zero_spectrum_leaves_the_other_projector_standing():
    """The reference floors each norm before normalising, so a spectrum that is
    identically zero does not collapse the pair to nothing: the other rank-1
    projector remains, and its Frobenius norm is 1. The compiled path returned 0
    here, which is where the documented T[i,F] = 1 reading was being lost."""
    ref = _ref()
    b = np.abs(np.random.default_rng(0).normal(size=8))
    z = np.zeros(8)
    assert np.isclose(l_gb_scalar(z, b), 1.0, atol=1e-12)
    assert np.isclose(l_gb_scalar(b, z), 1.0, atol=1e-12)
    assert np.isclose(l_gb_scalar(z, b), ref.l_gb_scalar(z, b)["frob"], atol=1e-12)
    assert np.isclose(l_gb_scalar(b, z), ref.l_gb_scalar(b, z)["frob"], atol=1e-12)


def test_both_zero_and_parallel_still_couple_at_nothing():
    z = np.zeros(6)
    assert np.isclose(l_gb_scalar(z, z), 0.0, atol=1e-12)
    a = np.array([1.0, 2.0, 3.0])
    assert np.isclose(l_gb_scalar(a, 2.0 * a), 0.0, atol=1e-12)


def test_the_degenerate_eigenvalues_are_zero_and_minus_one():
    """`-P_b` has spectrum {-1, 0, ...}, so the pair is not symmetric there. The
    +-sqrt(spread) form is the both-ordinary case, not the general one."""
    ref = _ref()
    b = np.abs(np.random.default_rng(1).normal(size=7))
    z = np.zeros(7)
    got = ref.l_gb_scalar(z, b)
    assert np.isclose(got["top_eig"], 0.0, atol=1e-12)
    assert np.isclose(got["bot_eig"], -1.0, atol=1e-12)
    flipped = ref.l_gb_scalar(b, z)
    assert np.isclose(flipped["top_eig"], 1.0, atol=1e-12)
    assert np.isclose(flipped["bot_eig"], 0.0, atol=1e-12)


def test_the_closed_form_matches_the_reference_across_every_regime():
    """Zero, tiny-but-nonzero, parallel, ragged lengths and ordinary, together."""
    ref = _ref()
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(400):
        n = int(rng.integers(1, 30))
        m = int(rng.integers(1, 30))
        a = np.abs(rng.normal(size=n)) * rng.choice([0.0, 1.0, 1e-15])
        b = np.abs(rng.normal(size=m)) * rng.choice([0.0, 1.0, 1e-15])
        worst = max(worst, abs(l_gb_scalar(a, b) - ref.l_gb_scalar(a, b)["frob"]))
    assert worst < 1e-12, worst


def test_the_tower_matches_the_reference_on_a_real_solid():
    from rexgraph.core._l_gb import l_gb_tower
    ref = _ref()
    Bs = [np.asarray(b.todense(), float) for b in _c60_solid()]
    for got, want in zip(l_gb_tower(Bs), ref.l_gb_tower(Bs), strict=True):
        assert got["pair"] == want["pair"]
        for k in ("top_eig", "bot_eig", "frob", "localization"):
            assert np.isclose(got[k], want[k], atol=1e-12), (got["pair"], k)


#### numerical stability, which the closed form got wrong first
def test_identical_spectra_do_not_amplify_to_1e_minus_8():
    """cos^2 for identical vectors lands at 1 - 2e-16, and sqrt(2 - 2cos^2) turns
    that into 3e-8. Taking sin^2 from the component of b orthogonal to a keeps the
    cancellation in the vector space, where it is exact."""
    s = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    assert l_gb_scalar(s, s.copy()) < 1e-14
    assert l_gb_scalar(s, 3.0 * s) < 1e-14


def test_near_parallel_is_not_flattened_to_zero():
    """The clamp that hid the first error drove this to a flat 0. The true value is
    small and nonzero, and the reference reports it."""
    ref = _ref()
    s = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    b = s + 1e-9 * np.random.default_rng(0).normal(size=5)
    got = l_gb_scalar(s, b)
    want = ref.l_gb_scalar(s, b)["frob"]
    assert got > 1e-12, "a real separation was reported as none"
    assert np.isclose(got, want, rtol=1e-9)


def test_every_regime_against_the_reference_including_the_hard_ones():
    ref = _ref()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(400):
        n = int(rng.integers(1, 25))
        base = np.abs(rng.normal(size=n))
        a = base * rng.choice([0.0, 1.0, 1e-15])
        style = int(rng.integers(0, 5))
        b = {0: base.copy(), 1: 3.0 * base,
             2: base + 1e-9 * rng.normal(size=n),
             3: np.abs(rng.normal(size=n)),
             4: np.abs(rng.normal(size=int(rng.integers(1, 25))))}[style]
        b = b * rng.choice([0.0, 1.0, 1.0, 1.0])
        worst = max(worst, abs(l_gb_scalar(a, b) - ref.l_gb_scalar(a, b)["frob"]))
    assert worst < 1e-14, worst


#### where the F channel actually goes, and the identity that needed it
def _uniformly_oriented():
    """Every vertex a pure source or a pure sink. T and G agree at every shared
    vertex there, so the signed/unsigned mismatch has nothing to measure."""
    return {"star out": ([0, 0, 0, 0], [1, 2, 3, 4]),
            "K33 left to right": ([0, 0, 0, 1, 1, 1, 2, 2, 2],
                                  [3, 4, 5, 3, 4, 5, 3, 4, 5]),
            "C4 source sink": ([0, 2, 2, 0], [1, 1, 3, 3]),
            "path alternating": ([0, 2, 2, 4], [1, 1, 3, 3])}


def test_uniform_orientation_reads_frustration_as_zero():
    """This family is where F vanishes, and it is carried at zero rather than
    dropped. The bundle keeps four hats and the F column reads 0."""
    for name, (src, tgt) in _uniformly_oriented().items():
        r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
        r._ensure_clean()
        bundle = r._rcf_bundle
        names = list(bundle["hat_names"])
        assert names == ["L1_down", "L_O", "L_SG", "L_C"], (name, names)
        assert float(np.asarray(bundle["trace_values"])[2]) == 0.0, name
        chi = np.asarray(bundle["chi"])
        assert np.allclose(chi[:, 2], 0.0), name
        assert np.allclose(chi.sum(axis=1), 1.0), name


def test_a_mixed_orientation_keeps_all_four():
    r = RexGraph(sources=np.array([0, 0, 2, 3], np.int32),
                 targets=np.array([1, 2, 0, 0], np.int32))
    r._ensure_clean()
    assert list(r._rcf_bundle["hat_names"]) == ["L1_down", "L_O", "L_SG", "L_C"]


def test_the_documented_identity_holds_natively():
    """T[i,F] = T[F,i] = 1 for i in T, G, C, straight off the bundle. It needs two
    things that are both true now: F carried as a channel even at zero mass, and
    the degenerate reading where a zero spectrum leaves the other projector
    standing, whose Frobenius norm is 1."""
    from rexgraph.core._l_gb import l_gb_channel_tensor
    for name, (src, tgt) in _uniformly_oriented().items():
        r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
        r._ensure_clean()
        hats = list(r._rcf_bundle["hats"])
        assert len(hats) == 4, name
        T = np.asarray(l_gb_channel_tensor(hats))
        for i in (0, 1, 3):
            assert np.isclose(T[i, 2], 1.0, atol=1e-12), (name, i)
            assert np.isclose(T[2, i], 1.0, atol=1e-12), (name, i)


def test_channel_delta_is_four_wide_whatever_the_orientation():
    """The reason the width matters: the reference refuses anything but four
    channels a side, so a dropped F made the tensor unbuildable exactly where F
    was the interesting reading."""
    for name, (src, tgt) in _uniformly_oriented().items():
        r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
        r._ensure_clean()
        assert channel_delta(r).shape == (4, 4), name


#### frustration is a first-class channel everywhere it is read
def test_a_massless_channel_reports_zero_mixing_time_not_infinity():
    """Its operator is the zero matrix, so e^{-tL} = I and every state is already
    stationary at t=0. Nothing equilibrates because nothing moves, which is a mixing
    time of zero and not a process that never settles. Reporting inf would also
    poison anything derived across channels."""
    r = RexGraph(sources=np.array([0, 0, 0, 0], np.int32),
                 targets=np.array([1, 2, 3, 4], np.int32))
    r._ensure_clean()
    times = np.asarray(r.per_channel_mixing_times, dtype=float)
    names = list(r.hat_names)
    assert len(times) == 4 and names[2] == "L_SG"
    assert np.isfinite(times).all(), times
    assert times[2] == 0.0


def test_the_exact_character_reads_zero_frustration_and_stays_on_the_simplex():
    from fractions import Fraction

    from rexgraph.rational_trig import exact_character
    r = RexGraph(sources=np.array([0, 0, 0, 0], np.int32),
                 targets=np.array([1, 2, 3, 4], np.int32))
    r._ensure_clean()
    chi, names = exact_character(r)
    assert names == ["L1_down", "L_O", "L_SG", "L_C"]
    for row in chi:
        assert row[2] == Fraction(0)
        assert sum(row) == Fraction(1)
    # and the float tower agrees column for column
    assert np.allclose([[float(x) for x in row] for row in chi],
                       np.asarray(r.structural_character), atol=1e-12)


def test_the_accessors_answer_rather_than_declining():
    """lagrangian_fields and cr_violation returned None when F dropped, to avoid the
    remaining three being read as T,G,F,C. The positions are fixed now."""
    r = RexGraph(sources=np.array([0, 0, 0], np.int32),
                 targets=np.array([1, 2, 3], np.int32))
    r._ensure_clean()
    fields = r.lagrangian_fields()
    assert fields is not None
    assert list(fields["channels"]) == ["L1_down", "L_O", "L_SG", "L_C"]
    assert r.cr_violation() is not None
