"""The K7 colour has an exposure, and it is solved for rather than picked."""

import numpy as np
import pytest

from rexgraph.color import BALMER_LIMIT, exposure, hex_color, spectral_color, spectral_colors
from rexgraph.graph import RexGraph


def _panel():
    """Six relations sharing no vertex, plus two that do: two distinct characters."""
    groups = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13],
              [14, 15, 16], [17, 18, 19], [0, 3, 20], [1, 7, 21]]
    ptr = np.cumsum([0] + [len(g) for g in groups]).astype(np.int32)
    idx = np.array([v for g in groups for v in g], np.int32)
    rex = RexGraph.from_hypergraph(ptr, idx)
    rex._ensure_clean()
    return rex


def test_the_fixed_exposure_is_untouched():
    """dLT=1, eps=1 is spore's setting and must give exactly what it always gave."""
    assert spectral_color([0.25] * 4) == pytest.approx(
        (0.0, 0.007361, 0.105192), abs=1e-5)
    assert spectral_color([4 / 13, 4 / 13, 5 / 26, 5 / 26]) == pytest.approx(
        (0.0, 0.347789, 1.0), abs=1e-5)


def test_a_real_character_can_fall_off_the_end_of_the_band():
    """The motivation, stated as a test.

    Six of this fixture's eight relations are black at spore's dLT = 1, and every one of
    the eight on the real BindingDB panel that prompted this was. The failure mode is the
    same either way: the picture goes dark and the darkness is not saying anything about
    the characters, only that their spectra left the band.
    """
    chi = np.asarray(_panel().structural_character, dtype=float)
    dark = [row for row in chi if spectral_color(row) == (0.0, 0.0, 0.0)]
    assert len(dark) == 6
    assert exposure(chi)["visible"] == 8


def test_the_solved_exposure_lights_the_complex_the_fixed_one_could_not():
    chi = np.asarray(_panel().structural_character, dtype=float)
    solved = exposure(chi)
    assert solved["visible"] > 0
    assert solved["dLT"] != 1.0
    lit = [c for c in spectral_colors(chi, dLT="auto", eps="auto") if any(c)]
    assert len(lit) == solved["visible"]


def test_the_chosen_dLT_really_is_inside_every_interval_it_claims():
    """The solve is only right if the eigenvalues it counts are genuinely visible.

    An eigenvalue is visible exactly when 360 <= B / (lam * dLT) <= 830, so this
    re-derives the condition rather than trusting the sweep that produced it.
    """
    chi = np.asarray(_panel().structural_character, dtype=float)
    dLT = exposure(chi)["dLT"]
    counted = 0
    for row in chi:
        colour = spectral_color(row, dLT=dLT)
        if any(colour):
            counted += 1
    assert counted > 0
    for row in chi:
        if any(spectral_color(row, dLT=dLT)):
            from rexgraph.color import _spectrum
            wl = BALMER_LIMIT / (_spectrum(row) * dLT)
            assert ((wl >= 360.0) & (wl <= 830.0)).any()


def test_the_solved_intensity_does_not_clip():
    """eps is set so the brightest cell reaches full without going white."""
    chi = np.asarray(_panel().structural_character, dtype=float)
    colours = spectral_colors(chi, dLT="auto", eps="auto")
    assert all(hex_color(c) != "#ffffff" for c in colours)
    peak = max(max(c) for c in colours)
    assert peak == pytest.approx(1.0, abs=1e-6)


def test_an_explicit_setting_is_never_overridden():
    chi = [[0.25] * 4]
    assert spectral_colors(chi, dLT=1.0, eps=1.0)[0] == spectral_color([0.25] * 4)


def test_a_complex_with_no_spectrum_reports_rather_than_guesses():
    assert exposure([[0.0, 0.0, 0.0, 0.0]])["visible"] == 0
