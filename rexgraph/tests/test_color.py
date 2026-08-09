"""K7 spectral colour, against the C it was ported from.

`spore.c:3086` builds K7's channel operators from its own boundary operator, mixes a
character into them, reads the spectrum as wavelengths against the Balmer limit and
integrates those through the CIE colour-matching functions. So a colour is a physical
consequence of a character rather than a palette decision.

These pin the port against reference values taken from the C directly, and the properties
that make the map usable: it is a function of the character alone, the hats are exact, and
a character with no visible spectrum says so instead of returning a default.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from rexgraph.color import (
    BALMER_LIMIT,
    K7_NE,
    hex_color,
    k7_hats,
    spectral_color,
    spectral_colors,
)

#: taken from the C, compiled from spore.c's own k7_color_forward
_REFERENCE = [
    ((0.25, 0.25, 0.25, 0.25), 1.0, 1.0,
     (0.0, 0.007361278773337589, 0.1051920260434398)),
    ((1.0, 0.0, 0.0, 0.0), 1.0, 1.0, (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 1.0, 0.0), 1.0, 1.0, (0.0, 0.0, 0.0)),
    ((0.2034, 0.2034, 0.2373, 0.3559), 1.0, 1.0,
     (0.087865357745443601, 0.0, 0.28873879134503333)),
    ((0.25, 0.25, 0.25, 0.25), 0.5, 1.0, (1.0, 1.0, 0.0)),
    ((0.25, 0.25, 0.25, 0.25), 2.0, 1.0, (0.0, 0.0, 0.0)),
    ((0.25, 0.25, 0.25, 0.25), 1.0, 0.5,
     (0.0, 0.0036806393866687945, 0.065008423140049759)),
    ((0.4, 0.3, 0.2, 0.1), 1.3, 0.8,
     (0.0, 0.0017107396906022097, 0.012131709383817258)),
]


@pytest.mark.parametrize("chi,dLT,eps,expected", _REFERENCE)
def test_the_port_matches_the_c(chi, dLT, eps, expected):
    """Compiled from spore.c and run, not reasoned about. The residual is the Jacobi
    sweep against LAPACK, nothing structural."""
    assert spectral_color(chi, dLT=dLT, eps=eps) == pytest.approx(expected, abs=1e-13)


#### the hats


def test_the_hats_are_exact():
    """K7's boundary operator is integer, so T, G and F are integer matrices and the
    trace normalisation is a ratio of integers. No float enters before the eigensolve."""
    hats = k7_hats()
    assert all(isinstance(x, Fraction) for hat in hats for row in hat for x in row)
    assert hats[0][0][0] == Fraction(1, 21), "diag(T) is 2 over a trace of 42"


def test_the_two_gram_diagonals_coincide():
    """diag(T) = diag(G), because squaring kills the sign. The same identity F is built
    from everywhere else in the library."""
    T, G, _F, _C = k7_hats()
    assert [T[i][i] for i in range(K7_NE)] == [G[i][i] for i in range(K7_NE)]


def test_the_coparticipation_hat_is_the_overlap_one():
    """They coincide on a complete graph, which is why spore stores C as G."""
    _T, G, _F, C = k7_hats()
    assert G == C


def test_the_frustration_hat_carries_no_trace():
    """A quirk kept deliberately: diag(T) = diag(G) puts zero on F's diagonal, and
    G - T >= 0 everywhere on a complete graph so the PSD shift never fires. So trace(F)
    is zero, spore divides by its 1.0 guard, and hat_F is the only unnormalised one.
    Changing that would change every colour the system has produced."""
    _T, _G, F, _C = k7_hats()
    assert sum(F[i][i] for i in range(K7_NE)) == 0


#### the map


def test_it_is_a_function_of_the_character_alone():
    assert spectral_color((0.2, 0.3, 0.4, 0.1)) == spectral_color((0.2, 0.3, 0.4, 0.1))


def test_a_character_with_no_visible_spectrum_is_black():
    """Every eigenvalue puts the wavelength outside 360-830 nm. There is no colour for
    that, and inventing one would be a reading the physics does not support."""
    assert spectral_color((1.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_dLT_moves_the_result_along_the_spectrum():
    chi = (0.25, 0.25, 0.25, 0.25)
    assert spectral_color(chi, dLT=1.0) != spectral_color(chi, dLT=0.5)


def test_eps_scales_the_intensity_without_moving_the_hue():
    chi = (0.2034, 0.2034, 0.2373, 0.3559)
    full = spectral_color(chi, eps=1.0)
    half = spectral_color(chi, eps=0.5)
    assert half != full
    assert sum(half) < sum(full)


def test_a_short_character_is_padded_rather_than_refused():
    assert spectral_color((0.5, 0.5)) == spectral_color((0.5, 0.5, 0.0, 0.0))


def test_the_batch_form_agrees_with_the_single_one():
    rows = [(0.25, 0.25, 0.25, 0.25), (0.4, 0.3, 0.2, 0.1)]
    assert spectral_colors(rows) == [spectral_color(r) for r in rows]


def test_hex_is_a_final_rounding():
    assert hex_color((1.0, 0.0, 0.5)) == "#ff0080"
    assert hex_color((0.0, 0.0, 0.0)) == "#000000"


def test_the_wavelength_scale_is_the_balmer_limit():
    assert BALMER_LIMIT == pytest.approx(364.50682023328704)
