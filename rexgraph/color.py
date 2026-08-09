"""Colour derived from the character, through K7's spectrum.

A palette is a decision someone made. This is not one: a cell's channel character is
mixed into the channel operators of K7, the complete graph on seven vertices, its spectrum
is read as wavelengths against the Balmer limit, and those are integrated through the CIE
colour-matching functions into sRGB. So the colour of a cell is a physical consequence of
its character rather than a lookup, two cells the same colour have the same character, and
nothing here has a range to normalise or a legend that could lie.

The construction, from `spore.c:3086`::

    B1        K7's boundary operator, 7 x 21, entries -1 and +1
    T         B1^T B1              the signed Gram
    G         |B1|^T |B1|          its unsigned twin
    F         G - T, shifted to PSD by its own most negative entry
    hats      each trace-normalised; C = G, since the two coincide on a complete graph
    M(chi)    chi_T hat_T + chi_G hat_G + chi_F hat_F + chi_C hat_C
    lambda    the spectrum of M
    wl_j      alpha / (lambda_j dLT),  alpha = 364.50682... nm, the Balmer limit
    XYZ       eps * sum_j lambda_j CMF(wl_j)
    sRGB      gamma(M_XYZ_TO_SRGB . XYZ)

Why K7 rather than the complex being drawn: the reference has to be fixed, or the same
character would take different colours in different complexes and the picture would stop
being comparable. K7 is the smallest complete graph whose 21 edges give a spectrum wide
enough to cover the visible band.

One quirk is inherited deliberately. `trace(F)` on K7 is ZERO, so the trace
normalisation divides by the 1.0 guard and `hat_F` is the only channel operator that is
not normalised. That follows from the model rather than from an oversight:
`diag(T) = diag(G)` identically, because squaring kills the sign, so F's diagonal is zero
before the shift; and `G - T >= 0` everywhere on a complete graph, so the PSD shift never
fires. Reproduced as it stands, since changing it would change every colour the system
has ever produced.

`dLT` positions the whole picture on the spectrum, because the wavelength is
`alpha / (lambda dLT)`, and the character eigenvalues of an ordinary complex sit where a
factor of two in dLT moves the result from deep blue to saturated yellow. It is a real
knob, not a fudge: the same dLT gives the same colour for the same character everywhere.

**The hats are exact.** K7's boundary operator is integer, so T, G and F are integer
matrices and the trace normalisation is a ratio of integers: `k7_hats` returns Fractions.
Everything after the eigensolve is float, which is the honest boundary, since a spectrum
is not a rational function of its matrix.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np

__all__ = ["exposure", "k7_hats", "spectral_color", "spectral_colors", "hex_color",
           "BALMER_LIMIT", "K7_NV", "K7_NE"]

#: K7: the complete graph on seven vertices, and its 21 relations
K7_NV = 7
K7_NE = 21

#: the Balmer series limit in nm, which sets the wavelength scale
BALMER_LIMIT = 364.50682023328704

#: linear sRGB from CIE XYZ
_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                         [-0.9689, 1.8758, 0.0415],
                         [0.0557, -0.2040, 1.0570]])


def _k7_boundary() -> np.ndarray:
    """K7's B1, 7 x 21, one column per relation with a -1 and a +1."""
    B1 = np.zeros((K7_NV, K7_NE), dtype=np.int64)
    e = 0
    for i in range(K7_NV):
        for j in range(i + 1, K7_NV):
            B1[i, e] = -1
            B1[j, e] = 1
            e += 1
    return B1


@lru_cache(maxsize=1)
def k7_hats() -> tuple:
    """The four trace-normalised channel operators of K7, as exact Fractions.

    Returned as a tuple of tuples so the cache can hold it. Integer throughout until the
    normalisation, which is a ratio of integers, so no float enters here at all.
    """
    B1 = _k7_boundary()
    T = B1.T @ B1
    G = np.abs(B1).T @ np.abs(B1)
    F = G - T
    # shift F onto the PSD cone by its own most negative entry, on the diagonal only
    minF = int(F.min())
    if minF < 0:
        F = F + np.diag(np.full(K7_NE, -minF, dtype=np.int64))

    hats = []
    for M in (T, G, F, G):                       # C = G: they coincide on a complete graph
        trace = int(np.trace(M)) or 1
        hats.append(tuple(tuple(Fraction(int(x), trace) for x in row) for row in M))
    return tuple(hats)


def _cie(wl):
    """CIE 1931 colour-matching functions, the Wyman-Sloan-Shirley Gaussian fit."""
    def lobe(w, peak, lo, hi):
        t = (w - peak) * np.where(w < peak, lo, hi)
        return np.exp(-0.5 * t * t)

    x = (0.362 * lobe(wl, 442.0, 0.0624, 0.0374)
         + 1.056 * lobe(wl, 599.8, 0.0264, 0.0323)
         - 0.065 * lobe(wl, 501.1, 0.0490, 0.0382))
    y = (0.821 * lobe(wl, 568.8, 0.0213, 0.0247)
         + 0.286 * lobe(wl, 530.9, 0.0613, 0.0322))
    z = (1.217 * lobe(wl, 437.0, 0.0845, 0.0278)
         + 0.681 * lobe(wl, 459.0, 0.0385, 0.0725))
    return x, y, z


def _gamma(u):
    """sRGB gamma encoding."""
    u = np.clip(u, 0.0, 1.0)
    return np.where(u <= 0.0031308, 12.92 * u, 1.055 * np.power(u, 1 / 2.4) - 0.055)


def spectral_color(chi, *, dLT: float = 1.0, eps: float = 1.0) -> tuple:
    """One cell's character as sRGB in [0, 1].

    `chi` is the four channel shares in CHANNEL_ORDER. `dLT` scales the wavelength, so it
    moves the whole picture along the spectrum; `eps` scales the intensity. Both default
    to 1, which is what `spore.c` uses when the caller does not say.

    A character whose spectrum falls entirely outside 360-830 nm returns black, because
    there is no visible colour for it, and reporting one would invent a reading.
    """
    linear = _linear_rgb(chi, dLT)
    if not linear.size:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) for v in _gamma(float(eps) * linear))


def _linear_rgb(chi, dLT) -> np.ndarray:
    """One cell's colour as LINEAR sRGB at unit intensity, before gamma and before eps.

    Split out because `exposure` has to know how bright the picture would be before the
    gamma encode clips it, and asking `spectral_color` would give the clipped answer.
    Empty when the cell's whole spectrum falls outside the visible band, which is an
    absence rather than a black.
    """
    lam = _spectrum(chi)
    if not lam.size:
        return np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        wl = BALMER_LIMIT / (lam * float(dLT))
    visible = np.isfinite(wl) & (wl >= 360.0) & (wl <= 830.0)
    if not visible.any():
        return np.zeros(0)
    lam, wl = lam[visible], wl[visible]
    cx, cy, cz = _cie(wl)
    XYZ = np.array([float(lam @ cx), float(lam @ cy), float(lam @ cz)])
    return _XYZ_TO_SRGB @ XYZ


def _spectrum(chi) -> np.ndarray:
    """The positive eigenvalues of one cell's channel-weighted K7 operator.

    The one place the character is mixed into the hats, so `spectral_color` and `exposure`
    read the same operator by construction rather than by two copies agreeing.
    """
    values = [float(x) for x in chi]
    values += [0.0] * (4 - len(values))
    M = np.zeros((K7_NE, K7_NE))
    for share, hat in zip(values, k7_hats(), strict=True):
        if share:
            M += share * np.array([[float(x) for x in row] for row in hat])
    evals = np.linalg.eigvalsh(M)
    return evals[evals > 1e-10]


def exposure(chi_rows) -> dict:
    """The `dLT` that puts the most of THIS complex inside the visible band.

    `dLT = 1` is spore's value and it is right there, but it is an exposure, not a
    reading, and at a fixed one most complexes come out black. Measured on a real binding
    panel: every one of eight relations returned `(0, 0, 0)` at `dLT = 1`, so the picture
    was grey throughout and the grey was not saying anything about the characters. It was
    saying the spectrum had fallen off the end of the band.

    Nothing has to be guessed to fix that, because the visibility condition is already an
    equation. A cell's colour comes from `wl = B / (lam * dLT)` kept where
    `360 <= wl <= 830`, so eigenvalue `lam` is visible for exactly

        dLT in [B / (830 * lam), B / (360 * lam)]

    one closed interval per eigenvalue. The exposure that shows the most of the complex is
    then the point covered by the most intervals, which a sweep finds exactly: sort the
    endpoints, run a counter, take the widest run at the maximum. The midpoint of that run
    is taken geometrically, since `dLT` enters as a reciprocal scale and the interval is a
    ratio rather than a difference.

    So this chooses nothing. It solves for where the function the owner wrote is defined,
    and reports the answer along with how much of the complex it reaches, because an
    exposure that only lights half the cells is a fact about the picture.
    """
    rows = np.asarray(chi_rows, dtype=float)
    rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
    spectra = [_spectrum(row) for row in rows]
    events = []
    for lam in np.concatenate(spectra) if spectra and len(rows) else []:
        events.append((BALMER_LIMIT / (830.0 * float(lam)), 1))
        events.append((BALMER_LIMIT / (360.0 * float(lam)), -1))
    if not events:
        return {"dLT": 1.0, "visible": 0, "of": int(len(rows)), "reason": "no spectrum"}

    events.sort()
    best_count, count, best = 0, 0, (1.0, 1.0)
    for k, (position, delta) in enumerate(events):
        count += delta
        if delta == 1 and count > best_count:
            nxt = next((q for q, _ in events[k + 1:]), position)
            best_count, best = count, (position, nxt)
        elif delta == 1 and count == best_count:
            nxt = next((q for q, _ in events[k + 1:]), position)
            if nxt / max(position, 1e-300) > best[1] / max(best[0], 1e-300):
                best = (position, nxt)
    lo, hi = best
    dLT = float(np.sqrt(lo * hi)) if lo > 0 and hi > 0 else 1.0

    # and the intensity, by the same rule. `eps` multiplies XYZ before the gamma encode,
    # which CLIPS, so an eps that overshoots turns every bright cell the same white and
    # loses exactly the cells whose character is strongest. The largest linear component
    # anywhere in the complex is what must land at 1, so eps is its reciprocal: nothing
    # clips, nothing is dimmed more than it has to be, and the ratios between cells are
    # untouched because it is one scalar over the whole picture.
    linear = [_linear_rgb(row, dLT) for row in rows]
    peak = max((float(v.max()) for v in linear if v.size), default=0.0)
    eps = 1.0 / peak if peak > 0 else 1.0
    lit = sum(1 for v in linear if v.size and float(v.max()) > 0.0)
    return {"dLT": dLT, "eps": float(eps), "visible": int(lit), "of": int(len(rows)),
            "band": [float(lo), float(hi)],
            "reading": ("the exposure covered by the most eigenvalue intervals, with eps "
                        "set so the brightest cell lands at full without clipping; "
                        f"{lit} of {len(rows)} cells land in the visible band there")}


def spectral_colors(chi_rows, *, dLT: float | str = 1.0, eps: float | str = 1.0) -> list:
    """`spectral_color` over every cell of a complex.

    `dLT="auto"` solves for the exposure through `exposure` first, and `eps="auto"` takes
    the intensity from the same solve. That is what a drawing wants: at a fixed pair most
    complexes come out either black, because their spectrum fell off the end of the band,
    or white, because the encode clipped.
    """
    rows = np.asarray(chi_rows, dtype=float)
    rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
    solved = exposure(rows) if "auto" in (dLT, eps) else None
    scale = solved["dLT"] if dLT == "auto" else float(dLT)
    gain = solved["eps"] if eps == "auto" else float(eps)
    return [spectral_color(row, dLT=scale, eps=gain) for row in rows]


def hex_color(rgb) -> str:
    """sRGB in [0, 1] as `#rrggbb`, for a path string."""
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(float(c) * 255))) for c in rgb)
