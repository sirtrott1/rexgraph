"""The edge signature must read channels by NAME, not by position.

`edge_signature` did `(chi[0], chi[1] + chi[3], chi[2])`, which assumes the character
always has four components in the order T, G, F, C. It does not. The character is
(nE, nhats) with nhats = 3 or 4: a channel whose trace vanishes is DROPPED, and then the
remaining columns close up.

Both failure modes follow, and both are silent-or-crash rather than approximate:

  * IndexError on chi[3] whenever nhats == 3. That is not exotic. Any consistently
    oriented complex has no head-to-tail disagreement, so trace(F) = 0 and F drops.
    A bipartite measurement complex is exactly that, and it is the shape most of this
    library's data takes.
  * worse when it does not crash: with F dropped the columns are (T, G, C), so chi[2] is
    C and the signature silently reads the co-participation channel where it means the
    frustration one.

`hat_names` says which channels are live, so the fix is to look them up.
"""

import numpy as np

from rexgraph.faces import autoface
from rexgraph.graph import RexGraph


def _bipartite(n_t=6, per_t=5):
    """Consistently oriented target -> ligand measurements. trace(F) = 0, so nhats = 3."""
    src, tgt, nxt = [], [], n_t
    for t in range(n_t):
        for _ in range(per_t):
            src.append(t); tgt.append(nxt); nxt += 1
    for t in range(n_t - 1):                       # share a ligand so it is connected
        src.append(t); tgt.append(n_t + (t + 1) * per_t)
    return RexGraph(sources=np.asarray(src, np.int32), targets=np.asarray(tgt, np.int32))


def test_a_consistently_oriented_complex_reads_zero_frustration():
    """The precondition. Frustration VANISHES here and is reported as zero rather
    than dropped: every vertex is a pure source or a pure sink, so the signed and
    unsigned overlaps agree at every shared vertex and there is no orientation
    conflict to measure. That is a reading, not an absent channel, so the character
    keeps its width and stays comparable with any other complex."""
    rex = _bipartite()
    assert int(rex.nhats) == 4
    chi = np.asarray(rex.structural_character)
    assert chi.shape[1] == 4
    names = list(rex.hat_names)
    assert names == ["L1_down", "L_O", "L_SG", "L_C"]
    assert np.allclose(chi[:, names.index("L_SG")], 0.0)
    assert np.allclose(chi.sum(axis=1), 1.0), "still on the simplex"


def test_edge_signature_does_not_crash_when_frustration_is_zero():
    rex = _bipartite()
    sig = rex.edge_signature(0)
    assert isinstance(sig, tuple)
    assert all(isinstance(x, float) for x in sig)


def test_group_edges_by_signature_works_when_frustration_is_zero():
    rex = _bipartite()
    groups = rex.group_edges_by_signature()
    assert sum(len(v) for v in groups.values()) == int(rex.nE)


def test_the_signature_reads_channels_by_name():
    """The silent half. With F dropped the third column is C, so a positional read
    reports co-participation as though it were frustration."""
    rex = _bipartite()
    chan = {"L1_down": "T", "L_O": "G", "L_SG": "F", "L_C": "C"}
    chi = np.asarray(rex.structural_character)[0]
    by_name = {chan[n]: v for n, v in zip(list(rex.hat_names), chi, strict=True)}
    sig = rex.edge_signature(0)
    assert np.isclose(sig[0], round(float(by_name.get("T", 0.0)), 6))
    assert np.isclose(sig[2], round(float(by_name.get("F", 0.0)), 6))
    assert np.isclose(sig[1], round(float(by_name.get("G", 0.0) + by_name.get("C", 0.0)), 6))


def test_a_four_channel_complex_is_unchanged():
    """Where all four are live the signature must read exactly as it always did."""
    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    autoface(rex, 3)
    assert int(rex.nhats) == 4
    chi = np.asarray(rex.structural_character)[0]
    assert rex.edge_signature(0) == (round(float(chi[0]), 6),
                                     round(float(chi[1] + chi[3]), 6),
                                     round(float(chi[2]), 6))


def test_structurally_identical_relations_share_a_signature():
    """What the signature is for: exact equivalence classes, not a threshold."""
    rex = _bipartite(n_t=4, per_t=4)
    groups = rex.group_edges_by_signature()
    assert len(groups) < int(rex.nE)               # some relations are interchangeable
    for sig, members in groups.items():
        for e in members:
            assert rex.edge_signature(e) == sig
