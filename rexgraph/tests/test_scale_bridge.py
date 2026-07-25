"""Scale-free moments and the varentropy self-diagnostic - the eigen-free scale engine.

The scale profile is closed-k-walk moments; the varentropy gap is the Renyi-2 minus
Renyi-3 harmonic log. These guard the scale/character self-diagnostics behaviorally
and against their closed-form oracles.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _rex(edges):
    s = np.array([e[0] for e in edges], dtype=np.int32)
    t = np.array([e[1] for e in edges], dtype=np.int32)
    return RexGraph.from_graph(s, t)


def test_vertex_scale_profile_separates_clustered_from_path():
    """Closed-k-walk moments diverge for a clustered vs a path vertex at scale >= 2 (script 15)."""
    rex = _rex([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)])   # triangle {0,1,2} + path tail 3-4
    vsp = rex.vertex_scale_profile
    assert vsp.shape[0] == rex.nV
    clustered, path = vsp[0], vsp[4]
    # the k=0,1 moments can coincide; the higher moments must separate them
    assert not np.allclose(clustered[2:], path[2:])
    assert clustered[2] > path[2]                          # more length-2 closed walks in the triangle


def test_harmonic_entropy_is_renyi2_of_rl4():
    """harmonic_entropy = -log(tr(RL4^2)/tr(RL4)^2), the eigen-free Renyi-2 (script 18)."""
    rex = _rex([(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (4, 5), (2, 5)])
    RL = np.asarray(rex.RL, dtype=float)
    want = -np.log(np.trace(RL @ RL) / np.trace(RL) ** 2)
    np.testing.assert_allclose(rex.harmonic_entropy, want, atol=1e-9)


def test_character_varentropy_gap_ties_to_harmonic_entropy():
    """varentropy gap = H2 - H3 >= 0, and H2 is the harmonic (Renyi-2) entropy (script 19)."""
    rex = _rex([(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (4, 5), (2, 5)])
    v = rex.character_varentropy
    assert v["gap"] == pytest.approx(v["H2"] - v["H3"], abs=1e-9)
    assert v["H2"] == pytest.approx(rex.harmonic_entropy, abs=1e-5)   # dict values are 6dp-rounded
    assert v["gap"] >= -1e-9
