"""Equiweight: the derived axiom, and what it is good for.

    Gamma D + D Gamma = 0,      Gamma = diag((-1)^grade)

Derived from the chain condition and the definition of D, not assumed. The proof is one
line of block bookkeeping: (Gamma D)[d-1,d] = (-1)^{d-1} B_d and (D Gamma)[d-1,d] =
(-1)^d B_d, so the blocks cancel. D connects only consecutive grades, and consecutive
grades have opposite parity, so every block of the anticommutator vanishes.

It is exactly zero on any relational complex, which makes measuring it on one vacuous.
The content is the contrapositive: an operator that does NOT satisfy it is not a graded
Dirac, and the residual is a distance from being one. That is the only interesting way to
call it, and it is why the measurement takes an arbitrary operator rather than a RexGraph.

Lem 4.8 is the consequence worth having: equiweight forces the Hodge symmetry
dim K^{p,q} = dim K^{q,p}, i.e. D's nonzero spectrum is symmetric about 0, because Gamma
conjugates D to -D and therefore pairs every eigenvector with one of opposite sign.
"""

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _branching(ptr, idx):
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int32), np.asarray(idx, np.int32))


def _triangle_with_face():
    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    rex.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1.0, 1.0, 1.0])])
    return rex


FIXTURES = {
    "pairwise triangle": lambda: RexGraph(sources=np.array([0, 1, 2], np.int32),
                                          targets=np.array([1, 2, 0], np.int32)),
    "triangle + face": _triangle_with_face,
    "lone k=3": lambda: _branching([0, 3], [0, 1, 2]),
    "k=4 + two legs": lambda: _branching([0, 4, 6, 8], [0, 1, 2, 3, 3, 4, 3, 5]),
    "double-T": lambda: _branching([0, 3, 6], [0, 1, 2, 0, 1, 3]),
}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_equiweight_is_exactly_zero(name):
    """Structural identity, so the tolerance is 0 and not an epsilon."""
    assert FIXTURES[name]().equiweight_residual == 0


@pytest.mark.parametrize("name", list(FIXTURES))
def test_equiweight_agrees_with_the_assembled_dirac(name):
    """The property is computed from the block structure without materialising D. Check
    it against the assembled operator so the shortcut is verified, not trusted."""
    rex = FIXTURES[name]()
    D = np.asarray(rex.dirac_operator, dtype=float)
    g = np.asarray(rex.dirac_grading, dtype=float)
    anti = g[:, None] * D + D * g[None, :]
    assert np.abs(anti).max() < 1e-12
    assert rex.equiweight_residual == 0


@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_grading_alternates_by_grade(name):
    rex = FIXTURES[name]()
    g = np.asarray(rex.dirac_grading)
    nV, nE, nF = int(rex.nV), int(rex.nE), int(rex.nF_hodge)
    assert g.shape == (nV + nE + nF,)
    assert np.all(g[:nV] == 1)
    assert np.all(g[nV:nV + nE] == -1)
    if nF:
        assert np.all(g[nV + nE:] == 1)


#### Lem 4.8: the consequence
@pytest.mark.parametrize("name", list(FIXTURES))
def test_hodge_symmetry_of_the_dirac_spectrum(name):
    """Gamma D Gamma = -D, so lambda is an eigenvalue iff -lambda is. This is what
    equiweight buys: the nonzero spectrum is symmetric about zero."""
    rex = FIXTURES[name]()
    ev = np.sort(np.asarray(rex.dirac_eigenvalues, dtype=float))
    assert np.allclose(ev, -ev[::-1], atol=1e-9), ev


@pytest.mark.parametrize("name", list(FIXTURES))
def test_hodge_symmetry_counts_match(name):
    """The integer form of the same statement: as many strictly positive modes as
    strictly negative ones, with the rest harmonic."""
    rex = FIXTURES[name]()
    sym = rex.hodge_symmetry
    assert sym["n_positive"] == sym["n_negative"]
    assert sym["n_positive"] + sym["n_negative"] + sym["n_harmonic"] == int(rex.dirac_dimension)
    assert sym["symmetric"] is True


def test_harmonic_count_is_the_betti_sum():
    """dim ker D = sum of the Betti numbers, an exact integer, no eigen threshold."""
    rex = _triangle_with_face()
    b = [int(x) for x in rex.betti]
    assert rex.hodge_symmetry["n_harmonic"] == sum(b) == int(rex.dirac_harmonic_count)


#### the useful direction: a foreign operator
def test_a_foreign_operator_has_a_nonzero_residual():
    """The contrapositive, which is the only non-vacuous reading. An operator with a
    block joining two grades of the SAME parity is not a graded Dirac, and the residual
    says so."""
    from rexgraph.dirac_propagator import equiweight_residual

    sizes = (2, 2, 2)
    n = sum(sizes)
    D = np.zeros((n, n))
    D[0, 2] = D[2, 0] = 1.0                      # grade 0 <-> grade 1: legal
    assert equiweight_residual(D, sizes) == 0.0
    D[0, 4] = D[4, 0] = 1.0                      # grade 0 <-> grade 2: same parity
    assert equiweight_residual(D, sizes) > 0.0


def test_the_residual_scales_with_the_offending_mass():
    """It is a distance, not a flag."""
    from rexgraph.dirac_propagator import equiweight_residual

    sizes = (1, 1, 1)
    out = []
    for a in (0.5, 1.0, 4.0):
        D = np.zeros((3, 3))
        D[0, 2] = D[2, 0] = a                    # grade 0 <-> grade 2
        out.append(equiweight_residual(D, sizes))
    assert out[0] < out[1] < out[2]


def test_a_diagonal_block_also_violates_it():
    """A grade talking to itself is the other way to fail: Gamma agrees with itself, so
    the anticommutator doubles rather than cancels."""
    from rexgraph.dirac_propagator import equiweight_residual

    D = np.zeros((3, 3))
    D[1, 1] = 1.0                                # within grade 1
    assert equiweight_residual(D, (1, 2, 0)) > 0.0


def test_residual_rejects_a_size_mismatch():
    from rexgraph.dirac_propagator import equiweight_residual

    with pytest.raises(ValueError):
        equiweight_residual(np.zeros((3, 3)), (1, 1, 1, 1))
