"""Native channel scores must never acquire an implicit eigenbasis dependency."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from rexgraph import sparse_interfacing as si
from rexgraph.graph import RexGraph


@pytest.mark.parametrize("n", [6, 41, 1100])
def test_native_scores_do_not_touch_eigen_or_dense_paths(monkeypatch, n):
    rex = RexGraph.from_graph(np.arange(n), np.roll(np.arange(n), -1))
    def forbidden(*args, **kwargs):
        pytest.fail("native interfacing reached a dense or spectral path")
    monkeypatch.setattr(si, "_rl_spectrum", forbidden)
    monkeypatch.setattr(RexGraph, "_rl_eigen", property(forbidden))
    for module in (np.linalg, la):
        monkeypatch.setattr(module, "eigh", forbidden)
    monkeypatch.setattr(sp.linalg, "eigsh", forbidden)
    for cls in (sp.csr_matrix, sp.csc_matrix):
        monkeypatch.setattr(cls, "toarray", forbidden)
        monkeypatch.setattr(cls, "todense", forbidden)
    got = rex.interfacing_vector([0], [1.], None)
    assert np.isfinite(got['scores']).all()
    assert got['scores'].shape == got['channel_direction'].shape == (3,)
    for key in ('schrodinger', 'coverage', 'iv', 'sphere_pos', 'confidence'):
        assert got[key] is None
    assert got['mode_diagnostics']['status'] == 'not-requested'


def test_repeated_eigenspace_rotation_changes_mode_diagnostics(monkeypatch):
    rex = SimpleNamespace(nE=2, nV=2, _rl4_sparse=sp.eye(2, format='csr'))
    base = {'rho': np.ones(2), 'psi': np.array([1., 0.]), 'scores': np.ones(3),
            'efficiency': .5}
    monkeypatch.setattr(si, 'build_interfacing_bundle_sparse', lambda *a, **kw: dict(base))
    identity = np.eye(2)
    rotated = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
    a = si.build_interfacing_bundle_oracle(rex, [], [], None, eigenbasis=(np.ones(2), identity))
    b = si.build_interfacing_bundle_oracle(rex, [], [], None, eigenbasis=(np.ones(2), rotated))
    assert a['schrodinger'] == pytest.approx(1.)
    assert b['schrodinger'] == pytest.approx(.5)
    assert a['coverage'] == .5 and b['coverage'] == 1.
    assert a['mode_diagnostics']['basis_digest'] != b['mode_diagnostics']['basis_digest']
    np.testing.assert_array_equal(a['scores'], b['scores'])


def test_partial_spectrum_requires_explicit_request():
    rex = RexGraph.from_graph([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])
    partial = rex.interfacing_vector_oracle([0], [1.], None, mode_count=2)
    assert not partial['mode_diagnostics']['complete']
    assert partial['mode_diagnostics']['mode_count'] == 2
    full = rex.interfacing_vector_oracle([0], [1.], None)
    assert full['mode_diagnostics']['complete']
    assert full['mode_diagnostics']['basis_dependent']
    assert full['iv'].shape == (4,)
    np.testing.assert_array_equal(partial['scores'], full['scores'])


@pytest.mark.parametrize('count', [True, 0, -1, 2.5, 3])
def test_partial_mode_count_is_checked(count):
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    with pytest.raises(ValueError, match='mode_count'):
        rex.interfacing_vector_oracle([0], [1.], None, mode_count=count)


@pytest.mark.parametrize('basis', [(np.ones(2), np.ones((3, 2))),
                                 (np.ones(3), np.zeros((3, 3))),
                                 (np.array([np.nan] * 3), np.eye(3)),
                                 (np.full(3, 999.), np.eye(3))])
def test_supplied_oracle_basis_is_checked(basis):
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    with pytest.raises(ValueError, match='oracle basis'):
        rex.interfacing_vector_oracle([0], [1.], None, eigenbasis=basis)


@pytest.mark.parametrize('indices,weights,signal', [([.5], [1.], None),
                                                    ([0], [1., 2.], None),
                                                    ([0], [np.nan], None),
                                                    ([0], [1.], [1., 2.]),
                                                    ([0], [1.], [1j, 1j, 1j])])
def test_native_inputs_are_validated(indices, weights, signal):
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    with pytest.raises(ValueError):
        rex.interfacing_vector(indices, weights, signal)
