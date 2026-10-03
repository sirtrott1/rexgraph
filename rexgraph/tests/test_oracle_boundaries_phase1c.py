"""Pre IO Phase -1C: production evaluators cannot fall through to dense oracles."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from rexgraph.evaluator import DenseEvaluationRefused
from rexgraph.graph import RexGraph


def _chain(n_edges: int) -> RexGraph:
    src = np.arange(n_edges, dtype=np.int32)
    tgt = src + 1
    return RexGraph(sources=src, targets=tgt)


def _disjoint_triangles(count: int) -> RexGraph:
    src, tgt, triangles = [], [], []
    for j in range(count):
        a, b, c = 3 * j, 3 * j + 1, 3 * j + 2
        src.extend((a, b, a))
        tgt.extend((b, c, c))
        triangles.append((a, b, c))
    return RexGraph.from_simplicial(
        np.asarray(src, np.int32),
        np.asarray(tgt, np.int32),
        np.asarray(triangles, np.int32),
    )


def test_spectral_perturbation_default_never_needs_dense_eigh(monkeypatch):
    rex = _chain(96)

    def forbidden(*_a, **_k):
        pytest.fail("default spectral perturbation touched a dense eigensolver")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    f_e, f_f = rex.spectral_perturbation(1)
    assert f_e.shape == (96,) and f_f.shape == (0,)
    assert np.linalg.norm(f_e) == pytest.approx(1.0, abs=1e-7)


def test_spectral_solver_failure_does_not_densify_large_operator(monkeypatch):
    rex = _chain(96)
    monkeypatch.setattr(sla, "eigsh", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forced")))

    def forbidden(*_a, **_k):
        pytest.fail("large spectral fallback densified")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    with pytest.raises(DenseEvaluationRefused, match="refusing implicit dense eigensolve"):
        rex.spectral_perturbation(1)


def test_field_psd_solver_failure_does_not_densify_large_block(monkeypatch):
    # 30 disjoint filled triangles -> nE+nF = 120, deliberately above the tiny fallback bound.
    rex = _disjoint_triangles(30)
    monkeypatch.setattr(sla, "eigsh", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forced")))

    original = sp.csr_matrix.toarray

    def forbidden(self, *args, **kwargs):  # noqa: ARG001
        pytest.fail("large field PSD fallback called CSR.toarray()")

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    try:
        with pytest.raises(DenseEvaluationRefused, match="refusing implicit dense eigensolve"):
            _ = rex.field_coupling_psd
    finally:
        # pytest would restore this too; explicit restoration keeps later code in this test
        # independent if the assertion above itself fails while debugging.
        monkeypatch.setattr(sp.csr_matrix, "toarray", original)


def test_measure_full_basis_obeys_configured_dense_limit():
    from rexgraph.core import _common

    rex = _chain(8)
    saved = int(_common.get_algorithm_config()["eigen_dense_limit"])
    try:
        _common.configure_algorithms(eigen_dense_limit=1)
        with pytest.raises(DenseEvaluationRefused, match="full eigenbasis"):
            rex.measure_in_eigenbasis(np.ones(rex.nE, dtype=np.complex128), method="auto")
        with pytest.raises(RuntimeError, match="not cached"):
            rex.measure_in_eigenbasis(np.ones(rex.nE, dtype=np.complex128), method="cached")
    finally:
        _common.configure_algorithms(eigen_dense_limit=saved)


def test_dense_spectral_oracle_is_explicit_and_agrees_as_an_eigensignal():
    rex = _chain(6)
    sparse, _ = rex.spectral_perturbation(1, method="sparse")
    dense, _ = rex.spectral_perturbation(1, method="dense_oracle")
    # Eigenvector sign is arbitrary; compare the rank one projector.
    assert np.allclose(np.outer(sparse, sparse), np.outer(dense, dense), atol=1e-7)


def test_legacy_analysis_solver_failure_is_also_bounded(monkeypatch):
    from rexgraph.analysis import _low_frequencies

    M = sp.eye(100, format="csr")
    monkeypatch.setattr(sla, "eigsh", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forced")))

    def forbidden(*_a, **_k):
        pytest.fail("legacy analysis densified a large sparse operator")

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    with pytest.raises(DenseEvaluationRefused, match="refusing implicit dense eigensolve"):
        _low_frequencies(M, 8)


def test_channel_gap_lanczos_failure_uses_sparse_kernel_robust_fallback(monkeypatch):
    from rexgraph.sparse_character import _smallest_pos_small_kernel

    M = sp.diags([0.0] + [1.0] * 599, format="csr")
    monkeypatch.setattr(sla, "eigsh", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forced")))
    original = sp.csr_matrix.toarray

    def forbid_full(self, *args, **kwargs):
        if self.shape == M.shape:
            pytest.fail("large channel-gap fallback densified the full operator")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbid_full)
    assert _smallest_pos_small_kernel(M) == pytest.approx(1.0, abs=1e-8)


def test_fiedler_dense_kernel_obeys_configured_eigen_limit():
    from rexgraph.core import _common
    from rexgraph.fiedler import _dense

    saved = int(_common.get_algorithm_config()["eigen_dense_limit"])
    try:
        _common.configure_algorithms(eigen_dense_limit=1)
        with pytest.raises(DenseEvaluationRefused, match="full eigenbasis"):
            _dense(sp.eye(2, format="csr"), 2, 2)
    finally:
        _common.configure_algorithms(eigen_dense_limit=saved)


def test_character_response_tiles_dense_rhs_by_memory_policy(monkeypatch):
    import rexgraph.fiedler as fiedler

    rex = _chain(12)
    seeds = np.arange(8, dtype=int)
    reference = rex.character_response(seeds)
    monkeypatch.setattr(fiedler, "solve_block_width", lambda *_a, **_k: 2)
    tiled = rex.character_response(seeds)
    assert tiled.shape == reference.shape
    assert np.allclose(tiled, reference, atol=1e-9)


def test_dense_materialization_chokepoint_checks_allocation(monkeypatch):
    import rexgraph.evaluator as evaluator
    from rexgraph.dense_matrix import ensure_dense

    seen = []
    monkeypatch.setattr(
        evaluator, "check_dense_allocation",
        lambda operation, rows, cols: seen.append((operation, rows, cols)),
    )
    M = sp.eye(3, format="csr")
    out = ensure_dense(M, operation="test dense boundary")
    assert out.shape == (3, 3)
    assert seen == [("test dense boundary", 3, 3)]
