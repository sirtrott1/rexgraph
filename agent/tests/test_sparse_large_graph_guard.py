"""Regression: the always-sparse spectral bundle truncates the L0 eigenbasis
for nV>2000 (k<<nV eigenpairs). Feeding that truncated basis to the dense
nV x nV L0 Cython kernels (build_edge_signal / build_response_operators) reads
out of bounds -> uncatchable C-level SIGSEGV. Two live agent paths did this:
corpus._spectral_score and the pipeline quality gate. These tests pin the fix:
guard on the full basis, and use the matrix-free B1^+ equivalent when truncated.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from agent.corpus import CorpusBuilder
from agent.pipeline_runner import _context_quality_gate


def _connected_graph(nV, extra_edges=0, seed=0):
    """A connected graph on nV vertices: a spanning path plus random chords."""
    rng = np.random.default_rng(seed)
    src = list(range(nV - 1))
    tgt = list(range(1, nV))
    for _ in range(extra_edges):
        a, b = int(rng.integers(nV)), int(rng.integers(nV))
        if a != b:
            src.append(a)
            tgt.append(b)
    return RexGraph(sources=np.array(src, dtype=np.int32),
                    targets=np.array(tgt, dtype=np.int32))


def _labels(nV):
    return [f"w{i}" for i in range(nV)]


class _Doc:
    def __init__(self, rex, labels):
        self.rex = rex
        self.vertex_labels = labels


class _EC:
    def __init__(self, labels):
        self.vertex_labels = labels


def test_large_graph_bundle_basis_is_truncated():
    """Precondition: for nV>2000 the sparse bundle carries a truncated L0 basis
    (this is exactly what makes the dense kernels unsafe)."""
    rex = _connected_graph(2500, extra_edges=1500)
    sb = rex.spectral_bundle
    evecs_L0 = sb['evecs_L0']
    assert rex.nV > 2000
    assert evecs_L0.shape[0] == rex.nV
    assert evecs_L0.shape[1] < rex.nV  # truncated -> dense kernel would segfault


def test_spectral_score_large_graph_does_not_segfault():
    """corpus._spectral_score on an nV>2000 doc returns a finite score via the
    LSQR (B1^+) fallback instead of crashing the process."""
    rex = _connected_graph(2500, extra_edges=1500)
    labels = _labels(rex.nV)
    doc = _Doc(rex, labels)
    query_ec = _EC(["W0", "W1", "W2", "W3"])  # >=2 shared tokens
    score = CorpusBuilder._spectral_score(None, doc, query_ec)
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_quality_gate_large_graph_is_permissive_not_crash():
    """The pipeline quality gate skips (returns True) on a truncated-basis graph
    rather than feeding it to the dense response-operator kernel."""
    rex = _connected_graph(2500, extra_edges=1500)
    labels = _labels(rex.nV)
    ok = _context_quality_gate(rex, labels, "w0 w1 w2 w3 topic")
    assert ok is True


def test_quality_gate_small_graph_runs_dense_path():
    """On a small graph (full basis) the gate runs the real dense kernels and
    returns a bool without error."""
    rex = _connected_graph(40, extra_edges=30)
    labels = _labels(rex.nV)
    ok = _context_quality_gate(rex, labels, "w0 w1 w2 w3 w4 topic")
    assert isinstance(ok, bool)


def test_lsqr_matches_dense_edge_signal_on_small_graph():
    """The matrix-free fallback psi = B1^+ rho (LSQR) equals the dense-kernel
    psi = B1^T L0^+ rho on a small graph where the full basis is available."""
    from rexgraph.core._interfacing import build_edge_signal
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    rex = _connected_graph(30, extra_edges=25)
    sb = rex.spectral_bundle
    assert sb['evecs_L0'].shape[1] == rex.nV  # full basis on the small graph

    rng = np.random.default_rng(1)
    rho = rng.standard_normal(rex.nV).astype(np.float64)

    B1 = np.ascontiguousarray(rex.B1, dtype=np.float64)
    psi_dense = build_edge_signal(
        rho, B1, sb['evals_L0'],
        np.ascontiguousarray(sb['evecs_L0'], dtype=np.float64),
        rex.nV, rex.nE,
    )
    psi_lsqr = lsqr(csr_matrix(B1), rho, atol=1e-10, btol=1e-10)[0]
    assert np.allclose(psi_dense, psi_lsqr, atol=1e-6)
