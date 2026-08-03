"""Regression: the always-sparse spectral bundle truncates the L0 eigenbasis
for nV>2000 (k<<nV eigenpairs). Feeding that truncated basis to the dense
nV x nV L0 Cython kernels (build_edge_signal / build_response_operators) reads
out of bounds -> uncatchable C-level SIGSEGV. Two live agent paths did this:
corpus._spectral_score and the pipeline quality gate. These tests pin the fix:
guard on the full basis, and use the matrix-free B1^+ equivalent when truncated.

The corpus half now goes through agent.scoring.interfacing_score, so the guard sits
inside RexGraph.interfacing_vector (it routes to the sparse bundle on the same
condition) instead of being hand-rolled in the caller. The property under test is
unchanged: a truncated-basis graph must return a finite score, not a SIGSEGV.
"""
import numpy as np
from agent.corpus import CorpusBuilder
from agent.pipeline_runner import _context_quality_gate

from rexgraph.graph import RexGraph


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


def test_document_scoring_large_graph_does_not_segfault():
    """Ranking an nV>2000 document returns a finite score instead of crashing the
    process: interfacing_vector takes the sparse bundle when the basis is truncated."""
    from agent.scoring import interfacing_score

    rex = _connected_graph(2500, extra_edges=1500)
    labels = _labels(rex.nV)
    r = interfacing_score(rex, labels, ["W0", "W1", "W2", "W3"])  # >=2 shared tokens
    assert isinstance(r["score"], float)
    assert np.isfinite(r["score"])
    assert r["n_shared"] >= 2


def test_corpus_score_document_large_graph_does_not_segfault():
    """The same property through the corpus entry point callers actually use."""
    rex = _connected_graph(2500, extra_edges=1500)
    labels = _labels(rex.nV)
    score = CorpusBuilder._score_document(None, _Doc(rex, labels), _EC(["W0", "W1", "W2"]))
    assert isinstance(score, float) and np.isfinite(score)


def test_quality_gate_large_graph_is_permissive_not_crash():
    """The gate skips on a truncated-basis graph rather than feeding it to the dense
    response-operator kernel. Skipping measures nothing, so it cannot refuse."""
    from agent.pipeline_runner import GATE_OK

    rex = _connected_graph(2500, extra_edges=1500)
    labels = _labels(rex.nV)
    g = _context_quality_gate(rex, labels, "w0 w1 w2 w3 topic")
    assert g["verdict"] == GATE_OK
    assert any("truncated" in r for r in g["reasons"])


def test_quality_gate_small_graph_runs_dense_path():
    """On a small graph the full basis is available, so the dense kernels run and
    produce a channel score."""
    from agent.pipeline_runner import GATE_OK, GATE_WARN

    rex = _connected_graph(40, extra_edges=30)
    labels = _labels(rex.nV)
    g = _context_quality_gate(rex, labels, "w0 w1 w2 w3 w4 topic")
    assert g["verdict"] in (GATE_OK, GATE_WARN)
    assert g["score"] is not None


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
