"""
Tests for rexgraph.analysis -- dashboard analysis pipeline.

Verifies:
    - analyze: returns dict with expected top-level keys
    - Spectral data: spectra block has correct key set
    - Topology: Betti numbers, Euler characteristic correct
    - Hodge: percentages sum to ~100
    - Energy: E_kin/E_pot present, regime is a valid string
    - Vertices/edges: correct count, expected per-item keys
    - Faces: present for simplicial complex
    - Standard metrics: PageRank, betweenness, clustering, Louvain
    - RCF sections: structural_character, rcfe, void_complex (when available)
    - analyze_signal: returns signal sub-dict
    - analyze_quotient: returns quotient sub-dict
    - analyze_all: returns both signal and quotient
    - Helpers: _build_flow, _classify_attributes, _partition_from_fiedler
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.analysis import (
    analyze,
    analyze_signal,
    analyze_quotient,
    analyze_all,
    _build_flow,
    _classify_attributes,
    _partition_from_fiedler,
)


# Fixtures

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def k4_data(k4):
    return analyze(k4, run_perturbation=False)


@pytest.fixture
def triangle_data(triangle):
    return analyze(triangle, run_perturbation=False)


# Top-level structure

class TestAnalyzeKeys:

    def test_top_level_keys(self, k4_data):
        for key in ["meta", "vertices", "edges", "faces", "topology",
                     "coupling", "spectra", "hodge", "energy", "analysis",
                     "overlap"]:
            assert key in k4_data, f"Missing key: {key}"

    def test_meta(self, k4_data):
        meta = k4_data["meta"]
        assert meta["nV"] == 4
        assert meta["nE"] == 6
        assert meta["nF"] == 4


# Vertices and edges

class TestVerticesEdges:

    def test_vertex_count(self, k4_data):
        assert len(k4_data["vertices"]) == 4

    def test_edge_count(self, k4_data):
        assert len(k4_data["edges"]) == 6

    def test_vertex_keys(self, k4_data):
        v = k4_data["vertices"][0]
        for key in ["id", "x", "y", "degree", "role", "pagerank",
                     "betweenness", "clustering", "community"]:
            assert key in v, f"Missing vertex key: {key}"

    def test_edge_keys(self, k4_data):
        e = k4_data["edges"][0]
        for key in ["id", "source", "target", "type", "flow",
                     "gradRaw", "curlRaw", "harmRaw", "rho",
                     "Ekin", "Epot"]:
            assert key in e, f"Missing edge key: {key}"


# Topology

class TestTopology:

    def test_betti(self, k4_data):
        topo = k4_data["topology"]
        assert topo["b0"] == 1
        assert topo["b1_filled"] == 0
        assert topo["b2"] == 1

    def test_chain_valid(self, k4_data):
        assert k4_data["topology"]["chainOk"] is True

    def test_euler(self, k4_data):
        assert k4_data["topology"]["euler"] == 4 - 6 + 4  # = 2

    def test_triangle_betti(self, triangle_data):
        topo = triangle_data["topology"]
        assert topo["b0"] == 1
        assert topo["b1_filled"] == 1  # one unfilled cycle
        assert topo["b2"] == 0


# Spectra

class TestSpectra:

    def test_spectra_keys(self, k4_data):
        sp = k4_data["spectra"]
        for key in ["L0", "L1_down", "L1_full", "L2", "LO", "RL1", "RL"]:
            assert key in sp, f"Missing spectrum: {key}"

    def test_no_old_keys(self, k4_data):
        sp = k4_data["spectra"]
        assert "L1_alpha" not in sp
        assert "Lambda" not in sp

    def test_l0_length(self, k4_data):
        assert len(k4_data["spectra"]["L0"]) == 4  # nV eigenvalues


# Hodge

class TestHodge:

    def test_percentages(self, k4_data):
        h = k4_data["hodge"]
        total = h["gradPct"] + h["curlPct"] + h["harmPct"]
        assert abs(total - 100.0) < 1.0  # within 1%


# Energy

class TestEnergy:

    def test_energy_keys(self, k4_data):
        e = k4_data["energy"]
        assert "E_kin" in e
        assert "E_pot" in e
        assert "ratio" in e
        assert "regime" in e

    def test_regime_valid(self, k4_data):
        assert k4_data["energy"]["regime"] in ("kinetic", "potential", "crossover")


# Coupling

class TestCoupling:

    def test_coupling_keys(self, k4_data):
        c = k4_data["coupling"]
        assert "alpha_G" in c
        assert "alpha_T" in c
        assert "fiedler_RL1" in c


# Standard metrics

class TestStandardMetrics:

    def test_present(self, k4_data):
        sm = k4_data["analysis"]["standard_metrics"]
        assert "n_communities" in sm
        assert "modularity" in sm
        assert "top_pagerank" in sm


# Faces

class TestFaces:

    def test_face_count(self, k4_data):
        assert len(k4_data["faces"]) == 4

    def test_no_faces(self, triangle_data):
        assert len(triangle_data["faces"]) == 0


# RCF sections (may not be available in all builds)

class TestRCFSections:

    def test_structural_character(self, k4_data):
        if "structural_character" in k4_data:
            sc = k4_data["structural_character"]
            assert "per_edge" in sc
            assert "per_vertex" in sc
            assert len(sc["per_edge"]) == 6

    def test_rcfe(self, k4_data):
        if "rcfe" in k4_data:
            rcfe = k4_data["rcfe"]
            assert "curvature" in rcfe
            assert "strain" in rcfe
            assert "bianchi_ok" in rcfe

    def test_void_complex(self, k4_data):
        if "void_complex" in k4_data:
            vc = k4_data["void_complex"]
            assert "n_voids" in vc
            assert "n_potential" in vc

    def test_fiber_bundle(self, k4_data):
        if "fiber_bundle" in k4_data:
            fb = k4_data["fiber_bundle"]
            assert "mean_phi_sim" in fb
            assert "mean_fb_sim" in fb

    def test_dirac(self, k4_data):
        if "dirac" in k4_data:
            d = k4_data["dirac"]
            assert "spectrum" in d
            assert "total_dim" in d
            assert d["total_dim"] == 4 + 6 + 4  # nV + nE + nF

    def test_hypermanifold(self, k4_data):
        if "hypermanifold" in k4_data:
            hm = k4_data["hypermanifold"]
            assert "n_levels" in hm
            assert "manifolds" in hm


# Signal dashboard

class TestAnalyzeSignal:

    def test_returns_signal_key(self, k4):
        data = analyze_signal(k4, n_steps=10, t_max=1.0)
        assert "signal" in data
        assert data["meta"]["mode"] == "signal"


# Quotient dashboard

class TestAnalyzeQuotient:

    def test_returns_quotient_key(self, k4):
        data = analyze_quotient(k4, max_vertex_presets=2)
        assert "quotient" in data
        assert data["meta"]["mode"] == "quotient"


# Combined

class TestAnalyzeAll:

    def test_returns_both(self, k4):
        data = analyze_all(k4, signal_steps=10, signal_t_max=1.0,
                           max_vertex_presets=2)
        assert "signal" in data
        assert "quotient" in data
        assert data["meta"]["mode"] == "all"


# Helpers

class TestHelpers:

    def test_build_flow_default(self, triangle):
        flow = _build_flow(triangle, None, None, triangle.nE)
        assert flow.shape == (3,)
        assert np.all(flow > 0)

    def test_build_flow_negative(self, triangle):
        attrs = {"type": ["activation", "inhibition", "activation"]}
        flow = _build_flow(triangle, attrs, ["inhibition"], triangle.nE)
        assert flow[1] < 0  # inhibition edge is negative
        assert flow[0] > 0

    def test_classify_attributes_numeric(self):
        attrs = {"score": ["1.0", "2.0", "3.0"]}
        result = _classify_attributes(attrs, 3)
        assert len(result) == 1
        assert result[0]["kind"] == "numeric"

    def test_classify_attributes_categorical(self):
        attrs = {"type": ["a", "b", "a"]}
        result = _classify_attributes(attrs, 3)
        assert len(result) == 1
        assert result[0]["kind"] == "categorical"

    def test_partition_from_fiedler(self):
        vec = np.array([0.5, -0.3, 0.1, -0.8])
        parts = _partition_from_fiedler(vec)
        assert parts == ["A", "B", "A", "B"]


# Edge attrs integration

class TestEdgeAttrs:

    def test_with_type_column(self, triangle):
        attrs = {"type": ["binding", "inhibition", "binding"]}
        data = analyze(triangle, edge_attrs=attrs, run_perturbation=False)
        assert data["edges"][0]["type"] == "binding"
        assert data["edges"][1]["type"] == "inhibition"

    def test_with_vertex_labels(self, triangle):
        data = analyze(triangle, vertex_labels=["A", "B", "C"],
                       run_perturbation=False)
        ids = [v["id"] for v in data["vertices"]]
        assert "A" in ids
        assert "B" in ids
        assert "C" in ids
