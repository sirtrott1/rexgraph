"""
Tests for rexgraph.core._faces - face classification, extraction, and metrics.

Verifies:
    - Proper vs self-loop face classification
    - B2_hodge filtering preserves chain condition
    - Vertex face count correctness
    - Face descriptor extraction
    - Face metrics shapes and ranges
    - No-face graphs return empty results
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def k4():
    """K4 with all 4 faces filled."""
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
    )


@pytest.fixture
def filled_triangle():
    return RexGraph.from_simplicial(
        sources=np.array([0, 1, 0], dtype=np.int32),
        targets=np.array([1, 2, 2], dtype=np.int32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )


@pytest.fixture
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


# Face Classification

class TestClassification:

    def test_k4_all_proper(self, k4):
        """K4 faces are all proper (3 unique vertices each)."""
        from rexgraph.core._faces import classify_faces
        fc = classify_faces(k4._B2_dual, k4._sources, k4._targets)
        assert fc['n_proper'] == 4
        assert fc['n_self_loop'] == 0

    def test_proper_mask_shape(self, k4):
        from rexgraph.core._faces import classify_faces
        fc = classify_faces(k4._B2_dual, k4._sources, k4._targets)
        assert fc['proper_mask'].shape == (k4.nF,)
        assert fc['self_loop_mask'].shape == (k4.nF,)

    def test_masks_complementary(self, k4):
        from rexgraph.core._faces import classify_faces
        fc = classify_faces(k4._B2_dual, k4._sources, k4._targets)
        assert np.all(fc['proper_mask'] != fc['self_loop_mask'])


# B2_hodge Filtering

class TestB2Hodge:

    def test_chain_condition_holds(self, k4):
        """B1 @ B2_hodge = 0."""
        B2h = k4.B2_hodge
        product = k4.B1 @ B2h
        assert np.max(np.abs(product)) < 1e-10

    def test_nF_hodge_equals_proper(self, k4):
        """nF_hodge = n_proper."""
        from rexgraph.core._faces import classify_faces
        fc = classify_faces(k4._B2_dual, k4._sources, k4._targets)
        assert k4.nF_hodge == fc['n_proper']

    def test_triangle_b2_hodge_shape(self, filled_triangle):
        B2h = filled_triangle.B2_hodge
        assert B2h.shape == (filled_triangle.nE, 1)


# Vertex Face Count

class TestVertexFaceCount:

    def test_k4_uniform(self, k4):
        """Every vertex in K4 touches 3 faces."""
        from rexgraph.core._faces import vertex_face_count
        vfc = vertex_face_count(k4._B2_dual, k4._sources, k4._targets, k4.nV)
        assert np.all(vfc == 3)

    def test_triangle_all_vertices(self, filled_triangle):
        """All 3 vertices of a filled triangle touch 1 face."""
        from rexgraph.core._faces import vertex_face_count
        vfc = vertex_face_count(
            filled_triangle._B2_dual,
            filled_triangle._sources,
            filled_triangle._targets,
            filled_triangle.nV)
        assert np.all(vfc == 1)

    def test_tree_zero(self, tree):
        """Tree has no faces, so vertex face count is all zeros."""
        assert tree.nF == 0


# Face Extraction

class TestExtraction:

    def test_k4_face_count(self, k4):
        from rexgraph.core._faces import extract_faces, classify_faces
        fc = classify_faces(k4._B2_dual, k4._sources, k4._targets)
        vn = [f"v{i}" for i in range(k4.nV)]
        en = [f"e{i}" for i in range(k4.nE)]
        faces = extract_faces(k4._B2_dual, k4._sources, k4._targets,
                            vn, en, face_class=fc)
        assert len(faces) == 4

    def test_face_has_required_keys(self, k4):
        from rexgraph.core._faces import extract_faces, classify_faces
        fc = classify_faces(k4._B2_dual, k4._sources, k4._targets)
        vn = [f"v{i}" for i in range(k4.nV)]
        en = [f"e{i}" for i in range(k4.nE)]
        faces = extract_faces(k4._B2_dual, k4._sources, k4._targets,
                            vn, en, face_class=fc)
        for f in faces:
            assert 'id' in f
            assert 'boundary' in f
            assert 'vertices' in f
            assert 'size' in f
            assert 'is_self_loop' in f

    def test_triangle_face_size(self, filled_triangle):
        from rexgraph.core._faces import extract_faces, classify_faces
        fc = classify_faces(filled_triangle._B2_dual,
                          filled_triangle._sources, filled_triangle._targets)
        vn = [f"v{i}" for i in range(filled_triangle.nV)]
        en = [f"e{i}" for i in range(filled_triangle.nE)]
        faces = extract_faces(filled_triangle._B2_dual,
                            filled_triangle._sources, filled_triangle._targets,
                            vn, en, face_class=fc)
        assert faces[0]['size'] == 3  # triangle has 3 boundary edges

    def test_k4_no_self_loops(self, k4):
        from rexgraph.core._faces import extract_faces, classify_faces
        fc = classify_faces(k4._B2_dual, k4._sources, k4._targets)
        vn = [f"v{i}" for i in range(k4.nV)]
        en = [f"e{i}" for i in range(k4.nE)]
        faces = extract_faces(k4._B2_dual, k4._sources, k4._targets,
                            vn, en, face_class=fc)
        assert all(not f['is_self_loop'] for f in faces)


# Face Metrics

class TestFaceMetrics:

    def test_metrics_shapes(self, k4):
        from rexgraph.core._faces import compute_face_metrics_i32, vertex_face_count
        vfc = vertex_face_count(k4._B2_dual, k4._sources, k4._targets, k4.nV)
        rho = np.zeros(k4.nE, dtype=np.float64)
        m = compute_face_metrics_i32(
            k4._B2_dual, k4._sources, k4._targets,
            k4.nV, k4.nE, k4.nF, vfc, rho)
        assert m['v_avg_contrib'].shape == (k4.nV,)
        assert m['e_avg_contrib'].shape == (k4.nE,)
        assert m['e_bnd_asym'].shape == (k4.nE,)
        assert m['f_concentration'].shape == (k4.nF,)

    def test_asymmetry_range(self, k4):
        """Boundary asymmetry is in [0, 1]."""
        from rexgraph.core._faces import compute_face_metrics_i32, vertex_face_count
        vfc = vertex_face_count(k4._B2_dual, k4._sources, k4._targets, k4.nV)
        rho = np.zeros(k4.nE, dtype=np.float64)
        m = compute_face_metrics_i32(
            k4._B2_dual, k4._sources, k4._targets,
            k4.nV, k4.nE, k4.nF, vfc, rho)
        assert np.all(m['e_bnd_asym'] >= -1e-12)
        assert np.all(m['e_bnd_asym'] <= 1.0 + 1e-12)

    def test_k4_zero_asymmetry(self, k4):
        """K4 is vertex-transitive, so boundary asymmetry is zero."""
        from rexgraph.core._faces import compute_face_metrics_i32, vertex_face_count
        vfc = vertex_face_count(k4._B2_dual, k4._sources, k4._targets, k4.nV)
        rho = np.zeros(k4.nE, dtype=np.float64)
        m = compute_face_metrics_i32(
            k4._B2_dual, k4._sources, k4._targets,
            k4.nV, k4.nE, k4.nF, vfc, rho)
        assert np.allclose(m['e_bnd_asym'], 0, atol=1e-12)

    def test_concentration_nonneg(self, k4):
        """Face concentration (CV) is nonnegative."""
        from rexgraph.core._faces import compute_face_metrics_i32, vertex_face_count
        vfc = vertex_face_count(k4._B2_dual, k4._sources, k4._targets, k4.nV)
        rho = np.zeros(k4.nE, dtype=np.float64)
        m = compute_face_metrics_i32(
            k4._B2_dual, k4._sources, k4._targets,
            k4.nV, k4.nE, k4.nF, vfc, rho)
        assert np.all(m['f_concentration'] >= -1e-12)

    def test_correlation_bounded(self, k4):
        """Pearson correlation is in [-1, 1]."""
        from rexgraph.core._faces import compute_face_metrics_i32, vertex_face_count
        vfc = vertex_face_count(k4._B2_dual, k4._sources, k4._targets, k4.nV)
        rho = np.random.RandomState(42).rand(k4.nE).astype(np.float64)
        m = compute_face_metrics_i32(
            k4._B2_dual, k4._sources, k4._targets,
            k4.nV, k4.nE, k4.nF, vfc, rho)
        assert -1.0 - 1e-10 <= m['asym_rho_corr'] <= 1.0 + 1e-10


# build_face_data

class TestBuildFaceData:

    def test_returns_all_keys(self, k4):
        from rexgraph.core._faces import build_face_data
        vn = [f"v{i}" for i in range(k4.nV)]
        en = [f"e{i}" for i in range(k4.nE)]
        rho = np.zeros(k4.nE, dtype=np.float64)
        d = build_face_data(k4._B2_dual, k4._sources, k4._targets,
                          k4.nV, vn, en, rho)
        assert 'faces' in d
        assert 'face_class' in d
        assert 'vertex_face_count' in d
        assert 'metrics' in d

    def test_no_faces_returns_empty(self, tree):
        """Graph with no faces has nF = 0 and empty face data."""
        assert tree.nF == 0
        assert tree.nF_hodge == 0
        # B2_hodge on a faceless graph is an (nE, 0) matrix
        assert tree.B2_hodge.shape == (tree.nE, 0)
