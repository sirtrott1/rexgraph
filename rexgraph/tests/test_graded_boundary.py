"""Tests for the general, graded, mixed-arity boundary builder.

Covers:
  * ``build_graded_boundaries`` on mixed-arity edges (witness / pairwise / branching)
    and mixed n-gon faces - arities, signs and ``d^2 = 0``;
  * a genuine grade-3 complex (soccer-ball solid + octahedron solid + square pyramid)
    with ``B1 B2 = 0`` AND ``B2 B3 = 0`` sparsely, and correct Betti numbers;
  * ``RexGraph.from_cells`` round-tripping against ``from_simplicial`` and exposing
    the full graded boundary list;
  * back-compat: the classic constructors still yield correct ``graded_boundaries``.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
import rexgraph.graded_boundary as gb


# ---------------------------------------------------------------------------
# build_graded_boundaries: arity, signs, d^2 = 0
# ---------------------------------------------------------------------------

def test_mixed_arity_edges_positional_signs():
    """Witness (nnz=1), pairwise (nnz=2) and branching (nnz=3) edges in one grade,
    with positional signs (first -1, rest +1)."""
    cells = [
        4,                                  # 4 vertices
        [[0], [0, 1], [1, 2, 3]],           # witness, pairwise, branching
    ]
    B = gb.build_graded_boundaries(cells)
    assert len(B) == 1
    B1 = B[0]
    assert B1.shape == (4, 3)
    # Column arities = nnz per column.
    arities = np.diff(B1.tocsc().indptr)
    assert list(arities) == [1, 2, 3]
    B1d = B1.toarray()
    # Witness: single +... no, positional first entry is -1.
    assert B1d[0, 0] == -1.0
    # Pairwise: source -1, target +1.
    assert B1d[0, 1] == -1.0 and B1d[1, 1] == 1.0
    # Branching: first -1, rest +1.
    assert B1d[1, 2] == -1.0 and B1d[2, 2] == 1.0 and B1d[3, 2] == 1.0


def test_explicit_signed_cells():
    """Explicit ``[(index, sign), ...]`` form gives arbitrary orientations."""
    cells = [
        3,
        [[(0, -1), (1, 1)], [(1, -1), (2, 1)], [(0, -1), (2, 1)]],
    ]
    B = gb.build_graded_boundaries(cells)
    B1 = B[0].toarray()
    expected = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=float)
    assert np.array_equal(B1, expected)


def test_mixed_ngon_faces_chain_condition():
    """A triangle and a square (quadrilateral) face over a shared edge set: mixed
    grade-2 arity, and B1 B2 = 0 sparsely."""
    # Vertices 0..4. Edges around a triangle (0,1,2) and a square (0,2,3,4)? Keep it
    # simple: two independent faces sharing no edge, mixed arity 3 and 4.
    # Triangle 0-1-2 ; square 3-4-5-6 (disjoint) but reuse vertices to stay compact.
    verts = 6
    edges = [
        [0, 1], [1, 2], [0, 2],                 # triangle edges 0,1,2
        [3, 4], [4, 5], [5, 0], [0, 3],         # square edges 3,4,5,6  (0-3-4-5 loop)
    ]
    # Triangle face: +e0 +e1 -e2 (closes on vertices).
    tri = [(0, 1.0), (1, 1.0), (2, -1.0)]
    # Square face over loop 0->3->4->5->0 : e6(0-3)+e3(3-4)+e4(4-5)+e5(5-0)
    sq = [(6, 1.0), (3, 1.0), (4, 1.0), (5, 1.0)]
    cells = [verts, edges, [tri, sq]]
    B = gb.build_graded_boundaries(cells)
    assert [b.shape for b in B] == [(6, 7), (7, 2)]
    face_arities = sorted(np.diff(B[1].tocsc().indptr).tolist())
    assert face_arities == [3, 4]                # mixed arity
    ok, res = gb.verify_chain(B)
    assert ok and res == 0.0


# ---------------------------------------------------------------------------
# grade-3 complexes: B1B2 = 0, B2B3 = 0, Betti numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder,expect", [
    (gb.solid_octahedron_3rex, dict(nV=6, nE=12, nF=8, arities={3})),
    (gb.square_pyramid_3rex, dict(nV=5, nE=8, nF=5, arities={3, 4})),
    (gb.truncated_icosahedron_3rex, dict(nV=60, nE=90, nF=32, arities={5, 6})),
])
def test_grade3_solid_is_contractible(builder, expect):
    """Every solid 3-ball: d^2=0 through both consecutive pairs, and Betti = (1,0,0,0)
    (contractible). The truncated icosahedron is the mixed 5/6-gon soccer ball."""
    cbg = builder()
    assert cbg[0] == expect["nV"]
    assert len(cbg[1]) == expect["nE"]
    assert len(cbg[2]) == expect["nF"]
    assert len(cbg[3]) == 1                       # exactly one volume
    assert set(len(f) for f in cbg[2]) == expect["arities"]

    B = gb.build_graded_boundaries(cbg)
    assert [b.shape for b in B] == [
        (expect["nV"], expect["nE"]),
        (expect["nE"], expect["nF"]),
        (expect["nF"], 1),
    ]
    # d^2 = 0 for BOTH consecutive pairs (B1B2 and B2B3), sparsely.
    ok, res = gb.verify_chain(B)
    assert ok and res == 0.0
    # Explicit per-pair check that neither product is silently empty-by-shape.
    assert (B[0] @ B[1]).nnz == 0 or np.abs((B[0] @ B[1]).data).max() < 1e-12
    assert (B[1] @ B[2]).nnz == 0 or np.abs((B[1] @ B[2]).data).max() < 1e-12
    # Euler characteristic of the solid = 1.
    assert expect["nV"] - expect["nE"] + expect["nF"] - 1 == 1
    assert gb.betti_numbers(B) == [1, 0, 0, 0]


def test_soccer_ball_shell_is_2sphere():
    """The truncated-icosahedron SHELL (drop the volume) is a topological 2-sphere:
    Betti = (1, 0, 1) - beta_2 = 1 detects the enclosed void."""
    cbg = gb.truncated_icosahedron_3rex()
    B_shell = gb.build_graded_boundaries(cbg[:3])       # B1, B2 only
    ok, res = gb.verify_chain(B_shell)
    assert ok and res == 0.0
    assert gb.betti_numbers(B_shell) == [1, 0, 1]


def test_octahedron_shell_is_2sphere():
    cbg = gb.solid_octahedron_3rex()
    B_shell = gb.build_graded_boundaries(cbg[:3])
    assert gb.betti_numbers(B_shell) == [1, 0, 1]


# ---------------------------------------------------------------------------
# graded_laplacians
# ---------------------------------------------------------------------------

def test_graded_laplacians_shapes_and_psd():
    """Per-grade Hodge Laplacians L_0..L_G have the right shapes and are symmetric
    PSD; harmonic dimension (ker L_g) matches Betti."""
    cbg = gb.solid_octahedron_3rex()
    B = gb.build_graded_boundaries(cbg)
    L = gb.graded_laplacians(B)
    sizes = [B[0].shape[0]] + [b.shape[1] for b in B]   # 6, 12, 8, 1
    assert [Lg.shape for Lg in L] == [(n, n) for n in sizes]
    betti = gb.betti_numbers(B)
    for g, Lg in enumerate(L):
        A = Lg.toarray()
        assert np.allclose(A, A.T)                        # symmetric
        w = np.linalg.eigvalsh(A)
        assert w.min() > -1e-9                            # PSD
        nker = int(np.sum(w < 1e-9))
        assert nker == betti[g]                           # ker L_g == beta_g


# ---------------------------------------------------------------------------
# RexGraph.from_cells: round-trip and graded_boundaries contract
# ---------------------------------------------------------------------------

def test_from_cells_matches_from_simplicial_single_triangle():
    src = np.array([0, 1, 0]); tgt = np.array([1, 2, 2])
    tris = np.array([[0, 1, 2]])
    rs = RexGraph.from_simplicial(src, tgt, tris)
    rc = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]],
                              [[(0, 1), (1, 1), (2, -1)]]])
    assert np.array_equal(rs.B1, rc.B1)
    assert np.array_equal(rs.B2, rc.B2)
    assert rc.chain_valid


def test_from_cells_2rex_tetrahedron_chain_valid_and_grades():
    """A tetrahedron 2-rex built via from_cells: same dimensions as from_simplicial,
    chain-valid, and graded_boundaries returns exactly 2 grades."""
    src = np.array([0, 0, 0, 1, 1, 2]); tgt = np.array([1, 2, 3, 2, 3, 3])
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    rs = RexGraph.from_simplicial(src, tgt, tris)

    edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    # Reuse the simplicial B2 columns as explicit signed cells so the two agree.
    B2s = rs.B2
    faces = []
    for f in range(B2s.shape[1]):
        col = B2s[:, f]
        faces.append([(int(e), float(col[e])) for e in np.nonzero(col)[0]])
    rc = RexGraph.from_cells([4, edges, faces])

    assert (rc.nV, rc.nE, rc.nF) == (rs.nV, rs.nE, rs.nF)
    assert rc.chain_valid
    assert np.array_equal(rc.B1, rs.B1)
    assert np.array_equal(rc.B2, rs.B2)
    assert len(rc.graded_boundaries()) == 2


def test_from_cells_grade3_exposes_three_boundaries():
    cbg = gb.solid_octahedron_3rex()
    r = RexGraph.from_cells(cbg)
    B = r.graded_boundaries()
    assert len(B) == 3
    assert [b.shape for b in B] == [(6, 12), (12, 8), (8, 1)]
    ok, res = gb.verify_chain(B)
    assert ok and res == 0.0
    assert gb.betti_numbers(B) == [1, 0, 0, 0]
    assert r._graded_duals is not None and len(r._graded_duals) == 1


def test_from_cells_soccer_ball_grade3():
    cbg = gb.truncated_icosahedron_3rex()
    r = RexGraph.from_cells(cbg)
    assert (r.nV, r.nE, r.nF) == (60, 90, 32)
    B = r.graded_boundaries()
    assert [b.shape for b in B] == [(60, 90), (90, 32), (32, 1)]
    ok, res = gb.verify_chain(B)
    assert ok and res == 0.0
    assert gb.betti_numbers(B) == [1, 0, 0, 0]


def test_from_cells_respects_isolated_vertex_count():
    """A declared vertex not touched by any edge is still counted."""
    r = RexGraph.from_cells([5, [[0, 1], [1, 2]]])     # vertices 3,4 isolated
    assert r.nV == 5
    assert r.graded_boundaries()[0].shape == (5, 2)


# ---------------------------------------------------------------------------
# Back-compat: classic constructors -> correct graded_boundaries
# ---------------------------------------------------------------------------

def test_hypergraph_is_1rex():
    he_idx = np.array([0, 1, 2, 0, 1, 1, 2], dtype=np.int64)
    he_ptr = np.array([0, 3, 5, 7], dtype=np.int64)
    r = RexGraph.from_hypergraph(he_ptr, he_idx)
    B = r.graded_boundaries()
    assert len(B) == 1                                  # 1-rex: only B1
    assert B[0].shape == (3, 3)


def test_simplicial_is_2rex():
    src = np.array([0, 0, 0, 1, 1, 2]); tgt = np.array([1, 2, 3, 2, 3, 3])
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    r = RexGraph.from_simplicial(src, tgt, tris)
    B = r.graded_boundaries()
    assert len(B) == 2                                  # 2-rex: B1, B2
    assert B[0].shape == (4, 6) and B[1].shape == (6, 4)
    ok, res = gb.verify_chain(B)
    assert ok and res == 0.0
    # Tetrahedron surface = 2-sphere: beta_2 = 1.
    assert gb.betti_numbers(B) == [1, 0, 1]


def test_graph_is_1rex():
    r = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    B = r.graded_boundaries()
    assert len(B) == 1
    assert B[0].shape == (3, 3)
    assert gb.betti_numbers(B) == [1, 1]                # a 3-cycle: beta_1 = 1
