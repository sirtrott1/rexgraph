"""Public endpoint algorithms must not project a primary branching C1 relation.

The primary carrier is the boundary column.  These tests pin the migration
boundary: an old pairwise algorithm can still operate on an explicitly derived
pairwise section, but no public route may choose the first two participants of
a branching relation on the caller's behalf.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex


def _branching() -> RexGraph:
    return RexGraph.from_hypergraph(
        np.array([0, 4], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )


def test_primary_branching_c1_has_no_implicit_endpoint_shadow():
    rex = _branching()

    assert rex.sources is None
    assert rex.targets is None
    with pytest.raises(ValueError, match="two-endpoint representation"):
        rex._ensure_src_tgt()

    # The primary boundary remains present, with its exact canonical share.
    assert np.allclose(rex.B1[:, 0], [-1.0, 1 / 3, 1 / 3, 1 / 3])


def test_package_root_and_public_sparse_boundary_expose_the_primary_carrier():
    """A core-only user can import the public type without forcing a dense B1."""
    from rexgraph import RexGraph as RootRexGraph
    from rexgraph import TemporalRex as RootTemporalRex
    from rexgraph import relation_identity
    from rexgraph.core._sparse import DualCSR, to_scipy_csr

    rex = _branching()
    first = rex.B1_sparse
    assert RootRexGraph is RexGraph
    assert RootTemporalRex is TemporalRex
    assert isinstance(first, DualCSR)
    assert to_scipy_csr(first).shape == (4, 1)
    assert relation_identity(rex, 0) == (0, 1, 2, 3)

    # Mutation invalidates the public sparse carrier just like the old private cache.
    rex.add_hyperedges([[4, 5, 6]])
    current = rex.B1_sparse
    assert current is not first
    assert to_scipy_csr(current).shape == (7, 2)


@pytest.mark.parametrize(
    "operation",
    [
        lambda r: r._adjacency_bundle,
        lambda r: r.context_face_selection(np.ones((1, r.nV), dtype=np.uint8)),
        lambda r: r.typed_face_selection(np.zeros(r.nE, dtype=np.int32)),
        lambda r: r.face_data([], [], np.zeros(r.nE)),
        lambda r: r.void_complex,
        lambda r: r.L_frustration_weighted,
        lambda r: r.partition_communities(),
    ],
)
def test_legacy_endpoint_operations_refuse_primary_branching_c1(operation):
    with pytest.raises(ValueError, match="pairwise C1 complex"):
        operation(_branching())


def test_legacy_join_refuses_but_primary_join_preserves_arity():
    left, right = _branching(), _branching()

    with pytest.raises(ValueError, match="pairwise C1 complex"):
        left.inner_join(right, np.arange(4, dtype=np.int32))

    joined, report = left.join(right, correspondence={i: i for i in range(4)})
    assert report["shared_relations"] == 1
    assert joined.nE == 1
    assert np.diff(joined.boundary_ptr).tolist() == [4]
    assert np.allclose(joined.B1[:, 0], [-1.0, 1 / 3, 1 / 3, 1 / 3])


def test_branching_c1_does_not_use_the_pairwise_fiedler_kernel():
    rex = _branching()
    with pytest.raises(ValueError, match="exact ker\\(B1.T\\) deflation basis"):
        _ = rex.eigenvalues_L0

    from rexgraph.fiedler import deflated_operator, kernel_from_boundary

    with pytest.raises(ValueError, match="not defined for branching C1"):
        kernel_from_boundary(rex.graded_boundaries()[0])
    with pytest.raises(ValueError, match="not defined for branching C1"):
        deflated_operator(rex.graded_boundaries()[0])


def test_from_cells_canonicalizes_declared_branching_c1_orientation():
    rex = RexGraph.from_cells([
        3,
        [[(1, 1), (2, 1), (0, -1)]],
    ])

    assert np.diff(rex.boundary_ptr).tolist() == [3]
    assert np.allclose(rex.B1[:, 0], [-1.0, 1 / 2, 1 / 2])
    assert np.allclose(rex.graded_boundaries()[0].toarray()[:, 0],
                       [-1.0, 1 / 2, 1 / 2])


def test_from_cells_refuses_noncanonical_c1_signing_and_invalid_branching_c2():
    with pytest.raises(ValueError, match="one negative distinguished"):
        RexGraph.from_cells([3, [[(0, 1), (1, 1), (2, 1)]]])

    with pytest.raises(ValueError, match="repeats a relation"):
        RexGraph.from_cells([
            3,
            [[0, 1, 2]],
            [[(0, 1), (0, -1)]],
        ])


def test_from_cells_derives_exact_branching_c2_from_relation_support():
    rex = RexGraph.from_cells([
        3,
        [[0, 1, 2], [0, 1], [0, 2]],
        [[0, 1, 2]],
    ])

    assert np.array_equal(rex.B2[:, 0], [2.0, -1.0, -1.0])
    assert rex.chain_valid
    B1, B2 = rex.graded_boundaries()
    assert np.allclose((B1 @ B2).toarray(), 0.0)
    assert list(rex.betti) == [1, 0, 0]


def test_from_cells_accepts_exact_explicit_branching_c2_coefficients():
    rex = RexGraph.from_cells([
        3,
        [[0, 1, 2], [0, 1], [0, 2]],
        [[(0, 2), (1, -1), (2, -1)]],
    ])

    assert np.array_equal(rex.B2[:, 0], [2.0, -1.0, -1.0])
    assert rex.chain_valid


def test_from_cells_accepts_branching_higher_grade_only_when_the_exact_tower_closes():
    cells = [
        3,
        [[0, 1, 2], [0, 1], [0, 2]],
        [[0, 1, 2], [0, 1, 2]],
        [[0, 1]],
    ]
    rex = RexGraph.from_cells(cells)
    B1, B2, B3 = rex.graded_boundaries()

    assert np.allclose((B1 @ B2).toarray(), 0.0)
    assert np.array_equal((B2 @ B3).toarray(), np.zeros((3, 1)))

    with pytest.raises(ValueError, match="exact B2 B3 = 0 chain condition"):
        RexGraph.from_cells([*cells[:3], [[0]]])


def test_standard_temporal_store_declines_branching_snapshot_after_its_carrier_is_set():
    timeline = TemporalRex([
        (np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)),
    ])
    with pytest.raises(ValueError, match="standard TemporalRex snapshot"):
        timeline.append_snapshot(_branching())


def test_explicit_general_temporal_store_keeps_branching_snapshot():
    timeline = TemporalRex([], general=True)
    timeline.append_snapshot(_branching())

    restored = timeline.reconstruct_at(0)
    assert np.diff(restored.boundary_ptr).tolist() == [4]
    assert np.allclose(restored.B1[:, 0], [-1.0, 1 / 3, 1 / 3, 1 / 3])


def test_empty_temporal_store_infers_general_carrier_from_its_first_snapshot():
    timeline = TemporalRex([])
    timeline.append_snapshot(_branching())

    assert timeline._general
    assert np.diff(timeline.reconstruct_at(0).boundary_ptr).tolist() == [4]


@pytest.mark.parametrize(
    ("ptr", "idx", "match"),
    [
        ([0.0, 2.0], [0, 1], "integral"),
        ([0, 2], [0.5, 1.0], "integral"),
        ([0, 2], [True, False], "integral"),
        ([1, 3], [0, 1, 2], "start at zero"),
        ([0, 2, 1], [0, 1], "nondecreasing"),
        ([0, 2], [0], "terminal"),
        ([0, 0], [], "empty C1"),
        ([0, 2], [0, -1], "negative"),
        ([0, 3], [0, 0, 1], r"only an exact \[v, v\]"),
    ],
)
def test_primary_c1_carrier_refuses_nonexact_or_malformed_input(ptr, idx, match):
    with pytest.raises(ValueError, match=match):
        RexGraph.from_hypergraph(np.asarray(ptr), np.asarray(idx))


def test_primary_c1_carrier_preserves_witness_branch_and_deliberate_self_loop():
    rex = RexGraph.from_hypergraph(
        np.array([0, 1, 4, 6], dtype=np.int32),
        np.array([0, 0, 1, 2, 3, 3], dtype=np.int32),
    )
    assert np.diff(rex.boundary_ptr).tolist() == [1, 3, 2]
    assert np.allclose(rex.B1[:, 2], 0.0)


def test_primary_factories_carry_relation_identity_metric_and_sign_channels():
    ids = np.array([41, 42], dtype=np.int64)
    weights = np.array([Fraction(3, 2), Fraction(5, 3)], dtype=object)
    signs = np.array([-1, 1], dtype=np.int32)
    ptr = np.array([0, 3, 5], dtype=np.int32)
    idx = np.array([0, 1, 2, 2, 3], dtype=np.int32)

    rex = RexGraph.from_hypergraph(
        ptr, idx, relation_ids=ids, w_E=weights, signs=signs, directed=True,
    )
    assert rex.relation_ids.tolist() == [41, 42]
    assert rex.w_E.tolist() == [Fraction(3, 2), Fraction(5, 3)]
    assert rex._signs.tolist() == [-1, 1]
    assert rex._directed

    # The graded constructor carries the same C1 attribution without changing its
    # exact boundary declaration.
    cells = RexGraph.from_cells(
        [4, [[0, 1, 2], [2, 3]]], relation_ids=ids, w_E=weights, signs=signs,
    )
    assert cells.relation_ids.tolist() == [41, 42]
    assert cells.w_E.tolist() == [Fraction(3, 2), Fraction(5, 3)]
    assert cells._signs.tolist() == [-1, 1]


def test_public_sparse_c2_carriers_distinguish_declared_from_hodge_faces():
    from rexgraph.core._sparse import DualCSR, to_scipy_csr

    rex = RexGraph.from_cells([
        3,
        [[0, 1], [1, 2], [0, 2]],
        [[(0, 1), (1, 1), (2, -1)]],
    ])
    declared = rex.B2_sparse
    hodge = rex.B2_hodge_sparse
    assert declared is rex._B2_dual
    assert hodge is rex._B2_hodge_dual
    assert isinstance(declared, DualCSR)
    assert isinstance(hodge, DualCSR)
    assert to_scipy_csr(declared).shape == (3, 1)
    assert to_scipy_csr(hodge).shape == (3, 1)

    # C2 mutation invalidates both public sparse carriers, including the Hodge view.
    rex.remove_faces(np.array([1], dtype=np.int32))
    assert rex.B2_sparse is None
    assert rex.B2_hodge_sparse is None


def test_permissive_face_staging_keeps_a_nonclosing_declaration_out_of_hodge():
    """A candidate C2 remains inspectable while the exact Hodge stack excludes it."""
    from rexgraph.core._sparse import to_scipy_csr

    rex = RexGraph(
        sources=np.array([0, 1, 2, 2], dtype=np.int32),
        targets=np.array([1, 2, 0, 3], dtype=np.int32),
    )
    # The first three relations close a triangle; relation 3 leaves C2 open at C0=3.
    rex.add_faces([[0, 1, 3]], [[1.0, 1.0, 1.0]])

    assert rex.nF == 1
    assert rex.nF_hodge == 0
    assert not rex.chain_valid
    assert rex.chain_report()["unbounded"] == [0]
    assert to_scipy_csr(rex.B2_sparse).shape == (4, 1)
    assert rex.B2_hodge_sparse is None


def test_primary_staging_uses_the_same_exact_c1_contract():
    rex = RexGraph(sources=np.array([0]), targets=np.array([1]))
    with pytest.raises(ValueError, match="integral"):
        rex.add_edges(np.array([1.5]), np.array([2]))
    with pytest.raises(ValueError, match="negative"):
        rex.add_edges(np.array([-1]), np.array([2]))
    with pytest.raises(ValueError, match="empty C1"):
        rex.add_hyperedges([[]])
    with pytest.raises(ValueError, match=r"only an exact \[v, v\]"):
        rex.add_hyperedges([[0, 0, 1]])

    rex.add_hyperedges([[2, 2]])
    rex._ensure_clean()
    assert np.allclose(rex.B1[:, 1], 0.0)


def test_declared_import_closes_every_chain_pair_and_refuses_empty_c2_cells():
    # Pairwise C1 used to skip the B1B2 import check; a declared complex cannot.
    with pytest.raises(ValueError, match="exact B1 B2 = 0 chain condition"):
        RexGraph.from_cells([3, [[0, 1], [1, 2]], [[0, 1]]])
    with pytest.raises(ValueError, match="empty boundary support"):
        RexGraph.from_cells([2, [[0, 1]], [[]]])

    rex = RexGraph(sources=np.array([0]), targets=np.array([1]))
    with pytest.raises(ValueError, match="nonempty relation support"):
        rex.add_faces([[]], [np.array([], dtype=float)])
    with pytest.raises(ValueError, match="integral"):
        rex.add_faces([[0.5]], [[1.0]])
    with pytest.raises(ValueError, match="cannot repeat"):
        rex.add_faces([[0, 0]], [[1.0, -1.0]])
    with pytest.raises(ValueError, match="nonempty relation support"):
        RexGraph(
            sources=np.array([0]), targets=np.array([1]),
            B2_col_ptr=np.array([0, 0]),
            B2_row_idx=np.array([], dtype=np.int32),
            B2_vals=np.array([], dtype=float),
        )


@pytest.mark.parametrize(
    "cells",
    [
        [2.5, [[0, 1]]],
        [2, [[0.5, 1]]],
        [2, [[(0.5, -1), (1, 1)]]],
        [2, [[0, 1]], [[0.5]]],
    ],
)
def test_declared_graded_basis_indices_are_never_float_coerced(cells):
    with pytest.raises(ValueError, match="exact integral|vertex count"):
        RexGraph.from_cells(cells)
