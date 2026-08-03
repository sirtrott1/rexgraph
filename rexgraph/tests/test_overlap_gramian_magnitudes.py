"""The overlap Gramian must carry per-entry boundary magnitudes, not a support count.

`overlap_gramian` and `overlap_gramian_sparse` both document themselves as
``K = |B1|^T |B1|``. On the general (branching) path they did not compute that: the
builder wrote ``np.ones(...)`` into M and then forced ``M.data[:] = 1.0``, so K was a
count of shared vertices. Under the star column every magnitude is 1 and the two agree,
which is why nothing caught it; they diverge the moment a boundary entry has a
magnitude other than 1, and they would diverge on every branching relation once the
column carries the share 1/(k-1).

Three properties hold here.

PER ENTRY, NOT PER ACCUMULATED ENTRY. The magnitudes have to be read off the boundary
structure, not off a dense signed B1. A self-loop lists its vertex twice with -1 and +1;
the dense form has already summed those to 0, and |0| != |-1| + |+1|. Recovering K from
the dense signed view is therefore impossible, which is what
`test_self_loop_limitations_that_remain_are_pinned` records. The compiled standard-only
kernel gets this right and is left alone here.

DENSE AND SPARSE AGREE. They are documented as the same quantity in two shapes.

THE COUNT IS STILL AVAILABLE where a count is what is wanted: the C channel is defined
on shared-vertex counts, and that is a different quantity from G, not a cheaper version
of it.
"""

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _entry_gramian(rex):
    """K built directly from the boundary structure, per entry, at unit magnitude.

    Deliberately independent of the implementation under test: it walks the CSR the
    same way the definition does rather than calling any RexGraph helper.
    """
    bp = np.asarray(rex._boundary_ptr)
    bi = np.asarray(rex._boundary_idx)
    nE, nV = int(rex.nE), int(rex.nV)
    M = np.zeros((nE, nV), dtype=float)
    for e in range(nE):
        s, t = bp[e], bp[e + 1]
        k = t - s
        # the magnitude profile, written out from the definition rather than read from
        # the implementation: |-1| at the distinguished entry, |1/(k-1)| on the rest,
        # unit magnitudes below arity 3 where the share is 1 anyway
        share = 1.0 / (k - 1) if k > 2 else 1.0
        for j in range(s, t):
            M[e, bi[j]] += 1.0 if j == s else share   # per entry: a repeat counts twice
    return M @ M.T


def _branching(ptr, idx):
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int32), np.asarray(idx, np.int32))


BRANCHING = {
    "lone k=3": ([0, 3], [0, 1, 2]),
    "lone k=5": ([0, 5], [0, 1, 2, 3, 4]),
    "two k=3 sharing a vertex": ([0, 3, 6], [0, 1, 2, 2, 3, 4]),
    "k=4 with two pairwise legs": ([0, 4, 6, 8], [0, 1, 2, 3, 3, 4, 3, 5]),
    "double-T": ([0, 3, 6], [0, 1, 2, 0, 1, 3]),
    "repeated vertex in a boundary": ([0, 3], [0, 1, 1]),
}


@pytest.mark.parametrize("name", list(BRANCHING))
def test_sparse_gramian_is_the_entry_gramian(name):
    """The branching path must sum magnitudes per boundary entry."""
    rex = _branching(*BRANCHING[name])
    got = rex.overlap_gramian_sparse.toarray()
    assert np.allclose(got, _entry_gramian(rex), atol=1e-12), name


@pytest.mark.parametrize("name", list(BRANCHING))
def test_dense_matches_sparse(name):
    """One quantity, two shapes."""
    rex = _branching(*BRANCHING[name])
    assert np.allclose(np.asarray(rex.overlap_gramian),
                       rex.overlap_gramian_sparse.toarray(), atol=1e-12), name


def test_a_repeated_boundary_entry_counts_twice():
    """Boundary [0,1,1] is arity 3, so the share is 1/2 and vertex 1 receives it twice.
    M is [1, 1] and K is 1^2 + 1^2 = 2.

    The point is that the two entries at vertex 1 are summed rather than collapsed. The
    old builder forced every entry to 1 AND collapsed the duplicate, which reaches the
    same 2 here by a route that does not generalise: at unit magnitudes the sum would
    have been 2, giving 5. The parametrised test above is what pins the general case.
    """
    rex = _branching([0, 3], [0, 1, 1])
    assert np.isclose(rex.overlap_gramian_sparse.toarray()[0, 0], 2.0)


def test_reading_the_gramian_does_not_mutate_the_complex():
    """A read must not move the model.

    `np.ascontiguousarray` does not copy an already-contiguous array and `csr_matrix`
    aliases the indptr and indices it is given, so a `sum_duplicates()` inside the
    builder reaches the graph's own `_boundary_ptr` unless the buffers are copied first.
    An arity-3 relation naming a vertex twice would have its pointer go [0, 3] -> [0, 2]
    on the first read of either Gramian, permanently, and read as arity 2 thereafter.
    """
    for accessor in ("overlap_gramian_sparse", "overlap_gramian"):
        rex = _branching([0, 3], [0, 1, 1])
        before = np.asarray(rex._boundary_ptr).tolist()
        idx_before = np.asarray(rex._boundary_idx).tolist()
        getattr(rex, accessor)
        assert np.asarray(rex._boundary_ptr).tolist() == before, accessor
        assert np.asarray(rex._boundary_idx).tolist() == idx_before, accessor
    # and the arity the graph reports is still the declared one
    rex = _branching([0, 3], [0, 1, 1])
    _ = rex.overlap_gramian_sparse
    ptr = np.asarray(rex._boundary_ptr)
    assert int(ptr[1] - ptr[0]) == 3


def test_repeated_reads_are_stable():
    """A second consequence of aliasing: the first read changed what the second read
    saw, so the accessor was not idempotent."""
    rex = _branching([0, 3], [0, 1, 1])
    first = rex.__class__.overlap_gramian_sparse.func(rex).toarray()
    second = rex.__class__.overlap_gramian_sparse.func(rex).toarray()
    assert np.array_equal(first, second)


def test_simple_graphs_are_untouched():
    """The standard-only path is a different kernel and is correct, including on
    self-loops where the dense signed B1 cannot express the magnitudes at all."""
    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    K = rex.overlap_gramian_sparse.toarray()
    assert np.array_equal(K, np.array([[2., 1., 1.],
                                       [1., 2., 1.],
                                       [1., 1., 2.]]))
    assert np.allclose(np.asarray(rex.overlap_gramian), K)


def test_the_c_channel_still_reads_counts():
    """G and C are different quantities. C is the weighted line-graph Laplacian over
    shared-vertex counts; making G a true Gramian must not silently redefine C."""
    from rexgraph.sparse_character import build_sparse_channels

    rex = _branching([0, 3, 5], [0, 1, 2, 2, 3])
    ch = dict(build_sparse_channels(rex))
    L_C = ch["L_C"].toarray()
    assert np.allclose(L_C.sum(axis=1), 0.0, atol=1e-12)     # a Laplacian
    assert (np.diag(L_C) >= 0).all()
