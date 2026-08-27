"""What a caller outside rexgraph actually passes in.

These readings are meant to be used from an agent, a model's feature path, or a
layer someone writes later, so the input door matters as much as the arithmetic.
Two failures were legible only to someone who already knew the internals: a torch
tensor carrying grad surfaced torch's own "Can't call numpy() on Tensor that
requires grad", and a wrong-length signal surfaced a scipy matmul dimension
mismatch naming neither nE nor the reading being taken.
"""

import numpy as np
import pytest

import rexgraph as rg
from rexgraph.graph import RexGraph
from rexgraph.harmonic_sparse import as_edge_signal


def _tri():
    r = RexGraph(sources=np.array([0, 1, 2], np.int32),
                 targets=np.array([1, 2, 0], np.int32))
    r._ensure_clean()
    return r


@pytest.mark.parametrize("make", [
    lambda n: np.arange(n, dtype=np.int64),
    lambda n: np.arange(n, dtype=np.float64),
    lambda n: np.arange(n, dtype=np.float32),
    lambda n: list(range(n)),
    lambda n: np.arange(n * 2, dtype=np.float64)[::2],      # non-contiguous
    lambda n: np.arange(n, dtype=np.float64).reshape(-1, 1),  # 2-D column
])
def test_every_ordinary_container_is_accepted(make):
    r = _tri()
    n = int(r.nE)
    out = np.asarray(r.harmonic_winding(make(n)))
    assert out.shape == (1,)


@pytest.mark.parametrize("reading", ["winding", "coords", "projection"])
def test_a_torch_tensor_is_accepted_and_detached(reading):
    """Accepted, and detached on purpose: a winding is an exact integer count, not a
    differentiable function of the signal, so there is no gradient to carry. The
    contract is 'use these to build features', and it must not read as a crash."""
    torch = pytest.importorskip("torch")
    r = _tri()
    t = torch.arange(int(r.nE), dtype=torch.float32, requires_grad=True)
    if reading == "winding":
        out = r.harmonic_winding(t)
    elif reading == "coords":
        out = rg.hodge_coords(r, t).harmonic
    else:
        out = rg.harmonic_projection(rg.harmonic_basis(r), t)
    assert isinstance(np.asarray(out), np.ndarray)
    assert not hasattr(out, "requires_grad")


@pytest.mark.parametrize("call", [
    lambda r, v: r.harmonic_winding(v),
    lambda r, v: rg.hodge_coords(r, v),
    lambda r, v: rg.harmonic_coords(r, v),
    lambda r, v: rg.harmonic_projection(rg.harmonic_basis(r), v),
])
def test_a_wrong_length_names_what_was_expected(call):
    r = _tri()
    with pytest.raises(ValueError, match=r"Expected 3 values for the edge flow, got 9"):
        call(r, np.ones(9))


def test_the_helper_says_nE_and_the_role():
    with pytest.raises(ValueError, match=r"Expected 4 values for the edge weight, got 2"):
        as_edge_signal(np.ones(2), 4, what="weight")


@pytest.mark.parametrize("src,tgt", [
    ([], []),                 # empty
    ([0], [1]),               # one relation, no cycle
    ([0, 0], [1, 2]),         # a tree
    ([0, 1], [0, 1]),         # self-loops
])
def test_the_degenerate_complexes_do_not_raise(src, tgt):
    """An agent sweeping a corpus meets all of these, and none of them is an error."""
    r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
    r._ensure_clean()
    n = int(r.nE)
    assert rg.harmonic_basis(r).shape[0] == n
    assert rg.multiplicity_dimension(r) >= 0
    assert rg.simple_cycle_dimension(r) >= 0
    assert rg.coordinate_dims(r)["independent"] == n
    assert np.asarray(r.harmonic_winding(np.ones(n))).size == rg.harmonic_basis(r).shape[1]


def test_add_faces_does_not_leave_a_stale_reading():
    """An agent that mutates a complex and reads it again must not get the old
    answer. Filling a bigon takes beta_1 to zero and every reading follows."""
    r = RexGraph(sources=np.array([0, 0], np.int32), targets=np.array([1, 1], np.int32))
    r._ensure_clean()
    assert (int(r.betti[1]), rg.harmonic_basis(r).shape[1]) == (1, 1)
    r.add_faces([[0, 1]], signs=[[1.0, -1.0]])
    assert (int(r.betti[1]), rg.harmonic_basis(r).shape[1]) == (0, 0)
    assert rg.simple_cycle_dimension(r) == 0


def test_the_new_surface_is_reachable_from_the_package_root():
    """A reading nobody can import is a reading nobody uses."""
    for name in ("harmonic_winding", "harmonic_basis", "harmonic_frame",
                 "hodge_coords", "coordinate_dims", "multiplicity_dimension",
                 "simple_cycle_dimension", "multiplicity_groups",
                 "multiplicity_homology_dimension", "cycle_vector", "cycle_vectors",
                 "minimum_cycle_basis", "complex_structure"):
        assert name in rg.__all__, name
        assert hasattr(rg, name), name
