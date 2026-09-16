"""The native support transpose preserves both integer coordinate widths."""
import numpy as np
import pytest

from rexgraph.core import _rex


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_full_support_and_readonly_inputs(dtype):
    ptr = np.array([0, 4, 4, 5], dtype=dtype)
    idx = np.array([0, 1, 2, 3, 1], dtype=dtype)
    ptr.flags.writeable = idx.flags.writeable = False
    vp, vi = _rex.build_vertex_to_edge_csr_general(5, 3, ptr, idx)
    np.testing.assert_array_equal(vp, [0, 1, 3, 4, 5, 5])
    np.testing.assert_array_equal(vi, [0, 0, 2, 0, 0])
    assert vp.dtype == vi.dtype == dtype


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("nv,ne,ptr,idx", [
    (-1, 0, [0], []), (1, -1, [0], []), (2, 2, [0, 1], [0]),
    (2, 1, [1, 1], [0]), (2, 1, [0, 2], [0]),
    (2, 3, [0, 2, 1, 2], [0, 1]), (2, 1, [0, 1], [-1]),
    (2, 1, [0, 1], [2]),
])
def test_invalid_support_is_refused_before_native_access(dtype, nv, ne, ptr, idx):
    with pytest.raises(ValueError):
        _rex.build_vertex_to_edge_csr_general(
            nv, ne, np.array(ptr, dtype=dtype), np.array(idx, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_empty_support(dtype):
    vp, vi = _rex.build_vertex_to_edge_csr_general(
        0, 0, np.array([0], dtype=dtype), np.array([], dtype=dtype))
    assert vp.tolist() == [0] and vi.tolist() == []
