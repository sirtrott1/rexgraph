"""Host side HIP ownership/ABI tests; these do not claim device execution."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pytest

from rexgraph import hip_ternary as H, ternary as tn
from rexgraph.core import _ternary


class HostRuntime:
    """ctypes backed memory with the compiled CPU product as a launch oracle."""
    def __init__(self, *, fail_alloc=0, fail_upload=0, launch_error=0, wrong_answer=False):
        self.live = {}
        self.allocations = self.uploads = self.frees = self.launches = 0
        self.fail_alloc, self.fail_upload = fail_alloc, fail_upload
        self.launch_error, self.wrong_answer = launch_error, wrong_answer

    def ternary_alloc(self, ptr, nbytes):
        self.allocations += 1
        if self.allocations == self.fail_alloc:
            return 2
        buffer = ctypes.create_string_buffer(nbytes)
        addr = ctypes.addressof(buffer)
        self.live[addr] = buffer
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0] = addr
        return 0

    def ternary_free(self, ptr):
        self.frees += 1
        del self.live[ptr.value]
        return 0

    def ternary_upload(self, dst, src, nbytes):
        self.uploads += 1
        if self.uploads == self.fail_upload:
            return 1
        ctypes.memmove(dst, src, nbytes)
        return 0

    def ternary_download(self, dst, src, nbytes):
        ctypes.memmove(dst, src, nbytes)
        return 0

    @staticmethod
    def array(ptr, dtype, shape):
        n = int(np.prod(shape))
        ctype = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
        return np.ctypeslib.as_array((ctype * n).from_address(ptr.value)).reshape(shape)

    def ternary_pm1_launch(self, P, S, X, K, out, rows, words, block):
        self.launches += 1
        if self.launch_error:
            return self.launch_error
        time.sleep(0.001)   # releases the GIL to exercise the reused buffer contract
        result = _ternary.matvec_pm1(self.array(P, np.uint64, (rows, words)),
                                   self.array(S, np.uint64, (rows, words)),
                                   self.array(X, np.uint64, (words,)),
                                   self.array(K, np.int64, (rows,)), 1)
        if self.wrong_answer:
            result += 1
        ctypes.memmove(out, result.ctypes.data, result.nbytes)
        return 0

    def ternary_f64_launch(self, P, S, V, out, rows, words, block):
        self.launches += 1
        result = _ternary.matvec_f64(self.array(P, np.uint64, (rows, words)),
                                   self.array(S, np.uint64, (rows, words)),
                                   self.array(V, np.float64, (words * 64,)), 1)
        ctypes.memmove(out, result.ctypes.data, result.nbytes)
        return 0


class DeviceHostRuntime(HostRuntime):
    def __init__(self):
        super().__init__()
        self.device = 0
        self.fail_set = False
        self.used_devices = []

    def ternary_get_device(self, ptr):
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = self.device
        return 0

    def ternary_set_device(self, device):
        if self.fail_set:
            return 101
        self.device = device
        return 0

    def ternary_pm1_launch(self, *args):
        self.used_devices.append(self.device)
        return super().ternary_pm1_launch(*args)

    def ternary_free(self, ptr):
        self.used_devices.append(self.device)
        return super().ternary_free(ptr)


def test_resident_launch_and_close_select_owner_and_restore_current_device(monkeypatch):
    lib = DeviceHostRuntime()
    monkeypatch.setattr(H, "_load", lambda: lib)
    handle = H.resident(tn.pack([[1, -1]]))
    lib.device = 1
    np.testing.assert_array_equal(handle.matvec([1, -1]), [2])
    assert lib.device == 1
    lib.fail_set = True
    with pytest.raises(RuntimeError):
        handle.close()
    assert lib.live  # failed selection must not discard ownership
    lib.fail_set = False
    handle.close()
    assert not lib.live and lib.device == 1
    assert lib.used_devices and set(lib.used_devices) == {0}


def test_availability_requalifies_when_current_device_changes(monkeypatch):
    lib = DeviceHostRuntime()
    monkeypatch.setattr(H, "_load", lambda: lib)
    monkeypatch.setattr(H, "_DEVICE_OK", None)
    monkeypatch.setattr(H, "_DEVICE_KEY", None)
    assert H.available()
    lib.device = 1
    lib.wrong_answer = True
    assert not H.available()
    assert not H.available()
    assert lib.launches == 2


@pytest.fixture
def runtime(monkeypatch):
    lib = HostRuntime()
    monkeypatch.setattr(H, "_load", lambda: lib)
    return lib


@pytest.mark.parametrize("block", [0, 1, 2, 3, 6, 96, 192, 255, 257, -1, 8.0, True])
def test_invalid_reduction_block_never_allocates(runtime, block):
    with pytest.raises(ValueError, match="block"):
        H.resident(tn.pack(np.ones((2, 128), dtype=np.int8)), block=block)
    assert runtime.allocations == 0


@pytest.mark.parametrize("values", [[257, 1], [1.1, -1], [0, 1], [np.nan, 1], [1j, -1], ["1", "-1"]])
def test_pm1_validation_precedes_narrowing_and_upload(runtime, values):
    with H.resident(tn.pack([[1, -1]])) as r:
        uploads = runtime.uploads
        with pytest.raises(ValueError, match="vector"):
            r.matvec(values)
        assert runtime.uploads == uploads
        with pytest.raises(ValueError, match="vector"):
            tn._pm1_cpu(tn.pack([[1, -1]]), np.asarray(values))


@pytest.mark.parametrize("alloc,upload", [(2, 0), (7, 0), (0, 1), (0, 3)])
def test_constructor_failure_releases_every_successful_allocation(monkeypatch, alloc, upload):
    lib = HostRuntime(fail_alloc=alloc, fail_upload=upload)
    monkeypatch.setattr(H, "_load", lambda: lib)
    with pytest.raises(RuntimeError):
        H.resident(tn.pack([[1, -1]]))
    assert not lib.live
    assert lib.frees == lib.allocations - bool(alloc)


def test_closed_handle_refuses_products_and_reentry(runtime):
    r = H.resident(tn.pack([[1, -1]]))
    r.close()
    r.close()
    for fn in (lambda: r.matvec([1, -1]), lambda: r.matvec_f64([.5, 2.]), r.__enter__):
        with pytest.raises(RuntimeError, match="closed"):
            fn()
    assert not runtime.live


@pytest.mark.parametrize("shape", [(0, 65), (4, 0), (0, 0)])
def test_empty_products_keep_shape_dtype_and_avoid_device_calls(runtime, shape):
    with H.resident(tn.pack(np.zeros(shape, dtype=np.int8))) as r:
        np.testing.assert_array_equal(r.matvec(np.ones(shape[1], dtype=np.int64)), np.zeros(shape[0], np.int64))
        np.testing.assert_array_equal(r.matvec_f64(np.ones(shape[1])), np.zeros(shape[0]))
    assert runtime.allocations == runtime.launches == 0


def test_resident_copies_and_concurrent_products_are_independent(runtime):
    rng = np.random.default_rng(4)
    a = rng.choice([-1, 0, 1], size=(5, 130)).astype(np.int8)
    vectors = rng.choice([-1, 1], size=(16, 130))
    with H.resident(tn.pack(a)) as r:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(r.matvec, vectors))
        for x, result in zip(vectors, results, strict=True):
            np.testing.assert_array_equal(result, a.astype(np.int64) @ x)
        v = rng.standard_normal(130)
        np.testing.assert_allclose(r.matvec_f64(v), a @ v, rtol=1e-12, atol=1e-12)
    assert not runtime.live
    assert runtime.allocations == runtime.frees == 7


@pytest.mark.parametrize("launch_error,wrong_answer,ok", [(0, False, True), (209, False, False), (0, True, False)])
def test_availability_requires_a_working_kernel_image_and_exact_result(monkeypatch, launch_error, wrong_answer, ok):
    lib = HostRuntime(launch_error=launch_error, wrong_answer=wrong_answer)
    monkeypatch.setattr(H, "_load", lambda: lib)
    monkeypatch.setattr(H, "_DEVICE_OK", None)
    assert H.available() is ok
    assert H.available() is ok
    assert lib.allocations == lib.frees == lib.launches == 1
    assert not lib.live


def test_an_incompatible_shared_object_is_not_a_backend(monkeypatch):
    monkeypatch.setattr(H, "_TRIED", False)
    monkeypatch.setattr(H, "_LIB", None)
    monkeypatch.setattr(H.os.path, "exists", lambda _: True)
    monkeypatch.setattr(H.ctypes, "CDLL", lambda _: object())
    assert H._load() is None


@pytest.mark.parametrize("mutation", ["shape", "dtype", "padding", "arity"])
def test_inconsistent_packed_operator_is_rejected_before_allocation(runtime, mutation):
    op = tn.pack([[1, -1]])
    if mutation == "shape":
        op = tn.TernaryOperator(op.P, op.S, (2, 2), op.K)
    elif mutation == "dtype":
        op = tn.TernaryOperator(op.P.astype(np.int64), op.S, op.shape, op.K)
    elif mutation == "padding":
        op.P[0, 0] |= np.uint64(1 << 63)
    else:
        op.K[0] = 1
    with pytest.raises(ValueError):
        H.resident(op)
    assert runtime.allocations == 0


@pytest.mark.parametrize("bp,bi,nv,w,coef", [
    ([1, 2], [0, 1], 2, None, None),
    ([0, 2, 1], [0], 2, None, None),
    ([0, 3], [0, 1], 2, None, None),
    ([0., 2.], [0, 1], 2, None, None),
    ([0, 2], [0, 2**32 + 1], 2, None, None),
    ([0, 2], [0, -1], 2, None, None),
    ([0, 2], [0, 2], 2, None, None),
    ([0, 2], [0, 1], 2.5, None, None),
    ([0, 2], [0, 1], 2, [], None),
    ([0, 2], [0, 1], 2, [[1]], None),
    ([0, 2], [0, 1], 2, [np.nan], None),
    ([0, 2], [0, 1], 2, [1j], None),
    ([0, 2], [0, 1], 2, None, [[-1, 1]]),
    ([0, 2], [0, 1], 2, None, [-1, np.inf]),
])
def test_invalid_tower_inputs_are_rejected_before_native_transpose(runtime, bp, bi, nv, w, coef):
    with pytest.raises(ValueError):
        H.channel_tower(bp, bi, nv, w, coefficients=coef)
    assert runtime.allocations == 0


@pytest.mark.parametrize("which", [1, 4, 6])
def test_tower_upload_failure_releases_all_owned_buffers(monkeypatch, which):
    lib = HostRuntime(fail_upload=which)
    monkeypatch.setattr(H, "_load", lambda: lib)
    with pytest.raises(RuntimeError, match="upload"):
        H.channel_tower([0, 2], [0, 1], 2)
    assert lib.allocations == lib.frees == which
    assert not lib.live


@pytest.mark.parametrize("bp,nv", [([0], 0), ([0], 5), ([0, 0, 0], 0)])
def test_empty_towers_are_exact_zero_without_a_library(monkeypatch, bp, nv):
    monkeypatch.setattr(H, "_load", lambda: pytest.fail("empty tower attempted HIP load"))
    for out in H.channel_tower(bp, [], nv):
        np.testing.assert_array_equal(out, np.zeros(len(bp)-1))


@pytest.mark.parametrize("block", [4, 8, 16, 32, 64, 128, 256])
def test_actual_hip_reductions_at_all_supported_blocks(block):
    if not H.available():
        pytest.skip("HIP device and compatible kernel image are required")
    rng = np.random.default_rng(80)
    a = rng.choice([-1, 0, 1], size=(7, 4097)).astype(np.int8)
    x = rng.choice([-1, 1], size=4097)
    v = rng.standard_normal(4097)
    with H.resident(tn.pack(a), block=block) as r:
        np.testing.assert_array_equal(r.matvec(x), a.astype(np.int64) @ x)
        np.testing.assert_allclose(r.matvec_f64(v), a @ v, rtol=1e-11, atol=1e-11)
