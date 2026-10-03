
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rexgraph.nn import PackedTernaryLinear


def test_transpose_construction_unpacks_bounded_tiles(monkeypatch):
    from rexgraph.core import _ternary
    original = _ternary.unpack
    sizes = []
    def unpack(P, S, columns):
        sizes.append(len(P))
        return original(P, S, columns)
    monkeypatch.setattr(_ternary, "unpack", unpack)
    a = np.random.default_rng(35).integers(-1, 2, size=(131, 129), dtype=np.int8)
    layer = PackedTernaryLinear(a)
    x = torch.randn(3, 129, dtype=torch.float64, requires_grad=True)
    result = layer(x)
    result.sum().backward()
    torch.testing.assert_close(result, x @ torch.tensor(a, dtype=x.dtype).T)
    torch.testing.assert_close(x.grad, torch.tensor(a.sum(axis=0), dtype=x.dtype).expand_as(x))
    assert sizes == [64, 64, 3]


def test_construction_movement_and_assigned_state_work_inside_inference_mode():
    with torch.inference_mode():
        layer = PackedTernaryLinear([[1, -1]])
        layer._apply(lambda t: t.clone())  # force new buffers on this CPU only host
        x = torch.tensor([[2., 3.]])
        torch.testing.assert_close(layer(x), torch.tensor([[-1.]]))
        state = {key: value.clone() for key, value in layer.state_dict().items()}
        layer.load_state_dict(state, assign=True)
        assert all(not torch.is_inference(t) for t in (layer.P, layer.S, layer.PT, layer.ST))
        layer.S.zero_()
        torch.testing.assert_close(layer(x), torch.tensor([[5.]]))


def test_external_stream_lifetime_is_registered_before_partial_launch_failure(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from rexgraph.nn import packed as M
    from rexgraph import hip_ternary as H
    events = []
    class Tensor:
        dtype = torch.float32
        device = SimpleNamespace(type="cuda")
        def __init__(self, shape, name):
            self.shape, self.name = shape, name
        def reshape(self, *shape): return self
        def contiguous(self): return self
        def new_empty(self, shape): return Tensor(shape, "output")
        def data_ptr(self): return 4096
        def element_size(self): return 4
        def record_stream(self, stream): events.append(self.name)
    class Library:
        def ternary_f32_batch_async(self, *args):
            events.append("launch")
            return 0 if events.count("launch") == 1 else 1
    monkeypatch.setattr(torch.version, "hip", "host-test")
    monkeypatch.setattr(torch.cuda, "device", lambda d: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda d: SimpleNamespace(cuda_stream=123))
    monkeypatch.setattr(H, "_load", lambda: Library())
    with pytest.raises(RuntimeError, match="launch failed"):
        M._product(Tensor((65536, 64), "input"), Tensor((1, 1), "presence"),
                   Tensor((1, 1), "sign"), 1, 64, "hip", 256, 1)
    assert events == ["presence", "sign", "input", "output", "launch", "launch"]
from rexgraph.nn.packed import _launch_batches


@pytest.mark.parametrize("cols", [1, 63, 64, 65, 130])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_forward_and_transpose_gradients_match_dense(cols, batch_shape):
    a = np.random.default_rng(cols).integers(-1, 2, size=(7, cols), dtype=np.int8)
    layer = PackedTernaryLinear(a)
    x = torch.randn(*batch_shape, cols, dtype=torch.float64, requires_grad=True)
    expected = x @ torch.tensor(a.T.copy(), dtype=x.dtype)
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x, create_graph=True)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual_grad.sum(), x)[0],
                               torch.autograd.grad(expected_grad.sum(), x)[0], rtol=1e-12, atol=1e-12)


def test_packed_custom_function_passes_finite_difference_first_and_second_order_checks():
    layer = PackedTernaryLinear([[1, -1, 0], [0, 1, -1]])
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,))
    assert torch.autograd.gradgradcheck(layer, (x,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_input_precision_is_preserved_with_finite_gradients(dtype):
    layer = PackedTernaryLinear([[1, 0, -1], [-1, 1, 0]])
    x = torch.tensor([[1., 2., 3.]], dtype=dtype, requires_grad=True)
    out = layer(x)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), torch.tensor([[-2., 1.]]))
    out.sum().backward()
    assert x.grad.dtype == dtype and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("shape,batch", [((0, 3), (2, 3)), ((4, 0), (2, 0)), ((4, 3), (0, 3))])
def test_empty_maps_and_batches_have_the_correct_adjoint_shapes(shape, batch):
    layer = PackedTernaryLinear(np.zeros(shape, np.int8))
    x = torch.zeros(batch, dtype=torch.float64, requires_grad=True)
    y = layer(x)
    assert y.shape == (batch[0], shape[0])
    y.sum().backward()
    assert x.grad.shape == x.shape and not x.grad.count_nonzero()


def test_checkpoint_rebuilds_the_transpose_and_mutation_invalidates_it():
    layer = PackedTernaryLinear([[1, -1, 0], [0, 1, -1]])
    other = PackedTernaryLinear([[-1, 0, 1], [1, -1, 0]])
    assert set(layer.state_dict()) == {"P", "S"}
    layer.load_state_dict(other.state_dict())
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    y = layer(x)
    expected = x @ torch.tensor([[-1., 1.], [0., -1.], [1., 0.]], dtype=x.dtype)
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])
    layer.S.zero_()                    # all present entries become positive
    x = x.detach().requires_grad_(True)
    y = layer(x)
    torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], torch.tensor([[2., 1., 1.], [2., 1., 1.]], dtype=x.dtype))
    bad = {k: v.float() for k, v in other.state_dict().items()}
    with pytest.raises(RuntimeError, match="int64"):
        layer.load_state_dict(bad)


def test_explicit_hip_cannot_compute_on_cpu_inputs():
    with pytest.raises(ValueError, match="ROCm"):
        PackedTernaryLinear([[1, -1]], backend="hip")(torch.ones(2))


def test_asynchronous_abi_chunks_vectors_without_losing_tail_or_offsets():
    x = torch.ones(65536, 64)
    out = torch.empty(65536, 3)
    plane = torch.zeros(3, 1, dtype=torch.int64)
    calls = []
    class Library:
        def ternary_f32_batch_async(self, P, S, v, y, rows, words, nvec, block, stream):
            calls.append((v.value, y.value, nvec, stream.value))
            return 0
    _launch_batches(Library(), plane, plane, x, out, 3, 1, 256, 123)
    assert calls == [(x.data_ptr(), out.data_ptr(), 65535, 123),
                     (x.data_ptr()+65535*64*4, out.data_ptr()+65535*3*4, 1, 123)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_hip_stream_forward_and_backward_agree_with_torch(dtype):
    from rexgraph import hip_ternary as H
    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available() or not H.tensor_available():
        pytest.skip("ROCm device and current HIP tensor kernels are required")
    a = np.random.default_rng(7).integers(-1, 2, size=(9, 65), dtype=np.int8)
    layer = PackedTernaryLinear(a, backend="hip").cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.randn(4, 65, dtype=dtype, device="cuda", requires_grad=True)
        y = layer(x)
        expected = x @ torch.tensor(a.T.copy(), device="cuda", dtype=dtype)
        gradient = torch.autograd.grad(y.square().sum(), x)[0]
        want_grad = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(y, expected, rtol=1e-5 if dtype==torch.float32 else 1e-12,
                                   atol=1e-5 if dtype==torch.float32 else 1e-12)
        torch.testing.assert_close(gradient, want_grad, rtol=1e-5 if dtype==torch.float32 else 1e-12,
                                   atol=1e-4 if dtype==torch.float32 else 1e-11)
    stream.synchronize()
