"""Degenerate operators and model placement preserve the advertised NN contracts."""
import numpy as np
import pytest


@pytest.mark.parametrize("n", [0, 3])
def test_fixed_zero_wave_identity_and_gradients(n):
    torch = pytest.importorskip("torch")
    from rexgraph.nn.rcf_torch import wave_apply
    x = torch.arange(n * 2, dtype=torch.float64).reshape(n, 2).requires_grad_()
    time = torch.tensor(1e20, dtype=torch.float64, requires_grad=True)
    real, imag = wave_apply(torch.zeros((n, n), dtype=x.dtype), x, time)
    assert torch.equal(real, x)
    assert torch.equal(imag, torch.zeros_like(x))
    dx, dt = torch.autograd.grad((real + imag).sum(), (x, time))
    assert torch.equal(dx, torch.ones_like(x))
    assert dt.item() == 0


def test_learnable_zero_wave_matches_matrix_exponential_derivative():
    torch = pytest.importorskip("torch")
    from rexgraph.nn.rcf_torch import wave_apply
    lap = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[1.0], [3.0]], dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    real, imag = wave_apply(lap, x, time)
    expected = torch.matrix_exp(-1j * time * lap) @ x.to(torch.complex128)
    torch.testing.assert_close(real, expected.real)
    torch.testing.assert_close(imag, expected.imag)
    actual_grads = torch.autograd.grad((real + imag).sum(), (lap, x, time), retain_graph=True)
    reference_grads = torch.autograd.grad((expected.real + expected.imag).sum(), (lap, x, time))
    for actual, reference in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_sparse_propagators_match_dense_output_and_input_time_gradients(layout):
    torch = pytest.importorskip("torch")
    from rexgraph.nn.rcf_torch import heat_apply, spectral_bound, wave_apply
    lap = torch.tensor([[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]], dtype=torch.float64)
    sparse = lap.to_sparse() if layout == "coo" else lap.to_sparse_csr()
    assert spectral_bound(sparse) == spectral_bound(lap) == 4
    x = torch.arange(6, dtype=lap.dtype).reshape(3, 2).requires_grad_()
    time = torch.tensor(.3, dtype=lap.dtype, requires_grad=True)
    for function in (heat_apply, wave_apply):
        reference = function(lap, x, time)
        actual = function(sparse, x, time)
        if function is heat_apply:
            reference, actual = (reference,), (actual,)
        for result, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(result, expected)
        actual_grads = torch.autograd.grad(sum(a.sum() for a in actual), (x, time), retain_graph=True)
        reference_grads = torch.autograd.grad(sum(a.sum() for a in reference), (x, time), retain_graph=True)
        for result, expected in zip(actual_grads, reference_grads, strict=True):
            torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("matrix_free", [False, True])
def test_constant_chebyshev_polynomial_needs_no_matvec(matrix_free):
    torch = pytest.importorskip("torch")
    from rexgraph.nn.rcf_torch import cheb_apply, cheb_apply_op
    x = torch.tensor([[2.], [3.]], requires_grad=True)
    coeffs = torch.tensor([4.], requires_grad=True)
    if matrix_free:
        def forbidden(_):
            pytest.fail("a constant polynomial must not evaluate its operator")
        out = cheb_apply_op(forbidden, x, coeffs, 0.)
    else:
        out = cheb_apply(torch.zeros(2, 2), x, coeffs)
    torch.testing.assert_close(out, 4 * x)
    dx, dc = torch.autograd.grad(out.sum(), (x, coeffs))
    torch.testing.assert_close(dx, torch.full_like(x, 4))
    torch.testing.assert_close(dc, torch.tensor([5.]))


def test_single_token_bidirectional_attention_has_finite_output_and_gradients():
    torch = pytest.importorskip("torch")
    from rexgraph.nn.relational_attention import PropagatorAttention
    model = PropagatorAttention(4, 2).double()
    x = torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True)
    out, diagnostic = model(x, return_diag=True)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert all(np.isfinite(v) for v in diagnostic.values())
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("placement", ["constructor", "to"])
def test_float32_cochain_trains_and_roundtrips_without_dtype_drift(placement, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from rexgraph.graph import RexGraph
    from rexgraph.flow.cochain import CoParticipationCochain
    graph = RexGraph(sources=np.array([0, 0], np.int32), targets=np.array([1, 2], np.int32))
    model = (CoParticipationCochain(graph, 2, dtype=torch.float32) if placement == "constructor"
             else CoParticipationCochain(graph, 2).float())
    model.fit([0, 1], [True, True], epochs=2)
    assert model._adj.dtype == model.Z.dtype == torch.float32
    model.save_safetensors(tmp_path / "model.safetensors")
    loaded = CoParticipationCochain.load_safetensors(tmp_path / "model.safetensors")
    assert loaded.Z.dtype == loaded._adj.dtype == model.Z.dtype
    assert torch.equal(loaded.Z, model.Z)
    assert torch.equal(loaded._adj.to_dense(), model._adj.to_dense())
    loaded.fit([0, 1], [True, True], epochs=1)


@pytest.mark.parametrize("method", ["greens", "hodge", "hodgesgd"])
def test_custom_optimizer_closure_runs_with_gradients_enabled(method):
    torch = pytest.importorskip("torch")
    from rexgraph.nn.optim import build_optimizer
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = build_optimizer([parameter], method=method, lr=.1)
    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss
    result = optimizer.step(closure)
    assert result.item() == 4 and parameter.item() < 2


@pytest.mark.parametrize("scale", [0., 1e-20, 1., 1e20])
def test_spectral_entropy_is_scale_invariant_and_zero_on_zero_mass(scale):
    torch = pytest.importorskip("torch")
    from rexgraph.nn.rcf_torch import renyi2, renyi_order
    lap = torch.diag(torch.tensor([1., 2.], dtype=torch.float64)) * scale
    expected2 = 0. if scale == 0 else -np.log(5 / 9)
    expected3 = 0. if scale == 0 else -.5 * np.log(9 / 27)
    assert renyi2(lap).item() == pytest.approx(expected2)
    assert renyi_order(lap, 3).item() == pytest.approx(expected3)
