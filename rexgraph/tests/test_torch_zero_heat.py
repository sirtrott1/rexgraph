"""Zero and empty operators retain the heat identity and training gradients."""
import pytest


@pytest.mark.parametrize("n", [0, 3])
def test_fixed_zero_heat_is_identity_with_input_and_time_gradients(n):
    torch = pytest.importorskip("torch")
    from rexgraph.nn.rcf_torch import heat_apply
    X = torch.arange(n * 2, dtype=torch.float64).reshape(n, 2).requires_grad_()
    t = torch.tensor(1e20, dtype=torch.float64, requires_grad=True)
    out = heat_apply(torch.zeros((n, n), dtype=torch.float64), X, t)
    assert torch.equal(out, X)
    dx, dt = torch.autograd.grad(out.sum(), (X, t))
    assert torch.equal(dx, torch.ones_like(X))
    assert dt.item() == 0


def test_learnable_zero_operator_retains_its_heat_derivative():
    torch = pytest.importorskip("torch")
    from rexgraph.nn.rcf_torch import heat_apply
    L = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    X = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    t = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    out = heat_apply(L, X, t)
    expected = torch.matrix_exp(-t * L) @ X
    torch.testing.assert_close(out, expected)
    actual_grads = torch.autograd.grad(out.sum(), (L, t), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (L, t))
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)
