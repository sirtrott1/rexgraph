"""Shared polynomial work preserves each filter and its differentiable operator."""
import pytest

torch = pytest.importorskip("torch")

from rexgraph.nn import rcf_torch as R  # noqa: E402


@pytest.mark.parametrize("layout", ["dense", "coo", "csr", "batched"])
@pytest.mark.parametrize("order", [1, 16])
def test_shared_filters_match_independent_polynomials_and_gradients(layout, order):
    lap = torch.tensor([[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]], dtype=torch.float64)
    if layout == "batched":
        lap = torch.stack((lap, 2 * lap)).requires_grad_()
        x = torch.randn(2, 3, 2, dtype=lap.dtype, requires_grad=True)
    else:
        if layout == "coo":
            lap = lap.to_sparse()
        elif layout == "csr":
            lap = lap.to_sparse_csr()
        lap.requires_grad_()
        x = torch.randn(3, 2, dtype=lap.dtype, requires_grad=True)
    t = torch.tensor(.3, dtype=lap.dtype, requires_grad=True)
    bound = R.spectral_bound(lap)
    functions = (lambda l: torch.exp(-t * l), lambda l: torch.cos(t * l),
                 lambda l: -torch.sin(t * l))
    reference = [R.cheb_apply(lap, x, R.cheb_coeffs(f, order, bound, dtype=x.dtype), bound)
                 for f in functions]
    actual = R.propagator_apply(lap, x, t, K=order)
    for a, b in zip(actual, reference, strict=True):
        torch.testing.assert_close(a, b)
    variables = (x, t) if layout == "csr" else (lap, x, t)
    a_grads = torch.autograd.grad(sum(a.square().sum() for a in actual), variables,
                                  retain_graph=True, allow_unused=True)
    b_grads = torch.autograd.grad(sum(a.square().sum() for a in reference), variables, allow_unused=True)
    for a, b in zip(a_grads, b_grads, strict=True):
        if a is None or b is None:
            assert a is b is None  # a constant polynomial does not evaluate L
            continue
        if a.layout != torch.strided:
            a, b = a.to_dense(), b.to_dense()
        torch.testing.assert_close(a, b)


def test_shared_channels_follow_requested_order_and_keep_zero_operator_exact():
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(1e20, dtype=x.dtype, requires_grad=True)
    out = R.propagator_apply(torch.zeros(3, 3, dtype=x.dtype), x, t,
                             ("curl", "heat", "gradient", "curl"))
    for a, b in zip(out, (x * 0, x, x, x * 0), strict=True):
        assert torch.equal(a, b)
    dx, dt = torch.autograd.grad(sum(a.sum() for a in out), (x, t))
    assert torch.equal(dx, torch.full_like(x, 2)) and dt.item() == 0


@pytest.mark.parametrize("channels", [(), ("missing",)])
def test_invalid_channel_selection_fails_even_on_zero_operator(channels):
    with pytest.raises(ValueError, match="channels"):
        R.propagator_apply(torch.zeros(2, 2), torch.ones(2, 1), .3, channels)


def test_three_filters_use_only_one_operator_recurrence(monkeypatch):
    lap = torch.tensor([[1., -1.], [-1., 1.]])
    calls = []
    original = torch.Tensor.__matmul__
    def counted(self, other):
        if self is lap:
            calls.append(1)
        return original(self, other)
    monkeypatch.setattr(torch.Tensor, "__matmul__", counted)
    R.propagator_apply(lap, torch.randn(2, 4), .3, K=16, lam_max=2.)
    assert len(calls) == 15


@pytest.mark.parametrize("importance", [False, True])
def test_attention_outputs_and_all_gradients_match_independent_filters(monkeypatch, importance):
    import copy
    from rexgraph.nn.relational_attention import PropagatorAttention
    model = PropagatorAttention(8, 2, importance=importance).double()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    actual = model(x)[0]
    actual.sum().backward()
    def independent(lap, values, t, channels, K, lam_max):
        functions = {"heat": lambda l: torch.exp(-t * l), "gradient": lambda l: torch.cos(t * l),
                     "curl": lambda l: -torch.sin(t * l)}
        return tuple(R.cheb_apply(lap, values, R.cheb_coeffs(functions[c], K, lam_max,
                                                            dtype=values.dtype), lam_max)
                     for c in channels)
    monkeypatch.setattr(R, "propagator_apply", independent)
    x_ref = x.detach().clone().requires_grad_()
    expected = reference(x_ref)[0]
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, x_ref.grad)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
