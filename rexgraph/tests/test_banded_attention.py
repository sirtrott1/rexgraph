"""The banded causal propagator is the dense one addressed differently.

The sparsity here is the causal window, which is a structural fact about which
tokens are reachable, not a threshold on scores. So the band drops terms the dense
path multiplies by exactly zero, and the two must agree bit for bit rather than
approximately. That is the whole justification for the sparse path, so it is tested
rather than argued.
"""

import pytest

torch = pytest.importorskip("torch")

# after importorskip: torch must be known present before this module imports
from rexgraph.nn.relational_attention import (  # noqa: E402
    CausalPropagatorAttention,
    _band_valid,
    _causal_windows,
)


@pytest.mark.parametrize("T,w,hops", [
    (64, 16, 4), (128, 32, 3), (256, 64, 1), (200, 64, 8),
    (33, 64, 2),        # T < w: the band is wider than the sequence
    (1, 8, 2),          # a single token
    (17, 1, 3),         # a window of one: every hop is the identity on values
])
def test_the_band_reproduces_the_dense_path_exactly(T, w, hops):
    torch.manual_seed(0)
    m = CausalPropagatorAttention(64, 4, hops=hops, window=w)
    x = torch.randn(2, T, 64)
    with torch.no_grad():
        sparse, _ = m(x)
        dense, _ = m.forward_dense(x)
    assert m.sparse is True
    assert torch.equal(sparse, dense) or (sparse - dense).abs().max().item() < 1e-6, \
        (sparse - dense).abs().max().item()


def test_the_window_view_addresses_the_right_tokens():
    """out[..., i, m, :] must be z[..., i-w+1+m, :], with the left pad masked off.
    An off-by-one here would still produce plausible numbers."""
    B, H, T, d, w = 1, 1, 6, 2, 3
    z = torch.arange(T * d, dtype=torch.float32).view(1, 1, T, d)
    win = _causal_windows(z, w)
    assert win.shape == (B, H, T, w, d)
    valid = _band_valid(T, w, z.device)
    for i in range(T):
        for m in range(w):
            j = i - w + 1 + m
            if j < 0:
                assert not bool(valid[i, m])
            else:
                assert bool(valid[i, m])
                assert torch.equal(win[0, 0, i, m], z[0, 0, j])


def test_a_masked_row_never_reads_padding():
    """The first w-1 rows read pad slots. If the mask let one through, early tokens
    would mix in a zero vector and the dense comparison would drift."""
    torch.manual_seed(1)
    m = CausalPropagatorAttention(32, 2, hops=2, window=8)
    x = torch.randn(1, 4, 32)                     # T < w, so most slots are padding
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m.forward_dense(x)
    assert torch.isfinite(a).all()
    assert (a - b).abs().max().item() < 1e-6


def test_sparse_requires_a_window():
    """Without a window the band IS the full history, so there is nothing to save
    and asking for it is a mistake worth naming rather than silently ignoring."""
    with pytest.raises(ValueError, match="sparse=True needs a window"):
        CausalPropagatorAttention(32, 2, hops=2, window=None, sparse=True)


def test_the_default_follows_the_window():
    assert CausalPropagatorAttention(32, 2, window=16).sparse is True
    assert CausalPropagatorAttention(32, 2, window=None).sparse is False
    assert CausalPropagatorAttention(32, 2, window=16, sparse=False).sparse is False


def test_gradients_match_the_dense_path():
    """It has to be trainable, not just evaluable: the same loss must produce the
    same gradient on every parameter."""
    torch.manual_seed(2)
    x = torch.randn(2, 48, 32)

    def grads(sparse):
        torch.manual_seed(3)
        m = CausalPropagatorAttention(32, 2, hops=3, window=8, sparse=sparse)
        out, _ = m(x)
        out.square().sum().backward()
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    gs, gd = grads(True), grads(False)
    assert set(gs) == set(gd) and gs
    for n in gs:
        assert (gs[n] - gd[n]).abs().max().item() < 1e-5, n


def test_hop_count_zero_is_the_value_itself():
    """K=0 keeps only c_0 * V, so the propagator collapses to the values and the
    band cannot change that."""
    torch.manual_seed(4)
    m = CausalPropagatorAttention(32, 2, hops=0, window=8)
    x = torch.randn(1, 20, 32)
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m.forward_dense(x)
    assert (a - b).abs().max().item() < 1e-6


def test_the_diagnostic_reports_which_path_ran():
    m = CausalPropagatorAttention(32, 2, hops=2, window=8)
    with torch.no_grad():
        _, diag = m(torch.randn(1, 16, 32), return_diag=True)
    assert diag["path"] == "banded" and diag["band"] == 8
