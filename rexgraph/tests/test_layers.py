"""Composable relational NN layers (rexgraph.nn.layers): they compose into a trainable net and the
GreenResolvent block differentiates through the self-adjoint solve."""
import numpy as np
import pytest

pytest.importorskip("torch")
import torch                                                # noqa: E402

import rexgraph.nn as R                                     # noqa: E402
from rexgraph.graph import RexGraph                         # noqa: E402


def _L():
    rex = RexGraph.from_simplicial(np.array([0, 0, 0, 1, 1, 2], np.int32),
                                   np.array([1, 2, 3, 2, 3, 3], np.int32),
                                   np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32))
    return torch.tensor(np.asarray(rex.L0, dtype=np.float64), dtype=torch.float32), rex.nV


class _Net(torch.nn.Module):
    def __init__(self, d=8, n_cls=3):
        super().__init__()
        self.emb = torch.nn.Linear(4, d)
        self.b1 = R.RelationalBlock(d, op="heat")           # matrix-free propagator block
        self.b2 = R.RelationalBlock(d, op="green")          # implicit self-adjoint block
        self.head = torch.nn.Linear(d, n_cls)

    def forward(self, X, L):
        return self.head(self.b2(self.b1(self.emb(X), L), L))


def test_blocks_compose_and_train_with_hodgeadam():
    L, nV = _L()
    net = _Net()
    X = torch.randn(nV, 4)
    y = torch.randint(0, 3, (nV,))
    opt = R.HodgeAdam(net.parameters(), lr=0.05)
    first = last = None
    for i in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(X, L), y)
        loss.backward()
        opt.step()
        first = loss.item() if i == 0 else first
        last = loss.item()
    assert last < first                                     # the composed net learns
    assert net.b1.mix.log_t.grad is not None                # learnable propagator scale optimized
    assert net.b2.mix.log_alpha.grad is not None            # learnable resolvent alpha optimized


def test_green_resolvent_self_adjoint_backward():
    """The implicit block's backward is the same CG solve; gradient reaches X and alpha."""
    L, nV = _L()
    layer = R.GreenResolvent(4, 4)
    X = torch.randn(nV, 4, requires_grad=True)
    layer(X, L).sum().backward()
    assert X.grad is not None and layer.log_alpha.grad is not None
