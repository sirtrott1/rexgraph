import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from rexgraph.nn.optim import GreensCochain


def _ring_task(n=60, C=3):
    # a homophilous ring of C blocks; labels piecewise-constant along the ring; a few train nodes
    y = np.repeat(np.arange(C), n // C)[:n]
    rows = np.arange(n); cols = (np.arange(n) + 1) % n
    A = sp.coo_matrix((np.ones(2 * n), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n, n))
    A = (A + sp.eye(n)).tocsr(); d = np.asarray(A.sum(1)).ravel(); dinv = 1 / np.sqrt(np.maximum(d, 1e-9))
    Ah = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocoo()
    adj = torch.sparse_coo_tensor(np.vstack([Ah.row, Ah.col]), torch.tensor(Ah.data, dtype=torch.float32), (n, n)).coalesce()
    block = n // C
    train = np.zeros(n, bool)
    for c in range(C):
        train[c * block + block // 2] = True   # one seed at each block's center (symmetric on the ring)
    return adj, torch.tensor(y), torch.tensor(train)


def _fit(opt_kind):
    torch.manual_seed(0)
    adj, y, train = _ring_task()
    n, C = adj.shape[0], int(y.max()) + 1
    Z = nn.Parameter(torch.zeros(n, C))
    if opt_kind == "greens":
        opt = GreensCochain([{"params": [Z], "green_adj": adj, "green_channel": "low", "green_lam": 4.0}], lr=0.5)
    else:
        opt = torch.optim.Adam([Z], lr=0.5)
    for _ in range(200):
        opt.zero_grad(); F.cross_entropy(Z[train], y[train]).backward(); opt.step()
    test = ~train
    return float((Z.argmax(1)[test] == y[test]).float().mean())


def test_greens_propagates_where_adam_cannot():
    adam_acc = _fit("adam")
    greens_acc = _fit("greens")
    # Adam only updates the few train-node params -> test nodes stay at chance; Greens propagates
    assert adam_acc < 0.5
    assert greens_acc > 0.75
    assert greens_acc > adam_acc + 0.3


def test_greens_falls_back_to_adam_without_adj():
    # a group with no green_adj must behave like Adam (a simple quadratic converges)
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(20))
    opt = GreensCochain([w], lr=0.1)
    for _ in range(300):
        opt.zero_grad(); (w * w).sum().backward(); opt.step()
    assert float((w * w).sum()) < 1e-2


def test_khop_channels_run_and_propagate():
    # the 2-hop / 3-hop channels are first-class and still propagate on a homophilous ring
    for ch in ("twohop", "threehop"):
        adj, y, train = _ring_task()
        n, C = adj.shape[0], int(y.max()) + 1
        Z = torch.nn.Parameter(torch.zeros(n, C))
        opt = GreensCochain([{"params": [Z], "green_adj": adj, "green_channel": ch, "green_lam": 4.0}], lr=0.5)
        for _ in range(200):
            opt.zero_grad(); F.cross_entropy(Z[train], y[train]).backward(); opt.step()
        acc = float((Z.argmax(1)[~train] == y[~train]).float().mean())
        assert acc > 0.5, f"{ch} failed to propagate (acc={acc})"


def test_generator_selects_the_best_scoring_channel():
    from rexgraph.nn.optim import generate_khop_channel
    # a mock context signal that prefers 2-hop; the generator must pick it
    scores = {"low": 0.4, "twohop": 0.8, "threehop": 0.5}
    best, got = generate_khop_channel(lambda ch: scores[ch], channels=("low", "twohop", "threehop"))
    assert best == "twohop"
    assert got == scores
