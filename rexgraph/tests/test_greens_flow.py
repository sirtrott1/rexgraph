"""GreensFlow: the same preconditioning over an operator that reads both grades.

`coparticipation_adjacency` is |B1|^T|B1| and never touches B2, so a model trained
through it is blind to every face in the complex. That is measurable rather than
arguable: the operator is bit-identical on an open complex and the same complex closed,
so an ablation over the two reports the same number for a reason that has nothing to do
with the data.

`flow_adjacency` is L1_down + alpha * L1_up, so the gradient tier and the curl tier both
carry signal, and alpha defaults to c0_squared, the exact rational coupling.

This is additive. GreensCochain is unchanged and stays right for a cochain over a
face-free complex, where there is no curl tier to miss.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.faces import autoface
from rexgraph.graph import RexGraph

torch = pytest.importorskip("torch")

from rexgraph.nn import GreensCochain, GreensFlow  # noqa: E402

_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _tetra(filled: bool):
    rex = RexGraph(sources=np.array([e[0] for e in _EDGES], dtype=np.int32),
                   targets=np.array([e[1] for e in _EDGES], dtype=np.int32))
    if filled:
        autoface(rex)
    rex._ensure_clean()
    return rex


@pytest.fixture
def closed():
    return _tetra(True)


@pytest.fixture
def open_complex():
    return _tetra(False)


#### the operator sees what the old one could not


def test_coparticipation_is_blind_to_faces(closed, open_complex):
    """The premise. Not a claim about the learner, a claim about the operator."""
    from rexgraph.flow import coparticipation_adjacency

    assert closed.nF_hodge > 0 and open_complex.nF_hodge == 0
    assert torch.allclose(coparticipation_adjacency(closed).to_dense(),
                          coparticipation_adjacency(open_complex).to_dense())


def test_the_flow_operator_is_not(closed, open_complex):
    assert not torch.allclose(GreensFlow.build_adjacency(closed).to_dense(),
                              GreensFlow.build_adjacency(open_complex).to_dense())


def test_the_curl_tier_changes_the_preconditioned_gradient(closed, open_complex):
    """Where it has to show: the same gradient, whitened through the two complexes."""
    g = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    out = {}
    for tag, rex in (("closed", closed), ("open", open_complex)):
        z = torch.zeros(6, 2, dtype=torch.float64, requires_grad=True)
        opt = GreensFlow([{"params": [z]}], rex=rex, lr=0.3)
        group = opt.param_groups[0]
        op, low = GreensFlow._channel_op(group["green_adj"], "low", group)
        out[tag] = GreensFlow._greens(op, g, 1.0, low, 12)
    assert not torch.allclose(out["closed"], out["open"])


def test_the_trajectories_separate(closed, open_complex):
    """One Adam step is sign-only, so a single step cannot show this and does not."""
    ends = []
    target = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    for rex in (closed, open_complex):
        z = torch.zeros(6, 2, dtype=torch.float64, requires_grad=True)
        opt = GreensFlow([{"params": [z]}], rex=rex, lr=0.1)
        for _ in range(20):
            opt.zero_grad()
            ((z - target) ** 2).sum().backward()
            opt.step()
        ends.append(z.detach().clone())
    assert not torch.allclose(*ends)


#### the exchange rate


def test_alpha_defaults_to_the_exact_coupling(closed):
    """c0_squared is the rational geometry<->topology coupling, not a number picked for
    the run."""
    assert Fraction(str(closed.c0_squared)) == Fraction(3, 4)
    default = GreensFlow.build_adjacency(closed).to_dense()
    explicit = GreensFlow.build_adjacency(closed, alpha=float(closed.c0_squared)).to_dense()
    assert torch.allclose(default, explicit)


def test_a_different_alpha_gives_a_different_operator(closed):
    assert not torch.allclose(GreensFlow.build_adjacency(closed, alpha=0.1).to_dense(),
                              GreensFlow.build_adjacency(closed, alpha=5.0).to_dense())


def test_it_says_when_there_is_no_curl_tier(closed, open_complex):
    """On a face-free complex this degrades to GreensCochain rather than pretending to a
    tier that is not there."""
    z = torch.zeros(6, 2, dtype=torch.float64, requires_grad=True)
    assert GreensFlow([{"params": [z]}], rex=closed).reads_faces is True
    assert GreensFlow([{"params": [z]}], rex=open_complex).reads_faces is False


#### it is additive


def test_greens_cochain_is_untouched():
    """No rex, no operator: the existing optimizer behaves exactly as before."""
    z = torch.zeros(6, 2, dtype=torch.float64, requires_grad=True)
    assert GreensCochain([{"params": [z]}]).param_groups[0]["green_adj"] is None


def test_flow_without_a_complex_is_plain_adam():
    z = torch.zeros(6, 2, dtype=torch.float64, requires_grad=True)
    assert GreensFlow([{"params": [z]}]).param_groups[0]["green_adj"] is None


def test_an_explicit_operator_is_not_overridden(closed):
    z = torch.zeros(6, 2, dtype=torch.float64, requires_grad=True)
    mine = GreensFlow.build_adjacency(closed, alpha=2.0)
    opt = GreensFlow([{"params": [z], "green_adj": mine}], rex=closed)
    assert opt.param_groups[0]["green_adj"] is mine


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_operator_follows_the_parameter_dtype(closed, dtype):
    """The operator is assembled in float64 because it comes off exact rational
    structure; the parameters need not be, and a mismatch is a dtype error at the first
    CG matvec rather than anything meaningful."""
    z = torch.zeros(closed.nE, 2, dtype=dtype, requires_grad=True)
    opt = GreensFlow([{"params": [z]}], rex=closed, lr=0.3)
    (z - 1).pow(2).sum().backward()
    opt.step()
    assert opt.param_groups[0]["green_adj"].dtype == dtype
    assert float(z.abs().sum()) > 0
