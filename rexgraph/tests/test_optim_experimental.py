"""HodgeAdam / HodgeSGD live in rexgraph.nn._experimental, not in rexgraph.nn.optim.

They tie plain Adam on standard feature-space models, so they are kept off the
rexgraph.nn top-level surface and out of the live optimizer module. optim.py re-exports
them so callers naming them directly keep resolving.
"""
import pytest

import rexgraph.nn as R
from rexgraph.nn import _experimental, optim


def test_defined_in_experimental_not_optim():
    assert _experimental.HodgeAdam.__module__ == "rexgraph.nn._experimental"
    assert _experimental.HodgeSGD.__module__ == "rexgraph.nn._experimental"


def test_optim_reexport_is_the_same_object():
    from rexgraph.nn.optim import HodgeAdam, HodgeSGD
    assert HodgeAdam is _experimental.HodgeAdam
    assert HodgeSGD is _experimental.HodgeSGD
    assert optim.HodgeAdam is _experimental.HodgeAdam        # attribute access, the factory's path
    assert optim.HodgeSGD is _experimental.HodgeSGD
    assert R.optim.HodgeAdam is _experimental.HodgeAdam       # aliased-module access


def test_absent_from_nn_surface():
    assert "HodgeAdam" not in R.__all__ and "HodgeSGD" not in R.__all__
    assert not hasattr(R, "HodgeAdam") and not hasattr(R, "HodgeSGD")


def test_named_construction_still_works():
    pytest.importorskip("torch")
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Linear(4, 3))
    opt = optim.build_optimizer(model.parameters(), method="hodge")
    assert isinstance(opt, _experimental.HodgeAdam)
    opt = optim.build_optimizer(model.parameters(), method="hodgesgd")
    assert isinstance(opt, _experimental.HodgeSGD)

    arch, label = R.make_optimizer("hodge-arch", model, list(model.parameters()))
    assert isinstance(arch, _experimental.HodgeAdam)
    assert label.startswith("HodgeAdam(arch:")
