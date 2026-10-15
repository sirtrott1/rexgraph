"""Check default optimizer routing for ordinary and complex parameter groups.
"""
import pytest

pytest.importorskip("torch")

import numpy as np
import torch

import rexgraph.nn as R
from rexgraph.nn.optim import GreensCochain






def test_auto_routes_feature_space_model_to_adam():
    model = torch.nn.Linear(4, 2)
    opt, label = R.make_optimizer("auto", model, model.parameters(), lr=1e-3)
    assert type(opt) is torch.optim.Adam
    assert label == "Adam(auto)"


def test_auto_routes_cochain_to_greens():
    """A CoParticipationCochain exposes greens_groups(), so "auto" hands it GreensCochain."""
    from rexgraph.flow.cochain import CoParticipationCochain
    from rexgraph.graph import RexGraph
    rex = RexGraph(sources=np.array([0, 1, 2, 0], np.int32),
                   targets=np.array([1, 2, 3, 3], np.int32))
    model = CoParticipationCochain(rex, 2)
    opt, label = R.make_optimizer("auto", model, model.parameters(), lr=0.3)
    assert isinstance(opt, GreensCochain)
    assert label.startswith("GreensCochain(auto:")
