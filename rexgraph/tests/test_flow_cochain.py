"""The co-participation cochain classifier: the optimizer propagates the class through arity>2
branching hyperedges where a structure-blind optimizer (Adam) cannot. Synthetic hub-hypergraph, no
external data. Mirrors the real-data finding (a target bound by many ligands is a branching hyperedge;
co-participation through it carries the label) on a controlled complex."""
import numpy as np
import pytest

from rexgraph.graph import RexGraph

torch = pytest.importorskip("torch")

from rexgraph.flow.cochain import CoParticipationCochain, coparticipation_adjacency  # noqa: E402
from rexgraph.nn.factory import make_optimizer  # noqa: E402


def _hub_hypergraph(groups=8, per_group=15, n_classes=2):
    """`groups` hubs, each a target vertex bound by `per_group` ligands -> an arity-`per_group`
    branching hyperedge. Every edge in a group shares that hub, so co-participation groups them; a
    ligand is unique to its edge, so the ONLY shared structure is the hub. class = group % n_classes.
    Hub ids 0..groups-1, ligand ids groups.. (disjoint, so no vertex collision). Returns
    (rex, labels, per_group, groups)."""
    src, dst, labels = [], [], []
    lig = groups
    for g in range(groups):
        for _ in range(per_group):
            src.append(g)
            dst.append(lig); lig += 1
            labels.append(g % n_classes)
    rex = RexGraph(sources=np.array(src, dtype=np.int32), targets=np.array(dst, dtype=np.int32))
    return rex, np.array(labels), per_group, groups


def _masked_acc(pred, labels, is_m):
    return float((pred[is_m] == labels[is_m]).mean())


def test_coparticipation_adjacency_is_sparse_and_carries_hub_arity():
    rex, labels, arity, groups = _hub_hypergraph()
    adj = coparticipation_adjacency(rex)
    n_edges = len(labels)
    assert adj.shape == (n_edges, n_edges)
    assert adj.is_sparse
    # every edge co-participates with the other (arity-1) edges of its hub: a genuine arity>2 clique,
    # NOT a pairwise 2-neighbourhood. Count off-diagonal co-participants for edge 0.
    dense = adj.to_dense()
    off = (dense[0] != 0).sum().item() - 1  # minus the self-loop from renormalisation
    assert off == arity - 1 >= 2, f"edge 0 should see {arity - 1} hub co-participants, saw {off}"


def test_auto_routes_the_cochain_to_greens_but_a_plain_module_to_adam():
    rex, labels, _arity, _groups = _hub_hypergraph()
    model = CoParticipationCochain(rex, 2)
    _opt, label = make_optimizer("auto", model, model.parameters())
    assert label.startswith("GreensCochain"), label

    class _Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Z = torch.nn.Parameter(torch.zeros(len(labels), 2, dtype=torch.float64))

        def forward(self):
            return self.Z

    plain = _Plain()
    _opt2, label2 = make_optimizer("auto", plain, plain.parameters())
    assert label2 == "Adam(auto)", label2


def test_coparticipation_channel_propagates_where_adam_cannot():
    rex, labels, arity, _groups = _hub_hypergraph()
    assert arity >= 3  # a real branching hyperedge, not a pairwise edge
    rng = np.random.RandomState(0)
    is_m = rng.rand(len(labels)) < 0.3
    obs = ~is_m

    # Adam floor: a bare cochain trained with plain Adam never sends a gradient to a masked edge,
    # so its masked predictions stay at the zero-init argmax (chance).
    floor = CoParticipationCochain(rex, 2)
    opt = torch.optim.Adam(floor.parameters(), lr=0.3)
    lab_t = torch.as_tensor(labels, dtype=torch.long)
    obs_t = torch.as_tensor(obs, dtype=torch.bool)
    for _ in range(400):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(floor.Z[obs_t], lab_t[obs_t]).backward()
        opt.step()
    adam_acc = _masked_acc(floor.predict(), labels, is_m)
    assert adam_acc < 0.65, f"Adam should be near chance on masked edges, got {adam_acc}"

    # co-participation Green's channel via make_optimizer("auto"): the class is carried through the
    # shared hub to the masked edges.
    model = CoParticipationCochain(rex, 2).fit(labels, obs, epochs=400, lr=0.3)
    greens_acc = _masked_acc(model.predict(), labels, is_m)
    assert greens_acc > 0.9, f"co-participation should propagate the class, got {greens_acc}"


def test_restricting_connectors_to_the_non_hub_side_kills_propagation():
    # ablation: restrict co-participation to the LIGAND vertices (unique per edge) -> no shared cell,
    # so the channel cannot propagate and collapses to the Adam floor. Proves it is the branching HUB
    # doing the work, not co-participation generically.
    rex, labels, _arity, groups = _hub_hypergraph()
    n_vertices = abs(_scipy_incidence(rex)).shape[0]
    ligand_only = np.arange(n_vertices) >= groups  # hubs are ids 0..groups-1
    rng = np.random.RandomState(0)
    is_m = rng.rand(len(labels)) < 0.3
    obs = ~is_m
    model = CoParticipationCochain(rex, 2, restrict_vertices=ligand_only).fit(
        labels, obs, epochs=400, lr=0.3)
    acc = _masked_acc(model.predict(), labels, is_m)
    assert acc < 0.65, f"ligand-only (no shared hub) must not propagate, got {acc}"


def _scipy_incidence(rex):
    from rexgraph.core._sparse import to_scipy_csr
    return to_scipy_csr(rex._B1_dual)


def test_coparticipation_operator_survives_safetensors_roundtrip(tmp_path):
    # the complex that DEFINES co-participation must round-trip losslessly, or a reloaded model would
    # build a different operator. Assert the operator is bit-identical after the safetensors bridge.
    pytest.importorskip("safetensors")
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex

    rex, _labels, _arity, _groups = _hub_hypergraph()
    a0 = coparticipation_adjacency(rex).to_dense()
    p = tmp_path / "complex.safetensors"
    rex_to_safetensors(rex, str(p))
    a1 = coparticipation_adjacency(safetensors_to_rex(str(p))).to_dense()
    assert a0.shape == a1.shape
    assert torch.equal(a0, a1)


def test_trained_model_roundtrips_through_safetensors(tmp_path):
    # a TRAINED model (complex + cochain + knobs) persists in one safetensors file and reloads to
    # identical predictions. This is the full flow: canonical complex serializer + namespaced cochain.
    pytest.importorskip("safetensors")
    rex, labels, _arity, _groups = _hub_hypergraph()
    rng = np.random.RandomState(0)
    is_m = rng.rand(len(labels)) < 0.3
    obs = ~is_m
    model = CoParticipationCochain(rex, 2, green_lam=3.0, green_iters=15).fit(
        labels, obs, epochs=300, lr=0.3)
    pred0 = model.predict()

    p = tmp_path / "model.safetensors"
    model.save_safetensors(str(p))
    reloaded = CoParticipationCochain.load_safetensors(str(p))

    assert np.array_equal(reloaded.predict(), pred0)               # identical predictions
    assert torch.equal(reloaded.Z, model.Z)                        # identical cochain
    assert (reloaded.green_lam, reloaded.green_iters, reloaded.green_channel) == (3.0, 15, "low")
    # the rebuilt operator matches too (complex round-tripped, same restriction)
    assert torch.equal(reloaded._adj.to_dense(), model._adj.to_dense())


def test_restricted_model_roundtrips_through_safetensors(tmp_path):
    # a model with a connector restriction must restore the SAME restriction (else the operator drifts)
    pytest.importorskip("safetensors")
    rex, labels, _arity, groups = _hub_hypergraph()
    n_vertices = abs(_scipy_incidence(rex)).shape[0]
    hub_only = np.arange(n_vertices) < groups
    model = CoParticipationCochain(rex, 2, restrict_vertices=hub_only)
    p = tmp_path / "restricted.safetensors"
    model.save_safetensors(str(p))
    reloaded = CoParticipationCochain.load_safetensors(str(p))
    assert torch.equal(reloaded._adj.to_dense(), model._adj.to_dense())
