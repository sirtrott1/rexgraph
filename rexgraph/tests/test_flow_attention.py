import os
import tempfile

import numpy as np
import pytest
from rexgraph.graph import RexGraph
from rexgraph.flow.attention import (
    CoParticipationAttention,
    coparticipation_neighbors,
    coparticipation_attention,
)

# A real-data check, pointed at whatever the operator supplies. The path used to
# be a literal naming a specific private dataset, which put a dataset -- and its
# location on one machine -- into a tree that is meant to be dataset-agnostic.
# Unset, the test skips, which is what it already did when the file was absent.
_REAL_DATA_ENV = "REXGRAPH_TEST_BINDING_TSV"
_REAL_DATA_PATH = os.path.expanduser(os.environ.get(_REAL_DATA_ENV, ""))


def _submode_task(seed=1, nT=80, E=5000):
    rng = np.random.RandomState(seed)
    t = rng.randint(0, nT, E)
    lig = rng.randint(0, 3000, E)
    submode = rng.randint(0, 2, E)
    sm_aff = rng.randn(nT, 2) * 2.0
    y = sm_aff[t, submode] + rng.randn(E) * 0.2
    f = (submode + rng.randn(E) * 0.25).reshape(-1, 1)          # the inside feature reveals the sub-mode
    rex = RexGraph(sources=t.astype(np.int32), targets=(lig + nT).astype(np.int32))
    return rex, y, f, submode


def _r2(pred, y, mask):
    base = float(np.mean((y[mask] - y[mask].mean()) ** 2))
    return 1.0 - float(np.mean((pred[mask] - y[mask]) ** 2)) / base


def test_attention_beats_uniform_on_submode_task():
    rex, y, f, submode = _submode_task()
    rng = np.random.RandomState(0); is_m = rng.rand(len(y)) < 0.2; obs = ~is_m
    ptr, idx = coparticipation_neighbors(rex)
    uni = coparticipation_attention(ptr, idx, f, y, obs, gamma=0.0)              # uniform settle
    att = coparticipation_attention(ptr, idx, f, y, obs, proj=np.eye(1), gamma=4.0)  # inside-compat attention
    assert _r2(uni, y, is_m) < 0.6                       # uniform blurs the sub-modes
    assert _r2(att, y, is_m) >= 0.85                     # attention resolves them (grounded 0.92)


def test_fit_learns_the_compatibility():
    rex, y, f, submode = _submode_task()
    rng = np.random.RandomState(0); is_m = rng.rand(len(y)) < 0.2; obs = ~is_m
    m = CoParticipationAttention(inside_dim=f.shape[1])
    m.fit_self_supervised(rex, f, np.where(obs, y, 0.0), mask_frac=0.2, seed=2)
    pred = m.predict(rex, f, y, obs)
    assert _r2(pred, y, is_m) >= 0.80          # the fit (self-supervised) recovers most of the attention gain


def _load_binding_subcomplex(n_rows=6000):
    """Real affinity-table subcomplex: read the first `n_rows` data rows to a temp TSV (fast), build
    the edge-primal complex (ID2=target as source, ID1=ligand as destination), and the pKd target.
    Returns None if the preserved data file is absent (so CI without the file still passes)."""
    if not os.path.exists(_REAL_DATA_PATH):
        return None
    from agent.warehouse import source as S

    with open(_REAL_DATA_PATH) as f_in:
        header = f_in.readline()
        rows = []
        for i, line in enumerate(f_in):
            rows.append(line)
            if i + 1 >= n_rows:
                break

    fd, tmp_path = tempfile.mkstemp(suffix=".tsv")
    try:
        with os.fdopen(fd, "w") as f_out:
            f_out.write(header)
            f_out.writelines(rows)
        ed = S.load_edges(tmp_path, source="ID2", target="ID1", weight="Y", usecols=["ID1", "ID2", "Y"])
    finally:
        os.remove(tmp_path)

    rex = S.edge_complex(ed)
    y = -np.log10(np.clip(ed.weight, 1e-3, None) * 1e-9)  # pKd
    inside = np.concatenate(
        [np.asarray(rex.structural_character, dtype=np.float64),
         np.asarray(rex.rcfe_curvature, dtype=np.float64).reshape(-1, 1)],
        axis=1,
    )
    return rex, y, inside


def test_attention_path_is_matrix_free(monkeypatch):
    import numpy.linalg as nla
    import rexgraph.core._linalg as _linalg
    import rexgraph.core._sparse as _sparse
    import scipy.sparse.linalg as ssla
    calls = []
    def spy(mod, name):
        if hasattr(mod, name):
            orig = getattr(mod, name)
            monkeypatch.setattr(mod, name, lambda *a, _n=name, _o=orig, **k: (calls.append(_n), _o(*a, **k))[1])
    for n in ("lstsq",): spy(_linalg, n)
    for n in ("spmm_AAt_dense_f64", "spmm_AtA_dense_f64"): spy(_sparse, n)
    for n in ("eigsh", "svds"): spy(ssla, n)
    for n in ("eig", "eigh", "eigvals", "eigvalsh", "svd", "lstsq", "pinv"): spy(nla, n)
    # build the sub-mode task, run neighbors + attention + the fit, all on the attention path
    rex, y, f, submode = _submode_task()
    obs = np.ones(len(y), dtype=bool)
    ptr, idx = coparticipation_neighbors(rex)
    coparticipation_attention(ptr, idx, f, y, obs, gamma=1.0)
    m = CoParticipationAttention(inside_dim=f.shape[1])
    m.fit_self_supervised(rex, f, y, mask_frac=0.2, seed=0)
    m.predict(rex, f, y, obs)
    assert calls == [], f"attention path used a dense/eigensolver: {calls}"


def test_binding_subcomplex_honest_ceiling():
    loaded = _load_binding_subcomplex()
    if loaded is None:
        pytest.skip(f"set {_REAL_DATA_ENV} to a tab-separated affinity table "
                    f"to run the real-data check")
    rex, y, inside = loaded

    rng = np.random.RandomState(0)
    is_m = rng.rand(len(y)) < 0.2
    obs = ~is_m
    all_obs = np.ones(len(y), dtype=bool)

    ptr, idx = coparticipation_neighbors(rex)

    # face 1: unsupervised - the zero-parameter uniform co-participation settle (gamma=0)
    uniform = coparticipation_attention(ptr, idx, inside, y, obs, gamma=0.0)

    # face 2: self-supervised - fit (proj, gamma) from the observed 80% ALONE, predict the held-out 20%
    m_self = CoParticipationAttention(inside_dim=inside.shape[1])
    m_self.fit_self_supervised(rex, inside, np.where(obs, y, 0.0), obs_mask=obs, mask_frac=0.2, seed=2)
    self_supervised = m_self.predict((ptr, idx), inside, y, obs)

    # face 3 (supervised): fit with the TRUE pKd visible on every edge (the ceiling if labels were free)
    m_sup = CoParticipationAttention(inside_dim=inside.shape[1])
    m_sup.fit_self_supervised(rex, inside, y, obs_mask=all_obs, mask_frac=0.2, seed=3)
    supervised = m_sup.predict((ptr, idx), inside, y, all_obs)

    r2_uniform = _r2(uniform, y, is_m)
    r2_self_supervised = _r2(self_supervised, y, is_m)
    r2_supervised = _r2(supervised, y, is_m)

    print(
        "binding subcomplex (nE=%d, nV=%d) three faces: "
        "unsupervised(uniform)=%.4f  self-supervised(fitted, masked)=%.4f  "
        "supervised(fit-on-all)=%.4f"
        % (rex.nE, rex.nV, r2_uniform, r2_self_supervised, r2_supervised)
    )

    # co-participation structure alone predicts binding strength well above chance
    assert r2_uniform > 0.4
    # honest ceiling: the learned compatibility must not HURT relative to the uniform settle;
    # binding is structure-dominated and thin (one Kd scalar per relation), so a large gain over
    # uniform is NOT expected here and must not be asserted; only that attention does no harm
    assert r2_self_supervised >= r2_uniform - 0.02
