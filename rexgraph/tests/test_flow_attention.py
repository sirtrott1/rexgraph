import numpy as np

from rexgraph.flow.attention import (
    CoParticipationAttention,
    coparticipation_attention,
    coparticipation_neighbors,
)
from rexgraph.graph import RexGraph



def _submode_task(seed=1, nT=80, E=5000):
    rng = np.random.RandomState(seed)
    t = rng.randint(0, nT, E)
    lig = rng.randint(0, 3000, E)
    submode = rng.randint(0, 2, E)
    sm_aff = rng.randn(nT, 2) * 2.0
    y = sm_aff[t, submode] + rng.randn(E) * 0.2
    f = (submode + rng.randn(E) * 0.25).reshape(-1, 1)          # the inside feature reveals the sub mode
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
    att = coparticipation_attention(ptr, idx, f, y, obs, proj=np.eye(1), gamma=4.0)  # inside compat attention
    assert _r2(uni, y, is_m) < 0.6                       # uniform blurs the sub modes
    assert _r2(att, y, is_m) >= 0.85                     # attention separates the generated submodes


def test_fit_learns_the_compatibility():
    rex, y, f, submode = _submode_task()
    rng = np.random.RandomState(0); is_m = rng.rand(len(y)) < 0.2; obs = ~is_m
    m = CoParticipationAttention(inside_dim=f.shape[1])
    m.fit_self_supervised(rex, f, np.where(obs, y, 0.0), obs_mask=obs, mask_frac=0.2, seed=2)
    pred = m.predict(rex, f, y, obs)
    assert _r2(pred, y, is_m) >= 0.80          # the fit (self supervised) recovers most of the attention gain




def test_attention_path_is_matrix_free(monkeypatch):
    import numpy.linalg as nla
    import scipy.sparse.linalg as ssla

    import rexgraph.core._linalg as _linalg
    import rexgraph.core._sparse as _sparse
    calls = []
    def spy(mod, name):
        if hasattr(mod, name):
            orig = getattr(mod, name)
            monkeypatch.setattr(mod, name, lambda *a, _n=name, _o=orig, **k: (calls.append(_n), _o(*a, **k))[1])
    for n in ("lstsq",): spy(_linalg, n)
    for n in ("spmm_AAt_dense_f64", "spmm_AtA_dense_f64"): spy(_sparse, n)
    for n in ("eigsh", "svds"): spy(ssla, n)
    for n in ("eig", "eigh", "eigvals", "eigvalsh", "svd", "lstsq", "pinv"): spy(nla, n)
    # build the sub mode task, run neighbors + attention + the fit, all on the attention path
    rex, y, f, submode = _submode_task()
    obs = np.ones(len(y), dtype=bool)
    ptr, idx = coparticipation_neighbors(rex)
    coparticipation_attention(ptr, idx, f, y, obs, gamma=1.0)
    m = CoParticipationAttention(inside_dim=f.shape[1])
    m.fit_self_supervised(rex, f, y, obs_mask=obs, mask_frac=0.2, seed=0)
    m.predict(rex, f, y, obs)
    assert calls == [], f"attention path used a dense/eigensolver: {calls}"
