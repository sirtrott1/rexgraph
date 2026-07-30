"""
Track-1 stage-1: the edge-primary relational complex net vs a matched pairwise GNN, on a task
where HIGHER-ORDER structure is decisive — substructure (triangle / 4-cycle) counting, the
canonical failure mode of pairwise message passing (1-WL can't count them).

The premise being tested: carrying V/E/F cochains and flowing through the boundary operators
B₁/B₂ (so FACES — bounded, relevance-selected triangles — are first-class) lets the model count
what a pairwise model structurally cannot. The clean attribution is the faces-ablation:
edge-primary WITH faces vs the SAME net with faces off (≈ pairwise reach).

Everything sparse scatter/gather on B₁/B₂ (O(nnz)), differentiable, trained by vector HodgeAdam.
Run:  python -m agent.benchmarks.bench_relational_model [--quick]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rexgraph.nn.optim import HodgeAdam, pick_device


# ─────────────────────────── data: random graphs + substructure counts ───────────────────────────

def gen_graph(nV, p, rng):
    """Erdős–Rényi graph → (src, tgt) undirected edge endpoints (i<j), oriented i→j."""
    iu = np.triu_indices(nV, k=1)
    mask = rng.random(iu[0].shape[0]) < p
    src = iu[0][mask].astype(np.int64); tgt = iu[1][mask].astype(np.int64)
    return src, tgt


def gen_graph_regular(nV, deg, rng):
    """Random d-REGULAR graph (constant degree) → substructure count is DECORRELATED from
    degree/density, so a pairwise GNN (all nodes look identical locally) has nothing to exploit
    and must genuinely count — the clean test of the higher-order advantage."""
    import networkx as nx
    seed = int(rng.integers(0, 2 ** 31 - 1))
    G = nx.random_regular_graph(deg, nV, seed=seed)
    e = np.array(list(G.edges()), dtype=np.int64)
    lo = np.minimum(e[:, 0], e[:, 1]); hi = np.maximum(e[:, 0], e[:, 1])
    return lo, hi


def find_triangles(src, tgt, nV):
    """Relevance-gated faces = the actual triangles (bounded, NOT combinatorial enumeration).
    Returns face_edges [nF,3] (edge indices) and face_signs [nF,3] (∂ orientation +,+,−)."""
    eidx = {(int(s), int(t)): e for e, (s, t) in enumerate(zip(src, tgt))}
    adj = {v: set() for v in range(nV)}
    for s, t in zip(src, tgt):
        adj[int(s)].add(int(t)); adj[int(t)].add(int(s))
    faces = []
    for e, (i, j) in enumerate(zip(src, tgt)):
        i, j = int(i), int(j)
        for k in adj[i] & adj[j]:                       # common neighbour ⇒ triangle
            a, b, c = sorted((i, j, k))
            if k > j or (k < i):                        # dedup: count each triangle once
                pass
            faces.append((a, b, c))
    faces = sorted(set(faces))
    fe, fs = [], []
    for (a, b, c) in faces:                             # boundary a-b + b-c − a-c
        fe.append([eidx[(a, b)], eidx[(b, c)], eidx[(a, c)]]); fs.append([1.0, 1.0, -1.0])
    return (np.array(fe, dtype=np.int64).reshape(-1, 3),
            np.array(fs, dtype=np.float32).reshape(-1, 3), len(faces))


def count_4cycles(src, tgt, nV):
    """# of 4-cycles (squares) = ½ Σ_{i<j} C(common_nbrs(i,j), 2)."""
    A = np.zeros((nV, nV))
    A[src, tgt] = 1; A[tgt, src] = 1
    co = A @ A                                          # common-neighbour counts
    tri = np.triu(co, 1)
    return float((tri * (tri - 1) / 2).sum() - _triangle_total(A) * 0) / 2.0


def _triangle_total(A):
    return float(np.trace(A @ A @ A) / 6.0)


def make_batch(bs, rng, device, nmin=8, nmax=14, target="triangles", graphs="er"):
    """Block-diagonal batch of bs graphs; target = per-graph substructure count (normalized).
    graphs="er" (Erdős–Rényi, count correlates with density) or "regular" (fixed-size d-regular,
    count decorrelated from degree — the clean higher-order test)."""
    S, T, VB, EB = [], [], [], []
    FE, FS, FB = [], [], []
    voff = eoff = 0
    y = []
    for g in range(bs):
        if graphs == "regular":
            nV = 16; src, tgt = gen_graph_regular(nV, 4, rng)      # fixed n, fixed degree
        else:
            nV = int(rng.integers(nmin, nmax + 1))
            src, tgt = gen_graph(nV, float(rng.uniform(0.25, 0.55)), rng)
        if src.shape[0] < 3:
            src = np.array([0, 1, 0]); tgt = np.array([1, 2, 2]); nV = max(nV, 3)
        fe, fs, nf = find_triangles(src, tgt, nV)
        tgt_val = nf if target == "triangles" else count_4cycles(src, tgt, nV)
        S.append(src + voff); T.append(tgt + voff)
        VB.append(np.full(nV, g)); EB.append(np.full(src.shape[0], g))
        if nf > 0:
            FE.append(fe + eoff); FS.append(fs); FB.append(np.full(nf, g))
        y.append(tgt_val)
        voff += nV; eoff += src.shape[0]
    src = torch.tensor(np.concatenate(S), device=device)
    tgt = torch.tensor(np.concatenate(T), device=device)
    vb = torch.tensor(np.concatenate(VB), device=device)
    eb = torch.tensor(np.concatenate(EB), device=device)
    if FE:
        fe = torch.tensor(np.concatenate(FE), device=device)
        fs = torch.tensor(np.concatenate(FS), device=device)
        fb = torch.tensor(np.concatenate(FB), device=device)
    else:
        fe = torch.zeros(0, 3, dtype=torch.long, device=device)
        fs = torch.zeros(0, 3, device=device); fb = torch.zeros(0, dtype=torch.long, device=device)
    y = torch.tensor(np.array(y, dtype=np.float32), device=device)
    y = torch.log1p(y)                                  # compress the count range
    return dict(src=src, tgt=tgt, vb=vb, eb=eb, fe=fe, fs=fs, fb=fb, nV=int(voff),
                nE=int(eoff), nF=int(fe.shape[0]), bs=bs), y


# ─────────────────────────── sparse boundary operators (scatter/gather) ───────────────────────────

def seg_mean(x, idx, n):
    out = torch.zeros(n, x.shape[-1], device=x.device, dtype=x.dtype)
    cnt = torch.zeros(n, 1, device=x.device, dtype=x.dtype)
    out.index_add_(0, idx, x); cnt.index_add_(0, idx, torch.ones_like(x[:, :1]))
    return out / cnt.clamp_min(1.0)

def B1_E2V(xE, b):                     # B₁ x_E : edge→vertex (signed)
    out = torch.zeros(b["nV"], xE.shape[-1], device=xE.device, dtype=xE.dtype)
    out.index_add_(0, b["tgt"], xE); out.index_add_(0, b["src"], -xE); return out
def B1T_V2E(xV, b):                    # B₁ᵀ x_V : vertex→edge (x_tgt − x_src)
    return xV[b["tgt"]] - xV[b["src"]]
def absB1_E2V(xE, b):
    out = torch.zeros(b["nV"], xE.shape[-1], device=xE.device, dtype=xE.dtype)
    out.index_add_(0, b["tgt"], xE); out.index_add_(0, b["src"], xE); return out
def absB1T_V2E(xV, b):
    return xV[b["tgt"]] + xV[b["src"]]
def B2_F2E(xF, b):                     # B₂ x_F : face→edge
    out = torch.zeros(b["nE"], xF.shape[-1], device=xF.device, dtype=xF.dtype)
    if b["nF"] == 0:
        return out
    vals = (b["fs"].unsqueeze(-1) * xF.unsqueeze(1)).reshape(-1, xF.shape[-1])
    out.index_add_(0, b["fe"].reshape(-1), vals); return out
def B2T_E2F(xE, b):                    # B₂ᵀ x_E : edge→face
    if b["nF"] == 0:
        return torch.zeros(0, xE.shape[-1], device=xE.device, dtype=xE.dtype)
    return (b["fs"].unsqueeze(-1) * xE[b["fe"]]).sum(1)


# ─────────────────────────── models ───────────────────────────

class EdgeLayer(nn.Module):
    def __init__(self, d, use_faces):
        super().__init__()
        self.uf = use_faces
        self.e = nn.ModuleDict({k: nn.Linear(d, d) for k in
                                (["self", "dn", "T", "G"] + (["up", "curl"] if use_faces else []))})
        self.v = nn.ModuleDict({k: nn.Linear(d, d) for k in ["self", "up", "L0"]})
        if use_faces:
            self.f = nn.ModuleDict({k: nn.Linear(d, d) for k in ["self", "dn"]})
        self.lnE = nn.LayerNorm(d); self.lnV = nn.LayerNorm(d)
        self.lnF = nn.LayerNorm(d) if use_faces else None

    def forward(self, XV, XE, XF, b):
        e = self.e["self"](XE) + self.e["dn"](B1T_V2E(XV, b)) \
            + self.e["T"](B1T_V2E(B1_E2V(XE, b), b)) + self.e["G"](absB1T_V2E(absB1_E2V(XE, b), b))
        if self.uf:
            e = e + self.e["up"](B2_F2E(XF, b)) + self.e["curl"](B2_F2E(B2T_E2F(XE, b), b))
        XE = XE + F.gelu(self.lnE(e))
        v = self.v["self"](XV) + self.v["up"](B1_E2V(XE, b)) + self.v["L0"](B1_E2V(B1T_V2E(XV, b), b))
        XV = XV + F.gelu(self.lnV(v))
        if self.uf and b["nF"] > 0:
            fnew = self.f["self"](XF) + self.f["dn"](B2T_E2F(XE, b))
            XF = XF + F.gelu(self.lnF(fnew))
        return XV, XE, XF


class ComplexNet(nn.Module):
    def __init__(self, d=48, n_layers=3, use_faces=True):
        super().__init__()
        self.uf = use_faces; self.d = d
        self.v0 = nn.Linear(1, d); self.e0 = nn.Linear(1, d); self.f0 = nn.Linear(1, d)
        self.layers = nn.ModuleList([EdgeLayer(d, use_faces) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear((3 if use_faces else 2) * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, b):
        XV = self.v0(torch.ones(b["nV"], 1, device=b["src"].device))
        XE = self.e0(torch.ones(b["nE"], 1, device=b["src"].device))
        XF = self.f0(torch.ones(max(b["nF"], 1), 1, device=b["src"].device))[:b["nF"]] if self.uf \
            else torch.zeros(0, self.d, device=b["src"].device)
        for l in self.layers:
            XV, XE, XF = l(XV, XE, XF, b)
        gV = seg_mean(XV, b["vb"], b["bs"]); gE = seg_mean(XE, b["eb"], b["bs"])
        feats = [gV, gE]
        if self.uf:
            gF = seg_mean(XF, b["fb"], b["bs"]) if b["nF"] > 0 else torch.zeros(b["bs"], self.d, device=gV.device)
            feats.append(gF)
        return self.head(torch.cat(feats, -1)).squeeze(-1)


class PairwiseGNN(nn.Module):
    """Matched pairwise message-passing baseline (vertices only, no faces)."""
    def __init__(self, d=48, n_layers=3):
        super().__init__()
        self.d = d; self.v0 = nn.Linear(1, d)
        self.msg = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(n_layers)])
        self.upd = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(n_layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, b):
        XV = self.v0(torch.ones(b["nV"], 1, device=b["src"].device))
        for msg, upd, ln in zip(self.msg, self.upd, self.ln):
            m = F.gelu(msg(torch.cat([XV[b["src"]], XV[b["tgt"]]], -1)))
            agg = torch.zeros_like(XV)
            agg.index_add_(0, b["tgt"], m); agg.index_add_(0, b["src"], m)
            XV = XV + F.gelu(ln(upd(torch.cat([XV, agg], -1))))
        return self.head(seg_mean(XV, b["vb"], b["bs"])).squeeze(-1)


# ─────────────────────────── stage 2: Green's-function (implicit) layers ───────────────────────────

def edge_L1_matvec(b):
    """Matrix-free edge Hodge Laplacian L₁ = B₁ᵀB₁ + B₂B₂ᵀ (down + up), for the batch."""
    def mv(xE):
        return B1T_V2E(B1_E2V(xE, b), b) + B2_F2E(B2T_E2F(xE, b), b)
    return mv


class GreenLayer(nn.Module):
    """Edge-primary layer whose edge diffusion is the SOLVED equilibrium (I+αL₁)⁻¹ — one
    implicit layer captures all propagation depth (self-adjoint forward=backward, dynamic CG)."""
    def __init__(self, d, use_faces=True):
        super().__init__()
        self.uf = use_faces
        keys = ["self", "dn", "green"] + (["up"] if use_faces else [])
        self.e = nn.ModuleDict({k: nn.Linear(d, d) for k in keys})
        self.v = nn.ModuleDict({k: nn.Linear(d, d) for k in ["self", "up"]})
        if use_faces:
            self.f = nn.ModuleDict({k: nn.Linear(d, d) for k in ["self", "dn"]})
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.lnE = nn.LayerNorm(d); self.lnV = nn.LayerNorm(d)
        self.lnF = nn.LayerNorm(d) if use_faces else None

    def forward(self, XV, XE, XF, b):
        from rexgraph.nn.rcf_torch import green_resolvent
        green = green_resolvent(XE, self.alpha.abs() + 1e-3, edge_L1_matvec(b), tol=1e-4, max_iter=30)
        e = self.e["self"](XE) + self.e["dn"](B1T_V2E(XV, b)) + self.e["green"](green)
        if self.uf:
            e = e + self.e["up"](B2_F2E(XF, b))
        XE = XE + F.gelu(self.lnE(e))
        XV = XV + F.gelu(self.lnV(self.v["self"](XV) + self.v["up"](B1_E2V(XE, b))))
        if self.uf and b["nF"] > 0:
            XF = XF + F.gelu(self.lnF(self.f["self"](XF) + self.f["dn"](B2T_E2F(XE, b))))
        return XV, XE, XF


class GreenComplexNet(nn.Module):
    def __init__(self, d=48, n_layers=1, use_faces=True):
        super().__init__()
        self.uf = use_faces; self.d = d
        self.v0 = nn.Linear(1, d); self.e0 = nn.Linear(1, d); self.f0 = nn.Linear(1, d)
        self.layers = nn.ModuleList([GreenLayer(d, use_faces) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear((3 if use_faces else 2) * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, b):
        dev = b["src"].device
        XV = self.v0(torch.ones(b["nV"], 1, device=dev))
        XE = self.e0(torch.ones(b["nE"], 1, device=dev))
        XF = self.f0(torch.ones(max(b["nF"], 1), 1, device=dev))[:b["nF"]] if self.uf \
            else torch.zeros(0, self.d, device=dev)
        for l in self.layers:
            XV, XE, XF = l(XV, XE, XF, b)
        gV = seg_mean(XV, b["vb"], b["bs"]); gE = seg_mean(XE, b["eb"], b["bs"])
        feats = [gV, gE]
        if self.uf:
            gF = seg_mean(XF, b["fb"], b["bs"]) if b["nF"] > 0 else torch.zeros(b["bs"], self.d, device=gV.device)
            feats.append(gF)
        return self.head(torch.cat(feats, -1)).squeeze(-1)


# ─────────────────────────── stage 3: Lagrangian-monitored predictor–corrector ───────────────────────────

def lagrangian_monitor(XE, b):
    """Action + real-line coherence ρ. gradient energy ‖B₁X‖² is the coherent (real-line) part;
    curl energy ‖B₂ᵀX‖² is the off-real (rotational) part. ρ = grad/(grad+curl) ∈ (0,1] — cheap,
    no eigensolve. Low ρ ⇒ the state has drifted off the real line ⇒ correct."""
    gradE = (B1_E2V(XE, b) ** 2).sum()
    curlE = (B2T_E2F(XE, b) ** 2).sum() if b["nF"] > 0 else XE.new_zeros(())
    action = (XE * XE).sum()
    rho = gradE / (gradE + curlE + 1e-6)
    return action, rho


class LagrangianGreenLayer(nn.Module):
    """Green's predictor + Lagrangian-monitored corrector. The predictor forward-propagates
    (resolvent); the monitor reads the real-line coherence ρ; the corrector is a curl-directed
    back-solve, gated by (1−ρ) — 'on the real line ⇒ carry the coherent flow forward; off it ⇒
    correct via the curl component.' The direction is chosen by the action monitor, per layer."""
    def __init__(self, d, use_faces=True):
        super().__init__()
        self.uf = use_faces
        keys = ["self", "dn", "green", "corr"] + (["up"] if use_faces else [])
        self.e = nn.ModuleDict({k: nn.Linear(d, d) for k in keys})
        self.v = nn.ModuleDict({k: nn.Linear(d, d) for k in ["self", "up"]})
        if use_faces:
            self.f = nn.ModuleDict({k: nn.Linear(d, d) for k in ["self", "dn"]})
        self.alpha = nn.Parameter(torch.tensor(1.0)); self.beta = nn.Parameter(torch.tensor(1.0))
        self.lnE = nn.LayerNorm(d); self.lnV = nn.LayerNorm(d)
        self.lnF = nn.LayerNorm(d) if use_faces else None
        self.last_rho = None

    def forward(self, XV, XE, XF, b):
        from rexgraph.nn.rcf_torch import green_resolvent
        mv = edge_L1_matvec(b)
        green = green_resolvent(XE, self.alpha.abs() + 1e-3, mv, tol=1e-4, max_iter=30)   # predictor
        action, rho = lagrangian_monitor(green, b)
        self.last_rho = float(rho.detach())
        curl_comp = B2_F2E(B2T_E2F(green, b), b) if b["nF"] > 0 else torch.zeros_like(green)  # up-Lap=curl
        corr = green_resolvent(curl_comp, self.beta.abs() + 1e-3, mv, tol=1e-4, max_iter=30)  # corrector
        e = (self.e["self"](XE) + self.e["dn"](B1T_V2E(XV, b))
             + self.e["green"](rho * green) + self.e["corr"]((1 - rho) * corr))
        if self.uf:
            e = e + self.e["up"](B2_F2E(XF, b))
        XE = XE + F.gelu(self.lnE(e))
        XV = XV + F.gelu(self.lnV(self.v["self"](XV) + self.v["up"](B1_E2V(XE, b))))
        if self.uf and b["nF"] > 0:
            XF = XF + F.gelu(self.lnF(self.f["self"](XF) + self.f["dn"](B2T_E2F(XE, b))))
        return XV, XE, XF


class LagrangianGreenNet(nn.Module):
    def __init__(self, d=48, n_layers=2, use_faces=True):
        super().__init__()
        self.uf = use_faces; self.d = d
        self.v0 = nn.Linear(1, d); self.e0 = nn.Linear(1, d); self.f0 = nn.Linear(1, d)
        self.layers = nn.ModuleList([LagrangianGreenLayer(d, use_faces) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear((3 if use_faces else 2) * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, b):
        dev = b["src"].device
        XV = self.v0(torch.ones(b["nV"], 1, device=dev))
        XE = self.e0(torch.ones(b["nE"], 1, device=dev))
        XF = self.f0(torch.ones(max(b["nF"], 1), 1, device=dev))[:b["nF"]] if self.uf \
            else torch.zeros(0, self.d, device=dev)
        for l in self.layers:
            XV, XE, XF = l(XV, XE, XF, b)
        gV = seg_mean(XV, b["vb"], b["bs"]); gE = seg_mean(XE, b["eb"], b["bs"])
        feats = [gV, gE]
        if self.uf:
            gF = seg_mean(XF, b["fb"], b["bs"]) if b["nF"] > 0 else torch.zeros(b["bs"], self.d, device=gV.device)
            feats.append(gF)
        return self.head(torch.cat(feats, -1)).squeeze(-1)

    def query(self, b, XE, seed_edges):
        """Shared-complex inference: query the model's OWN complex by reweighting — put a signal
        on seed edges, propagate via the Green's resolvent, return the surfaced response. Runs on
        the same structure being trained (model = complex = memory)."""
        from rexgraph.nn.rcf_torch import green_resolvent
        sig = torch.zeros(b["nE"], XE.shape[-1], device=XE.device)
        sig[seed_edges] = 1.0
        return green_resolvent(sig, torch.tensor(1.0, device=XE.device), edge_L1_matvec(b),
                               tol=1e-4, max_iter=30)


# ─────────────────────────── verify the boundary ops before trusting results ───────────────────────────

def _verify_ops():
    dev = "cpu"
    rng = np.random.default_rng(0)
    b, _ = make_batch(2, rng, dev)
    nV, nE, nF = b["nV"], b["nE"], b["nF"]
    B1 = torch.zeros(nV, nE); B1[b["src"], torch.arange(nE)] = -1; B1[b["tgt"], torch.arange(nE)] = 1
    B2 = torch.zeros(nE, nF)
    for f in range(nF):
        for j in range(3):
            B2[b["fe"][f, j], f] = b["fs"][f, j]
    assert torch.allclose(B1 @ B2, torch.zeros(nV, nF), atol=1e-5), "∂²=0 violated"
    xE = torch.randn(nE, 3); xV = torch.randn(nV, 3); xF = torch.randn(max(nF, 1), 3)[:nF]
    assert torch.allclose(B1_E2V(xE, b), B1 @ xE, atol=1e-5)
    assert torch.allclose(B1T_V2E(xV, b), B1.T @ xV, atol=1e-5)
    if nF:
        assert torch.allclose(B2_F2E(xF, b), B2 @ xF, atol=1e-5)
        assert torch.allclose(B2T_E2F(xE, b), B2.T @ xE, atol=1e-5)
    return nF


# ─────────────────────────── train / eval ───────────────────────────

def train(model_fn, seed, steps, device, target="triangles", graphs="er"):
    torch.manual_seed(seed)
    model = model_fn().to(device)
    opt = HodgeAdam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        b, y = make_batch(64, rng, device, target=target, graphs=graphs)
        loss = F.mse_loss(model(b), y)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval R² on fresh graphs
    with torch.no_grad():
        b, y = make_batch(512, np.random.default_rng(seed + 999), device, target=target, graphs=graphs)
        pred = model(b)
        ss_res = ((pred - y) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot).item()
    n_params = sum(p.numel() for p in model.parameters())
    return r2, n_params


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default=None); ap.add_argument("--target", default="triangles")
    ap.add_argument("--graphs", default="er", choices=["er", "regular"])
    ap.add_argument("--stage2", action="store_true", help="explicit vs Green's-implicit layers")
    ap.add_argument("--stage3", action="store_true", help="Green's vs Lagrangian-monitored")
    a = ap.parse_args(argv)
    device = a.device or pick_device()
    steps = 300 if a.quick else 800
    seeds = [0, 1] if a.quick else [0, 1, 2]
    nf = _verify_ops()
    print("boundary ops verified (∂²=0, B₁/B₂ match dense); sample batch had %d triangle-faces" % nf)
    print("device=%s steps=%d seeds=%s  target=%s  graphs=%s  (R², higher=better)\n"
          % (device, steps, seeds, a.target, a.graphs))
    if a.stage3:
        configs = [
            ("Green's implicit (2 layers)", lambda: GreenComplexNet(48, 2, use_faces=True)),
            ("Lagrangian-monitored (2 lyr)", lambda: LagrangianGreenNet(48, 2, use_faces=True)),
        ]
    elif a.stage2:
        configs = [
            ("explicit (3 layers, stage-1)", lambda: ComplexNet(48, 3, use_faces=True)),
            ("Green's implicit (1 layer)", lambda: GreenComplexNet(48, 1, use_faces=True)),
            ("Green's implicit (2 layers)", lambda: GreenComplexNet(48, 2, use_faces=True)),
        ]
    else:
        configs = [
            ("complex (edge-primary, faces)", lambda: ComplexNet(48, 3, use_faces=True)),
            ("complex (faces OFF, ablation)", lambda: ComplexNet(48, 3, use_faces=False)),
            ("pairwise GNN (baseline)", lambda: PairwiseGNN(48, 3)),
        ]
    for name, fn in configs:
        r2s, npar = [], 0
        for s in seeds:
            r2, npar = train(fn, s, steps, device, a.target, a.graphs)
            r2s.append(r2)
        print("  %-34s R²=%.3f±%.3f  (%d params)" % (name, float(np.mean(r2s)), float(np.std(r2s)), npar))


if __name__ == "__main__":
    main()
