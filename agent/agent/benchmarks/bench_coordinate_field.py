"""The complex supplies the coordinate field; nothing has to be traced or reduced.

A network that wants to consume a signal on a relational complex normally gets
handed the raw edge vector and left to find structure in it, or gets an embedding
traced out in Euclidean space and cut down to a chosen width. The complex already
carries a coordinate system: the Hodge chart. Its harmonic block has one axis per
independent hole, the axes are cycles with entries in {0, +1, -1}, and a signal's
position along them is exact.

The task here is harmonic by construction, so it is a fair place to ask what the
coordinates buy. Each sample is an edge signal `f = H c + B1^T phi`: a harmonic
part with winding `c`, plus a gradient part that carries no holes. The label is
which hole dominates, `argmax |c|`, which is a function of the harmonic part alone
and is invisible to anything that cannot separate it out.

Three readings, each fed to the same model so the comparison is about the
features and not the model:

    raw          the edge vector, nE wide
    pca          the edge vector reduced to dim_H by PCA on the training set,
                 which is the ordinary "embed then reduce" route
    coords       G^-1 H^T f, dim_H wide, no fitting and no choice of width

Two targets, because the model class has to be able to express the label.
`argmax |c|` is not linear in the coordinates, so a linear readout fails on it
whatever it is fed; `sign(w . c)` is. Both are reported.

There is also a row with no training at all. `G^-1 H^T (H c + B1^T phi) = c`
exactly, since the frame lies in ker B1 and kills the gradient part, so
`argmax |coords|` IS the label and reads off directly. That row is the point:
the coordinate field is not a feature that helps a network, it is the answer the
network was being asked to reconstruct.

Run: python -m agent.benchmarks.bench_coordinate_field
"""

from __future__ import annotations

import numpy as np

from rexgraph.graph import RexGraph
from rexgraph.hodge_coords import harmonic_coords, harmonic_frame

N_TRAIN, N_TEST, STEPS, LR = 3000, 1000, 400, 0.5


def build_graph(nV, nE, seed):
    """A connected graph with nE - nV + 1 independent holes."""
    rng = np.random.default_rng(seed)
    edges = [(i, i + 1) for i in range(nV - 1)]          # a spanning path
    while len(edges) < nE:
        a, b = int(rng.integers(0, nV)), int(rng.integers(0, nV))
        if a != b and (min(a, b), max(a, b)) not in {(min(x, y), max(x, y)) for x, y in edges}:
            edges.append((a, b))
    r = RexGraph(sources=np.array([a for a, _ in edges], np.int32),
                 targets=np.array([b for _, b in edges], np.int32))
    r._ensure_clean()
    return r


def samples(rex, H, n, rng, noise, w=None):
    """`f = H c + B1^T phi`, with both labels: which hole dominates, and the
    sign of a fixed linear functional of the winding."""
    from rexgraph.core._sparse import rmatvec

    k = H.shape[1]
    C = rng.normal(size=(n, k))
    phi = rng.normal(size=(n, rex.nV)) * noise
    Hd = np.asarray(H.todense())
    F = C @ Hd.T + np.array([np.asarray(rmatvec(rex.B1_sparse, p)).ravel() for p in phi])
    return F, np.argmax(np.abs(C), axis=1), (C @ w > 0).astype(int)


def softmax_fit(Xtr, ytr, Xte, yte, k, seed, hidden=0):
    """Plain gradient descent, so the comparison is about the features."""
    rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = np.eye(k)[ytr]
    if hidden:
        W1 = rng.normal(size=(Xtr.shape[1], hidden)) * 0.1
        b1 = np.zeros(hidden)
        W2 = rng.normal(size=(hidden, k)) * 0.1
        b2 = np.zeros(k)
        for _ in range(STEPS):
            Z = np.maximum(Xtr @ W1 + b1, 0)
            P = np.exp(Z @ W2 + b2 - (Z @ W2 + b2).max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            dZ = (P - Y) @ W2.T * (Z > 0)
            W2 -= LR * Z.T @ (P - Y) / len(Y)
            b2 -= LR * (P - Y).mean(0)
            W1 -= LR * Xtr.T @ dZ / len(Y)
            b1 -= LR * dZ.mean(0)
        Zt = np.maximum(Xte @ W1 + b1, 0)
        return float((np.argmax(Zt @ W2 + b2, 1) == yte).mean())
    W = rng.normal(size=(Xtr.shape[1], k)) * 0.1
    b = np.zeros(k)
    for _ in range(STEPS):
        S = Xtr @ W + b
        P = np.exp(S - S.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        W -= LR * Xtr.T @ (P - Y) / len(Y)
        b -= LR * (P - Y).mean(0)
    return float((np.argmax(Xte @ W + b, 1) == yte).mean())


def pca(Xtr, Xte, d):
    """The ordinary embed-then-reduce route, fitted on the training set only."""
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return (Xtr - mu) @ Vt[:d].T, (Xte - mu) @ Vt[:d].T


def run(nV=40, nE=50, noise=1.0, seeds=(0, 1, 2, 3, 4)):
    rows = []
    for seed in seeds:
        rex = build_graph(nV, nE, seed)
        H = harmonic_frame(rex)
        k = H.shape[1]
        rng = np.random.default_rng(1000 + seed)
        w = rng.normal(size=k)
        Xtr, ytr, str_ = samples(rex, H, N_TRAIN, rng, noise, w)
        Xte, yte, ste = samples(rex, H, N_TEST, rng, noise, w)
        Ctr = np.array([harmonic_coords(rex, f, frame=H) for f in Xtr])
        Cte = np.array([harmonic_coords(rex, f, frame=H) for f in Xte])
        Ptr, Pte = pca(Xtr, Xte, k)
        rows.append({
            "seed": seed, "nE": rex.nE, "dim_H": k,
            "chance argmax": 1.0 / k, "chance sign": 0.5,
            # argmax |c| is not linear, so every feature set gets the MLP
            "argmax raw mlp": softmax_fit(Xtr, ytr, Xte, yte, k, seed, hidden=128),
            "argmax pca mlp": softmax_fit(Ptr, ytr, Pte, yte, k, seed, hidden=128),
            "argmax coords mlp": softmax_fit(Ctr, ytr, Cte, yte, k, seed, hidden=128),
            # no fitting of any kind: the coordinates are the winding
            "argmax coords read": float((np.argmax(np.abs(Cte), 1) == yte).mean()),
            # sign(w . c) is linear in the coordinates, so a linear readout is fair
            "sign raw linear": softmax_fit(Xtr, str_, Xte, ste, 2, seed),
            "sign pca linear": softmax_fit(Ptr, str_, Pte, ste, 2, seed),
            "sign coords linear": softmax_fit(Ctr, str_, Cte, ste, 2, seed),
        })
    return rows


def main():
    rows = run()
    groups = [
        ("argmax |c|  (nonlinear label, MLP for every feature set)", "chance argmax",
         ["argmax raw mlp", "argmax pca mlp", "argmax coords mlp",
          "argmax coords read"]),
        ("sign(w . c)  (linear label, linear readout for every feature set)",
         "chance sign",
         ["sign raw linear", "sign pca linear", "sign coords linear"]),
    ]
    print(f"  nE {rows[0]['nE']}, dim_H {rows[0]['dim_H']}, "
          f"{N_TRAIN} train / {N_TEST} test, {len(rows)} seeds")
    for title, chance, cols in groups:
        print()
        print(f"  {title}")
        print(f"    chance {rows[0][chance]:.3f}")
        print(f"    {'seed':>4} " + " ".join(f"{c.split(' ', 1)[1]:>13}" for c in cols))
        for r in rows:
            print(f"    {r['seed']:>4} " + " ".join(f"{r[c]:>13.3f}" for c in cols))
        print("    " + "-" * (5 + 14 * len(cols)))
        print("    worst" + " ".join(f"{min(r[c] for r in rows):>13.3f}" for c in cols))


if __name__ == "__main__":
    main()
