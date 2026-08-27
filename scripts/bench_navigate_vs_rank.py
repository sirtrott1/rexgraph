"""Navigating a tree against ranking it."""
import glob, os, random, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
from rexgraph.document import build_document, read_document, section_text
from rexgraph.sectioning import sectionings_of
from rexgraph.core._sparse import to_scipy_csr

CHAIN = ("chapter", "paragraph", "sentence", "span")


def layers_of(rex, info):
    st = sectionings_of(rex)
    base = info["base_layer"]
    chain = [c for c in CHAIN if c in st]
    if base not in chain:
        return None
    nE = int(rex.nE)
    owners = {}
    for name in chain:
        s = st[name]
        s = s.resolved(st) if s.is_derived else s
        owners[name] = np.asarray(s.owner_cochain(nE), dtype=np.int64)
    return chain, owners, base


def accumulate(owner, resp_cells, mask=None):
    keep = owner >= 0 if mask is None else (owner >= 0) & mask
    out = np.zeros(int(owner.max()) + 1 if owner.size else 0)
    np.add.at(out, owner[keep], resp_cells[keep])
    return out


def cell_reading(B, vertex_field):
    C = B.tocoo()
    out = np.zeros(B.shape[1])
    np.add.at(out, C.col, np.abs(vertex_field[C.row]))
    return out


def greens_potential(B, q, iters=200):
    n = B.shape[0]
    qq = q - q.mean()
    L = spla.LinearOperator((n, n), matvec=lambda x: B @ (B.T @ x), dtype=np.float64)
    u, _info = spla.cg(L, qq, rtol=1e-8, maxiter=iters)
    return u


def tree_potential(chain, owners, leaf_scores, base):
    nodes, offset = {}, 0
    for name in chain:
        n_i = int(owners[name].max()) + 1
        nodes[name] = (offset, n_i)
        offset += n_i
    root = offset
    nV = offset + 1

    rows, cols, vals, e = [], [], [], 0
    for k, name in enumerate(chain):
        off, n_i = nodes[name]
        up = chain[k + 1] if k + 1 < len(chain) else None
        for j in range(n_i):
            cells = np.flatnonzero(owners[name] == j)
            if cells.size == 0:
                parent = root
            elif up is None:
                parent = root
            else:
                pj = owners[up][cells[0]]
                parent = (nodes[up][0] + int(pj)) if pj >= 0 else root
            rows += [off + j, parent]; cols += [e, e]; vals += [-1.0, 1.0]; e += 1
    B = sp.csc_matrix((vals, (rows, cols)), shape=(nV, e))

    q = np.zeros(nV)
    off, n_i = nodes[base]
    m = min(n_i, leaf_scores.size)
    q[off:off + m] = leaf_scores[:m]
    q[root] = -q.sum()
    L = spla.LinearOperator((nV, nV), matvec=lambda x: B @ (B.T @ x), dtype=np.float64)
    u, _i = spla.cg(L, q - q.mean(), rtol=1e-8, maxiter=400)
    return u, nodes, root


paths = [p for p in sorted(glob.glob(os.path.expanduser(
    '~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')))
    if 80_000 < os.path.getsize(p) < 300_000][:10]

rng = random.Random(23)
same_B, same_C, same_D = [], [], []
true_A, true_B, true_C, true_D = [], [], [], []
touched_A, touched_B = [], []
t_rank = t_walk = 0.0

for p in paths:
    raw, _ = read_document(p)
    rex, info = build_document(raw, profile=ENGLISH_GUTENBERG)
    L = layers_of(rex, info)
    if L is None:
        continue
    chain, owners, base = L
    if len(chain) < 3:
        continue
    B = to_scipy_csr(rex._B1_dual).tocsr()
    deg = np.asarray(rex.degree, dtype=np.float64)
    vocab = {str(v).lower(): i for i, v in enumerate(info["vocab"])}
    n_base = int(owners[base].max()) + 1

    for _ in range(12):
        i = rng.randrange(n_base)
        q = section_text(rex, base, i, raw).strip()
        if not (40 < len(q) < 400):
            continue
        seeds = [vocab[w] for w, _a, _b in tokenize(q, ENGLISH_GUTENBERG) if w in vocab]
        if not seeds:
            continue
        x = np.zeros(int(rex.nV))
        sd = np.asarray(seeds)
        x[sd] = 1.0 / np.maximum(deg[sd], 1.0)

        # A: rank every leaf
        t0 = time.perf_counter()
        applied = np.abs(B @ (B.T @ x))
        cells_A = cell_reading(B, applied)
        a = int(np.argmax(accumulate(owners[base], cells_A)))
        t_rank += time.perf_counter() - t0
        touched_A.append(n_base)

        # B: descend on the same reading
        t0 = time.perf_counter()
        mask = np.ones(int(rex.nE), dtype=bool)
        seen = 0
        for name in chain:
            sc = accumulate(owners[name], cells_A, mask)
            if not sc.size or sc.max() <= 0:
                break
            seen += int((sc > 0).sum())
            pick = int(np.argmax(sc))
            mask = mask & (owners[name] == pick)
        b = int(np.argmax(accumulate(owners[base], cells_A, mask))) if mask.any() else -1
        t_walk += time.perf_counter() - t0
        touched_B.append(seen)

        # C: descend on the Green's potential
        cells_C = cell_reading(B, greens_potential(B, x))
        mask = np.ones(int(rex.nE), dtype=bool)
        for name in chain:
            sc = accumulate(owners[name], cells_C, mask)
            if not sc.size or sc.max() <= 0:
                break
            mask = mask & (owners[name] == int(np.argmax(sc)))
        c = int(np.argmax(accumulate(owners[base], cells_C, mask))) if mask.any() else -1

        # D: Theorem 26's own setting: the potential solved ON THE TREE
        leaves = accumulate(owners[base], cells_A)
        u, nodes, root = tree_potential(chain, owners, leaves, base)
        cur, d = root, -1
        for name in reversed(chain):                 # root -> chapter -> ... -> span
            off, n_i = nodes[name]
            child = np.arange(n_i)
            # only children of `cur`
            keep = []
            for j in child:
                cells = np.flatnonzero(owners[name] == j)
                if cells.size == 0:
                    continue
                up = chain[chain.index(name) + 1] if chain.index(name) + 1 < len(chain) else None
                par = root if up is None else nodes[up][0] + int(owners[up][cells[0]])
                if par == cur:
                    keep.append(j)
            if not keep:
                break
            pick = max(keep, key=lambda j: u[off + j])
            cur, d = off + pick, pick
        true_D.append(d == i); same_D.append(d == a)

        same_B.append(b == a); same_C.append(c == a)
        true_A.append(a == i); true_B.append(b == i); true_C.append(c == i)
    print(f"  {os.path.basename(p):12s} n={len(true_A)}", flush=True)

n = len(true_A)
print(f"\nn = {n} queries over {len(paths)} documents, chain {CHAIN}")
print(f"\n  arrives at the TRUE span")
print(f"    A rank all leaves        {np.mean(true_A)*100:5.1f}%")
print(f"    B descend, applied field {np.mean(true_B)*100:5.1f}%")
print(f"    C descend, Green's       {np.mean(true_C)*100:5.1f}%")
print(f"    D descend, TREE potential {np.mean(true_D)*100:5.1f}%   <- Thm 26's setting")
print(f"\n  agrees with A's answer")
print(f"    B {np.mean(same_B)*100:5.1f}%     C {np.mean(same_C)*100:5.1f}%     "
      f"D {np.mean(same_D)*100:5.1f}%")
print(f"\n  sections evaluated: rank {int(np.median(touched_A))}  "
      f"walk {int(np.median(touched_B))}")
print(f"  time: rank {t_rank:.2f}s  walk {t_walk:.2f}s")
