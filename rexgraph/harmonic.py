"""
rexgraph/harmonic.py

The harmonic plane of numbers: thin wrapper over _harmonic Cython module.
Falls back to pure Python when compiled module is unavailable.
See rexgraph.core._harmonic for full documentation.

Nothing imports this module. It is not in `rexgraph.__init__`, has no callers and
no tests, and it reaches the harmonic plane the retired way: `B1_dense`,
`B2_dense`, a dense nE x nE L1 and `np.linalg.eigh` against a hardcoded cutoff.

The live surface is `rexgraph.harmonic_sparse` for the frame and projection, and
`rexgraph.hodge_coords` for coordinates, the metric, closure and the Gram
determinant. `harmonic_product_structure` here and `harmonic_closure` there
compute the same object; the latter does it off the small Gram with no
eigendecomposition. Kept for the prime-coupling experiments, which have no
equivalent elsewhere.
"""

from __future__ import annotations

import numpy as np

# re-exported on purpose: this module is a wrapper, so `from rexgraph.harmonic import
# harmonic_projectors` is part of what it is for. The names flagged below are unused
# HERE and reachable THROUGH here, which is the distinction noqa is recording.
try:
    from .core._harmonic import (
        harmonic_basis,
        harmonic_channel_character,
        harmonic_decode,  # noqa: F401
        harmonic_encode,  # noqa: F401
        harmonic_leakage,  # noqa: F401
        harmonic_product_table,
        harmonic_projectors,  # noqa: F401
        prime_coupling,
        prime_removal_analysis,
    )
    _COMPILED = True
except ImportError:
    _COMPILED = False

from .graph import RexGraph


def build_prime_complex(k: int, removed_vertex: int | None = None):
    """Build K_k on the first k primes, optionally removing faces for one vertex."""
    try:
        from sympy import primerange
        primes = list(primerange(2, max(500, k * 50)))[:k]
    except ImportError:
        primes, c = [], 2
        while len(primes) < k:
            if all(c % p != 0 for p in primes): primes.append(c)
            c += 1

    src, tgt = [], []
    for i in range(k):
        for j in range(i + 1, k):
            src.append(i); tgt.append(j)
    all_tri = [(i,j,m) for i in range(k) for j in range(i+1,k) for m in range(j+1,k)]
    triangles = [t for t in all_tri if removed_vertex not in t] if removed_vertex is not None else all_tri
    rex = RexGraph.from_simplicial(
        np.array(src, dtype=np.int32), np.array(tgt, dtype=np.int32),
        np.array(triangles, dtype=np.int32) if triangles else np.zeros((0,3), dtype=np.int32))
    return rex, primes


def _py_harmonic_basis(B1, B2):
    L1 = B1.T @ B1 + B2 @ B2.T
    evals, evecs = np.linalg.eigh(L1)
    return evecs[:, evals < 1e-10], evals


def get_harmonic_basis(rex: RexGraph):
    """Extract harmonic basis from a RexGraph."""
    B1, B2 = rex.B1_dense, rex.B2_dense
    if _COMPILED:
        return harmonic_basis(B1, B2)
    return _py_harmonic_basis(B1, B2)


def prime_removal_test(k: int) -> dict:
    """Test beta_1=k-2 and (k-1)/k concentration for all prime removals."""
    try:
        from sympy import primerange
        primes = list(primerange(2, max(500, k * 50)))[:k]
    except ImportError:
        primes, c = [], 2
        while len(primes) < k:
            if all(c % p != 0 for p in primes): primes.append(c)
            c += 1

    log_p = np.log(np.array(primes, dtype=np.float64))
    src, tgt = [], []
    for i in range(k):
        for j in range(i+1, k): src.append(i); tgt.append(j)
    src_arr, tgt_arr = np.array(src, dtype=np.int32), np.array(tgt, dtype=np.int32)
    all_tri = [(i,j,m) for i in range(k) for j in range(i+1,k) for m in range(j+1,k)]

    results = []
    for p_idx in range(k):
        partial = [t for t in all_tri if p_idx not in t]
        tri_arr = np.array(partial, dtype=np.int32)
        rex = RexGraph.from_simplicial(src_arr, tgt_arr, tri_arr)
        B1, B2 = rex.B1_dense, rex.B2_dense
        if _COMPILED:
            r = prime_removal_analysis(k, p_idx, src_arr, tgt_arr, B1, B2, log_p)
        else:
            hb, _ = _py_harmonic_basis(B1, B2)
            P = hb @ hb.T
            sig = np.array([log_p[src_arr[e]] + log_p[tgt_arr[e]] for e in range(rex.nE)])
            harm = P @ sig
            edges_p = [e for e in range(rex.nE) if src_arr[e]==p_idx or tgt_arr[e]==p_idx]
            h_on = sum(harm[e]**2 for e in edges_p)
            h_total = float(np.sum(harm**2))
            r = {'beta_1': hb.shape[1], 'concentration': h_on/h_total if h_total>1e-30 else 0,
                 'harm_norm': float(np.linalg.norm(harm)),
                 'expected_beta_1': k-2, 'expected_concentration': (k-1)/k}
        r['prime'] = primes[p_idx]; r['prime_idx'] = p_idx
        results.append(r)

    return {'k': k, 'primes': primes, 'results': results,
            'beta_1_exact': all(r['beta_1']==k-2 for r in results),
            'concentration_exact': all(abs(r['concentration']-(k-1)/k)<0.01 for r in results)}


def harmonic_product_structure(rex: RexGraph) -> dict:
    """Characterize H's algebraic structure under Hadamard product."""
    B1, B2 = rex.B1_dense, rex.B2_dense
    if _COMPILED:
        hb, _ = harmonic_basis(B1, B2)
    else:
        hb, _ = _py_harmonic_basis(B1, B2)
    if hb.shape[1] == 0:
        return {'dim_H': 0, 'error': 'trivial harmonic subspace'}
    if _COMPILED:
        result = harmonic_product_table(hb)
        chi = rex.structural_character
        chi_H, iso = harmonic_channel_character(hb, chi)
        result['channel_character'] = chi_H.tolist()
        result['channel_isotropic'] = bool(iso)
        return result
    # Python fallback
    n_harm = hb.shape[1]; P = hb @ hb.T
    closure = np.zeros((n_harm, n_harm))
    for i in range(n_harm):
        for j in range(n_harm):
            prod = hb[:,i]*hb[:,j]; proj = P@prod
            t = np.linalg.norm(prod)**2
            closure[i,j] = np.linalg.norm(proj)**2/t if t>1e-30 else 0
    chi = rex.structural_character; chi_H = np.zeros(4)
    for h_idx in range(n_harm):
        h = hb[:,h_idx]; h_sq = h**2; h_sq /= max(h_sq.sum(), 1e-30)
        chi_H += chi.T @ h_sq
    chi_H /= max(n_harm, 1)
    return {'dim_H': n_harm, 'mean_closure': float(closure.mean()),
            'channel_character': chi_H.tolist(), 'channel_isotropic': bool(np.std(chi_H)<0.02)}


def prime_coupling_matrix(k: int) -> dict:
    """Compute pairwise cosine coupling between prime tensor positions."""
    try:
        from sympy import primerange
        primes = list(primerange(2, max(500, k*50)))[:k]
    except ImportError:
        primes, c = [], 2
        while len(primes) < k:
            if all(c%p!=0 for p in primes): primes.append(c)
            c += 1
    log_p = np.log(np.array(primes, dtype=np.float64))
    src, tgt = [], []
    for i in range(k):
        for j in range(i+1,k): src.append(i); tgt.append(j)
    src_arr, tgt_arr = np.array(src, dtype=np.int32), np.array(tgt, dtype=np.int32)
    all_tri = [(i,j,m) for i in range(k) for j in range(i+1,k) for m in range(j+1,k)]

    if _COMPILED:
        result = prime_coupling(k, all_tri, src_arr, tgt_arr, log_p)
    else:
        projs = {}
        for p_idx in range(k):
            partial = [t for t in all_tri if p_idx not in t]
            tri_arr = np.array(partial, dtype=np.int32)
            rex = RexGraph.from_simplicial(src_arr, tgt_arr, tri_arr)
            hb, _ = _py_harmonic_basis(rex.B1_dense, rex.B2_dense)
            P = hb @ hb.T
            sig = np.array([log_p[src_arr[e]]+log_p[tgt_arr[e]] for e in range(rex.nE)])
            projs[p_idx] = P @ sig
        coupling = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                ni,nj = np.linalg.norm(projs[i]), np.linalg.norm(projs[j])
                if ni>1e-10 and nj>1e-10: coupling[i,j]=float(np.dot(projs[i],projs[j])/(ni*nj))
        off = [coupling[i,j] for i in range(k) for j in range(k) if i!=j]
        result = {'coupling': coupling.tolist(), 'mean_coupling': float(np.mean(off)),
                  'max_coupling': float(np.max(np.abs(off))), 'asymptotically_orthogonal': float(np.mean(off))<0.5}
    result['k'] = k; result['primes'] = primes
    return result
