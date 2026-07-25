"""Test _linalg module: LAPACK/BLAS wrappers and full RL pipeline."""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import pinvh
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import compiled module; skip if not available
try:
    from rexgraph.core._linalg import (
        eigh, svd, lstsq, matrix_rank,
        gemm_nn, gemm_nt, gemm_tn,
        pinv_spectral, pinv_matvec,
        rl_pipeline,
    )
    HAS_LINALG = True
except ImportError:
    HAS_LINALG = False

pytestmark = pytest.mark.skipif(not HAS_LINALG, reason="_linalg not compiled")

class TestEigh:
    def test_eigenvalues(self):
        A = np.array([[4,1,0],[1,3,1],[0,1,2]], dtype=np.float64)
        ev, _ = eigh(A)
        ev_ref = np.linalg.eigh(A)[0]
        assert np.allclose(sorted(ev), sorted(ev_ref), atol=1e-12)

    def test_eigenvectors(self):
        A = np.array([[4,1,0],[1,3,1],[0,1,2]], dtype=np.float64)
        ev, evec = eigh(A)
        for k in range(3):
            assert norm(A @ evec[:, k] - ev[k] * evec[:, k]) < 1e-10

class TestSVD:
    def test_reconstruction(self):
        M = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
        U, S, Vt = svd(M)
        assert np.allclose(U[:,:2] @ np.diag(S) @ Vt[:2,:], M, atol=1e-10)

class TestLstsq:
    def test_solution(self):
        A = np.array([[1,1],[1,2],[1,3]], dtype=np.float64)
        b = np.array([1,2,2], dtype=np.float64)
        x, rank = lstsq(A, b)
        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
        assert np.allclose(x, x_ref, atol=1e-10)

class TestGemm:
    def test_nn(self):
        A = np.random.randn(4, 5).astype(np.float64)
        B = np.random.randn(5, 3).astype(np.float64)
        assert np.allclose(gemm_nn(A, B), A @ B, atol=1e-12)

    def test_nt(self):
        A = np.random.randn(4, 5).astype(np.float64)
        assert np.allclose(gemm_nt(A, A), A @ A.T, atol=1e-12)

    def test_tn(self):
        A = np.random.randn(4, 5).astype(np.float64)
        assert np.allclose(gemm_tn(A, A), A.T @ A, atol=1e-12)

class TestPinv:
    def test_spectral(self):
        RL = np.array([[2,1,0],[1,3,1],[0,1,2]], dtype=np.float64)
        ev, evec = eigh(RL)
        RLp = pinv_spectral(ev, evec)
        assert np.allclose(RLp, pinvh(RL), atol=1e-10)

    def test_matvec(self):
        RL = np.array([[2,1,0],[1,3,1],[0,1,2]], dtype=np.float64)
        ev, evec = eigh(RL)
        x = np.array([1, 0, 0], dtype=np.float64)
        assert np.allclose(pinv_matvec(ev, evec, x), pinvh(RL) @ x, atol=1e-10)

class TestRLPipeline:
    @pytest.fixture
    def drct(self):
        edges = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
                 (1,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,7),(1,7),(2,3)]
        nV, nE = 8, 16
        B1 = np.zeros((nV, nE), dtype=np.float64)
        for j, (s,t) in enumerate(edges): B1[s,j] = -1; B1[t,j] = 1
        from numpy.linalg import svd as np_svd
        adj = [set() for _ in range(nV)]
        emap = {}
        for i, (s,t) in enumerate(edges):
            adj[s].add(t); adj[t].add(s); emap[(min(s,t),max(s,t))] = i
        tris = []
        for i in range(nV):
            for j in adj[i]:
                if j<=i: continue
                for k in adj[i]&adj[j]:
                    if k<=j: continue
                    tris.append((emap[(min(i,j),max(i,j))], emap[(min(i,k),max(i,k))], emap[(min(j,k),max(j,k))]))
        B2 = np.zeros((nE, min(10,len(tris))), dtype=np.float64)
        for fi, (e0,e1,e2) in enumerate(tris[:10]):
            sub = B1[:,[e0,e1,e2]]; _,_,Vt = np_svd(sub)
            kv = Vt[-1,:]; kv = np.sign(kv/np.max(np.abs(kv)))
            B2[e0,fi],B2[e1,fi],B2[e2,fi] = kv
        L1 = B1.T@B1+B2@B2.T
        K1 = np.abs(B1).T@np.abs(B1); rs=K1.sum(1)
        D = np.diag(np.where(rs>1e-12,1/np.sqrt(rs),0))
        L_O = 0.5*((np.eye(nE)-D@K1@D)+(np.eye(nE)-D@K1@D).T)
        deg = np.zeros(nV)
        for s,t in edges: deg[s]+=1;deg[t]+=1
        W = np.diag([1/np.log(d+np.e) for d in deg])
        Ks = B1.T@W@B1; Ks_off=Ks.copy(); np.fill_diagonal(Ks_off,0)
        L_SG = np.diag(np.abs(Ks_off).sum(1))-Ks_off
        return B1, L1, L_O, L_SG, nV, nE

    def test_trace(self, drct):
        B1, L1, L_O, L_SG, nV, nE = drct
        r = rl_pipeline(B1, L1, L_O, L_SG)
        assert abs(np.trace(r['RL']) - 3) < 1e-10

    def test_chi_simplex(self, drct):
        B1, L1, L_O, L_SG, nV, nE = drct
        r = rl_pipeline(B1, L1, L_O, L_SG)
        assert max(abs(r['chi'][e].sum()-1) for e in range(nE)) < 1e-14

    def test_phi_simplex(self, drct):
        B1, L1, L_O, L_SG, nV, nE = drct
        r = rl_pipeline(B1, L1, L_O, L_SG)
        assert max(abs(r['phi'][v].sum()-1) for v in range(nV)) < 1e-14

    def test_kappa_range(self, drct):
        B1, L1, L_O, L_SG, nV, nE = drct
        r = rl_pipeline(B1, L1, L_O, L_SG)
        assert np.all(r['kappa'] >= -1e-10) and np.all(r['kappa'] <= 1+1e-10)

    def test_phi_reference(self, drct):
        B1, L1, L_O, L_SG, nV, nE = drct
        r = rl_pipeline(B1, L1, L_O, L_SG)
        def tn(L):
            tr=np.trace(L); return (L/tr,tr) if tr>1e-15 else (np.zeros_like(L),0)
        h1,_=tn(L1);hO,_=tn(L_O);hSG,_=tn(L_SG)
        RL=h1+hO+hSG; RLp=pinvh(RL); S0=B1@RLp@B1.T
        phi_ref=np.zeros((nV,3))
        for v in range(nV):
            if abs(S0[v,v])>1e-15:
                for k in range(3):
                    phi_ref[v,k]=(B1@RLp@[h1,hO,hSG][k]@RLp@B1.T)[v,v]/S0[v,v]
        assert np.max(np.abs(r['phi']-phi_ref)) < 1e-12
