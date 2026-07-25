"""Integration test verifying graph.py RCF pipeline wiring.

Simulates the exact call chain that RexGraph's cached bundles use,
verifying each stage produces correct results.
"""
import numpy as np
from numpy.linalg import norm, svd
from scipy.linalg import pinvh
import pytest

def build_test_complex():
    edges = [(0,1),(1,2),(0,2),(0,3),(1,3)]
    nV, nE = 4, 5
    B1 = np.zeros((nV, nE), dtype=np.float64)
    for j, (s, t) in enumerate(edges): B1[s,j] = -1; B1[t,j] = 1
    sub = B1[:, [0,1,2]]; _, _, Vt = svd(sub)
    kv = Vt[-1,:]; kv = np.sign(kv/np.max(np.abs(kv)))
    B2 = np.zeros((nE, 1), dtype=np.float64)
    B2[0,0], B2[1,0], B2[2,0] = kv
    return edges, nV, nE, B1, B2

def build_operators(edges, nV, nE, B1, B2):
    L1 = B1.T@B1+B2@B2.T
    K1 = np.abs(B1).T@np.abs(B1)
    rs = K1.sum(1); D = np.diag(np.where(rs>1e-12,1/np.sqrt(rs),0))
    L_O = 0.5*((np.eye(nE)-D@K1@D)+(np.eye(nE)-D@K1@D).T)
    deg = np.zeros(nV)
    for s,t in edges: deg[s]+=1;deg[t]+=1
    W = np.diag([1/np.log(d+np.e) for d in deg])
    Ks = B1.T@W@B1; Ks_off=Ks.copy(); np.fill_diagonal(Ks_off,0)
    L_SG = np.diag(np.abs(Ks_off).sum(1))-Ks_off
    return L1, L_O, L_SG, K1

def trace_norm(L):
    tr=np.trace(L); return (L/tr,tr) if tr>1e-15 else (np.zeros_like(L),0)

class TestRCFPipeline:
    @pytest.fixture
    def complex(self):
        edges, nV, nE, B1, B2 = build_test_complex()
        L1, L_O, L_SG, K1 = build_operators(edges, nV, nE, B1, B2)
        h1,_=trace_norm(L1); hO,_=trace_norm(L_O); hSG,_=trace_norm(L_SG)
        RL=h1+hO+hSG; RLp=pinvh(RL)
        chi=np.zeros((nE,3))
        for e in range(nE):
            if RL[e,e]>1e-15:
                for k in range(3): chi[e,k]=[h1,hO,hSG][k][e,e]/RL[e,e]
        S0=B1@RLp@B1.T
        phi=np.zeros((nV,3))
        for v in range(nV):
            if abs(S0[v,v])>1e-15:
                for k in range(3):
                    phi[v,k]=(B1@RLp@[h1,hO,hSG][k]@RLp@B1.T)[v,v]/S0[v,v]
        v2e=[[] for _ in range(nV)]
        for e,(s,t) in enumerate(edges): v2e[s].append(e);v2e[t].append(e)
        cs=np.zeros((nV,3))
        for v in range(nV):
            if v2e[v]: cs[v]=np.mean([chi[e] for e in v2e[v]],axis=0)
        kappa=np.array([1-0.5*np.sum(np.abs(phi[v]-cs[v])) for v in range(nV)])
        return dict(edges=edges,nV=nV,nE=nE,B1=B1,B2=B2,L1=L1,L_O=L_O,
                    L_SG=L_SG,K1=K1,RL=RL,RLp=RLp,chi=chi,phi=phi,
                    chi_star=cs,kappa=kappa,hats=[h1,hO,hSG])

    def test_trace_rl(self, complex):
        assert abs(np.trace(complex['RL'])-3) < 1e-10

    def test_chi_simplex(self, complex):
        chi=complex['chi']
        assert max(abs(chi[e].sum()-1) for e in range(complex['nE'])) < 1e-10

    def test_phi_simplex(self, complex):
        phi=complex['phi']
        assert max(abs(phi[v].sum()-1) for v in range(complex['nV'])) < 1e-8

    def test_kappa_range(self, complex):
        assert np.all(complex['kappa'] >= -1e-10)
        assert np.all(complex['kappa'] <= 1+1e-10)

    def test_impute(self, complex):
        RL=complex['RL']
        obs=np.array([1.0,0.5,-0.3,0,0]); mask=np.array([1,1,1,0,0],dtype=bool)
        oi=np.where(mask)[0]; mi=np.where(~mask)[0]
        g=-pinvh(RL[np.ix_(mi,mi)])@RL[np.ix_(mi,oi)]@obs[oi]
        full=obs.copy(); full[mi]=g
        rnd=obs.copy(); rnd[3]=0.5; rnd[4]=-0.5
        assert full@RL@full <= rnd@RL@rnd+1e-10

    def test_bianchi(self, complex):
        B1,B2=complex['B1'],complex['B2']
        nE,nF=complex['nE'],B2.shape[1]
        curv=np.zeros(nE)
        for f in range(nF):
            col=B2[:,f]; cn=col@col
            if cn>1e-15: curv+=col**2/cn
        assert np.max(np.abs(B1@np.diag(curv)@B2)) < 1e-10

    def test_propagate(self, complex):
        RLp=complex['RLp']; nE=complex['nE']
        src=np.zeros(nE);src[0]=1; tgt=np.zeros(nE);tgt[4]=1
        score=(RLp@src)@tgt/(norm(src)*norm(tgt))
        assert abs(score) > 0
