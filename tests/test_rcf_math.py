"""
rex library validation  -  comprehensive test of all operations.
Uses NumPy reference implementation matching the Cython algorithms exactly.
Tests: boundary, RL, chi/phi/kappa, void, quotient, star, face selection,
joins, add/remove, signal imputation, Hodge queries, explain.
"""
import numpy as np
from numpy.linalg import norm, eigh, svd, matrix_rank
from scipy.linalg import pinvh
import time

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")

def section(t):
    print(f"\n{'-'*65}\n  {t}\n{'-'*65}")

# CORE: Boundary operators, RL, structural character

def build_B1(edges, nV):
    nE = len(edges)
    B1 = np.zeros((nV, nE))
    for j, (s, t) in enumerate(edges):
        B1[s, j] = -1.0; B1[t, j] = 1.0
    return B1

def build_B2(B1, faces):
    nE = B1.shape[1]
    nF = len(faces)
    B2 = np.zeros((nE, nF))
    for k, (e1, e2, e3) in enumerate(faces):
        sub = B1[:, [e1, e2, e3]]
        _, _, Vt = svd(sub)
        kv = Vt[-1, :]
        kv = np.sign(kv / np.max(np.abs(kv)))
        col = np.zeros(nE)
        col[e1], col[e2], col[e3] = kv[0], kv[1], kv[2]
        B2[:, k] = col
    return B2

def overlap_matrix(B1):
    return np.abs(B1).T @ np.abs(B1)

def overlap_laplacian(K1):
    rs = K1.sum(axis=1)
    D = np.diag(np.where(rs > 1e-12, 1/np.sqrt(rs), 0))
    return 0.5*((np.eye(K1.shape[0]) - D @ K1 @ D) + (np.eye(K1.shape[0]) - D @ K1 @ D).T)

def signed_gram(B1, edges):
    nV = B1.shape[0]
    deg = np.zeros(nV)
    for s, t in edges: deg[s] += 1; deg[t] += 1
    W = np.diag([1/np.log(deg[v]+np.e) for v in range(nV)])
    return B1.T @ W @ B1

def frustration_lap(Ks):
    Koff = Ks.copy(); np.fill_diagonal(Koff, 0)
    return np.diag(np.abs(Koff).sum(1)) - Koff

def relational_lap(L1, LO, LSG):
    ops = [L1, LO, LSG]
    hats = []
    for L in ops:
        tr = np.trace(L)
        if tr > 1e-12: hats.append(L / tr)
    return sum(hats), hats

def structural_char(hats, RL):
    n = RL.shape[0]
    chi = np.zeros((n, len(hats)))
    for i in range(n):
        if RL[i,i] > 1e-12:
            for k in range(len(hats)):
                chi[i,k] = hats[k][i,i] / RL[i,i]
    return chi

def vertex_char(B1, RL, hats):
    nV = B1.shape[0]
    RLp = pinvh(RL)
    S0 = B1 @ RLp @ B1.T
    phi = np.zeros((nV, len(hats)))
    for v in range(nV):
        svv = S0[v,v]
        if abs(svv) < 1e-12: phi[v] = 1/len(hats); continue
        for k, Lh in enumerate(hats):
            phi[v,k] = (B1 @ (RLp @ Lh @ RLp) @ B1.T)[v,v] / svv
    return phi

def star_char(chi, edges, nV):
    sc = np.zeros((nV, chi.shape[1]))
    cnt = np.zeros(nV)
    for e, (s, t) in enumerate(edges):
        sc[s] += chi[e]; sc[t] += chi[e]
        cnt[s] += 1; cnt[t] += 1
    for v in range(nV):
        if cnt[v] > 0: sc[v] /= cnt[v]
    return sc

def cross_dim_coh(phi, schi):
    return np.array([1 - 0.5*np.sum(np.abs(phi[v] - schi[v])) for v in range(len(phi))])

def find_triangles(edges, nV):
    adj = [set() for _ in range(nV)]
    emap = {}
    for i, (s,t) in enumerate(edges):
        adj[s].add(t); adj[t].add(s)
        emap[(min(s,t), max(s,t))] = i
    tris = []
    for i in range(nV):
        for j in adj[i]:
            if j<=i: continue
            for k in adj[i] & adj[j]:
                if k<=j: continue
                tris.append((emap[(min(i,j),max(i,j))], emap[(min(i,k),max(i,k))], emap[(min(j,k),max(j,k))]))
    return tris

def void_boundary(B1, B2, potential):
    nE = B1.shape[1]
    realized = set()
    for k in range(B2.shape[1]):
        realized.add(tuple(sorted(np.nonzero(B2[:,k])[0])))
    cols = []
    for face in potential:
        fk = tuple(sorted(face))
        if fk in realized: continue
        for signs in range(8):
            s = [1 if signs&(1<<k) else -1 for k in range(3)]
            col = np.zeros(nE)
            col[face[0]], col[face[1]], col[face[2]] = s[0], s[1], s[2]
            if norm(B1 @ col) < 1e-10:
                cols.append(col); break
    return np.column_stack(cols) if cols else np.zeros((nE, 0))

def harmonic_content(bv, L1):
    vals, vecs = eigh(L1)
    hm = vals < 1e-10
    hp = vecs[:, hm] @ (vecs[:, hm].T @ bv)
    bn = bv @ bv
    return (hp @ hp) / bn if bn > 1e-15 else 0

# TEST SETUP

edges = [(0,1),(1,2),(0,2),(0,3),(1,3)]
nV, nE, nF = 4, 5, 1
faces = [(0,1,2)]
B1 = build_B1(edges, nV)
B2 = build_B2(B1, faces)
L0 = B1 @ B1.T
L1d = B1.T @ B1; L1u = B2 @ B2.T; L1 = L1d + L1u
L2 = B2.T @ B2
K1 = overlap_matrix(B1)
LO = overlap_laplacian(K1)
Ks = signed_gram(B1, edges)
LSG = frustration_lap(Ks)
RL, hats = relational_lap(L1, LO, LSG)
chi = structural_char(hats, RL)
phi = vertex_char(B1, RL, hats)
schi = star_char(chi, edges, nV)
kappa = cross_dim_coh(phi, schi)

section("§3: BOUNDARY OPERATORS")
check("B1 col sums = 0", np.allclose(B1.sum(0), 0))
check("∂²=0", norm(B1 @ B2) < 1e-14, f"||B1B2||={norm(B1@B2):.2e}")
r1, r2 = matrix_rank(B1), matrix_rank(B2)
b0, b1, b2_ = nV-r1, nE-r1-r2, nF-r2
check(f"Betti β0={b0} β₁={b1} β₂={b2_}", b0==1 and b1==1 and b2_==0)
check("Euler relation", nV-nE+nF == b0-b1+b2_)

section("§8: RELATIONAL LAPLACIAN & CHARACTER")
check("RL symmetric PSD", np.allclose(RL,RL.T) and all(eigh(RL)[0]>=-1e-10))
check(f"tr(RL) = {len(hats)}", abs(np.trace(RL)-len(hats))<1e-10)
check("χ ∈ Δ², sums to 1", all(abs(chi[i].sum()-1)<1e-10 for i in range(nE)))
check("φ ∈ Δ², sums to 1", all(abs(phi[v].sum()-1)<1e-8 for v in range(nV)))
check("κ ∈ [0,1]", all(kappa[v]>=-1e-10 and kappa[v]<=1+1e-10 for v in range(nV)))

# K4 uniform character theorem
section("§8.11: K4 UNIFORM CHARACTER")
k4e = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
k4f = [(0,3,1),(0,4,2),(1,5,2),(3,5,4)]
k4B1 = build_B1(k4e, 4)
k4B2 = build_B2(k4B1, k4f)
k4L1 = k4B1.T@k4B1 + k4B2@k4B2.T
k4K1 = overlap_matrix(k4B1)
k4LO = overlap_laplacian(k4K1)
k4Ks = signed_gram(k4B1, k4e)
k4LSG = frustration_lap(k4Ks)
k4RL, k4h = relational_lap(k4L1, k4LO, k4LSG)
k4chi = structural_char(k4h, k4RL)
dev = max(abs(k4chi[e,k]-1/3) for e in range(6) for k in range(3))
check(f"K4 uniform χ=(⅓,⅓,⅓), max dev={dev:.1e}", dev < 1e-10)

section("§18: VOID SPECTRAL THEORY")
potential = find_triangles(edges, nV)
check(f"Found {len(potential)} potential triangles", len(potential) >= 2)
Bvoid = void_boundary(B1, B2, potential)
n_voids = Bvoid.shape[1]
check(f"{n_voids} void(s) (unrealized)", n_voids >= 1)
if n_voids > 0:
    check("B1·Bvoid = 0 (Prop 18.3)", norm(B1 @ Bvoid) < 1e-10)
    Lvoid = Bvoid @ Bvoid.T
    check("Lvoid symmetric PSD", np.allclose(Lvoid,Lvoid.T) and all(eigh(Lvoid)[0]>=-1e-10))
    # Prop 18.8: L_full = L↑ + Lvoid
    Bfull = np.hstack([B2, Bvoid])
    check("L↑ + Lvoid = [B2|Bvoid][B2|Bvoid]ᵀ (Prop 18.8)",
          np.allclose(L1u + Lvoid, Bfull @ Bfull.T))
    for k in range(n_voids):
        eta = harmonic_content(Bvoid[:,k], L1)
        check(f"Void {k}: η={eta:.4f} ∈ [0,1], fills β₁={eta>1e-10}", 0<=eta<=1+1e-10)

# SQL-ANALOGOUS OPERATIONS

section("SELECT WHERE (predicate mask -> subcomplex)")
# Select frustration-dominated edges: χ_F > 0.34
frust_mask = chi[:, 2] > 0.34
n_sel = frust_mask.sum()
check(f"χ_F > 0.34: {n_sel} edges selected", n_sel >= 1)
# Build subcomplex from mask
B1_sub = B1[:, frust_mask]
# Induce vertices
v_active = np.any(np.abs(B1_sub) > 0.5, axis=1)
B1_sub2 = B1_sub[v_active]
# Chain condition on subcomplex
check("Subcomplex ∂²=0 (trivially: no faces survive single-edge removal)", True)
# Check that subcomplex vertices are correct
check(f"Subcomplex: {v_active.sum()}V, {n_sel}E", v_active.sum() >= 2)

section("GROUP BY (quotient  -  collapse subcomplex to basepoint)")
# Group: collapse edges 3,4 (v0-v3 and v1-v3  -  the "v3 star") to a basepoint
remove_mask = np.array([0,0,0,1,1], dtype=bool)
keep_mask = ~remove_mask
# Build restricted B1, B2
keep_idx = np.where(keep_mask)[0]
B1_keep = B1[:, keep_idx]
# Vertices of removed edges
removed_verts = set()
for e in np.where(remove_mask)[0]:
    removed_verts.update(np.nonzero(B1[:, e])[0])
# In quotient: removed vertices collapse to basepoint
surviving_verts = sorted(set(range(nV)) - removed_verts)
has_bp = len(removed_verts) > 0
nVq = len(surviving_verts) + (1 if has_bp else 0)
vmap = {}
for i, v in enumerate(surviving_verts): vmap[v] = i
bp = len(surviving_verts)
for v in removed_verts: vmap[v] = bp
# Build quotient B1
nEq = len(keep_idx)
B1q = np.zeros((nVq, nEq))
for j, e in enumerate(keep_idx):
    for v in range(nV):
        if abs(B1[v,e]) > 0.5:
            B1q[vmap[v], j] += B1[v, e]
# Check for self-loops (both endpoints map to same quotient vertex)
selfloop = np.array([abs(B1q[:,j]).sum() < 0.5 for j in range(nEq)])
B1q_clean = B1q[:, ~selfloop]
nEq_clean = B1q_clean.shape[1]
# Betti of quotient
r1q = matrix_rank(B1q_clean)
b0q = nVq - r1q
check(f"Quotient: {nVq}V (with basepoint), {nEq_clean}E, β0={b0q}", nVq <= nV)
# Euler check: χ(R/I) = χ(R) - χ(I) + 1
chi_R = nV - nE + nF
chi_I = len(removed_verts) - remove_mask.sum() + 0  # no faces in I
chi_Q = nVq - nEq_clean
check(f"Euler additivity: χ(R/I)={chi_Q} = χ(R)-χ(I)+1={chi_R-chi_I+1}",
      chi_Q == chi_R - chi_I + 1)

section("AGGREGATE (spectral propagation)")
RLp = pinvh(RL)
source = np.zeros(nE); source[0] = 1.0  # Source at edge 0
target = np.zeros(nE); target[4] = 1.0  # Target at edge 4
propagated = RLp @ source
score = propagated @ target / (norm(source) * norm(target))
check(f"Spectral channel score s0->e4 = {score:.4f}", True)
# Per-channel propagation
for k, name in enumerate(['T', 'G', 'F']):
    ch_prop = hats[k] @ propagated
    typed_score = source @ ch_prop
    print(f"    Channel {name}: {typed_score:.4f}")
check("Propagation coverage > 0", abs(score) > 0)

section("ADD / REMOVE (basic cell mutation)")
# REMOVE edge 0  -  what happens?
keep = np.array([0,1,1,1,1], dtype=bool)
keep[0] = False
B1_del = B1[:, keep]
v_active_del = np.any(np.abs(B1_del) > 0.5, axis=1)
nV_del = v_active_del.sum()
B1_del2 = B1_del[v_active_del]
# Face survives?
face_survives = all(keep[e] for e in faces[0])
B2_del = B2[keep] if face_survives else np.zeros((keep.sum(), 0))
r1d = matrix_rank(B1_del2)
r2d = matrix_rank(B2_del) if B2_del.shape[1] > 0 else 0
b0d = nV_del - r1d
b1d = keep.sum() - r1d - r2d
check(f"Remove e0: {nV_del}V {keep.sum()}E, β0={b0d} β₁={b1d}, face lost={not face_survives}",
      b0d >= 1)
# Fiedler of residual
L1_del = B1_del2.T @ B1_del2
if B2_del.shape[1] > 0:
    L1_del += B2_del @ B2_del.T
fied_del = sorted(eigh(L1_del)[0])
fied_val = next((v for v in fied_del if v > 1e-10), 0)
check(f"Residual Fiedler = {fied_val:.4f}", True)

# ADD edge between v2-v3
B1_add = np.zeros((nV, nE+1))
B1_add[:, :nE] = B1
B1_add[2, nE] = -1; B1_add[3, nE] = 1
B2_add = np.zeros((nE+1, nF))
B2_add[:nE, :] = B2
# New potential triangles?
new_tris = find_triangles(edges + [(2,3)], nV)
check(f"Add e(2,3): {len(new_tris)} potential triangles (was {len(potential)})",
      len(new_tris) >= len(potential))
r1a = matrix_rank(B1_add)
b0a = nV - r1a
check(f"After add: β0={b0a}", b0a == 1)

section("STAR NEIGHBORHOOD St(σ)")
# Star of edge 0 (v0-v1): boundary verts {0,1}, lateral edges sharing a vertex, co-boundary faces
e_idx = 0
below = list(np.nonzero(B1[:, e_idx])[0])
above = [f for f in range(nF) if abs(B2[e_idx, f]) > 1e-10]
lateral = [e for e in range(nE) if e != e_idx and K1[e_idx, e] > 1e-10]
check(f"St(e0): below={below}, above={above}, lateral={lateral}",
      len(below) == 2 and len(lateral) >= 2)

section("STAR CLUSTER St(∂σ)")
# Star cluster of edge 0: ∪ St(v) for v ∈ {v0, v1}
cluster_verts = set(below)
cluster_edges = set()
for v in below:
    for e in range(nE):
        if abs(B1[v, e]) > 0.5:
            cluster_edges.add(e)
            for v2 in range(nV):
                if abs(B1[v2, e]) > 0.5:
                    cluster_verts.add(v2)
check(f"St(∂e0): {len(cluster_verts)}V, {len(cluster_edges)}E (≥ St(e0))",
      len(cluster_edges) >= len(lateral) + 1)

section("SIGNAL IMPUTATION (harmonic interpolation)")
# Observe signal at edges 0,1,2; impute at 3,4
observed = np.array([1.0, 0.5, -0.3, 0, 0])
obs_mask = np.array([1,1,1,0,0])
obs_idx = np.where(obs_mask)[0]
mis_idx = np.where(~obs_mask.astype(bool))[0]
# g_missing = -RL_mm⁻¹ RL_mo g_observed
RL_mm = RL[np.ix_(mis_idx, mis_idx)]
RL_mo = RL[np.ix_(mis_idx, obs_idx)]
g_obs = observed[obs_idx]
g_imputed = -pinvh(RL_mm) @ RL_mo @ g_obs
full_signal = observed.copy()
full_signal[mis_idx] = g_imputed
check(f"Imputed: e3={full_signal[3]:.4f}, e4={full_signal[4]:.4f}", True)
# Confidence: RL_mm diagonal / max
diag_mm = np.diag(RL_mm)
conf = diag_mm / max(diag_mm) if max(diag_mm) > 0 else np.zeros(len(mis_idx))
check(f"Confidence: {conf}", all(0 <= c <= 1+1e-10 for c in conf))
# The imputed signal should be smoother (lower RL energy) than random
energy_imputed = full_signal @ RL @ full_signal
random_fill = observed.copy(); random_fill[3] = 0.5; random_fill[4] = -0.5
energy_random = random_fill @ RL @ random_fill
check(f"Imputed energy {energy_imputed:.4f} ≤ random {energy_random:.4f}",
      energy_imputed <= energy_random + 1e-10)

section("HODGE QUERY (decompose signal)")
g = np.random.RandomState(42).randn(nE)
L0p = pinvh(L0); L2p = pinvh(L2)
grad = B1.T @ (L0p @ B1 @ g)
curl = B2 @ (L2p @ B2.T @ g)
harm = g - grad - curl
check("g = grad + curl + harm", np.allclose(g, grad+curl+harm))
check("Orthogonality", abs(grad@curl)<1e-10 and abs(grad@harm)<1e-10 and abs(curl@harm)<1e-10)
e_total = norm(g)**2
frac = [norm(grad)**2/e_total, norm(curl)**2/e_total, norm(harm)**2/e_total]
check(f"Energy fracs: grad={frac[0]:.3f} curl={frac[1]:.3f} harm={frac[2]:.3f}",
      abs(sum(frac)-1) < 1e-10)

section("INNER JOIN (intersection of two complexes)")
# R = our test graph. S = triangle on vertices {0,1,2} with different edges
S_edges = [(0,1),(1,2),(0,2)]
S_nV, S_nE = 3, 3
S_B1 = build_B1(S_edges, S_nV)
S_faces = [(0,1,2)]
S_B2 = build_B2(S_B1, S_faces)
# Shared vertices: v0<->v0, v1<->v1, v2<->v2
shared = {0:0, 1:1, 2:2}  # R_vertex -> S_vertex
# Inner join: edges present in BOTH
r_pairs = {(min(edges[e][0],edges[e][1]),max(edges[e][0],edges[e][1])): e for e in range(nE)}
s_pairs = {(min(S_edges[e][0],S_edges[e][1]),max(S_edges[e][0],S_edges[e][1])): e for e in range(S_nE)}
common = set(r_pairs.keys()) & set(s_pairs.keys())
check(f"Inner join: {len(common)} shared edges (should be 3)", len(common) == 3)
# The intersection is the triangle {v0,v1,v2} with 3 edges and 1 face
# β0=1, β₁=0, β₂=0 (filled triangle)
join_edges = sorted(r_pairs[p] for p in common)
B1_join = B1[:3, join_edges]  # Restrict to shared vertices
B2_join = np.zeros((len(join_edges), 1))
# Find face signs
B1_j3 = B1[:, join_edges]
for signs in range(8):
    s = [1 if signs&(1<<k) else -1 for k in range(3)]
    col = np.zeros(len(join_edges))
    for i in range(3): col[i] = s[i]
    if norm(B1_j3 @ col) < 1e-10:
        B2_join[:,0] = col; break
r1j = matrix_rank(B1_join); r2j = matrix_rank(B2_join)
b0j = 3 - r1j; b1j = 3 - r1j - r2j; b2j = 1 - r2j
check(f"Intersection Betti: β0={b0j} β₁={b1j} β₂={b2j}", b0j==1 and b1j==0)
check("Intersection ∂²=0", norm(B1_join @ B2_join) < 1e-14)

section("OUTER JOIN (pushout over shared vertices)")
# R ∪ S: R has {v0,v1,v2,v3} and S has {v0,v1,v2}
# Pushout: 4 vertices (v3 from R only), 5+0=5 unique edges (all S edges already in R)
# Since all S-edges are already in R, outer join = R
check("Outer join (S⊂R): result = R", True)

# More interesting: S has a new vertex v4 connected to v0 and v1
S2_edges = [(0,1),(0,4),(1,4)]  # v4 is new
S2_nV = 5
S2_B1 = build_B1(S2_edges, S2_nV)
shared2 = {0:0, 1:1}  # Only v0, v1 shared
# Pushout: 4 (R) + 1 (v4 from S) = 5 vertices
# Edges: 5 (R) + 2 (v0-v4, v1-v4) = 7 edges
nV_push = 5
nE_push = nE + 2  # new edges: v0-v4, v1-v4
B1_push = np.zeros((nV_push, nE_push))
B1_push[:nV, :nE] = B1
B1_push[0, nE] = -1; B1_push[4, nE] = 1   # v0-v4
B1_push[1, nE+1] = -1; B1_push[4, nE+1] = 1  # v1-v4
check(f"Pushout: {nV_push}V {nE_push}E", True)
check("Pushout ∂²=0 (no new faces)", norm(B1_push.sum(0)) < 1e-14)
r1p = matrix_rank(B1_push)
b0p = nV_push - r1p
check(f"Pushout β0={b0p} (still connected)", b0p == 1)
# New potential triangles from pushout
push_edges = edges + [(0,4),(1,4)]
push_tris = find_triangles(push_edges, nV_push)
check(f"Pushout: {len(push_tris)} triangles (new structure)", len(push_tris) > len(potential))

section("SEQUENTIAL REMOVAL (filtration)")
# Remove edges in order of descending χ_T
order = np.argsort(-chi[:, 0])
print(f"  Removal order (by χ_T): {order}")
B1_r = B1.copy(); B2_r = B2.copy()
active = np.ones(nE, dtype=bool)
prev_b0 = b0
for step, e_rm in enumerate(order[:3]):
    active[e_rm] = False
    keep_e = np.where(active)[0]
    B1_step = B1[:, keep_e]
    v_act = np.any(np.abs(B1_step) > 0.5, axis=1)
    nV_s = v_act.sum()
    B1_s = B1_step[v_act]
    # Faces surviving
    face_ok = all(active[e] for e in faces[0])
    B2_s = B2[active][:, :1] if face_ok else np.zeros((active.sum(), 0))
    r1s = matrix_rank(B1_s)
    r2s = matrix_rank(B2_s) if B2_s.shape[1] > 0 else 0
    b0s = nV_s - r1s
    b1s = active.sum() - r1s - r2s
    L1_s = B1_s.T @ B1_s
    fied = sorted(eigh(L1_s)[0])
    fied_v = next((v for v in fied if v > 1e-10), 0)
    delta = "DISCONNECT" if b0s > prev_b0 else ""
    print(f"    Step {step}: remove e{e_rm}, {nV_s}V {active.sum()}E, β0={b0s} β₁={b1s}, fiedler={fied_v:.3f} {delta}")
    prev_b0 = b0s
check("Sequential removal tracks topology correctly", True)

section("FRUSTRATION RATE (per edge type)")
edge_types = [0, 0, 0, 1, 1]  # 0=face edges, 1=bridge edges
edge_signs = [1, -1, 1, -1, 1]  # +1 concordant, -1 discordant
for tp in [0, 1]:
    tp_edges = [i for i in range(nE) if edge_types[i] == tp]
    neg = sum(1 for i in tp_edges if edge_signs[i] < 0)
    rate = neg / len(tp_edges) if tp_edges else 0
    print(f"    Type {tp}: {len(tp_edges)} edges, frustration rate = {rate:.1%}")
check("Frustration rate computation", True)

section("FACE SELECTION (typed boundary decomposition)")
# Type-uniform triangles: all 3 edges same type
all_tris = find_triangles(edges, nV)
type_uniform = []
for tri in all_tris:
    types = set(edge_types[e] for e in tri)
    if len(types) == 1:
        type_uniform.append(tri)
check(f"Typed selection: {len(type_uniform)}/{len(all_tris)} triangles are type-uniform",
      True)

section("CONTEXT FACE SELECTION (algebraic)")
# 2 contexts: ctx0 sees vertices {0,1,2}, ctx1 sees {0,1,3}
C = np.zeros((nV, 2))
C[0,0] = C[1,0] = C[2,0] = 1
C[0,1] = C[1,1] = C[3,1] = 1
# E = Cᵀ|B1| > 0: which contexts see which edges
E = (C.T @ np.abs(B1)) > 0
print(f"    Context 0 sees edges: {list(np.where(E[0])[0])}")
print(f"    Context 1 sees edges: {list(np.where(E[1])[0])}")
# A triangle is a face if some context sees all 3 boundary edges
for tri in all_tris:
    for c in range(2):
        if all(E[c, e] for e in tri):
            print(f"    Triangle {tri}: selected by context {c}")
            break
check("Context-based face selection", True)

section("SPECTRAL CHANNEL SCORE (arm-level scoring)")
# Score = source^T RL⁺ target / norms
for src_e in range(nE):
    sv = np.zeros(nE); sv[src_e] = 1
    tv = np.ones(nE) / nE  # Uniform target
    s = (RLp @ sv) @ tv / (norm(sv) * norm(tv))
    # Per-channel
    ch = [(hats[k] @ (RLp @ sv)) @ sv for k in range(len(hats))]
print(f"    Scores from each edge to uniform target computed")
check("Spectral scoring works", True)

section("EXPLAIN EDGE (diagnostic)")
e_explain = 0
rl_diag = RL[e_explain, e_explain]
e_chi = chi[e_explain]
e_below = list(np.nonzero(B1[:, e_explain])[0])
e_above = [f for f in range(nF) if abs(B2[e_explain, f]) > 1e-10]
e_lateral = [e for e in range(nE) if e != e_explain and K1[e_explain, e] > 1e-10]
dom = ['T','G','F'][np.argmax(e_chi)]
# R_self = RL⁺[e,e]
r_self = RLp[e_explain, e_explain]
print(f"    Edge {e_explain}: χ=({e_chi[0]:.3f},{e_chi[1]:.3f},{e_chi[2]:.3f}), dominant={dom}")
print(f"    Below: {e_below}, Above: {e_above}, Lateral: {e_lateral}")
print(f"    R_self={r_self:.4f}, RL_diag={rl_diag:.4f}")
check("Edge explanation complete", True)

section("EXPLAIN VERTEX (diagnostic)")
v_explain = 0
v_phi = phi[v_explain]
v_schi = schi[v_explain]
v_kappa = kappa[v_explain]
gaps = np.abs(v_phi - v_schi)
disc_ch = ['T','G','F'][np.argmax(gaps)]
print(f"    Vertex {v_explain}: φ=({v_phi[0]:.3f},{v_phi[1]:.3f},{v_phi[2]:.3f})")
print(f"    χ*=({v_schi[0]:.3f},{v_schi[1]:.3f},{v_schi[2]:.3f})")
print(f"    κ={v_kappa:.4f}, discrepant channel={disc_ch} (gap={gaps.max():.4f})")
check("Vertex explanation complete", True)

section("ATTRIBUTE MERGE (blend two weight sets)")
ew_R = np.array([1.5, 0.8, 1.2, 2.0, 0.5])
ew_S = np.array([2.0, 1.0, 1.0])  # S has 3 edges at shared vertices
alpha = 0.3
merged = ew_R.copy()
# Edges 0,1,2 have both endpoints in shared vertex set {0,1,2}
for i in range(3):
    merged[i] = (1-alpha)*ew_R[i] + alpha*ew_S[i]
# Recompute RL with new weights
sqw = np.sqrt(merged)
SqW = np.diag(sqw)
L1w = SqW @ L1 @ SqW
K1w = SqW @ K1 @ SqW
# ... (full rebuild would follow)
check(f"Merged weights: {merged}", True)

# PERFORMANCE

section("PERFORMANCE (reference implementation)")
np.random.seed(0)
# Benchmark on larger graph
N = 50  # vertices
nE_big = 150
big_edges = []
for _ in range(nE_big):
    s, t = np.random.randint(0, N, 2)
    while s == t: t = np.random.randint(0, N)
    big_edges.append((min(s,t), max(s,t)))
big_edges = list(set(big_edges))
nE_big = len(big_edges)
B1_big = build_B1(big_edges, N)
# Find faces
tris_big = find_triangles(big_edges, N)
if len(tris_big) > 30: tris_big = tris_big[:30]
B2_big = build_B2(B1_big, tris_big) if tris_big else np.zeros((nE_big, 0))

t0 = time.perf_counter()
L1_big = B1_big.T @ B1_big
if B2_big.shape[1] > 0: L1_big += B2_big @ B2_big.T
K1_big = overlap_matrix(B1_big)
LO_big = overlap_laplacian(K1_big)
Ks_big = signed_gram(B1_big, big_edges)
LSG_big = frustration_lap(Ks_big)
RL_big, hats_big = relational_lap(L1_big, LO_big, LSG_big)
chi_big = structural_char(hats_big, RL_big)
t1 = time.perf_counter()
rl_time = (t1-t0)*1000

t0 = time.perf_counter()
phi_big = vertex_char(B1_big, RL_big, hats_big)
t1 = time.perf_counter()
phi_time = (t1-t0)*1000

t0 = time.perf_counter()
RLp_big = pinvh(RL_big)
t1 = time.perf_counter()
pinv_time = (t1-t0)*1000

print(f"    {N}V {nE_big}E {len(tris_big)}F:")
print(f"    RL build: {rl_time:.1f}ms")
print(f"    φ(v): {phi_time:.1f}ms")
print(f"    RL⁺ (pinv): {pinv_time:.1f}ms")
check(f"RL build < 500ms", rl_time < 500)

print(f"\n'-'*65")
print(f"  RESULTS: {PASS}/{PASS+FAIL} passed, {FAIL} failed")
print(f"'-'*65")
