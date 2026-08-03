"""rexgraph.nn._experimental: preserved-but-demoted optimizers.

``HodgeAdam`` and ``HodgeSGD`` are correct and verified: they decompose each parameter
tensor's gradient on the weight-neuron parameter graph and precondition the coordinated and
rotational parts separately. What benchmarking showed is that this buys nothing. On standard
feature-space models they TIE plain Adam, because the coordinated component engages only a few
percent of the gradient in the weight-neuron geometry, which is not the model's data complex.

The live path is ``factory.make_optimizer("auto", model, params)``. It routes to
``GreensCochain`` (in ``optim.py``) when the model exposes ``greens_groups()``, i.e. when its
parameters are cochains on a relational complex and the preconditioning happens in that
complex's own geometry; otherwise it returns plain Adam. That is where the structural gain
actually is: the model carrying relational structure, not a clever optimizer on an
unstructured one.

These two stay importable only so existing callers keep working. ``optim.py`` re-exports both
names, so ``rexgraph.nn.optim.HodgeAdam``, ``build_optimizer("hodge")`` and
``make_optimizer("hodge-arch", ...)`` all still resolve. They are off the ``rexgraph.nn``
top-level surface on purpose: the name sounds native, and models were reaching for it by name.
"""
from __future__ import annotations

try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:                                    # torch is an optional dep
    _HAS_TORCH = False


if _HAS_TORCH:

    class HodgeSGD(_torch.optim.Optimizer):
        """SGD whose per-matrix gradient is Hodge-decomposed on the bipartite parameter graph
        and recombined with separate gains and momenta for the coordinated (potential) and
        rotational parts. Reduces exactly to SGD/SGD-momentum when ``gamma_grad == gamma_curl``
        and the two momenta are equal.

        ``gamma_grad``, ``gamma_curl`` are gains on the potential (coordinated) and rotational
        parts; ``gamma_curl < 1`` damps oscillatory per-weight flow. ``momentum`` and
        ``curl_momentum`` are per-component momenta; ``curl_momentum`` defaults to ``momentum``,
        and a lower ``curl_momentum`` accelerates coordinated descent while limiting rotational
        velocity. ``min_side`` is the minimum number of elements to decompose; only scalars fall
        back to standard momentum SGD, every other rank (2-tensor, conv 4-tensor, 1-tensor bias,
        k-rex) is Hodge-decomposed via the general functional-ANOVA split. ``track`` accumulates
        mean pct_grad/pct_rot per step for ``hodge_report()``.

        Back-compat only: see the module docstring, and prefer ``make_optimizer("auto", ...)``.
        """

        def __init__(self, params, lr: float = 1e-2, gamma_grad: float = 1.0,
                     gamma_curl: float = 0.5, momentum: float = 0.0,
                     curl_momentum: float | None = None, weight_decay: float = 0.0,
                     min_side: int = 2, track: bool = True):
            if lr <= 0:
                raise ValueError("lr must be > 0")
            cm = momentum if curl_momentum is None else curl_momentum
            defaults = dict(lr=lr, gamma_grad=gamma_grad, gamma_curl=gamma_curl,
                            momentum=momentum, curl_momentum=cm, weight_decay=weight_decay,
                            min_side=min_side)
            super().__init__(params, defaults)
            self._track = track
            self._hist: dict[str, list[float]] = {"pct_grad": [], "pct_rot": []}

        @staticmethod
        def _decompose(G):
            # general functional-ANOVA / higher-order Hodge split for any k-tensor: potential =
            # Σ per-mode marginal means (main effects, inclusion-exclusion), residual = interaction.
            # Reduces to row+col-grand for a 2-tensor, mean vs fluctuation for a 1-tensor. Same
            # construction as HodgeAdam, verified against the core's hodge_matrix_decompose for 2D.
            if G.dim() <= 1:
                gm = G.mean()
                return gm.expand_as(G), G - gm
            k = G.dim(); gm = G.mean()
            potential = _torch.zeros_like(G)
            for i in range(k):
                dims = tuple(d for d in range(k) if d != i)
                potential = potential + G.mean(dim=dims, keepdim=True)
            potential = potential - (k - 1) * gm
            return potential, G - potential

        @_torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                lr = group["lr"]; gg = group["gamma_grad"]; gc = group["gamma_curl"]
                mom = group["momentum"]; cmom = group["curl_momentum"]
                wd = group["weight_decay"]; min_side = group["min_side"]
                eg_sum = 0.0; er_sum = 0.0; n_dec = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    G = p.grad
                    if wd:
                        G = G.add(p, alpha=wd)
                    st = self.state[p]
                    do_hodge = (G.dim() >= 1 and G.numel() >= max(min_side, 2))
                    if do_hodge:
                        # every rank decomposed (2-tensor, conv 4-tensor, 1-tensor bias, k-rex)
                        pot, rot = self._decompose(G)
                        if mom or cmom:
                            vp = st.get("v_pot");  vr = st.get("v_rot")
                            if vp is None:
                                vp = _torch.zeros_like(pot); vr = _torch.zeros_like(rot)
                                st["v_pot"] = vp; st["v_rot"] = vr
                            vp.mul_(mom).add_(pot); vr.mul_(cmom).add_(rot)
                            upd = gg * vp + gc * vr
                        else:
                            upd = gg * pot + gc * rot
                        if self._track:
                            eg_sum += float((pot * pot).sum()); er_sum += float((rot * rot).sum())
                            n_dec += 1
                    else:
                        # scalars only: standard momentum SGD
                        if mom:
                            v = st.get("v")
                            if v is None:
                                v = _torch.zeros_like(G); st["v"] = v
                            v.mul_(mom).add_(G); upd = v
                        else:
                            upd = G
                    p.add_(upd, alpha=-lr)
                if self._track and n_dec:
                    tot = eg_sum + er_sum
                    if tot > 0:
                        self._hist["pct_grad"].append(eg_sum / tot)
                        self._hist["pct_rot"].append(er_sum / tot)
            return loss

        def hodge_report(self) -> dict[str, list[float]]:
            """Per-step mean energy fractions (coordinated vs rotational) across all
            decomposed matrices. Feed to ``optim.save_hodge_trajectory`` for timeline tracking."""
            return {k: list(v) for k, v in self._hist.items()}


    class HodgeAdam(_torch.optim.Optimizer):
        """Adam whose adaptive moments are maintained independently within each Hodge component
        of a matrix gradient. Per-coordinate adaptive scaling operates inside the Hodge
        decomposition rather than across the flat matrix: the coordinated (potential) and
        rotational parts get their own first/second-moment estimates and their own effective
        step, then recombine with ``gamma_grad`` / ``gamma_curl`` gains. The grad/rotational
        split is a trackable, verified structure.

        Empirically this ties plain Adam on standard weight matrices (the coordinated component
        engages only a few percent of the gradient in the weight-neuron geometry, not the model's
        data complex); for relational-native models whose parameters are cochains, use
        ``optim.GreensCochain``, which preconditions in the complex's own geometry. Kept for
        back-compat; not a recommended default.

        Every rank is decomposed relationally: 2-tensors via the weighted bipartite Hodge flow,
        any other rank (1-tensor biases, 3/4-tensor conv kernels, k-rex) via the general
        functional-ANOVA / higher-order Hodge split (`_decompose`); only scalars fall back to plain
        Adam. At ``gamma_grad == gamma_curl == 1`` the recombination carries the full gradient; it
        is not identical to Adam because the √v normalization is per-component, not global."""

        def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                     gamma_grad: float = 1.0, gamma_curl: float = 1.0, weight_decay: float = 0.0,
                     min_side: int = 2, structure: str = "vector", heat_time: float = 0.3,
                     cheb_order: int = 12, max_side: int = 512, refresh: int = 16,
                     topk: int = 8, cg_iters: int = 12, blocks: int = 1, track: bool = True):
            if lr <= 0:
                raise ValueError("lr must be > 0")
            defaults = dict(lr=lr, betas=betas, eps=eps, gamma_grad=gamma_grad,
                            gamma_curl=gamma_curl, weight_decay=weight_decay, min_side=min_side,
                            structure=structure, heat_time=heat_time, cheb_order=cheb_order,
                            max_side=max_side, refresh=refresh, topk=topk, cg_iters=cg_iters,
                            blocks=blocks)
            super().__init__(params, defaults)
            self._track = track
            self._hist: dict[str, list[float]] = {"pct_grad": [], "pct_rot": []}

        @staticmethod
        def _decompose(G):
            """Functional-ANOVA / higher-order Hodge split of any k-tensor gradient, the general
            construction rather than a 2-tensor special case. The coordinated (potential) part is
            the order-1 additive structure: the sum of each mode's marginal mean (its "main
            effect"), combined by inclusion-exclusion; the residual is all higher-order interaction.
            This is the Hodge/ANOVA decomposition on the product complex, the same object
            statistics calls analysis-of-variance. It reduces to row+col-grand means for a
            2-tensor, to mean vs fluctuation for a 1-tensor, and generalizes to conv kernels and
            any k-rex."""
            if G.dim() <= 1:
                gm = G.mean()
                return gm.expand_as(G), G - gm
            k = G.dim(); gm = G.mean()
            potential = _torch.zeros_like(G)
            for i in range(k):
                dims = tuple(d for d in range(k) if d != i)     # marginal mean over every other mode
                potential = potential + G.mean(dim=dims, keepdim=True)
            potential = potential - (k - 1) * gm
            return potential, G - potential

        @staticmethod
        def _cg(matvec, b, iters=12, tol=1e-7):
            """Matrix-free conjugate gradient solve of L φ = b (no autograd needed - this runs
            inside the no_grad optimizer step)."""
            x = _torch.zeros_like(b); r = b - matvec(x); pdir = r.clone()
            rs = (r * r).sum()
            for _ in range(iters):
                Ap = matvec(pdir); alpha = rs / ((pdir * Ap).sum() + 1e-20)
                x = x + alpha * pdir; r = r - alpha * Ap
                rs_new = (r * r).sum()
                if rs_new < tol * tol:
                    break
                pdir = r + (rs_new / (rs + 1e-20)) * pdir; rs = rs_new
            return x

        def _vhodge_block(self, G, W, st, group, key):
            """Vector Hodge gradient-flow decomposition of one (sub)matrix block: coordinated =
            B₁ᵀφ (Green's-projected potential flow), residual = rest. Sparse weighted bipartite
            (edges = the block's own strong weights) with matrix-free CG; produces vectors, not a
            scalar filter. ANOVA is the uniform-complete-graph limit."""
            m, n = G.shape
            tc = st.get(key + "_t", 0)
            if st.get(key) is None or (tc % group["refresh"] == 0):
                k = min(group["topk"], n)
                vals, idx = W.abs().topk(k, dim=1)                      # strong edges per row
                ar = _torch.arange(m, device=G.device).repeat_interleave(k)
                st[key] = (ar, idx.reshape(-1), (vals.reshape(-1) + 1e-6), m + n)
            ar, cols, w, nV = st[key]
            st[key + "_t"] = tc + 1
            src = ar; tgt = cols + m                                     # bipartite: inputs after m
            g = G[ar, cols]                                             # edge-flow of the gradient

            def matvec(phi):                                            # L₀^w φ = B₁ diag(w) B₁ᵀ φ
                flow = w * (phi[tgt] - phi[src])
                out = _torch.zeros(nV, device=G.device, dtype=G.dtype)
                out.index_add_(0, tgt, flow); out.index_add_(0, src, -flow)
                return out

            wg = w * g
            div = _torch.zeros(nV, device=G.device, dtype=G.dtype)
            div.index_add_(0, tgt, wg); div.index_add_(0, src, -wg)
            div = div - div.mean()                                      # deflate the constant (β₀)
            phi = self._cg(matvec, div, iters=group["cg_iters"])
            phi = phi - phi.mean()
            coordinated = _torch.zeros_like(G)
            coordinated[ar, cols] = phi[tgt] - phi[src]                 # Hodge gradient component
            return coordinated

        def _vector_decompose(self, G, p, st, group):
            """Architecture-aware vector Hodge decomposition. ``blocks`` splits the output
            (row) dim into independent relational sub-complexes, e.g. one per attention head,
            since heads are independent subspaces the flat bipartite would wrongly entangle.
            blocks=1 (MLP/embedding/generic) is the plain weighted-bipartite vector split."""
            m, n = G.shape
            if max(m, n) > group["max_side"] or m < 3:
                return self._decompose(G)
            blocks = group.get("blocks", 1)
            if blocks < 1 or m % blocks != 0 or (m // blocks) < 3:
                blocks = 1
            Wd = p.detach()
            if blocks == 1:
                pot = self._vhodge_block(G, Wd, st, group, "_e0")
                return pot, G - pot
            bs = m // blocks
            pot = _torch.zeros_like(G)
            for b in range(blocks):                                    # one sub-complex per head
                sl = slice(b * bs, (b + 1) * bs)
                pot[sl] = self._vhodge_block(G[sl], Wd[sl], st, group, "_e%d" % b)
            return pot, G - pot

        def _topo_decompose(self, G, p, st, group):
            """Topology split: heat-smooth the gradient over the graph induced by the layer's
            own weights (|W| neuron similarity), via the eigen-free heat propagator. The
            coordinated part is the low-frequency/global component on the network structure
            (correlated neurons updated coherently), the residual is the local/high-frequency
            part. ANOVA is the degenerate complete-graph limit; falls back to it for oversized
            matrices."""
            m, n = G.shape
            if max(m, n) > group["max_side"] or m < 3:
                return self._decompose(G)
            from rexgraph.nn import rcf_torch as _R
            tc = st.get("_topo_t", 0)
            if st.get("_L") is None or (tc % group["refresh"] == 0):
                Wd = p.detach().abs()
                A = Wd @ Wd.t()                                   # [m,m] output-neuron similarity
                A = A - _torch.diag_embed(_torch.diagonal(A))
                L = _torch.diag_embed(A.sum(dim=1)) - A           # PSD neuron-graph Laplacian
                st["_L"] = L; st["_lam"] = _R.spectral_bound(L)
            st["_topo_t"] = tc + 1
            coordinated = _R.heat_apply(st["_L"], G, group["heat_time"],
                                        K=group["cheb_order"], lam_max=st["_lam"])
            return coordinated, G - coordinated

        @staticmethod
        def _adam_step(comp, st, key, b1, b2, eps):
            m = st.get(key + "_m"); v = st.get(key + "_v"); t = st.get(key + "_t", 0) + 1
            if m is None:
                m = _torch.zeros_like(comp); v = _torch.zeros_like(comp)
            m = b1 * m + (1 - b1) * comp
            v = b2 * v + (1 - b2) * comp * comp
            st[key + "_m"] = m; st[key + "_v"] = v; st[key + "_t"] = t
            mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
            return mhat / (vhat.sqrt() + eps)

        @_torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                lr = group["lr"]; b1, b2 = group["betas"]; eps = group["eps"]
                gg = group["gamma_grad"]; gc = group["gamma_curl"]
                wd = group["weight_decay"]; min_side = group["min_side"]
                eg_sum = 0.0; er_sum = 0.0; n_dec = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    G = p.grad
                    if wd:
                        G = G.add(p, alpha=wd)
                    st = self.state[p]
                    dec = None
                    if G.dim() == 2 and min(G.shape) >= min_side:
                        # weighted vector/topo Hodge on the 2-tensor's own bipartite complex
                        if group["structure"] == "vector":
                            dec = self._vector_decompose(G, p, st, group)
                        elif group["structure"] == "topo":
                            dec = self._topo_decompose(G, p, st, group)
                        else:
                            dec = self._decompose(G)
                    elif G.dim() >= 1 and G.numel() >= max(min_side, 2):
                        # any other rank (1-tensor bias, 3/4-tensor conv, k-rex): the general
                        # functional-ANOVA / higher-order Hodge split, no Adam fallback.
                        dec = self._decompose(G)
                    if dec is not None:
                        pot, rot = dec
                        u_pot = self._adam_step(pot, st, "pot", b1, b2, eps)
                        u_rot = self._adam_step(rot, st, "rot", b1, b2, eps)
                        upd = gg * u_pot + gc * u_rot
                        if self._track:
                            eg_sum += float((pot * pot).sum()); er_sum += float((rot * rot).sum())
                            n_dec += 1
                    else:
                        upd = self._adam_step(G, st, "full", b1, b2, eps)      # scalars / 1-element
                    p.add_(upd, alpha=-lr)
                if self._track and n_dec:
                    tot = eg_sum + er_sum
                    if tot > 0:
                        self._hist["pct_grad"].append(eg_sum / tot)
                        self._hist["pct_rot"].append(er_sum / tot)
            return loss

        def hodge_report(self) -> dict[str, list[float]]:
            return {k: list(v) for k, v in self._hist.items()}


else:

    class HodgeSGD:                                  # pragma: no cover (env without torch)
        def __init__(self, *a, **k):
            raise ImportError(
                "HodgeSGD requires PyTorch (an optional dependency). Install a torch build "
                "for your backend (CUDA/ROCm/CPU/MPS), or use the framework-agnostic core: "
                "rexgraph.nn.optim.hodge_matrix_precondition / hodge_flow_precondition.")

    class HodgeAdam:                                 # pragma: no cover (env without torch)
        def __init__(self, *a, **k):
            raise ImportError(
                "HodgeAdam requires PyTorch (an optional dependency). Install a torch build "
                "for your backend (CUDA/ROCm/CPU/MPS), or use the framework-agnostic core.")
