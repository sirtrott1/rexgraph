"""
rexgraph.nn.optim: Hodge-structured optimization, gradient descent on the Helmholtz-Hodge
structure of the gradient field, not the coordinate-wise view of SGD/Adam.

A gradient on a weight matrix W (out × in) is a flow on the complete bipartite parameter
graph K_{m,n}: out-neurons ∪ in-neurons are vertices, each weight is an edge, and ∂L/∂W[i,j]
is the flow on that edge. The Hodge decomposition splits that flow into orthogonal parts:

  * potential (gradient, im B1ᵀ): the part explained by a per-neuron scalar potential; the
    coordinated descent every neuron agrees on. Closed form on K_{m,n}:
        potential[i,j] = rowmean_i + colmean_j - grandmean          (two-way ANOVA main effects)
  * rotational (curl + harmonic, ker B1): the interaction/per-weight residual; the rotational
    flow that causes oscillation and that momentum/Adam fight coordinate-wise.

Mixing the components (γ_grad·potential + γ_curl·rotational) is a structural preconditioner:
γ_curl < 1 damps the rotational flow while preserving the coordinated descent. γ_grad =
γ_curl = 1 reproduces plain SGD exactly. It is O(mn) (the closed form is row/column means)
and equals the Hodge grad-projection the compiled ``rex.hodge`` kernel computes on K_{m,n}
(see tests).

Two layers:
  * framework-agnostic numpy core (``hodge_matrix_*``, ``hodge_flow_*``): usable directly,
    and where the math is verified against rexgraph's kernels. The structured/general path
    (arbitrary parameter graph, real harmonic component) goes through
    ``hodge_flow_precondition`` and the compiled core.
  * ``torch.optim.Optimizer`` bindings (guarded; torch is optional). ``GreensCochain`` is the
    live one: Green's-function preconditioning for models whose parameters are cochains on a
    complex. ``HodgeAdam`` / ``HodgeSGD`` moved to ``rexgraph.nn._experimental`` (back-compat;
    they tie plain Adam on standard feature-space models) and are re-exported here.

The numpy core is pure BLAS; the torch binding runs on whatever backend torch is built for
(CUDA/ROCm/CPU/MPS). Training dynamics can be logged as a vector corpus via
``save_hodge_trajectory`` (the same ``rexgraph.io`` path used for embeddings), so a run's
grad/rotational balance is a trackable timeline.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# back-compat re-export: HodgeAdam / HodgeSGD were demoted to _experimental, but build_optimizer
# and factory.make_optimizer reach them by name here, as do external callers. _experimental
# carries its own torch guard and its own no-torch stubs, so this covers both branches below.
from rexgraph.nn._experimental import HodgeAdam, HodgeSGD  # noqa: F401

# framework-agnostic core

def hodge_matrix_decompose(G) -> tuple:
    """Analytic Helmholtz-Hodge decomposition of a 2D gradient matrix on the complete
    bipartite parameter graph K_{m,n}. Returns ``(potential, rotational, info)``:

        potential[i,j] = rowmean_i + colmean_j - grandmean   (= im B1ᵀ, coordinated descent)
        rotational     = G - potential                        (= ker B1, interaction/rotational)

    Exact projection (potential ⟂ rotational in the edge inner product), O(mn)."""
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"hodge_matrix_decompose expects a 2D matrix, got shape {G.shape!r}")
    rm = G.mean(axis=1, keepdims=True)          # per out-neuron potential
    cm = G.mean(axis=0, keepdims=True)          # per in-neuron potential
    gm = float(G.mean())
    potential = rm + cm - gm
    rotational = G - potential
    e_pot = float(np.sum(potential * potential))
    e_rot = float(np.sum(rotational * rotational))
    tot = e_pot + e_rot
    denom = tot if tot > 0 else 1.0
    info = {"pct_grad": e_pot / denom, "pct_rot": e_rot / denom,
            "energy": tot, "grad_norm": e_pot ** 0.5, "rot_norm": e_rot ** 0.5}
    return potential, rotational, info


def hodge_matrix_precondition(G, gamma_grad: float = 1.0, gamma_rot: float = 1.0) -> tuple:
    """Hodge-preconditioned gradient: ``gamma_grad·potential + gamma_rot·rotational``.
    ``(1.0, 1.0)`` returns G unchanged (plain SGD). Returns ``(update, info)``."""
    potential, rotational, info = hodge_matrix_decompose(G)
    return gamma_grad * potential + gamma_rot * rotational, info


def hodge_flow_decompose(rex, flow) -> tuple:
    """Full grad/curl/harmonic decomposition of an edge ``flow`` on an arbitrary relational
    complex, via the compiled ``rex.hodge`` kernel. Returns ``(grad, curl, harm)``. Use this
    (not the matrix closed form) when the parameter graph is not complete bipartite (a
    sparsified neuron-similarity graph, conv locality, an ontology) where the harmonic
    (topologically protected) component is nonzero and carries signal."""
    grad, curl, harm = rex.hodge(np.ascontiguousarray(flow, dtype=np.float64))
    return grad, curl, harm


def hodge_flow_precondition(rex, flow, gamma_grad: float = 1.0, gamma_curl: float = 1.0,
                            gamma_harm: float = 1.0) -> tuple:
    """Recombine an edge flow's Hodge components with per-component gains on a general
    complex. Returns ``(update, info)`` where info carries the energy fractions."""
    grad, curl, harm = hodge_flow_decompose(rex, flow)
    e = [float(np.sum(c * c)) for c in (grad, curl, harm)]
    tot = sum(e) or 1.0
    info = {"pct_grad": e[0] / tot, "pct_curl": e[1] / tot, "pct_harm": e[2] / tot,
            "energy": sum(e)}
    return gamma_grad * grad + gamma_curl * curl + gamma_harm * harm, info


# training-dynamics as a corpus

def save_hodge_trajectory(report: dict[str, list[float]], path: str, *,
                          optimizer: str = "HodgeSGD", **meta) -> str:
    """Persist a per-step Hodge trajectory (pct_grad / pct_rot / ...) as a labeled vector
    corpus through the same ``rexgraph.io`` path used for embeddings, so a run's
    coordinated-vs-rotational gradient balance is a trackable timeline. Returns the path."""
    from rexgraph.io import save_vectors  # direct import: the substrate never imports upward
    keys = [k for k, v in report.items() if isinstance(v, list) and v]
    if not keys:
        raise ValueError("empty trajectory report")
    n = min(len(report[k]) for k in keys)
    matrix = np.array([[float(report[k][i]) for k in keys] for i in range(n)], dtype=np.float32)
    labels = np.array(["step_%d" % i for i in range(n)])
    md = {"kind": "hodge_trajectory", "source": str(optimizer)}
    md.update({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in meta.items()})
    return str(save_vectors(matrix, labels, path, feature_names=keys, metadata=md))


# torch.optim binding

try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:                                    # torch is an optional dep
    _HAS_TORCH = False


if _HAS_TORCH:

    class GreensCochain(_torch.optim.Optimizer):
        """Adam whose gradient is preconditioned by the Green's function of a relational complex,
        for models whose parameters are COCHAINS on that complex (a value per cell).

        For a param group carrying a complex operator (`green_adj`, a sparse normalized adjacency
        of shape [n_cells, n_cells]), each parameter's gradient (first dim = n_cells) is whitened in
        the complex geometry: solve (I + t L) x = g with L = I - A_hat by matrix-free CG, returning
        the low-pass component x (Green's-smoothed, for a homophilous complex) or the high-pass
        residual g - x, then Adam moments are applied to the result. Groups without `green_adj`, or
        params whose first dim does not match, get plain Adam.

        `green_channel` selects the k-hop operator (A_hat**k, cached; the 0s keep it sparse):
        "low"/"high" walk one hop; "twohop"/"threehop" walk the sparse 2/3-hop operator, which
        carries the structure a heterophilous complex needs (2-hop neighbours agree where 1-hop
        neighbours disagree). Use `generate_khop_channel` to auto-select the channel per task from a
        cheap self-supervised score rather than fixing it.

        This is the native optimizer for relational-native models where the complex IS the model and
        the parameters are cochains: the Green's preconditioning does the relational propagation a
        structure-blind optimizer cannot (empirically a bare-cochain node model goes from chance to
        strong generalization, because the optimizer itself carries the training signal across the
        complex). On STANDARD feature-space models it offers nothing over Adam: the structure is
        already in the forward pass, so use plain Adam there. Requires torch."""

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                     green_lam=1.0, green_iters=12, green_channel="low"):
            if lr <= 0:
                raise ValueError("lr must be > 0")
            super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                             green_lam=green_lam, green_iters=green_iters,
                             green_channel=green_channel, green_adj=None))

        # channel -> (operator power on A_hat, low-pass?): low/high walk 1 hop; twohop/threehop
        # walk the sparse k-hop A_hat**k (the 0s keep it sparse; k-hop carries structure a
        # heterophilous complex needs (2-hop neighbours agree where 1-hop neighbours disagree).
        _CH_POWER = {"low": 1, "high": 1, "twohop": 2, "threehop": 3}

        @staticmethod
        def _channel_op(adj, channel, group):
            power = GreensCochain._CH_POWER.get(channel, 1)
            cache = group.setdefault("_op_cache", {})
            if channel not in cache:
                op = adj
                for _ in range(power - 1):
                    op = _torch.sparse.mm(adj, op).coalesce()      # sparse A_hat**k, stays sparse
                cache[channel] = op
            return cache[channel], (channel != "high")

        @staticmethod
        def _greens(op, g, t, low, iters):
            g2 = g if g.dim() >= 2 else g.unsqueeze(1)
            def mv(X):
                return (1.0 + t) * X - t * _torch.sparse.mm(op, X)
            X = _torch.zeros_like(g2); R = g2 - mv(X); P = R.clone()
            rs = (R * R).sum(0, keepdim=True)
            for _ in range(iters):
                AP = mv(P); a = rs / ((P * AP).sum(0, keepdim=True) + 1e-20)
                X = X + a * P; R = R - a * AP; rs2 = (R * R).sum(0, keepdim=True)
                P = R + (rs2 / (rs + 1e-20)) * P; rs = rs2
            out = X if low else (g2 - X)
            return out if g.dim() >= 2 else out.squeeze(1)

        @_torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
                wd = group["weight_decay"]; adj = group.get("green_adj")
                t = group["green_lam"]; ch = group["green_channel"]; it = group["green_iters"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if wd:
                        g = g.add(p, alpha=wd)
                    if adj is not None and p.dim() >= 1 and p.shape[0] == adj.shape[0]:
                        op, low = self._channel_op(adj, ch, group)
                        g = self._greens(op, g, t, low, it)
                    st = self.state[p]; ts = st.get("t", 0) + 1; st["t"] = ts
                    m = st.get("m"); v = st.get("v")
                    if m is None:
                        m = _torch.zeros_like(g); v = _torch.zeros_like(g)
                    m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
                    st["m"], st["v"] = m, v
                    mh = m / (1 - b1 ** ts); vh = v / (1 - b2 ** ts)
                    p.add_(mh / (vh.sqrt() + eps), alpha=-lr)
            return loss

    class GreensFlow(GreensCochain):
        """GreensCochain over the operator that reads BOTH grades.

        Same preconditioning, different complex operator. `coparticipation_adjacency` is
        built from |B1| alone and never touches B2, so a model trained through it is blind
        to every face: attaching hyperfaces leaves its operator bit-identical, and an
        ablation over open against closed complexes reports the same number for a reason
        that has nothing to do with the data. The curl tier exists and the optimizer
        cannot see it.

        `flow_adjacency` is L1_down + alpha * L1_up, so the gradient tier and the curl
        tier both carry signal. `alpha` is the exchange rate between them and defaults to
        `rex.c0_squared`, the exact rational geometry<->topology coupling, rather than to
        a number chosen for the run.

        This is ADDITIVE. GreensCochain is unchanged and stays the right choice for a
        cochain over a face-free complex, where there is no curl tier to miss. Use this
        one where the complex is closed and the faces are supposed to matter.
        """

        def __init__(self, params, *, rex=None, alpha=None, lr=1e-3, betas=(0.9, 0.999),
                     eps=1e-8, weight_decay=0.0, green_lam=1.0, green_iters=12,
                     green_channel="low"):
            super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                             green_lam=green_lam, green_iters=green_iters,
                             green_channel=green_channel)
            self._rex = rex
            self._alpha = alpha
            if rex is not None:
                adj = self.build_adjacency(rex, alpha=alpha)
                for group in self.param_groups:
                    if group.get("green_adj") is not None:
                        continue
                    # the operator is assembled in float64 because it comes off exact
                    # rational structure; the parameters need not be, and a mismatch is a
                    # dtype error at the first CG matvec rather than anything meaningful
                    dtypes = {p.dtype for p in group["params"] if p.is_floating_point()}
                    group["green_adj"] = (adj if len(dtypes) != 1
                                          else adj.to(dtypes.pop()).coalesce())

        @staticmethod
        def build_adjacency(rex, *, alpha=None):
            """The two-grade operator, with c0_squared as the default exchange rate."""
            from rexgraph.flow.hyperflow import flow_adjacency

            if alpha is None:
                alpha = float(rex.c0_squared)
            return flow_adjacency(rex, alpha=float(alpha))

        @property
        def reads_faces(self) -> bool:
            """Whether the operator in use actually carries a curl tier.

            False on a face-free complex, where this degrades to GreensCochain exactly
            rather than pretending to a tier that is not there.
            """
            return self._rex is not None and int(self._rex.nF_hodge) > 0

    def generate_khop_channel(score_fn, channels=("low", "twohop", "threehop")):
        """Context-aware k-hop GENERATOR: pick the propagation channel that fits the task, cheaply.

        Standard optimizers cannot do this because they have no cheap structural signal to detect
        which operator fits; a relational-native model does. `score_fn(channel)` is that signal: a
        callback returning a higher-is-better score for a candidate channel (e.g. a self-supervised
        inner-val reconstruction accuracy: fit the cochain on an inner-train split with that channel
        and score the held-out inner-val). This returns ``(best_channel, {channel: score})``; the
        caller then builds/sets GreensCochain with the selected channel. Empirically the generator
        auto-picks 2-hop for a heterophilous complex and low/3-hop for a homophilous one, with no
        task-specific hardcoding: the detection nobody wires into an optimizer because the cheap
        structural math is missing. A deeper (run-nothing) detector reads the channel straight off
        the complex's spectral moments; this self-supervised version is the first working form."""
        scores = {ch: float(score_fn(ch)) for ch in channels}
        best = max(scores, key=scores.get)
        return best, scores


else:

    class GreensCochain:                             # pragma: no cover (env without torch)
        def __init__(self, *a, **k):
            raise ImportError(
                "GreensCochain requires PyTorch (an optional dependency). Install a torch build "
                "for your backend (CUDA/ROCm/CPU/MPS), or use the framework-agnostic core.")

    class GreensFlow:                                # pragma: no cover (env without torch)
        def __init__(self, *a, **k):
            raise ImportError(
                "GreensFlow requires PyTorch (an optional dependency). Install a torch build "
                "for your backend (CUDA/ROCm/CPU/MPS), or use the framework-agnostic core.")

    def generate_khop_channel(*a, **k):             # pragma: no cover (env without torch)
        raise ImportError("generate_khop_channel requires PyTorch (an optional dependency).")


# training-backend exposure
# CUDA, ROCm, CPU, and Apple MPS are all supported for the training path, as the inference
# path exposes Vulkan/ROCm/CUDA/Metal/CPU (local_runtime.detect_hardware). The software adapts
# to whichever torch wheel is installed rather than assuming one vendor.

def training_backends() -> dict[str, Any]:
    """Detect the torch training backend on this host and whether the GPU actually runs. A
    wheel can see a device it has no compute kernels for (e.g. gfx1151 on a ROCm 6.3 wheel ->
    hipErrorNoBinaryForGpu); this probes a tiny op so ``gpu_usable`` reflects a real op, with
    fallback to CPU. Works with torch absent (numpy core still usable)."""
    if not _HAS_TORCH:
        return {"torch": None, "flavor": None, "devices": ["cpu"], "gpu": None,
                "gpu_usable": False, "recommended_device": "cpu",
                "note": "torch not installed - the numpy hodge_* core still works"}
    t = _torch
    hip = getattr(t.version, "hip", None)
    cuda_ver = getattr(t.version, "cuda", None)
    flavor = "rocm" if hip else ("cuda" if cuda_ver else "cpu")
    devices = ["cpu"]
    gpu = None
    usable = False
    note = None
    try:
        if t.cuda.is_available():                    # ROCm reuses the cuda namespace
            devices.append("cuda")
            try:
                gpu = t.cuda.get_device_name(0)
            except Exception:
                gpu = flavor
            try:                                     # does a real op run, or just "visible"?
                (t.ones(1, device="cuda") + 1).cpu()
                usable = True
            except Exception as e:
                note = f"GPU visible but no compute kernels ({flavor}): {type(e).__name__}"
    except Exception:
        pass
    try:
        if getattr(getattr(t.backends, "mps", None), "is_available", lambda: False)():
            devices.append("mps")
            gpu = gpu or "Apple MPS"
            usable = True
    except Exception:
        pass
    if "cuda" in devices and usable:
        rec = "cuda"
    elif "mps" in devices:
        rec = "mps"
    else:
        rec = "cpu"
    return {"torch": t.__version__, "flavor": flavor, "devices": devices, "gpu": gpu,
            "gpu_usable": usable, "recommended_device": rec, "note": note}


# compute-backend name (rexgraph.compute) / device alias -> torch device string. 'cuda' covers
# ROCm (torch reuses the cuda namespace); Apple Metal -> 'mps'; everything else trains on cpu.
_BACKEND_DEVICE: dict[str, str] = {
    "cuda": "cuda", "rocm": "cuda", "hip": "cuda", "gpu": "cuda",
    "mps": "mps", "metal": "mps",
    "cpu": "cpu", "openmp": "cpu", "vulkan": "cpu",
}


def _cuda_usable() -> bool:
    """True only when a CUDA/ROCm torch device is present AND a real op runs on it. A wheel can
    see a device it has no compute kernels for (e.g. gfx1151 on a ROCm 6.3 wheel), so 'visible' is
    not 'usable'. Honors the compute stack's ``gpu_count()`` (REXGRAPH_MAX_GPUS-capped) when present,
    falling back to torch's own view. Never raises."""
    if not _HAS_TORCH:
        return False
    try:
        from rexgraph import compute as _compute
        if _compute.gpu_count() <= 0:
            return False
    except Exception:
        try:
            if not _torch.cuda.is_available():
                return False
        except Exception:
            return False
    try:                                             # does a real op run, or is the device just visible?
        (_torch.ones(1, device="cuda") + 1).cpu()
        return True
    except Exception:
        return False


def _mps_usable() -> bool:
    if not _HAS_TORCH:
        return False
    try:
        return bool(getattr(getattr(_torch.backends, "mps", None), "is_available", lambda: False)())
    except Exception:
        return False


def _resolve_device(name) -> str:
    """Map a compute-backend name / device string to a usable torch device, guarding GPU
    availability so a GPU request on a CPU-only (or visible-but-unusable-GPU) host degrades to
    'cpu'. Passes through indexed forms like 'cuda:1' when a cuda device is usable."""
    d = _BACKEND_DEVICE.get(str(name).lower(), str(name))
    base = d.split(":")[0].lower()
    if base in ("cuda", "rocm", "hip", "gpu"):
        return d if _cuda_usable() else "cpu"
    if base in ("mps", "metal"):
        return "mps" if _mps_usable() else "cpu"
    return d


def pick_device(prefer: str | None = None) -> str:
    """The torch training/inference device, resolved through the ``rexgraph.compute`` execution
    stack. Always returns a device string.

    ``prefer`` None or ``"auto"`` -> ask ``rexgraph.compute.recommended_backend()`` what backend
    this host resolves to (dynamic per machine, honoring REXGRAPH_BACKEND), map it to a torch device,
    and confirm the GPU actually runs (``gpu_count() > 0`` plus a live op), so a visible-but-unusable
    GPU never leaks through. When the compute stack has no recommendation, fall back to torch's own
    validated probe (``training_backends``).

    An explicit ``prefer`` ('cpu' / 'cuda' / 'mps' / 'cuda:1' / a compute-backend name such as
    'rocm'/'openmp') is honored, still guarded: 'cpu' always forces CPU, and a GPU request on a host
    without a usable GPU degrades to 'cpu'. This is the ``ComputeSpec.backend`` -> device bridge."""
    if prefer is not None and str(prefer).lower() != "auto":
        return _resolve_device(prefer)
    rec = None
    try:                                             # the host's dynamic recommendation (source of truth)
        from rexgraph import compute as _compute
        rec = _compute.recommended_backend()
    except Exception:
        rec = None
    if rec:
        return _resolve_device(rec)
    return training_backends()["recommended_device"]   # compute stack unavailable: torch's own view


def build_optimizer(params, method: str = "adam", lr: float | None = None, **kwargs):
    """Construct a training optimizer. The honest menu, routed by empirical result:
      * ``"adam"``    -> plain Adam (DEFAULT; the right choice for standard feature-space models,
                        where the relational structure lives in the forward pass, not the optimizer)
      * ``"adamw"`` / ``"sgd"`` -> the traditional optimizers, interoperable, opt-in.
      * ``"greens"`` / ``"greenscochain"`` -> GreensCochain, Green's-function preconditioning of the
                        gradient in a complex's own geometry; the native optimizer for
                        relational-native models whose parameters are COCHAINS on that complex
                        (lr default 1e-3). Plain Adam elsewhere ties it, so it is opt-in, not default.
      * ``"hodge"`` / ``"hodgeadam"`` -> HodgeAdam (lr default 1e-3), back-compat only: it ties
                        plain Adam on standard weight matrices.
      * ``"hodgesgd"`` -> HodgeSGD, the structural preconditioner (lr default 1e-2), back-compat.
    Requires torch."""
    if not _HAS_TORCH:
        raise ImportError("build_optimizer needs torch; use the numpy hodge_* core otherwise.")
    m = method.lower()
    if m in ("hodge", "hodgeadam"):
        return HodgeAdam(params, lr=1e-3 if lr is None else lr, **kwargs)
    if m == "hodgesgd":
        return HodgeSGD(params, lr=1e-2 if lr is None else lr, **kwargs)
    if m in ("greens", "greenscochain"):
        return GreensCochain(params, lr=1e-3 if lr is None else lr, **kwargs)
    if m == "sgd":
        return _torch.optim.SGD(params, lr=1e-2 if lr is None else lr, **kwargs)
    if m in ("adam", "default"):
        return _torch.optim.Adam(params, lr=1e-3 if lr is None else lr, **kwargs)
    if m == "adamw":
        return _torch.optim.AdamW(params, lr=1e-3 if lr is None else lr, **kwargs)
    raise ValueError(f"unknown optimizer method {method!r} (adam|adamw|sgd|greens|hodge|hodgesgd)")


def hodge_groups(model, n_heads: int = 1):
    """Architecture-aware HodgeAdam param groups. Attention projection weights get
    ``blocks = n_heads`` (or 3·n_heads for a fused qkv), since each head is an independent
    relational subspace the flat bipartite would wrongly entangle; everything else stays
    ``blocks=1`` (flat weighted-bipartite: correct for MLP / embeddings / generic). Heuristic by
    parameter name; pass the result straight to ``HodgeAdam(hodge_groups(model, n_heads), lr=...)``."""
    if not _HAS_TORCH:
        raise ImportError("hodge_groups needs torch.")
    groups = []
    for name, p in model.named_parameters():
        low = name.lower()
        is_attn = any(t in low for t in ("qkv", "attn", "attention", "in_proj", ".q.", ".k.", ".v."))
        blocks = 1
        if p.dim() == 2 and is_attn and n_heads > 1:
            for cand in (3 * n_heads, n_heads):                        # fused qkv, else single proj
                if p.shape[0] % cand == 0 and p.shape[0] // cand >= 3:
                    blocks = cand; break
        groups.append({"params": [p], "blocks": blocks})
    return groups
