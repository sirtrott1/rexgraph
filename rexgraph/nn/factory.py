"""
nn.factory: component registry and builders, keyed by one string per component.

The registry holds every swappable component:

    attention:  relational (PropagatorAttention, default) | standard (torch MHA)
    optimizer:  adam (default, standard feature-space models) | adamw | sgd |
               greens (GreensCochain, relational-native cochain models) |
               hodge / hodgesgd (back-compat, tie plain Adam)
    model:      registered externally (the library ships no model)

RexGraph-native pieces are the registered defaults; the traditional option is selected by name
and is interoperable. Each `build_*` falls back to a safe default with a logged note on an
unavailable or mis-constructed component rather than raising. `inventory()` reports what is
available.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("rexgraph.nn")

try:
    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F
    _HAS_TORCH = True
except Exception:                                    # pragma: no cover (torch optional)
    _HAS_TORCH = False
    _nn = type("nn", (), {"Module": object})()


def available() -> bool:
    return _HAS_TORCH


# the registry

_REG: Dict[str, Dict[str, dict]] = {"attention": {}, "optimizer": {}, "model": {}}


def register(kind: str, name: str, factory: Callable, *, native: bool = False,
             default: bool = False, description: str = "", available_fn: Callable = None):
    """Register a component. `native` marks it as a RexGraph-native piece; `default` makes it the
    one chosen when no name is given."""
    _REG.setdefault(kind, {})[name] = {
        "factory": factory, "native": native, "default": default,
        "description": description, "available_fn": available_fn or (lambda: _HAS_TORCH)}


def _avail(meta) -> bool:
    try:
        return bool(meta["available_fn"]())
    except Exception:
        return False


def default_name(kind: str) -> Optional[str]:
    for n, m in _REG.get(kind, {}).items():
        if m["default"] and _avail(m):
            return n
    for n, m in _REG.get(kind, {}).items():          # any available if no default is
        if _avail(m):
            return n
    return None


def list_components(kind: str) -> List[dict]:
    return [{"name": n, "native": m["native"], "default": m["default"],
             "description": m["description"], "available": _avail(m)}
            for n, m in _REG.get(kind, {}).items()]


def inventory() -> dict:
    """Available components per kind, each with native/default/available flags, plus the torch
    version."""
    return {"torch": (_t.__version__ if _HAS_TORCH else None),
            "components": {k: list_components(k) for k in _REG}}


def build_attention(name: Optional[str], d: int, n_head: int, **kw):
    """Construct an attention block by name. An unknown/unavailable/failing choice falls back to
    the default (then to 'standard'), with a log note. Returns (module, used)."""
    reg = _REG["attention"]
    chosen = name or default_name("attention")
    tried = []
    for cand in [chosen, default_name("attention"), "standard"]:
        if not cand or cand in tried or cand not in reg or not _avail(reg[cand]):
            continue
        tried.append(cand)
        try:
            mod = reg[cand]["factory"](d, n_head, **kw)
            if cand != (name or chosen):
                logger.warning("attention %r -> fell back to %r", name, cand)
            return mod, cand
        except Exception as e:
            logger.warning("attention %r failed to build (%s) - trying fallback", cand, e)
    raise RuntimeError("no attention component could be built (torch installed? %s)" % _HAS_TORCH)


def build_optimizer(params, name: Optional[str] = None, lr: Optional[float] = None, **kw):
    """Construct the optimizer chosen by name (defaults to plain Adam, the honest default for
    standard feature-space models). Delegates to `optim.build_optimizer`; requires torch."""
    from rexgraph.nn import optim
    return optim.build_optimizer(params, method=(name or "adam"), lr=lr, **kw)


# attention components

if _HAS_TORCH:
    import math

    class StandardCausalAttention(_nn.Module):
        """Scaled-dot-product causal multi-head attention (PyTorch), wrapped to the same
        (out, diag) interface as the relational block so they are interchangeable."""
        def __init__(self, d: int, n_head: int, **kw):
            super().__init__()
            assert d % n_head == 0
            self.h, self.dk = n_head, d // n_head
            self.qkv = _nn.Linear(d, 3 * d)
            self.proj = _nn.Linear(d, d)

        def forward(self, x, return_diag: bool = False):
            B, T, d = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(B, T, self.h, self.dk).transpose(1, 2)
            k = k.view(B, T, self.h, self.dk).transpose(1, 2)
            v = v.view(B, T, self.h, self.dk).transpose(1, 2)
            s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
            i = _t.arange(T, device=x.device)
            s = s.masked_fill(~(i[:, None] >= i[None, :]), float("-inf"))
            out = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d)
            return self.proj(out), None

    def _relational(d, n_head, **kw):
        from rexgraph.nn.relational_attention import CausalPropagatorAttention
        return CausalPropagatorAttention(d, n_head, hops=int(kw.get("hops", 4)),
                                         window=kw.get("window"))

    def _relational_bidir(d, n_head, **kw):
        from rexgraph.nn.relational_attention import PropagatorAttention
        return PropagatorAttention(d, n_head, channels=kw.get("channels", ("heat", "curl")))

    register("attention", "relational", _relational, native=True, default=True,
             description="Causal propagator attention - multi-hop routing on the token graph (your math).")
    register("attention", "relational-bidir", _relational_bidir, native=True,
             description="Bidirectional propagator (heat/curl channels) - for encoders/analysis.")
    register("attention", "standard", lambda d, h, **kw: StandardCausalAttention(d, h),
             native=False, description="Standard causal multi-head attention (PyTorch default).")

    for _m in ("hodge", "hodgesgd", "adam", "sgd", "adamw", "greens"):
        register("optimizer", _m, None, native=(_m.startswith("hodge") or _m == "greens"),
                 default=(_m == "adam"),
                 description={"hodge": "HodgeAdam - vector-Hodge preconditioned Adam (back-compat; ties plain Adam).",
                              "hodgesgd": "HodgeSGD - pure structural preconditioner (back-compat; ties plain Adam).",
                              "adam": "Adam (PyTorch) - default; the honest choice for standard feature-space models.",
                              "sgd": "SGD (PyTorch).",
                              "adamw": "AdamW (PyTorch).",
                              "greens": "GreensCochain - Green's-function preconditioned Adam, the native "
                                        "optimizer for relational-native cochain models."}[_m])
    register("optimizer", "hodge-arch", None, native=True,
             description="HodgeAdam, ARCHITECTURE-AWARE - attention heads as independent Hodge "
                         "blocks (Track-2, where the structural edge should show).")

# Example models and a training-demo loop are not shipped here. rexgraph.nn provides the parts
# to build and train models (registry, builders, optimizers, attention blocks, propagators);
# assembled example nets live outside the library.


# architecture-aware optimizer names that need the model (not just params) to build head-blocks
_ARCH_OPT = {"hodge-arch", "hodgearch", "hodge-groups", "hodgegroups"}


def make_optimizer(name: str, model, trainable, *, n_heads: int = 1, lr=None, **kw):
    """Build the optimizer, routing per model type. This is the honest router: ``"auto"`` (the
    default) picks GreensCochain for a relational-native model whose parameters are cochains on a
    complex, and plain Adam for a standard feature-space model, because the two families were
    benchmarked and neither optimizer is universally better, they are correct for different model
    shapes. Specifically, ``"auto"``:

      * if `model` exposes a ``greens_groups()`` method returning a list of param-group dicts
        (each with ``"params"`` and a ``"green_adj"`` complex operator, optionally
        ``"green_channel"`` / ``"green_lam"``), builds GreensCochain over those groups plus one
        remainder group for the model's other trainable params, and returns
        ``GreensCochain(auto: N cochain groups)``.
      * otherwise builds plain ``torch.optim.Adam`` over `trainable` and returns ``Adam(auto)``.

    Named modes still work: ``"greens"`` -> GreensCochain directly; ``"hodge-arch"`` -> the
    architecture-aware HodgeAdam, which needs the model to group attention-projection weights into
    per-head Hodge blocks (back-compat); ``"hodge"`` / ``"hodgesgd"`` / ``"adam"`` / ``"sgd"`` /
    ``"adamw"`` -> the matching optim.build_optimizer path. Extra kwargs (e.g. ``gamma_curl``) pass
    through to the underlying optimizer. On any failure ``"auto"``/``"hodge-arch"`` fall back to
    plain Adam. Returns (optimizer, label)."""
    from rexgraph.nn import optim
    nm = (name or "auto").lower()
    if nm == "auto":
        trainable = list(trainable)
        greens_groups_fn = getattr(model, "greens_groups", None)
        if greens_groups_fn is not None:
            try:
                groups = list(greens_groups_fn())
                if groups:
                    covered = set()
                    for g in groups:
                        covered.update(id(p) for p in g["params"])
                    remainder = [p for p in trainable if id(p) not in covered]
                    all_groups = list(groups)
                    if remainder:
                        all_groups.append({"params": remainder})
                    opt = optim.GreensCochain(all_groups, lr=1e-3 if lr is None else lr, **kw)
                    return opt, f"GreensCochain(auto: {len(groups)} cochain groups)"
            except Exception as e:
                logger.warning("auto greens_groups build failed (%s) -> Adam", e)
        opt = _t.optim.Adam(trainable, lr=1e-3 if lr is None else lr)
        return opt, "Adam(auto)"
    if nm in _ARCH_OPT:
        try:
            groups = [g for g in optim.hodge_groups(model, n_heads)
                      if g["params"][0].requires_grad]
            blocked = sum(1 for g in groups if g.get("blocks", 1) > 1)
            opt = optim.HodgeAdam(groups, lr=1e-3 if lr is None else lr, **kw)
            return opt, f"HodgeAdam(arch:{blocked} head-blocked)"
        except Exception as e:
            logger.warning("hodge-arch build failed (%s) -> plain adam", e)
            nm = "adam"
    if nm in ("greens", "greenscochain"):
        opt = optim.build_optimizer(list(trainable), method="greens", lr=lr, **kw)
        return opt, "GreensCochain"
    opt = optim.build_optimizer(trainable, method=nm, lr=lr, **kw)
    return opt, type(opt).__name__


def build_model(spec: dict):
    """Build a model from a spec dict, e.g. {model:'my-net', ...kwargs}. The library ships no model
    of its own; register one with `register("model", name, Factory)` first. Unknown name falls
    back to the default if one exists. Returns (model, resolved_name)."""
    if not _HAS_TORCH:
        raise ImportError("build_model needs PyTorch (optional dependency).")
    reg = _REG["model"]
    if not reg:
        raise ValueError("no model is registered - register one with "
                         "rexgraph.nn.register('model', <name>, <Factory>) first "
                         "(the library ships the builders, not a model).")
    name = spec.get("model") or default_name("model")
    if name not in reg:
        name = default_name("model")
    kw = {k: v for k, v in spec.items() if k != "model"}
    return reg[name]["factory"](**kw), name
