"""rexgraph.compute - the execution backend layer: backends, thread control, and op dispatch.

A modular spine so parallel and device-specialized kernels register their implementations and are
routed here, instead of hardcoding a backend at each call site. Same registry pattern as
rexgraph.nn.factory.

  backends   cpu (serial/BLAS, always) | openmp (parallel CPU kernels, tuned by the thread count) |
             cuda (NVIDIA/ROCm) | mps (Apple Metal) - each with an availability probe
  threads    set_threads(n) / get_threads() - the CPU parallel width for the OpenMP kernels
  ops        register_op(name, backend, fn); dispatch(name, ...) routes to the best available
             implementation, preferring a requested backend, falling back to cpu

Registering a GPU implementation later is register_op(name, 'cuda', fn) - no call-site change.

The preferred backend is not hardcoded: when a call names none and no default/config is set, dispatch
resolves the best backend for THIS host lazily via rexgraph._env (auto-detected per machine), still
overridable by REXGRAPH_BACKEND or an explicit set_default_backend/apply_config. CPU is always a
valid fallback, so the library runs on a machine with no GPU or toolchain.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

__all__ = [
    "register_backend", "backends", "available_backends", "best_backend",
    "set_threads", "get_threads", "parallel_map",
    "set_default_backend", "get_default_backend", "apply_config",
    "register_op", "ops", "dispatch", "inventory",
    "recommended_backend",
    "gpu_count", "gpu_devices", "multi_gpu_min_work",
]


# backends

_BACKENDS: Dict[str, dict] = {}


def register_backend(name: str, *, available: Callable[[], bool], kind: str = "cpu",
                     description: str = "") -> None:
    """Register a compute backend with an availability probe. kind is 'cpu' or 'gpu'."""
    _BACKENDS[name] = {"name": name, "available": available, "kind": kind, "description": description}


def _ok(b) -> bool:
    try:
        return bool(b["available"]())
    except Exception:
        return False


def backends() -> List[dict]:
    """Every registered backend with its kind, description, and current availability."""
    return [{"name": b["name"], "kind": b["kind"], "description": b["description"],
             "available": _ok(b)} for b in _BACKENDS.values()]


def available_backends() -> List[str]:
    return [b["name"] for b in _BACKENDS.values() if _ok(b)]


def best_backend(prefer: Optional[str] = None) -> str:
    """The best available backend: `prefer` if available, else a GPU backend, else cpu."""
    if prefer and prefer in _BACKENDS and _ok(_BACKENDS[prefer]):
        return prefer
    for b in _BACKENDS.values():
        if b["kind"] == "gpu" and _ok(b):
            return b["name"]
    return "cpu"


# thread control (the CPU parallel width for the compiled OpenMP kernels)

def set_threads(n: Optional[int]) -> None:
    """Set the CPU parallel width for the OpenMP kernels via OMP_NUM_THREADS (most effective set
    before the first heavy kernel call). None restores the default (all cores)."""
    if n is None:
        os.environ.pop("OMP_NUM_THREADS", None)
    else:
        os.environ["OMP_NUM_THREADS"] = str(int(n))
    try:                                                     # apply at runtime when possible
        import threadpoolctl
        threadpoolctl.threadpool_limits(n)
    except Exception:
        pass


def get_threads() -> Optional[int]:
    v = os.environ.get("OMP_NUM_THREADS")
    return int(v) if v else None


# the preferred backend for dispatch when a call does not name one (set from the active setup)
_DEFAULT_BACKEND: Optional[str] = None


def set_default_backend(name: Optional[str]) -> None:
    """Set the backend dispatch prefers when a call passes no `prefer` (None / 'auto' clears it,
    so dispatch falls back to the best available)."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = name if (name and name != "auto") else None


def get_default_backend() -> Optional[str]:
    return _DEFAULT_BACKEND


# --- host-aware auto backend (lazy, cached, never hardcoded) -----------------------------------
# When no backend is explicitly requested (no `prefer`, no default set, no config), dispatch
# resolves the preferred backend DYNAMICALLY for THIS host via rexgraph._env.recommend_backend()
# instead of assuming one machine's GPU. The _env recommendation (cuda/rocm/vulkan/metal/cpu) is
# mapped onto a registered backend name; ROCm reuses the 'cuda' backend (torch/cupy share the
# namespace), Metal maps to 'mps'. A recommendation is used ONLY if that backend is registered AND
# available AND the op implements it - otherwise dispatch falls through to best_backend()/cpu, so
# a recommended GPU without a working runtime never breaks a call. The REXGRAPH_BACKEND env var
# (honored inside recommend_backend) wins over auto-detection; an explicit set_default_backend /
# apply_config wins over both.

# _env names -> registered compute-backend names
_BACKEND_ALIAS: Dict[str, str] = {
    "cuda": "cuda", "rocm": "cuda", "metal": "mps", "cpu": "cpu", "openmp": "openmp",
}
_ENV_MOD = None            # cached rexgraph._env module (or None if unavailable)
_ENV_TRIED = False
_DETECTED = None           # cached detect_compute_backends() result (detection is the costly part)


def _env_module():
    """Import rexgraph._env best-effort. Tries the normal import first (works under the test
    conftest / a full install), then falls back to loading _env.py by file path from beside this
    module (robust for the editable install, where new pure-Python files are not in the finder's
    map). Returns the module or None. Never raises."""
    global _ENV_MOD, _ENV_TRIED
    if _ENV_TRIED:
        return _ENV_MOD
    _ENV_TRIED = True
    mod = None
    try:
        from . import _env as mod  # type: ignore
    except Exception:
        try:
            import importlib.util
            path = os.path.join(os.path.dirname(__file__), "_env.py")
            spec = importlib.util.spec_from_file_location("rexgraph._env", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except Exception:
            mod = None
    _ENV_MOD = mod
    return mod


def recommended_backend() -> Optional[str]:
    """The raw host recommendation from rexgraph._env (cuda/rocm/vulkan/metal/cpu), honoring
    REXGRAPH_BACKEND. Diagnostic; None if _env is unavailable. Not necessarily a registered
    backend name (e.g. 'vulkan' / 'rocm')."""
    global _DETECTED
    mod = _env_module()
    if mod is None:
        return None
    try:
        if _DETECTED is None:
            _DETECTED = mod.detect_compute_backends()
        return mod.recommend_backend(_DETECTED)
    except Exception:
        return None


def _auto_backend() -> Optional[str]:
    """The host recommendation mapped onto a REGISTERED, currently-AVAILABLE backend name, or None
    if there is no such backend (then dispatch falls through to best_backend()/cpu). The costly
    detection is cached; the REXGRAPH_BACKEND override is re-read on each call. Never raises."""
    rec = recommended_backend()
    if not rec:
        return None
    name = _BACKEND_ALIAS.get(rec, rec)
    if name in _BACKENDS and _ok(_BACKENDS[name]):
        return name
    return None


def apply_config(config: Optional[dict]) -> dict:
    """Apply a compute config {threads, backend} (e.g. from a setup profile): sets the CPU thread
    width and the preferred dispatch backend. Absent/None keys keep the current default. Returns the
    effective config, for run logging."""
    if config:
        if "threads" in config:
            set_threads(config.get("threads"))
        set_default_backend(config.get("backend"))
    return {"threads": get_threads(), "backend": get_default_backend() or "auto",
            "available": available_backends()}


def _inner_thread_limiter(inner: Optional[int]):
    """Context manager that caps the INNER native threadpools (OpenBLAS / MKL / OpenMP that
    numpy / scipy dispatch to) to `inner` threads for the duration of a parallel region, via
    ``threadpoolctl`` when available, and a no-op (graceful) when it is not. This is the seam that
    prevents nested-parallelism oversubscription: while the outer pool runs W worker threads, each
    inner BLAS call is held to `inner` threads, so W * inner stays within the thread budget instead
    of W * (all cores). ``inner <= 0`` (or None) leaves the inner pools untouched."""
    import contextlib
    if not inner or inner <= 0:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=int(inner))
    except Exception:
        return contextlib.nullcontext()


def parallel_map(fn, items, *, threads=None, inner_threads=None):
    """Map `fn` over `items` on a thread pool, returning results in order. Threads parallelize
    numpy / scipy / BLAS work because those release the GIL during their C kernels; pure-Python
    bodies stay GIL-bound and gain nothing. Falls back to a serial map for a single worker or item.

    Nested-parallelism safety (automatic, not a per-call threshold): the machine's CPU thread
    BUDGET is `get_threads()` (the configured width) else `os.cpu_count()`. The number of WORKERS
    is `min(threads or budget, len(items))`, and while they run the inner native threadpools are
    held to `inner_threads`, defaulting to the BUDGET ARITHMETIC `max(1, budget // workers)` - so
    workers * inner tracks the budget: no oversubscription when a task calls multi-threaded BLAS,
    and no under-utilization when there are few large tasks (e.g. 4 workers -> inner = cores/4, all
    cores used). Pass `inner_threads=0` to leave the inner pools uncapped, or an explicit int to
    pin them. The inner cap needs the optional `threadpoolctl` dependency to take effect; without
    it the fan-out still runs (as before), just without the inner limit."""
    items = list(items)
    if not items:
        return []
    budget = get_threads() or (os.cpu_count() or 1)              # total CPU thread budget
    cap = threads if threads is not None else budget            # caller's worker cap
    workers = min(cap, len(items))
    if workers <= 1:
        return [fn(x) for x in items]                          # serial: inner keeps all cores
    inner = inner_threads if inner_threads is not None else max(1, budget // workers)
    from concurrent.futures import ThreadPoolExecutor
    with _inner_thread_limiter(inner):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, items))


# --- GPU enumeration for multi-GPU column tiling -----------------------------------------------
# The GPU propagators/solvers apply a SHARED sparse operator to a BLOCK of RHS columns, and that
# block splits EXACTLY and independently across GPUs: replicate the (small, sparse) operator to each
# device, hand each device a disjoint column tile, run the SAME on-device kernel per tile, then
# concatenate the tiles back. These helpers expose how many GPUs are usable and their indices
# (capped by REXGRAPH_MAX_GPUS), plus the (larger) work gate at which replicating the operator across
# devices actually pays off. On a 0- or 1-GPU host gpu_count() < 2, so the caller keeps the existing
# single-GPU/CPU path unchanged - multi-GPU is a pure, size-gated extension, never a new default.

def gpu_count() -> int:
    """Number of GPUs usable for on-device column tiling: ``torch.cuda.device_count()`` (CUDA or
    ROCm) capped by the ``REXGRAPH_MAX_GPUS`` env override, or 0 when no CUDA/ROCm torch device is
    present. This is the device count the multi-GPU dispatch partitions RHS columns across; a value
    < 2 means single-device (the existing path). The env cap is re-read on each call so it can be
    tuned at runtime; never raises."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        n = int(torch.cuda.device_count())
    except Exception:
        return 0
    if n <= 0:
        return 0
    cap = os.environ.get("REXGRAPH_MAX_GPUS")
    if cap:
        try:
            c = int(cap)
            if c >= 0:
                n = min(n, c)
        except Exception:
            pass
    return max(0, n)


def gpu_devices() -> List[int]:
    """The usable GPU device indices ``[0 .. gpu_count()-1]`` (honoring ``REXGRAPH_MAX_GPUS``); empty
    on a CPU-only host. The multi-GPU column tiling assigns one column tile per index in this list."""
    return list(range(gpu_count()))


# Multi-GPU work gate: replicating the sparse operator to each device and concatenating the tiles
# back has fixed overhead, so multi-GPU tiling only engages above a work size LARGER than the
# single-GPU gate (rexgraph.scale_propagator._GPU_MIN_WORK, ~4.2M). Overridable per host via
# REXGRAPH_MULTI_GPU_MIN_WORK; default 1<<24 (4x the single-GPU crossover). Like the single-GPU
# gate this is a PURE performance gate - the tiled result is exact either way, never a correctness
# branch.
_MULTI_GPU_MIN_WORK = int(os.environ.get("REXGRAPH_MULTI_GPU_MIN_WORK", 1 << 24))


def multi_gpu_min_work() -> int:
    """The work threshold (``n * order * columns`` for a Chebyshev apply, ``n * tile`` for a block
    solve) at or above which multi-GPU column tiling engages. Re-reads ``REXGRAPH_MULTI_GPU_MIN_WORK``
    each call so it can be tuned at runtime; falls back to the module default. Never raises."""
    v = os.environ.get("REXGRAPH_MULTI_GPU_MIN_WORK")
    if v:
        try:
            return int(v)
        except Exception:
            pass
    return _MULTI_GPU_MIN_WORK


# op dispatch (device-specialized kernels register their implementations here)

_OPS: Dict[str, Dict[str, Callable]] = {}


def register_op(name: str, backend: str, fn: Callable) -> None:
    """Register an implementation of op `name` for `backend`. Several backends per op are allowed;
    dispatch routes to the best available."""
    _OPS.setdefault(name, {})[backend] = fn


def ops() -> List[dict]:
    return [{"name": n, "backends": sorted(impls)} for n, impls in sorted(_OPS.items())]


def dispatch(name: str, *args, prefer: Optional[str] = None, **kw):
    """Run op `name` on the best available backend that implements it - preferring `prefer`, then
    the best available backend, then any available one, then cpu. Raises if the op is unknown."""
    impls = _OPS.get(name)
    if not impls:
        raise KeyError(f"no op registered as {name!r}")
    pref = prefer or _DEFAULT_BACKEND                        # the active setup's preference, if any
    if pref is None:                                         # nothing explicit -> host-recommended
        pref = _auto_backend()
    order: List[str] = []
    if pref:
        order.append(pref)
    order.append(best_backend(pref))
    order.extend(available_backends())
    order.append("cpu")
    for be in order:
        if be in impls:
            return impls[be](*args, **kw)
    return next(iter(impls.values()))(*args, **kw)           # last resort: any registered impl


def inventory() -> dict:
    """Backends (with availability), the thread width, the preferred backend, and registered ops."""
    return {"backends": backends(), "threads": get_threads(),
            "backend": get_default_backend() or "auto", "recommended": recommended_backend(),
            "ops": ops()}


# built-in backends

def _cuda_available() -> bool:
    try:
        import torch
        if torch.cuda.is_available():                        # ROCm reuses the cuda namespace
            return True
    except Exception:
        pass
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    except Exception:
        return False


register_backend("cpu", available=lambda: True, kind="cpu",
                 description="Serial / BLAS CPU (always available).")
register_backend("openmp", available=lambda: (os.cpu_count() or 1) > 1, kind="cpu",
                 description="Parallel CPU: the compiled OpenMP kernels, tuned by the thread width.")
register_backend("cuda", available=_cuda_available, kind="gpu",
                 description="NVIDIA / ROCm GPU (via torch or cupy).")
register_backend("mps", available=_mps_available, kind="gpu",
                 description="Apple Metal (via torch mps).")
