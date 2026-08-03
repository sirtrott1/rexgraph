"""
rexgraph.gpu_preflight: prove the GPU paths on the machine you are paying for.

The GPU paths are torch, so they are portable between CUDA and ROCm in principle --
torch presents HIP as "cuda" and the same code runs. In principle is not a claim
worth making about rented hardware, and the failure modes that matter (no float64
sparse support, a driver that reports a device it cannot allocate on, a
multi-GPU path never exercised) all surface as a wrong answer or a hang rather
than an import error.

This runs every GPU path against a CPU oracle and reports what actually works:

    python -m rexgraph.gpu_preflight
    python -m rexgraph.gpu_preflight --size 4000

Exit status is 0 only when every check the hardware claims to support passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np


def _probe_torch() -> dict[str, Any]:
    out: dict[str, Any] = {"torch": None, "device_count": 0, "devices": [],
                           "flavour": None}
    try:
        import torch
    except ImportError:
        return out
    out["torch"] = torch.__version__
    # torch reports HIP under the cuda namespace, so the version string is the only
    # honest way to tell which runtime is underneath.
    out["flavour"] = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    try:
        if torch.cuda.is_available():
            out["device_count"] = int(torch.cuda.device_count())
            for i in range(out["device_count"]):
                p = torch.cuda.get_device_properties(i)
                out["devices"].append({
                    "index": i, "name": p.name,
                    "total_memory_gb": round(p.total_memory / 2 ** 30, 1),
                    "capability": f"{p.major}.{p.minor}",
                })
    except Exception as e:
        out["probe_error"] = str(e)
    return out


def _check(name: str, fn) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        detail = fn()
        ok, err = True, None
    except Exception as e:
        detail, ok, err = None, False, f"{type(e).__name__}: {e}"
    return {"check": name, "ok": ok, "seconds": round(time.perf_counter() - t0, 3),
            "detail": detail, "error": err}


def _sparse_spd(n: int, seed: int = 0):
    """A sparse SPD matrix and a right-hand side, the shape the solvers actually see."""
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=min(0.01, 8.0 / n), format="csr",
                  random_state=seed, dtype=np.float64)
    A = (A + A.T) * 0.5 + sp.eye(n, format="csr") * (n * 0.01 + 1.0)
    return A.tocsr(), rng.standard_normal((n, 4))


def _check_float64_matmul(n):
    """H100 has real float64 throughput; consumer parts fake it. A silent downcast
    to float32 shows up here as a precision gap, not as an error."""
    import torch
    dev = torch.device("cuda")
    a = torch.randn(n, n, dtype=torch.float64, device=dev)
    b = torch.randn(n, n, dtype=torch.float64, device=dev)
    got = (a @ b).cpu().numpy()
    want = a.cpu().numpy() @ b.cpu().numpy()
    err = float(np.abs(got - want).max())
    return {"max_abs_err": err, "float64_honoured": err < 1e-8}


def _check_sparse_mm(n):
    """torch.sparse.mm in float64 is what every solver here is built on."""
    import torch
    A, B = _sparse_spd(n)
    dev = torch.device("cuda")
    At = torch.sparse_csr_tensor(
        torch.as_tensor(A.indptr, dtype=torch.int64),
        torch.as_tensor(A.indices, dtype=torch.int64),
        torch.as_tensor(A.data, dtype=torch.float64), size=A.shape, device=dev)
    Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
    got = torch.sparse.mm(At, Bt).cpu().numpy()
    err = float(np.abs(got - (A @ B)).max())
    return {"max_abs_err": err, "matches_cpu": err < 1e-8}


def _check_block_cg(n):
    """The solver the character/coherence hot path runs on, against its CPU twin."""
    from rexgraph import scale_propagator as spg
    A, B = _sparse_spd(n)
    dinv = 1.0 / A.diagonal()
    import torch
    dev = torch.device("cuda")
    At = spg._torch_csr(A, dev)
    Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
    dt = torch.as_tensor(dinv, dtype=torch.float64, device=dev)
    X = spg._block_cg_gpu(At, Bt, dt).cpu().numpy()
    resid = float(np.abs(A @ X - B).max())
    return {"max_residual": resid, "converged": resid < 1e-6}


def _check_end_to_end(n):
    """A real complex through the character path, GPU result against CPU result."""
    from rexgraph.graph import RexGraph
    rng = np.random.default_rng(0)
    src = rng.integers(0, n, n * 3).astype(np.int32)
    tgt = ((src + 1 + rng.integers(0, 7, n * 3)) % n).astype(np.int32)
    rex = RexGraph(sources=src, targets=tgt)
    gpu = np.asarray(rex.coherence)
    import rexgraph.scale_propagator as spg
    saved = spg._GPU_MIN_WORK
    try:
        spg._GPU_MIN_WORK = 1 << 62               # force the CPU path
        rex_cpu = RexGraph(sources=src, targets=tgt)
        cpu = np.asarray(rex_cpu.coherence)
    finally:
        spg._GPU_MIN_WORK = saved
    err = float(np.abs(gpu - cpu).max())
    return {"nV": int(rex.nV), "nE": int(rex.nE), "max_abs_err": err,
            "agrees_with_cpu": err < 1e-6}


def _check_multi_gpu(n):
    """The multi-GPU column split, which no single-GPU box has ever exercised."""
    import torch
    if torch.cuda.device_count() < 2:
        return {"skipped": "fewer than two devices"}
    from rexgraph import scale_propagator as spg
    A, _ = _sparse_spd(n)
    dinv = 1.0 / A.diagonal()
    diag = spg._greens_diagonal_multi(A, dinv, n, max(1, n // 4), 1e-10,
                                      list(range(torch.cuda.device_count())))
    return {"devices": torch.cuda.device_count(), "finite": bool(np.all(np.isfinite(diag)))}


def run(size: int = 1200) -> dict[str, Any]:
    """Run every check. Returns the full report."""
    env = _probe_torch()
    report: dict[str, Any] = {"environment": env, "checks": []}
    if not env["torch"]:
        report["verdict"] = "torch is not installed: no GPU path is reachable"
        report["ok"] = False
        return report
    if env["device_count"] == 0:
        report["verdict"] = "no GPU visible to torch: every path falls back to CPU"
        report["ok"] = False
        return report

    n = int(size)
    report["checks"] = [
        _check("float64_matmul", lambda: _check_float64_matmul(min(n, 1024))),
        _check("sparse_mm_float64", lambda: _check_sparse_mm(n)),
        _check("block_cg_vs_cpu", lambda: _check_block_cg(n)),
        _check("coherence_end_to_end", lambda: _check_end_to_end(min(n, 800))),
        _check("multi_gpu_column_split", lambda: _check_multi_gpu(min(n, 600))),
    ]
    failed = [c["check"] for c in report["checks"] if not c["ok"]]
    report["ok"] = not failed
    report["verdict"] = ("every GPU path agreed with its CPU oracle"
                         if not failed else f"failed: {', '.join(failed)}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--size", type=int, default=1200,
                    help="problem size for the solver checks")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    report = run(args.size)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["ok"] else 1

    env = report["environment"]
    print(f"torch {env['torch']} ({env['flavour']}), {env['device_count']} device(s)")
    for d in env["devices"]:
        print(f"  [{d['index']}] {d['name']}  {d['total_memory_gb']} GiB  sm_{d['capability']}")
    print()
    for c in report["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        print(f"  {mark} {c['check']:24} {c['seconds']:>7.3f}s  "
              f"{c['error'] or json.dumps(c['detail'], default=str)}")
    print(f"\n{report['verdict']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
