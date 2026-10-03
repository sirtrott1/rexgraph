"""
rexgraph.gpu_preflight: prove the GPU paths on the machine you are paying for.

The GPU paths are torch, so they are portable between CUDA and ROCm in principle:
torch presents HIP as "cuda" and the same code runs. In principle is not a claim
worth making about rented hardware, and the failure modes that matter (no float64
sparse support, a driver that reports a device it cannot allocate on, a
multi GPU path never exercised) all surface as a wrong answer or a hang rather
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
from rexgraph.compute import sparse_mm
from rexgraph.gpu_access import device_access, diagnose, probe_vulkan


def _probe_torch() -> dict[str, Any]:
    out: dict[str, Any] = {"torch": None, "device_count": 0, "devices": [],
                           "flavour": None}
    try:
        import torch
    except ImportError:
        return out
    out["torch"] = torch.__version__
    out["flavour"] = ("rocm" if getattr(torch.version, "hip", None) else
                      "cuda" if getattr(torch.version, "cuda", None) else "cpu")
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
    detail = None
    try:
        detail = fn()
        failed = [key for key, value in detail.items()
                  if isinstance(value, (bool, np.bool_)) and not value]
        if failed:
            raise ArithmeticError(f"numerical qualification failed: {', '.join(failed)}")
        ok, err = True, None
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    return {"check": name, "ok": ok, "seconds": round(time.perf_counter() - t0, 3),
            "detail": detail, "error": err}


def _sparse_spd(n: int, seed: int = 0):
    """A sparse SPD matrix and a right hand side, the shape the solvers actually see."""
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=min(0.01, 8.0 / n), format="csr",
                  random_state=seed, dtype=np.float64)
    A = (A + A.T) * 0.5
    A.setdiag(0)
    # Symmetric strict diagonal dominance establishes SPD at every requested size.
    A = A + sp.diags(np.asarray(abs(A).sum(axis=1)).ravel() + 1.0, format="csr")
    return A.tocsr(), rng.standard_normal((n, 4))


def _check_float64_matmul(n):
    """Check float64 arithmetic; device throughput is a separate measurement."""
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
    from rexgraph.compute import sparse_csr_tensor

    At = sparse_csr_tensor(
        torch.as_tensor(A.indptr, dtype=torch.int64),
        torch.as_tensor(A.indices, dtype=torch.int64),
        torch.as_tensor(A.data, dtype=torch.float64), size=A.shape, device=dev)
    Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
    got = sparse_mm(At, Bt).cpu().numpy()
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
    """A real complex through resident GPU propagation against its CPU twin.

    Automatic character dispatch could route both sides to CPU. Call the resident
    device recurrence directly, without a CPU fallback.
    """
    from rexgraph.graph import RexGraph
    rng = np.random.default_rng(0)
    src = rng.integers(0, n, n * 3).astype(np.int32)
    tgt = ((src + 1 + rng.integers(0, 7, n * 3)) % n).astype(np.int32)
    rex = RexGraph(sources=src, targets=tgt)
    import rexgraph.scale_propagator as spg
    import torch
    from rexgraph.nn.rcf_torch import heat_apply
    L = rex.L1_sparse.tocsr()
    signals = rng.standard_normal((rex.nE, 4))
    bound = spg._gershgorin_bound(L)
    Lt = spg._torch_csr(L, torch.device("cuda"))
    Xt = torch.as_tensor(signals, dtype=torch.float64, device="cuda")
    gpu = heat_apply(Lt, Xt, 0.2, K=64, lam_max=bound).cpu().numpy()
    cpu = spg.heat_apply(L, signals, 0.2, order=64, lam_max=bound, backend="cpu")
    err = float(np.abs(gpu - cpu).max())
    return {"nV": int(rex.nV), "nE": int(rex.nE), "max_abs_err": err,
            "agrees_with_cpu": err < 1e-6}


def _check_multi_gpu(n):
    """The multi GPU column split, which no single GPU box has ever exercised."""
    import torch
    if torch.cuda.device_count() < 2:
        return {"skipped": "fewer than two devices"}
    from rexgraph import scale_propagator as spg
    A, _ = _sparse_spd(n)
    dinv = 1.0 / A.diagonal()
    diag = spg._greens_diagonal_multi(A, dinv, n, max(1, n // 4), 1e-10,
                                      list(range(torch.cuda.device_count())))
    oracle = spg.greens_diagonal(A, chunk=max(1, n // 4), backend="cpu")
    err = float(np.abs(diag - oracle).max())
    return {"devices": torch.cuda.device_count(), "finite": bool(np.all(np.isfinite(diag))),
            "max_abs_err": err, "agrees_with_cpu": bool(np.allclose(diag, oracle, rtol=1e-7, atol=1e-9))}


def _check_autograd(n):
    """Resident heat/wave filters and Green resolvent, including learned scalars."""
    import torch
    from rexgraph.nn.rcf_torch import propagator_apply, green_resolvent
    A, X = _sparse_spd(n)
    matrix = A.toarray()
    bound = float(np.abs(matrix).sum(axis=1).max())
    def execute(device):
        L = torch.as_tensor(matrix, dtype=torch.float64, device=device)
        x = torch.tensor(X, dtype=torch.float64, device=device, requires_grad=True)
        t = torch.tensor(.2, dtype=torch.float64, device=device, requires_grad=True)
        alpha = torch.tensor(.3, dtype=torch.float64, device=device, requires_grad=True)
        outputs = propagator_apply(L, x, t, K=32, lam_max=bound)
        outputs += (green_resolvent(x, alpha, lambda v: L @ v, tol=1e-11, max_iter=100),)
        loss = sum(y.square().sum() for y in outputs)
        grads = torch.autograd.grad(loss, (x, t, alpha))
        return [y.detach().cpu().numpy() for y in (*outputs, *grads)]
    cpu, gpu = execute("cpu"), execute("cuda")
    err = max(float(np.abs(a - b).max()) for a, b in zip(cpu, gpu, strict=True))
    return {"max_abs_err": err, "forward_and_gradients_agree":
            all(np.allclose(a, b, rtol=1e-7, atol=1e-8) for a, b in zip(cpu, gpu, strict=True))}


def _check_packed_training(n):
    """Native HIP tensor ABI on a nondefault Torch stream, forward and adjoint."""
    import torch
    from rexgraph import hip_ternary as H
    if not getattr(torch.version, "hip", None):
        return {"skipped": "native HIP check requires ROCm"}
    lib = H._load()
    if lib is None or not hasattr(lib, "ternary_f32_batch_async"):
        return {"skipped": "optional native HIP tensor library is not built"}
    from rexgraph.nn import PackedTernaryLinear
    rng = np.random.default_rng(19)
    a = rng.integers(-1, 2, size=(17, n + 1), dtype=np.int8)
    errors = {}
    default = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(default)
    for dtype in (torch.float32, torch.float64):
        with torch.cuda.stream(stream):
            layer = PackedTernaryLinear(a, backend="hip").cuda()
            x = torch.tensor(rng.normal(size=(7, n + 1)), dtype=dtype, device="cuda", requires_grad=True)
            dense = torch.tensor(a, dtype=dtype, device="cuda")
            out = layer(x)
            oracle = x @ dense.T
            dx = torch.autograd.grad(out.square().sum(), x, retain_graph=True)[0]
            expected = torch.autograd.grad(oracle.square().sum(), x)[0]
            forward_err = float((out - oracle).abs().max())
            backward_err = float((dx - expected).abs().max())
            tol = 1e-4 if dtype == torch.float32 else 1e-10
            errors[str(dtype)] = {"forward_abs_err": forward_err, "backward_abs_err": backward_err}
            if not (torch.allclose(out, oracle, rtol=tol, atol=tol) and
                    torch.allclose(dx, expected, rtol=tol, atol=tol)):
                raise ArithmeticError(f"packed {dtype} forward/adjoint mismatch: {errors[str(dtype)]}")
    stream.synchronize()
    return {"dtype_checks": errors, "forward_and_gradients_agree": True}


def run(size: int = 1200) -> dict[str, Any]:
    """Run every check. Returns the full report."""
    n = int(size)
    if n < 2:
        raise ValueError("GPU qualification size must be at least two")
    env = _probe_torch()
    report: dict[str, Any] = {"environment": env, "checks": []}
    if not env["torch"]:
        report["access"] = device_access()
        report["vulkan"] = probe_vulkan()
        report["verdict"] = "torch is not installed: Torch GPU qualification did not run"
        report["ok"] = False
        return report
    if env["device_count"] == 0:
        report["access"] = device_access()
        report["vulkan"] = probe_vulkan()
        report["verdict"] = "no GPU visible to this Torch process: GPU qualification did not run"
        report["next_step"] = (
            "Vulkan hardware was enumerated independently; it does not provide a Torch training backend."
            if report["vulkan"]["available"] else
            "Check this process's device grants and visibility filters; these results do not establish host GPU absence."
        )
        report["ok"] = False
        return report

    report["checks"] = [
        _check("float64_matmul", lambda: _check_float64_matmul(min(n, 1024))),
        _check("sparse_mm_float64", lambda: _check_sparse_mm(n)),
        _check("block_cg_vs_cpu", lambda: _check_block_cg(n)),
        _check("coherence_end_to_end", lambda: _check_end_to_end(min(n, 800))),
        _check("propagator_green_autograd", lambda: _check_autograd(min(n, 128))),
        _check("native_packed_training", lambda: _check_packed_training(min(n, 1024))),
        _check("multi_gpu_column_split", lambda: _check_multi_gpu(min(n, 600))),
    ]
    failed = [c["check"] for c in report["checks"] if not c["ok"]]
    report["ok"] = not failed
    report["verdict"] = ("all executed GPU checks passed (see individual skips)"
                         if not failed else f"failed: {', '.join(failed)}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--size", type=int, default=1200,
                    help="problem size for the solver checks")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument("--diagnose", action="store_true",
                    help="inspect device access and enumerate HIP/Vulkan without numerical qualification")
    args = ap.parse_args(argv)

    if args.diagnose:
        report = diagnose()
        print(json.dumps(report, indent=2))
        return 0 if report["hip"]["available"] or report["vulkan"]["available"] else 1
    report = run(args.size)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["ok"] else 1

    env = report["environment"]
    print(f"torch {env['torch']} ({env['flavour']}), {env['device_count']} device(s)")
    for d in env["devices"]:
        print(f"  [{d['index']}] {d['name']}  {d['total_memory_gb']} GiB  capability {d['capability']}")
    print()
    for c in report["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        print(f"  {mark} {c['check']:24} {c['seconds']:>7.3f}s  "
              f"{c['error'] or json.dumps(c['detail'], default=str)}")
    print(f"\n{report['verdict']}")
    if report.get("next_step"):
        print(report["next_step"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
