"""Qualify native HIP against Core and measure residency and CPU/GPU contention.

Products return host arrays, so times include vector uploads, downloads and native
synchronization. This benchmarks numerical kernels, without autograd or NN quality.
An explicitly requested HIP device never falls back to CPU.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from rexgraph import compute, ternary as tn
from rexgraph.core import _channel_tower


def _timing(fn, iterations):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {"median_ms": statistics.median(samples) * 1000,
            "min_ms": min(samples) * 1000, "iterations": iterations}


def _run_many(fn, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return time.perf_counter() - start


def _tower_cases(H):
    """Witness, empty column, branching, signs, weights and declared coefficients."""
    bp = np.array([0, 1, 3, 7, 7, 10], dtype=np.int32)
    bi = np.array([0, 0, 1, 2, 0, 1, 3, 1, 2, 3], dtype=np.int32)
    weights = np.array([-2., 0., 3., 1., -.5])
    coefficients = np.array([1., -.7, .7, -1., .2, .3, .5, -1., .25, .75])
    results = []
    for coef in (None, coefficients):
        oracle = _channel_tower.channel_diagonals_any_arity(bp, bi, 5, weights, coefficients=coef)
        for block in (1, 31, 64, 256):
            actual = H.channel_tower(bp, bi, 5, weights, block=block, coefficients=coef)
            for got, want in zip(actual, oracle, strict=True):
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
        results.append({"declared": coef is not None, "blocks": [1, 31, 64, 256], "agrees": True})
    return results


def run(*, device="hip", rows=256, cols=4097, block=256, threads=1, iterations=20, seed=0, contention=False):
    if device not in {"cpu", "hip"}:
        raise ValueError("device must be cpu or hip")
    if min(rows, cols, threads, iterations) < 1:
        raise ValueError("rows, cols, threads and iterations must be positive")
    H = None
    if device == "hip":
        from rexgraph import hip_ternary as H
        H._block(block)
        if not H.available():
            raise RuntimeError("HIP device with compatible native kernels is unavailable; CPU fallback is disabled")
    elif contention:
        raise ValueError("contention measurement requires --device hip")
    compute.set_threads(threads)
    rng = np.random.default_rng(seed)
    a = rng.integers(-1, 2, size=(rows, cols), dtype=np.int8)
    op = tn.pack(a)
    x = rng.choice(np.array([-1, 1], np.int64), size=cols)
    v = rng.standard_normal(cols)
    cpu_pm1 = lambda: tn._pm1_cpu(op, x, threads)
    cpu_f64 = lambda: tn._f64_cpu(op, v, threads)
    oracle_pm1, oracle_f64 = cpu_pm1(), cpu_f64()
    report = {"ok": True, "gpu_qualified": False,
              "config": {"device": device, "rows": rows, "cols": cols, "block": block,
                         "threads": threads, "iterations": iterations, "seed": seed},
              "operator_bytes": op.nbytes, "parameters_trained": 0,
              "environment": {"interpreter": sys.executable, "core": str(Path(tn.__file__).resolve()),
                              "core_version": metadata.version("rexgraph"),
                              "cpu_extension": _channel_tower.__file__, "threads": threads},
              "cpu": {"pm1": _timing(cpu_pm1, iterations), "float64": _timing(cpu_f64, iterations)}}
    if H is None:
        return report
    start = time.perf_counter()
    resident = H.resident(op, block=block)
    report["resident_setup_ms"] = (time.perf_counter() - start) * 1000
    with resident as handle:
        hip_pm1 = lambda: handle.matvec(x)
        hip_f64 = lambda: handle.matvec_f64(v)
        np.testing.assert_array_equal(hip_pm1(), oracle_pm1)
        np.testing.assert_allclose(hip_f64(), oracle_f64, rtol=1e-11, atol=1e-10)
        report["hip_resident"] = {"pm1": _timing(hip_pm1, iterations), "float64": _timing(hip_f64, iterations)}
        def one_shot():
            with H.resident(op, block=block) as r:
                return r.matvec(x)
        report["hip_one_shot_pm1"] = _timing(one_shot, iterations)
        report["channel_tower"] = _tower_cases(H)
        if contention:
            # Independent exact products: no shared model weights or optimizer updates.
            cpu_solo = _run_many(cpu_pm1, iterations)
            gpu_solo = _run_many(hip_pm1, iterations)
            with ThreadPoolExecutor(max_workers=2) as pool:
                start = time.perf_counter()
                c = pool.submit(_run_many, cpu_pm1, iterations)
                g = pool.submit(_run_many, hip_pm1, iterations)
                cpu_busy, gpu_busy = c.result(), g.result()
                together = time.perf_counter() - start
            report["contention"] = {
                "cpu_solo_seconds": cpu_solo, "hip_solo_seconds": gpu_solo,
                "cpu_busy_seconds": cpu_busy, "hip_busy_seconds": gpu_busy,
                "together_seconds": together,
                "vs_sequential_speedup": (cpu_solo + gpu_solo) / together,
                "aggregate_products_per_second": 2 * iterations / together,
                "same_task_count_on_fastest_device_seconds": 2 * min(cpu_solo, gpu_solo),
                "vs_fastest_device_speedup": 2 * min(cpu_solo, gpu_solo) / together,
                "scope": "fixed equal split of independent pm1 products; sweep threads and shapes"}
        report["gpu_qualified"] = True
        report["hip_library"] = str(Path(os.environ.get("REXGRAPH_TERNARY_HIP") or H.library_path()).resolve())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "hip"), default="hip")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=4097)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contention", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output
    if output.exists():
        parser.error(f"output already exists: {output}")
    options = vars(args).copy()
    options.pop("output")
    try:
        report = run(**options)
    except Exception as exc:
        report = {"ok": False, "gpu_qualified": False, "config": options,
                  "error": f"{type(exc).__name__}: {exc}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
