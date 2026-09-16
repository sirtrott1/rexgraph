"""Measure exact ratio kernels against Q on explicit supplied coordinates.

Run from a neutral directory with an installed Core. An explicit extension
path can select a previously built reference kernel. It changes this benchmark
only and never changes runtime registration. Timings exclude the Q oracle.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


def expected(args, mode, owner, count, exact=False):
    item, carried, seed, deg, den, n, _ = args
    signed, mass = [Fraction(0)]*n, [Fraction(0)]*n
    for i, c, v in zip(item, carried, seed, strict=True):
        term = Fraction(int(c), int(deg[v]))
        signed[i] += term
        mass[i] += abs(term)
    out = [Fraction(0)]*count
    for i in range(n):
        value = signed[i] if mode == 0 else abs(signed[i]) if mode == 1 else mass[i]-abs(signed[i])
        g = i if owner is None else int(owner[i])
        if g >= 0:
            out[g] += value / int(den[i])
    return np.array(out, dtype=object) if exact else np.array([float(v) for v in out])


def measure(kernel, args, mode, owner, count, repeats, exact=False):
    want = expected(args, mode, owner, count, exact)
    kwargs = {"mode": mode, "group": owner, "n_groups": count if owner is not None else 0}
    if exact:
        kwargs["exact"] = True
    got = kernel.axis_ratio(*args, **kwargs)
    timings = []
    calls = max(1, 10000//max(len(args[0]), 1))
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(calls):
            got = kernel.axis_ratio(*args, **kwargs)
        timings.append((time.perf_counter_ns()-start)/1e9/calls)
    return {"mode": ("sum", "absolute", "coverage")[mode], "grouped": owner is not None,
            "median_seconds": statistics.median(timings), "samples_seconds": timings,
            "exact_rounding": bool(np.array_equal(got, want)), "output_sum": float(sum(got)),
            "items": args[5], "seeds": len(args[3]), "coordinates": len(args[0]), "outputs": count,
            "calls_per_sample": calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--exact", action="store_true", help="measure Fraction output; requires an exact output kernel")
    parser.add_argument("--include-wide", action="store_true",
                        help="include wide sparse axes; do not use with the former grid kernel")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.extension is None:
        from rexgraph.core import _exact_ratio as kernel
    else:
        spec = importlib.util.spec_from_file_location("_exact_ratio", args.extension.resolve())
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)

    readings = []
    for n, ns, m in ((64, 8, 128), (2500, 64, 5000), (25000, 64, 50000)):
        rng = np.random.default_rng(73)
        inputs = (rng.integers(n, size=m, dtype=np.int64), rng.integers(1, 4, size=m, dtype=np.int64),
                  rng.integers(ns, size=m, dtype=np.int64), rng.integers(1, 10, size=ns, dtype=np.int64),
                  rng.integers(1, 10, size=n, dtype=np.int64), n, kernel.frac_bits_for(3*m, ns, n))
        # SUM repeats the original positive fixture. Other modes include cancellation.
        for mode in (0, 1, 2):
            carried = inputs[1].copy()
            if mode:
                carried[::3] *= -1
            case = (inputs[0], carried, *inputs[2:])
            for grouped in (False, True):
                count = max(n//10, 1) if grouped else n
                owner = np.arange(n, dtype=np.int64) % count if grouped else None
                readings.append(measure(kernel, case, mode, owner, count, args.repeats, args.exact))
    a = lambda values: np.asarray(values, dtype=np.int64)
    d = 2**62
    case = (a([0, 0]), a([1, -1]), a([0, 1]), a([d-1, d]), a([1]), 1, kernel.frac_bits_for(1, 2, 1))
    cancellation = measure(kernel, case, 0, None, 1, args.repeats, args.exact)
    if args.include_wide:
        n = 20000
        case = (a([0, n-1]), a([1, 3]), a([n-1, 0]), np.ones(n, np.int64),
                np.ones(n, np.int64), n, kernel.frac_bits_for(3, n, n))
        readings.append(measure(kernel, case, 0, None, n, args.repeats, args.exact))
    print(json.dumps({"interpreter": sys.executable, "kernel": kernel.__file__, "numpy": np.__version__,
                      "repeats": args.repeats, "exact": args.exact,
                      "readings": readings, "cancellation": cancellation}, indent=2))
    return 0 if all(r["exact_rounding"] for r in readings+[cancellation]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
