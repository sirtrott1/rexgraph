"""
agent.bench_stack: the numbers, measured the same way every time.

Every performance claim in this stack was arrived at by measuring, and two of them
were wrong the first time. A benchmark reusing one payload across backends charged
the whole of structural_signature to whichever backend ran first, because RexGraph
caches betti and coherence on the instance -- a 157x difference that read as a
backend result. A vocabulary lookup that had gone from 0.5 ms to 3.5 ms was a list
scan hiding inside an index whose whole purpose was to remove scans.

Both were caught by re-measuring rather than by reasoning, so this exists to make
re-measuring the cheap thing to do:

    python -m agent.bench_stack                 # run and print
    python -m agent.bench_stack --save out.json # record it
    python -m agent.bench_stack --compare out.json   # against a recorded run

Every payload is fresh per backend, so nothing inherits a warm cache from whoever
ran before it. Numbers are medians, and the host is recorded alongside them because
a figure without the machine it came from is not a measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any

import numpy as np

#: how far a figure may drift from a recorded one before it is worth looking at.
#: Wide on purpose: a benchmark that cries wolf on ordinary machine noise gets
#: ignored, which costs more than the regression it was meant to catch.
TOLERANCE = 2.0


def _payload(n_records: int, vocab_size: int = 20000, seed: int = 0):
    """Fresh complexes. Never reused across backends: RexGraph caches betti and
    coherence on the instance, so a shared payload hands the second backend a
    157x head start and calls it a result."""
    from rexgraph.graph import RexGraph

    rng = np.random.default_rng(seed)
    vocab = [f"t{i:05d}" for i in range(vocab_size)]
    out = []
    for _ in range(n_records):
        n = int(rng.integers(8, 20))
        labels = list(rng.choice(vocab, size=n, replace=False))
        rex = RexGraph(sources=np.arange(n - 1, dtype=np.int32),
                       targets=np.arange(1, n, dtype=np.int32))
        rex._agent_meta = {"vertex_labels": labels}
        out.append((rex, labels))
    return out


def _median_ms(fn: Callable, reps: int) -> float:
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(xs), 4)


def bench_put_breakdown(n: int = 120) -> dict[str, float]:
    """Where a put's time actually goes. The headline: the signature dominates, so
    the backend is not what makes ingest slow."""
    from agent import rcdb

    sig, ser, store = [], [], []
    sink = rcdb.MemoryStore()
    for k, (rex, labels) in enumerate(_payload(n, seed=1)):
        t0 = time.perf_counter()
        s = rcdb.structural_signature(rex, {"vertex_labels": labels})
        sig.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        blob = rcdb.serialize_complex(rex)
        ser.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        sink._blobs[(f"x{k}", 1)] = blob
        store.append(time.perf_counter() - t0)
    m = {"signature_ms": round(statistics.median(sig) * 1000, 3),
         "serialize_ms": round(statistics.median(ser) * 1000, 3),
         "store_write_ms": round(statistics.median(store) * 1000, 5)}
    total = sum(m.values())
    m["signature_share"] = round(m["signature_ms"] / total, 3) if total else 0.0
    return m


def bench_backend(kind: str, n: int, root: str) -> dict[str, Any]:
    """One backend, from a cold start, with its own fresh payload."""
    from agent import rcdb

    def _open():
        if kind == "memory":
            return rcdb.MemoryStore()
        if kind == "file":
            return rcdb.FileStore(root)
        if kind == "sqlite":
            return rcdb.SQLStore(f"sqlite:///{root}/rc.sqlite")
        return rcdb.open_store(f"rex://{root}")

    payload = _payload(n, seed=2)
    store = _open()
    # warm the serialization path so it is not charged to whoever goes first
    store.put("__warm__", payload[0][0], meta={"vertex_labels": payload[0][1]})
    store.delete("__warm__")

    t0 = time.perf_counter()
    for k, (rex, labels) in enumerate(payload):
        store.put(f"r{k:06d}", rex,
                  meta={"doc_id": f"r{k:06d}", "vertex_labels": labels})
    ingest = time.perf_counter() - t0

    probe = payload[n // 2][1][0]
    reopen = None if kind == "memory" else _median_ms(_open, 3)
    # MemoryStore has nothing to reopen: calling _open() again would hand back an
    # EMPTY store and every query below would time a scan over nothing. That is
    # exactly the hollow figure this harness exists to prevent, and it was in the
    # harness -- it reported 0.004 ms where the real query takes 25 ms.
    live = store if kind == "memory" else _open()
    out = {
        "records": n,
        "ingest_s": round(ingest, 3),
        "records_per_s": round(n / ingest, 1),
        "reopen_ms": reopen,
        "get_ms": _median_ms(lambda: live.get(f"r{n // 2:06d}"), 20),
        "vocab_query_ms": _median_ms(
            lambda: live.query(labels_any=[probe], limit=20), 20),
        "structural_query_ms": _median_ms(
            lambda: live.query(min_nE=12, limit=10 ** 9), 5),
    }
    assert live.list(limit=1), f"{kind}: the store used for queries is empty"
    if kind == "rex":
        live.write_index()
        out["reopen_indexed_ms"] = _median_ms(_open, 3)
        indexed = _open()
        indexed.query(labels_any=[probe], limit=20)          # first touch
        out["vocab_query_indexed_warm_ms"] = _median_ms(
            lambda: indexed.query(labels_any=[probe], limit=20), 20)
    return out


def bench_analytics(n: int = 2000) -> dict[str, Any] | None:
    """The queries no store can answer, if duckdb is installed."""
    from agent import rcdb
    try:
        from agent import analytics
        if not analytics.available():
            return None
    except ImportError:
        return None

    store = rcdb.MemoryStore()
    for k, (rex, labels) in enumerate(_payload(n, seed=3)):
        store.put(f"r{k:06d}", rex, meta={"doc_id": f"r{k:06d}",
                                          "vertex_labels": labels,
                                          "source": "a" if k % 2 else "b"})
    view = analytics.signature_view(store)
    return {
        "records": n,
        "build_ms": _median_ms(view.refresh, 3),
        "filter_ms": _median_ms(lambda: view.ids("nE >= 12"), 20),
        "aggregate_ms": _median_ms(
            lambda: view.sql("SELECT source, count(*), avg(kappa_mean) "
                             "FROM signatures GROUP BY source"), 20),
        "store_same_filter_ms": _median_ms(
            lambda: store.query(min_nE=12, limit=10 ** 9), 5),
    }


def run(n: int = 2000, backends=("rex", "file", "sqlite", "memory")) -> dict[str, Any]:
    from rexgraph import hardware

    report: dict[str, Any] = {
        "host": hardware.summary(),
        "hardware": hardware.detect(),
        "put_breakdown": bench_put_breakdown(),
        "backends": {},
    }
    for kind in backends:
        with tempfile.TemporaryDirectory() as root:
            report["backends"][kind] = bench_backend(kind, n, root)
    a = bench_analytics(min(n, 2000))
    if a:
        report["analytics"] = a
    return report


def _flatten(report: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in (report.get("put_breakdown") or {}).items():
        out[f"put.{k}"] = v
    for kind, vals in (report.get("backends") or {}).items():
        for k, v in vals.items():
            if isinstance(v, (int, float)):
                out[f"{kind}.{k}"] = v
    for k, v in (report.get("analytics") or {}).items():
        if isinstance(v, (int, float)):
            out[f"analytics.{k}"] = v
    return out


def compare(current: dict[str, Any], recorded: dict[str, Any],
            tolerance: float = TOLERANCE) -> list[dict[str, Any]]:
    """Figures that moved by more than `tolerance`x, slower or faster.

    Faster is reported too: an unexplained speedup usually means the benchmark
    stopped measuring what it thought it was.
    """
    a, b = _flatten(current), _flatten(recorded)
    out = []
    for key in sorted(set(a) & set(b)):
        was, now = b[key], a[key]
        if not was or not now:
            continue
        ratio = now / was
        if ratio > tolerance or ratio < 1.0 / tolerance:
            out.append({"metric": key, "recorded": was, "current": now,
                        "ratio": round(ratio, 2),
                        "direction": "slower" if ratio > 1 else "faster"})
    return out


def _print(report: dict[str, Any]) -> None:
    print(f"host: {report['host']}\n")
    pb = report["put_breakdown"]
    print(f"put breakdown (fresh complex): signature {pb['signature_ms']:.2f} ms | "
          f"serialize {pb['serialize_ms']:.2f} ms | store {pb['store_write_ms']:.4f} ms")
    print(f"  the signature is {pb['signature_share']*100:.0f}% of a put: "
          f"ingest is topology, not storage\n")
    hdr = (f"{'backend':8} {'rec/s':>8} {'reopen':>9} {'get':>8} "
           f"{'vocab-q':>9} {'struct-q':>9}")
    print(hdr)
    print("-" * len(hdr))
    for kind, v in report["backends"].items():
        reo = f"{v['reopen_ms']:.1f}m" if v["reopen_ms"] is not None else "-"
        print(f"{kind:8} {v['records_per_s']:8.0f} {reo:>9} {v['get_ms']:7.3f}m "
              f"{v['vocab_query_ms']:8.3f}m {v['structural_query_ms']:8.2f}m")
    rex = report["backends"].get("rex", {})
    if "reopen_indexed_ms" in rex:
        print(f"\nrex with a tensor index: reopen {rex['reopen_indexed_ms']:.1f} ms "
              f"(vs {rex['reopen_ms']:.1f} replaying), "
              f"warm vocab query {rex['vocab_query_indexed_warm_ms']:.3f} ms")
    a = report.get("analytics")
    if a:
        print(f"\nanalytics over {a['records']} signatures: filter {a['filter_ms']:.2f} ms "
              f"(store: {a['store_same_filter_ms']:.2f} ms), "
              f"aggregate {a['aggregate_ms']:.2f} ms (no store can)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("-n", "--records", type=int, default=2000)
    ap.add_argument("--save", metavar="PATH", help="record this run as JSON")
    ap.add_argument("--compare", metavar="PATH", help="compare against a recorded run")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = run(args.records)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print(report)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nrecorded to {args.save}")
    if args.compare:
        with open(args.compare, encoding="utf-8") as fh:
            recorded = json.load(fh)
        moved = compare(report, recorded, args.tolerance)
        if not moved:
            print(f"\nnothing moved by more than {args.tolerance}x")
        else:
            print(f"\nmoved by more than {args.tolerance}x:")
            for m in moved:
                print(f"  {m['direction']:7} {m['ratio']:5.2f}x  {m['metric']:36} "
                      f"{m['recorded']} -> {m['current']}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
