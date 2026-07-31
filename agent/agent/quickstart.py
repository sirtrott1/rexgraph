"""
agent.quickstart: look at the host, look at the data, and say what to do.

The stack has good defaults but they were scattered across the pieces that hold
them -- hardware detection knows the allocation, the store registry knows which
backends exist, auto_rex knows which files it can read, and nothing put those
together and told you what your particular machine and your particular directory
add up to. On a rented GPU node that answer is worth having before you start
paying rather than after.

    from agent import quickstart
    plan = quickstart.plan("/data/corpus")
    print(plan.summary())            # what is here, what it will do, what is missing
    quickstart.install(plan)         # optional, and it always asks first
    result = quickstart.run(plan)    # ingest -> build -> persist -> index -> ready

Nothing is installed and nothing is written without being asked. `plan()` only
reads: it is safe to run anywhere, including somewhere you are only inspecting.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: file types worth counting separately when deciding what a directory IS.
_TEXTUAL = {".txt", ".md", ".rst", ".json", ".csv", ".tsv", ".pdf"}

#: optional packages, what they unlock, and whether anything here needs them.
OPTIONAL = {
    "duckdb": "columnar queries over signatures (group-by, aggregates)",
    "s3fs": "s3:// object storage",
    "gcsfs": "gs:// object storage",
    "adlfs": "az:// object storage",
    "h5py": "HDF5 and AnnData (.h5ad) containers",
    "pyarrow": "Arrow export, parquet",
    "sqlalchemy": "sql:// backends (postgres, mysql, sqlite)",
}


def _have(module: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Plan:
    """What was found and what will happen. Inspect it before running it."""

    path: str
    host: Dict[str, Any]
    files: Dict[str, int]
    n_files: int
    total_bytes: int
    readable: int
    unreadable: List[str]
    backend: str
    backend_reason: str
    store_uri: str
    depth: str
    depth_reason: str
    missing: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def install_command(self) -> Optional[str]:
        if not self.missing:
            return None
        return f"{sys.executable} -m pip install " + " ".join(sorted(self.missing))

    def summary(self) -> str:
        gb = self.total_bytes / 2 ** 30
        lines = [
            f"host    {self.host['summary']}",
            f"data    {self.n_files} files, {gb:.2f} GiB, in {self.path}",
        ]
        if self.files:
            top = sorted(self.files.items(), key=lambda kv: -kv[1])[:6]
            lines.append("        " + ", ".join(f"{ext or '(no ext)'} x{n}"
                                                for ext, n in top))
        if self.unreadable:
            lines.append(f"        {len(self.unreadable)} not readable by any adapter: "
                         + ", ".join(sorted(set(self.unreadable))[:6]))
        lines += [
            f"store   {self.backend} -> {self.store_uri}",
            f"        {self.backend_reason}",
            f"depth   {self.depth} ({self.depth_reason})",
        ]
        for note in self.notes:
            lines.append(f"note    {note}")
        if self.missing:
            lines.append("missing " + ", ".join(f"{k} ({v})" for k, v in
                                                sorted(self.missing.items())))
            lines.append(f"        {self.install_command()}")
        return "\n".join(lines)


def _scan(path: str, *, max_files: int = 200_000) -> Dict[str, Any]:
    """Count files by extension and total size. Reads no file contents."""
    from agent.adapters.formats import reader_for

    counts: Dict[str, int] = {}
    unreadable: List[str] = []
    readable = 0
    total = 0
    n = 0
    known_text = _TEXTUAL
    root = Path(path)
    if root.is_file():
        entries = [root]
    else:
        entries = (p for p in root.rglob("*") if p.is_file())
    for p in entries:
        n += 1
        if n > max_files:
            break
        ext = p.suffix.lower()
        counts[ext] = counts.get(ext, 0) + 1
        try:
            total += p.stat().st_size
        except OSError:
            pass
        if ext in known_text or reader_for(p) is not None:
            readable += 1
        else:
            unreadable.append(ext)
    return {"counts": counts, "n_files": n, "total_bytes": total,
            "readable": readable, "unreadable": unreadable}


def plan(path: str = ".", *, store: Optional[str] = None) -> Plan:
    """Inspect the host and the data, and decide. Reads only; changes nothing."""
    from rexgraph import hardware
    from agent import rcdb

    hw = hardware.detect()
    hw["summary"] = hardware.summary()
    scan = _scan(path)

    store_root = store or os.path.join(
        path if os.path.isdir(path) else os.path.dirname(path) or ".", ".rexdb")
    rec = rcdb.recommend_backend(store_root, uri=store or "")
    backend = rec["backend"]
    uri = store if (store and "://" in store) else f"{backend}://{store_root}"

    # depth follows the work, not a preference: the signature is ~94% of a put, so
    # the number of files is what decides whether the full pipeline is affordable.
    n = max(scan["readable"], 1)
    if n <= 200:
        depth, why = "standard", f"{n} readable files: the full pipeline is cheap here"
    elif n <= 5000:
        depth, why = "standard", f"{n} readable files"
    else:
        depth, why = "quick", (f"{n} readable files: analysis is ~94% of ingest, so "
                               f"quick first and deepen what matters")

    missing = {}
    if not _have("duckdb"):
        missing["duckdb"] = OPTIONAL["duckdb"]
    if ".h5ad" in scan["counts"] or ".loom" in scan["counts"] or ".h5" in scan["counts"]:
        if not _have("h5py"):
            missing["h5py"] = OPTIONAL["h5py"]
    scheme = (uri.split("://", 1)[0] if "://" in uri else "")
    for sch, mod in (("s3", "s3fs"), ("gs", "gcsfs"), ("gcs", "gcsfs"),
                     ("az", "adlfs"), ("abfs", "adlfs")):
        if scheme == sch and not _have(mod):
            missing[mod] = OPTIONAL[mod]

    notes = []
    if hw["gpu_count"]:
        notes.append(f"{hw['gpu_count']} gpu visible; run "
                     f"`python -m rexgraph.gpu_preflight` before a long job")
    else:
        notes.append("no gpu visible: every path falls back to cpu, which is exact "
                     "but slower on the character/propagator solves")
    if hw.get("scheduler"):
        notes.append(f"{hw['scheduler']} allocation detected; threads follow it "
                     f"({hw['cpus']} cpu) rather than the node")
    if hw["cloud"]["provider"]:
        notes.append(f"running on {hw['cloud']['provider']}"
                     + (" in kubernetes" if hw["cloud"]["kubernetes"] else ""))
    if scan["unreadable"]:
        notes.append("unreadable types are skipped, not fatal")

    return Plan(path=str(path), host=hw, files=scan["counts"], n_files=scan["n_files"],
                total_bytes=scan["total_bytes"], readable=scan["readable"],
                unreadable=scan["unreadable"], backend=backend,
                backend_reason=rec["reason"], store_uri=uri, depth=depth,
                depth_reason=why, missing=missing, notes=notes)


def install(p: Plan, *, yes: bool = False) -> Dict[str, Any]:
    """Install what the plan says is missing. Asks first unless `yes`."""
    cmd = p.install_command()
    if not cmd:
        return {"installed": [], "skipped": True, "reason": "nothing missing"}
    if not yes:
        return {"installed": [], "skipped": True,
                "reason": "not confirmed", "command": cmd}
    proc = subprocess.run(cmd.split(), capture_output=True, text=True)
    return {"installed": sorted(p.missing) if proc.returncode == 0 else [],
            "skipped": False, "returncode": proc.returncode,
            "stderr": proc.stderr[-2000:] if proc.returncode else ""}


def run(p: Plan, *, limit: Optional[int] = None, persist: bool = True,
        progress=None) -> Dict[str, Any]:
    """Ingest, build, persist and index, on the plan's own terms.

    Returns the corpus and the store. The store is indexed before returning, so the
    next process to open it memory-maps rather than replaying: with the pipeline
    ending here, that is the one moment it is certainly worth paying for.
    """
    import time

    from agent import rcdb
    from agent.corpus import CorpusBuilder

    t0 = time.perf_counter()
    corpus = CorpusBuilder()
    added = corpus.add_directory(p.path) if os.path.isdir(p.path) else \
        [corpus.add_document(p.path)]
    if limit:
        corpus.documents = corpus.documents[:limit]
    if progress:
        progress(f"ingesting {len(corpus.documents)} documents at depth={p.depth}")
    corpus.build(depth=p.depth)
    built = time.perf_counter() - t0

    store = None
    ids: List[str] = []
    if persist:
        store = rcdb.open_store(p.store_uri)
        ids = corpus.persist(store)
        if hasattr(store, "write_index"):
            # the pipeline ends here, so this is the moment the snapshot is
            # certainly worth its cost: every later open memory-maps instead.
            store.write_index()
    return {
        "corpus": corpus, "store": store, "ids": ids,
        "n_documents": len(corpus.documents),
        "n_added": len(added),
        "build_seconds": round(built, 2),
        "total_seconds": round(time.perf_counter() - t0, 2),
        "store_uri": p.store_uri if persist else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--store", help="store uri or directory (default: <path>/.rexdb)")
    ap.add_argument("--run", action="store_true", help="execute the plan")
    ap.add_argument("--install", action="store_true", help="install what is missing")
    ap.add_argument("--limit", type=int, help="cap the documents ingested")
    args = ap.parse_args(argv)

    p = plan(args.path, store=args.store)
    print(p.summary())
    if args.install:
        out = install(p, yes=True)
        print(f"\ninstall: {out}")
        p = plan(args.path, store=args.store)
    if args.run:
        print()
        out = run(p, limit=args.limit, progress=lambda m: print(f"        {m}"))
        print(f"\nbuilt {out['n_documents']} documents in {out['build_seconds']}s")
        if out["store_uri"]:
            print(f"stored {len(out['ids'])} records in {out['store_uri']}")
    elif not args.install:
        print("\n(nothing was changed; pass --run to execute)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
