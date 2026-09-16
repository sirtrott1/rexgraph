#!/usr/bin/env python3
"""Read Git commits as primary relations and check RCQL through an RCDB store.

Run in an installed environment from a neutral directory. The destination must
be new. Git is read only; no checkout, index, production store or install changes.
Only explicitly listed queries execute. This is a dataset experiment, not an
operator coverage or performance claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def history(repo, revision="HEAD", limit=100):
    """Read the last first parent commits, oldest first, with literal path bytes.

    A merge records its difference from its first parent. Renames are a deletion
    and addition of path identities. Empty commits remain metadata, not invented
    relations. The first lexical path is the declared head, not a causal claim.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("history limit must be a positive integer")
    tip = git(repo, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}").decode().strip()
    hashes = git(repo, "rev-list", "--first-parent", "--reverse", f"--max-count={limit}", tip).decode().splitlines()
    commits, identities = [], {}
    for sha in hashes:
        fields = git(repo, "show", "-s", "--format=%ct %P", sha).decode().split()
        if len(fields) > 1:
            data = git(repo, "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                       "--name-only", "-z", fields[1], sha, "--")
        else:
            data = git(repo, "diff-tree", "--root", "--no-commit-id", "--no-renames",
                       "-r", "--name-only", "-z", sha, "--")
        paths = sorted(set(p.decode("utf-8", "surrogateescape") for p in data.split(b"\0") if p))
        identity = int.from_bytes(hashlib.sha256(sha.encode()).digest()[:8], "big") & ((1 << 63) - 1)
        if identity in identities:
            raise ValueError("commit relation ID collision; full commit hashes remain authoritative")
        identities[identity] = sha
        commits.append({"sha": sha, "time": int(fields[0]), "parents": fields[1:],
                        "relation_id": identity, "paths": paths})
    return {"revision": tip, "commits": commits}


def projection(data, min_touches=1):
    """Retain every nonempty projected relation, including witnesses and parallels."""
    if isinstance(min_touches, bool) or not isinstance(min_touches, int) or min_touches <= 0:
        raise ValueError("minimum touch count must be a positive integer")
    counts = Counter(path for commit in data["commits"] for path in commit["paths"])
    files = sorted(path for path, count in counts.items() if count >= min_touches)
    positions = {path: index for index, path in enumerate(files)}
    cells = []
    for commit in data["commits"]:
        support = sorted(positions[path] for path in commit["paths"] if path in positions)
        if support:
            cells.append({**commit, "support": support})
    return files, cells


def snapshot(files, cells):
    import numpy as np
    from rexgraph.graph import RexGraph
    rex = RexGraph.from_cells([len(files), [c["support"] for c in cells]],
                             relation_ids=np.array([c["relation_id"] for c in cells], dtype=np.int64))
    rex._agent_meta = {"vertex_labels": files,
                       "commit_sha_by_relation_id": {str(c["relation_id"]): c["sha"] for c in cells},
                       "orientation": "first lexical path; not causality"}
    return rex


def measure(rex):
    """Use Core actions through RCQL; report residuals independently of identities."""
    import numpy as np
    from rcql import Executor, call, query, source
    from rexgraph.cochain import Cochain
    from rexgraph.linear_operator import boundary_operator

    executor = Executor(sources={"r": rex})
    def read(*expressions):
        return executor.execute(query(source("r"), *expressions)).values

    b0, b1, character, approximate, topology, geometry = read(
        call("BETTI", 0), call("BETTI", 1), call("CHARACTER", True), call("CHARACTER"),
        call("CHANNEL", "T"), call("CHANNEL", "G"))
    dt, dg = topology.diagonal(exact=True), geometry.diagonal(exact=True)
    if not np.array_equal(dt, dg):
        raise AssertionError("T and G channel diagonals disagree")
    norms = {}
    arities = np.diff(rex.boundary_ptr)
    for index, arity in enumerate(arities):
        if int(arity) in norms:
            continue
        (norm,) = read(call("QUADRANCE", call("BOUNDARY", call("CELL", 1, index)), True))
        expected = Fraction(1) if arity == 1 else 1 + Fraction(1, int(arity) - 1)
        if norm != expected:
            raise AssertionError(f"primary boundary quadrance differs at arity {arity}")
        norms[int(arity)] = str(norm)
    exact = np.asarray(character["values"], dtype=object)
    floating = np.asarray(approximate)
    errors = [abs(q - Fraction.from_float(float(f))) for q, f in zip(exact.flat, floating.flat, strict=True)]
    flow = Cochain(1, np.asarray(arities, dtype=float), source=rex)
    (split,) = read(call("HODGE", flow))
    grad, curl, harm = (split[key].values for key in ("gradient", "curl", "harmonic"))
    boundary = boundary_operator(rex, 1)
    hres = float(np.linalg.norm(boundary.apply(harm)))
    reconstruction = float(np.linalg.norm(flow.values - grad - curl - harm))
    scale = max(1.0, float(np.linalg.norm(flow.values)))
    cross = {"gradient_harmonic": float(abs(np.vdot(grad, harm))),
             "gradient_curl": float(abs(np.vdot(grad, curl))),
             "curl_harmonic": float(abs(np.vdot(curl, harm)))}
    # This is an experiment tolerance, not a declaration of exact harmonicity.
    if reconstruction > 1e-8 * scale or hres > 1e-8 * scale or max(cross.values()) > 1e-8 * scale**2:
        raise AssertionError("Hodge residual exceeds the experiment tolerance")
    if np.any(curl != 0):
        raise AssertionError("a source without C2 must have zero curl")
    return {"vertices": rex.nV, "relations": rex.nE, "betti0": b0, "betti1": b1,
            "arity_counts": dict(sorted(Counter(map(int, arities)).items())),
            "diagonal_T_equals_G_exact": True, "boundary_quadrance_by_arity": norms,
            "character_max_absolute_error": str(max(errors, default=Fraction(0))),
            "hodge": {"signal": "relation arity", "reconstruction_norm": reconstruction,
                      "boundary_harmonic_norm": hres, "cross_inner_products": cross,
                      "squared_norms": {key: float(np.vdot(split[key].values, split[key].values).real)
                                        for key in split}, "relative_tolerance": 1e-8,
                      "curl_exact_zero": True}}


def run(repo, output, revision="HEAD", limit=100, min_touches=1, stride=20):
    import numpy as np
    import rexgraph
    from rexgraph.core import _sparse
    from rexgraph.graph import TemporalRex
    from rexgraph.io.catalog import object_digest
    from rcql import Executor, parse
    from rcdb import RexStore

    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("snapshot stride must be a positive integer")
    data = history(repo, revision, limit)
    files, cells = projection(data, min_touches)
    if not cells:
        raise ValueError("history projection has no nonempty relations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "input.json").write_text(json.dumps(data, indent=2) + "\n")
    marks = [*range(stride, len(cells), stride), len(cells)]
    timeline = TemporalRex([])
    records = []
    with closing(RexStore(str(output / "history.rex"))) as store:
        for step, count in enumerate(marks):
            rex = snapshot(files, cells[:count])
            # Git timestamps may tie or reverse. Temporal steps follow ancestry;
            # RCDB valid time retains the unmodified commit clock separately.
            timeline.append_snapshot(rex, at=float(step))
            store.put("history", rex, valid_from=float(cells[count-1]["time"]),
                      meta={"commit": cells[count-1]["sha"], "relations": count})
            records.append({"version": step + 1, "valid_time": cells[count-1]["time"],
                            "state_digest": object_digest(rex), **measure(rex)})
    with closing(RexStore(str(output / "history.rex"))) as store:
        executor = Executor(sources={"db": store, "timeline": timeline})
        for record in records:
            result = executor.execute(parse(
                f'FROM RCDB_VERSION(RCDB("db"), "history", {record["version"]}) '
                'RETURN STATE_HASH(), COUNT(CELLS(1)), BETTI(0), BETTI(1)')).values
            if result != (record["state_digest"], record["relations"], record["betti0"], record["betti1"]):
                raise AssertionError("RCDB version reopen changed the native state")
        commit_time = records[-1]["valid_time"]
        valid_digest = executor.execute(parse(
            f'FROM RCDB_VALID_AT(RCDB("db"), "history", {commit_time}) RETURN STATE_HASH()')).values[0]
        if valid_digest != records[-1]["state_digest"]:
            raise AssertionError("valid time did not select the latest qualifying version")
        earliest_transaction = min(r.tx_from for r in store.history("history"))
        # A commit may have a future timestamp. Only assert absence when the
        # measured commit time actually precedes this store's ingestion clock.
        as_of_absent = None
        if commit_time < earliest_transaction:
            try:
                executor.execute(parse(
                    f'FROM RCDB_AS_OF(RCDB("db"), "history", {commit_time}) RETURN STATE_HASH()'))
            except KeyError:
                as_of_absent = True
            else:
                raise AssertionError("transaction time returned a state before its ingestion")
        transitions = []
        for step in range(1, timeline.T):
            delta, structural, existence, head, orientation, signing = executor.execute(parse(
                f'FROM $timeline LET d = TEMPORAL_DELTA(step={step}) RETURN d, '
                'STRUCTURAL_DELTA(d), EXISTENCE_DELTA(d), HEAD_DELTA(d), ORIENTATION_DELTA(d), SIGNING_DELTA(d)')).values
            expected = {c["relation_id"] for c in cells[marks[step-1]:marks[step]]}
            if set(delta.keys) != expected or any(event.existence != 1 for event in delta.events):
                raise AssertionError("history births lost commit identities")
            previous_ids = set(map(int, delta.previous.relation_ids))
            if not previous_ids < set(map(int, delta.current.relation_ids)):
                raise AssertionError("history did not preserve previous relation identities")
            transitions.append({"step": step, "births": len(expected), "relation_ids": sorted(expected),
                                "channels": [structural, existence, head, orientation, signing]})
    arities = [len(c["support"]) for c in cells]
    report = {"input_revision": data["revision"], "input_commits": len(data["commits"]),
              "min_touches": min_touches, "file_basis": files, "retained_relations": len(cells),
              "omitted_empty_projections": len(data["commits"]) - len(cells),
              "witnesses": arities.count(1), "branching_relations": sum(k > 2 for k in arities),
              "clique_pair_occurrences_not_materialized": sum(k*(k-1)//2 for k in arities),
              "snapshots": records, "transitions": transitions,
              "clocks": {"commit_time": commit_time, "first_transaction": earliest_transaction,
                         "valid_at_matches_latest": True, "as_of_before_ingestion_absent": as_of_absent},
              "interpretation": {"history": "first parent ancestry; merge difference against first parent",
                  "paths": "literal paths, rename means distinct old and new path identities",
                  "projection": "global retrospective frequency filter; witnesses retained",
                  "orientation": "first lexical path is the head; no causal interpretation",
                  "betti0": "dimension C0/im B1, not an incidence connectivity count",
                  "betti1": "dimension ker B1; C2 is absent, no inferred faces",
                  "clique_pairs": "pair occurrences, not unique edges or a runtime comparison",
                  "character": "exact and numerical native readings, not a dense oracle",
                  "clock": "TemporalRex uses snapshot step; RCDB valid time uses Git commit time",
                  "coverage": "named assertions only; not a catalogue coverage percentage"},
              "environment": {"python": sys.executable, "rexgraph": rexgraph.__file__,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "compiled_sparse": _sparse.__file__, "numpy": np.__version__,
                  "versions": {name: importlib.metadata.version(name)
                               for name in ("rexgraph", "rexgraph-rcql", "rexgraph-rcdb")}}}
    def encode(value):
        if isinstance(value, Fraction):
            return {"numerator": value.numerator, "denominator": value.denominator}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unhandled report value {type(value).__name__}")
    (output / "report.json").write_text(json.dumps(report, default=encode, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-touches", type=int, default=1)
    parser.add_argument("--stride", type=int, default=20)
    args = parser.parse_args()
    report = run(args.repo, args.output, args.revision, args.limit, args.min_touches, args.stride)
    print(json.dumps({"revision": report["input_revision"], "relations": report["retained_relations"],
                      "snapshots": len(report["snapshots"]), "report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()
