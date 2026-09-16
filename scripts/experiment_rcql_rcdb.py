#!/usr/bin/env python3
"""Reproducible local RCQL/RCDB lifecycle acceptance, not a scale benchmark.

Run with a non editable platform environment from a neutral directory::

    /path/to/venv/bin/python -I /path/to/experiment_rcql_rcdb.py \
        --output /new/disposable/directory --code-revision <wheel-source-sha>

The output directory must not exist. No production stores or installation are
changed. RCQL queries, checks, exact values, timings and fresh process reopen
results remain in report.json. Documented unavailable workflows are recorded as
UNSUPPORTED, not silently implemented in Python. Core assisted candidate creation,
file export and cross store copy are explicitly identified. All optional local
backends/formats used here must be installed; missing dependencies are failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction as Q
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from rcql import Executor, parse
from rcdb import FileStore, MemoryStore, ObjectStore, RexStore, SQLStore, copy_record
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.io import load, save
from rexgraph.io.catalog import FileCatalog, object_digest
from rexgraph.native_sparse import csr_carrier, empty_native


BACKENDS = ("memory", "file", "rex", "sql", "object")
FORMATS = ("rcbd", "rex", "safetensors", "h5", "zarr")


def json_value(value):
    if isinstance(value, Q):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "values": json_value(value.tolist())}
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (RexGraph, TemporalRex)):
        return {"type": type(value).__name__, "state_digest": object_digest(value)}
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "display": str(value)}


def open_backend(kind, root):
    root.mkdir(parents=True, exist_ok=True)
    if kind == "memory":
        return MemoryStore()
    if kind == "file":
        return FileStore(str(root / "file"))
    if kind == "rex":
        return RexStore(str(root / "rex"))
    if kind == "sql":
        return SQLStore(f"sqlite:///{root / 'sql.db'}")
    if kind == "object":
        return ObjectStore(f"file://{root / 'object'}")
    raise ValueError(kind)


def fixtures():
    def branch(sign=1):
        r = RexGraph(boundary_ptr=np.array([0, 4, 6], np.int32),
                     boundary_idx=np.array([2, 0, 1, 3, 0, 2], np.int32),
                     w_E=np.array([Q(2, 3), sign * Q(5, 7)], dtype=object))
        r._agent_meta = {"vertex_labels": ["alpha", "βeta", "head", "tail"],
                         "source": "disposable RCQL lifecycle experiment"}
        r.attach_metadata(1, 0, "author/note", "branch → relation; not a clique")
        r.attach_metadata(0, 0, "schema", RexGraph.from_graph([0], [1]))
        return r
    n = 2**60
    high = RexGraph.from_graph([0], [1])
    high._graded_duals = [empty_native((0, 2)).dual,
        csr_carrier(np.array([0, 2, 4], np.int32), np.array([0, 1, 0, 1], np.int32),
                    np.array([n, n+1, n-1, n], np.int64), (2, 2))]
    triangle = RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    witness = RexGraph(boundary_ptr=np.array([0, 1], np.int32),
                       boundary_idx=np.array([0], np.int32))
    return {"branch": branch(), "negative": branch(-1), "triangle": triangle,
            "tower": high, "witness": witness, "empty": RexGraph.from_graph([], [])}


class Experiment:
    def __init__(self):
        self.events = []
        self.queries = []
        self.case = "setup"

    def check(self, condition, label, **details):
        if not condition:
            raise AssertionError(f"{label}: {json_value(details)}")
        self.events.append({"case": self.case, "status": "PASS", "check": label,
                            **json_value(details)})

    def run(self, name, fn):
        parent_case = self.case
        self.case = name
        start = time.perf_counter()
        try:
            fn()
        except Exception as exc:
            self.events.append({"case": name, "status": "FAIL", "error": str(exc),
                                "traceback": traceback.format_exc()})
        elapsed = time.perf_counter() - start
        failures = [e for e in self.events if e["case"] == name and e["status"] == "FAIL"]
        print(f"{'FAIL' if failures else 'PASS'} {name} ({elapsed:.3f}s)", flush=True)
        self.events.append({"case": name, "status": "TIMING", "seconds": elapsed})
        self.case = parent_case

    def query(self, text, sources, params=None):
        row = {"case": self.case, "query": text}
        self.queries.append(row)
        start = time.perf_counter()
        try:
            result = Executor(sources=sources, params=params).execute(parse(text))
            row.update(status="OK", values=json_value(result.values),
                       exactness=json_value(result.exactness),
                       aliases=json_value(result.aliases),
                       execution=json_value(result.execution),
                       provenance=json_value(result.provenance),
                       native_plan=json_value(result.native_plan))
            return result
        except Exception as exc:
            row.update(status="ERROR", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            row["seconds"] = time.perf_counter() - start

    def unavailable(self, text, sources, params, errors, reason):
        try:
            self.query(text, sources, params)
        except errors as exc:
            self.events.append({"case": self.case, "status": "UNSUPPORTED",
                                "query": text, "reason": reason,
                                "observed": f"{type(exc).__name__}: {exc}"})
        else:
            raise AssertionError(f"Expected documented boundary no longer applies: {text}")

    def maths(self, name, rex, prefix="FROM $r", sources=None):
        sources = {"r": rex} if sources is None else sources
        result = self.query(prefix + " RETURN STATE_HASH(), RANK(1), NULLITY(1), "
                            "BETTI(0), BETTI(1)", sources)
        expected = {"branch": (2, 0, 2, 0), "negative": (2, 0, 2, 0),
                    "triangle": (2, 1, 1, 0), "tower": (1, 0, 1, 0),
                    "witness": (1, 0, 0, 0), "empty": (0, 0, 0, 0)}[name]
        self.check(result.values == (object_digest(rex), *expected),
                   "state identity and hand-predicted rank/nullity/Betti", fixture=name)
        if name in {"branch", "negative"}:
            expected_channels = {
                "T": [Q(16, 27), Q(-40, 63)], "G": [Q(16, 27), Q(40, 63)],
                "F": [Q(80, 63), Q(-80, 63)], "C": [Q(4, 3), Q(-4, 3)]}
            for channel, expected in expected_channels.items():
                if name == "negative" and channel != "C":
                    expected = [expected[0], -expected[1]]
                action = self.query(prefix + f' RETURN APPLY(CHANNEL("{channel}"), '
                                    'INDICATOR(CELL(1,0)), true)', sources)
                self.check(action.values[0].values.tolist() == expected and
                           action.exactness[0].value == "rational",
                           "exact full channel action", channel=channel, expected=expected)
            desc = self.query(prefix + ' AS r LET selected = r.CELLS(indices=[0],grade=1) '
                              'RETURN ARITY(CELL(1,0)) AS arity, HEAD(CELL(1,0)), '
                              'SHARE(CELL(1,0)), SHARE_SUPPORT(CELL(1,0)), '
                              'INDICATOR(selected), r.DESCRIBE().nE AS relations', sources)
            self.check(desc.named_values == {"arity": 4, "relations": 2},
                       "relation navigation, source alias, LET and named fields")
        if name == "tower":
            result = self.query(prefix + " RETURN RANK(4), NULLITY(4), BETTI(3), BETTI(4)", sources)
            self.check(result.values == (2, 0, 0, 0), "determinant-one integer tower above 2**53")


def file_case(exp, root, name, value, suffix):
    path = root / "files" / "nested β with spaces" / f"{name}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    save(str(path), value)
    restored = load(str(path))
    if isinstance(value, TemporalRex):
        from rexgraph.io.temporal_state import to_temporal_state
        original_state, restored_state = to_temporal_state(value), to_temporal_state(restored)
        changed_headers = {key: {"original": original_state.header.get(key),
                                 "restored": restored_state.header.get(key)}
                           for key in original_state.header.keys() | restored_state.header.keys()
                           if original_state.header.get(key) != restored_state.header.get(key)}
        changed_tensors = [key for key in original_state.tensors.keys() | restored_state.tensors.keys()
                           if key not in original_state.tensors or key not in restored_state.tensors or
                           original_state.tensors[key].dtype != restored_state.tensors[key].dtype or
                           not np.array_equal(original_state.tensors[key], restored_state.tensors[key])]
        exp.events.append({"case": exp.case, "status": "MEASUREMENT", "path": str(path),
                           "temporal_header_differences": changed_headers,
                           "temporal_tensor_differences": changed_tensors})
    exp.check(object_digest(restored) == object_digest(value),
              "core-assisted full-state file roundtrip", path=str(path))
    if isinstance(value, RexGraph):
        exp.maths(name, restored)
    catalog = FileCatalog([root / "files"])
    relative = "root0/" + path.relative_to(root / "files").as_posix()
    before = catalog.hash(relative)
    if isinstance(value, RexGraph):
        exp.maths(name, value, f'FROM FILE("files", {json.dumps(relative)})', {"files": catalog})
    else:
        hashed = exp.query(f'FROM FILE("files", {json.dumps(relative)}) RETURN STATE_HASH()', {"files": catalog})
        exp.check(hashed.values == (object_digest(value),), "whole temporal file canonical hash")
        for step in range(value.T):
            expected = value.reconstruct_at(step)
            for source in (f'AT(FILE("files", {json.dumps(relative)}), {step})',
                           f'AT_TIME(FILE("files", {json.dumps(relative)}), {value._times[step]})'):
                result = exp.query(f'FROM {source} RETURN STATE_HASH(), RANK(1)', {"files": catalog})
                exp.check(result.values == (object_digest(expected), step+1),
                          "RCQL selects and queries temporal file snapshots", step=step)
    exp.check(catalog.hash(relative) == before, "read-only RCQL leaves physical file hash unchanged")
    if suffix == "safetensors":
        qname = json.dumps(relative)
        result = exp.query(f'FROM CATALOG("files") RETURN FILE_INFO({qname}), '
                           f'FILE_HASH({qname}), TENSORS({qname}, limit=1000), '
                           f'SEARCH_TENSORS({qname}, "boundary", limit=100)', {"files": catalog})
        exp.check(result.values[1] == before and bool(result.values[2]), "file and tensor metadata navigation")
        exp.check(all("boundary" in t["name"] for t in result.values[3]), "literal tensor name search")


def store_case(exp, root, kind, values, timeline, manifests):
    store = open_backend(kind, root)
    sources = {"db": store}
    try:
        # Exercise the read/serialization contract separately from the mutation
        # package contract, so a failed exact commit cannot hide read coverage.
        for name, value in values.items():
            store.put(name, value, analytics=False)
            exp.maths(name, value, f'FROM RCDB_GET($db, "{name}")', sources)
        store.put("timeline", timeline, analytics=False)
        temporal_hash = exp.query('FROM RCDB("db") RETURN RCDB_HASH("timeline")', sources)
        exp.check(temporal_hash.values == (object_digest(timeline),), "core-assisted whole-history store roundtrip")
        for step in range(timeline.T):
            result = exp.query(f'FROM AT(RCDB_GET($db,"timeline"),{step}) RETURN STATE_HASH(), RANK(1)', sources)
            exp.check(result.values == (object_digest(timeline.reconstruct_at(step)), step+1),
                      "RCQL navigates stored temporal snapshots", step=step)
        store.configure_security(require_commits=True)
        for name, value in values.items():
            def commit_fixture(name=name, value=value):
                target = f"committed/{name}"
                query = f'FROM RCDB("db") MUTATE "{target}" SET state=$candidate, actor="experiment", '
                query += 'expected_version=0, valid_from=10, valid_to=20 COMMIT'
                before = store.state_digest()
                plan = exp.query("EXPLAIN " + query, sources, {"candidate": value})
                exp.check(not plan.execution and store.state_digest() == before,
                          "EXPLAIN performs no publication", record=target)
                try:
                    commit = exp.query(query, sources, {"candidate": value})
                except Exception:
                    exp.check(store.state_digest() == before and store.read_record(target) is None and
                              store.commit_history(target) == [], "failed commit published neither state nor lineage")
                    raise
                exp.check(commit.values[0].version == 1 and store.verify_commits(target),
                          "RCQL creates a verifiable first version", record=target)
                exp.maths(name, value, f'FROM RCDB_GET($db, "{target}")', sources)
            exp.run(f"commit/{kind}/{name}", commit_fixture)
        # Candidate transformations are core/Python work. The commits themselves
        # (including reading a source record into LET) are RCQL operations.
        result = exp.query('FROM RCDB("db") LET candidate=RCDB_GET("triangle") '
                           'MUTATE "workflow" SET state=candidate, actor="genesis", '
                           'expected_version=0, valid_from=10, valid_to=20 COMMIT', sources)
        exp.check(result.values[0].version == 1, "RCQL creates workflow record from a selected record")
        result = exp.query('FROM RCDB("db") LET candidate=RCDB_GET("tower") '
                           'MUTATE "workflow" SET state=candidate, actor="replacement", '
                           'expected_version=1, valid_from=20, valid_to=30 COMMIT', sources)
        exp.check(result.values[0].version == 2, "RCQL copies a selected record into a replacement version")
        result = exp.query('FROM RCDB("db") LET candidate=RCDB_GET("witness") '
                           'MUTATE "workflow" SET state=candidate, actor="structural replacement", '
                           'expected_version=2, valid_from=30 COMMIT', sources)
        exp.check(result.values[0].version == 3, "RCQL replacement changes relation and face structure")
        # Both read and write are RCQL; the host passes a returned native value
        # between two statements (not a claimed multi statement IMPORT grammar).
        catalog = FileCatalog([root.parent / "files"])
        from_file = exp.query('FROM FILE("files", "root0/nested β with spaces/triangle.rcbd") '
                              'AS document RETURN document, STATE_HASH()', {"files": catalog})
        document = from_file.values[0]
        exp.check(object_digest(document) == from_file.values[1], "RCQL ingestion selects queried file state")
        imported = exp.query('FROM RCDB("db") MUTATE "from-file" SET state=$document, '
                             'expected_version=0 COMMIT', sources, {"document": document})
        exp.check(imported.values[0].version == 1 and
                  store.read_record("from-file").state_digest == object_digest(values["triangle"]),
                  "RCQL file candidate is committed through RCQL")
        before = store.state_digest()
        try:
            exp.query('FROM RCDB("db") MUTATE "workflow" SET state=$candidate, '
                      'expected_version=1 COMMIT', sources, {"candidate": values["empty"]})
        except Exception as exc:
            from rcdb import VersionConflictError
            if not isinstance(exc, VersionConflictError):
                raise
            exp.check(store.state_digest() == before, "stale version rejected without logical-store change")
        else:
            raise AssertionError("stale update succeeded")
        history = store.history("workflow")
        by_version = {r.version: r for r in history}
        for version, name, valid_at in ((1, "triangle", 15), (2, "tower", 20), (3, "witness", 30)):
            for source in (f'RCDB_VERSION($db,"workflow",{version})',
                           f'RCDB_AS_OF($db,"workflow",{by_version[version].tx_from!r})',
                           f'RCDB_VALID_AT($db,"workflow",{valid_at})'):
                exp.maths(name, values[name], f"FROM {source}", sources)
        result = exp.query('FROM RCDB("db") RETURN RCDB_LIST(limit=100), '
                           'RCDB_LIST(limit=2,offset=1), RCDB_SEARCH("alpha"), '
                           'RCDB_HISTORY("workflow"), RCDB_STATS(), RCDB_HASH("workflow"), '
                           'RCDB_COMMITS("workflow"), RCDB_VERIFY("workflow"), RCDB_STATE_HASH()', sources)
        rows, page, search, history_view, stats, digest, commits, verified, logical = result.values
        exp.check(len(rows) == len(store.list()) and len(page) == 2, "record listing and pagination")
        exp.check(any(r["id"] == "negative" for r in search), "label search finds stored relation state")
        exp.check(len(history_view) == 3 and verified and digest == object_digest(values["witness"]),
                  "history and current version identity")
        exp.check(len(commits) == 3 and commits[1]["parent"] == commits[0]["link"] and
                  commits[2]["parent"] == commits[1]["link"] and
                  commits[1]["previous_state"] == object_digest(values["triangle"]) and
                  commits[1]["resulting_state"] == object_digest(values["tower"]),
                  "commit parent links and exact transition endpoint identities")
        # One detached read cannot mutate the stored payload or metadata.
        snap = store.read_record("negative")
        snap.value.w_E[0] = Q(99)
        snap.record.signature["nE"] = 999
        exp.check(store.read_record("negative").state_digest == object_digest(values["negative"]) and
                  store.read_record("negative").record.signature["nE"] == 2,
                  "read snapshot edits cannot modify published state")
        exported = root / "exports" / f"{kind}-current.rcbd"
        exported.parent.mkdir(exist_ok=True)
        save(str(exported), store.read_record("workflow").value)
        catalog = FileCatalog([exported.parent])
        exp.maths("witness", values["witness"],
                  f'FROM FILE("files","root0/{exported.name}")', {"files": catalog})
        exp.check(store.state_digest() == logical, "core-assisted export does not alter store")
        manifests[kind] = {"logical": logical, "records": {
            record.id: {"digest": store.read_record(record.id).state_digest, "version": record.version}
            for record in store.list()},
            "workflow_versions": [object_digest(values[name]) for name in ("triangle", "tower", "witness")]}
    finally:
        store.close()


def reopen(exp, root, kind, manifest):
    store = open_backend(kind, root)
    try:
        result = exp.query('FROM RCDB("db") RETURN RCDB_STATE_HASH(), RCDB_VERIFY("workflow")', {"db": store})
        exp.check(result.values == (manifest["logical"], True), "fresh handle preserves logical state and lineage")
        for name, expected in manifest["records"].items():
            snapshot = store.read_record(name)
            exp.check(snapshot.state_digest == expected["digest"] and
                      snapshot.record.version == expected["version"], "persisted record", record=name)
        values = fixtures()
        for name in ("tower", "branch", "negative"):
            exp.maths(name, values[name], f'FROM RCDB_GET($db,"{name}")', {"db": store})
        for step in range(2):
            result = exp.query(f'FROM AT(RCDB_GET($db,"timeline"),{step}) RETURN RANK(1)', {"db": store})
            exp.check(result.values == (step+1,), "fresh-process temporal snapshot query", step=step)
        for v, digest in enumerate(manifest["workflow_versions"], 1):
            result = exp.query(f'FROM RCDB_VERSION($db,"workflow",{v}) RETURN STATE_HASH()', {"db": store})
            exp.check(result.values == (digest,), "persisted historical version", version=v)
    finally:
        store.close()


def copy_cases(exp, root, values):
    # All 20 directed pairs, using RCDB's single authorized migration primitive.
    sources = {k: open_backend(k, root / "copy-source") for k in BACKENDS}
    try:
        for store in sources.values():
            store.put("copy", values["tower"], meta={"nested": {"unicode": "β"}},
                      tags=["roundtrip"], valid_from=10, valid_to=20, analytics=False)
            store.put("copy", values["branch"], analytics=False)
        for left in BACKENDS:
            for right in BACKENDS:
                if left == right:
                    continue
                def copy(left=left, right=right):
                    dst = open_backend(right, root / f"copy-{left}-to-{right}").configure_security(require_commits=True)
                    try:
                        selected = sources[left].read_record("copy", version=1).record
                        copied = copy_record(sources[left], dst, selected, actor="experiment copy", expected_version=0)
                        snap = dst.read_record("copy")
                        exp.check(copied.version == 1 and snap.state_digest == object_digest(values["tower"]),
                                  "core-assisted cross-backend copy uses selected old version")
                        exp.check(snap.record.meta == {"nested": {"unicode": "β"}} and
                                  snap.record.valid_from == 10 and snap.record.valid_to == 20 and
                                  "roundtrip" in snap.record.signature["tags"] and dst.verify_commits("copy"),
                                  "copied metadata, tags, valid interval and fresh destination lineage")
                        exp.maths("tower", values["tower"], 'FROM RCDB_GET($db,"copy")', {"db": dst})
                    finally:
                        dst.close()
                exp.run(f"copy/{left}-to-{right}", copy)
    finally:
        for store in sources.values():
            store.close()


def navigation(exp, root, values, manifests):
    catalog = FileCatalog([root / "files"])
    result = exp.query('FROM CATALOG("files") RETURN FILES(limit=1000), '
                       'FILES(limit=2,offset=1), SEARCH("branch"), HASH_FILES()', {"files": catalog})
    all_rows, page, found, hashed = result.values
    exp.check(len(all_rows) == 5 * (len(values) + 1) and len(page) == 2,
              "catalog discovers supported static and temporal files, pagination", entries=len(all_rows))
    exp.check(len(found) == 5, "catalog literal file search")
    exp.check(all(e.sha256 for e in catalog.list(limit=1000)), "hash all cataloged files")
    # A directory store is loadable through the catalog's explicit RCDB loader
    # injection; there is no core -> RCDB package dependency.
    handles = []
    def rcdb_loader(path):
        store = RexStore(str(path)) if Path(path).name == "rex" else FileStore(str(path))
        handles.append(store)
        return store
    stores = FileCatalog([root / "stores"], loaders={"rcdb": rcdb_loader})
    try:
        for kind in ("rex", "file"):
            result = exp.query(f'FROM FILE("files","root0/{kind}") RETURN RCDB_HASH("workflow"), '
                               'RCDB_VERIFY("workflow")', {"files": stores})
            exp.check(result.values == (object_digest(values["witness"]), True),
                      "catalog navigation into an injected RCDB directory", backend=kind)
            before = stores.hash(f"root0/{kind}")
            committed = exp.query(f'FROM FILE("files","root0/{kind}") '
                                  'LET candidate=RCDB_GET("triangle") MUTATE "catalog-edit" '
                                  'SET state=candidate, expected_version=0, actor="catalog edit" COMMIT',
                                  {"files": stores})
            exp.check(committed.values[0].version == 1 and stores.hash(f"root0/{kind}") != before,
                      "RCQL mutation through cataloged RCDB directory changes its physical files", backend=kind)
            exp.maths("triangle", values["triangle"],
                      f'FROM RCDB_GET(FILE("files","root0/{kind}"),"catalog-edit")', {"files": stores})
            state = exp.query(f'FROM FILE("files","root0/{kind}") RETURN RCDB_STATE_HASH(), '
                               'RCDB_VERIFY("catalog-edit")', {"files": stores})
            exp.check(state.values[1], "catalog mutation has persisted verifiable lineage", backend=kind)
            manifests[kind]["logical"] = state.values[0]
            manifests[kind]["records"]["catalog-edit"] = {"digest": object_digest(values["triangle"]), "version": 1}
    finally:
        for store in handles:
            store.close()


def native_editing(exp, root, values):
    for suffix in FORMATS:
        path = root / 'editable' / f'branch.{suffix}'
        path.parent.mkdir(exist_ok=True)
        save(str(path), values['branch'])
        catalog = FileCatalog([path.parent])
        name = f'root0/branch.{suffix}'
        read = exp.query(f'FROM FILE("files","{name}") AS document RETURN document, STATE_HASH()', {"files": catalog})
        before = catalog.hash(name)
        params = {'document': read.values[0]}
        text = f'FROM CATALOG("files") MUTATE "{name}" SET state=$document, expected_hash="{before}" REMOVE [1] ADD [[0,1,2,3,4]] COMMIT'
        explained = exp.query('EXPLAIN ' + text, {'files': catalog}, params)
        exp.check(not explained.execution and catalog.hash(name) == before, 'file edit EXPLAIN does not publish')
        committed = exp.query(text, {'files': catalog}, params).values[0]
        reopened = FileCatalog([path.parent])
        result = exp.query(f'FROM FILE("files","{name}") MATCH e IN CELLS(1) WHERE ARITY(e) >= 4 RETURN e.index, ARITY(e) ORDER BY e.index', {'files': reopened})
        exp.check(result.values == (((0, 4), (1, 5)),), 'native RCQL edit and MATCH after file reopen', format=suffix)
        backup = path.parent / committed.backup.split('/', 1)[1]
        exp.check(object_digest(load(str(backup))) == read.values[1], 'file recovery copy retains original exact state')
    for kind in BACKENDS:
        store = open_backend(kind, root / 'inline' / kind).configure_security(require_commits=True)
        try:
            text = 'FROM $db MUTATE "r" SET state=$r, expected_version=0 REMOVE [1] ADD [[0,1,2,3,4]] COMMIT'
            exp.query(text, {'db': store}, {'r': values['branch']})
            result = exp.query('FROM RCDB_GET($db,"r") MATCH e IN CELLS(1) WHERE ARITY(e) >= 4 RETURN e.index, ARITY(e)', {'db': store})
            exp.check(result.values == (((0, 4), (1, 5)),) and store.verify_commits('r'), 'inline structural edit has native results and verified RCDB lineage', backend=kind)
        finally:
            store.close()


def sheaf_case(exp, root, values):
    """Assigned local section diagnostics across selected store backed stalks."""
    from rcql import PhraseSheaf, PhraseStalk, PhraseCorrespondence, SourceRef, SourcePolicy, bind
    store = open_backend("rex", root / "stores")
    try:
        snapshots = [store.read_record(name, version=1) for name in ("branch", "negative")]
        stalks = tuple(PhraseStalk(name, bind(name, snap.value, SourcePolicy.allow("read", "identity"),
                                             source_ref=SourceRef(name=name, record_id=name, record_version=1,
                                                                  state_digest=snap.state_digest)))
                       for name, snap in zip(("branch", "negative"), snapshots, strict=True))
        section = PhraseSheaf(stalks, (PhraseCorrespondence("shared", ("branch", "negative")),),
                              stalk_dims={"branch": 2, "negative": 2}, correspondence_dims={"shared": 1})
        for name, snap, sign in zip(("branch", "negative"), snapshots, (1, -1), strict=True):
            section.assign(name, snap.value.edge_metric_exact)
            section.restrict(name, "shared", [[0, sign]])
        text = 'FROM PHRASE($section) RETURN SECTION_CHECK($section), GLUE($section).ratio'
        result = exp.query(text, {}, {"section": section})
        exp.check(result.values[0].compatible and result.values[1] == Q(1),
                  "explicit rectangular restrictions reconcile selected rational stalks")
        exp.check([r.state_digest for r in result.values[0].contributors] ==
                  [snap.state_digest for snap in snapshots], "section contributors keep selected state identities")
        section.restrict("negative", "shared", [[0, 1]])
        result = exp.query(text, {}, {"section": section})
        exp.check(not result.values[0].compatible and result.values[1] == Q(0),
                  "mismatching restriction reports obstruction, not a false global glue")
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", default="unspecified; caller must identify wheel source")
    parser.add_argument("--reopen-one", choices=BACKENDS[1:])
    args = parser.parse_args()
    root = args.output.resolve()
    exp = Experiment()
    if args.reopen_one:
        manifests = json.loads((root / "reopen.json").read_text())
        exp.run(f"fresh-process/{args.reopen_one}",
                lambda: reopen(exp, root / "stores", args.reopen_one, manifests[args.reopen_one]))
        report = {"events": exp.events, "queries": exp.queries}
        (root / f"reopen-{args.reopen_one}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return int(any(e["status"] == "FAIL" for e in exp.events))
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    modules = {name: str(Path(importlib.import_module(name).__file__).resolve())
               for name in ("rexgraph", "rcql", "rcdb", "rexgraph.core._boundary", "rexgraph.core._sparse")}
    versions = {}
    for name in ("rexgraph", "rexgraph-rcql", "rexgraph-rcdb", "numpy", "scipy", "safetensors", "h5py", "zarr", "sqlalchemy", "fsspec"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    environment = {"python": sys.executable, "version": sys.version, "modules": modules,
                   "versions": versions, "wheel_source_revision": args.code_revision}
    print(json.dumps(environment, indent=2), flush=True)
    exp.run("installed-import-boundary", lambda: exp.check(
        all(Path(p).is_relative_to(Path(sys.prefix)) for p in modules.values()),
        "runtime packages and compiled kernels all inside selected installation"))
    values = fixtures()
    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_graph([0], [1]), at=1)
    timeline.append_snapshot(RexGraph.from_graph([0, 1], [1, 2]), at=2)
    for name, value in values.items():
        exp.run(f"reference/{name}", lambda n=name, v=value: exp.maths(n, v))
    for name, value in {**values, "temporal": timeline}.items():
        for suffix in FORMATS:
            exp.run(f"file/{name}/{suffix}", lambda n=name, v=value, s=suffix: file_case(exp, root, n, v, s))
    manifests = {}
    for kind in BACKENDS:
        exp.run(f"store/{kind}", lambda k=kind: store_case(exp, root / "stores", k, values, timeline, manifests))
    exp.run("catalog-navigation", lambda: navigation(exp, root, values, manifests))
    (root / "reopen.json").write_text(json.dumps(manifests, indent=2) + "\n")
    for kind in BACKENDS[1:]:
        if kind not in manifests:
            continue
        def fresh(kind=kind):
            result = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()),
                                     "--output", str(root), "--reopen-one", kind],
                                    cwd="/tmp", text=True, capture_output=True, timeout=120)
            exp.check(result.returncode == 0, "new interpreter reopens and verifies persisted store",
                      stdout=result.stdout, stderr=result.stderr)
            child = json.loads((root / f"reopen-{kind}.json").read_text())
            exp.events.extend(child["events"])
            exp.queries.extend(child["queries"])
        exp.run(f"reopen/{kind}", fresh)
    exp.run("cross-store-copy-setup", lambda: copy_cases(exp, root / "copies", values))
    exp.run("native-file-and-record-editing", lambda: native_editing(exp, root, values))
    exp.run("selected-store-stalks", lambda: sheaf_case(exp, root, values))
    counts = dict(Counter(e["status"] for e in exp.events))
    report = {"environment": environment, "seconds": time.perf_counter() - started,
              "counts": counts, "query_count": len(exp.queries),
              "scope": "local functional acceptance; no remote providers, scale, crash or multi-process writer claims",
              "python_assisted": ["fixture/candidate construction and initial RCDB read-fixture put",
                                  "native file save/load/export and file-to-mutation candidate handoff",
                                  "copy_record cross-store transfer", "RCDB catalog loader injection"],
              "events": exp.events, "queries": exp.queries}
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(root / "report.json"), "counts": counts,
                      "queries": len(exp.queries), "seconds": report["seconds"]}), flush=True)
    return int(counts.get("FAIL", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
