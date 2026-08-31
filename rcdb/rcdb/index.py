"""The record index IS the complex.

A store's index is a relation between records and the terms they are accessioned by,
and that is a boundary operator, not a document and not a join table. So it is stored as
one: vertices are the records and one shared vocabulary, and a record's accession is a
single branching relation over them.

    vertices    [0, nR)          the records
                [nR, nR + nT)    the vocabulary, one table for every kind of term
    relations   CSR over those vertices, the record FIRST in its own span

The trichotomy is the layout, and each part is carried by exactly one thing:

    existence     the sparsity pattern: which (record, term) pairs are present at all
    orientation   position 0 of a span. The record is the distinguished vertex, which
                  is what "accessioned BY these terms" means, and it is where the -1
                  goes. Nothing stores a sign.
    share         1/(k-1) over the rest of the span, from its width. Nothing stores a
                  weight, and the column sums to zero at every arity.

So `rel/ptr` and `rel/idx` carry all three, and they are `from_hypergraph`'s own
arguments: a stored index hands the library a complex without converting anything.

That removes the tables rather than compressing them. There is no join table between
records and labels because the incidence IS the join, and no separate table per kind of
term either: a single-valued field like `source` is the k=2 case of the same relation,
where the share is 1/(2-1) = 1 and the column is the ordinary (-1, +1). Kind is a
1-cochain over the relations.

What is per-record and not relational stays a cochain: the counts and measurements are
0-cochains over the record vertices, so a predicate over them is a vectorised read while
a predicate over terms is a boundary operation.
"""
from __future__ import annotations

import struct as _struct
from collections.abc import Sequence

import numpy as np

FORMAT_VERSION = 2

#: 0-cochains over the record vertices. Measurements, not relations: a query filters on
#: these without touching the incidence.
MEASURES = (
    ("version", np.int64), ("created", np.float64),
    ("tx_from", np.float64), ("tx_to", np.float64),
    ("valid_from", np.float64), ("valid_to", np.float64),
    ("nV", np.int64), ("nE", np.int64), ("nF", np.int64),
    ("betti0", np.int64), ("betti1", np.int64), ("betti2", np.int64),
    ("kappa_mean", np.float64), ("kappa_greens_mean", np.float64),
    ("chain_valid", np.int8), ("n_voids", np.int64), ("n_labels", np.int64),
    ("structural_perplexity", np.float64), ("effective_modes", np.float64),
    ("varentropy_gap", np.float64),
)
#: the kinds of accession relation, in code order. A single-valued kind is the k=2 case
#: of the same relation, not a separate mechanism.
KINDS = ("source", "object_type", "coherence_method",
         "tags", "labels_sample", "vertex_labels")
#: kinds holding one term rather than a list. Reconstruction only.
SINGLE = frozenset(("source", "object_type", "coherence_method"))
#: kinds read from `meta` rather than from the signature.
FROM_META = frozenset(("vertex_labels",))

_KIND_CODE = {k: i for i, k in enumerate(KINDS)}


def _fit(values, bound):
    """`values` in the narrowest unsigned type that holds `bound`.

    A code is an index into a table, so its width follows the table rather than the
    machine.
    """
    for dt in (np.uint8, np.uint16, np.uint32):
        if bound <= np.iinfo(dt).max:
            return np.asarray(values, dtype=dt)
    return np.asarray(values, dtype=np.int64)


def _pack_strings(table):
    """A string table as one utf-8 blob plus its offsets, so a read is a slice."""
    blob = bytearray()
    offs = [0] * (len(table) + 1)
    for i, s in enumerate(table):
        blob += s.encode("utf-8")
        offs[i + 1] = len(blob)
    return bytes(blob), _fit(offs, max(len(blob), 1))


def _unpack_strings(blob, offs):
    """The table as a list of str.

    `offs` is converted once, since indexing a numpy array per bound returns numpy
    scalars. Where the blob is ASCII the byte offsets are character offsets too, so one
    decode and a slice per entry replaces a decode per entry.
    """
    o = offs.tolist() if hasattr(offs, "tolist") else list(offs)
    if not o or o[-1] == 0:
        return [""] * max(len(o) - 1, 0)
    if blob.isascii():
        text = blob.decode("ascii")
        return [text[o[i]:o[i + 1]] for i in range(len(o) - 1)]
    return [blob[o[i]:o[i + 1]].decode("utf-8", "replace") for i in range(len(o) - 1)]


class StringTable(Sequence):
    """A packed string table, decoded per entry rather than all at once.

    Indexed, iterated and measured like a list. Entries decode as they are asked for
    and are cached, so a caller that reads a handful of a large vocabulary decodes a
    handful. Repeated access costs one dict lookup.
    """

    __slots__ = ("_blob", "_offs", "_cache")

    def __init__(self, blob, offs):
        self._blob = blob
        self._offs = offs
        self._cache: dict[int, str] = {}

    def __len__(self):
        return max(len(self._offs) - 1, 0)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        i = int(i)
        if i < 0:
            i += len(self)
        got = self._cache.get(i)
        if got is None:
            got = self._blob[int(self._offs[i]):int(self._offs[i + 1])].decode(
                "utf-8", "replace")
            self._cache[i] = got
        return got


def _row_values(record) -> dict:
    """The MEASURES of one record, from its signature and its own fields.

    One reader for both writers. The cochains and the log frame carry the same values,
    so they have to agree on what each one means, and a field promoted to a cochain is
    added here once.

    None stays None: an absent bitemporal bound and an uncomputed measurement are not
    zero, and the readers turn it back into None.
    """
    sig = record.signature or {}
    betti = sig.get("betti") or []
    return {
        "version": record.version, "created": record.created,
        "tx_from": record.tx_from, "tx_to": record.tx_to,
        "valid_from": record.valid_from, "valid_to": record.valid_to,
        "nV": sig.get("nV", 0), "nE": sig.get("nE", 0), "nF": sig.get("nF", 0),
        "betti0": betti[0] if len(betti) > 0 else 0,
        "betti1": betti[1] if len(betti) > 1 else 0,
        "betti2": betti[2] if len(betti) > 2 else 0,
        "kappa_mean": sig.get("kappa_mean"),
        "kappa_greens_mean": sig.get("kappa_greens_mean"),
        "chain_valid": bool(sig.get("chain_valid")),
        "n_voids": sig.get("n_voids", 0), "n_labels": sig.get("n_labels", 0),
        "structural_perplexity": sig.get("structural_perplexity"),
        "effective_modes": sig.get("effective_modes"),
        "varentropy_gap": sig.get("varentropy_gap"),
    }


def _terms_of(record) -> list:
    """[(kind_code, [term, ...])] for one record, empty kinds dropped.

    A kind with nothing in it produces NO relation. Absence is the sparsity pattern,
    which is the existence channel doing its own job rather than a stored empty.
    """
    sig = record.signature or {}
    meta = record.meta or {}
    out = []
    for kind in KINDS:
        src = meta if kind in FROM_META else sig
        v = src.get(kind)
        if kind in SINGLE:
            terms = [str(v)] if v not in (None, "") else []
        else:
            terms = [str(x) for x in (v or [])]
        if terms:
            out.append((_KIND_CODE[kind], terms))
    return out


def build(records) -> dict:
    """The complex and the cochains for `records`, an iterable of (id, ComplexRecord).

    Rows keep the order given and relations are emitted in row order, so a record owns a
    contiguous run of relations and the reader recovers it without an index of its own.
    """
    rows = list(records)
    n = len(rows)
    ids = [r[0] for r in rows]
    recs = [r[1] for r in rows]

    # ONE vocabulary for every kind of term. A value that is both a source and a tag is
    # one vertex, which is the point: the co-occurrence is structure, not a coincidence
    # between two tables.
    vocab, vocab_idx = [], {}

    def _term(s):
        j = vocab_idx.get(s)
        if j is None:
            j = len(vocab); vocab_idx[s] = j; vocab.append(s)
        return j

    rel_idx, rel_ptr, rel_kind = [], [0], []
    for i, r in enumerate(recs):
        for kcode, terms in _terms_of(r):
            rel_idx.append(i)                        # distinguished: the record
            rel_idx.extend(n + _term(t) for t in terms)
            rel_ptr.append(len(rel_idx))
            rel_kind.append(kcode)

    rowvals = [_row_values(r) for r in recs]
    measures = {}
    for name, dt in MEASURES:
        if dt == np.float64:
            measures[name] = np.fromiter(
                (np.nan if v[name] is None else v[name] for v in rowvals), dt, n)
        else:
            measures[name] = np.fromiter(
                (0 if v[name] is None else int(v[name]) for v in rowvals), dt, n)

    nV = n + len(vocab)
    return {"n": n, "n_terms": len(vocab), "nV": nV,
            "ids": ids, "vocab": vocab,
            "rel_ptr": np.asarray(rel_ptr, np.int64),
            "rel_idx": _fit(rel_idx, max(nV, 1)),
            "rel_kind": _fit(rel_kind, len(KINDS)),
            "measures": measures,
            "residual": _residual_columns(recs)}


#### the residual: what the schema does not name, on the same principle
#
# A record can carry keys no cochain and no kind covers: the temporal branch adds T and
# checkpoint_times, and meta is caller supplied and arbitrary. Written per row as a
# document, every key name is stored again for every record that has it.
#
# So a key PATH is a term like any other, and the only question is whether it deserves a
# column. That is not a threshold, it is the bridge/cycle split on the record-path
# incidence: a path belonging to exactly one record is a BRIDGE and shares nothing, so a
# column for it is a column holding one row. Paths on cycles are shared by construction
# and a column is exactly right.
#
# Shared paths get a column each; bridge paths pool into ONE sparse relation of
# (row, path, value). Without that split, one record whose meta held a list of 5000
# small dicts produced 5000 single-row columns and an index 15.7x LARGER than json.
#
# A path is a sequence of (segment, is_index) pairs, kept as segment codes rather than a
# joined string, so a key containing any character at all is unambiguous.

K_INT, K_FLOAT, K_BOOL, K_STR, K_NONE = 0, 1, 2, 3, 4
K_EMPTY_LIST, K_EMPTY_DICT = 5, 6
K_LIST_INT, K_LIST_FLOAT, K_LIST_BOOL, K_LIST_STR = 7, 8, 9, 10
_LIST_KINDS = (K_LIST_INT, K_LIST_FLOAT, K_LIST_BOOL, K_LIST_STR)

#: caller data is arbitrary, so the walk is bounded. A cycle in `meta` would otherwise
#: recurse until the interpreter dies and take the store's write path with it.
_MAX_DEPTH = 64


def _scalar_kind(v):
    if v is None:
        return K_NONE
    if isinstance(v, (bool, np.bool_)):
        return K_BOOL
    if isinstance(v, (int, np.integer)):
        return K_INT
    if isinstance(v, (float, np.floating)):
        return K_FLOAT
    return K_STR                                # str, and anything else as its str


def _list_kind(vals):
    """The packed kind for a homogeneous scalar list, or None if it is not one."""
    def _isnum(v):
        return isinstance(v, (int, float, np.integer, np.floating)) and \
            not isinstance(v, (bool, np.bool_))
    if all(isinstance(v, (bool, np.bool_)) for v in vals):
        return K_LIST_BOOL
    if all(_isnum(v) and isinstance(v, (int, np.integer)) for v in vals):
        return K_LIST_INT
    if all(_isnum(v) for v in vals):
        return K_LIST_FLOAT
    if all(isinstance(v, str) for v in vals):
        return K_LIST_STR
    return None


def _flatten(value, path, out, depth=0, seen=None):
    """Append (path, kind, value) leaves. Dicts and ragged lists recurse by path.

    A container already on the current walk is written as its repr rather than followed,
    so a self-referential meta is recorded instead of fatal.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        if (seen and id(value) in seen) or depth >= _MAX_DEPTH:
            out.append((path, K_STR, repr(value)[:200]))
            return
        seen = (seen or frozenset()) | {id(value)}
    if isinstance(value, dict):
        if not value:
            out.append((path, K_EMPTY_DICT, None))
            return
        for k, v in value.items():
            _flatten(v, path + ((str(k), False),), out, depth + 1, seen)
        return
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if not vals:
            out.append((path, K_EMPTY_LIST, None))
            return
        kind = _list_kind(vals)
        if kind is not None:
            out.append((path, kind, vals))
            return
        for i, v in enumerate(vals):
            _flatten(v, path + ((str(i), True),), out, depth + 1, seen)
        return
    out.append((path, _scalar_kind(value), value))


def _residual_columns(recs):
    """{(scope, path, kind): {row: value}} for every key outside the schema.

    Scope 0 is the signature and 1 is the meta. A path that holds different types in
    different records becomes one column per type, which keeps every row exact rather
    than promoting them all to a common one.
    """
    known_sig = {c[0] for c in MEASURES} | set(KINDS) | {"betti"}
    cells: dict = {}
    for i, r in enumerate(recs):
        for scope, d, known in ((0, r.signature or {}, known_sig),
                                (1, r.meta or {}, set(KINDS))):
            for k, v in (d or {}).items():
                if k in known:
                    continue
                leaves = []
                _flatten(v, ((str(k), False),), leaves)
                for path, kind, val in leaves:
                    cells.setdefault((scope, path, kind), {})[i] = val
    return cells


def _leaf_strings(kind, value):
    """A leaf's value as strings. The kind reconstructs the type, and repr round trips a
    float exactly, so passing through text costs no precision."""
    if kind in (K_NONE, K_EMPTY_LIST, K_EMPTY_DICT):
        return []
    if kind in _LIST_KINDS:
        return [repr(v) if kind == K_LIST_FLOAT else str(v) for v in value]
    return [repr(value) if kind == K_FLOAT else str(value)]


def _leaf_value(kind, parts):
    if kind == K_NONE:
        return None
    if kind == K_EMPTY_LIST:
        return []
    if kind == K_EMPTY_DICT:
        return {}
    if kind == K_STR:
        return parts[0]
    if kind == K_BOOL:
        return parts[0] == "True"
    if kind == K_INT:
        return int(parts[0])
    if kind == K_FLOAT:
        return float(parts[0])
    if kind == K_LIST_STR:
        return list(parts)
    if kind == K_LIST_BOOL:
        return [x == "True" for x in parts]
    if kind == K_LIST_INT:
        return [int(x) for x in parts]
    return [float(x) for x in parts]


def _assign(root, path, value):
    """Place `value` at `path`, growing the dicts and lists the path implies."""
    cur = root
    for k, (seg, is_idx) in enumerate(path):
        key = int(seg) if is_idx else seg
        last = k == len(path) - 1
        child = value if last else ([] if path[k + 1][1] else {})
        if is_idx:
            while len(cur) <= key:
                cur.append(None)
            if last or cur[key] is None:
                cur[key] = child
            cur = cur[key]
        else:
            if last or cur.get(key) is None:
                cur[key] = child
            cur = cur[key]
    return root


def _path_tensors(klist, t, prefix, code):
    """The (scope, path, kind) triples of a set of residual keys, as tensors."""
    pflat, pisidx = [], []
    spans = np.zeros((len(klist), 2), np.int64)
    kinds = np.zeros(len(klist), np.int8)
    scopes = np.zeros(len(klist), np.int8)
    for j, (scope, path, kind) in enumerate(klist):
        spans[j, 0] = len(pflat)
        for sname, is_idx in path:
            pflat.append(code(sname)); pisidx.append(1 if is_idx else 0)
        spans[j, 1] = len(pflat)
        kinds[j], scopes[j] = kind, scope
    t[f"{prefix}/path/flat"] = _fit(pflat, max(pflat, default=0) + 1)
    t[f"{prefix}/path/isidx"] = np.asarray(pisidx, np.int8)
    t[f"{prefix}/path/spans"] = _fit(spans.ravel(),
                                     max(len(pflat), 1)).reshape(spans.shape)
    t[f"{prefix}/kinds"] = kinds
    t[f"{prefix}/scope"] = scopes


def _residual_tensors(cells, t):
    """Shared paths get a column each; bridge paths pool into one sparse relation."""
    segs, seg_index = [], {}

    def _code(sname):
        j = seg_index.get(sname)
        if j is None:
            j = len(segs); seg_index[sname] = j; segs.append(sname)
        return j

    keys = sorted(cells, key=lambda k: (k[0], [s for s, _ in k[1]], k[2]))
    shared = [k for k in keys if len(cells[k]) > 1]
    bridge = [k for k in keys if len(cells[k]) == 1]

    for j, key in enumerate(shared):
        rowmap = cells[key]
        rows = sorted(rowmap)
        vals = [rowmap[r] for r in rows]
        kind = key[2]
        g = f"residual/g{j}/"
        t[g + "rows"] = _fit(rows, max(rows) + 1)
        if kind in (K_INT, K_BOOL):
            t[g + "vals"] = np.asarray([int(v) for v in vals], np.int64)
        elif kind == K_FLOAT:
            t[g + "vals"] = np.asarray([float(v) for v in vals], np.float64)
        elif kind == K_STR:
            blob, offs = _pack_strings([str(v) for v in vals])
            t[g + "sbytes"] = np.frombuffer(blob, np.uint8).copy()
            t[g + "soffs"] = offs
        elif kind in _LIST_KINDS:
            spans = np.zeros((len(vals), 2), np.int64)
            flat: list = []
            for i, seq in enumerate(vals):
                spans[i, 0] = len(flat); flat.extend(seq); spans[i, 1] = len(flat)
            t[g + "spans"] = _fit(spans.ravel(), max(len(flat), 1)).reshape(spans.shape)
            if kind == K_LIST_FLOAT:
                t[g + "flat"] = np.asarray(flat, np.float64)
            elif kind == K_LIST_STR:
                blob, offs = _pack_strings([str(x) for x in flat])
                t[g + "sbytes"] = np.frombuffer(blob, np.uint8).copy()
                t[g + "soffs"] = offs
            else:
                t[g + "flat"] = np.asarray([int(v) for v in flat], np.int64)

    brows, bvals, bspans = [], [], []
    for key in bridge:
        (row, val), = cells[key].items()
        parts = _leaf_strings(key[2], val)
        brows.append(row)
        bspans.append(len(bvals))
        bvals.extend(parts)
        bspans.append(len(bvals))
    blob, offs = _pack_strings(bvals)
    t["bridge/rows"] = _fit(brows, max(brows, default=0) + 1)
    t["bridge/spans"] = _fit(bspans, max(len(bvals), 1)).reshape(-1, 2)
    t["bridge/sbytes"] = np.frombuffer(blob, np.uint8).copy()
    t["bridge/soffs"] = offs
    _path_tensors(shared, t, "residual", _code)
    _path_tensors(bridge, t, "bridge", _code)
    blob, offs = _pack_strings(segs)
    t["residual/seg/table"] = np.frombuffer(blob, np.uint8).copy()
    t["residual/seg/offsets"] = offs


def _paths_from(t, prefix, segs):
    spans = t[f"{prefix}/path/spans"]
    pflat, pisidx = t[f"{prefix}/path/flat"], t[f"{prefix}/path/isidx"]
    kinds, scopes = t[f"{prefix}/kinds"], t[f"{prefix}/scope"]
    out = []
    for j in range(len(kinds)):
        lo, hi = int(spans[j, 0]), int(spans[j, 1])
        out.append({"scope": int(scopes[j]), "kind": int(kinds[j]),
                    "path": tuple((segs[int(pflat[k])], bool(pisidx[k]))
                                  for k in range(lo, hi))})
    return out


def _residual_groups(t):
    """The reader's view: shared columns with their tensors, and the pooled bridges."""
    if "residual/kinds" not in t:
        return [], []
    segs = _unpack_strings(t["residual/seg/table"].tobytes(), t["residual/seg/offsets"])
    shared = _paths_from(t, "residual", segs)
    for j, e in enumerate(shared):
        g = f"residual/g{j}/"
        e["rows"] = t[g + "rows"]
        for suffix in ("vals", "flat", "spans"):
            if g + suffix in t:
                e[suffix] = t[g + suffix]
        if g + "sbytes" in t:
            e["strings"] = _unpack_strings(t[g + "sbytes"].tobytes(), t[g + "soffs"])
    bridge = _paths_from(t, "bridge", segs)
    bstr = _unpack_strings(t["bridge/sbytes"].tobytes(), t["bridge/soffs"])
    for j, e in enumerate(bridge):
        e["row"] = int(t["bridge/rows"][j])
        lo, hi = int(t["bridge/spans"][j, 0]), int(t["bridge/spans"][j, 1])
        e["parts"] = bstr[lo:hi]
    return shared, bridge


def _group_value(e, i):
    """The value a shared column holds at its i-th row."""
    kind = e["kind"]
    if kind == K_NONE:
        return None
    if kind == K_EMPTY_LIST:
        return []
    if kind == K_EMPTY_DICT:
        return {}
    if kind == K_STR:
        return e["strings"][i]
    if kind == K_BOOL:
        return bool(e["vals"][i])
    if kind == K_INT:
        return int(e["vals"][i])
    if kind == K_FLOAT:
        return float(e["vals"][i])
    lo, hi = int(e["spans"][i, 0]), int(e["spans"][i, 1])
    if kind == K_LIST_STR:
        return [e["strings"][k] for k in range(lo, hi)]
    run = e["flat"][lo:hi]
    if kind == K_LIST_BOOL:
        return [bool(v) for v in run]
    if kind == K_LIST_INT:
        return [int(v) for v in run]
    return [float(v) for v in run]


def _tensors(index: dict) -> dict:
    """The index as named tensors: the incidence, the cochains, and the tables."""
    t = {}
    t["rel/ptr"] = np.asarray(index["rel_ptr"], np.int64)
    t["rel/idx"] = np.asarray(index["rel_idx"])
    t["rel/kind"] = np.asarray(index["rel_kind"])
    for name, dt in MEASURES:
        t[f"measure/{name}"] = np.ascontiguousarray(index["measures"][name], dtype=dt)
    for key, table in (("ids", index["ids"]), ("vocab", index["vocab"])):
        blob, offs = _pack_strings(table)
        t[f"{key}/table"] = np.frombuffer(blob, np.uint8).copy()
        t[f"{key}/offsets"] = offs
    _residual_tensors(index.get("residual") or {}, t)
    return t


def write(path, index: dict, *, extra: dict | None = None) -> None:
    """Store the index as safetensors, with a digest over the payload.

    The digest is the one every other tensor artifact here carries, so a truncated or
    edited index is caught on read instead of returning as data.

    `extra` carries a backend's own aligned arrays, one entry per row, under `extra/`.
    The digest covers them with everything else, which is why they belong here rather
    than in a second file or in the metadata.
    """
    from rexgraph.io.rex_state import DIGEST_ALGO, state_digest
    from safetensors.numpy import save_file
    t = _tensors(index)
    for name, arr in (extra or {}).items():
        t[f"extra/{name}"] = np.ascontiguousarray(arr)
    save_file(t, str(path), metadata={
        "format": str(FORMAT_VERSION), "n": str(index["n"]),
        "n_terms": str(index["n_terms"]),
        # see wiktionary.write_index: the digest framing has to travel with the file or
        # anything written under the older one reads as corrupt
        "digest_algo": str(DIGEST_ALGO), "digest": state_digest(t)})


def read(path, *, verify: bool = True) -> dict:
    """The inverse of `write`. A digest that does not match raises."""
    from rexgraph.io.rex_state import state_digest
    from safetensors import safe_open
    from safetensors.numpy import load_file
    with safe_open(str(path), "numpy") as f:
        meta = f.metadata() or {}
    if int(meta.get("format", 0)) != FORMAT_VERSION:
        raise ValueError(f"index format {meta.get('format')} != {FORMAT_VERSION}")
    t = load_file(str(path))
    algo = int(meta.get("digest_algo", 1))
    if verify and meta.get("digest") and state_digest(t, algo=algo) != meta["digest"]:
        raise ValueError("index digest mismatch: the file is not what was written")
    n, nt = int(meta["n"]), int(meta["n_terms"])

    def _table(prefix):
        return StringTable(t[f"{prefix}/table"].tobytes(), t[f"{prefix}/offsets"])
    shared, bridge = _residual_groups(t)
    ptr, idx = t["rel/ptr"], t["rel/idx"]
    return {"n": n, "n_terms": nt, "nV": n + nt,
            "ids": _table("ids"), "vocab": _table("vocab"),
            "rel_ptr": ptr, "rel_idx": idx, "rel_kind": t["rel/kind"],
            "measures": {name: t[f"measure/{name}"] for name, _dt in MEASURES},
            "residual_groups": shared, "residual_bridges": bridge,
            "extra": {k[len("extra/"):]: v for k, v in t.items()
                      if k.startswith("extra/")}}


def complex_of(index: dict):
    """The index AS a RexGraph, from the CSR it already stores.

    No conversion: `rel_ptr` and `rel_idx` are `from_hypergraph`'s own arguments, so
    every reading in the library is one call from a stored index. Records are vertices
    [0, n) and vocabulary terms are [n, n + n_terms).
    """
    from rexgraph.graph import RexGraph
    return RexGraph.from_hypergraph(np.asarray(index["rel_ptr"], np.int64),
                                    np.asarray(index["rel_idx"], np.int64))


def _term_codes(index: dict) -> dict:
    """{term: vertex} built once per index and cached on it."""
    codes = index.get("_term_codes")
    if codes is None:
        n = index["n"]
        codes = {t: n + i for i, t in enumerate(index["vocab"])}
        index["_term_codes"] = codes
    return codes


def boundary_operator(index: dict):
    """`B1` for the stored index as a scipy CSC matrix, built from the arrays it holds.

    The index already stores `rel_ptr` and `rel_idx`, and those ARE the column structure:
    relation `e` occupies `rel_idx[rel_ptr[e]:rel_ptr[e+1]]`, the record first. So the
    boundary operator is those two arrays plus the values the arity determines:

        data[rel_ptr[e]]        = -1            the record, which the relation is OF
        data[rel_ptr[e]+1:...]  = 1/(k-1)       its terms, sharing

   , which is a vectorised fill, not a build. Going through `complex_of` and
    `to_scipy_csr(rex._B1_dual)` reconstructs a whole RexGraph to arrive at the same
    matrix and measured 40 s on the 61,353-record store, paid by the first query of every
    process. This is about a second, and nothing is cached that a restart has to earn
    back.

    Cached on the index anyway, because a query does not need it rebuilt.
    """
    import scipy.sparse as sp

    hit = index.get("_B1_csc")
    if hit is not None:
        return hit
    ptr = np.asarray(index["rel_ptr"], dtype=np.int64)
    idx = np.asarray(index["rel_idx"], dtype=np.int64)
    width = np.diff(ptr)
    data = np.empty(idx.size, dtype=np.float64)
    # every entry is a share, then position 0 of each column is overwritten with the -1
    data[:] = np.repeat(1.0 / np.maximum(width - 1, 1), width)
    heads = ptr[:-1][width > 0]
    data[heads] = -1.0
    B = sp.csc_matrix((data, idx, ptr), shape=(int(index["nV"]), int(width.size)))
    index["_B1_csc"] = B
    return B


def _existence_operator(index: dict, B):
    """`B1`'s sparsity pattern with every entry 1, cached beside it.

    The integer/existence tower: which record names which term, before the `1/(k-1)`
    share says how much of the record that term is. Same indices and indptr as `B`, so
    this is one array of ones and no rebuild.
    """
    hit = index.get("_B1_existence")
    if hit is not None:
        return hit
    import scipy.sparse as sp
    E = sp.csc_matrix((np.ones(B.nnz, dtype=np.float64), B.indices, B.indptr),
                      shape=B.shape)
    index["_B1_existence"] = E
    return E


def _row_operator(index: dict, P, reading: str):
    """`P` with row access, cached beside it.

    A seed set names a handful of TERM vertices and the relations they reach are those
    vertices' rows, which `P` being CSC does not address. Built once per index: 3.3 s at
    391M entries, against 840 ms per query for walking the whole operator instead.
    """
    key = "_B1_csr" if reading == "share" else "_B1_existence_csr"
    hit = index.get(key)
    if hit is not None:
        return hit
    index[key] = P.tocsr()
    return index[key]


def _seed_flux(index: dict, P, reading: str, seeds, weights):
    """`g = P^T x` for a seed vector, computed over the seeds' rows alone.

    `x` is nonzero only at the seeds, so `g` is nonzero only on the relations naming
    one and every other column of the product is an exact zero. This reads the seed rows
    and scatters them: the same arithmetic, in the same order of accumulation per
    column.
    """
    R = _row_operator(index, P, reading)
    rows = R[np.asarray(seeds, dtype=np.int64)]
    g = np.zeros(P.shape[1], dtype=np.float64)
    if rows.nnz:
        per = np.diff(rows.indptr)
        np.add.at(g, rows.indices, rows.data * np.repeat(weights, per))
    return g


def _vertex_degree_exact(index: dict, B):
    """Incidences per vertex as INTEGERS, cached. `_vertex_degree` returns the float
    array the matvec wants; a rational needs the integer it was rounded from."""
    hit = index.get("_deg_int")
    if hit is not None:
        return hit
    index["_deg_int"] = np.bincount(B.indices, minlength=B.shape[0])
    return index["_deg_int"]


def _arity(index: dict, B):
    """Cells per relation, cached. Counting them is one pass over `indptr`, but doing it
    per call on a 391M-entry operator is not free."""
    hit = index.get("_arity")
    if hit is not None:
        return hit
    index["_arity"] = np.diff(B.indptr)
    return index["_arity"]


def _vertex_degree(index: dict, B):
    """Incidences per VERTEX, cached beside the operator that shares its arrays.

    `B.indices` IS `rel_idx` already widened by the operator build, so counting off it
    costs nothing extra; re-reading `index["rel_idx"]` and casting to int64 allocated a
    fresh 3.1 GB copy of 391M entries on EVERY call, which is most of what a warm
    `record_response` was paying.

    Cached for the same reason `boundary_operator` is: a query does not need it rebuilt.
    """
    hit = index.get("_deg")
    if hit is not None:
        return hit
    deg = np.bincount(B.indices, minlength=B.shape[0]).astype(np.float64)
    index["_deg"] = deg
    return deg


def channel_diagonals(index: dict):
    """The four channel diagonals for the accession complex, per relation.

    `rexgraph.rational_trig.exact_channel_diagonals` computes these from an assembled
    complex; at corpus grade that is not available, because F and C are defined by
    OFF-DIAGONAL sums over relation pairs and every pair of records sharing a common word
    co-participates: the same density that makes RL4 unusable here. So each is taken
    from the structure directly:

        T[e,e] = G[e,e] = 1 + 1/(k-1)     from the arity, exactly, no matvec
        F[e,e] = 0                        STRUCTURALLY, see below
        C[e,e] = (|B|^T (|B| 1))_e - G[e,e]   two matvecs, O(nnz)

    F is identically zero here and that is a property of the accession relation rather
    than of this data. F measures where the signed and unsigned readings disagree, which
    needs a vertex that heads one relation and is an argument of another. Records occupy
    `[0, n)` and terms `[n, n + n_terms)`, disjoint, and a record heads every relation it
    is in while a term is always an argument, so no vertex is ever both, and the
    mismatch cannot arise. Verified on a built index: max |T - G| off-diagonal is 0.0e+00.

    What is left is TOPOLOGY against CO-PARTICIPATION, which is exactly the axis that
    separates a document answering a query from one that merely shares vocabulary with it.

    Returns `(diagonals, names)` with `diagonals` an `(nE, 4)` array.
    """
    B = boundary_operator(index)
    ptr = np.asarray(index["rel_ptr"], dtype=np.int64)
    idx = np.asarray(index["rel_idx"], dtype=np.int64)
    k = np.diff(ptr).astype(np.float64)
    t = 1.0 + 1.0 / np.maximum(k - 1.0, 1.0)          # = sum_v c_e[v]^2 at arity k

    # (G 1)_e = sum_f G[e,f] = |B|^T (|B| 1_nE). Both passes run on the DATA ARRAYS: the
    # sparsity is `ptr`/`idx` and the magnitudes are |data|, so nothing needs |B| built.
    # `abs(B)` copies a 391-million-nonzero matrix at Gutenberg scale, which is most of
    # a 36 s first call.
    mag = np.abs(B.data)
    rowsum = np.zeros(B.shape[0], dtype=np.float64)
    np.add.at(rowsum, idx, mag)                        # (|B| 1_nE)_v
    c = np.zeros(B.shape[1], dtype=np.float64)
    np.add.at(c, np.repeat(np.arange(k.size), k.astype(np.int64)), mag * rowsum[idx])
    c -= t                                             # drop the diagonal term
    d = np.zeros((B.shape[1], 4), dtype=np.float64)
    d[:, 0] = t
    d[:, 1] = t
    d[:, 2] = 0.0
    d[:, 3] = np.maximum(c, 0.0)
    return d, ["topology", "geometry", "frustration", "coparticipation"]


def _chi_of(index: dict):
    """`chi[e]`, the character, cached on the index.

    Same construction as `rational_trig.exact_character`: normalise each channel by its
    own trace, then normalise the row. A channel whose trace is zero contributes zero
    rather than dividing by it: F is exactly that case here.
    """
    hit = index.get("_chi")
    if hit is not None:
        return hit
    d, names = channel_diagonals(index)
    traces = d.sum(axis=0)
    hats = np.zeros_like(d)
    live = traces > 0
    hats[:, live] = d[:, live] / traces[live]
    rl = hats.sum(axis=1, keepdims=True)
    chi = np.divide(hats, rl, out=np.full_like(hats, 1.0 / hats.shape[1]),
                    where=rl > 0)
    index["_chi"] = (chi, names)
    return chi, names


def record_response(index: dict, terms, *, steps: int = 1, rex=None,
                    channels: bool = False, reading: str = "share"):
    """How strongly each RECORD answers a set of terms, by diffusion on the index.

    This is the accession relation read in the direction it was built for. A record's
    column says "this record is accessioned BY these terms": the record carries the -1
    at position 0 and its terms share `1/(k-1)`. Seeding the TERM vertices and letting the
    field run is the inverse: given terms, which records do they reach.

    It is not an inverted index and the difference is the point. An inverted index answers
    "which records contain this term", one term at a time, and something outside the
    structure then combines the lists. The field answers where the query's energy GOES,
    which at `steps > 1` routes through co-participation: a record is reached through
    terms it shares with records the query already reached, not only through terms the
    query named. That is the local-to-global bridge, and it is a matvec rather than a scan.

    **Matrix-free, and it has to be.** The propagation is `L0 x = B1 (B1^T x)`, applied,
    never formed. Neither operator this complex would otherwise build is affordable:
    `RL4` is nE x nE and every pair of records sharing any common word co-participates, so
    at 306,765 relations it is effectively dense; `L0` is nV x nV at 11.6 million. Two
    sparse matvecs over nnz(B1) is the whole cost. `rex.propagate_signal` goes through
    RL4 and is the wrong propagator HERE, not in general.

    Seeds are weighted `1/deg`: inverse document frequency derived from the complex's own
    incidence rather than imported as a corpus statistic. A term used by 9,000 records
    says little about which one answers, and its degree already says so. Nothing is
    excluded and no threshold is applied.

    `steps` is the SCALE, not a threshold, and one step is a true reading: the seeded mass
    lands on exactly the relations that name the query's terms, so `steps=1` is accession
    proper. Each further step is one more moment of the same propagator, reaching records
    through shared vocabulary.

    MEASURED on the 61,353-document Gutenberg store, and the default follows it: one step
    ranks the right book at 1, 1, 2, 3 and 7 on five title queries, and two steps is
    WORSE on every one of them: Alice falls from 115 to 8474, the second Frankenstein
    from 179 to 37201. At this grade the vocabulary is shared so widely that a second
    moment smears rather than bridges. That is a property of the accession complex, not
    of the method: one grade down, inside a document, the further scales are what reach.

    `channels=True` returns the response RESOLVED INTO THE CHANNELS instead of summed:
    `(n_records, 4)` over (topology, geometry, frustration, coparticipation), plus their
    names. A record then answers with a PROFILE rather than a scalar. The scalar is that
    profile summed over its channels, so asking for it adds nothing to the computation:
    it stops throwing the axes away, and the axes are where the difference between
    answering and merely sharing vocabulary lives. See `channel_diagonals` for what each
    channel is at this grade and why F is structurally zero.

    `reading` chooses WHICH TOWER is read, and the two answer different questions.

        "share"      the boundary as built, so each term contributes `1/(k-1)`. A
                     record's response is therefore a DENSITY: what fraction of this
                     record the query is. Measured on the 8 documents that all hold
                     `221b baker street`, the resulting order agrees with the ordering
                     by accession width at rank correlation +1.000: a 3,206-term
                     pamphlet mentioning a page number beat the 8,332-term Adventures
                     of Sherlock Holmes, which sat at 37.
        "existence"  the {0,1} incidence pattern, so a term contributes its seed weight
                     and nothing is divided by the width. A record's response is then the
                     MASS of query terms it holds. On the same query that puts Holmes at
                     5 and the pamphlet at 22.

    Both are exact and neither is a normalisation of the other: they are the integer and
    the share towers of the same boundary, and accession asks "does this record hold the
    query's terms", which is the mass question. The share stays in `B1` either way: it
    is load-bearing for the zero-sum column and nothing here touches it.

    Returns `(scores, ids)`, or `(profiles, ids, channel_names)` with `channels`.
    """

    codes = _term_codes(index)
    n = int(index["n"])
    # the table itself, not a list: materialising it decodes every id on a
    # call that returns a ranking over a handful of them.
    ids = index["ids"]
    seeds = [codes[w] for w in {str(x).lower() for x in terms} if w in codes]
    if not seeds:
        return np.zeros(n, dtype=np.float64), ids
    B = boundary_operator(index)
    if str(reading) not in ("share", "existence"):
        raise ValueError(f"reading must be 'share' or 'existence', got {reading!r}")
    # the existence tower is the same sparsity pattern with every entry 1: the {0,1}
    # incidence, which is what "holds this term" means before any share is applied.
    P = B if reading == "share" else _existence_operator(index, B)
    sd = np.asarray(seeds, dtype=np.int64)
    # incidences per VERTEX. `B` is CSC, so `np.diff(B.indptr)` counts per COLUMN, per
    # relation: a different array of a different length, which mis-weights every seed
    # and raises once a vertex id exceeds nE.
    deg = _vertex_degree(index, B)
    w = 1.0 / np.maximum(deg[sd], 1.0)
    n_steps = max(1, int(steps))
    if n_steps == 1 and not channels:
        # the reading is a rational and is computed as one. The float returned is that
        # rational divided once, not an accumulation of roundings, and the ordering it
        # carries is the ordering of the exact values.
        got = _response_terms(index, P, str(reading), sd)
        if got is None:
            return np.zeros(n, dtype=np.float64), ids
        rows, carried, which, deg, den = got
        return _render(rows, carried, which, deg, den, n), ids
    if n_steps == 1:
        # One step is accession proper and is the default, and at one step the answer is
        # confined to the seeds' own star. Both matvecs below compute it over the whole
        # operator, which on the Gutenberg index is 391M entries to reach a result that
        # is nonzero on a few thousand columns.
        g = _seed_flux(index, P, reading, sd, w)
    else:
        x = np.zeros(B.shape[0], dtype=np.float64)
        x[sd] = w
        for _ in range(n_steps):
            g = P.T @ x
            x = np.abs(P @ g)
    if not channels:
        if n_steps == 1:
            # a record vertex sits at position 0 of the relations it owns and in no
            # other column, so its row of `P g` is the sum of `g` over those relations.
            # The rest of `P g` is the term half, which this return discards.
            owner0 = rel_owner(index)
            acc = np.zeros(n, dtype=np.float64)
            keep0 = (owner0 >= 0) & (owner0 < n)
            np.add.at(acc, owner0[keep0], g[keep0])
            return np.abs(acc), ids
        return x[:n], ids

    # A record is the HEAD of every relation it owns, so relation e hands it |g_e|. The
    # profile is that contribution carried through the relation's character instead of
    # collapsed into a total. (`abs` is safe HERE and not in a document: at this grade a
    # vertex is a record or a term and never both, so the sign is constant per class and
    # carries nothing. Inside a document a word heads one span and argues in another, and
    # there the same `abs` would cancel real orientation.)
    chi, names = _chi_of(index)
    owner = rel_owner(index)
    contrib = np.abs(g)
    if n_steps == 1:
        # the reading the scalar takes, one index down. `g` is exact here, so the
        # profile summed over its channels is the scalar `record_response` returns.
        contrib = _relation_flux(index, P, str(reading), sd)
    prof = np.zeros((n, chi.shape[1]), dtype=np.float64)
    keep = (owner >= 0) & (owner < n)
    for c in range(chi.shape[1]):
        np.add.at(prof[:, c], owner[keep], contrib[keep] * chi[keep, c])
    return prof, ids, names


def _record_denominator(index: dict, B):
    """`den[r]`, the product of `(k-1)` over the relations record `r` owns, cached.

    Every denominator the share reading introduces sits inside this one number, and a
    record owning five relations keeps it small: 2**23.3 at the widest over a
    61,353-document store, so the reading's integer form holds in an int64.
    """
    hit = index.get("_den")
    if hit is not None:
        return hit
    n = int(index["n"])
    owner = rel_owner(index)
    km1 = np.maximum(_arity(index, B).astype(np.int64) - 1, 1)
    keep = (owner >= 0) & (owner < n)
    den = np.ones(n, dtype=np.int64)
    np.multiply.at(den, owner[keep], km1[keep])
    index["_den"] = den
    return den


def _response_terms(index: dict, P, reading: str, seeds):
    """The reading's contributions, before either denominator is divided out.

        row[i]      the record contribution i lands on
        carried[i]  den[row]/(k_e - 1), or 1 for the existence tower
        seed[i]     which seed carried it
        deg[v]      the incidence count of seed v
        den[r]      the product of (k_e - 1) over the relations r owns

    so that

        response[r] = ( SUM over v of a[r,v]/deg[v] ) / den[r]
        a[r,v]      = SUM over contributions on (r, v) of carried

    Both denominators stay factored. Their common multiple grows without bound as
    seeds are added, where dividing by each axis separately does not, and the two are
    the same number.
    """
    n = int(index["n"])
    B = boundary_operator(index)
    owner = rel_owner(index)
    degree = _vertex_degree_exact(index, B)
    R = _row_operator(index, P, str(reading))
    share = str(reading) == "share"
    den = _record_denominator(index, B) if share else np.ones(n, dtype=np.int64)
    km1 = np.maximum(_arity(index, B).astype(np.int64) - 1, 1)

    rows, carried, which = [], [], []
    for j, v in enumerate(seeds):
        cols = R[int(v)].indices
        r = owner[cols]
        ok = (r >= 0) & (r < n)
        r, cols = r[ok], cols[ok]
        rows.append(r.astype(np.int64))
        carried.append((den[r] // km1[cols]) if share
                       else np.ones(r.size, dtype=np.int64))
        which.append(np.full(r.size, j, dtype=np.int64))
    if not rows:
        return None
    deg = np.array([int(degree[v]) for v in seeds], dtype=np.int64)
    return (np.concatenate(rows), np.concatenate(carried), np.concatenate(which),
            deg, den)


def _accumulate(rows, carried, which):
    """`{row: {seed: a}}` from the contributions. Shared by the exact reading and the
    rendered one, so the two cannot drift."""
    acc: dict[int, dict[int, int]] = {}
    for i in range(len(rows)):
        per = acc.setdefault(int(rows[i]), {})
        v = int(which[i])
        per[v] = per.get(v, 0) + int(carried[i])
    return acc


def _sum_axes(per_seed, deg, den_r):
    """`( SUM_v a_v/deg_v ) / den_r` as an exact Fraction."""
    from fractions import Fraction
    total = Fraction(0)
    for v, a in per_seed.items():
        total += Fraction(int(a), int(deg[v]))
    return total / int(den_r)


def _relation_flux(index: dict, P, reading: str, seeds):
    """`g[e]`, what each relation carries from the seeds, exactly.

        g[e] = ( SUM over seeds v in e of 1/deg[v] ) / (k_e - 1)

    The record reading's two axes, indexed by relation rather than by record, so the
    same kernel evaluates it. A seed is a TERM vertex and a term never sits at position
    0, so every entry it meets is the positive share and `g` carries no sign. The
    existence tower drops the share, leaving a denominator of 1.
    """
    n_rel = int(P.shape[1])
    B = boundary_operator(index)
    degree = _vertex_degree_exact(index, B)
    R = _row_operator(index, P, str(reading))
    share = str(reading) == "share"
    den = (np.maximum(_arity(index, B).astype(np.int64) - 1, 1) if share
           else np.ones(n_rel, dtype=np.int64))
    rows, which = [], []
    for j, v in enumerate(seeds):
        cols = np.asarray(R[int(v)].indices, dtype=np.int64)
        rows.append(cols)
        which.append(np.full(cols.size, j, dtype=np.int64))
    if not rows:
        return np.zeros(n_rel, dtype=np.float64)
    rows = np.concatenate(rows)
    which = np.concatenate(which)
    carried = np.ones(rows.size, dtype=np.int64)
    deg = np.array([int(degree[v]) for v in seeds], dtype=np.int64)
    return _render(rows, carried, which, deg, den, n_rel)


def _render(rows, carried, which, deg, den, n):
    """The contributions as float64, one correctly rounded division per record.

    `rexgraph.core._exact_ratio` divides the seed axis and the relation axis separately
    in 128 bits, so the arithmetic is exact at any seed count and the only rounding is
    the one that makes a double. Without the kernel the same identity runs on python
    ints: a different dtype, not a different answer.
    """
    n = int(n)
    if _exact_ratio is not None:
        widest = int(carried.max(initial=1)) * max(len(rows), 1)
        frac = _exact_ratio.frac_bits_for(widest, len(deg))
        return _exact_ratio.axis_ratio(
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(carried, dtype=np.int64),
            np.ascontiguousarray(which, dtype=np.int64),
            np.ascontiguousarray(deg, dtype=np.int64),
            np.ascontiguousarray(den, dtype=np.int64), n, int(frac))
    out = np.zeros(n, dtype=np.float64)
    for r, per_seed in _accumulate(rows, carried, which).items():
        out[r] = float(_sum_axes(per_seed, deg, den[r]))
    return out


def record_response_exact(index: dict, terms, *, reading: str = "share"):
    """The accession reading over the RATIONALS, with no float anywhere.

    `{row: Fraction}` for the records a seed reaches, and nothing for the rest, which
    answer exactly zero. Every quantity the reading is built from is an exact rational:
    a boundary entry is -1 or `1/(k-1)`, a seed weight is `1/deg`, and a record's answer
    is a sum of five of them. Nothing here needs a tolerance, so nothing here has one.

        response[r] = SUM over the relations e that r owns of
                          (SUM over seeds v in e of 1/deg[v]) / (k_e - 1)

    `record_response` returns the same quantity as float64, which is what a query
    ranks on. This is the reading it answers to, and the tests assert the two orderings
    are identical rather than close.
    """
    codes = _term_codes(index)
    B = boundary_operator(index)
    P = B if str(reading) == "share" else _existence_operator(index, B)
    if str(reading) not in ("share", "existence"):
        raise ValueError(f"reading must be 'share' or 'existence', got {reading!r}")
    seeds = [codes[w] for w in {str(x).lower() for x in terms} if w in codes]
    if not seeds:
        return {}
    got = _response_terms(index, P, str(reading), seeds)
    if got is None:
        return {}
    rows, carried, which, deg, den = got
    out = {}
    for r, per_seed in _accumulate(rows, carried, which).items():
        value = _sum_axes(per_seed, deg, den[r])
        if value:
            out[r] = value
    return out


def rel_owner(index: dict):
    """The record each relation belongs to.

    DERIVED, never stored: the record vertex sits at position 0 of every relation, which
    is the orientation convention the whole index rests on, so the owner is a read of the
    CSR and not a fact about it. Deriving it here is what keeps `build` and `read` the
    same object; storing it in one and not the other made a built index silently unable
    to answer `relations_of`.
    """
    own = index.get("_rel_owner")
    if own is None:
        ptr, idx = index["rel_ptr"], index["rel_idx"]
        own = (np.asarray(idx)[np.asarray(ptr)[:-1]].astype(np.int64)
               if len(ptr) > 1 else np.zeros(0, np.int64))
        index["_rel_owner"] = own
    return own


def _rel_bounds(index: dict):
    """Where each record's run of relations starts, for all records at once.

    The owner is sorted because relations are emitted in row order, so one searchsorted
    over every boundary replaces two per record.
    """
    b = index.get("_rel_bounds")
    if b is None:
        b = np.searchsorted(rel_owner(index), np.arange(index["n"] + 1))
        index["_rel_bounds"] = b
    return b


def relations_of(index: dict, row: int) -> range:
    """The relation indices belonging to record `row`."""
    b = _rel_bounds(index)
    return range(int(b[row]), int(b[row + 1]))


def terms_of(index: dict, row: int) -> dict:
    """{kind: [term, ...]} for one record, read off the incidence.

    Each span is taken as a slice rather than entry by entry: reading a term through a
    numpy scalar costs an extraction per entry, and converting the whole CSR up front
    would charge a page for the whole store. A slice per relation is neither.
    """
    ptr, idx, kind = index["rel_ptr"], index["rel_idx"], index["rel_kind"]
    vocab, n = index["vocab"], index["n"]
    b = _rel_bounds(index)
    lo_e, hi_e = int(b[row]), int(b[row + 1])
    if lo_e == hi_e:
        return {}
    bounds = ptr[lo_e:hi_e + 1].tolist()
    kinds = kind[lo_e:hi_e].tolist()
    out: dict = {}
    for j, k in enumerate(kinds):
        out[KINDS[k]] = [vocab[v - n]
                         for v in idx[bounds[j] + 1:bounds[j + 1]].tolist()]
    return out


def incidence(index: dict):
    """|B1| with the distinguished entries dropped: term vertex x relation, cached.

    The signed operator answers "how does this relation move the complex"; this one
    answers "which relations carry this term", so it is the unsigned twin restricted to
    the shared part of each span. Built once per index straight off the CSR, with no
    pass over relations in python.
    """
    A = index.get("_incidence")
    if A is None:
        import scipy.sparse as sp
        ptr = np.asarray(index["rel_ptr"], np.int64)
        idx = np.asarray(index["rel_idx"], np.int64)
        nE = ptr.size - 1
        keep = np.ones(idx.size, bool)
        keep[ptr[:-1]] = False                  # position 0 of a span is the record
        rows = idx[keep]
        cols = np.repeat(np.arange(nE, dtype=np.int64), np.diff(ptr) - 1)
        A = sp.csr_matrix((np.ones(rows.size, np.float64), (rows, cols)),
                          shape=(index["nV"], nE))
        A.data[:] = 1.0                         # a term repeated in one relation is one
        index["_incidence"] = A
    return A


def records_with_terms(index: dict, terms, *, mode: str = "any",
                       kinds=None) -> np.ndarray:
    """The rows whose accession touches `terms`, as a boundary operation.

    The unsigned incidence applied to the indicator on those term vertices, then summed
    onto the record each relation belongs to. Flat in how many terms are asked for,
    where a per-record set intersection grows with it. `mode="all"` requires every term.
    """
    n = index["n"]
    codes = _term_codes(index)
    want = [codes[t] for t in dict.fromkeys(terms) if t in codes]
    if not want:
        return np.zeros(0, np.int64)
    A = incidence(index)
    per_rel = np.asarray(A[np.asarray(want, np.int64), :].sum(axis=0)).ravel()
    if kinds is not None:
        kc = np.asarray([_KIND_CODE[k] for k in kinds])
        per_rel = np.where(np.isin(np.asarray(index["rel_kind"]), kc), per_rel, 0.0)
    hit = np.zeros(n)
    np.add.at(hit, rel_owner(index), per_rel)
    need = len(want) if mode == "all" else 1
    return np.flatnonzero(hit >= need).astype(np.int64)


#: query key -> (cochain, comparison). The narrowing pass covers every scalar key the
#: caller's predicate supports; term keys go through the incidence.
_NUMERIC = {
    "min_nV": ("nV", "ge"), "max_nV": ("nV", "le"),
    "min_nE": ("nE", "ge"), "max_nE": ("nE", "le"),
    "min_nF": ("nF", "ge"),
    "min_betti1": ("betti1", "ge"), "max_betti1": ("betti1", "le"),
    "min_kappa": ("kappa_mean", "ge"), "max_kappa": ("kappa_mean", "le"),
}
#: query key -> (which kinds to search, mode). None means every kind.
_TERMS = {
    "source": (("source",), "any"),
    "tags_any": (("tags",), "any"), "tags_all": (("tags",), "all"),
    "labels_any": (None, "any"), "labels_all": (None, "all"),
}


def rows_for(index: dict, *, ids=None, as_of=None, **predicate) -> np.ndarray:
    """Row indices matching the predicate: a vectorised read per cochain, and a boundary
    operation for the term keys.

    A superset. Only the keys backed by a cochain or by the incidence narrow, so a
    caller still evaluates the full predicate on what comes back. An unknown key is
    ignored rather than rejected, because the caller owns the key vocabulary.
    """
    n = index["n"]
    keep = np.ones(n, dtype=bool)
    c = index["measures"]
    if ids is not None:
        tbl = index["ids"]
        pos = {rid: i for i, rid in enumerate(tbl)}
        m = np.zeros(n, bool)
        want = [pos[i] for i in ids if i in pos]
        if want:
            m[np.asarray(want, np.int64)] = True
        keep &= m
    for key, (col, how) in _NUMERIC.items():
        v = predicate.get(key)
        if v is None:
            continue
        a = c[col]
        if a.dtype == np.float64:
            # an uncomputed measurement is NaN, and the caller's predicate reads a
            # missing number as zero. Coerce the same way or this stops being a superset
            a = np.nan_to_num(a, nan=0.0)
        keep &= (a >= v) if how == "ge" else (a <= v)
    if predicate.get("chain_valid") is not None:
        keep &= c["chain_valid"].astype(bool) == bool(predicate["chain_valid"])
    if predicate.get("has_voids") is not None:
        keep &= (c["n_voids"] > 0) == bool(predicate["has_voids"])
    if as_of is not None:
        keep &= (c["tx_from"] <= as_of) & (np.isnan(c["tx_to"]) | (c["tx_to"] > as_of))
    for key, (kinds, mode) in _TERMS.items():
        v = predicate.get(key)
        if v is None:
            continue
        terms = [v] if isinstance(v, str) else list(v)
        m = np.zeros(n, bool)
        hit = records_with_terms(index, terms, mode=mode, kinds=kinds)
        if hit.size:
            m[hit] = True
        keep &= m
    return np.flatnonzero(keep)


def ids_of(index: dict, rows) -> list:
    """The record ids for `rows`."""
    tbl = index["ids"]
    return [tbl[int(r)] for r in np.asarray(rows, dtype=np.int64)]


def payload_at(index: dict, row: int) -> dict:
    """{"s": ..., "m": ...} - the residual for one row, from its columns and bridges."""
    out = {"s": {}, "m": {}}
    for e in index.get("residual_groups") or ():
        rows = e["rows"]
        i = int(np.searchsorted(rows, row))
        if i >= rows.size or int(rows[i]) != row:
            continue
        _assign(out["s" if e["scope"] == 0 else "m"], e["path"], _group_value(e, i))
    for e in index.get("residual_bridges") or ():
        if e["row"] == row:
            _assign(out["s" if e["scope"] == 0 else "m"], e["path"],
                    _leaf_value(e["kind"], e["parts"]))
    return out


def record_at(index: dict, row: int):
    """Rebuild a ComplexRecord from its cochains and its relations.

    NaN reads back as None, which is what an open bitemporal bound and an absent
    optional measurement both mean.
    """
    from .core import ComplexRecord
    c = index["measures"]
    extra = payload_at(index, row)
    terms = terms_of(index, row)

    def _f(name):
        v = float(c[name][row])
        return None if np.isnan(v) else v

    def _one(name):
        return (terms.get(name) or [""])[0]
    sig = {
        "object_type": _one("object_type"), "source": _one("source"),
        "nV": int(c["nV"][row]), "nE": int(c["nE"][row]), "nF": int(c["nF"][row]),
        "betti": [int(c["betti0"][row]), int(c["betti1"][row]), int(c["betti2"][row])],
        "betti1": int(c["betti1"][row]),
        "chain_valid": bool(c["chain_valid"][row]),
        "kappa_mean": _f("kappa_mean"),
        "coherence_method": _one("coherence_method"),
        "n_voids": int(c["n_voids"][row]), "n_labels": int(c["n_labels"][row]),
        "tags": terms.get("tags", []),
        "labels_sample": terms.get("labels_sample", []),
    }
    for k in ("kappa_greens_mean", "structural_perplexity", "effective_modes",
              "varentropy_gap"):
        v = _f(k)
        if v is not None:
            sig[k] = v
    sig.update(extra.get("s", {}))
    meta = dict(extra.get("m", {}))
    labels = terms.get("vertex_labels", [])
    if labels:
        meta["vertex_labels"] = labels
    return ComplexRecord(
        id=index["ids"][row], signature=sig, created=float(c["created"][row]),
        meta=meta, version=int(c["version"][row]), tx_from=float(c["tx_from"][row]),
        tx_to=_f("tx_to"), valid_from=_f("valid_from"), valid_to=_f("valid_to"))


#### the append-only log, as frames rather than lines
#
# One line of JSON per change re-encodes the whole record as text on every put, so the
# same field names and the same number formatting are written again for every version.
# A frame carries values and never field names.

LOG_MAGIC = b"REXLOG\x00"
try:                                        # exact 128 bit accumulation and division
    from rexgraph.core import _exact_ratio
except ImportError:                         # source checkout: python ints below
    _exact_ratio = None

_OP_PUT, _OP_DELETE = 1, 2

try:                                        # the compiled frame scan, when built
    from rexgraph.core._recordlog import read_frames as _read_frames
except ImportError:                         # source checkout: the python codec below
    _read_frames = None

_FRAME_COLS = tuple(name for name, _dt in MEASURES)
#: position of each measurement in a frame's scalar row. Indexing the row directly is
#: what lets a record be built without a dict of twenty names standing between.
_COL = {name: i for i, name in enumerate(_FRAME_COLS)}
(_C_VERSION, _C_CREATED, _C_TXF, _C_TXT, _C_VFROM, _C_VTO, _C_NV, _C_NE, _C_NF,
 _C_B0, _C_B1, _C_B2, _C_KAPPA, _C_KGREENS, _C_CHAIN, _C_VOIDS, _C_NLAB,
 _C_PERP, _C_MODES, _C_VGAP) = (_COL[n] for n in _FRAME_COLS)


def _leaves_of(record):
    """The residual leaves of one record as (scope, path, kind, value)."""
    cells = _residual_columns([record])
    return [(sc, path, kind, rowmap[0]) for (sc, path, kind), rowmap in cells.items()]


def log_append(path, op: str, rid: str, record=None, extra=None) -> None:
    """Append one frame: op, id, the measurements, and the record's terms.

    Terms travel with the frame because a log entry has no vocabulary to reference yet;
    the vocabulary is built when the log is folded into an index.

    `extra` is a backend's own int64 row, written after the record and signalled by the
    presence byte reading 2 rather than 1. A log written before this reads unchanged,
    because it only ever wrote 0 or 1 there.
    """
    import os
    rb = rid.encode("utf-8")
    scal = np.zeros(len(_FRAME_COLS), dtype=np.float64)
    strings: list[str] = []
    leaves: list = []
    if record is not None:
        src = _row_values(record)
        for k, name in enumerate(_FRAME_COLS):
            v = src.get(name)
            scal[k] = np.nan if v is None else float(v)
        kinds = _terms_of(record)
        strings.append(str(len(kinds)))
        for kcode, terms in kinds:
            strings.append(str(kcode)); strings.append(str(len(terms)))
            strings.extend(terms)
        for scope, leaf_path, kind, value in _leaves_of(record):
            parts = _leaf_strings(kind, value)
            leaves.append((scope, kind, [b for _s, b in leaf_path], len(parts)))
            strings.extend(seg for seg, _b in leaf_path)
            strings.extend(parts)
    sblob, soffs = _pack_strings(strings)
    soffs = np.asarray(soffs, np.int64)         # the frame parser reads a fixed width
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "ab") as f:
        if new:
            f.write(LOG_MAGIC)
        f.write(np.int8(_OP_PUT if op == "put" else _OP_DELETE).tobytes())
        f.write(np.int32(len(rb)).tobytes()); f.write(rb)
        _has = 0 if record is None else (2 if extra is not None else 1)
        f.write(np.int8(_has).tobytes())
        if record is not None:
            f.write(scal.tobytes())
            f.write(np.int32(soffs.size).tobytes()); f.write(soffs.tobytes())
            f.write(np.int32(len(sblob)).tobytes()); f.write(sblob)
            f.write(np.int32(len(leaves)).tobytes())
            for scope, kind, isidx, nvals in leaves:
                f.write(np.int8([scope, kind]).tobytes())
                f.write(np.int32(len(isidx)).tobytes())
                f.write(np.asarray(isidx, np.int8).tobytes())
                f.write(np.int32(nvals).tobytes())
            if extra is not None:
                ex = np.asarray(extra, np.int64)
                f.write(np.int32(ex.size).tobytes()); f.write(ex.tobytes())
        f.flush()
        os.fsync(f.fileno())


def log_read(path, start: int = 0):
    """Yield (op, id, record|None, extra|None). A torn tail ends the read.

    `start` resumes at a byte offset a caller recorded earlier, which is how a store
    replays only the frames its snapshot does not already hold.

    The scan runs in `rexgraph.core._recordlog` where that is built. The python below
    is the same codec and stays the reference: a source checkout with no compiled core
    still reads its own logs.
    """
    import os
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        buf = f.read()
    if not buf.startswith(LOG_MAGIC):
        return
    at = len(LOG_MAGIC) if start <= len(LOG_MAGIC) else int(start)
    if _read_frames is not None:
        for op, rid, scal, kterms, rest, leaves, extra in _read_frames(
                buf, at, len(_FRAME_COLS)):
            rec = (_record_from_parts(rid, scal, kterms, rest, leaves)
                   if scal is not None else None)
            yield ("put" if op == _OP_PUT else "delete"), rid, rec, extra
        return
    n = len(buf)
    o = len(LOG_MAGIC) if start <= len(LOG_MAGIC) else int(start)
    nscal = len(_FRAME_COLS)
    # `struct` for the single scalars: np.frombuffer builds an array object per field,
    # which at eight fields a frame is most of a replay. Arrays still come off numpy.
    u_i8 = _struct.Struct("<b").unpack_from
    u_i32 = _struct.Struct("<i").unpack_from
    u_2i8 = _struct.Struct("<bb").unpack_from
    while o < n:
        try:
            (op,) = u_i8(buf, o); o += 1
            (ln,) = u_i32(buf, o); o += 4
            if o + ln > n:
                return
            rid = buf[o:o + ln].decode("utf-8"); o += ln
            (has,) = u_i8(buf, o); o += 1
            rec, extra = None, None
            if has:
                scal = np.frombuffer(buf, np.float64, nscal, o); o += 8 * nscal
                (no,) = u_i32(buf, o); o += 4
                soffs = np.frombuffer(buf, np.int64, no, o); o += 8 * no
                (bl,) = u_i32(buf, o); o += 4
                if o + bl > n:
                    return
                strings = _unpack_strings(buf[o:o + bl], soffs); o += bl
                (nl,) = u_i32(buf, o); o += 4
                leaves = []
                for _ in range(nl):
                    scope, kind = u_2i8(buf, o); o += 2
                    (ns,) = u_i32(buf, o); o += 4
                    isidx = np.frombuffer(buf, np.int8, ns, o).copy(); o += ns
                    (nv,) = u_i32(buf, o); o += 4
                    leaves.append((int(scope), int(kind), isidx, nv))
                rec = _record_from_frame(rid, scal, strings, leaves)
                if has == 2:
                    (ne,) = u_i32(buf, o); o += 4
                    extra = np.frombuffer(buf, np.int64, ne, o).copy(); o += 8 * ne
        except (ValueError, IndexError, UnicodeDecodeError, _struct.error):
            return                      # a torn tail: a short read is where it stopped
        yield ("put" if op == _OP_PUT else "delete"), rid, rec, extra


def _record_from_frame(rid, scal, strings, leaves=()):
    """The reference path: split the string table, then build the record."""
    it = iter(strings)
    terms = []
    for _ in range(int(next(it, "0") or 0)):
        code = int(next(it, "0"))
        cnt = int(next(it, "0") or 0)
        terms.append((code, [next(it, "") for _ in range(cnt)]))
    return _record_from_parts(rid, scal, terms, list(it), leaves)


def _record_from_parts(rid, scal, kind_terms, rest, leaves=()):
    """Build a record from a frame whose string table is already split.

    `rexgraph.core._recordlog` splits it during the scan, so the terms arrive as
    `[(kind code, [term])]` and `rest` holds what the residual leaves index.
    """
    from .core import ComplexRecord
    # one conversion to python floats, then `v != v` for the NaN test. Reading the
    # numpy scalars one at a time and calling np.isnan on each was the single largest
    # cost of a log replay.
    # `tolist` once, then read the row by position. Building a dict of twenty names to
    # look each one back out of was the last Python layer in a replay.
    v = scal.tolist()
    terms = {KINDS[code]: vs for code, vs in kind_terms}
    it = iter(rest)
    b1 = int(v[_C_B1] or 0)
    sig = {
        "object_type": (terms.get("object_type") or [""])[0],
        "source": (terms.get("source") or [""])[0],
        "coherence_method": (terms.get("coherence_method") or [""])[0],
        "nV": int(v[_C_NV] or 0), "nE": int(v[_C_NE] or 0), "nF": int(v[_C_NF] or 0),
        "betti": [int(v[_C_B0] or 0), b1, int(v[_C_B2] or 0)],
        "betti1": b1,
        "chain_valid": bool(v[_C_CHAIN]),
        "kappa_mean": None if v[_C_KAPPA] != v[_C_KAPPA] else v[_C_KAPPA],
        "n_voids": int(v[_C_VOIDS] or 0), "n_labels": int(v[_C_NLAB] or 0),
        "tags": terms.get("tags", []), "labels_sample": terms.get("labels_sample", []),
    }
    for name, k in (("kappa_greens_mean", _C_KGREENS),
                    ("structural_perplexity", _C_PERP),
                    ("effective_modes", _C_MODES), ("varentropy_gap", _C_VGAP)):
        if v[k] == v[k]:                      # NaN is how an absent measurement writes
            sig[name] = v[k]
    meta = ({"vertex_labels": terms["vertex_labels"]}
            if terms.get("vertex_labels") else {})
    for scope, kind, isidx, nvals in leaves:
        path = tuple((next(it, ""), bool(b)) for b in isidx)
        parts = [next(it, "") for _ in range(nvals)]
        _assign(sig if scope == 0 else meta, path, _leaf_value(kind, parts))
    return ComplexRecord(
        id=rid, signature=sig, created=float(v[_C_CREATED] or 0.0), meta=meta,
        version=int(v[_C_VERSION] or 1), tx_from=float(v[_C_TXF] or 0.0),
        tx_to=None if v[_C_TXT] != v[_C_TXT] else v[_C_TXT],
        valid_from=None if v[_C_VFROM] != v[_C_VFROM] else v[_C_VFROM],
        valid_to=None if v[_C_VTO] != v[_C_VTO] else v[_C_VTO])
