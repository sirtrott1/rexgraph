"""The canonical layered binary rex state: ONE (de)serializer every format delegates to.

to_state(rex) produces a RexState{tensors, header}. All numeric payload is named tensors; the
header is a KB scale json safe dict of version plus pointers, types, roles, and nested rex names.
Three tiers:
  tier 1 (pointers, identity): label_names, agent_meta.
  tier 2 (composite binary relations): boundary, B2, w_E, signs, edge_types, directed, g_channel,
    w_boundary.
  tier 3 (rexgraph outputs): signals plus the spectral signature scalars.
Nested rexes (a cell value that is itself a RexGraph) serialize recursively under a namespaced
tensor group.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field

import numpy as np

#: The codec spec rides as a tensor so the container digest covers it; see `to_state`.
CODEC_TENSOR = "codec_spec"

#: 2 added the tensor codec: derivatives and aranges in place of the arrays they generate.
#: A reader that does not know the key would hand a differenced pointer array to the
#: constructor and build a different complex in silence, so the version has to move; both
#: are accepted on READ, and a v1 bundle simply carries no codec.
FORMAT_VERSION = 2
READABLE_VERSIONS = (1, 2)


#: digest framing version. 1 was unframed and collided; 2 length-prefixes every field.
#: Written into the header so a bundle carries the rule it was sealed under.
DIGEST_ALGO = 2


@dataclass
class RexState:
    tensors: dict          # name to np.ndarray (all numeric payload)
    header: dict = field(default_factory=dict)   # KB json safe: version, types, roles, nested names


#### the one name codec
#
# Containers reserve different characters: a hierarchy backend (.rex, hdf5, zarr) cannot
# hold '/' in a name, a filesystem path additionally cannot hold '\@:*?"<>|', and
# safetensors reserves nothing (its keys are arbitrary strings, so '/' is stored
# verbatim). One reversible codec parameterized by the reserved set covers all of them;
# encoding '%' first is what keeps it collision-free, since '%2F' must not decode to '/'
# unless it was encoded as such.

#: hierarchy backends: .rex bundles, hdf5 groups, zarr groups
RESERVED_HIERARCHY = "/"
#: a single filesystem path component, plus '@' which the RCDB uses as its version separator
RESERVED_PATH = "/\\@:*?\"<>|"


def encode_name(name: str, reserved: str = RESERVED_HIERARCHY) -> str:
    """Reversible, collision-free encoding of `name` for a container reserving `reserved`."""
    out = name.replace("%", "%25")
    for ch in reserved:
        out = out.replace(ch, f"%{ord(ch):02X}")
    return out


def decode_name(name: str, reserved: str = RESERVED_HIERARCHY) -> str:
    """Inverse of :func:`encode_name` for the same reserved set."""
    for ch in reserved:
        name = name.replace(f"%{ord(ch):02X}", ch)
    return name.replace("%25", "%")


def fname_encode(name: str) -> str:
    """The hierarchy case of :func:`encode_name` (used by .rex, hdf5, zarr)."""
    return encode_name(name, RESERVED_HIERARCHY)


def fname_decode(name: str) -> str:
    return decode_name(name, RESERVED_HIERARCHY)


#### ragged, string, and metadata packing helpers (binary, no JSON of data)
def _pack_strings(strings):
    """Pack a list of strings into (utf8 bytes uint8 tensor, int64 offsets)."""
    enc = [s.encode("utf-8") for s in strings]
    offs = np.zeros(len(enc) + 1, dtype=np.int64)
    for i, b in enumerate(enc):
        offs[i + 1] = offs[i] + len(b)
    buf = np.frombuffer(b"".join(enc), dtype=np.uint8).copy() if enc else np.zeros(0, np.uint8)
    return buf, offs


def _unpack_strings(buf, offs):
    b = bytes(np.asarray(buf, np.uint8).tobytes())
    return [b[int(offs[i]):int(offs[i + 1])].decode("utf-8") for i in range(len(offs) - 1)]


def _pack_w_boundary(wb: dict):
    """{(edge, point) or int: scalar or array} maps to keys (K,2) int64, offsets (K+1,) int64,
    and flat values. This ragged CSR shape is what set_vertex_attribution and manual w_boundary
    entries actually produce; a plain dict of str/int keys in JSON would lose the tuple shape."""
    keys, offs, vals, scal = [], [0], [], []
    for k, v in wb.items():
        kk = list(k) if isinstance(k, tuple) else [int(k), -1]
        if len(kk) == 1:
            kk = [kk[0], -1]
        keys.append(kk[:2])
        is_scalar = np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0)
        arr = np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()
        vals.append(arr); offs.append(offs[-1] + arr.shape[0])
        scal.append(1 if is_scalar else 0)   # distinguish a stored scalar from a length-1 array
    kt = np.asarray(keys, np.int64).reshape(-1, 2) if keys else np.zeros((0, 2), np.int64)
    vt = np.concatenate(vals) if vals else np.zeros(0, np.float64)
    return kt, np.asarray(offs, np.int64), vt, np.asarray(scal, np.uint8)


def _unpack_w_boundary(kt, offs, vt, st=None) -> dict:
    out = {}
    kt = np.asarray(kt).reshape(-1, 2); offs = np.asarray(offs)
    st = None if st is None else np.asarray(st)
    for i in range(kt.shape[0]):
        e, p = int(kt[i, 0]), int(kt[i, 1])
        key = e if p == -1 else (e, p)
        seg = np.asarray(vt)[int(offs[i]):int(offs[i + 1])]
        scalar = bool(st[i]) if st is not None else (seg.shape[0] == 1)
        out[key] = (float(seg[0]) if scalar else seg.copy())   # scalars stay scalar; arrays stay arrays
    return out



#### the tensor codec
#
# Every array below is exact integers, and two of them are not the object of interest.
#
# A CSR pointer is the INTEGRAL of the arity vector: `boundary_ptr[i+1] - boundary_ptr[i]`
# is how many vertices relation `i` reaches. A string-offset array is the integral of the
# label lengths. Storing an integral of small numbers stores large numbers, and the large
# numbers are what a compressor then has to work on: measured on one document,
# `boundary_ptr` took 19.2 KiB compressed and its first difference took 4.1.
#
# An identity partition's `indptr`/`indices` are `arange`. That is the map saying "one
# section per cell", which is real structure and carries nothing beyond its own length;
# it compressed to 91% of itself because an ascending uint16 run has no redundancy a
# general compressor can name.
#
# So: store the DERIVATIVE and integrate on load, store an `arange` as its endpoints.
# Both are exactly invertible over the integers, which is the only licence a storage
# transform needs: no estimate, no tolerance, nothing to tune. What is NOT done here is
# reordering: the arguments of a boundary column share `1/(k-1)` and look interchangeable,
# but `_leaf_digests` hashes them in order, so sorting them would move the Merkle root.
#
# Applied last on write and first on read, so every digest: the container's and each
# sectioning's: is computed over the same arrays it is checked against.

def _is_arange(a) -> bool:
    """`a` is exactly `arange(a[0], a[0] + len(a))`, over the integers."""
    if a.ndim != 1 or a.size < 2 or not np.issubdtype(a.dtype, np.integer):
        return False
    x = a.astype(np.int64, copy=False)
    return bool(np.array_equal(x, np.arange(int(x[0]), int(x[0]) + x.size)))


def _monotone_columns(a) -> list:
    """Indices of the columns that never decrease, which are the ones worth differencing."""
    x = a.astype(np.int64, copy=False)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    return [j for j in range(x.shape[1]) if bool((np.diff(x[:, j]) >= 0).all())]


def _narrow(a):
    """The same integers in the smallest dtype that holds them. Exact, and the original
    dtype is recorded so the load restores it rather than guessing."""
    if a.size == 0:
        return a
    lo, hi = int(a.min()), int(a.max())
    for dt in ((np.uint8, np.uint16, np.uint32) if lo >= 0
               else (np.int8, np.int16, np.int32)):
        info = np.iinfo(dt)
        if lo >= info.min and hi <= info.max:
            return a.astype(dt, copy=False)
    return a


def encode_tensors(t: dict) -> dict:
    """Transform tensors in place; return the spec that inverts it.

    Only integer arrays of rank 1 or 2 are touched, and only where the array itself says
    the transform applies. Floats, strings-as-bytes and anything higher rank pass
    through untouched.
    """
    spec = {}
    for name in sorted(t):
        a = np.asarray(t[name])
        if not np.issubdtype(a.dtype, np.integer) or a.ndim not in (1, 2) or a.size < 2:
            continue
        if _is_arange(a):
            spec[name] = {"c": "arange", "start": int(a[0]), "n": int(a.size),
                          "dtype": a.dtype.str}
            del t[name]                          # the array IS its two endpoints
            continue
        cols = _monotone_columns(a)
        if not cols:
            continue
        x = a.astype(np.int64, copy=True)
        flat = x.ndim == 1
        x = x.reshape(-1, 1) if flat else x
        for j in cols:
            x[1:, j] = np.diff(x[:, j])          # x[0] keeps the true first value
        x = x.reshape(-1) if flat else x
        spec[name] = {"c": "delta", "cols": [int(j) for j in cols], "dtype": a.dtype.str}
        t[name] = np.ascontiguousarray(_narrow(x))
    return spec


def decode_tensors(t: dict, spec: dict) -> None:
    """Invert :func:`encode_tensors` in place."""
    for name, sp in (spec or {}).items():
        if sp["c"] == "arange":
            t[name] = np.arange(int(sp["start"]), int(sp["start"]) + int(sp["n"]),
                                dtype=np.dtype(sp["dtype"]))
            continue
        if name not in t:
            continue
        x = np.asarray(t[name]).astype(np.int64, copy=True)
        flat = x.ndim == 1
        x = x.reshape(-1, 1) if flat else x
        for j in sp["cols"]:
            x[:, j] = np.cumsum(x[:, j])         # the integral, back again
        x = x.reshape(-1) if flat else x
        t[name] = np.ascontiguousarray(x.astype(np.dtype(sp["dtype"]), copy=False))


#### the canonical (de)serializer
def to_state(rex) -> RexState:
    # Faces and edges added since the last read are PENDING: `add_faces` queues them
    # and `_ensure_clean` is what writes them into the boundary arrays. Reading
    # `_B2_col_ptr` without flushing first serialises an empty B2 under a header that
    # declares nF, so a complex saved straight after `add_faces` loses every face,
    # in every container, silently.
    ensure = getattr(rex, "_ensure_clean", None)
    if callable(ensure):
        ensure()

    t, h = {}, {"format_version": FORMAT_VERSION, "object_type": "RexGraph"}
    h["nV"], h["nE"], h["nF"] = int(rex._nV), int(rex._nE), int(rex._nF)
    h["directed"] = bool(rex._directed)
    h["g_channel"] = getattr(rex, "_g_channel", "raw")

    # tier 2: composite binary relations
    t["boundary_ptr"] = np.asarray(rex._boundary_ptr)
    t["boundary_idx"] = np.asarray(rex._boundary_idx)
    if rex._nF > 0:
        t["B2_col_ptr"] = np.asarray(rex._B2_col_ptr)
        t["B2_row_idx"] = np.asarray(rex._B2_row_idx)
        t["B2_vals"] = np.asarray(rex._B2_vals)
    if rex._w_E is not None:
        t["w_E"] = np.asarray(rex._w_E)
    if rex._signs is not None:
        t["signs"] = np.asarray(rex._signs, dtype=np.float64)
    # edge_types is a deterministic cached_property recomputed from the boundary on load, so it is
    # NOT stored: storing it is dead weight and forces a kernel classification on every save.
    if getattr(rex, "_w_boundary", None):
        kt, offs, vt, st = _pack_w_boundary(rex._w_boundary)
        t["wb_keys"], t["wb_offsets"], t["wb_values"], t["wb_scalar"] = kt, offs, vt, st
    # grades >= 3 (the graded duals list) are first-class Tier-2, stored as CSR triples so grade-
    # general homology round-trips (a bare B1/B2 store silently changes betti on a 3-complex).
    gd = getattr(rex, "_graded_duals", None)
    if gd:
        h["n_graded_duals"] = len(gd)
        h["graded_shapes"] = []
        for g, mat in enumerate(gd):
            csr = mat.tocsr()
            t[f"gd{g}_indptr"] = np.asarray(csr.indptr)
            t[f"gd{g}_indices"] = np.asarray(csr.indices)
            t[f"gd{g}_data"] = np.asarray(csr.data)
            h["graded_shapes"].append([int(csr.shape[0]), int(csr.shape[1])])

    # tier 1: identity, pointers
    am = getattr(rex, "_agent_meta", None)
    if am:
        labels = am.get("vertex_labels")
        if labels:
            buf, loffs = _pack_strings([str(x) for x in labels])
            t["label_bytes"], t["label_offsets"] = buf, loffs
        # small identity scalars/columns stay in the header (KB, not data)
        h["agent_meta"] = {k: v for k, v in am.items() if k != "vertex_labels"}

    # attributes and nesting
    nested = _pack_cell_metadata(getattr(rex, "_cell_metadata", None), t, h)
    h["nested"] = nested

    # sectionings: partitions of THIS field, each with its own digest, so a chapter or
    # paragraph layer can be checked and read without opening the complex it sections
    from rexgraph.sectioning import pack_sectionings
    sect = pack_sectionings(rex, t, h)
    if sect:
        h["sectionings"] = sect
        # the layer hierarchy hashed as its own Merkle tree: the interior nodes ARE the
        # paragraph and chapter digests, so this replaces per-layer hashing rather than
        # adding to it, and it carries inclusion proofs the flat digest cannot.
        from rexgraph.merkle import pack_merkle
        mk = pack_merkle(rex, t, h)
        if mk:
            h["merkle"] = mk

    # tier 3: user signals (deterministic fields recompute on load; user _signals are stored)
    sig = getattr(rex, "_signals", None)
    if isinstance(sig, np.ndarray):
        t["signals"] = np.asarray(sig)
    # names as well as value: a container may carry EXTRA tensors alongside the state
    # (a trained cochain, a mask), and those arrive in the same flat dict on load, so
    # the digest has to say which names it covered or it would be recomputed over a
    # different set than it was written over.
    # LAST, so every digest above (each sectioning's, over its own arrays) was taken
    # over the untransformed tensors, and the container digest below is taken over what
    # is actually written. The load mirrors it: verify, decode, then everything else.
    #
    # The spec is a TENSOR, not a header key, because `state_digest` covers the tensors
    # and nothing else. In the header it would be the one input to reconstruction that
    # the container seal does not reach: rewriting an `arange` codec's `start` would
    # hand the loader a different array with the digest still checking out. As a tensor
    # it is sealed with everything else.
    codec = encode_tensors(t)
    if codec:
        t[CODEC_TENSOR] = np.frombuffer(
            json.dumps(codec, sort_keys=True).encode("utf-8"), dtype=np.uint8).copy()
    h["digest_names"] = sorted(t)
    h["digest_algo"] = DIGEST_ALGO
    h["digest"] = state_digest(t)
    return RexState(t, h)


def state_digest(tensors: dict, names=None, *, algo: int = DIGEST_ALGO) -> str:
    """A sha256 over the tensor payload, order-independent.

    Here rather than in one container because every format delegates to `to_state`, so
    a digest computed at this seam covers `.rex`, hdf5, zarr, safetensors and the wire
    with one rule instead of five. What it answers is narrow and worth stating: whether
    these are the bytes that were written. It is NOT the structural check, which is the
    chain condition and lives on the complex; the two catch different failures and
    neither substitutes for the other. Nor is it a signature: anyone who can rewrite the
    payload can recompute it, so identity needs a key.

    Names are folded in with their bytes, so moving a payload between tensors changes
    the digest, and sorted so the dict's insertion order does not.

    Each field is LENGTH-PREFIXED, and it has to be. Concatenating name, dtype, shape and
    payload unframed leaves the field boundaries ambiguous, and that is not theoretical:
    `{"a": zeros(0), "b": zeros(0)}` and `{"auint8(0,)b": zeros(0)}` produce byte-identical
    streams and therefore the same sha256. Two different objects with one digest is the
    single thing a digest exists to prevent.

    `algo=1` reproduces the unframed stream, and exists only so a bundle written before
    the fix still verifies as what it is rather than reading as corrupt. Nothing writes
    it.
    """
    h = hashlib.sha256()
    legacy = int(algo) == 1
    for name in (sorted(tensors) if names is None else list(names)):
        arr = np.ascontiguousarray(tensors[name])
        for part in (name.encode("utf-8"), str(arr.dtype).encode("utf-8"),
                     str(arr.shape).encode("utf-8"), arr.tobytes()):
            if not legacy:
                h.update(len(part).to_bytes(8, "little"))
            h.update(part)
    return h.hexdigest()


def verify_state(state: RexState) -> bool:
    """Whether a state's tensors still match the digest recorded with them.

    True when no digest was recorded: a bundle written before this existed is old, not
    corrupt, and refusing it would turn an upgrade into data loss.

    A name the digest covered that is no longer present is a failure, not an absence:
    dropping a tensor is exactly the truncation this is here to catch.
    """
    declared = state.header.get("digest")
    if not declared:
        return True
    names = state.header.get("digest_names")
    if names is None:
        names = sorted(state.tensors)
    if any(n not in state.tensors for n in names):
        return False
    # a bundle sealed before the framing fix carries no stamp and must be checked under
    # the rule it was written with, or an upgrade would report every stored object corrupt
    algo = int(state.header.get("digest_algo", 1))
    return hmac.compare_digest(
        str(declared), state_digest(state.tensors, names, algo=algo))


def _pack_cell_metadata(cm, t, h):
    """Per (dim,key) attribute maps to a typed columnar tensor set; schema goes in the header. A
    value that is a RexGraph is stored as a NESTED state under tensor group 'nested/<name>/*'.
    Returns the list of nested entries."""
    nested = []
    if not cm:
        h["cell_meta"] = []
        return nested
    schema = []
    for dim, cells in cm.items():
        by_key = {}
        for idx, kv in cells.items():
            for k, v in kv.items():
                by_key.setdefault(k, []).append((int(idx), v))
        for key, pairs in by_key.items():
            gname = f"cm_{dim}_{key}"
            idxs = np.asarray([p[0] for p in pairs], np.int64)
            vals = [p[1] for p in pairs]
            from rexgraph.graph import RexGraph
            if all(isinstance(v, RexGraph) for v in vals):
                for j, sub in enumerate(vals):
                    sub_state = to_state(sub)
                    pref = f"nested/{gname}/{j}/"
                    for tn, ta in sub_state.tensors.items():
                        t[pref + tn] = ta
                    nested.append({"group": gname, "j": j, "header": sub_state.header})
                schema.append({"dim": int(dim), "key": key, "kind": "rex",
                               "idx": idxs.tolist()})
            elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                t[gname + "_idx"] = idxs
                t[gname + "_val"] = np.asarray([float(v) for v in vals], np.float64)
                schema.append({"dim": int(dim), "key": key, "kind": "num"})
            else:
                t[gname + "_idx"] = idxs
                buf, offs = _pack_strings([str(v) for v in vals])
                t[gname + "_valbytes"] = buf; t[gname + "_valoffs"] = offs
                schema.append({"dim": int(dim), "key": key, "kind": "str"})
    h["cell_meta"] = schema
    return nested


def _unpack_cell_metadata(rex, t, h):
    for col in h.get("cell_meta", []):
        dim, key, kind = col["dim"], col["key"], col["kind"]
        if kind == "num":
            idxs = np.asarray(t[f"cm_{dim}_{key}_idx"]); vals = np.asarray(t[f"cm_{dim}_{key}_val"])
            for i, v in zip(idxs, vals, strict=False):
                rex.attach_metadata(dim, int(i), key, float(v))
        elif kind == "str":
            idxs = np.asarray(t[f"cm_{dim}_{key}_idx"])
            vals = _unpack_strings(t[f"cm_{dim}_{key}_valbytes"], t[f"cm_{dim}_{key}_valoffs"])
            for i, v in zip(idxs, vals, strict=False):
                rex.attach_metadata(dim, int(i), key, v)
        elif kind == "rex":
            for entry in [n for n in h.get("nested", []) if n["group"] == f"cm_{dim}_{key}"]:
                pref = f"nested/cm_{dim}_{key}/{entry['j']}/"
                sub_t = {k[len(pref):]: v for k, v in t.items() if k.startswith(pref)}
                sub = from_state(RexState(sub_t, entry["header"]))
                rex.attach_metadata(dim, int(col["idx"][entry["j"]]), key, sub)


def from_state(state: RexState, *, verify: bool = True):
    """Rebuild the complex a state describes, checking its digest first.

    Checked here rather than in each reader, so `.rex`, hdf5, zarr, safetensors and the
    wire all refuse a payload whose tensors no longer match what was written, instead of
    four of them refusing and one loading it. A state carrying no digest predates this
    and loads unchanged.

    `verify=False` is for a caller that assembled the tensors itself and never wrote a
    digest to check against.
    """
    from rexgraph.graph import RexGraph
    h, t = state.header, state.tensors
    if verify and not verify_state(state):
        raise ValueError(
            "the stored tensors do not match the digest recorded with them: this "
            "object was modified or truncated after it was written")
    ver = h.get("format_version")
    if ver not in READABLE_VERSIONS:
        # A bundle written before the canonical rex-state carried its version under
        # "version" and named its arrays differently. Say that, rather than report
        # a version of None, so the reader knows the file is old and not corrupt.
        if ver is None and "version" in h:
            raise ValueError(
                f"this bundle was written by an older version of rexgraph "
                f"(manifest version {h['version']!r}, no format_version) and cannot "
                f"be read by the current reader, which expects format_version "
                f"{FORMAT_VERSION}. Rebuild it from its source.")
        raise ValueError(f"unsupported rex state format_version {ver!r} "
                         f"(this reader accepts {READABLE_VERSIONS})")
    if CODEC_TENSOR in t:
        t = dict(t)                              # never mutate the caller's state
        spec = json.loads(bytes(np.asarray(t.pop(CODEC_TENSOR)).tobytes()).decode("utf-8"))
        decode_tensors(t, spec)
    kw = {"boundary_ptr": t["boundary_ptr"], "boundary_idx": t["boundary_idx"],
          "directed": h.get("directed", False), "g_channel": h.get("g_channel", "raw")}
    for name in ("B2_col_ptr", "B2_row_idx", "B2_vals"):
        if name in t:
            kw[name] = t[name]
    if "w_E" in t:
        kw["w_E"] = t["w_E"]
    if "signs" in t:
        kw["signs"] = t["signs"]
    if "wb_keys" in t:
        kw["w_boundary"] = _unpack_w_boundary(t["wb_keys"], t["wb_offsets"], t["wb_values"],
                                              t.get("wb_scalar"))
    rex = RexGraph(**kw)
    # Honour the recorded vertex count. The boundary arrays only witness vertices
    # that carry a relation, so a 0-cell incident to nothing is invisible to them
    # and the constructor sizes below it. The header already records nV; dropping it
    # here meant an isolated vertex survived in memory and vanished on reload, which
    # moved beta_0 across a save.
    n_declared = int(h.get("nV", 0) or 0)
    if n_declared > rex.nV:
        rex._nV = n_declared
    # grades >= 3: restore the graded duals so grade-general homology round-trips
    n_gd = int(h.get("n_graded_duals", 0))
    if n_gd:
        from scipy.sparse import csr_matrix
        gd = []
        for g in range(n_gd):
            shape = tuple(h["graded_shapes"][g])
            gd.append(csr_matrix((t[f"gd{g}_data"], t[f"gd{g}_indices"], t[f"gd{g}_indptr"]),
                                 shape=shape))
        rex._graded_duals = gd
    # identity
    am = dict(h.get("agent_meta", {}))
    if "label_bytes" in t:
        am["vertex_labels"] = _unpack_strings(t["label_bytes"], t["label_offsets"])
    if am:
        rex._agent_meta = am
    if "signals" in t:
        rex._signals = np.asarray(t["signals"])
    _unpack_cell_metadata(rex, t, h)
    if h.get("sectionings"):
        from rexgraph.sectioning import unpack_sectionings
        unpack_sectionings(rex, t, h)
    if h.get("merkle"):
        from rexgraph.merkle import unpack_merkle
        unpack_merkle(rex, t, h)
    return rex
