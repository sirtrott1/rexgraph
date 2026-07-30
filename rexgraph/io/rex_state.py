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

from dataclasses import dataclass, field

import numpy as np

FORMAT_VERSION = 1


@dataclass
class RexState:
    tensors: dict          # name to np.ndarray (all numeric payload)
    header: dict = field(default_factory=dict)   # KB json safe: version, types, roles, nested names


# --- the one name codec ---
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
        out = out.replace(ch, "%%%02X" % ord(ch))
    return out


def decode_name(name: str, reserved: str = RESERVED_HIERARCHY) -> str:
    """Inverse of :func:`encode_name` for the same reserved set."""
    for ch in reserved:
        name = name.replace("%%%02X" % ord(ch), ch)
    return name.replace("%25", "%")


def fname_encode(name: str) -> str:
    """The hierarchy case of :func:`encode_name` (used by .rex, hdf5, zarr)."""
    return encode_name(name, RESERVED_HIERARCHY)


def fname_decode(name: str) -> str:
    return decode_name(name, RESERVED_HIERARCHY)


# --- ragged, string, and metadata packing helpers (binary, no JSON of data) ---
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


# --- the canonical (de)serializer ---
def to_state(rex) -> RexState:
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

    # tier 3: user signals (deterministic fields recompute on load; user _signals are stored)
    sig = getattr(rex, "_signals", None)
    if isinstance(sig, np.ndarray):
        t["signals"] = np.asarray(sig)
    return RexState(t, h)


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
            for i, v in zip(idxs, vals):
                rex.attach_metadata(dim, int(i), key, float(v))
        elif kind == "str":
            idxs = np.asarray(t[f"cm_{dim}_{key}_idx"])
            vals = _unpack_strings(t[f"cm_{dim}_{key}_valbytes"], t[f"cm_{dim}_{key}_valoffs"])
            for i, v in zip(idxs, vals):
                rex.attach_metadata(dim, int(i), key, v)
        elif kind == "rex":
            for entry in [n for n in h.get("nested", []) if n["group"] == f"cm_{dim}_{key}"]:
                pref = f"nested/cm_{dim}_{key}/{entry['j']}/"
                sub_t = {k[len(pref):]: v for k, v in t.items() if k.startswith(pref)}
                sub = from_state(RexState(sub_t, entry["header"]))
                rex.attach_metadata(dim, int(col["idx"][entry["j"]]), key, sub)


def from_state(state: RexState):
    from rexgraph.graph import RexGraph
    h, t = state.header, state.tensors
    ver = h.get("format_version")
    if ver != FORMAT_VERSION:
        raise ValueError(f"unsupported rex state format_version {ver!r} (expected {FORMAT_VERSION})")
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
    return rex
