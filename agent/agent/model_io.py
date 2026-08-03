"""
model_io - agent-side IO for MODEL artifacts (GGUF / safetensors weights) and the shared
embedding-corpus persistence path.

This is the agent's counterpart to ``rexgraph.io``: rexgraph.io serializes *relational
complexes* and *vector corpora* (format-level, self-sufficient, no runtime deps); this
module reads *model files* - which is inference-layer concern and must NOT live in the
self-sufficient core (it would drag GGUF/transformers deps into a BLAS-only package and
make a public core path depend on a runtime). Everything here that produces vectors
EMITS into ``rexgraph.io`` via the one container (``save_vectors``/``load_vectors``), so
there is a single on-disk format for embeddings across the whole stack.

Two jobs:
  1. Inspect a model file WITHOUT loading weights - GGUF/safetensors header, tensor
     inventory, arch/params/embedding-dim/quant. Feeds ``local_runtime`` (real size for
     offload decisions) and the UI.
  2. ``save_embedding_corpus`` / ``load_embedding_corpus`` - the ONE embedding-corpus
     round-trip, wrapping ``rexgraph.io`` and stamping model provenance. ``model_introspect``
     and any weight-extraction path both go through here (no duplicated persistence).
"""
from __future__ import annotations

import json
import os
import struct
from typing import Any

import numpy as np

# ggml tensor type id -> name (quantization). Covers the common set; unknown ids fall back
# to ``type{n}`` so a new quant never crashes the reader.
_GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16",
}

# Common token-embedding tensor names across architectures (safetensors weight files).
_EMBED_TENSOR_NAMES = (
    "model.embed_tokens.weight", "embed_tokens.weight", "tok_embeddings.weight",
    "transformer.wte.weight", "wte.weight", "gpt_neox.embed_in.weight",
    "model.tok_embeddings.weight", "embeddings.word_embeddings.weight",
)


# GGUF header

def _gguf_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _gguf_value(f, vtype: int):
    if vtype == 0:  return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:  return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:  return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:  return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:  return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:  return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:  return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:  return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == 8:  return _gguf_str(f)
    if vtype == 9:                                   # ARRAY
        (elem_type,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        # Consume every element (to keep the cursor aligned for later KVs / tensor infos)
        # but only KEEP a small sample - token/merge arrays can be 100k+ entries.
        sample: list[Any] = []
        for i in range(n):
            v = _gguf_value(f, elem_type)
            if i < 8:
                sample.append(v)
        return {"_array_len": int(n), "sample": sample}
    if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11: return struct.unpack("<q", f.read(8))[0]
    if vtype == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_metadata(path: str) -> dict[str, Any]:
    """Parse a GGUF header (magic, version, metadata KVs, tensor inventory) WITHOUT reading
    tensor data. Native parser - no llama.cpp / gguf package needed. Returns
    {version, n_tensors, kv, tensors:[{name, shape, ggml_type, type}]}."""
    p = os.path.expanduser(path)
    with open(p, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file (magic {magic!r}): {path}")
        (version,) = struct.unpack("<I", f.read(4))
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))
        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _gguf_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            kv[key] = _gguf_value(f, vtype)
        tensors: list[dict[str, Any]] = []
        for _ in range(tensor_count):
            name = _gguf_str(f)
            (ndim,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack("<%dQ" % ndim, f.read(8 * ndim)))
            (ttype,) = struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))            # offset - not needed for a summary
            tensors.append({"name": name, "shape": dims, "ggml_type": int(ttype),
                            "type": _GGML_TYPE.get(int(ttype), "type%d" % ttype)})
    return {"version": int(version), "n_tensors": int(tensor_count),
            "kv": kv, "tensors": tensors}


# safetensors header

def read_safetensors_header(path: str) -> dict[str, Any]:
    """Read the safetensors header (tensor names/dtypes/shapes + __metadata__) WITHOUT
    loading any tensor data. Returns {metadata, tensors:{name:{dtype,shape,n_bytes}}}."""
    p = os.path.expanduser(path)
    with open(p, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", {}) or {}
    tensors = {name: {"dtype": v["dtype"], "shape": v["shape"],
                      "n_bytes": int(v["data_offsets"][1] - v["data_offsets"][0])}
               for name, v in hdr.items()}
    return {"metadata": meta, "tensors": tensors, "n_tensors": len(tensors)}


# unified summary

def model_summary(path: str) -> dict[str, Any]:
    """Backend-agnostic summary of a model file (GGUF or safetensors) without loading
    weights: format, on-disk size, architecture, parameter count, layer count, embedding
    dimension, and quant/dtype. Used by ``local_runtime`` for offload decisions and by the
    UI. Missing fields are ``None`` rather than an error."""
    p = os.path.expanduser(path)
    ext = os.path.splitext(p)[1].lower()
    size_gb = round(os.path.getsize(p) / (1024 ** 3), 2) if os.path.exists(p) else None
    out: dict[str, Any] = {"format": ext.lstrip("."), "file_gb": size_gb, "arch": None,
                           "n_params": None, "n_layers": None, "embedding_dim": None,
                           "context_length": None, "quant": None, "n_tensors": None}
    try:
        if ext == ".gguf":
            info = read_gguf_metadata(p)
            kv, tensors = info["kv"], info["tensors"]
            arch = kv.get("general.architecture")
            out["arch"] = arch
            out["n_tensors"] = info["n_tensors"]
            if arch:
                out["n_layers"] = kv.get(f"{arch}.block_count")
                out["embedding_dim"] = kv.get(f"{arch}.embedding_length")
                out["context_length"] = kv.get(f"{arch}.context_length")
            out["n_params"] = int(sum(int(np.prod(t["shape"])) for t in tensors if t["shape"]))
            # dominant tensor quant = the modal ggml type over the big tensors
            types = [t["type"] for t in tensors]
            if types:
                out["quant"] = max(set(types), key=types.count)
        elif ext == ".safetensors":
            h = read_safetensors_header(p)
            ts = h["tensors"]
            out["n_tensors"] = h["n_tensors"]
            out["n_params"] = int(sum(int(np.prod(v["shape"])) for v in ts.values() if v["shape"]))
            dtypes = [v["dtype"] for v in ts.values()]
            if dtypes:
                out["quant"] = max(set(dtypes), key=dtypes.count)
            for name in _EMBED_TENSOR_NAMES:
                if name in ts and len(ts[name]["shape"]) == 2:
                    out["embedding_dim"] = ts[name]["shape"][1]
                    break
            out["arch"] = h["metadata"].get("architecture") or h["metadata"].get("model_type")
    except Exception as e:                               # never let inspection crash a caller
        out["error"] = str(e)
    return out


# embedding-table extraction (weights)

def extract_embedding_table(path: str, *, limit: int | None = None) -> dict[str, Any]:
    """Load a model's token-embedding matrix from a ``.safetensors`` weight file (vocab × dim)
    so it can be analyzed as a relational complex or persisted via ``save_embedding_corpus``.
    GGUF is not supported here - its embedding tensor is quantized and needs the runtime to
    dequantize; use ``model_introspect.embed`` against the running server instead."""
    p = os.path.expanduser(path)
    ext = os.path.splitext(p)[1].lower()
    if ext == ".gguf":
        raise NotImplementedError(
            "GGUF embedding tensors are quantized - dequant needs the llama.cpp runtime. "
            "Start the model and use agent.model_introspect.embed() (server /v1/embeddings).")
    if ext != ".safetensors":
        raise ValueError(f"expected a .safetensors weight file, got {ext!r}")
    from safetensors import safe_open
    with safe_open(p, framework="numpy") as st:
        keys = set(st.keys())
        name = next((n for n in _EMBED_TENSOR_NAMES if n in keys), None)
        if name is None:
            raise KeyError("no known token-embedding tensor in {} (looked for {})".format(path, ", ".join(_EMBED_TENSOR_NAMES)))
        mat = np.asarray(st.get_tensor(name))
    if limit is not None and mat.shape[0] > limit:
        mat = mat[:limit]
    return {"name": name, "matrix": mat.astype(np.float32, copy=False),
            "vocab": int(mat.shape[0]), "dim": int(mat.shape[1])}


# shared embedding-corpus persistence

def save_embedding_corpus(matrix, labels, path: str, *, model: str | None = None,
                          source: str | None = None,
                          feature_names: list[str] | None = None,
                          block_offsets: dict[str, Any] | None = None,
                          **meta) -> str:
    """Persist an embedding corpus (matrix + labels + provenance) through the ONE
    ``rexgraph.io`` vector container. The single home for embedding round-trips - both
    ``model_introspect`` and weight extraction call this, so there is no duplicated format
    code. Returns the written path."""
    from rexgraph.io import save_vectors
    md: dict[str, Any] = {"kind": "embedding_corpus"}
    if model:
        md["model"] = str(model)
    if source:
        md["source"] = str(source)
    md.update({k: (v if isinstance(v, (int, float, str, bool)) else str(v))
               for k, v in meta.items()})
    labs = None if labels is None else np.asarray(labels)
    return str(save_vectors(np.asarray(matrix), labs, path,
                            feature_names=feature_names, block_offsets=block_offsets,
                            metadata=md))


def load_embedding_corpus(path: str):
    """Load an embedding corpus written by ``save_embedding_corpus``. Returns
    ``(matrix, labels, feature_names, metadata)`` - the ``rexgraph.io`` vector tuple."""
    from rexgraph.io import load_vectors
    return load_vectors(path)
