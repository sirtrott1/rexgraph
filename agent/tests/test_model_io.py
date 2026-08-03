"""
Tests for the model/embedding IO seam:

  * rexgraph.io vector front-door (save_vectors/load_vectors + load() object_type routing)
  * agent.model_io header parsers (GGUF native + safetensors) and model_summary
  * the ONE shared embedding-corpus persist path (model_io <-> rexgraph.io)
  * model_introspect re-analysing a cached corpus without re-embedding

The corpus round-trip and the front-door need the compiled rexgraph.io.safetensors bridge;
they skip cleanly if safetensors is absent. The GGUF/safetensors HEADER parsers are pure
Python and always run.
"""

import struct

import numpy as np
import pytest

# GGUF native header parser

def _write_mini_gguf(path):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    buf = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 4)
    buf += s("general.architecture") + struct.pack("<I", 8) + s("llama")
    buf += s("llama.block_count") + struct.pack("<I", 4) + struct.pack("<I", 32)
    buf += s("llama.embedding_length") + struct.pack("<I", 4) + struct.pack("<I", 4096)
    # a string ARRAY (exercises sampling/cursor alignment)
    buf += s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) \
        + struct.pack("<Q", 3) + s("a") + s("b") + s("c")
    # one tensor info
    buf += s("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 4096, 128000) \
        + struct.pack("<I", 12) + struct.pack("<Q", 0)      # ggml type 12 = Q4_K
    path.write_bytes(buf)


def test_gguf_header_parse(tmp_path):
    from agent import model_io
    p = tmp_path / "mini.gguf"
    _write_mini_gguf(p)
    info = model_io.read_gguf_metadata(str(p))
    assert info["version"] == 3
    assert info["n_tensors"] == 1
    assert info["kv"]["general.architecture"] == "llama"
    assert info["kv"]["llama.block_count"] == 32
    # large arrays are consumed but only sampled
    assert info["kv"]["tokenizer.ggml.tokens"]["_array_len"] == 3
    assert info["kv"]["tokenizer.ggml.tokens"]["sample"] == ["a", "b", "c"]
    t = info["tensors"][0]
    assert t["name"] == "token_embd.weight" and t["type"] == "Q4_K"
    assert t["shape"] == [4096, 128000]


def test_gguf_summary(tmp_path):
    from agent import model_io
    p = tmp_path / "mini.gguf"
    _write_mini_gguf(p)
    s = model_io.model_summary(str(p))
    assert s["format"] == "gguf"
    assert s["arch"] == "llama"
    assert s["n_layers"] == 32
    assert s["embedding_dim"] == 4096
    assert s["quant"] == "Q4_K"
    assert s["n_params"] == 4096 * 128000
    assert "error" not in s


def test_gguf_bad_magic(tmp_path):
    from agent import model_io
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(ValueError):
        model_io.read_gguf_metadata(str(p))
    # model_summary swallows the error into the dict instead of raising
    assert "error" in model_io.model_summary(str(p))


# safetensors header parser

def _write_safetensors(path, tensors):
    """Minimal safetensors writer: {name: ndarray} -> file."""
    import json
    header = {}
    blobs = []
    off = 0
    dt = {"float32": "F32", "int32": "I32", "float64": "F64"}
    for name, arr in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dt[str(arr.dtype)], "shape": list(arr.shape),
                        "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def test_safetensors_header_and_summary(tmp_path):
    from agent import model_io
    p = tmp_path / "w.safetensors"
    _write_safetensors(p, {
        "model.embed_tokens.weight": np.zeros((100, 16), np.float32),
        "layer.0.weight": np.zeros((16, 16), np.float32),
    })
    h = model_io.read_safetensors_header(str(p))
    assert h["n_tensors"] == 2
    assert h["tensors"]["model.embed_tokens.weight"]["shape"] == [100, 16]
    s = model_io.model_summary(str(p))
    assert s["format"] == "safetensors"
    assert s["embedding_dim"] == 16                 # from the embed tensor
    assert s["n_params"] == 100 * 16 + 16 * 16
    assert s["quant"] == "F32"


# shared embedding-corpus round-trip (needs rexgraph.io)

def _have_vectors():
    try:
        from rexgraph.io import load_vectors, save_vectors  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_vectors(), reason="rexgraph.io safetensors bridge unavailable")
def test_vector_front_door_roundtrip(tmp_path):
    import rexgraph.io as io
    M = np.random.RandomState(0).randn(6, 8).astype(np.float32)
    labs = np.array(["a", "a", "b", "b", "c", "c"])
    p = str(tmp_path / "emb.safetensors")
    io.save_vectors(M, labs, p, block_offsets={"sem": (0, 4), "syn": (4, 8)},
                    metadata={"model": "demo"})
    # generic load() must route by object_type, NOT assume a rex complex
    M2, labs2, _names, meta = io.load(p)
    assert np.allclose(M, M2)
    assert list(labs2) == list(labs)
    assert meta["block_offsets"] == {"sem": (0, 4), "syn": (4, 8)}
    assert meta["model"] == "demo"


@pytest.mark.skipif(not _have_vectors(), reason="rexgraph.io safetensors bridge unavailable")
def test_embedding_corpus_shared_path(tmp_path):
    from agent import model_io
    rng = np.random.RandomState(1)
    centers = rng.randn(3, 16) * 3
    V = np.vstack([centers[i] + rng.randn(4, 16) * 0.3 for i in range(3)]).astype(np.float32)
    labels = np.array(["%s_%d" % (n, i) for n in ("x", "y", "z") for i in range(4)])
    p = str(tmp_path / "corpus.safetensors")
    model_io.save_embedding_corpus(V, labels, p, model="demo", source="http://local")
    M2, labs2, _n, meta = model_io.load_embedding_corpus(p)
    assert np.allclose(V, M2)
    assert list(labs2) == list(labels)
    assert meta["kind"] == "embedding_corpus" and meta["model"] == "demo"


@pytest.mark.skipif(not _have_vectors(), reason="rexgraph.io safetensors bridge unavailable")
def test_embedding_complex_from_corpus(tmp_path):
    """Re-analysis of a cached corpus runs the RCF math with no server call."""
    from agent import model_introspect, model_io
    rng = np.random.RandomState(2)
    centers = rng.randn(3, 16) * 3
    V = np.vstack([centers[i] + rng.randn(5, 16) * 0.25 for i in range(3)]).astype(np.float32)
    labels = np.array(["c%d_%d" % (c, i) for c in range(3) for i in range(5)])
    p = str(tmp_path / "corpus.safetensors")
    model_io.save_embedding_corpus(V, labels, p)
    res = model_introspect.embedding_complex_from_corpus(p, top_p=0.9)
    assert res["n_items"] == 15
    assert len(res["betti"]) == 3
    assert res["structural"].get("structural_perplexity") is not None
    assert isinstance(res.get("bridges"), list)
