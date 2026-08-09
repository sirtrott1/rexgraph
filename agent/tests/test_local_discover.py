"""Auto-detection of models already on disk (local_runtime.discover_local_models)."""
import json
import os

from agent import local_runtime


def _touch(path, size_bytes=1024):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size_bytes)


def _blob(path, header=b"", size_bytes=1024):
    """A blob of the stated size, written SPARSE past the header.

    Sizes here are GB-scale on purpose: discovery reports size_gb rounded to two
    decimals, so a megabyte-scale fixture rounds to 0.0 and cannot exercise an
    assertion about the reported size. truncate() makes that free on any filesystem
    that supports sparse files, which is every one the suite runs on."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        if size_bytes > len(header):
            f.truncate(size_bytes)


def _ollama_manifest(path, digest):
    """A minimal ollama manifest: schemaVersion + a single model-weight layer pointing at
    `digest` (the ollama registry manifest format - see docs.ollama.com/api - trimmed to
    the fields discover_local_models actually needs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest = {"schemaVersion": 2,
                "layers": [{"mediaType": "application/vnd.ollama.image.model",
                            "digest": digest, "size": 1}]}
    with open(path, "w") as f:
        json.dump(manifest, f)


def test_discover_finds_gguf_and_skips_vocab_fixtures(tmp_path, monkeypatch):
    d = tmp_path / "models"
    _touch(str(d / "qwen3-8b-q4_k_m.gguf"), 5_000_000)
    _touch(str(d / "ggml-vocab-qwen2.gguf"), 2048)          # llama.cpp test fixture - must be skipped
    monkeypatch.setenv("REXGRAPH_MODEL_DIRS", str(d))

    models = local_runtime.discover_local_models()
    names = {m["name"] for m in models}
    assert "qwen3-8b-q4_k_m" in names
    assert not any(n.startswith("ggml-vocab-") for n in names)
    got = next(m for m in models if m["name"] == "qwen3-8b-q4_k_m")
    assert got["format"] == "gguf" and got["loadable"] == "llama.cpp"


def test_discover_dedupes_split_shards(tmp_path, monkeypatch):
    d = tmp_path / "big"
    _touch(str(d / "glm-4.6-00001-of-00003.gguf"), 3_000_000)
    _touch(str(d / "glm-4.6-00002-of-00003.gguf"), 3_000_000)
    _touch(str(d / "glm-4.6-00003-of-00003.gguf"), 3_000_000)
    monkeypatch.setenv("REXGRAPH_MODEL_DIRS", str(d))

    models = [m for m in local_runtime.discover_local_models() if "glm-4.6" in m["name"]]
    assert len(models) == 1  # one model, not three shards


def test_discover_reports_hf_transformers_snapshot(tmp_path, monkeypatch):
    hub = tmp_path / "huggingface" / "hub"
    snap = hub / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "abc123"
    _touch(str(snap / "config.json"), 200)
    _touch(str(snap / "model.safetensors"), 4_000_000)
    monkeypatch.setenv("REXGRAPH_MODEL_DIRS", "")
    # point _default_scan_dirs at our fake hub by monkeypatching HOME so the hf-cache path matches
    monkeypatch.setattr(local_runtime, "_default_scan_dirs", lambda: [str(hub)])

    models = local_runtime.discover_local_models()
    hits = [m for m in models if m["name"] == "Qwen/Qwen2.5-7B-Instruct"]
    assert hits and hits[0]["format"] == "transformers"
    assert hits[0]["loadable"] == "vllm/transformers"


def test_discover_finds_ollama_gguf_model_via_manifest(tmp_path, monkeypatch):
    # ollama stores models as content-addressed, EXTENSION-LESS blobs under blobs/ - the
    # real name only exists in the manifest, which we must parse to recover it.
    root = tmp_path / ".ollama" / "models"
    digest_hex = "a" * 64
    _blob(str(root / "blobs" / f"sha256-{digest_hex}"), header=b"GGUF", size_bytes=4_000_000)
    _ollama_manifest(str(root / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "8b"),
                      f"sha256:{digest_hex}")
    monkeypatch.setenv("REXGRAPH_MODEL_DIRS", str(root))

    models = local_runtime.discover_local_models()
    hits = [m for m in models if m["name"] == "qwen3:8b"]
    assert hits, f"ollama model not discovered: {[m['name'] for m in models]}"
    got = hits[0]
    assert got["source"] == "ollama"
    assert got["format"] == "gguf"
    assert got["loadable"] == "llama.cpp"


def test_discover_reports_non_gguf_ollama_model_as_not_loadable(tmp_path, monkeypatch):
    # ollama can also store non-GGUF (e.g. MLX) models. llama.cpp cannot load those, so we
    # must not lie and call them "gguf"/"llama.cpp" just because they came from ollama - sniff
    # the actual blob bytes rather than trusting the tag name.
    root = tmp_path / ".ollama" / "models"
    digest_hex = "b" * 64
    _blob(str(root / "blobs" / f"sha256-{digest_hex}"), header=b"\x00\x00\x00\x08{\"format\":\"mlx\"", size_bytes=2_000_000)
    _ollama_manifest(str(root / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "35b-mlx"),
                      f"sha256:{digest_hex}")
    monkeypatch.setenv("REXGRAPH_MODEL_DIRS", str(root))

    models = local_runtime.discover_local_models()
    hits = [m for m in models if m["name"] == "qwen3.5:35b-mlx"]
    assert hits, f"ollama model not discovered: {[m['name'] for m in models]}"
    got = hits[0]
    assert got["source"] == "ollama"
    assert got["format"] != "gguf"        # honest: not GGUF magic bytes
    assert got["loadable"] != "llama.cpp"  # honest: plan_hive/start() must not treat it as spawnable


def test_discover_reports_tensor_sharded_ollama_model_as_not_loadable(tmp_path, monkeypatch):
    # Real-world shape (verified against an actual installed `ollama pull` of an MLX model):
    # some ollama models have NO single "*.model" layer at all - they are split into many
    # per-tensor "*.tensor" layer blobs instead. There is no one file to hand llama-server, so
    # this can never be format=="gguf"/loadable=="llama.cpp" no matter what the tag says.
    root = tmp_path / ".ollama" / "models"
    digests = [f"{c * 64}" for c in ("c", "d")]
    for digest_hex in digests:
        _blob(str(root / "blobs" / f"sha256-{digest_hex}"), size_bytes=8_000_000_000)
    manifest_path = root / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "35b-mlx"
    os.makedirs(manifest_path.parent, exist_ok=True)
    manifest = {"schemaVersion": 2,
                "layers": [{"mediaType": "application/vnd.ollama.image.tensor",
                            "digest": f"sha256:{d}", "size": 8_000_000_000, "name": f"tensor.{i}"}
                           for i, d in enumerate(digests)]}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    monkeypatch.setenv("REXGRAPH_MODEL_DIRS", str(root))

    models = local_runtime.discover_local_models()
    hits = [m for m in models if m["name"] == "qwen3.5:35b-mlx"]
    assert hits, f"tensor-sharded ollama model not discovered: {[m['name'] for m in models]}"
    got = hits[0]
    assert got["source"] == "ollama"
    assert got["format"] != "gguf"
    assert got["loadable"] != "llama.cpp"
    assert got["size_gb"] > 0
