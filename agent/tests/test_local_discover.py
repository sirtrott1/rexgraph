"""Auto-detection of models already on disk (local_runtime.discover_local_models)."""
import os

from agent import local_runtime


def _touch(path, size_bytes=1024):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size_bytes)


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
