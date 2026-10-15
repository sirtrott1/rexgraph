"""Local model discovery rejects incomplete downloads and resolves cached revisions."""
import json
import asyncio
from types import SimpleNamespace

import pytest

from agent.model_files import cached_model_path, has_model_weights


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "hub" / "models--example--model" / "snapshots" / "revision"
    path.mkdir(parents=True)
    (path / "config.json").write_text('{}')
    return path


def test_partial_shards_do_not_make_a_model_available(snapshot):
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "layer.a": "model-00001-of-00002.safetensors",
        "layer.b": "model-00002-of-00002.safetensors",
    }}))
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"synthetic")
    assert not has_model_weights(snapshot)
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"synthetic")
    assert has_model_weights(snapshot)


@pytest.mark.parametrize("index", [{}, [], {"weight_map": {}},
                                  {"weight_map": {"layer": "../outside.safetensors"}},
                                  {"weight_map": {"layer": 4}}])
def test_invalid_weight_index_is_not_a_complete_download(snapshot, index):
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    assert not has_model_weights(snapshot)


def test_empty_or_configuration_only_download_is_unavailable(snapshot):
    assert not has_model_weights(snapshot)
    (snapshot / "model.safetensors").touch()
    assert not has_model_weights(snapshot)
    (snapshot / "model.safetensors").write_bytes(b"synthetic")
    assert has_model_weights(snapshot)
    (snapshot / "config.json").write_text('null')
    assert not has_model_weights(snapshot)


def test_hub_revision_and_local_path_resolve_without_network(snapshot, tmp_path):
    (snapshot / "pytorch_model.bin").write_bytes(b"synthetic")
    assert cached_model_path("example/model", tmp_path / "hub") == snapshot
    assert cached_model_path(str(snapshot), tmp_path / "absent") == snapshot
    refs = snapshot.parent.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text("incomplete")
    assert cached_model_path("example/model", tmp_path / "hub") is None
    (refs / "main").write_text(snapshot.name)
    assert cached_model_path("example/model", tmp_path / "hub") == snapshot


def test_model_manager_does_not_report_configuration_as_downloaded(snapshot, tmp_path, monkeypatch):
    from agent.cli import config
    from agent.model_manager import ModelManager
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("agent.model_manager.KNOWN_MODELS", {"example/model": {}})
    manager = ModelManager()
    assert manager.get_model_path("example/model") is None
    assert not manager.scan()[0].downloaded
    (snapshot / "model.safetensors").write_bytes(b"synthetic")
    assert manager.get_model_path("example/model") == str(snapshot)
    assert manager.scan()[0].downloaded


def test_ocr_status_honors_configured_model_paths_and_checks_partial_downloads(tmp_path, monkeypatch):
    from agent.server.routes.ocr import ocr_status
    from agent.cli import serve
    from agent import model_manager
    monkeypatch.setenv("REXGRAPH_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setattr(model_manager, "get_manager", lambda: SimpleNamespace(status=lambda: {}))
    monkeypatch.setattr(serve, "server_status", lambda: {"status": "stopped"})
    deepseek = tmp_path / "models" / "deepseek-ai--DeepSeek-OCR-2"
    deepseek.mkdir(parents=True)
    (deepseek / "config.json").write_text('{}')
    result = asyncio.run(ocr_status())
    assert not result["backends"]["unlimited_ocr"]["model_downloaded"]
    (deepseek / "model.safetensors").write_bytes(b"synthetic")
    result = asyncio.run(ocr_status())
    assert result["backends"]["unlimited_ocr"]["model_downloaded"]
