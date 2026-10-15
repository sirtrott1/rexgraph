"""Configuration errors and interrupted writes preserve usable application state."""
import json
from types import SimpleNamespace

import pytest

from agent.cli import config, serve, setup


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "PID_FILE", tmp_path / "server.pid")
    return tmp_path


@pytest.mark.parametrize("payload", [[], None, 42, "settings", b"\xff"])
def test_invalid_configuration_returns_defaults(files, payload):
    path = files / "config.json"
    path.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    assert config.load_config().gpu_server_port == 10000


def test_invalid_fields_do_not_replace_valid_configuration(files):
    (files / "config.json").write_text(json.dumps({
        "tesseract": "false", "gpu_server_port": True, "agent_server_port": 70000,
        "model_paths": {"example/model": []}, "cache_dir": None,
        "gpu_model": "example/model", "__dict__": {},
    }))
    cfg = config.load_config()
    assert cfg.tesseract is False
    assert cfg.gpu_server_port == 10000
    assert cfg.agent_server_port == 8000
    assert cfg.model_paths == {}
    assert isinstance(cfg.cache_dir, str)
    assert cfg.gpu_model == "example/model"


def test_empty_saved_directories_retain_defaults(files):
    (files / "config.json").write_text('{"cache_dir": "", "models_dir": ""}')
    cfg = config.load_config()
    assert cfg.cache_dir and cfg.models_dir


@pytest.mark.parametrize("writer", ["config", "pid"])
def test_failed_publication_preserves_the_previous_file(files, monkeypatch, writer):
    path = files / ("config.json" if writer == "config" else "server.pid")
    original = b'{"previous": true}'
    path.write_bytes(original)

    def fail(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(config.os, "replace", fail)
    with pytest.raises(OSError):
        if writer == "config":
            config.save_config(config.RexGraphConfig())
        else:
            config.save_pid(123, 10000, "vllm")
    assert path.read_bytes() == original
    assert not list(files.glob("*.tmp"))


@pytest.mark.parametrize("payload", [[], None, {"pid": "123", "port": 8000},
                                    {"pid": -1, "port": 8000}, {"pid": 123, "port": 0},
                                    {"pid": True, "port": 8000}])
def test_invalid_process_record_is_ignored(files, payload):
    (files / "server.pid").write_text(json.dumps(payload))
    assert config.read_pid() is None


@pytest.mark.parametrize("pid", [None, True, 0, -1, "123", 10**100])
def test_invalid_pid_is_not_a_live_process(pid):
    assert not config.process_alive(pid)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_foreground_server_reports_exit_and_clears_pid(files, monkeypatch, exit_code):
    monkeypatch.setattr(serve, "find_running_server", lambda: None)
    monkeypatch.setattr(serve.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(
        pid=123, wait=lambda: exit_code))
    assert serve.serve(model="example/model", foreground=True) is (exit_code == 0)
    assert not config.PID_FILE.exists()


def test_background_server_creates_and_closes_log(files, monkeypatch):
    cache = files / "new" / "cache"
    monkeypatch.setattr(serve, "load_config", lambda: config.RexGraphConfig(cache_dir=str(cache)))
    monkeypatch.setattr(serve, "find_running_server", lambda: None)
    monkeypatch.setattr(serve, "wait_for_health", lambda *a, **kw: True)
    outputs = []

    def start(*args, **kwargs):
        outputs.append(kwargs["stdout"])
        assert not outputs[-1].closed
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(serve.subprocess, "Popen", start)
    assert serve.serve(model="example/model")
    assert (cache / "server.log").is_file()
    assert outputs[0].closed


def test_amd_setup_requires_a_selected_runtime_or_existing_rocm(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(
        version=SimpleNamespace(hip=None)))
    monkeypatch.delenv("TORCH_INDEX_URL", raising=False)
    commands = []
    monkeypatch.setattr(setup.subprocess, "run", lambda cmd, **kw:
                        commands.append(cmd) or SimpleNamespace(returncode=0))
    platform = config.PlatformInfo(gpu="amd")
    assert not setup._install_torch_for_gpu(platform, interactive=False)
    assert not commands
    monkeypatch.setenv("TORCH_INDEX_URL", "https://example.invalid/selected-rocm")
    assert setup._install_torch_for_gpu(platform, interactive=False)
    assert commands[0][-1] == "https://example.invalid/selected-rocm"
    assert "--upgrade" in commands[0] and "--force-reinstall" in commands[0]
    commands.clear()
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(
        version=SimpleNamespace(hip="synthetic")))
    assert setup._install_torch_for_gpu(platform, interactive=False)
    assert not commands


def test_installed_user_checks_use_current_platform_and_corpus_apis(monkeypatch):
    from agent.cli import test_all
    monkeypatch.setattr(config, "detect_platform", lambda: config.PlatformInfo(
        os="linux", gpu="none", python="synthetic"))
    runner = test_all.TestRunner()
    test_all.test_platform(runner)
    test_all.test_corpus(runner)
    assert runner.passed == ["detect_platform", "load_config", "CorpusBuilder.build"]
    assert not runner.failed


def test_dependency_reconciliation_uses_distribution_requirements(monkeypatch):
    commands = []
    monkeypatch.setattr(setup.subprocess, "run", lambda cmd, **kwargs:
                        commands.append(cmd) or SimpleNamespace(returncode=0))
    assert setup._ensure_pip_packages(["paddlepaddle", "pymupdf>=1.22"], interactive=False)
    assert commands[0][-2:] == ["paddlepaddle", "pymupdf>=1.22"]


@pytest.mark.parametrize("gpu", ["amd", "apple"])
def test_non_cuda_backend_requires_a_platform_runtime_before_installing(monkeypatch, gpu):
    monkeypatch.delenv("VLLM_WHEEL_URL", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    calls = []
    monkeypatch.setattr(setup, "_install_torch_for_gpu", lambda *a, **kw: calls.append("torch"))
    monkeypatch.setattr(setup, "_ensure_pip_packages", lambda *a, **kw: calls.append("pip"))
    assert not setup.install_gpu_backend(config.PlatformInfo(gpu=gpu), model="example/model", interactive=False)
    assert not calls
