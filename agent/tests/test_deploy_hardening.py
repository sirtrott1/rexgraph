"""Generated launches use literal arguments and the actual persistent launcher."""
import json
import os
import subprocess
import sys

import pytest

from agent.deploy import DeploymentSpec, generate_bundle


def stub(path, output):
    path.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        f"Path({str(output)!r}).write_text(json.dumps({{'args':sys.argv[1:],'env':dict(os.environ)}}))\n")
    path.chmod(0o755)


def run_entrypoint(bundle, tmp_path, command, env=None):
    bin_path = tmp_path/"bin"; bin_path.mkdir()
    output = tmp_path/"capture.json"; stub(bin_path/command, output)
    root = tmp_path/"app"; (root/"inputs").mkdir(parents=True)
    (root/"inputs"/"input.txt").write_text("input")
    path = tmp_path/"entrypoint.sh"
    path.write_text(bundle["entrypoint.sh"].replace("/app/", str(root)+"/"))
    environ = dict(os.environ, PATH=str(bin_path)+os.pathsep+os.environ["PATH"])
    for name in ("CHAT_MODEL_URL", "RCF_HOST", "RCF_PORT"): environ.pop(name, None)
    environ.update(env or {})
    result = subprocess.run(["bash", str(path)], env=environ, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text())


def test_pipeline_shell_exec_preserves_literal_query_backend_and_endpoint(tmp_path):
    marker = tmp_path/"injected"
    query = f'hello "$(touch {marker})"; `touch {marker}`\nnext line'
    backend = f"backend'; touch {marker}; '"
    url = f"http://host/$(touch {marker})?q='literal'"
    captured = run_entrypoint(generate_bundle(DeploymentSpec(mode="pipeline", query=query,
        backend=backend, model_url=url)), tmp_path, "rexgraph-run")
    args = captured["args"]
    assert args[args.index("--query")+1] == query
    assert args[args.index("--backend")+1] == backend
    assert captured["env"]["CHAT_MODEL_URL"] == url
    assert not marker.exists()


@pytest.mark.parametrize("override", [False, True])
def test_service_entrypoint_reaches_the_environment_based_launcher_and_respects_overrides(tmp_path, override):
    env = {"RCF_HOST": "127.0.0.1", "RCF_PORT": "9002", "CHAT_MODEL_URL": ""} if override else {}
    captured = run_entrypoint(generate_bundle(DeploymentSpec(port=9001, model_url="http://model")),
                              tmp_path, "rcf-server", env)
    assert captured["args"] == []
    assert captured["env"]["RCF_HOST"] == ("127.0.0.1" if override else "0.0.0.0")
    assert captured["env"]["RCF_PORT"] == ("9002" if override else "9001")
    assert captured["env"]["CHAT_MODEL_URL"] == ("" if override else "http://model")


def test_service_bundle_persists_auth_store_audit_and_sessions_and_refuses_failed_local_installs():
    bundle = generate_bundle(DeploymentSpec(source="local"))
    docker = bundle["Dockerfile"]
    assert "|| true" not in docker and "--no-index --find-links /tmp/wheels/" in docker
    assert "REXGRAPH_CONFIG_DIR=/app/data/config" in docker
    assert "REXGRAPH_SESSION_DIR=/app/data/sessions" in docker
    assert "rexgraph_data:/app/data" in bundle["docker-compose.yml"]
    assert "env_file:" in bundle["docker-compose.yml"]
    assert "cp .env.example .env" in bundle["README.md"]


@pytest.mark.parametrize("options", [{"port": True}, {"port": 0}, {"port": 65536}, {"port": "1; echo bad"},
    {"python_version": "3.13\nRUN bad"}, {"python_version": "3.9"}, {"insecure": "false"},
    {"ontology": 1}, {"depth": "standard; bad"}, {"model_url": "http://host\nENV bad=1"}, {"query": "\x00"}])
def test_deployment_refuses_unsafe_or_ambiguous_runtime_fields(options):
    with pytest.raises(ValueError): generate_bundle(DeploymentSpec(**options))


def test_rcf_console_main_consumes_generated_env(monkeypatch):
    from agent.server import app, launch
    captured = {}
    monkeypatch.setattr(launch, "serve", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setenv("RCF_HOST", "0.0.0.0"); monkeypatch.setenv("RCF_PORT", "9001")
    app.main()
    assert captured["host"] == "0.0.0.0" and captured["port"] == 9001


@pytest.mark.parametrize("kind", ["partial-args", "partial-env", "missing-env", "partial-saved", "generation"])
def test_tls_request_never_downgrades_to_plain_http(kind, tmp_path, monkeypatch):
    from agent.server import launch, security
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("REXGRAPH_TLS_CERT", raising=False); monkeypatch.delenv("REXGRAPH_TLS_KEY", raising=False)
    kwargs = {}
    if kind == "partial-args": kwargs["ssl_cert"] = "missing"
    elif kind == "partial-env": monkeypatch.setenv("REXGRAPH_TLS_CERT", "missing")
    elif kind == "missing-env":
        monkeypatch.setenv("REXGRAPH_TLS_CERT", "missing"); monkeypatch.setenv("REXGRAPH_TLS_KEY", "missing")
    elif kind == "partial-saved":
        (tmp_path/"tls").mkdir(); (tmp_path/"tls"/"cert.pem").write_text("invalid")
    else:
        kwargs["https"] = True
        monkeypatch.setattr(security, "generate_self_signed_cert", lambda: {"error": "test failure"})
    with pytest.raises((ValueError, RuntimeError)): launch.resolve_tls(**kwargs)
