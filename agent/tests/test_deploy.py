"""Tests for agent.deploy - containerizing agents/pipelines for deployment."""

import io
import zipfile

import pytest
from agent.deploy import (
    DeploymentSpec,
    bundle_to_zip,
    generate_bundle,
    spec_from_dict,
    write_bundle,
)


class TestSpec:
    def test_name_sanitized(self):
        s = DeploymentSpec(name="My Agent!! v2").normalized()
        assert s.name == "my-agent-v2"

    def test_mode_and_source_validation(self):
        s = DeploymentSpec(mode="bogus", source="bogus").normalized()
        assert s.mode == "service" and s.source == "pypi"

    def test_extras_deduped_and_server_forced_in_service(self):
        s = DeploymentSpec(mode="service", extras=["ocr", "ocr", "bogus"]).normalized()
        assert "server" in s.extras and "bogus" not in s.extras
        assert s.extras == sorted(set(s.extras))

    def test_pipeline_gets_ocr(self):
        s = DeploymentSpec(mode="pipeline", extras=[]).normalized()
        assert "ocr" in s.extras

    def test_spec_from_dict_ignores_unknown_keys(self):
        s = spec_from_dict({"name": "x", "mode": "pipeline", "junk": 1})
        assert s.name == "x" and s.mode == "pipeline"


class TestBundle:
    def test_all_files_present(self):
        b = generate_bundle(DeploymentSpec())
        for f in ["Dockerfile", "docker-compose.yml", "entrypoint.sh",
                  "rexgraph-agent.json", ".dockerignore", ".env.example", "README.md"]:
            assert f in b and b[f].strip()

    def test_dockerfile_has_blas_and_nonroot(self):
        d = generate_bundle(DeploymentSpec())["Dockerfile"]
        assert "libopenblas-dev" in d       # the runtime BLAS symbols fix
        assert "useradd" in d and "USER rexuser" in d

    def test_service_entrypoint_runs_server(self):
        b = generate_bundle(DeploymentSpec(mode="service", port=9000))
        assert "rcf-server" in b["entrypoint.sh"]
        assert "9000" in b["entrypoint.sh"]
        assert "EXPOSE 9000" in b["Dockerfile"]

    def test_pipeline_entrypoint_runs_headless(self):
        b = generate_bundle(DeploymentSpec(mode="pipeline", query="findings?",
                                           backend="tesseract"))
        ep = b["entrypoint.sh"]
        assert "rexgraph-run" in ep
        assert "findings?" in ep and "tesseract" in ep
        assert "tesseract-ocr" in b["Dockerfile"]

    def test_model_url_wired(self):
        b = generate_bundle(DeploymentSpec(model_url="http://llm:8000"))
        assert "http://llm:8000" in b["Dockerfile"] or "http://llm:8000" in b["docker-compose.yml"]

    def test_builder_config_embedded(self):
        import json
        cfg = {"steps": [{"type": "ocr"}, {"type": "corpus_build"}]}
        b = generate_bundle(DeploymentSpec(builder_config=cfg))
        parsed = json.loads(b["rexgraph-agent.json"])
        assert parsed["builder_config"] == cfg

    def test_local_source_uses_wheels(self):
        b = generate_bundle(DeploymentSpec(source="local"))
        assert "wheels/" in b["Dockerfile"]

    def test_zip_roundtrip_and_exec_bit(self):
        b = generate_bundle(DeploymentSpec(name="z"))
        data = bundle_to_zip(b)
        zf = zipfile.ZipFile(io.BytesIO(data))
        assert set(zf.namelist()) == set(b.keys())
        ep = [i for i in zf.infolist() if i.filename == "entrypoint.sh"][0]
        assert (ep.external_attr >> 16) & 0o111  # executable

    def test_write_bundle(self, tmp_path):
        b = generate_bundle(DeploymentSpec())
        write_bundle(b, str(tmp_path / "out"))
        assert (tmp_path / "out" / "Dockerfile").exists()


@pytest.fixture(scope="module")
def client():
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    from fastapi.testclient import TestClient
    get_auth_manager().disable_auth()
    with TestClient(app) as c:
        yield c


class TestDeployRoutes:
    def test_preview(self, client):
        r = client.post("/api/v1/deploy/preview",
                        json={"name": "my-agent", "mode": "service"})
        assert r.status_code == 200
        assert "Dockerfile" in r.json()
        assert "libopenblas-dev" in r.json()["Dockerfile"]

    def test_bundle_download(self, client):
        r = client.post("/api/v1/deploy/bundle",
                        json={"name": "doc agent", "mode": "pipeline",
                              "query": "findings?",
                              "builder_config": {"steps": [{"type": "ocr"}]}})
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"
        assert "doc-agent-deploy.zip" in r.headers.get("content-disposition", "")
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert "Dockerfile" in zf.namelist()
