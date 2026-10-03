"""Agent's installed UI is served completely without runtime downloads."""
from html.parser import HTMLParser
from pathlib import Path
from urllib import request

import pytest
from fastapi.testclient import TestClient

from agent.server.app import app
from agent.server.launch import _ensure_ui_assets
from agent.ui_assets import frontend_dir


class Assets(HTMLParser):
    def __init__(self): super().__init__(); self.paths = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if (tag == "script" and key == "src") or (tag == "link" and key == "href"):
                self.paths.append(value)


def test_every_agent_index_dependency_is_served_with_its_license():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200 and 'id="root"' in response.text
        assets = Assets(); assets.feed(response.text)
        assert len(assets.paths) == 5
        for path in assets.paths:
            assert path.startswith("/static/")
            asset = client.get(path)
            assert asset.status_code == 200 and asset.content, path
            if path.endswith((".js", ".jsx")): assert "javascript" in asset.headers["content-type"]
        assert "MIT License" in client.get("/static/REACT-LICENSE.txt").text


def test_agent_launch_validates_ui_offline_without_writing_the_installation(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("startup downloaded or wrote UI code")
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    _ensure_ui_assets()


def test_agent_incomplete_installation_names_the_missing_asset(monkeypatch):
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: False if p.name == "app.jsx" else original(p))
    with pytest.raises(RuntimeError, match="app.jsx.*reinstall"): _ensure_ui_assets()


def test_agent_react_assets_match_the_existing_licensed_runtime():
    root = frontend_dir()
    assert b'"18.2.0"' in (root/"react.production.min.js").read_bytes()
    assert b'"18.2.0"' in (root/"react-dom.production.min.js").read_bytes()
