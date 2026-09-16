"""System's installed UI is self contained and never downloads runtime code."""
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from system.server.app import app
from system.server.launch import _ensure_ui_assets


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if (tag == "script" and key == "src") or (tag == "link" and key == "href"):
                self.paths.append(value)


def test_every_index_dependency_is_served_from_the_package():
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200 and 'id="root"' in page.text
        assets = Assets()
        assets.feed(page.text)
        assert len(assets.paths) >= 6
        for path in assets.paths:
            assert path.startswith("/static/")
            response = client.get(path)
            assert response.status_code == 200 and response.content, path
            if path.endswith((".js", ".jsx")):
                assert "javascript" in response.headers["content-type"]
        assert "MIT License" in client.get("/static/REACT-LICENSE.txt").text


def test_ui_validation_is_offline_and_does_not_write(monkeypatch):
    from pathlib import Path
    from urllib import request
    def forbidden(*args, **kwargs):
        pytest.fail("startup tried to download or write UI code")
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    _ensure_ui_assets()


def test_incomplete_installation_reports_the_missing_assets(monkeypatch):
    from pathlib import Path
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if p.name == "app.jsx" else original(p))
    with pytest.raises(RuntimeError, match="app.jsx.*reinstall"):
        _ensure_ui_assets()
