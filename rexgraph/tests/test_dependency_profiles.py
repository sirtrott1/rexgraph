"""Base installation and explicit analysis profiles have distinct dependencies."""
from pathlib import Path
import tomllib


def test_scipy_is_declared_only_where_required():
    root = Path(__file__).resolve().parents[2]
    projects = {name: tomllib.loads((root / path / "pyproject.toml").read_text())["project"]
                for name, path in (("core", "."), ("rcql", "rcql"), ("rcdb", "rcdb"), ("agent", "agent"))}
    for name in ("core", "rcql", "rcdb"):
        assert not any(dep.startswith("scipy") for dep in projects[name]["dependencies"])
    assert "scipy>=1.10" in projects["agent"]["dependencies"]
    for name, profile in (("core", "scipy"), ("core", "oracles"), ("rcdb", "scipy")):
        assert projects[name]["optional-dependencies"][profile] == ["scipy>=1.10"]
    assert "scipy" not in (root / "environment-minimal.yml").read_text().lower()
