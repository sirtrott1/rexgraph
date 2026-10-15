"""Base installation and explicit analysis profiles have distinct dependencies."""
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib
from packaging.requirements import Requirement


def test_scipy_is_declared_only_where_required():
    root = Path(__file__).resolve().parents[2]
    projects = {name: tomllib.loads((root / path / "pyproject.toml").read_text())["project"]
                for name, path in (("core", "."), ("rcql", "rcql"), ("rcdb", "rcdb"), ("agent", "agent"))}
    for name in ("core", "rcql", "rcdb"):
        assert "scipy" not in {Requirement(dep).name for dep in projects[name]["dependencies"]}
    assert "scipy" in {Requirement(dep).name for dep in projects["agent"]["dependencies"]}
    for name, profile in (("core", "scipy"), ("core", "oracles"), ("rcdb", "scipy")):
        assert {Requirement(dep).name for dep in projects[name]["optional-dependencies"][profile]} == {"scipy"}
    assert "scipy" not in (root / "environment-minimal.yml").read_text().lower()
