"""Metadata that PyPI would reject, caught here instead of at upload.

Classifiers are not free text. PyPI validates every one against a fixed list and refuses
the whole upload if any is unknown, so a single wrong string fails a release after the
wheels are built. That is a slow way to learn about a typo.

It was not hypothetical: `Topic :: Scientific/Engineering :: Bioinformatics` sat in
pyproject through six releases and is not a classifier. The real one is `Bio-Informatics`,
with the hyphen. Nothing local ever checks it, because nothing local ever uploads.
"""

from __future__ import annotations

import ast
import re
import runpy
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ("pyproject.toml", "agent/pyproject.toml")

pytestmark = pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(),
                                reason="not a source checkout")


def _project(name: str) -> dict:
    return tomllib.loads((ROOT / name).read_text())["project"]


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_every_classifier_is_a_real_one(manifest):
    trove = pytest.importorskip(
        "trove_classifiers",
        reason="pip install trove-classifiers to check against the published list")
    declared = _project(manifest).get("classifiers", [])
    assert declared, f"{manifest} declares no classifiers"
    unknown = [c for c in declared if c not in trove.classifiers]
    assert not unknown, f"{manifest} would be rejected by PyPI: {unknown}"


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_the_license_is_an_spdx_expression(manifest):
    """The identifier, not the license's prose name.

    `{text = "Apache License 2.0"}` is the deprecated table form AND not an SPDX id, so
    it lands in metadata as an opaque string rather than a License Expression.
    """
    licence = _project(manifest).get("license")
    assert isinstance(licence, str), (
        f"{manifest}: expected an SPDX string, got {licence!r}")
    assert licence == "Apache-2.0", licence


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_the_license_file_is_inside_its_own_project(manifest):
    """A path that escapes the project root is DROPPED, not rejected.

    The agent pointed at `../LICENSE` and built a wheel that declared Apache-2.0 while
    shipping no license at all, which the license itself requires it to ship. Nothing failed;
    the file was simply absent. So the check is that each declared path stays inside the
    manifest's own directory and actually exists there.
    """
    base = (ROOT / manifest).parent
    declared = _project(manifest).get("license-files", [])
    assert declared, f"{manifest} declares no license-files"
    for entry in declared:
        assert not entry.startswith(".."), (
            f"{manifest}: {entry!r} escapes the project root and will be dropped silently")
        assert (base / entry).exists(), f"{manifest}: {entry!r} does not exist"


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_the_project_says_where_it_lives(manifest):
    """A package with no URLs is a page with nowhere to go from."""
    urls = _project(manifest).get("urls", {})
    assert urls, f"{manifest} declares no project URLs"
    assert any("github.com" in v for v in urls.values()), urls


def test_all_top_level_runtime_modules_are_installed_by_meson():
    """Source imports must not mask omitted files in the built wheel.

    Keep the explicit Meson list, but require a decision for every runtime module.
    Tests live in a separate subdirectory and are not runtime modules.
    """
    package = ROOT / "rexgraph"
    manifest = (package / "meson.build").read_text()
    match = re.search(r"py\.install_sources\(\s*(\[.*?\])\s*,", manifest, re.S)
    assert match, "expected an explicit top-level py.install_sources list"
    declared = ast.literal_eval(match.group(1))
    assert len(declared) == len(set(declared)), "duplicate installed source entries"
    actual = {path.name for path in package.glob("*.py")}
    assert set(declared) == actual, {
        "missing_from_wheel": sorted(actual - set(declared)),
        "nonexistent_source": sorted(set(declared) - actual),
    }


@pytest.mark.parametrize("conftest", ["conftest.py", "rcql/conftest.py", "rexgraph/tests/conftest.py"])
def test_installed_test_mode_never_adds_source_fallbacks(conftest, monkeypatch, tmp_path):
    import rexgraph
    monkeypatch.setattr(rexgraph, "__path__", [str(tmp_path)])
    monkeypatch.setenv("REXGRAPH_TEST_INSTALLED", "1")
    before = tuple(sys.path)
    runpy.run_path(str(ROOT / conftest))
    assert tuple(sys.path) == before
    assert rexgraph.__path__ == [str(tmp_path)]
