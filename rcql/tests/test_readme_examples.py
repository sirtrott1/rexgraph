"""Public package examples execute against the same runtime as queries."""
from pathlib import Path
import re

import pytest


@pytest.mark.parametrize("package", ["rcql", "rcdb"])
def test_readme_python_examples(package, tmp_path, monkeypatch):
    namespace = {}
    if package == "rcdb":
        from contextlib import closing

        rcdb = pytest.importorskip("rcdb")
        from rexgraph import RexGraph

        # Recovery and migration examples read an existing legacy store.
        with closing(rcdb.FileStore(str(tmp_path / "legacy"), read_only=False)) as source:
            source.put("example", RexGraph.from_graph([0], [1]), analytics=False)
            namespace["known_store_id"] = source.store_id
    root = Path(__file__).resolve().parents[2]
    text = (root / package / "README.md").read_text()
    examples = re.findall(r"^```python\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)
    assert examples
    # Tutorials intentionally publish local files. Keep their destinations
    # outside the checkout and independent across source/installed test runs.
    monkeypatch.chdir(tmp_path)
    for example in examples:
        exec(compile(example, f"{package}/README.md", "exec"), namespace)
