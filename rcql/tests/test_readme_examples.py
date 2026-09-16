"""Public package examples execute against the same runtime as queries."""
from pathlib import Path
import re

import pytest


@pytest.mark.parametrize("package", ["rcql", "rcdb"])
def test_readme_python_examples(package):
    if package == "rcdb":
        pytest.importorskip("rcdb")
    root = Path(__file__).resolve().parents[2]
    text = (root / package / "README.md").read_text()
    examples = re.findall(r"^```python\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)
    assert examples
    namespace = {}
    for example in examples:
        exec(compile(example, f"{package}/README.md", "exec"), namespace)
