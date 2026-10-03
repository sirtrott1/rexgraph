"""Launch import contract probes without executing ambient site startup hooks."""
from __future__ import annotations

import os
from pathlib import Path
import site
import subprocess
import sys


def run_isolated(code: str, *, packages=(), **kwargs):
    roots = list(site.getsitepackages())
    if os.environ.get("REXGRAPH_TEST_INSTALLED") != "1":
        repository = Path(__file__).resolve().parents[1]
        source_roots = [str(repository if name == "rexgraph" else repository / name)
                        for name in packages]
        roots = source_roots + roots
    bootstrap = f"import sys; sys.path[:0] = {roots!r};\n"
    if "rexgraph" in packages:
        # Bootstrap Meson's compiled module finder without running other .pth
        # hooks. A non editable wheel only needs its installed site path.
        bootstrap += (
            "import importlib.util;\n"
            "if importlib.util.find_spec('_rexgraph_editable_loader') is not None:\n"
            " import _rexgraph_editable_loader\n"
            " if hasattr(_rexgraph_editable_loader, 'init'):\n"
            "  _rexgraph_editable_loader.init()\n"
        )
    return subprocess.run([sys.executable, "-I", "-S", "-c", bootstrap + code], **kwargs)
