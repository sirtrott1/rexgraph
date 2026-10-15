"""Launch import contract probes without executing ambient site startup hooks."""
from __future__ import annotations

import json
import os
import site
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import unquote, urlparse


def run_isolated(code: str, *, packages=(), **kwargs):
    roots = list(site.getsitepackages())
    # A registered dependency directory can come from a directory path in a .pth file.
    # Retain those directories without executing their startup hooks again.
    for path in sys.path:
        if Path(path).name in ("site-packages", "dist-packages") and path not in roots:
            roots.append(path)
    if os.environ.get("REXGRAPH_TEST_INSTALLED") != "1":
        repository = Path(__file__).resolve().parents[1]
        source_roots = [str(repository if name == "rexgraph" else repository / name)
                        for name in packages]
        roots = source_roots + roots
    else:
        # -S disables setuptools editable .pth hooks. Use only the source roots
        # named by installed editable metadata, retaining the wheel isolation
        # contract when no editable distribution is present.
        for name in packages:
            if name == "rexgraph":
                continue
            try:
                direct = distribution(f"rexgraph-{name}").read_text("direct_url.json")
            except PackageNotFoundError:
                continue
            info = json.loads(direct) if direct else {}
            if info.get("dir_info", {}).get("editable"):
                url = urlparse(info["url"])
                if url.scheme != "file" or url.netloc not in ("", "localhost"):
                    raise ValueError(f"nonlocal editable source for {name}")
                roots.insert(0, str(Path(unquote(url.path)).resolve()))
    bootstrap = f"import sys; sys.path[:0] = {roots!r};\n"
    try:
        direct = distribution("rexgraph").read_text("direct_url.json")
    except PackageNotFoundError:
        direct = None
    core_editable = bool((json.loads(direct) if direct else {}).get("dir_info", {}).get("editable"))
    if "rexgraph" in packages and core_editable:
        # Bootstrap Meson's compiled module finder without running other .pth
        # hooks. A non editable wheel only needs its installed site path.
        bootstrap += (
            "import importlib.util;\n"
            "if importlib.util.find_spec('_rexgraph_editable_loader') is not None:\n"
            " import _rexgraph_editable_loader\n"
            " if hasattr(_rexgraph_editable_loader, 'init'):\n"
            "  _rexgraph_editable_loader.init()\n"
        )
    return subprocess.run([sys.executable, "-B", "-I", "-S", "-c", bootstrap + code], **kwargs)
