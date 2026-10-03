"""Resolve sibling source packages during checkout tests.

Prepend distribution directories containing their corresponding package.
REXGRAPH_TEST_INSTALLED=1 disables checkout sibling resolution for wheel tests.
"""
from __future__ import annotations

import os
import pathlib
import sys

_repo = pathlib.Path(__file__).resolve().parent.parent

# An installed wheel run must not fill gaps with checkout siblings.
_siblings = () if os.environ.get("REXGRAPH_TEST_INSTALLED") == "1" else _repo.iterdir()
for _dist in sorted(p for p in _siblings if p.is_dir()):
    _name = _dist.name
    if not (_dist / _name / "__init__.py").is_file():
        continue
    _shim = sys.modules.get(_name)
    if _shim is not None and getattr(_shim, "__file__", None) is None:
        for _mod in [m for m in sys.modules if m == _name or m.startswith(_name + ".")]:
            del sys.modules[_mod]
    if str(_dist) not in sys.path:
        sys.path.insert(0, str(_dist))
