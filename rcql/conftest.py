"""Resolve sibling distributions to their real packages, not the directories shadowing them.

Every distribution here is laid out as <repo>/<name>/<name>, so from the repository root the
OUTER directory shadows the package: `import rcql` finds <repo>/rcql/, which has no
__init__.py, and Python returns a namespace package of unknown location. The editable finder
is appended to sys.meta_path, so the standard path finder answers first and the real package
is never reached.

The symptom is not uniform, which is why this is a fix rather than a note. A suite that
imports names at module scope errors during collection, which is loud. One that imports
inside the test body collects cleanly and fails individual tests, which reads as ordinary
red. Worst is an optional integration guarded by importorskip: a namespace shim imports
successfully, so the guard passes, the real package never executes, and the failure looks
like the feature rather than the path. That case cost time here twice.

Every sibling is resolved rather than a declared list, because a declared list drifts. The
first version of this file named only the siblings each suite touched at the time, and the
next test to reach for a new one reintroduced the bug it was written to prevent.

Prepending a distribution directory makes <repo>/<name>/<name>/__init__.py a regular package,
and a regular package wins over a namespace portion, so every invocation resolves the same
code: from the repository root, from inside this distribution, and from anywhere else.

Each distribution carries its own copy on purpose. Sharing one helper from the repository
root would make these packages depend on a file outside themselves, which is the coupling the
split exists to prevent.
"""
from __future__ import annotations

import pathlib
import sys

_repo = pathlib.Path(__file__).resolve().parent.parent

for _dist in sorted(p for p in _repo.iterdir() if p.is_dir()):
    _name = _dist.name
    if not (_dist / _name / "__init__.py").is_file():
        continue
    _shim = sys.modules.get(_name)
    if _shim is not None and getattr(_shim, "__file__", None) is None:
        for _mod in [m for m in sys.modules if m == _name or m.startswith(_name + ".")]:
            del sys.modules[_mod]
    if str(_dist) not in sys.path:
        sys.path.insert(0, str(_dist))
