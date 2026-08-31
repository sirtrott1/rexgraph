"""Reach the sibling rexgraph-rcql when it is not installed.

system depends on rcql as a separate distribution. Preferring an installed one and
falling back to the copy beside it means these tests exercise the real behaviour in a
source checkout without anything being installed into the caller's environment.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_rcql() -> None:
    try:
        import rcql
        if hasattr(rcql, "parse"):
            return
    except ImportError:
        pass
    # The repository root shadows both packages as namespace packages, so a bare import
    # can succeed and still be empty; drop that before pointing at the real one.
    sys.modules.pop("rcql", None)
    sibling = Path(__file__).resolve().parents[2] / "rcql"
    if sibling.is_dir():
        sys.path.insert(0, str(sibling))


_ensure_rcql()
