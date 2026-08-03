"""Module-level hive task fns for the coordinator proc (forkserver) lane. These MUST be top-level
(picklable and importable in a forkserver child), so a closure over hive state cannot live here."""
from __future__ import annotations


def structural_of(text: str) -> dict:
    """Build a query complex from `text` and return its structural metrics (the eigen-free
    varentropy-gap reliability signal). CPU-bound and pure in its string input, so it is safe on the
    proc lane. Returns {} when the text has no analyzable structure."""
    try:
        from agent.metrics import structural_metrics
        from agent.query_engine import build_query_rex
        rex, _ = build_query_rex(text or "")
        return structural_metrics(rex) if rex is not None else {}
    except Exception:
        return {}
