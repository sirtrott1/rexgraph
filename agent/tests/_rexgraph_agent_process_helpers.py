"""Importable workers for fresh process acceptance tests."""
import os


def append_journal(journal: str, n: int) -> None:
    os.environ["REXGRAPH_AUDIT_JOURNAL"] = journal
    from agent.server import audit
    audit.reset_cache()
    for i in range(n):
        audit.record("test.write", user=f"p{os.getpid()}", target=str(i))
