"""
agent.rcql_runtime: typed RCQL execution over live agent values.

RCQL evaluates against SOURCES it is handed, never against anything it imports, which is
what lets the language depend on rexgraph alone while still answering questions about a
store, a catalog or a live complex. This is the registry on the agent side: what is
currently bound, and under which policy.

`policy` is the part that matters. A source registered bare answers everything the
operator registry can ask of it. A source wrapped in a `SourcePolicy` answers only what
that policy permits, so a caller can be given records without identity, or history
without the ability to name a record, and the executor enforces it rather than the
caller remembering to.

Ported from the archived work, unchanged apart from taking the source snapshot under the
lock and executing outside it, which the archive already did and which matters because
execution is unbounded while registration is not.
"""

from __future__ import annotations

from threading import RLock

from rcql import BoundSource, Executor, MutationQuery, Query


class RCQLRuntime:
    """Bind live agent values to typed RCQL queries."""

    def __init__(self):
        self._lock = RLock()
        self._sources = {}

    def register(self, name: str, value, *, policy=None) -> None:
        name = str(name).strip()
        if not name:
            raise ValueError("source name cannot be empty")
        if policy is not None:
            value = BoundSource(value, policy)
        with self._lock:
            self._sources[name] = value

    def remove(self, name: str) -> None:
        with self._lock:
            self._sources.pop(str(name), None)

    def sources(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._sources))

    def execute(self, query: Query | MutationQuery, *, params=None):
        """Execute a typed query or mutation against the currently bound sources.

        A string is refused. Parsing belongs to the caller, so the thing executed here is
        already an AST that a policy can be reasoned about against, rather than text this
        layer would have to parse and therefore trust.
        """
        if not isinstance(query, (Query, MutationQuery)):
            raise TypeError(
                "agent RCQL execution requires a typed Query or MutationQuery")
        with self._lock:
            bound = dict(self._sources)
        return Executor(sources=bound, params=params).execute(query)


rcql_runtime = RCQLRuntime()
