"""agent.query_manager - a query is a living relational complex.

Each query is built into its own relational complex (agent.query_engine), but a
real interaction is not one query - it is a *trajectory*: the user (or an agent)
refines, follows up, narrows. This module gives that trajectory a lifecycle:

  * QueryState   - one snapshot: the query complex + its signature + how it maps
                   onto a schema (which tables it touches, whether they can join).
  * QuerySession - the evolving trajectory, with convergence dynamics (is it
                   approaching an end state, or drifting?).
  * QueryManager - owns sessions, links them to a schema complex, and persists
                   resolved queries to the RCDB (the agentic memory cache), so a
                   structurally similar past query can be recalled.

The schema integration is the point: because a schema IS a relational complex
(agent.schema_complex.schema_to_rex), a query complex can be laid over it - the
tables it references are a sub-complex, and the FK graph says whether those
tables are joinable or structurally disconnected (an invalid reference).
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import query_engine as qe
from . import rcdb
from . import schema_complex as sc


# --- schema mapping helpers ---------------------------------------------------

def _norm(w: str) -> str:
    return (w or "").strip().lower()


def _same_entity(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return a == b or a.rstrip("s") == b.rstrip("s")


def _schema_index(model: "sc.SchemaModel"):
    tables = {}
    for name in model.table_names():
        tables[_norm(name)] = name
    cols: Dict[str, List[str]] = {}
    for t in model.tables:
        for c in t.columns:
            cols.setdefault(_norm(c), []).append(t.name)
    return tables, cols


def _fk_adjacency(model: "sc.SchemaModel"):
    adj: Dict[str, set] = {n: set() for n in model.table_names()}
    for fk in model.foreign_keys:
        if fk.from_table in adj and fk.to_table in adj and fk.from_table != fk.to_table:
            adj[fk.from_table].add(fk.to_table)
            adj[fk.to_table].add(fk.from_table)
    return adj


def _shortest_path(adj, a: str, b: str) -> Optional[List[str]]:
    if a == b:
        return [a]
    seen = {a}
    q = deque([[a]])
    while q:
        path = q.popleft()
        for nb in adj.get(path[-1], ()):
            if nb in seen:
                continue
            if nb == b:
                return path + [b]
            seen.add(nb)
            q.append(path + [nb])
    return None


def _relate_to_schema(concepts: List[str], model: Optional["sc.SchemaModel"]) -> Dict[str, Any]:
    """Map a query's concepts onto the schema complex: which tables/columns it
    touches, whether those tables are joinable, and which concepts match nothing."""
    if model is None:
        return {"linked": False}
    tables, cols = _schema_index(model)
    touched, matched_cols, unmatched = [], [], []
    for c in concepts:
        nc = _norm(c)
        hit = next((tables[k] for k in tables if _same_entity(nc, k)), None)
        if hit:
            if hit not in touched:
                touched.append(hit)
            continue
        col_hit = next((cols[k] for k in cols if _same_entity(nc, k)), None)
        if col_hit:
            matched_cols.append({"column": c, "tables": col_hit})
            for tn in col_hit:
                if tn not in touched:
                    touched.append(tn)
            continue
        unmatched.append(c)

    # join analysis: are the touched tables connected in the FK graph?
    adj = _fk_adjacency(model)
    joinable, disconnected, path = True, [], None
    if len(touched) >= 2:
        anchor = touched[0]
        reach = {anchor}
        stack = [anchor]
        while stack:
            n = stack.pop()
            for nb in adj.get(n, ()):
                if nb not in reach:
                    reach.add(nb); stack.append(nb)
        disconnected = [t for t in touched[1:] if t not in reach]
        joinable = not disconnected
        for t in touched[1:]:
            p = _shortest_path(adj, anchor, t)
            if p and len(p) > 1:
                path = p
                break
    return {
        "linked": True,
        "touched_tables": touched,
        "matched_columns": matched_cols,
        "unmatched_concepts": unmatched,     # entity-like words the schema has no home for
        "joinable": joinable,
        "join_path": path,                   # a concrete FK path between touched tables
        "disconnected_tables": disconnected,  # touched tables with no relational path
    }


# --- query lifecycle ----------------------------------------------------------

@dataclass
class QueryState:
    step: int
    text: str
    signature: Dict[str, Any]
    concepts: List[str]
    schema: Dict[str, Any]
    rex: Any = field(default=None, repr=False)

    def public(self) -> dict:
        return {"step": self.step, "text": self.text, "signature": self.signature,
                "schema": self.schema}


def _build_state(step: int, text: str, model) -> QueryState:
    rex, ec = qe.build_query_rex(text)
    sig = qe.query_signature(rex, ec)
    concepts = list(sig.get("concepts", []) or [])
    return QueryState(step=step, text=text, signature=sig, concepts=concepts,
                      schema=_relate_to_schema(concepts, model), rex=rex)


def _delta(prev: QueryState, curr: QueryState) -> Dict[str, Any]:
    a, b = set(prev.concepts), set(curr.concepts)
    union = a | b
    overlap = len(a & b) / len(union) if union else 1.0
    return {"overlap": round(overlap, 3), "added": sorted(b - a), "dropped": sorted(a - b)}


class QuerySession:
    """An evolving query: a sequence of QueryStates with convergence dynamics."""

    def __init__(self, text: str, *, model=None, manager: "QueryManager" = None,
                 sid: Optional[str] = None):
        self.id = sid or uuid.uuid4().hex[:12]
        self.model = model
        self.manager = manager
        self.created = time.time()
        self.status = "open"
        self.answer: Optional[str] = None
        self.states: List[QueryState] = [_build_state(0, text, model)]

    def current(self) -> QueryState:
        return self.states[-1]

    def evolve(self, text: str) -> QueryState:
        """Refine/extend the query; a new complex state is appended to the trajectory."""
        st = _build_state(len(self.states), text, self.model)
        self.states.append(st)
        return st

    def convergence(self) -> Dict[str, Any]:
        """The dynamics of the trajectory, read from structure - no magnitude thresholds.

        - the persistent CORE is the exact intersection of every state's concepts: what the query
          has been about throughout. A non-empty core means a stable subject.
        - the trend is the SIGN of the change in per-step overlap (monotone up = converging,
          monotone down = drifting) plus the exact 'concept set stopped changing' case for stable.
        Both signals are exact-structural, not tuned cutoffs.
        """
        concept_sets = [set(s.concepts) for s in self.states]
        core = sorted(set.intersection(*concept_sets)) if concept_sets else []
        if len(self.states) < 2:
            return {"steps": len(self.states), "overlaps": [], "core": core,
                    "trend": "initial", "progressing": True}
        overlaps = [_delta(self.states[i - 1], self.states[i])["overlap"]
                    for i in range(1, len(self.states))]
        last = _delta(self.states[-2], self.states[-1])
        if not last["added"] and not last["dropped"]:
            trend = "stable"                                    # exact: concept set stopped changing
        elif len(overlaps) == 1:
            trend = "converging" if core else "drifting"        # exact: shares the subject, or disjoint
        elif all(overlaps[i] >= overlaps[i - 1] for i in range(1, len(overlaps))):
            trend = "converging"                                # honing: overlap monotone up
        elif all(overlaps[i] <= overlaps[i - 1] for i in range(1, len(overlaps))):
            trend = "drifting"                                  # wandering: overlap monotone down
        else:
            trend = "mixed"
        progressing = bool(core) and trend != "drifting"        # kept a subject, not wandering off it
        return {"steps": len(self.states), "overlaps": overlaps, "core": core,
                "trend": trend, "progressing": progressing}

    def progressing(self) -> bool:
        """True when the query kept a stable subject (a non-empty persistent core) and is not
        wandering off it - both read from structure, not a magnitude cutoff."""
        return self.convergence()["progressing"]

    def resolve(self, answer: str) -> "QuerySession":
        """Mark the end state and persist the converged query complex to memory."""
        self.status = "resolved"
        self.answer = answer
        if self.manager is not None:
            self.manager._remember(self)
        return self

    def trajectory(self) -> Dict[str, Any]:
        return {"id": self.id, "status": self.status, "answer": self.answer,
                "states": [s.public() for s in self.states],
                "convergence": self.convergence()}


class QueryManager:
    """Owns query sessions, links them to a schema complex, and persists resolved queries to
    the RCDB memory so a structurally similar past query can be recalled (the memory worker)."""

    def __init__(self, store: Optional[rcdb.RCStore] = None, schema=None):
        self.store = store or rcdb.default_store()
        self.schema = schema                       # a SchemaModel, or None
        self._sessions: Dict[str, QuerySession] = {}

    def open(self, text: str, *, schema=None) -> QuerySession:
        s = QuerySession(text, model=schema or self.schema, manager=self)
        self._sessions[s.id] = s
        return s

    def get(self, sid: str) -> Optional[QuerySession]:
        return self._sessions.get(sid)

    def evolve(self, sid: str, text: str) -> QueryState:
        return self._sessions[sid].evolve(text)

    def resolve(self, sid: str, answer: str) -> QuerySession:
        return self._sessions[sid].resolve(answer)

    def _remember(self, session: QuerySession) -> Optional[str]:
        """Memory worker: persist the resolved query complex + its schema footprint."""
        st = session.current()
        if st.rex is None:
            return None
        touched = st.schema.get("touched_tables", []) if st.schema.get("linked") else []
        meta = {"vertex_labels": st.concepts, "source": "query",
                "query_text": st.text, "answer": session.answer, "touched_tables": touched}
        tags = ["query"] + list(touched)
        self.store.put(session.id, st.rex, meta=meta, tags=tags)
        return session.id

    def recall(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find structurally similar past queries in memory, ranked by shared concepts."""
        _, ec = qe.build_query_rex(text)
        concepts = set(qe.query_signature(None, ec).get("concepts", [])) if ec else set()
        out = []
        for rec in self.store.query(limit=200, tags_any=["query"]):
            labels = set(rec.signature.get("labels_sample", []))
            union = labels | concepts
            ov = len(labels & concepts) / len(union) if union else 0.0
            if ov > 0:
                out.append({"id": rec.id, "overlap": round(ov, 3),
                            "query": rec.meta.get("query_text"),
                            "answer": rec.meta.get("answer")})
        out.sort(key=lambda d: -d["overlap"])
        return out[:limit]
