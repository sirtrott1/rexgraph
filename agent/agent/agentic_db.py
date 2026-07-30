"""agent.agentic_db: a live database as a first-class member of the agentic RCDB.

An AgenticDB wraps a live SQL database (any SQLAlchemy dialect) and makes it
interoperate with the hive and the Relational Complex Database (agent.rcdb):

  * Its schema is reflected into a relational complex and stored in the RCDB, so
    the database's *structure* is queryable topology (circular FK dependencies,
    hub tables, missing-FK voids) and part of agentic memory.
  * A natural-language question is mapped onto that schema complex by the query
    manager: the tables it touches, and - crucially - the JOIN is derived from
    the schema's FK graph (shortest paths, junction tables auto-inserted), not
    guessed by a model. A question referencing tables with no relational path is
    refused as an invalid reference.
  * search / extract / classify are read operations; modify is a guarded write
    (read-only by default; INSERT/UPDATE/DELETE only; DDL blocked).
  * attach_to_hive() registers these as worker bees, so agents operate the
    database through the swarm (hive.invoke), every call recorded in the monitor.

Safety: the connection URI is checked against the DB policy (agent.server.dbguard
- anti-SSRF / allow-lists) before use.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from . import rcdb
from . import schema_complex as sc
from .query_manager import QueryManager

_WRITE_OK = re.compile(r"^\s*(insert|update|delete)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(r"\b(drop|alter|truncate|create|attach|detach|pragma|grant|revoke|vacuum)\b",
                        re.IGNORECASE)


def _as_text(data) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        return data.get("q") or data.get("question") or data.get("text") or ""
    return str(data)


def _as_kwargs(data) -> dict:
    if isinstance(data, dict):
        return data
    return {"table": str(data)}


class AgenticDB:
    """A live SQL database, bridged into the hive and the RCDB."""

    def __init__(self, conn_str: str, *, store: Optional[rcdb.RCStore] = None,
                 writable: bool = False, with_weights: bool = False,
                 guard_uri: bool = True, schema_id: Optional[str] = None):
        if guard_uri:
            try:
                from .server.dbguard import check_db_uri
                check_db_uri(conn_str)
            except ImportError:
                pass
        from sqlalchemy import create_engine
        self.conn_str = conn_str
        self.writable = writable
        self.store = store or rcdb.default_store()
        self._engine = create_engine(conn_str)
        self.model = sc.reflect_schema(conn_str)

        weights = None
        if with_weights:
            try:
                weights, _ = sc.pull_cardinality_stats(conn_str, self.model)
            except Exception:
                weights = None
        self.rex, self.meta = sc.schema_to_rex(self.model, weights=weights)
        self.query_mgr = QueryManager(store=self.store, schema=self.model)
        self.schema_id = schema_id or ("db:" + hashlib.sha1(conn_str.encode()).hexdigest()[:10])
        if self.rex is not None:
            self.store.put(self.schema_id, self.rex, meta=dict(self.meta, source="db"),
                           tags=["schema", "db"])

    # -- structure -------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Full topological diagnosis of the live schema (circular FK deps, hierarchy vs
        tension, missing-FK voids, hub tables). The database's structure, as a complex."""
        return sc.diagnose_schema(self.model)

    def tables(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "columns": t.columns, "primary_key": t.primary_key}
                for t in self.model.tables]

    def _table_names(self) -> set:
        return set(self.model.table_names())

    # -- schema-topological SQL ------------------------------------------------

    def _pk(self, table: str) -> str:
        for t in self.model.tables:
            if t.name == table and t.primary_key:
                return t.primary_key[0]
        return "id"

    def _fk_edges(self):
        adj: Dict[str, List[Tuple[str, tuple]]] = defaultdict(list)
        for fk in self.model.foreign_keys:
            if fk.from_table == fk.to_table:
                continue
            fc = fk.from_cols[0] if fk.from_cols else f"{fk.to_table}_id"
            tc = fk.to_cols[0] if fk.to_cols else self._pk(fk.to_table)
            info = (fk.from_table, fc, fk.to_table, tc)
            adj[fk.from_table].append((fk.to_table, info))
            adj[fk.to_table].append((fk.from_table, info))
        return adj

    def _bfs_path(self, adj, sources: set, target: str):
        """Shortest FK path from any table in `sources` to `target`; list of (table, fk_info)."""
        seen = set(sources)
        q = deque((s, []) for s in sources)
        while q:
            node, acc = q.popleft()
            for nb, info in adj.get(node, ()):
                if nb in seen:
                    continue
                step = acc + [(nb, info)]
                if nb == target:
                    return step
                seen.add(nb)
                q.append((nb, step))
        return None

    def _join_plan(self, touched: List[str]):
        """A join tree covering all touched tables via FK paths (pulling in junctions).
        Returns (ordered_tables, joins) or None if some table is unreachable."""
        if not touched:
            return None
        adj = self._fk_edges()
        tables = [touched[0]]
        tree = {touched[0]}
        joins: List[Tuple[str, tuple]] = []
        for target in touched[1:]:
            if target in tree:
                continue
            path = self._bfs_path(adj, tree, target)
            if path is None:
                return None
            for tbl, info in path:
                if tbl not in tree:
                    tree.add(tbl); tables.append(tbl); joins.append((tbl, info))
        return tables, joins

    def _select_sql(self, plan, limit: int) -> str:
        tables, joins = plan
        sql = f"SELECT * FROM {tables[0]}"
        for tbl, (ft, fc, tt, tc) in joins:
            sql += f" JOIN {tbl} ON {ft}.{fc} = {tt}.{tc}"
        return sql + f" LIMIT {int(limit)}"

    def _execute(self, sql: str):
        from sqlalchemy import text
        with self._engine.connect() as conn:
            res = conn.execute(text(sql))
            cols = list(res.keys())
            rows = [dict(zip(cols, r)) for r in res.fetchall()]
        return rows, cols

    # -- read ------------------------------------------------------------------

    def classify(self, text: str) -> Dict[str, Any]:
        """Map arbitrary text/query onto the schema complex: touched tables, joinability,
        the entity words the schema has no home for. The structural read, no SQL run."""
        return self.query_mgr.open(text).current().schema

    def search(self, question: str, *, limit: int = 50) -> Dict[str, Any]:
        """Answer a question by mapping it to tables, building the join from the FK graph, and
        running the SELECT. Refuses references that cannot join."""
        session = self.query_mgr.open(question)
        sch = session.current().schema
        touched = sch.get("touched_tables", []) if sch.get("linked") else []
        if not touched:
            return {"question": question, "tables": [], "rows": [], "n": 0,
                    "note": "no schema tables referenced"}
        plan = self._join_plan(touched)
        if plan is None:
            return {"question": question, "tables": touched, "rows": [], "n": 0,
                    "error": "referenced tables have no relational path (unjoinable)",
                    "disconnected": sch.get("disconnected_tables")}
        sql = self._select_sql(plan, limit)
        try:
            rows, cols = self._execute(sql)
        except Exception as e:
            return {"question": question, "tables": touched, "sql": sql, "error": str(e)}
        session.resolve(f"{len(rows)} rows across {', '.join(plan[0])}")
        return {"question": question, "tables": touched, "join_tables": plan[0],
                "sql": sql, "columns": cols, "rows": rows, "n": len(rows), "session": session.id}

    def extract(self, table: str, *, columns=None, where: Optional[str] = None,
                limit: int = 100) -> Dict[str, Any]:
        """Read rows from one table (optionally projected/filtered)."""
        if table not in self._table_names():
            return {"error": f"unknown table {table!r}"}
        cols = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        sql += f" LIMIT {int(limit)}"
        try:
            rows, colnames = self._execute(sql)
        except Exception as e:
            return {"table": table, "sql": sql, "error": str(e)}
        return {"table": table, "columns": colnames, "rows": rows, "n": len(rows)}

    def data_complex(self, source: str, *, link_on, id_col: Optional[str] = None,
                     limit: int = 500) -> Dict[str, Any]:
        """Pull rows (a table name or a SELECT) and analyze the DATA as a relational complex:
        cluster records by shared values (connected components), rank them by structural centrality
        (coherence), and flag isolated outliers. The row-level companion to `health()`/schema
        topology - here the returned data is the complex, not the schema."""
        if source in self._table_names():
            rows = self.extract(source, limit=limit).get("rows", [])
        else:
            try:
                rows, _ = self._execute(source)
            except Exception as e:
                return {"error": str(e)}
        from .data_complex import analyze_rows
        r = analyze_rows(rows, link_on=link_on, id_col=id_col)
        r["source"] = source
        return r

    # -- guarded write ---------------------------------------------------------

    def modify(self, statement: str) -> Dict[str, Any]:
        """A guarded write. Read-only unless opened writable; a single INSERT/UPDATE/DELETE
        only; DDL and multi-statement input are refused."""
        if not self.writable:
            return {"ok": False, "error": "database is read-only (open with writable=True)"}
        s = statement.strip().rstrip(";")
        if ";" in s:
            return {"ok": False, "error": "only a single statement is allowed"}
        if _FORBIDDEN.search(s) or not _WRITE_OK.match(s):
            return {"ok": False, "error": "only INSERT/UPDATE/DELETE are permitted"}
        from sqlalchemy import text
        try:
            with self._engine.begin() as conn:
                res = conn.execute(text(s))
                return {"ok": True, "rowcount": res.rowcount}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # -- hive integration ------------------------------------------------------

    def attach_to_hive(self, hive, prefix: str = "db") -> List[str]:
        """Register the database's operations as worker bees so agents run them via hive.invoke().
        Read bees always; the write bee only when the database is writable."""
        added = []
        specs = [
            (f"{prefix}.search", lambda data, **kw: self.search(_as_text(data)), "analyze",
             "db:search", ["database", "sql", "query", prefix]),
            (f"{prefix}.classify", lambda data, **kw: self.classify(_as_text(data)), "analyze",
             "db:classify", ["database", "schema", prefix]),
            (f"{prefix}.schema", lambda data, **kw: self.health(), "analyze",
             "db:schema", ["database", "health", "topology", prefix]),
            (f"{prefix}.extract", lambda data, **kw: self.extract(**_as_kwargs(data)), "transform",
             "db:extract", ["database", "read", prefix]),
        ]
        for name, handler, cap, wtype, spec in specs:
            hive.add_worker(name, handler, capability=cap, worker_type=wtype, specialties=spec)
            added.append(name)
        if self.writable:
            hive.add_worker(f"{prefix}.modify", lambda data, **kw: self.modify(_as_text(data)),
                            capability="transform", worker_type="db:modify",
                            specialties=["database", "write", prefix])
            added.append(f"{prefix}.modify")
        return added
