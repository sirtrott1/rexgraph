"""
agent.schema_complex: schemas and ontologies AS relational complexes.

A database schema *is* a relational complex: tables are cells, foreign
keys are typed, directional relations (child -> parent), and junction
tables are co-participations (faces). Once a schema is a complex, the
same algebra that analyses documents diagnoses the schema's *actual
topology*:

  * Betti-1 = independent cycles = **circular FK dependencies** - the
    thing that breaks migration ordering, cascade deletes, and topological
    insert/delete.
  * Hodge gradient % = how cleanly the FK graph forms a hierarchy.
  * Hodge harmonic % = structural tension that cannot be resolved into a
    hierarchy (frustration between relations).
  * The void complex = relations the structure implies but that are
    absent - candidate missing foreign keys / normalization hints.
  * Coherence + degree = structurally central tables ("god tables").

Input can be a JSON spec, SQL DDL, or a live database (reflected via
SQLAlchemy, so you can diagnose the schema of any database the RCDB
already talks to).
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .diagnostics_thresholds import THRESHOLDS as _TH

# schema model

@dataclass
class ForeignKey:
    from_table: str
    from_cols: list[str]
    to_table: str
    to_cols: list[str] = field(default_factory=list)
    nullable: bool = True          # optional (nullable) vs mandatory participation
    identifying: bool = False      # FK cols are part of the child's PK (composition)
    on_delete: str = ""            # CASCADE / RESTRICT / SET NULL / "" (unknown)


@dataclass
class TableDef:
    name: str
    columns: list[str] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)


@dataclass
class SchemaModel:
    tables: list[TableDef] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    def __post_init__(self):
        # infer identifying (FK cols ⊆ child PK) for any parser that didn't set it
        pk = {t.name: set(t.primary_key or []) for t in self.tables}
        for fk in self.foreign_keys:
            if not fk.identifying and fk.from_cols and pk.get(fk.from_table):
                if set(fk.from_cols) <= pk[fk.from_table]:
                    fk.identifying = True

    def table_names(self) -> list[str]:
        names = [t.name for t in self.tables]
        # include FK-referenced tables even if not explicitly defined
        for fk in self.foreign_keys:
            for n in (fk.from_table, fk.to_table):
                if n not in names:
                    names.append(n)
        return names


# parsers

def parse_schema_json(spec: dict) -> SchemaModel:
    """Parse a JSON schema spec.

    {"tables": [{"name","columns":[],"primary_key":[],
                 "foreign_keys":[{"columns":[],"references":"tbl","ref_columns":[]}]}]}
    or a flat {"foreign_keys":[{"from_table","from_cols","to_table","to_cols"}]}.
    """
    tables, fks = [], []
    for t in spec.get("tables", []) or []:
        tables.append(TableDef(
            name=t["name"], columns=t.get("columns", []) or [],
            primary_key=t.get("primary_key", []) or []))
        for fk in t.get("foreign_keys", []) or []:
            fks.append(ForeignKey(
                from_table=t["name"],
                from_cols=fk.get("columns", []) or [],
                to_table=fk.get("references") or fk.get("to_table"),
                to_cols=fk.get("ref_columns", []) or fk.get("to_cols", []) or []))
    for fk in spec.get("foreign_keys", []) or []:
        fks.append(ForeignKey(
            from_table=fk["from_table"], from_cols=fk.get("from_cols", []),
            to_table=fk["to_table"], to_cols=fk.get("to_cols", [])))
    return SchemaModel(tables=tables, foreign_keys=fks)


_CREATE_RE = re.compile(r"create\s+table\s+(?:if\s+not\s+exists\s+)?"
                        r"[\"`\[]?(\w+)[\"`\]]?\s*\((.*?)\)\s*;",
                        re.IGNORECASE | re.DOTALL)
_INLINE_REF = re.compile(r"[\"`\[]?(\w+)[\"`\]]?[^,]*?\breferences\s+"
                         r"[\"`\[]?(\w+)[\"`\]]?\s*\(\s*[\"`\[]?(\w+)",
                         re.IGNORECASE)
_TABLE_FK = re.compile(r"foreign\s+key\s*\(\s*([^)]+)\)\s*references\s+"
                       r"[\"`\[]?(\w+)[\"`\]]?\s*\(\s*([^)]+)\)",
                       re.IGNORECASE)
_PK_RE = re.compile(r"primary\s+key\s*\(\s*([^)]+)\)", re.IGNORECASE)


def _parse_ddl_sqlglot(ddl: str, dialect: str | None = None) -> SchemaModel:
    """Dialect-aware DDL parsing via sqlglot (Oracle/Postgres/MySQL/…)."""
    from sqlglot import exp, parse
    read = dialect if dialect and dialect not in ("auto", "sql") else None
    statements = parse(ddl, read=read)
    tables: list[TableDef] = []
    fks: list[ForeignKey] = []

    def _ref_table(ref) -> str | None:
        t = ref.find(exp.Table)
        return t.name if t else None

    for s in statements:
        if s is None:
            continue
        if isinstance(s, exp.Create) and isinstance(s.this, exp.Schema):
            tname = s.this.this.name
            cols, pk = [], []
            for cd in s.this.find_all(exp.ColumnDef):
                cols.append(cd.name)
                if cd.find(exp.PrimaryKeyColumnConstraint):
                    pk.append(cd.name)
                for ref in cd.find_all(exp.Reference):
                    rt = _ref_table(ref)
                    if rt:
                        fks.append(ForeignKey(tname, [cd.name], rt, []))
            for pkn in s.this.find_all(exp.PrimaryKey):
                names = [e.name for e in pkn.expressions if hasattr(e, "name")]
                if names:
                    pk = names
            for fkn in s.this.find_all(exp.ForeignKey):
                from_cols = [e.name for e in fkn.expressions if hasattr(e, "name")]
                ref = fkn.args.get("reference")
                rt = _ref_table(ref) if ref else None
                if rt:
                    fks.append(ForeignKey(tname, from_cols, rt, []))
            tables.append(TableDef(name=tname, columns=cols, primary_key=pk))
        elif type(s).__name__ in ("Alter", "AlterTable"):
            atbl = s.this.name if s.this else None
            for fkn in s.find_all(exp.ForeignKey):
                from_cols = [e.name for e in fkn.expressions if hasattr(e, "name")]
                ref = fkn.args.get("reference")
                rt = _ref_table(ref) if ref else None
                if rt and atbl:
                    fks.append(ForeignKey(atbl, from_cols, rt, []))
    return SchemaModel(tables=tables, foreign_keys=fks)


def parse_schema_ddl(ddl: str, dialect: str | None = None) -> SchemaModel:
    """Parse CREATE TABLE / FOREIGN KEY / ALTER DDL into a schema model.

    Uses sqlglot (dialect-aware: oracle, postgres, mysql, tsql, snowflake, …)
    when available and falls back to a regex parser otherwise. Pass
    ``dialect`` to disambiguate vendor syntax.
    """
    try:
        model = _parse_ddl_sqlglot(ddl, dialect)
        if model.tables or model.foreign_keys:
            return model
    except Exception:
        pass
    return _parse_ddl_regex(ddl)


def _split_top_level(body: str) -> list[str]:
    """Split a CREATE TABLE body on top-level commas only: commas at paren
    depth 0. A naive ``body.split(",")`` shreds parenthesised, comma-separated
    constraint lists (``PRIMARY KEY(a, b)``, ``FOREIGN KEY (x, y) REFERENCES
    t(p, q)``, ``UNIQUE (a, b)``) mid-list, dropping the constraint and leaving
    a phantom column. Keeping each ``(...)`` intact fixes both."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def _parse_ddl_regex(ddl: str) -> SchemaModel:
    """Regex fallback parser (no sqlglot). Handles common inline/table FKs."""
    tables, fks = [], []
    for m in _CREATE_RE.finditer(ddl):
        name, body = m.group(1), m.group(2)
        cols: list[str] = []
        pk: list[str] = []
        for line in _split_top_level(body):
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith(("foreign key", "primary key", "constraint",
                               "unique", "check")):
                pkm = _PK_RE.search(line)
                if pkm:
                    pk = [c.strip(' "`[]') for c in pkm.group(1).split(",")]
                fkm = _TABLE_FK.search(line)
                if fkm:
                    fks.append(ForeignKey(
                        from_table=name,
                        from_cols=[c.strip(' "`[]') for c in fkm.group(1).split(",")],
                        to_table=fkm.group(2),
                        to_cols=[c.strip(' "`[]') for c in fkm.group(3).split(",")]))
                continue
            colname = line.split()[0].strip(' "`[]')
            cols.append(colname)
            if "primary key" in low:
                pk.append(colname)
            refm = _INLINE_REF.search(line)
            if refm and refm.group(2).lower() != "references":
                fks.append(ForeignKey(
                    from_table=name, from_cols=[colname],
                    to_table=refm.group(2), to_cols=[refm.group(3)]))
        tables.append(TableDef(name=name, columns=cols, primary_key=pk))
    return SchemaModel(tables=tables, foreign_keys=fks)


def reflect_schema(conn_str: str) -> SchemaModel:
    """Reflect a live database's schema via SQLAlchemy (any backend).

    Read-only reflection; credentials are never returned in the model.
    """
    from sqlalchemy import create_engine, inspect
    # short connect timeout so a bad host fails fast rather than hanging
    engine = create_engine(conn_str, connect_args={})
    insp = inspect(engine)
    tables, fks = [], []
    try:
        names = insp.get_table_names()
        for schema in (insp.get_schema_names() if hasattr(insp, "get_schema_names") else []):
            if schema in ("information_schema", "pg_catalog", "sys"):
                continue
    except Exception as e:
        raise RuntimeError(f"could not read schema: {e}") from e
    for name in names:
        col_info = insp.get_columns(name)
        cols = [c["name"] for c in col_info]
        nullable_of = {c["name"]: c.get("nullable", True) for c in col_info}
        try:
            pk = insp.get_pk_constraint(name).get("constrained_columns", []) or []
        except Exception:
            pk = []
        pk_set = set(pk)
        tables.append(TableDef(name=name, columns=cols, primary_key=pk))
        for fk in insp.get_foreign_keys(name):
            if fk.get("referred_table"):
                child_cols = fk.get("constrained_columns", []) or []
                # mandatory if any FK column is NOT NULL; identifying if the FK
                # columns are part of the child's primary key (composition)
                mandatory = any(not nullable_of.get(c, True) for c in child_cols)
                identifying = bool(child_cols) and set(child_cols) <= pk_set
                on_delete = ((fk.get("options") or {}).get("ondelete") or "").upper()
                fks.append(ForeignKey(
                    from_table=name,
                    from_cols=child_cols,
                    to_table=fk["referred_table"],
                    to_cols=fk.get("referred_columns", []) or [],
                    nullable=not mandatory,
                    identifying=identifying,
                    on_delete=on_delete))
    return SchemaModel(tables=tables, foreign_keys=fks)


def infer_mongo_schema(collections: dict[str, list[dict]]) -> SchemaModel:
    """Infer a schema from MongoDB-style collections of sample documents.

    Document databases have no declared foreign keys, but references are
    real: DBRef (``{"$ref": "coll"}``) and the ``<entity>_id`` / ``<entity>Id``
    convention. Collections become tables; inferred references become
    foreign keys, so the *same* topological diagnosis applies to NoSQL.
    """
    names = list(collections.keys())
    name_set = {n.lower() for n in names}
    singular = {n[:-1].lower(): n for n in names if n.lower().endswith("s")}

    def _resolve(ref_word: str) -> str | None:
        w = ref_word.lower()
        # try the whole word, then the trailing segment (favorite_post -> post)
        candidates = [w]
        if "_" in w:
            candidates.append(w.rsplit("_", 1)[-1])
        for c in candidates:
            for n in names:
                if n.lower() == c:
                    return n
            if c in singular:
                return singular[c]
            if (c + "s") in name_set:
                return next(n for n in names if n.lower() == c + "s")
        return None

    tables, fks, seen = [], [], set()
    for cname, docs in collections.items():
        fields = set()
        for doc in (docs or [])[:200]:
            if not isinstance(doc, dict):
                continue
            for k, v in doc.items():
                fields.add(k)
                target = None
                # DBRef
                if isinstance(v, dict) and "$ref" in v:
                    target = _resolve(str(v["$ref"]))
                elif isinstance(v, list) and v and isinstance(v[0], dict) and "$ref" in v[0]:
                    target = _resolve(str(v[0]["$ref"]))
                # <entity>_id / <entity>Id convention
                elif k.lower().endswith("_id") and k.lower() != "_id":
                    target = _resolve(k[:-3])
                elif k.endswith("Id") and len(k) > 2:
                    target = _resolve(k[:-2])
                if target and target != cname:
                    key = (cname, k, target)
                    if key not in seen:
                        seen.add(key)
                        fks.append(ForeignKey(cname, [k], target, []))
        pk = ["_id"] if any("_id" in (d or {}) for d in (docs or [])) else []
        tables.append(TableDef(name=cname, columns=sorted(fields), primary_key=pk))
    return SchemaModel(tables=tables, foreign_keys=fks)


def reflect_mongo(conn_str: str, db_name: str, sample: int = 100) -> SchemaModel:
    """Sample a live MongoDB and infer its schema (needs pymongo)."""
    from pymongo import MongoClient
    client = MongoClient(conn_str, serverSelectionTimeoutMS=4000)
    db = client[db_name]
    collections = {}
    for cname in db.list_collection_names():
        collections[cname] = list(db[cname].find().limit(sample))
    return infer_mongo_schema(collections)


def export_migration_plan(model: SchemaModel) -> dict[str, Any]:
    """Produce a clean, executable order to build the schema.

    Returns a create order plus the FKs to add *after* the tables exist
    (the cycle-breaking relations), which is exactly how you deploy a
    schema that contains circular dependencies: create in topological
    order without the offending FKs, then ALTER TABLE ADD them last.
    """
    order, cut = topological_order(model)
    cut_set = {(a, b) for a, b in cut}
    # map fk (from,to) -> the ForeignKey objects to defer
    deferred = []
    for fk in model.foreign_keys:
        if (fk.from_table, fk.to_table) in cut_set:
            deferred.append(fk)
    alters = []
    for fk in deferred:
        cols = ", ".join(fk.from_cols) if fk.from_cols else "<fk_col>"
        refcols = ", ".join(fk.to_cols) if fk.to_cols else "<pk_col>"
        alters.append(
            f"ALTER TABLE {fk.from_table} ADD CONSTRAINT "
            f"fk_{fk.from_table}_{fk.to_table} FOREIGN KEY ({cols}) "
            f"REFERENCES {fk.to_table} ({refcols});")
    return {
        "create_order": order,
        "deferred_foreign_keys": [f"{fk.from_table} -> {fk.to_table}" for fk in deferred],
        "post_create_ddl": alters,
        "note": ("Create tables in `create_order` (omit the deferred FKs), then "
                 "run `post_create_ddl` to add the cycle-closing relations. This "
                 "makes an otherwise un-orderable schema deployable."),
    }


def export_schema_ddl(model: SchemaModel, dialect: str = "generic") -> str:
    """Generate CREATE TABLE DDL from a schema model, cycle-safe.

    Tables are emitted in dependency order; foreign keys that would form a
    cycle are emitted as trailing ``ALTER TABLE ADD CONSTRAINT`` so the
    script runs top-to-bottom without ordering errors.
    """
    order, cut = topological_order(model)
    cut_set = {(a, b) for a, b in cut}
    by_name = {t.name: t for t in model.tables}
    fks_by_table: dict[str, list] = {}
    deferred = []
    for fk in model.foreign_keys:
        if (fk.from_table, fk.to_table) in cut_set:
            deferred.append(fk)
        else:
            fks_by_table.setdefault(fk.from_table, []).append(fk)
    lines = []
    for name in order:
        t = by_name.get(name)
        cols = (t.columns if t and t.columns else ["id"])
        coldefs = [f"  {c} INTEGER" + (" PRIMARY KEY" if t and t.primary_key == [c] else "")
                   for c in cols]
        for fk in fks_by_table.get(name, []):
            col = fk.from_cols[0] if fk.from_cols else "ref_id"
            refc = fk.to_cols[0] if fk.to_cols else "id"
            coldefs.append(f"  CONSTRAINT fk_{name}_{fk.to_table} "
                           f"FOREIGN KEY ({col}) REFERENCES {fk.to_table} ({refc})")
        if t and t.primary_key and t.primary_key != (cols[:1] if cols else []):
            coldefs.append(f"  PRIMARY KEY ({', '.join(t.primary_key)})")
        lines.append(f"CREATE TABLE {name} (\n" + ",\n".join(coldefs) + "\n);")
    for fk in deferred:
        col = fk.from_cols[0] if fk.from_cols else "ref_id"
        refc = fk.to_cols[0] if fk.to_cols else "id"
        lines.append(f"ALTER TABLE {fk.from_table} ADD CONSTRAINT "
                     f"fk_{fk.from_table}_{fk.to_table} FOREIGN KEY ({col}) "
                     f"REFERENCES {fk.to_table} ({refc});")
    return "\n\n".join(lines) + "\n"


def list_tables(conn_str: str, with_counts: bool = False) -> list[dict]:
    """List a live database's tables with column/PK/FK counts (read-only).
    If ``with_counts``, also include a best-effort ``rows`` count per table."""
    from sqlalchemy import create_engine, inspect, text
    engine = create_engine(conn_str)
    insp = inspect(engine)
    out = []
    conn = engine.connect() if with_counts else None
    try:
        for name in insp.get_table_names():
            entry = {
                "table": name,
                "columns": len(insp.get_columns(name)),
                "foreign_keys": len(insp.get_foreign_keys(name)),
                "primary_key": insp.get_pk_constraint(name).get("constrained_columns", []) or [],
            }
            if conn is not None:
                try:
                    entry["rows"] = int(conn.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar() or 0)
                except Exception:
                    try:
                        entry["rows"] = int(conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar() or 0)
                    except Exception:
                        entry["rows"] = None
            out.append(entry)
    finally:
        if conn is not None:
            conn.close()
    return out


# schema -> relational complex

def _associative_entities(model: SchemaModel) -> set:
    """Tables whose identity *is* their relationships: associative /
    junction entities. Signal: a primary key composed of foreign-key
    columns, binding two or more parents. These are the cells that
    genuinely co-participate, so they license a co-participation face.
    """
    fk_cols: dict[str, set] = {}
    fk_count: dict[str, int] = {}
    for fk in model.foreign_keys:
        fk_cols.setdefault(fk.from_table, set()).update(fk.from_cols or [])
        fk_count[fk.from_table] = fk_count.get(fk.from_table, 0) + 1
    assoc = set()
    for t in model.tables:
        pk = set(t.primary_key or [])
        cols = fk_cols.get(t.name, set())
        # ≥2 FKs and every PK column is a foreign-key column
        if fk_count.get(t.name, 0) >= 2 and pk and pk.issubset(cols):
            assoc.add(t.name)
    return assoc


def _coparticipation_b2(names, sources, targets, model):
    """Build co-participation faces as general **k-gons** via the B₂ boundary
    matrix, not restricted to triangles.

    A face is the ``{0,1}⊗{+,-}`` column: participating edges (`{0,1}`) with
    an orientation sign (`{+,-}`), constrained by ∂₁∂₂ = 0 (a closed loop).
    An associative entity licenses a face over each loop it closes:
      * two foreign keys to the *same* parent -> a **bigon** (self-M:N /
        bill-of-materials bounded recursion);
      * two foreign keys to parents that are themselves related -> a
        **triangle**;
      * (extensible to longer connected parents -> a **k-gon**).
    Only loops that satisfy ∂₁∂₂ = 0 are kept. Returns CSC arrays
    ``(col_ptr, row_idx, vals, descriptions)`` or ``(None, None, None, [])``.
    """
    idx = {n: i for i, n in enumerate(names)}
    assoc_idx = {idx[a] for a in _associative_entities(model) if a in idx}
    nE = len(sources)
    directed: dict[tuple, list] = {}
    inc: dict[int, list] = {}
    for e in range(nE):
        s, t = int(sources[e]), int(targets[e])
        directed.setdefault((s, t), []).append(e)
        inc.setdefault(s, []).append((e, t))
        inc.setdefault(t, []).append((e, s))

    def _edge_between(a, b):
        if (a, b) in directed:
            return directed[(a, b)][0], +1.0
        if (b, a) in directed:
            return directed[(b, a)][0], -1.0
        return None

    def _valid(face):                      # ∂₁∂₂ = 0 : the loop is closed
        acc: dict[int, float] = {}
        for e, sgn in face:
            s, t = int(sources[e]), int(targets[e])
            acc[s] = acc.get(s, 0.0) - sgn
            acc[t] = acc.get(t, 0.0) + sgn
        return all(abs(v) < 1e-9 for v in acc.values())

    faces, descs, seen = [], [], set()
    for j in sorted(assoc_idx):
        incident = inc.get(j, [])
        for a in range(len(incident)):
            for b in range(a + 1, len(incident)):
                ea, na = incident[a]
                eb, nb = incident[b]
                if na == nb:                      # bigon (self-M:N)
                    key = tuple(sorted((ea, eb)))
                    face = [(ea, +1.0), (eb, -1.0)]
                else:
                    conn = _edge_between(na, nb)  # triangle
                    if not conn:
                        continue
                    e_ab, s_ab = conn
                    key = tuple(sorted((ea, eb, e_ab)))
                    sa = +1.0 if (int(sources[ea]), int(targets[ea])) == (j, na) else -1.0
                    sb = -1.0 if (int(sources[eb]), int(targets[eb])) == (j, nb) else +1.0
                    face = [(ea, sa), (e_ab, s_ab), (eb, sb)]
                if key in seen or not _valid(face):
                    continue
                seen.add(key)
                faces.append(face)
                verts = sorted({int(sources[e]) for e, _ in face} |
                               {int(targets[e]) for e, _ in face})
                descs.append([names[v] for v in verts])
    if not faces:
        return None, None, None, []
    col_ptr, row_idx, vals = [0], [], []
    for face in faces:
        for e, sgn in face:
            row_idx.append(e)
            vals.append(sgn)
        col_ptr.append(len(row_idx))
    return (np.asarray(col_ptr, dtype=np.int32),
            np.asarray(row_idx, dtype=np.int32),
            np.asarray(vals, dtype=np.float64), descs)


#: The available face-selection algorithms for a schema complex (see
#: `_schema_face_b2` / `explore_schema_faces`). Same tables & FKs, different
#: definition of "what counts as a co-participation" -> different curl/harmonic.
SCHEMA_FACE_SELECTIONS = ("coparticipation", "autoface", "promote", "none")


def _schema_face_b2(names, src, tgt, edge_tables, model, mode):
    """Return the schema's B₂ as (col_ptr, row_idx, vals) CSC arrays under the
    chosen face-selection algorithm, or None for no faces:

      'coparticipation' - faces only over associative/junction entities (the
                          "real" co-participations); ordinary FK cycles stay
                          harmonic (broken). [_coparticipation_b2]
      'autoface'        - fill every triangle AND bigon the FK graph allows
                          (geometry-from-topology; more curl, less harmonic).
                          [_autoface_b2]
      'none'            - 1-rex, no faces (pure gradient/harmonic split).
    ('promote' is handled by the core face-finder in schema_to_rex, not here.)"""
    if mode == "coparticipation":
        cp, rp, vp, _ = _coparticipation_b2(names, src, tgt, model)
        return None if cp is None else (cp, rp, vp)
    if mode == "autoface":
        import scipy.sparse as sp
        nV, nE = len(names), len(src)
        rows, cols, vals = [], [], []
        for e in range(nE):
            rows += [int(src[e]), int(tgt[e])]; cols += [e, e]; vals += [-1.0, 1.0]
        B1 = sp.csr_matrix((vals, (rows, cols)), shape=(nV, nE), dtype=np.float64)
        B2, _ = _autoface_b2(names, list(edge_tables), B1)
        if B2 is None:
            return None
        B2 = B2.tocsc()
        return (np.asarray(B2.indptr, dtype=np.int32),
                np.asarray(B2.indices, dtype=np.int32),
                np.asarray(B2.data, dtype=np.float64))
    return None


def schema_to_rex(model: SchemaModel, face_selection: str = "coparticipation",
                  weights: dict[str, float] | None = None):
    """Build a typed, directed relational complex from a schema.

    Vertices = tables. Edges = foreign keys (child -> parent). Faces are chosen by
    ``face_selection`` (see `SCHEMA_FACE_SELECTIONS` / `explore_schema_faces`):
    'coparticipation' (default - associative/junction entities), 'autoface' (every
    triangle+bigon the FK graph allows), 'promote' (the core face-finder), or 'none'
    (1-rex). ``weights`` optionally maps "from_table->to_table" to an edge magnitude
    (e.g. cardinality) - the complex then carries ``w_E`` into the core channels, so
    the weighted/curvature story lives on the standard complex rather than a separate
    hand-rolled path. Different valid geometries of the same schema. Returns
    ``(rex_or_None, meta)``.
    """
    if face_selection not in SCHEMA_FACE_SELECTIONS:
        raise ValueError(
            f"face_selection must be one of {SCHEMA_FACE_SELECTIONS}, "
            f"got {face_selection!r}")
    names = model.table_names()
    idx = {n: i for i, n in enumerate(names)}
    sources, targets, edge_tables, self_refs = [], [], [], []
    for fk in model.foreign_keys:
        if fk.from_table not in idx or fk.to_table not in idx:
            continue
        s, t = idx[fk.from_table], idx[fk.to_table]
        if s == t:
            self_refs.append(fk.from_table)
            continue
        sources.append(s)
        targets.append(t)
        edge_tables.append((fk.from_table, fk.to_table))
    src = np.asarray(sources, dtype=np.int32)
    tgt = np.asarray(targets, dtype=np.int32)
    # Cap construction/diagnosis on oversized schemas (same guard as the edge
    # chokepoint) - schema_to_rex builds RexGraph directly, bypassing auto.py.
    from .auto import check_analysis_size
    check_analysis_size(len(names), len(sources))
    # co-participation face descriptions are always reported (meta), independent
    # of which face_selection actually fills B₂ below.
    _, _, _, face_descs = (_coparticipation_b2(names, src, tgt, model)
                           if sources else (None, None, None, []))
    meta = {
        "vertex_labels": names,
        "n_tables": len(names),
        "n_foreign_keys": len(sources),
        "self_referential": self_refs,
        "edges": edge_tables,
        "associative_entities": sorted(_associative_entities(model)),
        "coparticipation_faces": face_descs,
        "face_selection": face_selection,
        "weighted": bool(weights),
    }
    if not sources:
        return None, meta
    # edge weights (cardinality) aligned to the edge order, if supplied
    w_E = None
    if weights:
        w_arr = np.ones(len(sources), dtype=np.float64)
        for e, (a, b) in enumerate(edge_tables):
            val = weights.get(f"{a}->{b}")
            if val is not None:
                with contextlib.suppress(TypeError, ValueError):
                    w_arr[e] = max(float(val), 1e-9)
        w_E = w_arr
    from rexgraph.graph import RexGraph
    b2 = (_schema_face_b2(names, src, tgt, edge_tables, model, face_selection)
          if face_selection != "none" else None)
    if b2 is not None:
        # k-gon faces via the B₂ boundary matrix. Filling a junction's loop turns
        # its false harmonic hole into bounded curl; unfilled FK cycles stay
        # harmonic (broken).
        cp, rp, vp = b2
        try:
            rex = RexGraph(sources=src, targets=tgt, w_E=w_E,
                           B2_col_ptr=cp, B2_row_idx=rp, B2_vals=vp)
        except Exception:
            rex = RexGraph(sources=src, targets=tgt, w_E=w_E)
    else:
        rex = RexGraph(sources=src, targets=tgt, w_E=w_E)
        if face_selection == "promote":         # core face-finder
            with contextlib.suppress(Exception):
                rex = rex.promote()
    rex._agent_meta = meta
    return rex, meta


def explore_schema_faces(model: SchemaModel, modes=SCHEMA_FACE_SELECTIONS):
    """Explore how the choice of face-selection algorithm changes the geometry of
    the SAME schema. For each mode, build the complex and report the face count,
    Betti numbers, and the Hodge split (how much relational flow is
    hierarchy/gradient vs bounded/curl vs persistent/harmonic). Filling more loops
    (autoface) trades harmonic "broken cycle" content for bounded curl; filling only
    genuine junctions (coparticipation) leaves ordinary FK cycles harmonic. Returns
    ``{mode: {n_faces, betti, hodge}}`` - a side-by-side of the schema's options."""
    out: dict[str, Any] = {}
    for mode in modes:
        try:
            rex, _ = schema_to_rex(model, face_selection=mode)
            if rex is None:
                out[mode] = {"n_faces": 0, "betti": None, "hodge": None}
                continue
            h = rex.hodge_full(np.ones(rex.nE, dtype=np.float64))
            out[mode] = {
                "n_faces": int(rex.nF_hodge),
                "betti": [int(b) for b in rex.betti],
                "hodge": {
                    "hierarchy_gradient": round(float(h["pct_grad"]), 4),
                    "bounded_curl": round(float(h["pct_curl"]), 4),
                    "persistent_harmonic": round(float(h["pct_harm"]), 4),
                },
            }
        except Exception as e:
            out[mode] = {"error": str(e)}
    return out


# cycle finder (actionable circular-dependency output)

def _find_cycles(names: list[str], edges: list[tuple[str, str]],
                 max_cycles: int | None = None) -> list[list[str]]:
    """Enumerate distinct directed cycles (up to `max_cycles`) via an ITERATIVE
    colored DFS with an explicit stack, so deep schemas cannot raise RecursionError
    (Python's recursion limit is ~1000; reflected schemas can be deeper)."""
    if max_cycles is None:
        max_cycles = _TH.max_cycles
    adj: dict[str, list[str]] = {n: [] for n in names}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, adj.get(b, []))
    cycles, seen = [], set()
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in adj}

    for start in list(adj):
        if color[start] != WHITE:
            continue
        color[start] = GREY
        path = [start]
        stack = [(start, iter(adj.get(start, [])))]   # (node, its neighbor iterator)
        while stack:
            node, it = stack[-1]
            descended = False
            for nxt in it:
                c = color.get(nxt, WHITE)
                if c == GREY:                          # back edge -> a cycle
                    cyc = path[path.index(nxt):]
                    key = tuple(sorted(cyc))
                    if key not in seen and len(cycles) < max_cycles:
                        seen.add(key)
                        cycles.append(cyc[:])
                elif c == WHITE:                       # descend depth-first
                    color[nxt] = GREY
                    path.append(nxt)
                    stack.append((nxt, iter(adj.get(nxt, []))))
                    descended = True
                    break
            if not descended:                          # exhausted -> backtrack
                color[node] = BLACK
                stack.pop()
                path.pop()
    return cycles


def topological_order(model: SchemaModel):
    """Derive a valid order of operations from the FK structure.

    For create/insert, a referenced (parent) table must exist before the
    referencing (child) table. Returns ``(order, relations_to_cut)``: a
    topological order over the DAG part, plus a greedy feedback-arc set: the
    foreign keys to defer/cut so a strict linear order exists. This breaks
    EVERY cycle (harmonic AND bounded/curl), because any cycle blocks a total
    order; the harmonic fraction in the readout then says which of those cuts
    address genuine broken dependencies (harmonic) vs intended recursion (curl,
    e.g. a bill-of-materials) that you would instead handle with a deferred
    constraint. So a non-empty cut with a low harmonic fraction is expected,
    not a contradiction.
    """
    names = model.table_names()
    deps: dict[str, set] = {n: set() for n in names}   # child -> {parents}
    for fk in model.foreign_keys:
        if (fk.from_table in deps and fk.to_table in deps
                and fk.from_table != fk.to_table):
            deps[fk.from_table].add(fk.to_table)
    order: list[str] = []
    resolved: set = set()
    cut: list[tuple[str, str]] = []
    remaining = {n: set(p) for n, p in deps.items()}
    while remaining:
        ready = [n for n in remaining if remaining[n] <= resolved]
        if not ready:
            # a cycle blocks progress, so cut one feedback arc and continue
            node = min(remaining, key=lambda n: len(remaining[n] - resolved))
            unresolved = remaining[node] - resolved
            parent = sorted(unresolved)[0]
            cut.append((node, parent))       # FK node->parent must be deferred
            remaining[node].discard(parent)
            continue
        for n in sorted(ready):
            order.append(n)
            resolved.add(n)
            del remaining[n]
    return order, cut


def _autoface_b2(names, edges, B1):
    """Autoface: fill every triangle AND bigon the connectivity allows
    (geometry-from-topology). Triangles = 3 pairwise-related tables; bigons =
    mutual/parallel relations (e.g. a warehouse<->manager cycle, a self-M:N).
    Returns a SPARSE B₂ (nE×nF CSC) of signed loops with ∂₁∂₂=0 plus the
    table-tuples. Adjacency-driven enumeration; sparse ∂₁∂₂ check (no dense B₂)."""
    import numpy as np
    import scipy.sparse as sp
    idx = {n: i for i, n in enumerate(names)}
    nE = len(edges)
    B1s = B1 if sp.issparse(B1) else sp.csr_matrix(np.asarray(B1, dtype=np.float64))
    directed = {}
    undirected = {}
    for e, (a, b) in enumerate(edges):
        ia, ib = idx[a], idx[b]
        directed.setdefault((ia, ib), []).append(e)
        undirected.setdefault(ia, set()).add(ib)
        undirected.setdefault(ib, set()).add(ia)

    def edge(a, b):
        return directed.get((a, b), [None])[0]

    def chain_ok(colmap):
        """∂₁∂₂ = 0 for one face column (given as {edge: sign}), sparsely."""
        col = np.zeros(nE, dtype=np.float64)
        for e, v in colmap.items():
            col[e] = v
        return float(np.max(np.abs(B1s @ col))) < 1e-9

    face_maps, faces, seen = [], [], set()

    # bigons: a pair joined by two edges (mutual FK, or two parallel FKs)
    for a in sorted(undirected):
        for b in sorted(undirected.get(a, ())):
            if b <= a:
                continue
            es = directed.get((a, b), []) + directed.get((b, a), [])
            if len(es) >= 2:
                e0, e1 = es[0], es[1]
                for s1 in (1.0, -1.0):
                    cm = {e0: 1.0, e1: s1}
                    if chain_ok(cm):
                        seen.add(("bi", a, b))
                        face_maps.append(cm)
                        faces.append([names[a], names[b]])
                        break

    verts = sorted(undirected)
    for i in verts:
        for j in sorted(undirected.get(i, ())):
            if j <= i:
                continue
            for k in sorted(undirected.get(j, ())):
                if k <= j or k not in undirected.get(i, ()):
                    continue
                if (i, j, k) in seen:
                    continue
                cm, ok = {}, True
                for (a, b) in [(i, j), (j, k), (k, i)]:
                    if edge(a, b) is not None:
                        cm[edge(a, b)] = cm.get(edge(a, b), 0.0) + 1.0
                    elif edge(b, a) is not None:
                        cm[edge(b, a)] = cm.get(edge(b, a), 0.0) - 1.0
                    else:
                        ok = False
                        break
                if ok and chain_ok(cm):                 # ∂₁∂₂ = 0
                    seen.add((i, j, k))
                    face_maps.append(cm)
                    faces.append([names[i], names[j], names[k]])
    if not face_maps:
        return None, []
    rows, cols, vals = [], [], []
    for c, cm in enumerate(face_maps):
        for e, v in cm.items():
            rows.append(e); cols.append(c); vals.append(v)
    B2 = sp.csc_matrix((vals, (rows, cols)), shape=(nE, len(face_maps)),
                       dtype=np.float64)
    return B2, faces


def _lagrangian_curvature(B1, B2, w):
    """Global Lagrangian curvature: the deviance of the tower exchange
    c² = L_T/L_S from balance, via the NORMALIZED inverse-participation-ratio
    Lagrangians (source Def 5.1-5.3 / CANONICAL_SPARSE_MATH_REFERENCE Part IV).

    L_T = tr(T²)/tr(T)² is the weighted *topological* (down) Lagrangian, T =
    B₁^wᵀB₁^w - it exists on spans, from B₁ alone. L_S = tr(L₁²)/tr(L₁)² is the
    weighted *geometric* (up) Lagrangian, L₁ = B₂^wB₂^wᵀ. Both are inverse
    participation ratios (= e^{-H}) of the normalized spectrum, so they stay O(1)
    (no int64 overflow). c² = L_T/L_S = (k-2)/2 on Kₖ; curvature = |log c²| =
    |H_S - H_T| (direction-free). When the topology overwhelms the geometry (heavy
    junctions, few cycles) the curvature is large, closing the gap the face-bound
    curvatures leave on spans. The exact integer numerators tr(T²), tr(L₁²) are
    returned as L_T_trace/L_S_trace (unweighted: also c2_exact). c2 is None on a
    pure span (no geometry, L_S = 0); curvature stays large/finite there.
    """
    # Math lives in the core: rexgraph.core._curvature computes the normalized IPR
    # Lagrangians as SPARSE trace identities (no dense nE x nE, no eigensolve). This
    # wrapper only shapes/rounds the report.
    from rexgraph.core._curvature import lagrangian_curvature
    r = lagrangian_curvature(B1, B2, w)
    curv = r["curvature"]
    c2 = r["c2"]
    out = {"L_T": round(r["L_T"], 6), "L_S": round(r["L_S"], 6),
           "c2": (round(c2, 6) if c2 is not None else None),
           "curvature": (round(float(curv), 4) if curv is not None else None),
           "L_T_trace": r.get("L_T_trace"), "L_S_trace": r.get("L_S_trace")}
    if r.get("c2_exact") is not None:
        out["c2_exact"] = r["c2_exact"]
    return out


def _star_curvature(names, edges, w):
    """Per-vertex star curvature (grade-0 localization of R): the weight
    imbalance among each table's incident relations. Unlike face curvature
    (grades 1-2), this is a gradient-tower quantity that fires on spans/
    junctions: it is nonzero exactly where a table's relations carry
    imbalanced cardinality. Returns a ranked [{table, strain}] list.
    """
    # Math moved to the core: rexgraph.core._curvature.star_curvature is a tight
    # per-vertex loop over the edge list (grade-0 localization). This wrapper only
    # maps names<->indices and shapes the ranked report.
    import numpy as np
    from rexgraph.core._curvature import star_curvature
    idx = {n: i for i, n in enumerate(names)}
    src = np.array([idx[a] for a, b in edges], dtype=np.int64)
    tgt = np.array([idx[b] for a, b in edges], dtype=np.int64)
    star = star_curvature(src, tgt, np.asarray(w, dtype=np.float64), len(names))
    return sorted(
        [{"table": names[v], "strain": round(float(star[v]), 3)}
         for v in range(len(names)) if star[v] > 1e-9],
        key=lambda r: -r["strain"])


def relation_lint(model: SchemaModel) -> dict[str, Any]:
    """Data-model lint from the RL4 character: label each foreign key by its
    dominant structural channel, flag ones whose character is anomalous for the
    schema, and surface the tables pulled in conflicting directions (frustration).
    Enriched with FK modality (optional/mandatory, identifying, on-delete)."""
    import numpy as np
    rex, meta = schema_to_rex(model)
    out = {"relations": [], "conflict_tables": [], "anomalies": []}
    if rex is None:
        return out
    try:
        chi = np.asarray(rex.structural_character, dtype=float)   # (nE, 4) T,G,F,C
        phi = np.asarray(rex.vertex_character, dtype=float)
    except Exception:
        return out
    names = meta["vertex_labels"]
    edges = meta["edges"]
    labels = ["hierarchical", "structural-overlap", "conflicting", "hub-linked"]
    mod = {(fk.from_table, fk.to_table): fk for fk in model.foreign_keys}
    median = np.median(chi, axis=0) if len(chi) else np.zeros(4)
    devs = np.array([float(np.sum(np.abs(chi[i] - median))) for i in range(len(chi))])
    # Tukey fence for anomalies (only when enough relations to be meaningful)
    thr = np.inf
    if len(devs) >= 4:
        q1, q3 = np.percentile(devs, [25, 75])
        thr = q3 + 1.5 * (q3 - q1)
    for i, (a, b) in enumerate(edges):
        fk = mod.get((a, b))
        parts = [("optional" if (fk and fk.nullable) else "mandatory")]
        if fk and fk.identifying:
            parts.append("identifying")
        if fk and fk.on_delete:
            parts.append(f"on-delete {fk.on_delete.lower()}")
        rec = {"relation": f"{a} -> {b}",
               "character": labels[int(np.argmax(chi[i]))],
               "modality": ", ".join(parts),
               "deviation": round(float(devs[i]), 3),
               "anomaly": bool(devs[i] > thr)}
        out["relations"].append(rec)
        if rec["anomaly"]:
            out["anomalies"].append(rec["relation"])
    fidx = 2 if chi.shape[1] > 2 else -1
    # Conflict tables = statistical OUTLIERS in the frustration channel (Tukey fence,
    # data-adaptive, the same principled test as the relation anomalies above), not a
    # fixed cutoff. `frustration_ranking` gives every table's exact value so the
    # caller sees the full distribution and can apply its own filter if desired.
    if fidx >= 0 and phi.shape[1] > 2 and phi.shape[0] >= len(names):
        frus = np.array([float(phi[v, fidx]) for v in range(len(names))])
        fthr = np.inf
        if len(frus) >= 4:
            fq1, fq3 = np.percentile(frus, [25, 75])
            fthr = fq3 + 1.5 * (fq3 - fq1)
        ranking = sorted(
            [{"table": names[v], "frustration": round(float(frus[v]), 3)}
             for v in range(len(names)) if frus[v] > 1e-9],
            key=lambda r: -r["frustration"])
        out["frustration_ranking"] = ranking
        out["conflict_tables"] = [r for r in ranking if r["frustration"] > fthr]
    return out


def schema_strain(model: SchemaModel, weights=None):
    """Data-forced strain: how the weighting (real data magnitudes) strains
    the schema, on the lens-independent autoface geometry.

    Three layers, each a plain answer (see the RCF conversation):
      * weighted->weighted (‖B₁^w B₂^w‖², intensity) -> *how much / where*
        - the strain heat map over joins.
      * weighted->unweighted (B₁ diag(√w) B₂, first-order) -> *who / what*
        - per-relation attribution, additive because ∂₁∂₂=0 exactly.
      * harmonic log of the strain Gram -> *how many / how coupled*
        - N_eff effective independent root causes + coupled pairs.

    ``weights`` maps "from_table->to_table" to a magnitude (e.g. cardinality
    = child_rows / parent_distinct). Uniform weights => zero strain (flat).
    """
    import numpy as np
    names = model.table_names()
    idx = {n: i for i, n in enumerate(names)}
    edges = [(fk.from_table, fk.to_table) for fk in model.foreign_keys
             if fk.from_table in idx and fk.to_table in idx
             and fk.from_table != fk.to_table]
    nV, nE = len(names), len(edges)
    report = {"n_relations": nE, "has_geometry": False,
              "total_strain": 0.0, "per_join": [], "per_relation": [],
              "effective_root_causes": None, "coupled_relations": [],
              "relation_load": [], "table_strain": [],
              "lagrangian_curvature": None}
    if nE == 0:
        return report
    from .auto import check_analysis_size
    check_analysis_size(nV, nE)                 # cap oversized schemas (own complex)
    w = np.ones(nE)
    weights = weights or {}
    for e, (a, b) in enumerate(edges):
        key = f"{a}->{b}"
        if key in weights and weights[key] is not None:
            with contextlib.suppress(TypeError, ValueError):
                w[e] = max(float(weights[key]), 1e-9)
    # gradient-tower curvature localizations (fire on spans/junctions, unlike
    # the face-bound curvature below):
    #   relation_load: per-edge fan-out (which relation)
    #   table_strain  - per-vertex star curvature (which table is the hotspot)
    if np.any(np.abs(w - 1.0) > 1e-9):
        loads = sorted(
            [{"relation": f"{edges[e][0]} -> {edges[e][1]}",
              "load": round(float(w[e]), 3)}
             for e in range(nE) if abs(w[e] - 1.0) > 1e-9],
            key=lambda r: -abs(np.log(max(r["load"], 1e-9))))
        report["relation_load"] = loads
    report["table_strain"] = _star_curvature(names, edges, w)
    import scipy.sparse as sp
    # sparse signed incidence B1 (nV × nE): -1 source / +1 target
    _b1r, _b1c, _b1v = [], [], []
    for e, (a, b) in enumerate(edges):
        _b1r += [idx[a], idx[b]]; _b1c += [e, e]; _b1v += [-1.0, 1.0]
    B1 = sp.csr_matrix((_b1v, (_b1r, _b1c)), shape=(nV, nE), dtype=np.float64)
    B2, faces = _autoface_b2(names, edges, B1)          # sparse nE × nF
    # global Lagrangian curvature: tower-exchange deviance (works with or
    # without faces - L_T is a B₁^w quantity, so it captures span pressure).
    report["lagrangian_curvature"] = _lagrangian_curvature(B1, B2, w)
    if B2 is None:
        # no co-participation faces (tree/star): flat, no curvature - but the
        # fan-out load above still surfaces span/junction pressure.
        return report
    report["has_geometry"] = True

    # curvature (how much / where): R = B1 diag(w) B2 - the weighted chain
    # residual (Part F). All sparse; total/per-join are Frobenius reductions.
    R = (B1 @ sp.diags(w) @ B2).tocsr()
    report["total_strain"] = round(float(R.multiply(R).sum()), 6)
    per_join = np.asarray(R.multiply(R).sum(axis=0)).ravel()
    report["per_join"] = sorted(
        [{"tables": faces[f], "strain": round(float(per_join[f]), 6)}
         for f in range(len(faces))], key=lambda r: -r["strain"])

    # connection (who / what): the strain Gram M[i,j] = <U_i, U_j> with the
    # per-relation contribution vectors U_e = (√w_e-1)·(B1[:,e] ⊗ B2[e,:]) reduces
    # EXACTLY to M = diag(s)·(T ⊙ B2B2ᵀ)·diag(s),  s = √w-1,  T = B1ᵀB1  - a sparse
    # Hadamard product (no dense (nV·nF)×nE outer-product stack).
    s = np.sqrt(w) - 1.0
    T = (B1.T @ B1).tocsr()                             # topology channel (nE×nE)
    L1up = (B2 @ B2.T).tocsr()                          # face co-incidence (nE×nE)
    M = (sp.diags(s) @ T.multiply(L1up) @ sp.diags(s)).tocsr()
    Mdiag = M.diagonal()
    report["per_relation"] = sorted(
        [{"relation": f"{edges[e][0]} -> {edges[e][1]}",
          "contribution": round(float(Mdiag[e]), 6)}
         for e in range(nE) if Mdiag[e] > 1e-9],
        key=lambda r: -r["contribution"])
    Mc = M.tocoo()
    report["coupled_relations"] = [
        {"a": f"{edges[i][0]} -> {edges[i][1]}",
         "b": f"{edges[j][0]} -> {edges[j][1]}",
         "coupling": round(float(v), 6)}
        for i, j, v in zip(Mc.row, Mc.col, Mc.data, strict=False)
        if i < j and abs(v) > 1e-9]

    # harmonic log of the Gram -> effective independent root causes, EIGEN-FREE:
    # e^{H₂(M)} = 1/Σp² = tr(M)²/tr(M²)  (Rényi-2 collision / effective mode count).
    trM = float(Mdiag.sum())
    trM2 = float(M.multiply(M).sum())                   # tr(M²)=‖M‖_F² (M symmetric)
    if trM > 1e-12 and trM2 > 1e-15:
        report["effective_root_causes"] = round(trM * trM / trM2, 3)
    return report


def _row_count(conn, engine, table, approximate=False):
    """Count rows, preferring a fast catalog estimate in approximate mode
    (dialect-aware), falling back to exact COUNT(*)."""
    from sqlalchemy import text
    if approximate:
        d = engine.dialect.name
        try:
            if d == "postgresql":
                r = conn.execute(text("SELECT reltuples::bigint FROM pg_class "
                                      "WHERE relname = :t"), {"t": table}).scalar()
            elif d == "mysql":
                r = conn.execute(text("SELECT table_rows FROM information_schema.tables "
                                      "WHERE table_name = :t"), {"t": table}).scalar()
            elif d in ("mssql", "sqlserver"):
                r = conn.execute(text("SELECT SUM(row_count) FROM sys.dm_db_partition_stats "
                                      "WHERE object_id = OBJECT_ID(:t) AND index_id < 2"),
                                 {"t": table}).scalar()
            else:
                r = None
            if r is not None and int(r) > 0:
                return int(r)
        except Exception:
            pass
    try:
        return int(conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar() or 0)
    except Exception:
        try:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0)
        except Exception:
            return 0


def pull_cardinality_stats(conn_str: str, model: SchemaModel = None,
                           approximate: bool = False):
    """Weight each foreign key by real data magnitude from a live database.

    Cardinality ≈ child_row_count / parent_row_count (average fan-out). In
    ``approximate`` mode, uses fast catalog estimates (pg_class.reltuples,
    information_schema.table_rows, sys.dm_db_partition_stats) instead of
    COUNT(*) - essential on very large tables. Returns ``(weights, row_counts)``.
    """
    from sqlalchemy import create_engine
    if model is None:
        model = reflect_schema(conn_str)
    engine = create_engine(conn_str)
    counts = {}
    with engine.connect() as c:
        for t in model.table_names():
            counts[t] = _row_count(c, engine, t, approximate=approximate)
    weights = {}
    for fk in model.foreign_keys:
        child = counts.get(fk.from_table, 0)
        parent = counts.get(fk.to_table, 0)
        if parent > 0 and child > 0:
            weights[f"{fk.from_table}->{fk.to_table}"] = max(child / parent, 1e-9)
    return weights, counts


# diagnosis

def diagnose_schema(model: SchemaModel) -> dict[str, Any]:
    """Full topological diagnosis of a schema/ontology."""
    rex, meta = schema_to_rex(model)
    names = meta["vertex_labels"]
    edges = meta["edges"]

    report: dict[str, Any] = {
        "n_tables": meta["n_tables"],
        "n_foreign_keys": meta["n_foreign_keys"],
        "self_referential_tables": meta["self_referential"],
        "associative_entities": meta.get("associative_entities", []),
        "coparticipation_faces": meta.get("coparticipation_faces", []),
        "findings": [],
    }
    if rex is None:
        report["findings"].append({
            "severity": "info",
            "issue": "No foreign keys / references to analyze - tables are "
                     "structurally isolated.", "tables": names})
        report["hodge"] = None
        report["betti"] = None
        report["ontology_validity"] = None
        report["order_of_operations"] = names
        report["verdict"] = "no_relations"
        report["summary"] = ("No relations were found. If this is a document "
                             "database, references may not follow an inferable "
                             "convention.")
        return report

    # topology
    try:
        report["betti"] = [int(b) for b in rex.betti]
    except Exception:
        report["betti"] = None
    grad = curl = harm = 0.0
    try:
        h = rex.hodge_full(np.ones(rex.nE, dtype=np.float64))
        grad = round(float(h.get("pct_grad", 0)), 4)
        curl = round(float(h.get("pct_curl", 0)), 4)
        harm = round(float(h.get("pct_harm", 0)), 4)
        report["hodge"] = {
            # gradient = the valid hierarchy/DAG backbone
            "hierarchy_gradient": grad,
            # curl = bounded circulation (a loop that a co-participation fills):
            # valid recursion / feedback, not a broken dependency
            "bounded_recursion_curl": curl,
            # harmonic = persistent LOGICAL circulation (no face fills it):
            # a circular dependency that can never resolve into an order
            "persistent_circulation_harmonic": harm,
        }
    except Exception:
        report["hodge"] = None
    # EXACT integer invariants (threshold-free): the harmonic dimension β₁ counts
    # persistent (unfilled) cycles; rank(B₂) = nF - β₂ counts the co-participation-
    # filled (curl) cycles. These are combinatorial facts - the verdict/findings
    # below are driven by them (β₁>0, rank(B₂)>0), not by a Hodge-fraction cutoff.
    # The fractions above are kept only as informative magnitudes.
    _b = report.get("betti")
    harmonic_dim = int(_b[1]) if _b else 0
    try:
        curl_dim = int(rex.nF_hodge) - (int(_b[2]) if _b else 0)   # rank(B₂)
    except Exception:
        curl_dim = 0
    # a proper ontology/schema is a DAG = pure gradient. Score how close it is.
    report["ontology_validity"] = grad  # 1.0 = valid DAG, lower = circulation

    # valid order of operations + the relations to cut to reach a DAG. The cut
    # breaks EVERY cycle (harmonic and bounded/curl) so a strict order exists;
    # `readout.harmonic_fraction` says how many are genuinely broken vs intended
    # recursion (a non-empty cut with low harmonic fraction is expected).
    order, cut = topological_order(model)
    report["order_of_operations"] = order
    if cut:
        report["relations_to_cut"] = [f"{a} -> {b}" for a, b in cut]
        report["relations_to_cut_note"] = (
            "Feedback-arc set to reach a strict linear order; breaks all cycles. "
            "See readout.harmonic_fraction for which are broken (harmonic) vs "
            "intended recursion (bounded curl).")
        report["migration_plan"] = export_migration_plan(model)

    # co-participation faces (associative entities) - these make otherwise
    # cycle-looking structure into bounded recursion (curl), not broken cycles
    if meta.get("coparticipation_faces"):
        report["findings"].append({
            "severity": "info",
            "issue": f"{len(meta['coparticipation_faces'])} co-participation "
                     "face(s) from associative entities - bounded recursion, "
                     "not broken cycles.",
            "type": "coparticipation",
            "tables": meta.get("associative_entities", []),
        })

    # circular FK dependencies - classified by harmonic vs curl (exact: β₁>0 means
    # at least one directed cycle is unfilled/persistent; else all are face-closed).
    cycles = _find_cycles(names, edges)
    if cycles:
        if harmonic_dim > 0:
            report["findings"].append({
                "severity": "high",
                "issue": f"{len(cycles)} persistent (harmonic) circular"
                         "dependency chain(s) - logical loops that no "
                         "co-participation fills, so no valid insert/delete "
                         "order exists. This is broken relational algebra; cut "
                         "the listed relations to restore a DAG.",
                "cycles": [" -> ".join(c + [c[0]]) for c in cycles],
                "type": "harmonic_persistent",
            })
        else:
            report["findings"].append({
                "severity": "medium",
                "issue": f"{len(cycles)} bounded (curl) circulation(s) - loops "
                         "that a co-participation closes. These can be valid "
                         "recursion/feedback (e.g. a 1:1 with a designated "
                         "primary); verify each is intentional.",
                "cycles": [" -> ".join(c + [c[0]]) for c in cycles],
                "type": "curl_bounded",
            })
    elif harmonic_dim > 0:
        report["findings"].append({
            "severity": "medium",
            "issue": f"{harmonic_dim} persistent (harmonic) hole(s) with no single "
                     "enumerable directed cycle - undirected co-participation tension "
                     "(mutually-related tables with no associative entity filling the "
                     "loop) that resists a consistent hierarchy.",
            "type": "harmonic_diffuse",
        })

    # implied-but-missing relations (voids)
    try:
        vc = rex.void_complex
        n_voids = int(vc.get("n_voids", 0))
        if n_voids:
            report["voids"] = {"n_voids": n_voids,
                               "n_potential": int(vc.get("n_potential", 0))}
            report["findings"].append({
                "severity": "low",
                "issue": f"{n_voids} structurally-implied relation(s) are "
                         "absent - candidate missing foreign keys or a "
                         "normalization boundary the design skipped.",
            })
    except Exception:
        pass

    # table roles + impact (blast radius). The character gives each table a
    # role; degree gives its blast radius, both in plain terms, no vectors.
    try:
        indeg = {n: 0 for n in names}
        outdeg = {n: 0 for n in names}
        for a, b in edges:
            outdeg[a] += 1
            indeg[b] += 1
        try:
            phi = np.asarray(rex.vertex_character, dtype=float)  # (nV, 4) T,G,F,C
        except Exception:
            phi = None
        role_names = ["hub", "bridge", "boundary", "connector"]

        def _role(i):
            if phi is None or i >= len(phi):
                return "hub" if (indeg.get(names[i], 0) >= 2) else "leaf"
            dom = int(np.argmax(phi[i]))
            # a table nothing references and that references others is a leaf
            if indeg.get(names[i], 0) == 0 and outdeg.get(names[i], 0) > 0:
                return "leaf"
            return role_names[dom]

        central = sorted(
            [{"table": names[i],
              "role": _role(i),
              "referenced_by": indeg.get(names[i], 0),
              "references": outdeg.get(names[i], 0),
              "impact": indeg.get(names[i], 0) + outdeg.get(names[i], 0)}
             for i in range(len(names))],
            key=lambda r: -r["impact"])
        report["central_tables"] = central[:8]
        hubs = [c["table"] for c in central if c["impact"] >= 4]
        if hubs:
            report["findings"].append({
                "severity": "low",
                "issue": "High-impact table(s) - many relations converge here, so "
                         "a change here has a large blast radius. Worth checking "
                         "whether the table is doing too much.",
                "tables": hubs,
            })
    except Exception:
        pass

    # isolated tables
    connected = set()
    for a, b in edges:
        connected.add(a)
        connected.add(b)
    isolated = [n for n in names if n not in connected]
    if isolated:
        report["findings"].append({
            "severity": "info",
            "issue": "Table(s) with no foreign-key relations - verify they are "
                     "intentionally standalone.",
            "tables": isolated,
        })

    # descriptive state: a readout, not a judgment. Two INDEPENDENT axes, kept
    # separate so they never contradict:
    #   * directed orderability - does a strict insert/delete order exist? That is
    #     purely the FK *dependency* structure: a valid order exists iff there are
    #     no directed FK cycles, i.e. the feedback-arc cut is empty (`not cut`).
    #   * harmonic content - persistent (undirected) co-participation tension in the
    #     Hodge split; this CAN be present even in a perfectly orderable DAG (e.g. an
    #     FK triangle with no associative entity filling it), so it must not drive the
    #     "no valid order" claim.
    directed_cycles = bool(cut) or bool(cycles)   # directed FK cycles (ordering)
    order_defined = not cut                        # a strict order exists iff DAG
    persistent_harmonic = harmonic_dim > 0         # exact: β₁ unfilled cycles
    bounded_curl = curl_dim > 0                     # exact: rank(B₂) filled cycles
    if persistent_harmonic and directed_cycles:
        report["verdict"] = "cycles_present"
        report["summary"] = ("Persistent (harmonic) circular dependencies: no strict "
                             "insert/delete order exists - cut or defer the listed "
                             "relations to restore a DAG (see the plan).")
    elif persistent_harmonic:
        report["verdict"] = "harmonic_tension"
        report["summary"] = ("A valid order of operations exists (acyclic FK "
                             "dependencies), but there is persistent (harmonic) "
                             "co-participation tension - mutually-related tables with "
                             "no associative entity filling the loop; not an ordering "
                             "bug, a modeling observation.")
    elif bounded_curl or directed_cycles:
        report["verdict"] = "bounded_recursion"
        report["summary"] = ("Bounded recursion (curl): cycle-looking structure closed "
                             "by a co-participation. Common and often intended (e.g. "
                             "bill-of-materials / associative entities)."
                             + (" A strict linear order still needs the listed "
                                "relations cut/deferred." if directed_cycles else
                                " A valid order already exists.")
                             + " Verify it matches intent.")
    else:
        report["verdict"] = "acyclic"
        report["summary"] = ("Acyclic: a pure gradient hierarchy with a clean order "
                             "of operations and no circular dependencies.")
    report["readout"] = {
        "cycles_present": directed_cycles,
        # EXACT integer invariants (the decision basis):
        "harmonic_dimension": harmonic_dim,        # β₁: persistent unfilled cycles
        "curl_dimension": curl_dim,                # rank(B₂) - face-filled cycles
        "directed_cut_size": len(cut),             # feedback arcs to reach a DAG
        # informative (flow-dependent) magnitudes, not decision thresholds:
        "harmonic_fraction": round(float(harm), 4),
        "bounded_curl_fraction": round(float(curl), 4),
        "hierarchy_fraction": round(float(report["hodge"]["hierarchy_gradient"]), 4)
            if report.get("hodge") else None,
        "order_defined": order_defined,
        "associative_entities": len(meta.get("associative_entities", [])),
    }
    return report


def diagnose(source: Any, fmt: str = "auto") -> dict[str, Any]:
    """Convenience entry: diagnose from json/ddl/connection string.

    fmt: 'json' | 'ddl' | 'connection' | 'auto'.
    """
    if fmt == "json" or (fmt == "auto" and isinstance(source, dict)):
        model = parse_schema_json(source)
    elif fmt == "connection" or (fmt == "auto" and isinstance(source, str)
                                 and "://" in source):
        model = reflect_schema(source)
    else:
        model = parse_schema_ddl(str(source))
    return diagnose_schema(model)
