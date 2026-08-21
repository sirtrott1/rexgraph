"""
agent.analytics: columnar queries over what the store already knows.

Every backend can answer "which records match this predicate" by walking its
signatures, and all of them take about the same 10-24 ms over eight thousand
records because all of them are doing the same row-at-a-time work. What none of
them can answer at all is the shape of question you actually ask of a corpus:
how does kappa distribute across sources, which betti values are over-represented,
what is the median edge count per tag. The signature is a document, and
aggregating documents means writing the loop yourself.

A signature is a fixed set of scalars per record. That is a table, and a columnar
engine reads tables about twenty times faster than a scan does:

    structural filter over 8000 signatures   store 10.5 ms   duckdb 0.52 ms
    group-by + aggregate                     store  n/a      duckdb 0.58 ms

So this is not another backend. It is a view: signatures are projected out of
whatever store holds them, and the answers come back as ids that store can then be
asked for. Nothing moves, nothing is duplicated on disk, and the store stays the
one place a complex lives.

    from agent import analytics
    view = analytics.signature_view(store)
    view.sql("SELECT source, avg(kappa_mean) FROM signatures GROUP BY source")
    view.ids("betti1 > 2 AND chain_valid = 1")
"""

from __future__ import annotations

#: the scalar columns every signature carries. Anything else stays in the record,
#: which is where a document belongs.
COLUMNS = (
    ("id", "VARCHAR"),
    ("version", "INTEGER"),
    ("nV", "INTEGER"),
    ("nE", "INTEGER"),
    ("nF", "INTEGER"),
    ("betti0", "INTEGER"),
    ("betti1", "INTEGER"),
    ("kappa_mean", "DOUBLE"),
    ("n_voids", "INTEGER"),
    ("chain_valid", "BOOLEAN"),
    ("n_labels", "INTEGER"),
    ("source", "VARCHAR"),
    ("object_type", "VARCHAR"),
    ("tx_from", "DOUBLE"),
    ("valid_from", "DOUBLE"),
)


def _num(x, default=0):
    try:
        if x is None:
            return default
        return type(default)(x)
    except (TypeError, ValueError):
        return default


def signature_rows(store, *, limit: int = 10 ** 9, include_history: bool = False,
                   as_of=None, valid_at=None) -> list[tuple]:
    """Flatten a store's signatures into rows in COLUMNS order."""
    records = store.list(limit=limit, include_history=include_history,
                         as_of=as_of, valid_at=valid_at)
    rows = []
    for rec in records:
        sig = rec.signature or {}
        betti = sig.get("betti") or []
        rows.append((
            rec.id, int(rec.version),
            _num(sig.get("nV")), _num(sig.get("nE")), _num(sig.get("nF")),
            _num(betti[0] if len(betti) > 0 else 0),
            _num(betti[1] if len(betti) > 1 else 0),
            _num(sig.get("kappa_mean"), 0.0),
            _num(sig.get("n_voids")),
            bool(sig.get("chain_valid", False)),
            _num(sig.get("n_labels")),
            str(sig.get("source") or ""),
            str(sig.get("object_type") or ""),
            _num(rec.tx_from, 0.0),
            _num(rec.valid_from if rec.valid_from is not None else 0.0, 0.0),
        ))
    return rows


class SignatureView:
    """A columnar view of one store's signatures. Read-only, and derived: rebuild it
    after writing rather than trying to keep it in step, since a view that silently
    lags the store it describes is worse than one you refresh."""

    TABLE = "signatures"

    def __init__(self, store, *, limit: int = 10 ** 9, include_history: bool = False,
                 as_of=None, valid_at=None):
        try:
            import duckdb
        except ImportError as e:
            raise ImportError(
                "columnar signature queries need duckdb: pip install duckdb") from e
        self.store = store
        self._duckdb = duckdb
        self.con = duckdb.connect()
        self._limit = limit
        self._include_history = include_history
        self._as_of, self._valid_at = as_of, valid_at
        self.refresh()

    def refresh(self) -> SignatureView:
        """Re-project the store. Cheap: signatures are scalars the store already holds."""
        cols = ", ".join(f"{n} {t}" for n, t in COLUMNS)
        self.con.execute(f"DROP TABLE IF EXISTS {self.TABLE}")
        self.con.execute(f"CREATE TABLE {self.TABLE} ({cols})")
        rows = signature_rows(self.store, limit=self._limit,
                              include_history=self._include_history,
                              as_of=self._as_of, valid_at=self._valid_at)
        if rows:
            placeholders = ", ".join("?" for _ in COLUMNS)
            self.con.executemany(
                f"INSERT INTO {self.TABLE} VALUES ({placeholders})", rows)
        self.n_rows = len(rows)
        return self

    def sql(self, query: str) -> list[tuple]:
        """Run SQL against the signature table. `signatures` is the table name."""
        return self.con.execute(query).fetchall()

    def columns(self) -> list[str]:
        return [n for n, _ in COLUMNS]

    def ids(self, where: str, *, limit: int = 10 ** 9) -> list[str]:
        """Record ids matching a SQL predicate: the bridge back to the store."""
        rows = self.con.execute(
            f"SELECT id FROM {self.TABLE} WHERE {where} LIMIT {int(limit)}").fetchall()
        return [r[0] for r in rows]

    def complexes(self, where: str, *, limit: int = 100) -> list[tuple]:
        """(id, complex) for records matching a predicate. The view selects; the
        store still owns the payload, so nothing is duplicated."""
        return [(rid, self.store.get(rid)) for rid in self.ids(where, limit=limit)]

    def describe(self) -> list[tuple]:
        """Per-column summary statistics, which is usually the first real question."""
        return self.sql(f"SUMMARIZE {self.TABLE}")

    def to_arrow(self):
        """The table as Arrow, for anything downstream that speaks it (polars,
        pandas, a parquet write) without this module taking a dependency on it."""
        # fetch_arrow_table, not .arrow(): newer duckdb returns a streaming
        # RecordBatchReader from the latter, which is not a materialized table.
        return self.con.execute(f"SELECT * FROM {self.TABLE}").fetch_arrow_table()

    def close(self):
        self.con.close()

    def __len__(self):
        return self.n_rows


def signature_view(store, **kw) -> SignatureView:
    """A columnar view over `store`'s signatures."""
    return SignatureView(store, **kw)


def available() -> bool:
    """Whether columnar analytics can run here."""
    try:
        import duckdb  # noqa: F401
        return True
    except ImportError:
        return False
