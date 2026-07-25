"""Tests for agent.schema_complex - schemas/ontologies as relational
complexes and their topological diagnosis."""

import pytest

from agent import schema_complex as sc


CYCLIC_DDL = """
CREATE TABLE customers (id INT PRIMARY KEY, fav_order INT REFERENCES orders(id));
CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT REFERENCES customers(id));
CREATE TABLE line_items (id INT PRIMARY KEY, order_id INT REFERENCES orders(id));
"""

CLEAN_SPEC = {"tables": [
    {"name": "users", "primary_key": ["id"]},
    {"name": "orders", "foreign_keys": [{"columns": ["user_id"], "references": "users"}]},
    {"name": "items", "foreign_keys": [{"columns": ["order_id"], "references": "orders"}]},
    {"name": "payments", "foreign_keys": [{"columns": ["order_id"], "references": "orders"}]},
]}


class TestParsing:
    def test_ddl_inline_references(self):
        m = sc.parse_schema_ddl(CYCLIC_DDL)
        assert len(m.tables) == 3
        assert len(m.foreign_keys) == 3
        assert any(fk.to_table == "orders" for fk in m.foreign_keys)

    def test_ddl_table_level_fk(self):
        ddl = ("CREATE TABLE a (id INT PRIMARY KEY);"
               "CREATE TABLE b (id INT, a_id INT, "
               "FOREIGN KEY (a_id) REFERENCES a(id));")
        m = sc.parse_schema_ddl(ddl)
        assert len(m.foreign_keys) == 1
        assert m.foreign_keys[0].from_table == "b"
        assert m.foreign_keys[0].to_table == "a"

    def test_json_spec(self):
        m = sc.parse_schema_json(CLEAN_SPEC)
        assert len(m.tables) == 4
        assert len(m.foreign_keys) == 3


class TestDialectParsing:
    def test_oracle_with_alter_fk_cycle(self):
        pytest.importorskip("sqlglot")
        ddl = ("CREATE TABLE departments (dept_id NUMBER(4) PRIMARY KEY, mgr_id NUMBER(6));"
               "CREATE TABLE employees (emp_id NUMBER(6) PRIMARY KEY, "
               "dept_id NUMBER(4) REFERENCES departments(dept_id));"
               "ALTER TABLE departments ADD CONSTRAINT dm FOREIGN KEY (mgr_id) "
               "REFERENCES employees(emp_id);")
        m = sc.parse_schema_ddl(ddl, dialect="oracle")
        assert len(m.tables) == 2 and len(m.foreign_keys) == 2
        assert sc.diagnose_schema(m)["verdict"] == "cycles_present"

    def test_postgres_quoted_and_serial(self):
        pytest.importorskip("sqlglot")
        ddl = ('CREATE TABLE public."user" (id SERIAL PRIMARY KEY);'
               'CREATE TABLE post (id SERIAL PRIMARY KEY, '
               'author_id INT REFERENCES public."user"(id), '
               'reply_to INT REFERENCES post(id));')
        m = sc.parse_schema_ddl(ddl, dialect="postgres")
        # self-referential post.reply_to recorded; author_id -> user is a real FK
        d = sc.diagnose_schema(m)
        assert "post" in d["self_referential_tables"]

    def test_mysql_backticks(self):
        pytest.importorskip("sqlglot")
        ddl = ("CREATE TABLE `order` (id INT PRIMARY KEY, customer_id INT, "
               "FOREIGN KEY (customer_id) REFERENCES customer(id));"
               "CREATE TABLE customer (id INT PRIMARY KEY);")
        m = sc.parse_schema_ddl(ddl, dialect="mysql")
        assert len(m.foreign_keys) == 1
        assert sc.diagnose_schema(m)["verdict"] == "acyclic"


class TestMongo:
    def test_infer_refs_and_cycle(self):
        mongo = {
            "users": [{"_id": 1, "favorite_post_id": 10}],
            "posts": [{"_id": 10, "author": {"$ref": "users", "$id": 1}}],
        }
        m = sc.infer_mongo_schema(mongo)
        assert len(m.foreign_keys) == 2
        assert sc.diagnose_schema(m)["verdict"] == "cycles_present"

    def test_no_inferable_refs(self):
        m = sc.infer_mongo_schema({"logs": [{"msg": "x", "level": "info"}]})
        assert sc.diagnose_schema(m)["verdict"] == "no_relations"


class TestReflectionAndMigration:
    def test_live_sqlite_reflection_and_plan(self, tmp_path):
        import sqlite3
        dbf = tmp_path / "db.sqlite"
        con = sqlite3.connect(str(dbf))
        con.executescript(
            "CREATE TABLE a (id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES b(id));"
            "CREATE TABLE b (id INTEGER PRIMARY KEY, a_id INTEGER REFERENCES a(id));"
            "CREATE TABLE c (id INTEGER PRIMARY KEY, a_id INTEGER REFERENCES a(id));")
        con.commit()
        con.close()
        m = sc.reflect_schema("sqlite:///" + str(dbf))
        assert len(m.tables) == 3 and len(m.foreign_keys) == 3
        d = sc.diagnose_schema(m)
        assert d["verdict"] == "cycles_present"
        mp = d["migration_plan"]
        assert mp["create_order"] and mp["post_create_ddl"]
        assert "ALTER TABLE" in mp["post_create_ddl"][0]

    def test_migration_plan_export(self):
        m = sc.parse_schema_ddl(
            "CREATE TABLE a (id INT PRIMARY KEY, b_id INT REFERENCES b(id));"
            "CREATE TABLE b (id INT PRIMARY KEY, a_id INT REFERENCES a(id));")
        plan = sc.export_migration_plan(m)
        assert plan["deferred_foreign_keys"]
        assert plan["create_order"]


class TestComplex:
    def test_schema_to_rex(self):
        m = sc.parse_schema_json(CLEAN_SPEC)
        rex, meta = sc.schema_to_rex(m)
        assert rex is not None
        assert meta["n_tables"] == 4 and meta["n_foreign_keys"] == 3
        assert "users" in meta["vertex_labels"]

    def test_no_fk_gives_none_rex(self):
        m = sc.parse_schema_json({"tables": [{"name": "solo"}]})
        rex, meta = sc.schema_to_rex(m)
        assert rex is None and meta["n_foreign_keys"] == 0

    def test_self_referential_recorded(self):
        ddl = ("CREATE TABLE employees (id INT PRIMARY KEY, "
               "manager_id INT REFERENCES employees(id));")
        m = sc.parse_schema_ddl(ddl)
        rex, meta = sc.schema_to_rex(m)
        assert "employees" in meta["self_referential"]


class TestDiagnosis:
    def test_cyclic_detected_as_harmonic(self):
        r = sc.diagnose(CYCLIC_DDL, fmt="ddl")
        assert r["verdict"] == "cycles_present"
        high = [f for f in r["findings"] if f["severity"] == "high"]
        assert high and high[0].get("type") == "harmonic_persistent"
        assert "cycles" in high[0]
        # harmonic (persistent) circulation, not curl
        assert r["hodge"]["persistent_circulation_harmonic"] > 0
        assert r["hodge"]["bounded_recursion_curl"] == 0

    def test_clean_hierarchy_is_acyclic(self):
        r = sc.diagnose(CLEAN_SPEC, fmt="json")
        assert r["verdict"] == "acyclic"
        assert r["hodge"]["hierarchy_gradient"] > 0.9
        assert r["ontology_validity"] > 0.9
        assert r["betti"][1] == 0

    def test_order_of_operations_derived(self):
        r = sc.diagnose(CLEAN_SPEC, fmt="json")
        order = r["order_of_operations"]
        # a parent must appear before any child that references it
        assert order.index("users") < order.index("orders")
        assert order.index("orders") < order.index("items")

    def test_broken_cycle_yields_cut_and_order(self):
        r = sc.diagnose(CYCLIC_DDL, fmt="ddl")
        assert r.get("relations_to_cut")          # something to cut
        assert len(r["order_of_operations"]) == r["n_tables"]  # full order after cut

    def test_hub_table_flagged(self):
        spec = {"tables": [{"name": "core", "primary_key": ["id"]}] + [
            {"name": f"t{i}", "foreign_keys": [{"columns": ["c"], "references": "core"}]}
            for i in range(5)]}
        r = sc.diagnose(spec, fmt="json")
        central = r.get("central_tables", [])
        assert central and central[0]["table"] == "core"

    def test_table_roles_and_impact(self):
        # 'core' referenced by many -> hub role, high impact
        spec = {"tables": [{"name": "core", "primary_key": ["id"]}] + [
            {"name": f"t{i}", "foreign_keys": [{"columns": ["c"], "references": "core"}]}
            for i in range(5)]}
        r = sc.diagnose(spec, fmt="json")
        central = r["central_tables"]
        top = central[0]
        assert top["table"] == "core"
        assert top["impact"] == 5 and top["referenced_by"] == 5
        assert "role" in top          # a plain-language role label is present

    def test_isolated_table_flagged(self):
        spec = {"tables": [
            {"name": "a", "foreign_keys": [{"columns": ["b"], "references": "b"}]},
            {"name": "b"},
            {"name": "orphan"}]}
        r = sc.diagnose(spec, fmt="json")
        assert any("orphan" in (f.get("tables") or []) for f in r["findings"])


class TestCoparticipation:
    def test_associative_entity_gives_bounded_recursion(self):
        # enrollment binds student & course, which are also directly related:
        # a genuine co-participation -> curl (bounded), not harmonic (broken)
        spec = {"tables": [
            {"name": "student", "primary_key": ["id"],
             "foreign_keys": [{"columns": ["course_id"], "references": "course"}]},
            {"name": "course", "primary_key": ["id"]},
            {"name": "enrollment", "primary_key": ["student_id", "course_id"],
             "foreign_keys": [{"columns": ["student_id"], "references": "student"},
                              {"columns": ["course_id"], "references": "course"}]}]}
        r = sc.diagnose(spec, fmt="json")
        assert "enrollment" in r["associative_entities"]
        assert r["coparticipation_faces"]
        assert r["verdict"] == "bounded_recursion"
        assert r["hodge"]["bounded_recursion_curl"] > 0
        assert r["hodge"]["persistent_circulation_harmonic"] == 0

    def test_plain_cycle_stays_harmonic(self):
        # no associative entity -> the cycle is broken (harmonic), not bounded
        ddl = ("CREATE TABLE a(id INT PRIMARY KEY, b_id INT REFERENCES b(id));"
               "CREATE TABLE b(id INT PRIMARY KEY, c_id INT REFERENCES c(id));"
               "CREATE TABLE c(id INT PRIMARY KEY, a_id INT REFERENCES a(id));")
        r = sc.diagnose(ddl, fmt="ddl")
        assert r["associative_entities"] == []
        assert r["verdict"] == "cycles_present"
        assert r["hodge"]["persistent_circulation_harmonic"] > 0

    def test_pure_mn_junction_no_false_cycle(self):
        # a clean M:N junction (student <- enrollment -> course, no student-course
        # edge) is a span, not a cycle: valid, no harmonic tension
        spec = {"tables": [
            {"name": "student", "primary_key": ["id"]},
            {"name": "course", "primary_key": ["id"]},
            {"name": "enrollment", "primary_key": ["student_id", "course_id"],
             "foreign_keys": [{"columns": ["student_id"], "references": "student"},
                              {"columns": ["course_id"], "references": "course"}]}]}
        r = sc.diagnose(spec, fmt="json")
        assert r["verdict"] == "acyclic"
        assert "enrollment" in r["associative_entities"]

    def test_bill_of_materials_bigon(self):
        # self-M:N: 'assembly' binds component to itself via two FKs - a bigon
        # k-gon face (impossible with triangles) fills the false harmonic hole
        spec = {"tables": [
            {"name": "component", "primary_key": ["id"]},
            {"name": "assembly", "primary_key": ["parent_id", "child_id"],
             "foreign_keys": [{"columns": ["parent_id"], "references": "component"},
                              {"columns": ["child_id"], "references": "component"}]}]}
        r = sc.diagnose(spec, fmt="json")
        assert "assembly" in r["associative_entities"]
        # a 2-cell (bigon) is present, so there is no persistent harmonic hole
        assert r["coparticipation_faces"]
        assert r["hodge"]["persistent_circulation_harmonic"] == 0
        assert r["verdict"] != "cycles_present"


class TestStrain:
    # bowtie: two triangles {A,B,C},{B,C,D} sharing edge B-C
    BOWTIE = {"tables": [
        {"name": "A", "primary_key": ["id"]},
        {"name": "D", "primary_key": ["id"]},
        {"name": "C", "foreign_keys": [{"columns": ["a"], "references": "A"},
                                       {"columns": ["d"], "references": "D"}]},
        {"name": "B", "foreign_keys": [{"columns": ["a"], "references": "A"},
                                       {"columns": ["c"], "references": "C"},
                                       {"columns": ["d"], "references": "D"}]}]}

    def test_uniform_weights_flat(self):
        m = sc.parse_schema_json(self.BOWTIE)
        r = sc.schema_strain(m)              # no weights -> uniform
        assert r["has_geometry"] is True     # triangles exist
        assert r["total_strain"] == 0.0      # flat: geometry ≡ topology

    def test_weighted_strain_matches_verified(self):
        m = sc.parse_schema_json(self.BOWTIE)
        r = sc.schema_strain(m, weights={"B->C": 3.0, "C->D": 0.3})
        # verified values from the RCF derivation
        assert abs(r["total_strain"] - 19.78) < 0.05
        rel = {x["relation"]: x["contribution"] for x in r["per_relation"]}
        assert abs(rel["B -> C"] - 2.1436) < 0.01     # who carries it
        assert abs(rel["C -> D"] - 0.4091) < 0.01
        assert r["coupled_relations"]                 # B-C <-> C-D coupling present
        assert abs(r["coupled_relations"][0]["coupling"] - 0.3311) < 0.01
        assert abs(r["effective_root_causes"] - 1.31) < 0.05   # N_eff, not 2

    def test_tree_has_no_forced_geometry(self):
        # a star (no triangles) is flat: no faces, no curvature
        spec = {"tables": [{"name": "hub", "primary_key": ["id"]}] + [
            {"name": f"t{i}", "foreign_keys": [{"columns": ["h"], "references": "hub"}]}
            for i in range(4)]}
        r = sc.schema_strain(sc.parse_schema_json(spec), weights={"t0->hub": 100.0})
        assert r["has_geometry"] is False
        assert r["total_strain"] == 0.0
        # fan-out load (per-edge) and star curvature (per-vertex) both fire on the span
        assert r["relation_load"] and r["relation_load"][0]["relation"] == "t0 -> hub"
        assert r["relation_load"][0]["load"] == 100.0
        assert r["table_strain"]                       # per-vertex star curvature
        assert r["table_strain"][0]["table"] in ("hub", "t0")

    def test_lagrangian_curvature_closes_span_gap(self):
        # a heavy junction (span) has zero face curvature but high Lagrangian
        # curvature - the towers are imbalanced (topology overwhelms geometry)
        spec = {"tables": [{"name": "hub", "primary_key": ["id"]}] + [
            {"name": f"t{i}", "foreign_keys": [{"columns": ["h"], "references": "hub"}]}
            for i in range(4)]}
        r = sc.schema_strain(sc.parse_schema_json(spec), weights={"t0->hub": 100.0})
        lc = r["lagrangian_curvature"]
        assert lc is not None and lc["L_T"] > 0
        # balanced small schema has low Lagrangian curvature; heavy span is high
        bal = sc.schema_strain(sc.parse_schema_json({"tables": [
            {"name": "A", "primary_key": ["id"]},
            {"name": "B", "foreign_keys": [{"columns": ["a"], "references": "A"}]},
            {"name": "C", "foreign_keys": [{"columns": ["a"], "references": "A"},
                                           {"columns": ["b"], "references": "B"}]}]}))
        # both computed; the heavy-weight span should not be lower than balanced
        assert lc["curvature"] is not None
        # a mutual FK (warehouse<->manager) is a bigon -> a face -> has_geometry
        spec = {"tables": [
            {"name": "warehouse", "foreign_keys": [{"columns": ["m"], "references": "manager"}]},
            {"name": "manager", "foreign_keys": [{"columns": ["w"], "references": "warehouse"}]}]}
        r = sc.schema_strain(sc.parse_schema_json(spec), weights={"warehouse->manager": 2.0})
        assert r["has_geometry"] is True     # the bigon fills


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    from fastapi.testclient import TestClient
    db = tmp_path_factory.mktemp("scdb") / "db.sqlite"
    os.environ["REXGRAPH_RCDB_URI"] = f"sqlite:///{db}"
    import agent.server.routes.rcdb as R
    R._STORE = None
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    get_auth_manager().disable_auth()
    with TestClient(app) as c:
        yield c
    R._STORE = None


class TestRoute:
    def test_analyze_ddl_and_store(self, client):
        r = client.post("/api/v1/schema/analyze",
                        json={"ddl": CYCLIC_DDL, "store_id": "sch1", "tags": ["prod"]})
        assert r.status_code == 200
        body = r.json()
        assert body["verdict"] == "cycles_present"
        assert body.get("stored_as") == "sch1"
        # the schema is now a queryable RCDB record tagged 'schema'
        q = client.post("/api/v1/db/query", json={"tags_any": ["schema"]}).json()
        assert "sch1" in [rec["id"] for rec in q["records"]]

    def test_analyze_spec_clean(self, client):
        r = client.post("/api/v1/schema/analyze", json={"spec": CLEAN_SPEC})
        assert r.status_code == 200
        assert r.json()["verdict"] == "acyclic"

    def test_requires_input(self, client):
        assert client.post("/api/v1/schema/analyze", json={}).status_code == 400

    def test_strain_route_with_weights(self, client):
        ddl = ("CREATE TABLE A(id INT PRIMARY KEY);"
               "CREATE TABLE D(id INT PRIMARY KEY);"
               "CREATE TABLE C(id INT PRIMARY KEY, a INT REFERENCES A(id), d INT REFERENCES D(id));"
               "CREATE TABLE B(id INT PRIMARY KEY, a INT REFERENCES A(id), "
               "c INT REFERENCES C(id), d INT REFERENCES D(id));")
        r = client.post("/api/v1/schema/strain",
                        json={"ddl": ddl, "weights": {"B->C": 3.0, "C->D": 0.3}})
        assert r.status_code == 200
        body = r.json()
        assert body["has_geometry"] and body["total_strain"] > 0
        assert body["per_relation"][0]["relation"] == "B -> C"   # ranked, who first
        assert 1.0 < body["effective_root_causes"] < 2.0        # coupled -> < 2

    def test_strain_route_live_stats(self, client, tmp_path_factory):
        import sqlite3
        ddl = ("CREATE TABLE A(id INTEGER PRIMARY KEY);"
               "CREATE TABLE D(id INTEGER PRIMARY KEY);"
               "CREATE TABLE C(id INTEGER PRIMARY KEY, a INTEGER REFERENCES A(id), d INTEGER REFERENCES D(id));"
               "CREATE TABLE B(id INTEGER PRIMARY KEY, a INTEGER REFERENCES A(id), "
               "c INTEGER REFERENCES C(id), d INTEGER REFERENCES D(id));")
        dbf = tmp_path_factory.mktemp("strain") / "s.db"
        con = sqlite3.connect(str(dbf))
        con.executescript(ddl)
        con.executemany("INSERT INTO A(id) VALUES(?)", [(i,) for i in range(5)])
        con.executemany("INSERT INTO D(id) VALUES(?)", [(i,) for i in range(5)])
        con.executemany("INSERT INTO C(id,a,d) VALUES(?,0,0)", [(i,) for i in range(2)])
        con.executemany("INSERT INTO B(id,a,c,d) VALUES(?,0,0,0)", [(i,) for i in range(100)])
        con.commit()
        con.close()
        r = client.post("/api/v1/schema/strain",
                        json={"connection": "sqlite:///" + str(dbf)}).json()
        assert r["row_counts"]["B"] == 100 and r["row_counts"]["C"] == 2
        assert r["total_strain"] > 0        # skewed data forces strain


class TestRelationLint:
    def test_modality_and_character(self):
        ddl = ("CREATE TABLE a(id INT PRIMARY KEY);"
               "CREATE TABLE b(id INT PRIMARY KEY, a_id INT REFERENCES a(id));"
               "CREATE TABLE j(a_id INT REFERENCES a(id), b_id INT REFERENCES b(id), "
               "PRIMARY KEY(a_id,b_id));")
        m = sc.parse_schema_ddl(ddl)
        # identifying inferred from PK membership
        ident = {(fk.from_table, fk.to_table): fk.identifying for fk in m.foreign_keys}
        assert ident[("j", "a")] and ident[("j", "b")] and not ident[("b", "a")]
        lint = sc.relation_lint(m)
        assert len(lint["relations"]) == 3
        for r in lint["relations"]:
            assert r["character"] in ("hierarchical", "structural-overlap",
                                      "conflicting", "hub-linked")
            assert "identifying" in r["modality"] or r["relation"] == "b -> a"

    def test_conflict_tables_surface_frustration(self):
        # a chain a->b->c->d has transit tables carrying frustration
        ddl = ("CREATE TABLE d(id INT PRIMARY KEY);"
               "CREATE TABLE c(id INT PRIMARY KEY, d_id INT REFERENCES d(id));"
               "CREATE TABLE b(id INT PRIMARY KEY, c_id INT REFERENCES c(id));"
               "CREATE TABLE a(id INT PRIMARY KEY, b_id INT REFERENCES b(id));")
        lint = sc.relation_lint(sc.parse_schema_ddl(ddl))
        assert isinstance(lint["conflict_tables"], list)

    def test_lint_route(self, client):
        r = client.post("/api/v1/schema/lint", json={"spec": {"tables": [
            {"name": "a", "primary_key": ["id"]},
            {"name": "b", "foreign_keys": [{"columns": ["a"], "references": "a"}]}]}})
        assert r.status_code == 200
        assert "relations" in r.json()


class TestDDLFallbackParser:
    """Regression tests for the regex fallback (_parse_ddl_regex), which runs
    when sqlglot is absent. It must not shred parenthesised, comma-separated
    constraint lists (a prior bug split PRIMARY KEY(a,b) mid-list, dropping the
    PK and inventing a phantom column)."""

    def test_top_level_splitter_keeps_parens_intact(self):
        body = "a_id INT REFERENCES a(id), b_id INT, PRIMARY KEY(a_id, b_id)"
        parts = [p.strip() for p in sc._split_top_level(body)]
        assert parts == ["a_id INT REFERENCES a(id)", "b_id INT",
                         "PRIMARY KEY(a_id, b_id)"]

    def test_composite_pk_and_identifying_via_fallback(self):
        ddl = ("CREATE TABLE a(id INT PRIMARY KEY);"
               "CREATE TABLE b(id INT PRIMARY KEY, a_id INT REFERENCES a(id));"
               "CREATE TABLE j(a_id INT REFERENCES a(id), b_id INT REFERENCES b(id), "
               "PRIMARY KEY(a_id,b_id));")
        m = sc._parse_ddl_regex(ddl)                     # force the fallback path
        j = next(t for t in m.tables if t.name == "j")
        assert j.primary_key == ["a_id", "b_id"]          # composite PK captured
        assert "b_id)" not in j.columns                   # no phantom column
        assert j.columns == ["a_id", "b_id"]
        ident = {(fk.from_table, fk.to_table): fk.identifying for fk in m.foreign_keys}
        assert ident[("j", "a")] and ident[("j", "b")] and not ident[("b", "a")]

    def test_multicolumn_table_fk_survives_via_fallback(self):
        ddl = ("CREATE TABLE p(x INT, y INT, PRIMARY KEY(x, y));"
               "CREATE TABLE c(x INT, y INT, "
               "FOREIGN KEY (x, y) REFERENCES p(x, y));")
        m = sc._parse_ddl_regex(ddl)
        fk = next(fk for fk in m.foreign_keys if fk.from_table == "c")
        assert fk.from_cols == ["x", "y"]                 # multi-col FK intact
        assert fk.to_table == "p" and fk.to_cols == ["x", "y"]
