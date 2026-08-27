"""One entry point that looks at the host, looks at the data, and says what to do.

The defaults were all present and all scattered: hardware detection knew the
allocation, the store registry knew which backends existed, auto_rex knew which
files it could read, and nothing added them up for a particular machine and a
particular directory.

Planning reads only. Nothing is installed and nothing is written until asked,
which matters most on the machine where you are only looking.
"""

import os

import pytest

from agent import quickstart


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "a.txt").write_text(
        "The boundary map sends an edge to its endpoints. Composing two boundary "
        "maps gives zero. Orientation decides the sign of every entry.")
    (tmp_path / "b.txt").write_text(
        "Each typed channel weighs the edges differently. The overlap channel "
        "counts shared endpoints. Frustration tracks sign conflict.")
    (tmp_path / "c.bed").write_text("chr1\t100\t500\ta\nchr1\t400\t900\tb\n")
    (tmp_path / "d.unknownext").write_text("not something any adapter reads")
    return tmp_path


# planning

def test_a_plan_reads_and_changes_nothing(corpus_dir):
    before = sorted(p.name for p in corpus_dir.iterdir())
    quickstart.plan(str(corpus_dir))
    assert sorted(p.name for p in corpus_dir.iterdir()) == before


def test_a_plan_counts_what_is_actually_there(corpus_dir):
    p = quickstart.plan(str(corpus_dir))
    assert p.n_files == 4
    assert p.files[".txt"] == 2
    assert p.files[".bed"] == 1
    assert p.total_bytes > 0


def test_unreadable_types_are_named_not_hidden(corpus_dir):
    """Skipping a file quietly is how someone discovers three hours in that half
    their corpus was never ingested."""
    p = quickstart.plan(str(corpus_dir))
    assert ".unknownext" in p.unreadable
    assert p.readable == 3


def test_the_plan_reports_the_host_it_measured(corpus_dir):
    p = quickstart.plan(str(corpus_dir))
    assert p.host["cpus"] >= 1
    assert "summary" in p.host and p.host["summary"]
    assert "cpu" in p.summary().lower()


def test_a_backend_is_chosen_with_a_reason(corpus_dir):
    p = quickstart.plan(str(corpus_dir))
    assert p.backend
    assert p.backend_reason
    assert "://" in p.store_uri


def test_an_explicit_store_uri_is_respected(corpus_dir):
    p = quickstart.plan(str(corpus_dir), store="sqlite:///tmp/x.db")
    assert p.store_uri == "sqlite:///tmp/x.db"


def test_depth_follows_the_amount_of_work(corpus_dir):
    p = quickstart.plan(str(corpus_dir))
    assert p.depth in ("quick", "standard")
    assert p.depth_reason


def test_a_large_corpus_is_planned_shallow(corpus_dir, monkeypatch):
    """The signature is ~94% of a put, so file count is what decides affordability."""
    monkeypatch.setattr(quickstart, "_scan", lambda path, **kw: {
        "counts": {".txt": 50_000}, "n_files": 50_000, "total_bytes": 10 ** 10,
        "readable": 50_000, "unreadable": []})
    p = quickstart.plan(str(corpus_dir))
    assert p.depth == "quick"


def test_the_summary_is_readable_and_complete(corpus_dir):
    text = quickstart.plan(str(corpus_dir)).summary()
    for expected in ("host", "data", "store", "depth"):
        assert expected in text


def test_missing_extras_come_with_the_command_to_get_them(corpus_dir, monkeypatch):
    monkeypatch.setattr(quickstart, "_have", lambda m: False)
    p = quickstart.plan(str(corpus_dir))
    assert p.missing
    cmd = p.install_command()
    assert cmd and "pip install" in cmd
    assert "duckdb" in cmd


def test_a_cloud_store_asks_for_that_provider_s_driver(corpus_dir, monkeypatch):
    monkeypatch.setattr(quickstart, "_have", lambda m: m != "s3fs")
    p = quickstart.plan(str(corpus_dir), store="s3://bucket/prefix")
    assert "s3fs" in p.missing


def test_nothing_is_missing_when_everything_is_present(corpus_dir, monkeypatch):
    monkeypatch.setattr(quickstart, "_have", lambda m: True)
    p = quickstart.plan(str(corpus_dir))
    assert p.missing == {}
    assert p.install_command() is None


# installing

def test_install_does_not_run_without_being_asked(corpus_dir, monkeypatch):
    monkeypatch.setattr(quickstart, "_have", lambda m: False)
    called = []
    monkeypatch.setattr(quickstart.subprocess, "run",
                        lambda *a, **kw: called.append(a) or None)
    out = quickstart.install(quickstart.plan(str(corpus_dir)))
    assert out["skipped"] is True and not called
    assert "pip install" in out["command"]


def test_install_is_a_no_op_when_nothing_is_missing(corpus_dir, monkeypatch):
    monkeypatch.setattr(quickstart, "_have", lambda m: True)
    out = quickstart.install(quickstart.plan(str(corpus_dir)), yes=True)
    assert out["skipped"] is True and out["reason"] == "nothing missing"


# running

def test_running_the_plan_ingests_builds_and_persists(corpus_dir):
    p = quickstart.plan(str(corpus_dir))
    out = quickstart.run(p)
    assert out["n_documents"] >= 2
    assert out["store"] is not None
    assert out["ids"]
    assert out["store"].get(out["ids"][0]) is not None


def test_the_store_is_left_indexed_so_the_next_open_is_fast(corpus_dir):
    """The pipeline ends here, which is the one moment the snapshot is certainly
    worth its cost."""
    from agent import rcdb

    p = quickstart.plan(str(corpus_dir))
    out = quickstart.run(p)
    if out["store"].backend != "rex":
        pytest.skip("only the embedded store carries a tensor index")
    assert os.path.exists(out["store"]._index_path)
    again = rcdb.open_store(p.store_uri)
    assert again._index is not None


def test_the_result_is_immediately_queryable(corpus_dir):
    p = quickstart.plan(str(corpus_dir))
    out = quickstart.run(p)
    hits = out["store"].query(labels_any=["boundary"], limit=5)
    assert hits, "the corpus was persisted but is not searchable"


def test_running_without_persisting_still_builds(corpus_dir):
    p = quickstart.plan(str(corpus_dir))
    out = quickstart.run(p, persist=False)
    assert out["store"] is None
    assert out["n_documents"] >= 2


def test_a_limit_caps_the_work(corpus_dir):
    p = quickstart.plan(str(corpus_dir))
    out = quickstart.run(p, limit=1)
    assert out["n_documents"] == 1


def test_the_cli_plans_without_changing_anything(corpus_dir, capsys):
    rc = quickstart.main([str(corpus_dir)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "host" in text and "nothing was changed" in text
