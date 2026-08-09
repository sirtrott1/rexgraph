"""An ontology across its releases, as one changing complex.

An ontology is a series, not a file. What matters is what changed between releases and
whether it matters, and both of those are structural questions that a textual diff
answers badly or not at all.

The distinction the tests care most about: a term that stops appearing because it was
MERGED and one that stops appearing because it was DELETED look identical in a text
diff and mean different things to anything holding the old identifier. OBO records the
merge as an `alt_id` on the surviving term, so it can be read rather than guessed.
"""
from __future__ import annotations

import pytest
from agent.ontology_releases import (
    load_releases,
    merges,
    navigate,
    release_diff,
    shared_vocabulary,
    summary,
    temporal_complex,
    term_lifecycle,
)


def _obo(terms) -> str:
    """terms: (id, name, [parents], [alt_ids])"""
    out = "format-version: 1.2\n"
    for tid, name, parents, alts in terms:
        out += f"\n[Term]\nid: {tid}\nname: {name}\n"
        out += "".join(f"is_a: {p}\n" for p in parents)
        out += "".join(f"alt_id: {a}\n" for a in alts)
    return out


@pytest.fixture
def series(tmp_path):
    """Three releases: a term is added, then one is merged and another deleted."""
    v1 = _obo([("T:1", "alpha", ["T:3"], []), ("T:2", "beta", ["T:3"], []),
               ("T:3", "root", [], [])])
    v2 = _obo([("T:1", "alpha", ["T:3"], []), ("T:2", "beta", ["T:3"], []),
               ("T:3", "root", [], []), ("T:4", "delta", ["T:3"], [])])
    # T:1 absorbs T:2 (records it as alt_id); T:4 is deleted with no such record
    v3 = _obo([("T:1", "alpha", ["T:3"], ["T:2"]), ("T:3", "root", [], [])])
    paths = []
    for i, text in enumerate((v1, v2, v3)):
        p = tmp_path / f"v{i}.obo"
        p.write_text(text)
        paths.append(str(p))
    return load_releases(paths, labels=["2026-01", "2026-02", "2026-03"])


#### reading the series


def test_each_release_is_read_in_the_order_given(series):
    assert [r.label for r in series] == ["2026-01", "2026-02", "2026-03"]
    assert [r.n_terms for r in series] == [3, 4, 2]


def test_the_vocabulary_is_the_union_over_every_release(series):
    assert shared_vocabulary(series) == ["T:1", "T:2", "T:3", "T:4"]


def test_a_release_with_no_label_still_gets_one(tmp_path):
    p = tmp_path / "x.obo"
    p.write_text(_obo([("A", "a", ["B"], []), ("B", "b", [], [])]))
    assert load_releases([str(p)])[0].label == "release_0"


#### lifecycle


def test_lifecycle_records_when_each_term_appears_and_stops(series):
    life = term_lifecycle(series)
    assert life["T:4"]["first_seen"] == 1
    assert life["T:4"]["introduced"] is True
    assert life["T:4"]["obsoleted_at"] == 2
    assert life["T:1"]["obsoleted_at"] is None, "T:1 survives to the end"
    assert life["T:3"]["present_in"] == 3


def test_a_term_present_from_the_start_is_not_introduced(series):
    assert term_lifecycle(series)["T:1"]["introduced"] is False


#### merge against deletion, which is the point


def test_a_merge_is_read_from_the_alt_id_the_file_records(series):
    """OBO states a merge; nothing here infers it from a disappearance."""
    merged = [m for m in merges(series) if m["kind"] == "merged"]
    assert len(merged) == 1
    assert merged[0]["term"] == "T:2"
    assert merged[0]["merged_into"] == "T:1"
    assert merged[0]["release"] == "2026-03"


def test_a_deletion_is_reported_apart_from_a_merge(series):
    """T:4 vanishes with no surviving term claiming it, which is a different event
    for anything holding the old id."""
    removed = [m for m in merges(series) if m["kind"] == "removed"]
    assert [m["term"] for m in removed] == ["T:4"]
    assert removed[0]["merged_into"] is None


def test_the_two_are_not_conflated_in_the_summary(series):
    s = summary(series)
    assert [m["term"] for m in s["merges"]] == ["T:2"]
    assert [m["term"] for m in s["removals"]] == ["T:4"]
    assert s["n_obsoleted"] == 2, "both left, for different reasons"


#### diffs


def test_a_diff_reports_what_moved(series):
    d = release_diff(series[0], series[1])
    assert d["added_terms"] == ["T:4"]
    assert d["removed_terms"] == []
    assert d["n_added_relations"] == 1
    assert d["unchanged"] is False


def test_an_unchanged_pair_says_so(tmp_path):
    text = _obo([("A", "a", ["B"], []), ("B", "b", [], [])])
    for i in range(2):
        (tmp_path / f"s{i}.obo").write_text(text)
    rel = load_releases([str(tmp_path / f"s{i}.obo") for i in range(2)])
    assert release_diff(rel[0], rel[1])["unchanged"] is True


#### the temporal complex


def test_the_series_becomes_one_temporal_complex(series):
    temporal, vocab = temporal_complex(series)
    assert int(temporal.T) == 3
    assert len(vocab) == 4
    for t in range(int(temporal.T)):
        assert temporal.reconstruct_at(t) is not None


def test_a_term_keeps_its_index_across_snapshots(series):
    """The identity a temporal complex exists to keep. Parsed independently, each
    release would number its own terms and the same term would be a different vertex
    at every step."""
    temporal, vocab = temporal_complex(series)
    root = vocab.index("T:3")
    import numpy as np
    for t in range(int(temporal.T)):
        rex = temporal.reconstruct_at(t)
        touched = set(np.asarray(rex.sources).tolist()) | set(
            np.asarray(rex.targets).tolist())
        assert root in touched, f"T:3 is not vertex {root} at t={t}"


def test_the_snapshots_track_the_relation_counts(series):
    temporal, _vocab = temporal_complex(series)
    counts = [int(temporal.reconstruct_at(t).nE) for t in range(int(temporal.T))]
    assert counts == [len(r.relations) for r in series]


#### surprise


def _growing(tmp_path, n_releases=6):
    """Steady growth, then one release that reorganises the hierarchy."""
    import random

    def build(n_terms, extra=0, seed=0):
        random.seed(seed)
        out = "format-version: 1.2\n"
        for i in range(n_terms):
            out += f"\n[Term]\nid: T:{i}\nname: t{i}\n"
            if i > 0:
                out += f"is_a: T:{i - 1}\n"
            for _ in range(extra):
                j = random.randrange(0, max(1, i)) if i else 0
                if j != i:
                    out += f"is_a: T:{j}\n"
        return out

    texts = [build(10 + 2 * i) for i in range(n_releases)]
    texts.append(build(10 + 2 * n_releases, extra=3, seed=1))     # the shock
    texts.append(build(12 + 2 * n_releases))
    paths = []
    for i, text in enumerate(texts):
        p = tmp_path / f"g{i}.obo"
        p.write_text(text)
        paths.append(str(p))
    return load_releases(paths, labels=[f"r{i}" for i in range(len(texts))])


def test_steady_growth_costs_nothing(tmp_path):
    """Idle by design: the navigator observes the gate and does no flow work unless
    the change is a surprise against the trend."""
    out = navigate(_growing(tmp_path))
    steady = [s for s in out["steps"][:6]]
    assert not any(s["surprise"] for s in steady), \
        "ordinary growth fired the gate, so nothing would ever be quiet"


def test_a_reorganisation_is_the_step_that_reports(tmp_path):
    out = navigate(_growing(tmp_path))
    surprising = [s["t"] for s in out["surprising"]]
    assert surprising, "the reorganisation did not register at all"
    assert 6 in surprising, f"expected the shock at t=6, got {surprising}"


def test_the_report_is_localised_to_what_changed(tmp_path):
    out = navigate(_growing(tmp_path))
    for s in out["surprising"]:
        assert s["n_region"] > 0, "a surprise reported no region"


def test_the_work_is_proportional_to_the_surprises_not_the_series(tmp_path):
    out = navigate(_growing(tmp_path))
    assert out["flow_calls"] == len(out["surprising"])
    assert out["flow_calls"] < out["n_releases"], \
        "every step did flow work, so the gate bought nothing"


#### everything at once


def test_summary_answers_the_whole_series(series):
    s = summary(series)
    assert s["n_releases"] == 3
    assert s["n_terms_total"] == 4
    assert s["n_introduced"] == 1
    assert len(s["diffs"]) == 2


#### the route


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app
    from fastapi.testclient import TestClient
    yield TestClient(app)
    reset_default_store()


@pytest.fixture
def uploads():
    v1 = _obo([("T:1", "alpha", ["T:3"], []), ("T:2", "beta", ["T:3"], []),
               ("T:3", "root", [], [])])
    v2 = _obo([("T:1", "alpha", ["T:3"], []), ("T:2", "beta", ["T:3"], []),
               ("T:3", "root", [], []), ("T:4", "delta", ["T:3"], [])])
    v3 = _obo([("T:1", "alpha", ["T:3"], ["T:2"]), ("T:3", "root", [], [])])
    return [("files", (f"v{i}.obo", t.encode(), "text/plain"))
            for i, t in enumerate((v1, v2, v3))]


def test_the_route_reports_the_series(client, uploads):
    r = client.post("/api/v1/releases/analyze", files=uploads,
                    data={"labels": "2026-01,2026-02,2026-03"})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["n_releases"] == 3
    assert [m["term"] for m in body["merges"]] == ["T:2"]
    assert [m["term"] for m in body["removals"]] == ["T:4"]
    assert "navigation" in body


def test_the_labels_are_taken_in_upload_order(client, uploads):
    body = client.post("/api/v1/releases/analyze", files=uploads,
                       data={"labels": "first,second,third"}).json()
    assert [r["label"] for r in body["releases"]] == ["first", "second", "third"]


def test_the_series_downloads_as_a_temporal_complex(client, uploads, tmp_path):
    """A `TemporalRex` is not a `RexGraph`, and only the dispatching writer handles
    both: `rex_to_safetensors` fails on an attribute the temporal object does not
    have."""
    r = client.post("/api/v1/releases/analyze", files=uploads,
                    data={"download": "temporal"})
    assert r.status_code == 200, r.text[:300]
    path = str(tmp_path / "series.safetensors")
    with open(path, "wb") as fh:
        fh.write(r.content)
    from rexgraph.io.safetensors_bridge import load_safetensors
    obj = load_safetensors(path)["object"]
    assert type(obj).__name__ == "TemporalRex"
    assert int(obj.T) == 3


def test_one_file_is_not_a_series(client, uploads):
    r = client.post("/api/v1/releases/analyze", files=uploads[:1])
    assert r.status_code == 400
    assert "series" in r.json()["detail"]


#### reachable from the agent builder


def test_the_release_series_is_a_builder_step(tmp_path):
    """The new work has to be composable in the builder, or it is only reachable by
    someone who already knows the route exists."""
    from agent.builder import AgentBuilder
    assert "releases" in AgentBuilder.available_steps()

    v1 = _obo([("T:1", "a", ["T:2"], []), ("T:2", "b", [], [])])
    v2 = _obo([("T:1", "a", ["T:2"], ["T:3"]), ("T:2", "b", [], [])])
    paths = []
    for i, text in enumerate((v1, v2)):
        p = tmp_path / f"r{i}.obo"
        p.write_text(text)
        paths.append(str(p))

    res = AgentBuilder(AgentBuilder.template("releases")).run(files=paths, query="")
    step = res.steps[0]
    assert step.status == "ok", step.error
    assert step.data["n_releases"] == 2


def test_a_single_release_is_skipped_not_failed(tmp_path):
    from agent.builder import AgentBuilder
    p = tmp_path / "one.obo"
    p.write_text(_obo([("A", "a", ["B"], []), ("B", "b", [], [])]))
    res = AgentBuilder({"name": "x", "steps": [{"type": "releases"}]}).run(
        files=[str(p)], query="")
    assert res.steps[0].status == "ok"
    assert "skipped" in res.steps[0].data
