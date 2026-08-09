"""Routes hand back artifacts, not summaries of artifacts.

The library has a complete binary I/O stack: `.rex` bundles, safetensors, HDF5, Zarr,
the labeled vector container and the canonical per-cell table writers. A route that
computes a complex and returns only a JSON description of it has thrown the object
away.

Every test here takes the bytes a route returns and reads them back with the
library's own loader. A download that cannot be reloaded is not an artifact.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pytest
from tests.test_knowledge_roundtrip import BRCA_GAF, BRCA_OBO, GTF
from tests.test_ontology_reasoning import IMMUNE, OBO, REPAIR, _gaf

#: containers written as a directory, so downloaded zipped
ZIPPED = {"rex", "zarr"}
COMPLEX_FORMATS = ["rex", "safetensors", "hdf5", "zarr"]


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
def study_files():
    return [("files", (n, t.encode(), "text/plain")) for n, t in
            (("genes.gtf", GTF), ("goa.gaf", BRCA_GAF), ("go.obo", BRCA_OBO))]


def _materialise(tmp_path, content: bytes, fmt: str, stem: str) -> str:
    """Write the downloaded bytes back to a path the library can open."""
    if fmt in ZIPPED:
        out = str(tmp_path / f"{stem}.{fmt}")
        zipfile.ZipFile(io.BytesIO(content)).extractall(out)
        return out
    suffix = {"safetensors": ".safetensors", "hdf5": ".h5"}[fmt]
    path = str(tmp_path / f"{stem}{suffix}")
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def _load(path: str, fmt: str):
    if fmt == "rex":
        from rexgraph.io import load_rex
        return load_rex(path)
    if fmt == "safetensors":
        from rexgraph.io.safetensors_bridge import load_safetensors
        return load_safetensors(path)["object"]
    if fmt == "hdf5":
        from rexgraph.io import load_hdf5
        return load_hdf5(path)
    from rexgraph.io import load_zarr
    return load_zarr(path)


#### the joined complex comes back as a complex


@pytest.mark.parametrize("fmt", COMPLEX_FORMATS)
def test_the_join_hands_back_the_complex(client, study_files, tmp_path, fmt):
    r = client.post("/api/v1/knowledge/join", files=study_files,
                    data={"download": fmt})
    assert r.status_code == 200, r.text[:200]
    assert r.headers["content-type"] == "application/octet-stream"
    assert len(r.content) > 0
    back = _load(_materialise(tmp_path, r.content, fmt, "joined"), fmt)
    assert back.nV == 8 and back.nE == 9
    assert tuple(back.betti) == (1, 1, 0)


@pytest.mark.parametrize("fmt", COMPLEX_FORMATS)
def test_the_download_is_named_for_its_container(client, study_files, fmt):
    r = client.post("/api/v1/knowledge/join", files=study_files,
                    data={"download": fmt})
    disposition = r.headers["content-disposition"]
    assert ".zip" in disposition if fmt in ZIPPED else ".zip" not in disposition


def test_the_join_still_summarises_when_no_download_is_asked(client, study_files):
    r = client.post("/api/v1/knowledge/join", files=study_files)
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["n_entities"] == 8


def test_an_unknown_container_is_refused_with_the_list(client, study_files):
    r = client.post("/api/v1/knowledge/join", files=study_files,
                    data={"download": "csv"})
    assert r.status_code == 400
    assert "rex" in r.json()["detail"]


#### features go out as the labeled vector container


def test_features_come_back_as_vectors(client, study_files, tmp_path):
    """`save_vectors` is the container for exactly this, and a feature matrix
    returned as a JSON list of lists is not reusable by anything else here."""
    r = client.post("/api/v1/knowledge/join", files=study_files,
                    data={"download": "features"})
    assert r.status_code == 200, r.text[:200]
    path = str(tmp_path / "f.safetensors")
    with open(path, "wb") as fh:
        fh.write(r.content)

    from rexgraph.io import load_vectors
    matrix, labels, names, meta = load_vectors(path)
    assert matrix.shape == (9, 11), "one row per relation, one column per feature"
    assert len(names) == matrix.shape[1]
    assert names[0].startswith("char_")
    assert labels is not None and len(labels) == matrix.shape[0]
    assert "genes.gtf" in str(meta.get("classes", "")), \
        "the relation classes did not travel with the matrix"


def test_the_feature_columns_are_the_canonical_four_channels(client, study_files,
                                                             tmp_path):
    r = client.post("/api/v1/knowledge/join", files=study_files,
                    data={"download": "features"})
    path = str(tmp_path / "f.safetensors")
    with open(path, "wb") as fh:
        fh.write(r.content)
    from rexgraph.io import load_vectors
    _m, _l, names, _meta = load_vectors(path)
    for channel in ("L1_down", "L_O", "L_SG", "L_C"):
        assert f"char_{channel}" in names


#### enrichment goes out through the table writer


def test_enriched_terms_come_back_as_a_table(client, tmp_path):
    gaf = _gaf([(g, "GO:0006281") for g in REPAIR]
               + [(g, "GO:0006955") for g in IMMUNE])
    r = client.post("/api/v1/enrichment/run",
                    files=[("files", ("go.obo", OBO.encode(), "text/plain")),
                           ("files", ("goa.gaf", gaf.encode(), "text/plain"))],
                    data={"study": "BRCA1,BRCA2,ATM,RAD51", "download": "terms"})
    assert r.status_code == 200, r.text[:300]
    pq = pytest.importorskip("pyarrow.parquet")
    path = str(tmp_path / "e.parquet")
    with open(path, "wb") as fh:
        fh.write(r.content)
    table = pq.read_table(path)
    assert set(table.column_names) >= {"n_study", "n_term", "fold_enrichment",
                                       "p_value", "q_value"}
    assert table.num_rows > 0
    assert min(table.column("p_value").to_pylist()) == pytest.approx(15 / 495)


def test_a_table_download_with_nothing_enriched_says_so(client):
    r = client.post("/api/v1/enrichment/run",
                    files=[("files", ("go.obo", OBO.encode(), "text/plain"))],
                    data={"study": "NOT_A_GENE", "download": "terms"})
    assert r.status_code == 400


#### the stored complex, in every container


@pytest.fixture
def stored(client, tmp_path):
    from agent.knowledge import join
    from agent.rcdb import default_store

    def w(name, text):
        p = tmp_path / name
        p.write_text(text)
        return str(p)

    join(w("g.gtf", GTF), w("a.gaf", BRCA_GAF), w("o.obo", BRCA_OBO)).store(
        default_store(), "brca")
    return "brca"


@pytest.mark.parametrize("fmt", COMPLEX_FORMATS)
def test_a_stored_complex_exports_in_every_container(client, stored, tmp_path, fmt):
    r = client.get(f"/api/v1/db/export/{stored}?format={fmt}")
    assert r.status_code == 200, r.text[:200]
    back = _load(_materialise(tmp_path, r.content, fmt, "exported"), fmt)
    assert back.nE == 9


def test_export_defaults_to_the_storage_form(client, stored):
    """safetensors is what the RCDB stores, so it is the default and needs no
    conversion."""
    r = client.get(f"/api/v1/db/export/{stored}")
    assert r.status_code == 200
    assert "safetensors" in r.headers["content-disposition"]


def test_exporting_an_unknown_container_is_refused(client, stored):
    assert client.get(f"/api/v1/db/export/{stored}?format=csv").status_code == 400


def test_exporting_a_record_that_does_not_exist_is_a_404(client):
    assert client.get("/api/v1/db/export/nope").status_code == 404


#### the encoder trap


def test_kernel_output_survives_the_response_encoder(client, stored):
    """FastAPI's `jsonable_encoder` runs before the response class, so a route
    returning raw ndarray/np.generic 500s however the response is configured."""
    import json
    for dim in (0, 1):
        r = client.get(f"/api/v1/db/explain/{stored}?dim={dim}&idx=0")
        assert r.status_code == 200, r.text[:200]
        json.loads(r.text)


def test_plain_converts_every_numpy_shape():
    from agent.server.artifacts import plain
    out = plain({"a": np.arange(3), "b": np.float64(1.5),
                 "c": [np.int32(2), {"d": np.array([[1.0, 2.0]])}]})
    assert out == {"a": [0, 1, 2], "b": 1.5, "c": [2, {"d": [[1.0, 2.0]]}]}
    assert isinstance(out["b"], float) and isinstance(out["c"][0], int)


def test_no_scratch_files_are_left_behind(client, study_files):
    """The writers read their bytes and delete the scratch, because a streaming
    response outlives the handler that would otherwise clean up."""
    import glob
    import tempfile
    before = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                        "rexgraph_artifact_*")))
    for fmt in COMPLEX_FORMATS:
        client.post("/api/v1/knowledge/join", files=study_files,
                    data={"download": fmt})
    after = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                       "rexgraph_artifact_*")))
    assert after == before, f"scratch left behind: {sorted(after - before)}"


#### per-cell analysis tables


@pytest.fixture
def session(client, tmp_path):
    from agent.knowledge import join
    from agent.server.app import get_store

    def w(name, text):
        p = tmp_path / name
        p.write_text(text)
        return str(p)

    from tests.test_knowledge_roundtrip import BRCA_GAF, BRCA_OBO, GTF
    k = join(w("g.gtf", GTF), w("a.gaf", BRCA_GAF), w("o.obo", BRCA_OBO))
    s = get_store().create(name="tables")
    s.add_snapshot(rex=k.rex(), action="join", params={}, results={}, summary="")
    return s.session_id


@pytest.mark.parametrize("kind,column", [
    ("character", "chi_L1_down"),
    ("persistence", "birth"),
])
def test_an_analysis_table_comes_back_as_parquet(client, session, tmp_path,
                                                 kind, column):
    """Per-cell output goes through the canonical writer, so the SQL bridge and the
    warehouse read it without a conversion nobody wrote."""
    pq = pytest.importorskip("pyarrow.parquet")
    r = client.get(f"/api/analysis/{session}/table?kind={kind}")
    assert r.status_code == 200, r.text[:200]
    path = str(tmp_path / f"{kind}.parquet")
    with open(path, "wb") as fh:
        fh.write(r.content)
    table = pq.read_table(path)
    assert column in table.column_names
    assert table.num_rows > 0


def test_the_character_table_has_a_column_per_channel(client, session, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    r = client.get(f"/api/analysis/{session}/table?kind=character")
    path = str(tmp_path / "c.parquet")
    with open(path, "wb") as fh:
        fh.write(r.content)
    names = pq.read_table(path).column_names
    for channel in ("L1_down", "L_O", "L_SG", "L_C"):
        assert f"chi_{channel}" in names


def test_an_unknown_table_is_refused_with_the_list(client, session):
    r = client.get(f"/api/analysis/{session}/table?kind=bogus")
    assert r.status_code == 400 and "character" in r.json()["detail"]


def test_a_table_for_a_missing_session_is_a_404(client):
    assert client.get("/api/analysis/nope/table").status_code == 404
