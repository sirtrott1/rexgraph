"""
agent.server.routes.knowledge: schemas, ontologies and scientific files as one complex.

The upload takes any mix of them. What makes the result one thing rather than several
is the join: each file states its own cross-references, and following them is what
lets a genome annotation, a GO annotation set and the ontology share vertices instead
of sitting in three disconnected components.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/v1/knowledge")

#: how many relations a response lists back. The complex keeps all of them.
PREVIEW_LIMIT = 500


@router.get("/formats")
async def knowledge_formats():
    """Everything that can be joined, by extension.

    Both registries in one answer, because from this screen's point of view an
    ontology and a structure file are the same kind of thing: a file that names
    entities and relates them.
    """
    from agent.adapters.formats import available_extensions
    from agent.adapters.ontology_formats import available_formats

    onto = available_formats()
    onto_ext = sorted({e for exts in onto.values() for e in exts})
    science = {e: r for e, r in available_extensions().items()
               if not r.startswith("ontology_")}
    return {
        "ontology": onto,
        "ontology_extensions": onto_ext,
        "science": science,
        "science_extensions": sorted(science),
        "extensions": sorted(set(onto_ext) | set(science)),
    }


@router.post("/join")
async def join_files(
    files: list[UploadFile] = File(...),
    store_id: str = Form(""),
    tags: str = Form(""),
    face_selection: str = Form(""),
    download: str = Form(""),
):
    """Join uploaded files into one complex and report what connected.

    `download` returns the complex itself instead of the summary: `rex`,
    `safetensors`, `hdf5`, `zarr` for the complex, or `features` for the per-relation
    structural feature matrix in the labeled vector container. A relational complex
    is not a JSON document and the stack has containers for it.

    The report is the part worth reading: which entities were reached by more than
    one file, which identifiers two files disagreed about, and how many relations
    each contributed. A join that silently connected nothing looks exactly like a
    join that worked, so the counts are the answer, not a detail.
    """
    from agent.knowledge import join

    if not files:
        raise HTTPException(400, "No files uploaded")

    tmpdir = tempfile.mkdtemp(prefix="rexgraph_knowledge_")
    paths, origins, failures = [], [], []
    try:
        for f in files:
            name = os.path.basename(f.filename or "upload")
            path = os.path.join(tmpdir, name)
            with open(path, "wb") as fh:
                fh.write(await f.read())
            paths.append(path)
            origins.append(name)

        parts, kept_paths, kept_origins = [], [], []
        from agent.knowledge import as_part
        for path, name in zip(paths, origins, strict=False):
            try:
                parts.append(as_part(path, name))
                kept_paths.append(path)
                kept_origins.append(name)
            except Exception as e:                   # noqa: BLE001 - reported below
                failures.append({"file": name, "error": str(e)[:300]})

        if not parts:
            raise HTTPException(
                400, "; ".join(f"{x['file']}: {x['error']}" for x in failures)
                     or "nothing could be read")

        k = join(*parts, origins=kept_origins)
        try:
            rex = k.rex(face_selection=face_selection or None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

        if download.strip():
            from agent.server.artifacts import complex_file, vectors_file
            stem = "knowledge"
            if download.strip().lower() == "features":
                X, names, y, classes = k.features(rex=rex)
                return vectors_file(
                    X, y, stem, feature_names=names,
                    metadata={"classes": ",".join(classes),
                              "origins": ",".join(p.origin for p in k.parts)})
            return complex_file(rex, stem, download.strip())

        out = {
            "n_entities": k.nV,
            "n_relations": k.nE,
            "report": k.report,
            "nV": int(rex.nV), "nE": int(rex.nE), "nF": int(rex.nF),
            "betti": [int(b) for b in rex.betti],
            "relations": [list(t) for t in k.triples(with_origin=True)[:PREVIEW_LIMIT]],
            "truncated": k.nE > PREVIEW_LIMIT,
            "entities": {k.display(c): v for c, v in list(k.entities.items())[:200]},
        }
        if failures:
            out["failed_files"] = failures
        if store_id.strip():
            try:
                extra = [t.strip() for t in tags.split(",") if t.strip()]
                k.store(_store(), store_id.strip(), tags=extra, rex=rex)
                out["stored_as"] = store_id.strip()
            except Exception as e:                   # noqa: BLE001
                out["store_error"] = str(e)
        return out
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _store():
    from agent.rcdb import default_store
    return default_store()


@router.post("/health")
async def knowledge_health(files: list[UploadFile] = File(...)):
    """Whether load drains through the joined structure or gets trapped circulating.

    Reports the Hodge split of the coordination graph, the entities every path runs
    through, and any cycle holding harmonic content.
    """
    k, tmpdir = await _joined(files)
    try:
        return {"health": k.health(), "n_entities": k.nV, "n_relations": k.nE}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.post("/propagate")
async def knowledge_propagate(
    files: list[UploadFile] = File(...),
    seed: str = Form(...),
    t: float = Form(1.0),
    limit: int = Form(50),
):
    """Diffuse a seed across grades and report what it reaches.

    `seed` is a comma or whitespace separated list of entity names; every relation
    touching one of them starts at 1 and the coupled field operator carries it.
    """
    names = [x.strip() for x in str(seed).replace(",", " ").split() if x.strip()]
    if not names:
        raise HTTPException(400, "Provide 'seed' as one or more entity names")
    k, tmpdir = await _joined(files)
    try:
        import numpy as np
        field = k.propagate(names, t=float(t))
        triples = k.triples()
        order = np.argsort(-np.abs(field))[: int(limit)]
        return {
            "seed": names, "t": float(t),
            "reached": [{"relation": list(triples[int(i)]),
                         "value": float(field[int(i)])} for i in order],
        }
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.post("/bundle")
async def knowledge_training_bundle(
    files: list[UploadFile] = File(...),
    target: str = Form("relation"),
    weight_by: str = Form("degree"),
):
    """The joined complex as a training set, in the labeled vector container.

    Relations are the rows and the complex's own tensor fields are the columns, so a
    model learns from where a relation sits rather than from text about it.
    """
    from agent.server.artifacts import vectors_file
    from agent.warehouse.source import knowledge_bundle

    k, tmpdir = await _joined(files)
    try:
        b = knowledge_bundle(k, weight_by=weight_by, target=target)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    import numpy as np
    return vectors_file(
        np.asarray(b.X), np.asarray(b.y), "knowledge_training",
        feature_names=b.meta["feature_names"],
        metadata={"classes": ",".join(b.meta["classes"]),
                  "kind": b.kind, "target": target})


async def _joined(files):
    """Write the uploads to a scratch dir and join them. Caller removes the dir."""
    from agent.knowledge import join

    if not files:
        raise HTTPException(400, "No files uploaded")
    tmpdir = tempfile.mkdtemp(prefix="rexgraph_knowledge_")
    paths, origins = [], []
    for f in files:
        name = os.path.basename(f.filename or "upload")
        path = os.path.join(tmpdir, name)
        with open(path, "wb") as fh:
            fh.write(await f.read())
        paths.append(path)
        origins.append(name)
    try:
        return join(*paths, origins=origins), tmpdir
    except Exception as e:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(400, str(e)[:300]) from e
