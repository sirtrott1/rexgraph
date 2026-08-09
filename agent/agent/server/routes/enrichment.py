"""
agent.server.routes.enrichment: which terms a set of entities is concentrated in.

Takes the ontology and the annotations as files, joins them, and answers with both
readings: the hypergeometric one a reviewer expects, and the persistence one the
extra structure buys. Keeping them side by side is the only way to tell whether the
structure bought anything.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/v1/enrichment")

#: how many enriched terms a response lists back
LIMIT = 200


@router.post("/run")
async def run_enrichment(
    files: list[UploadFile] = File(...),
    study: str = Form(...),
    universe: str = Form(""),
    min_term_size: int = Form(1),
    download: str = Form(""),
):
    """Enrich a study set against the uploaded ontology and annotations.

    `study` and `universe` are comma or whitespace separated. Upload the annotation
    file together with the ontology it was made against, or the terms resolve to
    bare accessions and the hierarchy contributes nothing.
    """
    from agent.enrichment import enrich
    from agent.knowledge import join

    study_set = _split(study)
    if not study_set:
        raise HTTPException(400, "Provide a study set")

    tmpdir = tempfile.mkdtemp(prefix="rexgraph_enrich_")
    try:
        paths, origins = [], []
        for f in files:
            name = os.path.basename(f.filename or "upload")
            path = os.path.join(tmpdir, name)
            with open(path, "wb") as fh:
                fh.write(await f.read())
            paths.append(path)
            origins.append(name)
        try:
            k = join(*paths, origins=origins)
        except Exception as e:                     # noqa: BLE001 - caller's input
            raise HTTPException(400, str(e)[:300]) from e

        out = enrich(k, study_set, universe=_split(universe) or None,
                     min_term_size=int(min_term_size))
        if download.strip().lower() == "terms":
            import numpy as _np

            from agent.server.artifacts import metrics_file
            rows = out["terms"]
            if not rows:
                raise HTTPException(400, "no term was enriched, so there is no table")
            return metrics_file({
                "n_study": _np.asarray([r["n_study"] for r in rows]),
                "n_term": _np.asarray([r["n_term"] for r in rows]),
                "expected": _np.asarray([r["expected"] for r in rows]),
                "fold_enrichment": _np.asarray([r["fold_enrichment"] for r in rows]),
                "p_value": _np.asarray([r["p_value"] for r in rows]),
                "q_value": _np.asarray([r["q_value"] for r in rows]),
            }, "enrichment", index_name="term_rank")
        out["terms"] = out["terms"][:LIMIT]
        out["sources"] = k.report["sources"]
        out["n_joined"] = k.report["n_joined"]
        if not out["n_universe"]:
            out["warning"] = (
                "no entity in the complex carries an annotation relation, so there "
                "is nothing to enrich against. Upload an annotation file (.gaf or "
                ".gpad) alongside the ontology.")
        return out
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _split(value: str) -> list[str]:
    return [x.strip() for x in str(value).replace(",", " ").split() if x.strip()]
