"""
agent.server.routes.ontology: diagnose ontologies as complexes, from files or text.

An ontology arrives as a file. `.obo`, `.owl`, `.ttl`, `.nt` and OBO Graphs JSON are
what GO, HPO, MONDO, ChEBI and UBERON publish, and `.gaf`/`.gpad` is what relates
those terms to gene products. All of them read here, and several read together: an
annotation file loaded with its ontology resolves its term ids against the hierarchy.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/v1/ontology")

#: how many triples a response lists back. The complex keeps all of them.
PREVIEW_LIMIT = 500


def _count_predicates(triples) -> dict:
    out: dict[str, int] = {}
    for _s, p, _o in triples:
        out[p] = out.get(p, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _store_if_asked(model, report: dict, body: dict) -> None:
    if not (body.get("lineage_id") or body.get("store_id")):
        return
    from agent import ontology_complex as oc
    try:
        rex, meta = oc.ontology_to_rex(model)
        if rex is None:
            return
        from agent.rcdb import default_store as _store
        tags = (body.get("tags") or []) + ["ontology"]
        if body.get("lineage_id"):
            from agent.rcdb import put_version
            report["version"] = put_version(_store(), body["lineage_id"],
                                            rex, meta=meta, tags=tags)
        else:
            _store().put(body["store_id"], rex, meta=meta, tags=tags)
            report["stored_as"] = body["store_id"]
    except Exception as e:
        report["store_error"] = str(e)


def _report(parsed, body: dict) -> dict:
    """Diagnose a parsed ontology and attach what the file said about itself."""
    from agent import ontology_complex as oc
    from agent.adapters import ontology_formats as OF

    named = bool(body.get("use_names", True))
    triples = parsed.named_triples() if named else list(parsed.triples)
    if not triples:
        raise HTTPException(
            400, f"parsed as {parsed.meta.get('format')} but found no relations "
                 "between terms")
    model = oc.parse_rdf(triples)
    report = oc.diagnose_ontology(model)
    report["source"] = parsed.meta
    report["n_labels"] = len(parsed.labels)
    report["declarations"] = OF._count(parsed.declarations)
    report["predicates"] = _count_predicates(triples)
    report["triples"] = [list(t) for t in triples[:PREVIEW_LIMIT]]
    report["n_triples"] = len(triples)
    report["truncated"] = len(triples) > PREVIEW_LIMIT
    _store_if_asked(model, report, body)
    return report


@router.get("/formats")
async def ontology_formats():
    """The formats that can be read, and the extensions each claims.

    The screen builds its format selector from this, so a parser registered in
    `ontology_formats.PARSERS` reaches the UI without a frontend edit.
    """
    from agent.adapters import ontology_formats as OF
    formats = OF.available_formats()
    return {
        "formats": formats,
        "extensions": sorted(e for exts in formats.values() for e in exts),
        "detects_from_content": True,
        "annotation_formats": ["gaf", "gpad"],
    }


@router.post("/analyze")
async def analyze_ontology(body: dict = Body(...)):
    """Diagnose an ontology from triples or from text in any supported format.

    Body accepts either `triples` ([[s, p, o], ...]) or `text` with an optional
    `format`; with neither given, the format is detected from the content.
    Optional: `store_id` / `lineage_id` / `tags` to keep the complex, and
    `use_names` to report term ids rather than their labels.
    """
    from agent.adapters import ontology_formats as OF

    triples = body.get("triples")
    text = body.get("text")
    if triples:
        parsed = OF.ParsedOntology([tuple(t) for t in triples],
                                   meta={"format": "triples"})
    elif text and str(text).strip():
        try:
            parsed = OF.parse(text, body.get("format") or None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    else:
        raise HTTPException(
            400, "Provide 'triples' as [[subject, predicate, object], ...] or "
                 "'text' in a supported format")
    return _report(parsed, body)


@router.post("/reason")
async def reason_over_ontology(
    files: list[UploadFile] = File(None),
    text: str = Form(""),
    format: str = Form(""),
    terms: str = Form(""),
    download: str = Form(""),
):
    """Consistency, classification and module extraction over an ontology.

    Each answer is an exact integer invariant. A class that cannot have an instance
    comes back named, with both ancestor chains that made it so, because
    "unsatisfiable" on its own is not something a curator can act on.

    `download` returns the SIGNED complex the reasoning was done on rather than the
    report: `rex`, `safetensors`, `hdf5`, `zarr`. That complex is the object the
    answers came from (disjointness carried as a negative sign, subsumption oriented
    child to parent), so handing it back means the reasoning can be reproduced or
    extended somewhere else instead of only read.
    """
    from agent import ontology_reasoning as R

    parsed = await _collect(files, text, format)
    triples = parsed.named_triples()
    if not triples:
        raise HTTPException(400, "no relations between terms were found")

    if download.strip():
        from agent.server.artifacts import complex_file
        rc = R.build(triples)
        return complex_file(rc.rex, "reasoning", download.strip())

    wanted = [t.strip() for t in terms.split(",") if t.strip()]
    out = R.reason(triples, terms=wanted or None)
    out["source"] = parsed.meta
    return out


async def _collect(files, text, format):
    """Whatever the caller sent, as one parsed ontology."""
    from agent.adapters import ontology_formats as OF

    parts = []
    for f in (files or []):
        raw = await f.read()
        fmt = format or OF.format_for_extension(f.filename or "") or None
        try:
            p = OF.parse(raw.decode("utf-8", errors="replace"), fmt)
        except ValueError as e:
            raise HTTPException(400, f"{f.filename}: {e}") from e
        p.meta["file"] = f.filename
        parts.append(p)
    if text and str(text).strip():
        try:
            parts.append(OF.parse(text, format or None))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    if not parts:
        raise HTTPException(400, "Provide files or text")
    return parts[0] if len(parts) == 1 else OF.combine(*parts)


@router.post("/upload")
async def upload_ontology(
    files: list[UploadFile] = File(...),
    format: str = Form(""),
    store_id: str = Form(""),
    lineage_id: str = Form(""),
    use_names: bool = Form(True),
):
    """Diagnose one or more ontology files.

    Several files are combined into one complex, which is the point of uploading an
    annotation set beside its ontology: the `.gaf` names term ids and the `.obo`
    says what they mean and how they sit under each other, so a gene product ends up
    connected to the hierarchy rather than to a bare accession.
    """
    from agent.adapters import ontology_formats as OF

    if not files:
        raise HTTPException(400, "No files uploaded")

    parsed_all, failures = [], []
    for f in files:
        raw = await f.read()
        text = raw.decode("utf-8", errors="replace")
        fmt = format or OF.format_for_extension(f.filename or "") or None
        try:
            p = OF.parse(text, fmt)
        except ValueError as e:
            failures.append({"file": f.filename, "error": str(e)})
            continue
        if not p.triples:
            # A parser that reads a file it cannot understand returns nothing rather
            # than raising. Letting that through would drop the file silently and
            # report a diagnosis of whatever else was uploaded.
            failures.append({
                "file": f.filename,
                "error": f"read as {p.meta.get('format')} but no relations between "
                         "terms were found"})
            continue
        p.meta["file"] = f.filename
        parsed_all.append(p)

    if not parsed_all:
        raise HTTPException(
            400, "; ".join(f"{x['file']}: {x['error']}" for x in failures)
                 or "nothing could be parsed")

    parsed = parsed_all[0] if len(parsed_all) == 1 else OF.combine(*parsed_all)
    body = {"store_id": store_id, "lineage_id": lineage_id, "use_names": use_names}
    report = _report(parsed, body)
    report["files"] = [p.meta.get("file") for p in parsed_all]
    if failures:
        report["failed_files"] = failures
    return report
