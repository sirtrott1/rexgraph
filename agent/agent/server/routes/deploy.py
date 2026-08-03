"""agent.server.routes.deploy: containerize an agent/pipeline for deployment."""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response

from agent.deploy import bundle_to_zip, generate_bundle, spec_from_dict

router = APIRouter(prefix="/v1/deploy")


@router.post("/preview")
async def deploy_preview(body: dict = Body(...)):
    """Preview the generated deployment files without downloading.

    Body accepts any DeploymentSpec field: name, mode ('service'|'pipeline'),
    extras, port, model_url, depth, query, backend, ontology, source,
    builder_config.
    """
    try:
        bundle = generate_bundle(spec_from_dict(body))
    except Exception as e:
        raise HTTPException(400, f"Could not generate bundle: {e}")
    return {
        "files": list(bundle.keys()),
        "Dockerfile": bundle["Dockerfile"],
        "docker-compose.yml": bundle["docker-compose.yml"],
        "README.md": bundle["README.md"],
    }


@router.post("/bundle")
async def deploy_bundle(body: dict = Body(...)):
    """Generate and download a deployable container bundle as a zip."""
    try:
        spec = spec_from_dict(body).normalized()
        data = bundle_to_zip(generate_bundle(spec))
    except Exception as e:
        raise HTTPException(400, f"Could not build bundle: {e}")
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="{spec.name}-deploy.zip"'},
    )
