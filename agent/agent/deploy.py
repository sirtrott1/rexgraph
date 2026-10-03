"""
agent.deploy: turn a RexGraph agent/pipeline into a deployable container.

Generates a self contained deployment bundle (Dockerfile, compose file,
entrypoint, config, README) with no runtime dependencies beyond the
standard library. Docker itself is external tooling the operator already
has; nothing here imports it.

Two modes:
  * ``service``: containerize the full app: web UI + REST API (rcf server).
  * ``pipeline``: containerize a headless document processing agent that
    runs the user's configured analysis (depth / query / backend / model)
    over mounted inputs and writes JSON results.

Usage:

    from agent.deploy import DeploymentSpec, generate_bundle, bundle_to_zip
    spec = DeploymentSpec(name="my-agent", mode="pipeline",
                          depth="standard", query="key findings?")
    data = bundle_to_zip(generate_bundle(spec))
"""

from __future__ import annotations

import io
import json
import zipfile
import re
import shlex
from dataclasses import dataclass, field

# spec

_VALID_EXTRAS = {
    "server", "ocr", "training", "yaml",
    "langchain", "langgraph", "trustgraph",
}


@dataclass
class DeploymentSpec:
    name: str = "rexgraph-agent"
    mode: str = "service"                    # "service" | "pipeline"
    extras: list[str] = field(default_factory=lambda: ["server"])
    port: int = 8080
    python_version: str = "3.13"        # matches the pinned env (see environment.yml)
    # where to get the packages: "pypi" (pip install by name) or
    # "local" (copy a wheel from the build context)
    source: str = "pypi"
    # LLM the container should talk to (OpenAI compatible); optional
    model_url: str = ""
    # Auth posture. The server is secure by default (it enables auth and prints
    # a one time admin token to the container logs on first start). Set insecure
    # only when auth is terminated upstream or the port is firewalled.
    insecure: bool = False
    # pipeline mode settings (ignored in service mode)
    depth: str = "standard"
    query: str = ""
    backend: str = ""                        # ocr backend, e.g. "tesseract"
    ontology: bool = False
    # the full agent builder config, embedded for provenance/reference
    builder_config: dict | None = None

    def normalized(self) -> DeploymentSpec:
        self.mode = self.mode if self.mode in ("service", "pipeline") else "service"
        self.source = self.source if self.source in ("pypi", "local") else "pypi"
        extras = [e for e in self.extras if e in _VALID_EXTRAS]
        if self.mode == "service" and "server" not in extras:
            extras = ["server"] + extras
        if self.mode == "pipeline" and "ocr" not in extras:
            extras.append("ocr")
        self.extras = sorted(set(extras))
        if type(self.port) is str and re.fullmatch(r"[0-9]{1,5}", self.port): self.port = int(self.port)
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("deployment port must be an integer from 1 to 65535")
        if type(self.python_version) is not str or not re.fullmatch(r"3\.1[0-9](?:\.[0-9]{1,3})?", self.python_version):
            raise ValueError("deployment Python version must be a literal Python 3.10+ numeric tag")
        if type(self.insecure) is not bool or type(self.ontology) is not bool:
            raise ValueError("deployment switches must be booleans")
        for name in ("query", "backend", "model_url", "depth"):
            value = getattr(self, name)
            if type(value) is not str or "\x00" in value or len(value.encode("utf-8")) > 1024*1024:
                raise ValueError(f"deployment {name} requires bounded literal text")
        if self.depth not in {"quick", "standard", "full", "deep"}:
            raise ValueError("unsupported deployment analysis depth")
        if any(c in self.model_url for c in "\r\n"):
            raise ValueError("deployment model URL must occupy one line")
        # container name / image tag must be docker safe
        import re as _re
        safe = "".join(c if (c.isascii() and (c.isalnum() or c in "-_.")) else "-"
                       for c in (self.name or "rexgraph-agent")).lower()
        safe = _re.sub(r"-{2,}", "-", safe).strip("-.")
        self.name = safe or "rexgraph-agent"
        return self


# file templates

def _extras_str(spec: DeploymentSpec) -> str:
    return ",".join(spec.extras) if spec.extras else "server"


def _dockerfile(spec: DeploymentSpec) -> str:
    extras = _extras_str(spec)
    if spec.source == "local":
        install = (
            "# Local install: place rexgraph and rexgraph-agent wheels in ./wheels\n"
            "COPY wheels/ /tmp/wheels/\n"
            "RUN pip install --no-cache-dir --no-index --find-links /tmp/wheels/ /tmp/wheels/*.whl && \\\n"
            f'    pip install --no-cache-dir "rexgraph-agent[{extras}]" '
            "--no-index --find-links /tmp/wheels/"
        )
    else:
        install = (
            "# Install from PyPI. rexgraph must be published (or use source=local).\n"
            f'RUN pip install --no-cache-dir "rexgraph-agent[{extras}]"'
        )
    return f"""# Generated by rexgraph agent.deploy - {spec.name} ({spec.mode})
FROM python:{spec.python_version}-slim

# System deps. libopenblas-dev provides the BLAS/LAPACK symbols the
# compiled RexGraph kernels need at runtime (cblas_dgemm, dsyev_, ...).
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential libopenblas-dev {'tesseract-ocr ' if 'ocr' in spec.extras else ''}\\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

{install}

COPY rexgraph-agent.json /app/rexgraph-agent.json
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Non-root user for safety
RUN mkdir -p /app/data/cache /app/data/config /app/data/sessions /app/inputs /app/outputs && \\
    useradd -m -u 10001 rexuser && chown -R rexuser /app
USER rexuser

ENV REXGRAPH_CACHE_DIR=/app/data/cache \\
    REXGRAPH_CONFIG_DIR=/app/data/config \\
    REXGRAPH_SESSION_DIR=/app/data/sessions \\
    PYTHONUNBUFFERED=1

VOLUME ["/app/data", "/app/inputs", "/app/outputs"]
{f'EXPOSE {spec.port}' if spec.mode == 'service' else ''}
ENTRYPOINT ["/app/entrypoint.sh"]
"""


def _entrypoint(spec: DeploymentSpec) -> str:
    model = ("if [ -z \"${CHAT_MODEL_URL+x}\" ]; then\n  export CHAT_MODEL_URL="+
             shlex.quote(spec.model_url)+"\nfi\n")
    if spec.mode == "service":
        if spec.insecure:
            # Explicit opt out: run open. Required to both skip the secure
            # default and permit the public bind. Only safe behind upstream auth
            # or a firewalled port.
            auth = "export RCF_ALLOW_INSECURE=1\n"
        else:
            # Secure by default: the server enables auth on first start and
            # prints a one time admin token to the container logs.
            auth = ""
        return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/data/cache
{model}{auth}export RCF_HOST="${{RCF_HOST:-0.0.0.0}}"
export RCF_PORT="${{RCF_PORT:-{spec.port}}}"
exec rcf-server
"""
    # pipeline mode
    q = f" --query {shlex.quote(spec.query)}" if spec.query else ""
    b = f" --backend {shlex.quote(spec.backend)}" if spec.backend else ""
    onto = " --ontology" if spec.ontology else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail
# Headless agent: analyse everything under /app/inputs and write JSON to
# /app/outputs/results.json using the pipeline this agent was built with.
mkdir -p /app/data/cache /app/outputs
{model}export REXGRAPH_CONFIG_DIR="${{REXGRAPH_CONFIG_DIR:-/app/data/config}}"
if [ -z "$(ls -A /app/inputs 2>/dev/null)" ]; then
  echo "No inputs found. Mount files into /app/inputs (-v ./inputs:/app/inputs)."
  exit 2
fi
exec rexgraph-run --input-dir /app/inputs \\
  --depth {shlex.quote(spec.depth)}{q}{b}{onto} \\
  --output /app/outputs/results.json --json
"""


def _compose(spec: DeploymentSpec) -> str:
    if spec.mode == "service":
        ports = f'    ports:\n      - "{spec.port}:{spec.port}"\n'
        vols = ("    volumes:\n"
                "      - rexgraph_data:/app/data\n")
        cmd = ""
    else:
        ports = ""
        vols = ("    volumes:\n"
                "      - rexgraph_data:/app/data\n"
                "      - ./inputs:/app/inputs\n"
                "      - ./outputs:/app/outputs\n")
        cmd = ""
    env = "    env_file:\n      - .env\n"
    return f"""# Generated by rexgraph agent.deploy
services:
  {spec.name}:
    build: .
    image: {spec.name}:latest
    container_name: {spec.name}
    restart: unless-stopped
{ports}{env}{vols}{cmd}volumes:
  rexgraph_data:
"""


def _dockerignore() -> str:
    return "\n".join([
        "**/__pycache__", "*.pyc", "*.pyo", ".git", ".venv", "venv",
        "data/", "outputs/", "*.log", ".env",
    ]) + "\n"


def _env_example(spec: DeploymentSpec) -> str:
    return (
        "# Copy to .env and edit. Values here override the image defaults.\n"
        f"# CHAT_MODEL_URL points at any OpenAI-compatible LLM server.\n"
        "# CHAT_MODEL_URL=\n"
        "# CHAT_MODEL_NAME=my-model\n"
        "REXGRAPH_CACHE_DIR=/app/data/cache\n"
        "# Set to 1 to disable content-addressed caching\n"
        "# REXGRAPH_NO_CACHE=0\n"
    )


def _config_json(spec: DeploymentSpec) -> str:
    cfg = {
        "name": spec.name,
        "mode": spec.mode,
        "rexgraph_agent": {"extras": spec.extras},
        "runtime": {
            "depth": spec.depth,
            "query": spec.query,
            "backend": spec.backend,
            "ontology": spec.ontology,
            "model_url": spec.model_url,
        },
        "builder_config": spec.builder_config or {},
    }
    return json.dumps(cfg, indent=2) + "\n"


def _readme(spec: DeploymentSpec) -> str:
    if spec.mode == "service":
        run = (
            "## Run\n\n"
            "```bash\n"
            "cp .env.example .env\n"
            "docker compose up --build\n"
            "```\n\n"
            f"Then open the web UI / API at http://localhost:{spec.port} "
            f"(health check: `GET /api/health`).\n"
        )
        usage = (
            "This container runs the full RexGraph app: web UI, REST API, "
            "corpus, chat, and the ecosystem integrations."
        )
    else:
        run = (
            "## Run\n\n"
            "```bash\n"
            "cp .env.example .env\n"
            "mkdir -p inputs outputs data\n"
            "# The container writes as UID 10001; make outputs writable by that UID.\n"
            "# put PDFs / images / text files in ./inputs\n"
            "docker compose up --build\n"
            "```\n\n"
            "Results are written to `./outputs/results.json`.\n"
        )
        usage = (
            "This container is a headless analysis agent. It processes every "
            "file in `/app/inputs` with the pipeline settings baked in "
            f"(depth=`{spec.depth}`"
            + (f", query=`{spec.query}`" if spec.query else "")
            + ") and writes structural results as JSON."
        )
    src_note = ""
    if spec.source == "pypi":
        src_note = (
            "\n> **Note:** the image installs `rexgraph-agent` from PyPI. If the "
            "packages aren't published in your environment, regenerate the bundle "
            "with `source=\"local\"` and drop the `rexgraph`/`rexgraph-agent` "
            "wheels into a `wheels/` folder next to the Dockerfile.\n"
        )
    model_note = ""
    if spec.model_url:
        model_note = (
            f"\nThe agent will call the LLM at `{spec.model_url}` for answer "
            "synthesis. Change it via `CHAT_MODEL_URL` in `.env`.\n"
        )
    else:
        model_note = (
            "\nNo LLM is configured, so answers are structural. Set "
            "`CHAT_MODEL_URL` in `.env` to point at any OpenAI-compatible "
            "server (vLLM, Ollama, LM Studio, TGI, …) for narrative synthesis.\n"
        )
    return f"""# {spec.name}

{usage}
{src_note}{model_note}
{run}
Authentication, the RCDB store, audit and session files persist under the
`rexgraph_data` Docker volume. Back up that volume together. Bind mounts for
pipeline inputs and outputs must be readable/writable by container UID 10001.
For TLS, mount your certificate/key and configure `REXGRAPH_TLS_CERT` and
`REXGRAPH_TLS_KEY` in `.env`; for a reverse proxy, configure only its actual
address in `RCF_FORWARDED_ALLOW_IPS`. The generated bundle requires validation
on the chosen container host and any external database/object provider.

## What's in this bundle

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the image (installs system BLAS + rexgraph-agent). |
| `docker-compose.yml` | One-command build & run. |
| `entrypoint.sh` | Container startup ({spec.mode} mode). |
| `rexgraph-agent.json` | The agent/pipeline configuration, baked in. |
| `.env.example` | Environment overrides (LLM URL, cache). |

Generated by `rexgraph agent.deploy`. Image build needs its configured package
and OS dependency sources. Local wheel builds require every transitive wheel in
`wheels/`; installation failures stop the build. Runtime integrations contact
only the services you configure. The Agent UI and its licensed React runtime are
packaged for offline startup; incomplete installations report a reinstall error.
"""


# bundle assembly

def generate_bundle(spec: DeploymentSpec) -> dict[str, str]:
    """Return {filename: text_content} for the full deployment bundle."""
    spec = spec.normalized()
    return {
        "Dockerfile": _dockerfile(spec),
        "docker-compose.yml": _compose(spec),
        "entrypoint.sh": _entrypoint(spec),
        "rexgraph-agent.json": _config_json(spec),
        ".dockerignore": _dockerignore(),
        ".env.example": _env_example(spec),
        "README.md": _readme(spec),
    }


def bundle_to_zip(bundle: dict[str, str]) -> bytes:
    """Zip a bundle dict into bytes (for download)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in bundle.items():
            info = zipfile.ZipInfo(name)
            # make entrypoint executable inside the zip
            info.external_attr = (0o755 << 16) if name.endswith(".sh") else (0o644 << 16)
            zf.writestr(info, content)
    return buf.getvalue()


def write_bundle(bundle: dict[str, str], out_dir: str) -> str:
    """Write a bundle to a directory on disk. Returns the directory."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for name, content in bundle.items():
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write(content)
        if name.endswith(".sh"):
            os.chmod(path, 0o755)
    return out_dir


def spec_from_dict(d: dict) -> DeploymentSpec:
    """Build a DeploymentSpec from a loosely typed dict (API/CLI input)."""
    fields = {f for f in DeploymentSpec.__dataclass_fields__}
    return DeploymentSpec(**{k: v for k, v in (d or {}).items() if k in fields})
