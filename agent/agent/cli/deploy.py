"""
agent.cli.deploy - generate a deployable container bundle from the CLI.

    rexgraph-deploy --mode service --out ./deploy
    rexgraph-deploy --mode pipeline --query "key findings?" --backend tesseract --zip agent.zip
    rexgraph-deploy --config agent-config.json --mode pipeline --out ./my-agent
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.deploy import DeploymentSpec, generate_bundle, write_bundle, bundle_to_zip


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="rexgraph-deploy",
        description="Containerize a RexGraph agent or pipeline for deployment.")
    p.add_argument("--name", default="rexgraph-agent", help="image/container name")
    p.add_argument("--mode", choices=["service", "pipeline"], default="service",
                   help="service = full web app + API; pipeline = headless analysis agent")
    p.add_argument("--extras", default="",
                   help="comma-separated extras (server,ocr,training,langchain,...)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model-url", default="", help="OpenAI-compatible LLM URL")
    p.add_argument("--source", choices=["pypi", "local"], default="pypi")
    p.add_argument("--insecure", action="store_true",
                   help="run open (skip the secure default); only behind upstream auth or a firewall")
    # pipeline settings
    p.add_argument("--depth", default="standard")
    p.add_argument("--query", default="")
    p.add_argument("--backend", default="")
    p.add_argument("--ontology", action="store_true")
    p.add_argument("--config", help="agent-builder config JSON to embed")
    # output
    p.add_argument("--out", help="write bundle files to this directory")
    p.add_argument("--zip", dest="zip_path", help="write bundle as a .zip to this path")
    args = p.parse_args(argv)

    builder_config = None
    if args.config:
        try:
            with open(args.config) as f:
                builder_config = json.load(f)
        except Exception as e:
            print(f"error: could not read --config: {e}", file=sys.stderr)
            return 2

    spec = DeploymentSpec(
        name=args.name, mode=args.mode,
        extras=[e.strip() for e in args.extras.split(",") if e.strip()] or ["server"],
        port=args.port, model_url=args.model_url, source=args.source,
        insecure=args.insecure, depth=args.depth, query=args.query,
        backend=args.backend, ontology=args.ontology, builder_config=builder_config,
    )
    bundle = generate_bundle(spec)

    if not args.out and not args.zip_path:
        args.out = f"./{spec.normalized().name}-deploy"

    if args.out:
        path = write_bundle(bundle, args.out)
        print(f"Wrote {len(bundle)} files to {path}/")
        print(f"  cd {path} && docker compose up --build")
    if args.zip_path:
        with open(args.zip_path, "wb") as f:
            f.write(bundle_to_zip(bundle))
        print(f"Wrote bundle zip to {args.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
