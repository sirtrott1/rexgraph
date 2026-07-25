#!/usr/bin/env python3
"""
Start the RexGraph Agent (developer launcher).

    python run.py                                   # HTTP on localhost:8000
    python run.py --https                           # HTTPS with auto-generated cert
    python run.py --ssl-cert c.pem --ssl-key k.pem  # HTTPS with your cert

A thin wrapper that maps flags onto agent.server.launch.serve - the single
launch path also used by the `rcf-server` console script.
"""

import argparse

from agent.server.launch import serve


def main():
    parser = argparse.ArgumentParser(description="RexGraph Mathematical Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker process count (ignored with --reload)")
    parser.add_argument("--https", action="store_true",
                        help="Enable HTTPS (auto-generates cert if needed)")
    parser.add_argument("--ssl-cert", help="Path to TLS certificate")
    parser.add_argument("--ssl-key", help="Path to TLS private key")
    args = parser.parse_args()

    serve(
        host=args.host,
        port=args.port,
        reload=args.reload,
        https=args.https,
        ssl_cert=args.ssl_cert,
        ssl_key=args.ssl_key,
        workers=args.workers,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
