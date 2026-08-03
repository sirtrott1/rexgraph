"""
agent.cli.serve: localhost vLLM/SGLang process management.

    rexgraph-ocr serve                    # start with defaults
    rexgraph-ocr serve --port 10000       # custom port
    rexgraph-ocr serve --backend sglang   # SGLang instead of vLLM
    rexgraph-ocr serve --stop             # stop the running server
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import (
    clear_pid,
    load_config,
    process_alive,
    read_pid,
    save_config,
    save_pid,
)

logger = logging.getLogger(__name__)

DEFAULT_PORT = 10000
# A first run downloads a multi-GB model AND loads it onto the GPU via vLLM; 120s reported a false
# TIMEOUT while the server was still legitimately loading. Env-configurable; 15 min covers a cold
# download. (A genuinely broken model still fails fast via the process-died check below.)
HEALTH_TIMEOUT = int(os.environ.get("REXGRAPH_OCR_HEALTH_TIMEOUT", "900"))
HEALTH_INTERVAL = 2      # seconds between health checks


def serve(
    port: int = DEFAULT_PORT,
    model: str | None = None,
    backend: str = "vllm",
    extra_args: list | None = None,
    foreground: bool = False,
) -> bool:
    """Start the GPU inference server.

    If foreground, blocks until interrupted. Otherwise forks and
    waits for health check.
    """
    # Check for already running server
    existing = find_running_server()
    if existing:
        pid_data = read_pid()
        pid = pid_data.get("pid", "?") if pid_data else "?"
        print(f"Server already running on {existing} (PID {pid})")
        print("Stop it first: rexgraph-ocr serve --stop")
        return False

    # Resolve model path
    cfg = load_config()
    if model is None:
        model = cfg.gpu_model_path or cfg.gpu_model
    if not model:
        # Default to DeepSeek-OCR-2 if no model configured
        model = "deepseek-ai/DeepSeek-OCR-2"
        print(f"No model configured - defaulting to {model}")
        print("First run downloads the model (several GB) and loads it onto the GPU - this can take")
        print("a few minutes. Progress is shown below; run `rexgraph-ocr setup` to pick another model.")

    # Build the command
    if backend == "vllm":
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--trust-remote-code",
            "--max-model-len", "32768",
        ]
    elif backend == "sglang":
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(port),
            "--trust-remote-code",
        ]
    else:
        print(f"Unknown backend: {backend}. Use 'vllm' or 'sglang'.")
        return False

    if extra_args:
        cmd.extend(extra_args)

    print(f"Starting {backend} server on port {port}...")
    print(f"  Model: {model}")
    print(f"  Command: {' '.join(cmd)}")

    if foreground:
        # Blocking: run in the current process
        try:
            proc = subprocess.Popen(cmd)
            save_pid(proc.pid, port, backend)
            proc.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            proc.terminate()
            proc.wait(timeout=10)
            clear_pid()
        return True

    # Background: fork and wait for health
    try:
        log_file = Path(cfg.cache_dir) / "server.log"
        log_fd = open(log_file, "a")

        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,   # detach from terminal
        )
        save_pid(proc.pid, port, backend)

        print(f"  PID: {proc.pid}")
        print(f"  Log: {log_file}")
        print("  Waiting for health check...", end="", flush=True)

        if wait_for_health(f"http://localhost:{port}", timeout=HEALTH_TIMEOUT, log_file=log_file):
            print(" ready")
            print(f"\nServer running on http://localhost:{port}")

            # Update config
            cfg.gpu_server_port = port
            cfg.gpu_server_backend = backend
            save_config(cfg)
            return True
        else:
            print(" TIMEOUT")
            print("Server may still be loading. Check: rexgraph-ocr status")
            print(f"Logs: tail -f {log_file}")
            return False

    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        print(f"Is {backend} installed? Run: pip install {backend}")
        clear_pid()
        return False
    except Exception as e:
        print(f"Failed to start server: {e}")
        clear_pid()
        return False


def stop() -> bool:
    """Stop the running GPU server."""
    pid_data = read_pid()
    if not pid_data:
        print("No server PID file found.")
        return False

    pid = pid_data.get("pid")
    port = pid_data.get("port", "?")

    if not pid or not process_alive(pid):
        print(f"Server (PID {pid}) is not running. Cleaning up.")
        clear_pid()
        return True

    print(f"Stopping server (PID {pid}, port {port})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 10 seconds for graceful shutdown
        for _ in range(20):
            if not process_alive(pid):
                break
            time.sleep(0.5)
        else:
            # Force kill
            print("Forcing shutdown...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
    except ProcessLookupError:
        pass

    clear_pid()
    print("Server stopped.")
    return True


def find_running_server() -> str | None:
    """Check for a running server and return its URL, or None."""
    pid_data = read_pid()
    if not pid_data:
        return None

    pid = pid_data.get("pid")
    port = pid_data.get("port", DEFAULT_PORT)

    if pid and process_alive(pid):
        return f"http://localhost:{port}"

    # PID file exists but process is dead - clean up
    clear_pid()
    return None


def _log_tail(log_file, n=8):
    """The last n non-empty lines of the server log (for visibility during/after startup)."""
    if not log_file:
        return []
    try:
        lines = [l.rstrip() for l in Path(log_file).read_text(errors="replace").splitlines() if l.strip()]
        return lines[-n:]
    except Exception:
        return []


def wait_for_health(
    base_url: str,
    timeout: int = HEALTH_TIMEOUT,
    interval: int = HEALTH_INTERVAL,
    log_file=None,
) -> bool:
    """Poll /health and /v1/models until the server responds 200. Surfaces the server log (what it
    is actually doing - downloading, loading weights) instead of a wall of dots."""
    import urllib.error
    import urllib.request

    endpoints = [f"{base_url}/health", f"{base_url}/v1/models"]
    deadline = time.time() + timeout
    next_progress = time.time() + 10
    last_shown = ""
    while time.time() < deadline:
        for url in endpoints:
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, OSError, TimeoutError):
                pass

        # A broken model / crashed server fails fast here (not the full timeout)
        pid_data = read_pid()
        if pid_data and pid_data.get("pid") and not process_alive(pid_data["pid"]):
            print("\n  Server exited during startup. Last log lines:")
            for l in _log_tail(log_file):
                print("    " + l[:140])
            return False

        # every ~10s show the latest log line so you can see progress, not just dots
        if time.time() >= next_progress:
            tail = _log_tail(log_file, 1)
            msg = tail[0][:110] if tail else ""
            if msg and msg != last_shown:
                print("\n  " + msg, end="", flush=True)
                last_shown = msg
            else:
                print(".", end="", flush=True)
            next_progress = time.time() + 10
        time.sleep(interval)

    print(f"\n  Still loading after {timeout}s (raise REXGRAPH_OCR_HEALTH_TIMEOUT if needed). Last log:")
    for l in _log_tail(log_file):
        print("    " + l[:140])
    return False


def server_status() -> dict:
    """Return a status dict for the GPU server."""
    pid_data = read_pid()
    if not pid_data:
        return {"status": "stopped", "pid": None, "port": None}

    pid = pid_data.get("pid")
    port = pid_data.get("port", DEFAULT_PORT)
    backend = pid_data.get("backend", "")

    if not pid or not process_alive(pid):
        clear_pid()
        return {"status": "stopped", "pid": pid, "port": port}

    # Check health
    url = f"http://localhost:{port}"
    healthy = False
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url}/health", timeout=3) as resp:
            healthy = resp.status == 200
    except Exception:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=3) as resp:
                healthy = resp.status == 200
        except Exception:
            pass

    return {
        "status": "healthy" if healthy else "running",
        "pid": pid,
        "port": port,
        "url": url,
        "backend": backend,
    }


def main(argv=None) -> int:
    """CLI entry: rexgraph-serve start|stop|status."""
    import argparse
    import json as _json

    p = argparse.ArgumentParser(
        prog="rexgraph-serve",
        description="Manage the RexGraph GPU inference/OCR server.",
    )
    sub = p.add_subparsers(dest="command")
    ps = sub.add_parser("start", help="Start the server")
    ps.add_argument("--port", type=int, default=DEFAULT_PORT)
    ps.add_argument("--model", default=None)
    ps.add_argument("--backend", default="vllm")
    ps.add_argument("--foreground", action="store_true")
    sub.add_parser("stop", help="Stop the server")
    sub.add_parser("status", help="Show server status")

    args = p.parse_args(argv)
    if args.command == "start":
        return 0 if serve(port=args.port, model=args.model,
                          backend=args.backend,
                          foreground=args.foreground) else 1
    if args.command == "stop":
        return 0 if stop() else 1
    if args.command == "status":
        print(_json.dumps(server_status(), indent=2))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
