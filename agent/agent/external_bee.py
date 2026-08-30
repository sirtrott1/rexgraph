"""
agent.external_bee: an OpenAI-compatible endpoint whose replies come from a caller that polls.

`Hive.attach` takes any endpoint speaking `/v1/chat/completions`, and `_chat_full` is the only
way a chat bee is ever invoked. That contract assumes the responder holds a listening socket,
which rules out a responder that can call out but cannot be called: a CLI harness, a notebook, a
person at a terminal. This module inverts the direction without changing the contract. The hive
still POSTs a completion and blocks on the reply it always blocked on, while the responder polls
`/agent/next` for work and answers on `/agent/reply`, so the side with no inbound socket is the
side that connects.

Standing one up pulls in no web framework: it is stdlib `http.server`, a thread per request, so
the swarm keeps the property of running without the server layer present.

    from agent import external_bee
    broker = external_bee.serve(port=8799)
    hive.attach("claude", broker.url, role="worker", specialties=["topology"])

An unanswered request is a timeout rather than a hang. `reply_timeout` is set below the hive's
own 120s call timeout so the broker is the one that gives up first, returning 504; `_chat_full`
reads that as None and the hive routes elsewhere, which is the same fail-soft path an unreachable
bee already takes.

`token_ref` names an environment variable or secret-store entry, never a credential. It is
resolved per request through `agent.secrets.resolve_ref` and guards both faces of the broker, so
the reference a `Bee.api_key_ref` carries is the one the responder presents back.

Every request, reply and timeout is recorded through `agent.activity`, so the exchange lands in
the journal any other local process already tails.
"""
from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

# Below the 120s default of hive._chat_full, so a request that nobody claims ends as this
# broker's 504 rather than as the caller's read timeout. The distinction matters to the hive:
# a 504 names a bee that is up and idle, a read timeout names one that may be wedged.
DEFAULT_REPLY_TIMEOUT = 110.0

# How long GET /agent/next blocks before reporting an empty queue. Long enough that an idle
# responder is not spinning, short enough to stay inside any proxy's idle ceiling.
DEFAULT_POLL_WAIT = 25.0

MAX_BODY_BYTES = 8 * 1024 * 1024


@dataclass
class Pending:
    """One completion the hive is blocked on, and the slot its answer lands in."""
    id: str
    messages: list
    model: str
    params: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    done: threading.Event = field(default_factory=threading.Event)
    reply: dict | None = None
    claimed_at: float | None = None

    def public(self) -> dict:
        """What the responder is handed. `messages` is verbatim, so a tool loop that the hive
        opened arrives with its assistant turn and its `role: tool` results intact."""
        return {"id": self.id, "model": self.model, "messages": self.messages,
                "params": self.params, "created": self.created,
                "waited": round(time.time() - self.created, 3)}


class Broker:
    """The queue behind the endpoint: completions in from the hive, answers in from a responder.

    One lock guards `_pending`; the handoff itself rides on a `queue.Queue` for the claim side
    and a per-request `Event` for the answer side, so a request thread blocks on exactly the one
    event that will be set for it and no thread scans the table."""

    def __init__(self, *, name: str = "external", model: str = "external",
                 token_ref: str = "", reply_timeout: float = DEFAULT_REPLY_TIMEOUT,
                 poll_wait: float = DEFAULT_POLL_WAIT):
        self.name = name
        self.model = model
        self.token_ref = token_ref
        self.reply_timeout = float(reply_timeout)
        self.poll_wait = float(poll_wait)
        self._pending: dict[str, Pending] = {}
        self._ready: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._served = 0
        self._timed_out = 0

    #### the credential, resolved per call and never held
    def token(self) -> str:
        if not self.token_ref:
            return ""
        from .secrets import resolve_ref
        return resolve_ref(self.token_ref)

    def authorized(self, header: str) -> bool:
        """True when no token is configured, or when the header carries it. An unresolvable
        reference leaves the broker OPEN rather than bricked: the same choice `_chat_full` makes
        when it declines to send an unresolvable reference as a bearer token."""
        want = self.token()
        if not want:
            return True
        got = header[7:].strip() if header[:7].lower() == "bearer " else ""
        return got == want

    #### the hive's side: submit and block
    def submit(self, messages: list, *, model: str = "", params: dict | None = None) -> Pending:
        p = Pending(id="req_" + uuid.uuid4().hex[:16], messages=list(messages),
                    model=model or self.model, params=dict(params or {}))
        with self._lock:
            self._pending[p.id] = p
        self._ready.put(p.id)
        _record(self.name, "request", {"id": p.id, "n_messages": len(p.messages)})
        return p

    def wait(self, p: Pending) -> dict | None:
        """Block until the responder answers or the deadline passes. Returns None on timeout,
        and drops the request so a late answer has nothing to land on."""
        if p.done.wait(self.reply_timeout):
            with self._lock:
                self._pending.pop(p.id, None)
                self._served += 1
            return p.reply
        with self._lock:
            self._pending.pop(p.id, None)
            self._timed_out += 1
        _record(self.name, "timeout", {"id": p.id, "after_s": self.reply_timeout})
        return None

    #### the responder's side: claim and answer
    def claim(self, wait: float | None = None) -> Pending | None:
        """Take the next unanswered request, or None once `wait` seconds pass with the queue
        empty. Ids of requests that already timed out are skipped rather than handed out."""
        deadline = time.monotonic() + (self.poll_wait if wait is None else float(wait))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                rid = self._ready.get(timeout=remaining)
            except queue.Empty:
                return None
            with self._lock:
                p = self._pending.get(rid)
            if p is None:
                continue                            # timed out while queued, nothing to answer
            p.claimed_at = time.time()
            _record(self.name, "claim", {"id": p.id})
            return p

    def answer(self, request_id: str, reply: dict) -> bool:
        """Fill a claimed request and release the thread waiting on it. False when the id is
        unknown, which is what a caller sees after its request timed out."""
        with self._lock:
            p = self._pending.get(request_id)
        if p is None:
            return False
        p.reply = reply
        p.done.set()
        _record(self.name, "reply", {"id": request_id,
                                     "chars": len(reply.get("content") or ""),
                                     "tool_calls": len(reply.get("tool_calls") or [])})
        return True

    def status(self) -> dict:
        with self._lock:
            waiting = [p.public() for p in self._pending.values()]
        return {"name": self.name, "model": self.model, "waiting": len(waiting),
                "served": self._served, "timed_out": self._timed_out,
                "reply_timeout": self.reply_timeout, "pending": waiting}


def _record(name: str, action: str, detail: dict) -> None:
    """Journal one exchange. Recording must never break the exchange it describes."""
    try:
        from . import activity
        activity.record("worker:" + name, action, scope="worker", detail=detail)
    except Exception:
        logger.debug("activity record failed for %s/%s", name, action, exc_info=True)


def completion_payload(broker: Broker, reply: dict, model: str) -> dict:
    """The reply in the shape `_chat_full` parses: content, tool_calls, finish_reason and the
    reasoning trace, under one choice. Absent fields are omitted rather than sent as null, since
    a backend that never reasons should not appear to have returned an empty trace."""
    message: dict = {"role": "assistant", "content": reply.get("content")}
    if reply.get("tool_calls"):
        message["tool_calls"] = reply["tool_calls"]
    if reply.get("reasoning_content"):
        message["reasoning_content"] = reply["reasoning_content"]
    finish = reply.get("finish_reason") or ("tool_calls" if reply.get("tool_calls") else "stop")
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


class _Handler(BaseHTTPRequestHandler):
    """Routes for both faces. `broker` is set on the server object by `serve`."""

    protocol_version = "HTTP/1.1"
    server_version = "rexgraph-external-bee"

    @property
    def broker(self) -> Broker:
        return self.server.broker            # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        logger.debug("%s %s", self.address_string(), fmt % args)

    #### helpers
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if n > MAX_BODY_BYTES:
            self._drain()                        # refused, but the socket still has to be framed
            return None
        if n <= 0:
            return None
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, OSError):
            return None

    def _drain(self) -> None:
        """Consume the request body without using it. Keep-alive frames one request after the
        next on the same socket, so a body left unread is parsed as the next request line."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        remaining = min(n, MAX_BODY_BYTES)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                return
            remaining -= len(chunk)

    def _guard(self) -> bool:
        if self.broker.authorized(self.headers.get("Authorization", "")):
            return True
        self._drain()
        self._send(401, {"error": {"message": "unauthorized", "type": "invalid_request_error"}})
        return False

    def _query(self, key: str, default: float) -> float:
        _, _, qs = self.path.partition("?")
        for part in qs.split("&"):
            k, _, v = part.partition("=")
            if k == key:
                try:
                    return float(v)
                except ValueError:
                    return default
        return default

    #### routes
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/health":
            return self._send(200, {"ok": True, **self.broker.status()})
        if not self._guard():
            return None
        if route == "/v1/models":
            # what local_runtime.probe_endpoints reads, so attach_live() finds this endpoint
            return self._send(200, {"object": "list",
                                    "data": [{"id": self.broker.model, "object": "model"}]})
        if route == "/agent/next":
            p = self.broker.claim(self._query("wait", self.broker.poll_wait))
            if p is None:
                return self._send(204, {})
            return self._send(200, p.public())
        if route == "/agent/status":
            return self._send(200, self.broker.status())
        return self._send(404, {"error": {"message": "no such route", "type": "not_found"}})

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if not self._guard():
            return None
        body = self._body()
        if body is None:
            return self._send(400, {"error": {"message": "unreadable body",
                                              "type": "invalid_request_error"}})
        if route == "/v1/chat/completions":
            return self._completions(body)
        if route == "/agent/reply":
            rid = str(body.get("id") or "")
            if not rid:
                return self._send(400, {"error": {"message": "need 'id'",
                                                  "type": "invalid_request_error"}})
            ok = self.broker.answer(rid, body)
            # A late answer is not an error the responder can act on, so it reads as
            # accepted-but-dropped rather than as a failure to retry.
            return self._send(200 if ok else 409, {"ok": ok, "id": rid})
        return self._send(404, {"error": {"message": "no such route", "type": "not_found"}})

    def _completions(self, body: dict):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._send(400, {"error": {"message": "need 'messages'",
                                              "type": "invalid_request_error"}})
        params = {k: body[k] for k in ("max_tokens", "temperature", "tools", "tool_choice")
                  if k in body}
        p = self.broker.submit(messages, model=str(body.get("model") or ""), params=params)
        reply = self.broker.wait(p)
        if reply is None:
            return self._send(504, {"error": {"message": "no responder answered in time",
                                              "type": "timeout", "request_id": p.id}})
        return self._send(200, completion_payload(self.broker, reply, p.model))


def render_prompt(messages: list) -> str:
    """A completion request as one piece of text for a command that reads stdin.

    A tool-calling harness wants the structured messages and gets them from
    `/agent/next` untouched; a command line wants a transcript. Roles are kept as labels
    rather than dropped, because a system turn that arrives indistinguishable from the
    user's is a different request from the one the hive sent."""
    out = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content")
        if content is None and m.get("tool_calls"):
            content = json.dumps(m["tool_calls"])
        out.append(f"[{role}] {content}" if role != "user" else str(content))
    return "\n\n".join(out)


def respond_with(url: str, command: str, *, token: str = "", stop=None,
                 wait: float = 25.0, timeout: float = 300.0, cwd: str | None = None) -> None:
    """Answer this broker's work by running a command, so any CLI becomes a bee.

    The generalisation the broker was built for. A responder that can poll but not listen
    is the case; a subprocess that reads a prompt on stdin and writes an answer on stdout
    is the commonest shape of one, and naming the command at the call site means the hive
    gains a member without this module knowing which vendor it is.

    Runs until `stop` is set. A command that fails is answered with its stderr rather than
    left hanging: the hive's own timeout would eventually fire, but it would report a bee
    that is up and idle when in fact its command is broken, and those want different
    fixing. The command is run WITHOUT a shell.
    """
    import shlex
    import subprocess
    import urllib.error
    import urllib.request
    argv = shlex.split(command)
    auth = {"Authorization": f"Bearer {token}"} if token else {}

    def _call(path, data=None):
        req = urllib.request.Request(url.rstrip("/") + path, data=data, headers={
            **auth, **({"Content-Type": "application/json"} if data else {})})
        with urllib.request.urlopen(req, timeout=wait + 10) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else {})

    while not (stop is not None and stop.is_set()):
        try:
            status, job = _call(f"/agent/next?wait={wait}")
        except (urllib.error.URLError, OSError, ValueError):
            logger.debug("responder could not reach %s", url, exc_info=True)
            if stop is not None and stop.wait(1.0):
                return
            continue
        if status != 200 or not job.get("id"):
            continue
        try:
            done = subprocess.run(argv, input=render_prompt(job.get("messages") or []),
                                  capture_output=True, text=True, timeout=timeout, cwd=cwd)
            content = done.stdout.strip() or (done.stderr.strip()[-2000:] or "(no output)")
        except subprocess.TimeoutExpired:
            content = f"(the command did not finish inside {timeout:.0f}s)"
        except OSError as e:
            content = f"(the command could not run: {e})"
        with contextlib.suppress(Exception):
            _call("/agent/reply", json.dumps({"id": job["id"], "content": content}).encode())


class BrokerServer:
    """A running broker and the thread serving it."""

    def __init__(self, broker: Broker, httpd: ThreadingHTTPServer, thread: threading.Thread):
        self.broker = broker
        self.httpd = httpd
        self.thread = thread

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self, timeout: float = 5.0) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout)


def serve(*, host: str = "127.0.0.1", port: int = 0, broker: Broker | None = None,
          **kw) -> BrokerServer:
    """Start a broker on a daemon thread and return the handle. `port=0` takes any free port,
    which is what the tests use; the default host is loopback because the endpoint answers for
    whoever polls it, and that is not a decision to expose to a network by accident."""
    b = broker or Broker(**kw)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    httpd.broker = b                                 # type: ignore[attr-defined]
    t = threading.Thread(target=httpd.serve_forever, name="external-bee", daemon=True)
    t.start()
    _record(b.name, "serve", {"url": f"http://{host}:{httpd.server_address[1]}"})
    return BrokerServer(b, httpd, t)


def main(argv=None):
    """CLI: `python -m agent.external_bee --port 8799 --name claude`."""
    import argparse
    ap = argparse.ArgumentParser(prog="rexgraph-bee", description=(
        "An OpenAI-compatible endpoint answered by a process that polls it."))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--name", default="external", help="bee name used in the activity journal")
    ap.add_argument("--model", default="external", help="model id reported to the hive")
    ap.add_argument("--token-ref", default="",
                    help="env var or secret-store name holding the bearer token, never the token")
    ap.add_argument("--reply-timeout", type=float, default=DEFAULT_REPLY_TIMEOUT)
    ap.add_argument("--exec", dest="exec_cmd", default="",
                    help="answer this bee by running a command that reads the prompt on "
                         "stdin, e.g. 'codex exec --skip-git-repo-check -'. Without it the "
                         "endpoint waits for something else to poll it.")
    ap.add_argument("--exec-cwd", default=None, help="working directory for --exec")
    a = ap.parse_args(argv)
    srv = serve(host=a.host, port=a.port, name=a.name, model=a.model,
                token_ref=a.token_ref, reply_timeout=a.reply_timeout)
    print(f"external bee {a.name!r} on {srv.url}")
    print(f"  hive side:      POST {srv.url}/v1/chat/completions")
    print(f"  responder side: GET  {srv.url}/agent/next   POST {srv.url}/agent/reply")
    stop = threading.Event()
    if a.exec_cmd:
        print(f"  answered by:    {a.exec_cmd}")
        threading.Thread(target=respond_with, name="bee-exec", daemon=True,
                         kwargs={"url": srv.url, "command": a.exec_cmd,
                                 "token": srv.broker.token(), "stop": stop,
                                 "cwd": a.exec_cwd}).start()
    try:
        srv.thread.join()
    except KeyboardInterrupt:
        stop.set()
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
