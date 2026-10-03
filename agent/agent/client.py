"""
agent.client: Python client for a running rexgraph server.

Usage in a Jupyter notebook:

    from agent.client import RexClient

    rc = RexClient("https://team-server:8000", api_key="...")

    # Upload and analyze
    result = rc.upload("contract.pdf")
    print(result["betti"])

    # Build a corpus
    rc.corpus_add_text("TSMC manufactures chips.", doc_id="supply")
    rc.corpus_build()
    hits = rc.corpus_query("semiconductor", mode="spectral")

    # Chat with context
    response = rc.chat(result["session_id"], "What are the voids?")

    # Export
    data = rc.export_session(result["session_id"])
"""

from __future__ import annotations


class RexClient:
    """Client for a running rexgraph agent server."""

    def __init__(
        self,
        url: str = "http://localhost:8000",
        api_key: str | None = None,
        workspace: str = "default",
        frame_key: bytes | str | None = None,
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.workspace = workspace
        if frame_key is None:
            import os
            frame_key = os.environ.get("REXGRAPH_FRAME_KEY") or None
        self.frame_key = (frame_key.encode("utf-8")
                          if isinstance(frame_key, str) else frame_key)
        self._session = None

    def _headers(self) -> dict:
        h = {"X-Workspace": self.workspace}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    #### the native surface: binary frames, signed when the deployment signs
    #
    # A server configured with REXGRAPH_FRAME_KEY refuses an unsigned frame, so a
    # client that cannot sign cannot talk to it at all. Signing is the caller's side of
    # that contract and belongs with the caller; picking the key up from the same
    # environment variable the server reads means a local operator gets it for free.

    def _rex_headers(self, body: bytes | None = None, *, content_type=None) -> dict:
        from rexgraph.protocol import CONTENT_TYPE, sign
        h = self._headers()
        if body is not None:
            h["Content-Type"] = content_type or CONTENT_TYPE
            if self.frame_key is not None:
                h["X-Rex-Signature"] = sign(body, self.frame_key)
        return h

    def _check_reply(self, response, body=None) -> bytes:
        """The body of a binary reply, refused unless it is signed as expected.

        Both directions or neither: a client that authenticates what it sends and
        accepts anything back is still talking to whoever is in the path.
        """
        from rexgraph.protocol import verify_signature
        body = response.content if body is None else body
        if self.frame_key is not None and not verify_signature(
                body, response.headers.get("X-Rex-Signature", ""), self.frame_key):
            raise ValueError(
                "the server's reply is unsigned or its signature does not match; "
                "the response was altered in transit or the keys differ")
        return body

    def _record_reply(self, method, path, *, limit, content_type, **kwargs):
        """Bound an actual response stream before authenticating and decoding it."""
        import httpx
        with httpx.stream(method, self.url+path, timeout=300, **kwargs) as response:
            if not response.is_success:
                # Keep useful per record refusal details without buffering an
                # unbounded error page or leaving an unread streaming response.
                detail = bytearray()
                for chunk in response.iter_bytes():
                    detail.extend(chunk[:max(0, 64*1024-len(detail))])
                    if len(detail) == 64*1024:
                        break
                error_headers = {k: v for k, v in response.headers.items()
                                 if k.lower() not in {"content-encoding", "content-length"}}
                httpx.Response(response.status_code, headers=error_headers, content=bytes(detail),
                               request=response.request).raise_for_status()
            response.raise_for_status()
            if response.headers.get("content-type", "").split(";", 1)[0] != content_type:
                raise ValueError("server returned an unexpected record content type")
            declared = response.headers.get("content-length")
            if declared is not None:
                if not declared.isascii() or not declared.isdecimal():
                    raise ValueError("server returned an invalid content length")
                if len(declared) > 20 or int(declared) > limit:
                    raise ValueError("server record response exceeds its byte limit")
            body = bytearray()
            for chunk in response.iter_bytes():
                if len(body)+len(chunk) > limit:
                    raise ValueError("server record response exceeds its byte limit")
                body.extend(chunk)
            # HTTPX yields decoded bytes; `Content-Length` names encoded bytes.
            encoded = response.headers.get("content-encoding", "identity").lower() != "identity"
            if declared is not None and not encoded and int(declared) != len(body):
                raise ValueError("server response length differs from its content length")
            return self._check_reply(response, bytes(body))

    def rex_hello(self) -> dict:
        """What the server speaks and what it will not exceed. Read this first."""
        return self._get("/rex/v1/hello")

    def rex_verify(self, rex) -> dict:
        """Ask the server whether a complex is well formed, without storing it."""
        import httpx

        from rexgraph.protocol import encode
        body = encode(rex)
        r = httpx.post(self.url + "/rex/v1/verify", content=body,
                       headers=self._rex_headers(body), timeout=120)
        r.raise_for_status()
        return r.json()

    def rex_store(self, rex, **meta) -> dict:
        """Keep a complex in this client's workspace. Returns its record id."""
        import httpx

        from rexgraph.protocol import encode
        body = encode(rex, meta=meta or None)
        r = httpx.post(self.url + "/rex/v1/store", content=body,
                       headers=self._rex_headers(body), timeout=300)
        r.raise_for_status()
        return r.json()

    def rex_fetch(self, record_id: str):
        """A stored complex back, rebuilt and verified on arrival."""
        import httpx

        from rexgraph.protocol import decode, to_complex
        r = httpx.get(f"{self.url}/rex/v1/fetch/{record_id}",
                      headers=self._headers(), timeout=300)
        r.raise_for_status()
        return to_complex(decode(self._check_reply(r)))

    def rex_store_record(self, packet, *, source=None, courier=None):
        """Publish a RecordPacket; verify the receipt's source and payload identity."""
        from rcdb import CopyReceipt, RecordPacket
        from rcdb.packet import PACKET_CONTENT_TYPE, RECEIPT_CONTENT_TYPE
        from rcdb.transfer import RECEIPT_LIMIT
        from urllib.parse import quote
        if not isinstance(packet, RecordPacket):
            raise TypeError("rex_store_record requires a RecordPacket")
        body = packet.to_bytes()
        headers = self._rex_headers(body, content_type=PACKET_CONTENT_TYPE)
        for name, value in (("X-Rex-Source-Hive", source), ("X-Rex-Courier", courier)):
            if value is not None:
                if type(value) is not str or len(value.encode("utf-8")) > 256:
                    raise ValueError("courier header requires bounded text")
                headers[name] = quote(value, safe="")
        raw = self._record_reply("POST", "/rex/v1/records/store", limit=RECEIPT_LIMIT,
                                 content_type=RECEIPT_CONTENT_TYPE, content=body, headers=headers)
        receipt = CopyReceipt.from_bytes(raw)
        record = packet.record
        if ((receipt.source_store_id, receipt.source_record_id, receipt.source_version, receipt.source_digest)
                != (packet.source_store_id, record.id, record.version, packet.state_digest)
                or receipt.destination_digest != packet.state_digest):
            raise ValueError("server receipt differs from the submitted record packet")
        return receipt

    def rex_fetch_record(self, record_id: str, *, version=None, as_of=None, valid_at=None):
        """Fetch a portable selected version; materialize with packet.snapshot()."""
        from rcdb import RecordPacket
        from rcdb.core import _read_selector
        from rcdb.packet import PACKET_CONTENT_TYPE, PACKET_LIMIT
        _read_selector(record_id, version, as_of, valid_at)
        params = {"record_id": record_id, **{k: v for k, v in (("version", version), ("as_of", as_of), ("valid_at", valid_at)) if v is not None}}
        raw = self._record_reply("GET", "/rex/v1/records/fetch", limit=PACKET_LIMIT,
                                 content_type=PACKET_CONTENT_TYPE, params=params, headers=self._headers())
        packet = RecordPacket.from_bytes(raw)
        record = packet.record
        if record.id != record_id or (version is not None and record.version != version):
            raise ValueError("server returned a different record address")
        if (as_of is not None and (record.tx_from > as_of or (record.tx_to is not None and as_of >= record.tx_to))
                or valid_at is not None and (record.valid_from is not None and valid_at < record.valid_from
                    or record.valid_to is not None and valid_at >= record.valid_to)):
            raise ValueError("server returned a record outside the requested interval")
        return packet

    def rex_upload(self, filepath: str) -> dict:
        """Put a file in this workspace and get the handle that names it."""
        import os

        import httpx
        with open(filepath, "rb") as fh:
            body = fh.read()
        h = self._headers()
        h["X-Filename"] = os.path.basename(filepath)
        r = httpx.post(self.url + "/rex/v1/upload", content=body, headers=h,
                       timeout=300)
        r.raise_for_status()
        return r.json()

    def rex_files(self) -> dict:
        """The handles this workspace holds."""
        return self._get("/rex/v1/files")

    def rex_tools(self) -> dict:
        """Every capability this caller may run, with its schema."""
        return self._get("/rex/v1/tools")

    def rex_call(self, name: str, **arguments) -> dict:
        """Run one capability. File arguments are handles from `rex_upload`."""
        return self._post("/rex/v1/call",
                          json={"name": name, "arguments": arguments})

    def rex_audit(self, limit: int = 200) -> dict:
        """This workspace's trail, and whether the chain still verifies."""
        return self._get("/rex/v1/audit", limit=limit)

    def _get(self, path: str, **params) -> dict:
        import httpx
        r = httpx.get(
            self.url + path, headers=self._headers(),
            params=params, timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, **kwargs) -> dict:
        import httpx
        r = httpx.post(
            self.url + path, headers=self._headers(),
            timeout=120, **kwargs,
        )
        r.raise_for_status()
        return r.json()

    # Health
    def health(self) -> dict:
        return self._get("/api/health")

    def status(self) -> dict:
        return self._get("/api/v1/status")

    # Upload
    def upload(self, filepath: str) -> dict:
        """Upload a file and build a relational complex."""
        import httpx
        with open(filepath, "rb") as f:
            files = {"file": (filepath.split("/")[-1], f)}
            data = {"options": "{}"}
            r = httpx.post(
                self.url + "/api/upload",
                headers=self._headers(), files=files, data=data,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()

    # Analysis
    def analysis(self, session_id: str, depth: str = "standard") -> dict:
        return self._get(f"/api/analysis/{session_id}", depth=depth)

    # Chat
    def chat(self, session_id: str, message: str) -> dict:
        return self._post(f"/api/chat/{session_id}", json={"message": message})

    # Corpus
    def corpus_add_text(self, text: str, doc_id: str = None, date: str = None) -> dict:
        import httpx
        data = {"text": text}
        if doc_id:
            data["doc_id"] = doc_id
        if date:
            data["date"] = date
        r = httpx.post(
            self.url + "/api/v1/corpus/add",
            headers=self._headers(), data=data, timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def corpus_add_file(self, filepath: str, doc_id: str = None) -> dict:
        import httpx
        with open(filepath, "rb") as f:
            files = {"file": (filepath.split("/")[-1], f)}
            data = {}
            if doc_id:
                data["doc_id"] = doc_id
            r = httpx.post(
                self.url + "/api/v1/corpus/add",
                headers=self._headers(), files=files, data=data,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()

    def corpus_build(self, depth: str = "standard") -> dict:
        import httpx
        r = httpx.post(
            self.url + "/api/v1/corpus/build",
            headers=self._headers(), data={"depth": depth},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    def corpus_query(self, query: str, mode: str = "hybrid", top_k: int = 5) -> dict:
        import httpx
        r = httpx.post(
            self.url + "/api/v1/corpus/query",
            headers=self._headers(),
            data={"query": query, "mode": mode, "top_k": str(top_k)},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def corpus_summary(self) -> dict:
        return self._get("/api/v1/corpus/summary")

    def corpus_temporal(self) -> dict:
        return self._get("/api/v1/corpus/temporal")

    def corpus_bridge(self, doc_a: int, doc_b: int) -> dict:
        return self._get("/api/v1/corpus/bridge/%d/%d" % (doc_a, doc_b))

    def corpus_reset(self) -> dict:
        return self._post("/api/v1/corpus/reset", json={})

    # Models
    def models_list(self) -> dict:
        return self._get("/api/v1/models/list")

    def models_pull(self, model_id: str) -> dict:
        return self._post("/api/v1/models/pull", json={"model_id": model_id})

    def models_deploy(self, model_id: str, port: int = 10000, backend: str = "vllm") -> dict:
        return self._post("/api/v1/models/deploy",
                          json={"model_id": model_id, "port": port, "backend": backend})

    def models_stop(self) -> dict:
        return self._post("/api/v1/models/stop", json={})

    # Model chat
    def generate(self, prompt: str, session_id: str = None,
                 context: str = None, max_tokens: int = 1024) -> dict:
        body = {"prompt": prompt, "max_tokens": max_tokens}
        if session_id:
            body["session_id"] = session_id
        if context:
            body["context"] = context
        return self._post("/api/v1/model/generate", json=body)

    # Pipeline
    def pipeline(self, filepaths: list[str], query: str = None) -> dict:
        import httpx
        files = [("files", (p.split("/")[-1], open(p, "rb"))) for p in filepaths]
        data = {}
        if query:
            data["query"] = query
        try:
            r = httpx.post(
                self.url + "/api/v1/pipeline/run",
                headers=self._headers(), files=files, data=data,
                timeout=600,
            )
            r.raise_for_status()
            return r.json()
        finally:
            for _, (_, fh) in files:
                fh.close()

    # Export
    def export_session(self, session_id: str, format: str = "json") -> dict:
        return self._get(f"/api/v1/export/session/{session_id}", format=format)

    def export_workspace(self, format: str = "json") -> dict:
        return self._get("/api/v1/export/workspace", format=format)

    def export_queries(self, limit: int = 50) -> dict:
        return self._get("/api/v1/export/queries", limit=limit)

    # Admin
    def create_token(self, user_id: str, workspaces: list[str] = None,
                     role: str = "write") -> dict:
        return self._post("/api/v1/admin/token",
                          json={"user_id": user_id,
                                "workspaces": workspaces or ["default"],
                                "role": role})

    def list_workspaces(self) -> dict:
        return self._get("/api/v1/admin/workspaces")

    def workspace_activity(self) -> dict:
        return self._get("/api/v1/admin/workspace/activity")

    def __repr__(self):
        return f"RexClient({self.url}, workspace={self.workspace})"
