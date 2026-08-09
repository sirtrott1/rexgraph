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

    def _rex_headers(self, body: bytes | None = None) -> dict:
        from rexgraph.protocol import CONTENT_TYPE, sign
        h = self._headers()
        if body is not None:
            h["Content-Type"] = CONTENT_TYPE
            if self.frame_key is not None:
                h["X-Rex-Signature"] = sign(body, self.frame_key)
        return h

    def _check_reply(self, response) -> bytes:
        """The body of a binary reply, refused unless it is signed as expected.

        Both directions or neither: a client that authenticates what it sends and
        accepts anything back is still talking to whoever is in the path.
        """
        from rexgraph.protocol import verify_signature
        body = response.content
        if self.frame_key is not None and not verify_signature(
                body, response.headers.get("X-Rex-Signature", ""), self.frame_key):
            raise ValueError(
                "the server's reply is unsigned or its signature does not match; "
                "the response was altered in transit or the keys differ")
        return body

    def rex_hello(self) -> dict:
        """What the server speaks and what it will not exceed. Read this first."""
        return self._get("/rex/v1/hello")

    def rex_verify(self, rex) -> dict:
        """Ask the server whether a complex is well-formed, without storing it."""
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
