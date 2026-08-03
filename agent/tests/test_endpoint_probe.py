"""Live-endpoint probing (local_runtime.probe_endpoints) + monitor graph edges."""
from agent.agent_complex import AgentComplex

from agent import local_runtime


def test_probe_parses_openai_and_ollama(monkeypatch):
    # two fake live servers: an OpenAI-compatible one and an Ollama one
    class FakeResp:
        def __init__(self, payload):
            self._p = payload
            self.status_code = 200

        def json(self):
            return self._p

    served = {
        "http://127.0.0.1:8000/v1/models": {"data": [{"id": "Qwen2.5-7B"}, {"id": "phi-4"}]},
        "http://127.0.0.1:11434/api/tags": {"models": [{"name": "llama3.2:3b"}]},
    }

    class FakeHTTPX:
        @staticmethod
        def get(url, timeout=0.4):
            if url in served:
                return FakeResp(served[url])
            raise OSError("connection refused")

    import sys
    monkeypatch.setitem(sys.modules, "httpx", FakeHTTPX)
    monkeypatch.setenv("REXGRAPH_PROBE_URLS", "")

    live = local_runtime.probe_endpoints()
    by_url = {e["url"]: e for e in live}
    assert "http://127.0.0.1:8000" in by_url
    assert by_url["http://127.0.0.1:8000"]["models"] == ["Qwen2.5-7B", "phi-4"]
    assert by_url["http://127.0.0.1:11434"]["kind"] == "ollama"
    assert by_url["http://127.0.0.1:11434"]["models"] == ["llama3.2:3b"]


def test_probe_respects_extra_urls(monkeypatch):
    monkeypatch.setenv("REXGRAPH_PROBE_URLS", "http://box.local:9000,http://gpu:8899")
    urls = [t["url"] for t in local_runtime._default_probe_targets()]
    assert "http://box.local:9000" in urls and "http://gpu:8899" in urls


def test_monitor_emits_directed_graph_edges():
    ac = AgentComplex()
    ac.add_messages([
        {"from": "user", "to": "router", "text": "route this task now"},
        {"from": "router", "to": "bio", "text": "biology subtask for you"},
        {"from": "bio", "to": "router", "text": "here is the biology result"},
    ])
    mon = ac.monitor()
    assert "edges" in mon and mon["edges"]
    e = {(x["from"], x["to"]): x["weight"] for x in mon["edges"]}
    assert e[("router", "bio")] == 1 and e[("bio", "router")] == 1
    # every edge references a real agent that also appears as a node
    names = {a["agent"] for a in mon["agents"]}
    for x in mon["edges"]:
        assert x["from"] in names and x["to"] in names
