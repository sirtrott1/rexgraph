"""agent.hive: tool-calling and authenticated bees on the chat path.

The hive's only generation path is `hive._chat`. For a tool-driving harness it must be able to
(a) send `tools`/`tool_choice` and read back `tool_calls`/`finish_reason`/`reasoning_content`, and
(b) authenticate to a remote provider without ever persisting the credential.

Every test here stubs the HTTP layer (`httpx.Client`), so what is asserted is the *wire payload*
and the *headers* actually sent - not a mock of our own function, which could not catch a
credential that never leaves the process or a `tools` list that is silently dropped.
"""
import json

import pytest

from agent import agent_complex, hive, secrets


@pytest.fixture(autouse=True)
def clean():
    hive.reset_hive()
    agent_complex.reset_live()
    yield
    hive.reset_hive()
    agent_complex.reset_live()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """Captures the request the hive actually puts on the wire."""

    def __init__(self, capture, payload, **kw):
        self._capture = capture
        self._payload = payload
        capture["client_kwargs"] = kw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self._capture["url"] = url
        self._capture["json"] = json
        self._capture["headers"] = headers or {}
        return _FakeResponse(self._payload)


def _stub_http(monkeypatch, response_payload):
    """Replace httpx.Client so no real request is made; returns the capture dict."""
    import httpx
    capture = {}
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: _FakeClient(capture, response_payload, **kw))
    return capture


_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    },
}]

_TOOL_CALL_RESPONSE = {
    "choices": [{
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning_content": "the user wants a file read",
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "read_file",
                             "arguments": '{"path": "/etc/hosts"}'},
            }],
        },
    }],
}

_TEXT_RESPONSE = {
    "choices": [{"finish_reason": "stop",
                 "message": {"role": "assistant", "content": "  plain answer  "}}],
}


# --- the structured path: tools go out, tool_calls come back -------------------

def test_chat_full_forwards_tools_and_parses_tool_calls(monkeypatch):
    cap = _stub_http(monkeypatch, _TOOL_CALL_RESPONSE)
    res = hive._chat_full("http://bee", "qwen-local", "Read /etc/hosts.",
                          tools=_TOOLS, tool_choice="auto")

    # the tools genuinely reached the wire
    assert cap["json"]["tools"] == _TOOLS
    assert cap["json"]["tool_choice"] == "auto"

    # and the structured reply was parsed rather than discarded
    assert res is not None
    assert res.finish_reason == "tool_calls"
    assert res.content is None
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(res.tool_calls[0]["function"]["arguments"]) == {"path": "/etc/hosts"}
    assert res.reasoning_content == "the user wants a file read"


def test_chat_full_forwards_chat_template_kwargs(monkeypatch):
    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    hive._chat_full("http://bee", "qwen-local", "hi",
                    chat_template_kwargs={"enable_thinking": False})
    assert cap["json"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_full_accepts_full_message_history(monkeypatch):
    """A tool loop must be able to feed the assistant turn and the tool result back."""
    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    msgs = [
        {"role": "user", "content": "Read /etc/hosts."},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_abc", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "127.0.0.1 localhost"},
    ]
    hive._chat_full("http://bee", "qwen-local", None, messages=msgs)
    assert cap["json"]["messages"] == msgs


def test_chat_full_omits_optional_fields_when_unset(monkeypatch):
    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    hive._chat_full("http://bee", "m", "hi")
    for k in ("tools", "tool_choice", "chat_template_kwargs"):
        assert k not in cap["json"]


# --- backwards compatibility: the text path is untouched ----------------------

def test_chat_text_path_still_returns_a_bare_string(monkeypatch):
    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    out = hive._chat("http://bee", "m", "hi", system="be brief")
    assert out == "plain answer"                       # stripped str, exactly as before
    assert isinstance(out, str)
    assert "tools" not in cap["json"]                   # no new keys on the legacy payload
    assert cap["json"]["messages"] == [{"role": "system", "content": "be brief"},
                                       {"role": "user", "content": "hi"}]


def test_chat_text_path_returns_none_on_empty(monkeypatch):
    _stub_http(monkeypatch, {"choices": [{"message": {"content": "   "}}]})
    assert hive._chat("http://bee", "m", "hi") is None


def test_ask_still_returns_a_string(monkeypatch):
    """hive.dispatch/collaborate/consensus/guarded_ask all regex or concatenate this value."""
    _stub_http(monkeypatch, _TEXT_RESPONSE)
    h = hive.get_hive()
    h.attach("lead", "http://bee", role="queen", model="m")
    assert h.ask("lead", "hi") == "plain answer"


def test_ask_full_returns_structured_result(monkeypatch):
    cap = _stub_http(monkeypatch, _TOOL_CALL_RESPONSE)
    h = hive.get_hive()
    h.attach("lead", "http://bee", role="queen", model="m")
    res = h.ask_full("lead", "Read /etc/hosts.", tools=_TOOLS, tool_choice="auto")
    assert cap["json"]["tools"] == _TOOLS
    assert res.finish_reason == "tool_calls"
    assert res.tool_calls[0]["function"]["name"] == "read_file"


# --- authentication: a credential is resolved per call, never stored ----------

def test_no_authorization_header_without_a_credential(monkeypatch):
    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    hive._chat("http://bee", "m", "hi")
    assert "Authorization" not in cap["headers"]


def test_chat_sends_bearer_token_resolved_from_env_reference(monkeypatch):
    monkeypatch.setenv("MY_PROVIDER_KEY", "sk-live-secret-123")
    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    hive._chat("http://bee", "m", "hi", api_key_ref="MY_PROVIDER_KEY")
    assert cap["headers"]["Authorization"] == "Bearer sk-live-secret-123"


def test_ask_sends_bearer_token_for_an_authenticated_bee(monkeypatch):
    monkeypatch.setenv("MY_PROVIDER_KEY", "sk-live-secret-123")
    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    h = hive.get_hive()
    h.attach("remote", "https://api.deepseek.com", role="queen", model="deepseek-chat",
             api_key_ref="MY_PROVIDER_KEY")
    h.ask("remote", "hi")
    assert cap["headers"]["Authorization"] == "Bearer sk-live-secret-123"


def test_unresolvable_reference_sends_no_header(monkeypatch):
    """A missing secret must not crash the hive nor send a literal 'Bearer <ref>'."""
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    hive._chat("http://bee", "m", "hi", api_key_ref="NOT_SET_ANYWHERE")
    assert "Authorization" not in cap["headers"]


def test_resolve_ref_reads_env_then_secret_store(monkeypatch, tmp_path):
    monkeypatch.setenv("FROM_ENV", "env-value")
    assert secrets.resolve_ref("FROM_ENV") == "env-value"
    assert secrets.resolve_ref("") == ""
    assert secrets.resolve_ref("MISSING_EVERYWHERE") == ""

    store = secrets.FileSecretStore(str(tmp_path / "s.json"))
    store.put("prov", "sk-from-store", kind="llm")
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", "file://" + str(tmp_path / "s.json"))
    assert secrets.resolve_ref("prov") == "sk-from-store"


# --- the credential must never appear anywhere the hive serializes -----------

_SECRET = "sk-live-DO-NOT-LEAK-9999"


def _leak_surfaces(h):
    """Everything that turns a bee into data a user/log/disk could see."""
    from agent import hive_config
    prof = hive_config.HiveProfile(
        id="p", name="p", compose="manual",
        bees=[hive_config.BeeSpec(name="remote", role="queen", source="attach",
                                  url="https://api.deepseek.com", model="deepseek-chat",
                                  api_key_ref="LEAK_TEST_KEY")])
    bee = h.get("remote")
    return {
        "public": bee.public(),
        "status": h.status(),
        "snapshot": {"workers": h.snapshot()["workers"]},
        "profile": prof.to_dict(),
        "bee_repr": repr(bee),
        "bee_dataclass_fields": sorted(bee.__dataclass_fields__),
    }


def test_credential_never_leaks_through_any_serialization(monkeypatch):
    monkeypatch.setenv("LEAK_TEST_KEY", _SECRET)
    h = hive.get_hive()
    h.attach("remote", "https://api.deepseek.com", role="queen", model="deepseek-chat",
             api_key_ref="LEAK_TEST_KEY")

    surfaces = _leak_surfaces(h)
    blob = json.dumps({k: v for k, v in surfaces.items() if k != "bee_repr"}, default=str)
    blob += surfaces["bee_repr"]

    # the resolved secret must appear in NONE of them
    assert _SECRET not in blob, "resolved credential leaked into a serialized surface"

    # the Bee must not even have a field that could hold a raw key
    assert not any("api_key" == f or f.endswith("_key")
                   for f in surfaces["bee_dataclass_fields"]), \
        "Bee has a field that can hold a raw credential; store a reference instead"

    # what IS exposed is only the boolean fact that a credential is configured
    assert surfaces["public"]["has_api_key"] is True
    assert "LEAK_TEST_KEY" not in json.dumps(surfaces["public"])


def test_public_reports_no_credential_for_a_plain_bee():
    h = hive.get_hive()
    h.attach("local", "http://127.0.0.1:8080", role="queen", model="m")
    assert h.get("local").public()["has_api_key"] is False


def test_resolved_secret_is_not_cached_on_the_bee(monkeypatch):
    """Resolution happens per call; rotating the secret must take effect immediately."""
    monkeypatch.setenv("ROTATING_KEY", "first")
    h = hive.get_hive()
    h.attach("remote", "http://bee", role="queen", model="m", api_key_ref="ROTATING_KEY")

    cap = _stub_http(monkeypatch, _TEXT_RESPONSE)
    h.ask("remote", "hi")
    assert cap["headers"]["Authorization"] == "Bearer first"

    monkeypatch.setenv("ROTATING_KEY", "second")
    h.ask("remote", "hi")
    assert cap["headers"]["Authorization"] == "Bearer second"
    assert _SECRET not in json.dumps(h.get("remote").public())
