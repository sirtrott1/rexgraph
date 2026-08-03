"""Tests for the pluggable secret storage."""


from agent.secrets import EnvSecretStore, FileSecretStore, mask_uri, open_secret_store


def test_mask_hides_password():
    assert "secret" not in mask_uri("postgresql://u:secret@h/db")
    assert "****" in mask_uri("postgresql://u:secret@h/db")


def test_file_store_roundtrip_and_masked_list(tmp_path):
    s = FileSecretStore(str(tmp_path / "c.json"))
    s.put("prod", "postgresql://u:secret@h/db", "sql")
    assert s.get("prod") == "postgresql://u:secret@h/db"     # raw for resolve
    listed = s.list()
    assert listed[0]["name"] == "prod"
    assert all("secret" not in c["uri"] for c in listed)     # never leaks creds
    assert s.delete("prod") and not s.delete("prod")


def test_env_store_resolves_reference(tmp_path, monkeypatch):
    s = EnvSecretStore(str(tmp_path / "refs.json"))
    s.put("prod", "MY_DB_URI", "sql")                        # stores a reference
    monkeypatch.setenv("MY_DB_URI", "postgresql://u:secret@h/db")
    assert s.get("prod") == "postgresql://u:secret@h/db"     # fetched from env
    # the reference, not the secret, is what's stored/listed
    assert s.list()[0]["uri"] == "ref:MY_DB_URI"


def test_open_secret_store_selects_backend(tmp_path):
    assert isinstance(open_secret_store("file://" + str(tmp_path / "x.json")),
                      FileSecretStore)
    assert isinstance(open_secret_store("env://"), EnvSecretStore)
