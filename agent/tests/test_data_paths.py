"""Data configuration resolves consistently without creating or downloading resources."""
from pathlib import Path

import pytest
from agent.data_paths import cache_root, config_root, models_root, huggingface_cache_root
from agent.data_paths import lexical_path, lexical_root, lexical_status
from agent.data_paths import sessions_root


@pytest.fixture
def home(tmp_path, monkeypatch):
    for name in ("REXGRAPH_DATA_DIR", "REXGRAPH_LEXICAL_DIR", "XDG_DATA_HOME",
                 "REXGRAPH_WORDNET", "REXGRAPH_ROGET", "REXGRAPH_WIKTIONARY",
                 "REXGRAPH_CACHE_DIR", "REXGRAPH_CONFIG_DIR", "REXGRAPH_MODELS_DIR",
                 "REXGRAPH_MODEL_SCAN_DIRS",
                 "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "HF_HOME", "HF_HUB_CACHE",
                 "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "APPDATA", "LOCALAPPDATA",
                 "REXGRAPH_SESSION_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("agent.data_paths.platform.system", lambda: "Linux")
    return tmp_path


def test_new_user_data_location_is_not_a_source_checkout(home):
    root = home / ".local" / "share" / "rexgraph" / "lexical"
    assert lexical_root() == root
    assert not root.exists()
    assert not any(item["present"] for item in lexical_status().values())


def test_explicit_lexical_root_takes_precedence(home, monkeypatch):
    monkeypatch.setenv("REXGRAPH_LEXICAL_DIR", str(home / "explicit"))
    monkeypatch.setenv("REXGRAPH_DATA_DIR", str(home / "data"))
    assert lexical_root() == home / "explicit"
    assert lexical_path("roget") == home / "explicit" / "thesaurus" / "rogets-pg22.txt"


def test_data_root_contains_the_lexical_directory(home, monkeypatch):
    monkeypatch.setenv("REXGRAPH_DATA_DIR", str(home / "data"))
    assert lexical_root() == home / "data" / "lexical"


def test_xdg_configuration_does_not_fall_back_to_another_root(home, monkeypatch):
    legacy = home / "projects" / "rexgraph" / "data" / "lexical"
    legacy.mkdir(parents=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "configured"))
    assert lexical_root() == home / "configured" / "rexgraph" / "lexical"


def test_existing_legacy_data_remains_addressable(home):
    legacy = home / "projects" / "rexgraph" / "data" / "lexical"
    legacy.mkdir(parents=True)
    assert lexical_root() == legacy
    preferred = home / ".local" / "share" / "rexgraph" / "lexical"
    preferred.mkdir(parents=True)
    assert lexical_root() == preferred


def test_explicit_resource_file_and_presence_are_reported(home, monkeypatch):
    source = home / "custom-wordnet.xml"
    source.write_text("<LexicalResource/>\n")
    monkeypatch.setenv("REXGRAPH_WORDNET", str(source))
    assert lexical_path("wordnet") == source
    assert lexical_status()["wordnet"] == {"path": str(source), "present": True}


def test_unknown_resource_is_refused(home):
    with pytest.raises(KeyError):
        lexical_path("unknown")


def test_default_cache_and_configuration_do_not_create_directories(home):
    assert cache_root() == home / ".cache" / "rexgraph"
    assert config_root() == home / ".config" / "rexgraph"
    assert models_root() == cache_root() / "models"
    assert not cache_root().exists()
    assert not config_root().exists()


def test_xdg_paths_and_explicit_overrides_expand_user_directory(home, monkeypatch):
    expanded_home = Path("~").expanduser()
    monkeypatch.setenv("XDG_CACHE_HOME", "~/cache")
    monkeypatch.setenv("XDG_CONFIG_HOME", "~/config")
    assert cache_root() == expanded_home / "cache" / "rexgraph"
    assert config_root() == expanded_home / "config" / "rexgraph"
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", "~/custom cache")
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", "~/custom config")
    monkeypatch.setenv("REXGRAPH_MODELS_DIR", "~/custom models")
    assert cache_root() == expanded_home / "custom cache"
    assert config_root() == expanded_home / "custom config"
    assert models_root() == expanded_home / "custom models"
    from agent.cli import config
    assert config._xdg_cache_home() == cache_root()
    assert config._xdg_config_home() == config_root()


def test_empty_directory_settings_use_defaults(home, monkeypatch):
    for name in ("REXGRAPH_CACHE_DIR", "REXGRAPH_CONFIG_DIR", "REXGRAPH_MODELS_DIR",
                 "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(name, "")
    assert cache_root() == home / ".cache" / "rexgraph"
    assert config_root() == home / ".config" / "rexgraph"
    assert models_root() == cache_root() / "models"


def test_sessions_use_configured_root_and_leave_existing_legacy_data_alone(home, monkeypatch):
    legacy = home / ".rexgraph-agent" / "sessions"
    legacy.mkdir(parents=True)
    assert sessions_root() == legacy
    configured = home / "configured"
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(configured))
    assert sessions_root() == configured / "sessions"
    from agent.server.state import SessionStore
    from agent.session import create_session, list_sessions
    store = SessionStore()
    session = create_session()
    from rexgraph import RexGraph
    session.add_snapshot(RexGraph.from_graph([0], [1]), "upload")
    assert store.storage_dir == str(configured / "sessions")
    assert list_sessions()[0]["session_id"] == session.session_id
    assert legacy.is_dir() and not list(legacy.iterdir())
    monkeypatch.setenv("REXGRAPH_SESSION_DIR", str(home / "explicit"))
    assert sessions_root() == home / "explicit"


def test_windows_application_directories(home, monkeypatch):
    monkeypatch.setattr("agent.data_paths.platform.system", lambda: "Windows")
    assert config_root() == home / "AppData" / "Roaming" / "rexgraph"
    assert cache_root() == home / "AppData" / "Local" / "rexgraph" / "cache"
    monkeypatch.setenv("APPDATA", str(home / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "local"))
    assert config_root() == home / "roaming" / "rexgraph"
    assert cache_root() == home / "local" / "rexgraph" / "cache"


def test_huggingface_cache_overrides(home, monkeypatch):
    expanded_home = Path("~").expanduser()
    assert huggingface_cache_root() == home / ".cache" / "huggingface" / "hub"
    monkeypatch.setenv("XDG_CACHE_HOME", "~/cache")
    assert huggingface_cache_root() == expanded_home / "cache" / "huggingface" / "hub"
    monkeypatch.setenv("HF_HOME", "~/hf")
    assert huggingface_cache_root() == expanded_home / "hf" / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "~/legacy hub")
    assert huggingface_cache_root() == expanded_home / "legacy hub"
    monkeypatch.setenv("HF_HUB_CACHE", "~/hub")
    assert huggingface_cache_root() == expanded_home / "hub"


def test_ocr_cache_requires_weights_in_configured_location(home, monkeypatch):
    from agent.integrations.unlimited_ocr import GOTOCRClient
    monkeypatch.setenv("HF_HUB_CACHE", str(home / "hub"))
    client = GOTOCRClient(model_name="example/ocr")
    snapshot = home / "hub" / "models--example--ocr" / "snapshots" / "synthetic"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{}')
    assert not client._model_cached()
    (snapshot / "model.safetensors").write_bytes(b"synthetic")
    assert client._model_cached()


def test_model_discovery_deduplicates_directory_aliases(home, monkeypatch):
    from agent.local_runtime import _default_scan_dirs
    directory = home / "models"
    directory.mkdir()
    alias = home / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    monkeypatch.setenv("REXGRAPH_MODELS_DIR", str(directory))
    monkeypatch.setenv("REXGRAPH_MODEL_DIRS", str(alias))
    roots = _default_scan_dirs()
    assert sum(path.resolve() == directory for path in map(Path, roots)) == 1


def test_explicit_model_scan_roots_do_not_scan_unrequested_home_locations(home, monkeypatch):
    from agent.local_runtime import _default_scan_dirs
    selected = home / "selected"
    selected.mkdir()
    unrelated = home / "models"
    unrelated.mkdir()
    monkeypatch.setenv("REXGRAPH_MODEL_SCAN_DIRS", str(selected))
    assert _default_scan_dirs() == [str(selected)]


def test_explicit_hpc_model_location_precedes_cluster_defaults(home, monkeypatch):
    from agent.cli.hpc import resolve_model_cache
    cluster = home / "cluster"
    cluster.mkdir()
    monkeypatch.setenv("PROJECT", str(cluster))
    monkeypatch.setenv("REXGRAPH_MODELS_DIR", str(home / "models"))
    assert resolve_model_cache() == home / "models"
    assert not (cluster / "rexgraph_models").exists()


def test_configuration_consumers_use_same_root(home, monkeypatch):
    from agent.activity import _journal_default
    from agent.cli.auth import _cred_file
    from agent.hive_config import _config_dir
    from agent.lifecycle import _runs_dir
    from agent.server.handles import _root
    from agent.secrets import EnvSecretStore, open_secret_store

    monkeypatch.delenv("REXGRAPH_ACTIVITY_JOURNAL", raising=False)
    monkeypatch.delenv("REXGRAPH_NO_JOURNAL", raising=False)
    monkeypatch.delenv("REXGRAPH_SECRETS_URI", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "settings"))
    root = config_root()
    assert _journal_default() == root / "activity.jsonl"
    assert _cred_file() == root / "credentials.json"
    assert _config_dir() == root / "hive_profiles"
    assert _runs_dir() == root / "runs"
    assert _root() == root / "workspaces"
    assert Path(EnvSecretStore().path) == root / "secret_refs.json"
    assert Path(open_secret_store().path) == root / "connections.json"


def test_directory_overrides_replace_paths_saved_by_previous_setup(home, monkeypatch):
    from agent.cli import config
    file = home / "config.json"
    file.write_text('{"cache_dir": "/saved/cache", "models_dir": "/saved/models"}')
    monkeypatch.setattr(config, "CONFIG_FILE", file)
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(home / "cache"))
    monkeypatch.setenv("REXGRAPH_MODELS_DIR", str(home / "models"))
    saved = config.load_config()
    assert saved.cache_dir == str(cache_root())
    assert saved.models_dir == str(models_root())


def test_saved_directory_settings_remain_when_no_override_is_set(home, monkeypatch):
    from agent.cli import config
    file = home / "config.json"
    file.write_text('{"cache_dir": "/saved/cache", "models_dir": "/saved/models"}')
    monkeypatch.setattr(config, "CONFIG_FILE", file)
    saved = config.load_config()
    assert saved.cache_dir == "/saved/cache"
    assert saved.models_dir == "/saved/models"


def test_store_uses_resource_overrides_unless_a_root_is_supplied(home, monkeypatch):
    from agent.adapters.lexical_store import _paths

    source = home / "custom-wordnet.xml"
    monkeypatch.setenv("REXGRAPH_WORDNET", str(source))
    assert _paths()["wordnet"] == str(source)
    explicit = home / "store-resources"
    assert _paths(explicit)["wordnet"] == str(explicit / "wordnet" / "english-wordnet-2024.xml.gz")
