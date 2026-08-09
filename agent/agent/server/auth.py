"""
agent.server.auth: authentication and workspace management.

Bearer token auth with workspace-scoped isolation aligned with
TrustGraph's model. Each workspace is a namespace containing
its own corpus, sessions, conversations, and activity complex.

The workspace activity is itself a relational complex: users
are vertices, documents are vertices, queries and exchanges
are edges. The workspace's structural character tells you
what kind of work the team is doing.
"""

from __future__ import annotations

import contextlib
import json
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# Config path
_CONFIG_DIR = Path(os.environ.get("REXGRAPH_CONFIG_DIR",
    Path.home() / ".config" / "rexgraph"))

# Canonical roles for the shared multi-user model. A workspace has admin(s) who manage members,
# configuration, and consequential actions, and users who read and run build verbs. Legacy tokens
# stored with the older "write"/"read" roles are treated as users (only "admin" grants admin).
ROLE_ADMIN = "admin"
ROLE_USER = "user"


def _norm_role(role: str) -> str:
    """Canonicalize any role string to the two-role model ('write'/'read'/anything -> 'user')."""
    return ROLE_ADMIN if (role or "").strip().lower() == ROLE_ADMIN else ROLE_USER


def is_admin(entry: TokenEntry | None, workspace: str = "default") -> bool:
    """True if the token holds the admin role IN `workspace`. Roles are per-workspace; the root
    workspace 'default' is the instance, so admin-of-default is the instance administrator."""
    return entry is not None and entry.is_admin_in(workspace)


@dataclass
class TokenEntry:
    """A registered API token, with a per-workspace role map.

    `roles` maps a workspace name to 'admin' or 'user'; the key '*' grants that role in EVERY
    workspace (used only by the auth-disabled local identity). `workspaces` (the access list) and the
    legacy scalar `role` are kept as derived, back-compatible views so older callers and older
    persisted tokens keep working: a token loaded with only (role, workspaces) synthesizes `roles`.
    """
    token_hash: str
    user_id: str
    workspaces: list[str] = field(default_factory=list)  # access list (derived from roles)
    role: str = ROLE_USER                                # legacy scalar view (highest role held)
    created: float = 0.0
    roles: dict[str, str] = field(default_factory=dict)  # {workspace: 'admin'|'user'}; '*' = all

    def __post_init__(self):
        if not self.roles:                               # legacy token: synthesize from role+workspaces
            self.roles = {ws: _norm_role(self.role) for ws in (self.workspaces or [])}
        else:
            self.roles = {ws: _norm_role(r) for ws, r in self.roles.items()}
        self._resync()

    def _resync(self):
        """Keep the legacy `workspaces`/`role` views consistent with the `roles` map."""
        concrete = [w for w in self.roles if w != "*"]
        if concrete:
            self.workspaces = concrete
        self.role = ROLE_ADMIN if ROLE_ADMIN in self.roles.values() else ROLE_USER

    def role_in(self, workspace: str) -> str:
        """This token's role in `workspace` ('admin'/'user'), or '' if it has no access there."""
        return self.roles.get(workspace) or self.roles.get("*") or ""

    def is_admin_in(self, workspace: str) -> bool:
        return self.role_in(workspace) == ROLE_ADMIN

    def can_access(self, workspace: str) -> bool:
        return bool(self.role_in(workspace))


@dataclass
class WorkspaceState:
    """Per-workspace isolated state."""
    name: str
    corpus: object = None           # CorpusBuilder instance
    sessions: dict = field(default_factory=dict)
    trackers: dict = field(default_factory=dict)  # session_id -> ConversationTracker
    activity_edges: list = field(default_factory=list)  # (user, action, target, timestamp)
    created: float = 0.0

    def get_corpus(self):
        if self.corpus is None:
            from agent.corpus import CorpusBuilder
            self.corpus = CorpusBuilder()
        return self.corpus

    def get_tracker(self, session_id: str):
        from agent.conversation import ConversationTracker
        if session_id not in self.trackers:
            self.trackers[session_id] = ConversationTracker()
        return self.trackers[session_id]

    def record_activity(self, user_id: str, action: str, target: str):
        """Record a user action as an edge in the workspace activity graph."""
        self.activity_edges.append((user_id, action, target, time.time()))

    def activity_summary(self) -> dict:
        """Summarize workspace activity."""
        users = set()
        docs = set()
        queries = 0
        for user, action, target, _ts in self.activity_edges:
            users.add(user)
            if action in ("upload", "add_document"):
                docs.add(target)
            elif action == "query":
                queries += 1
        return {
            "n_users": len(users),
            "n_documents": len(docs),
            "n_queries": queries,
            "n_events": len(self.activity_edges),
            "users": sorted(users),
        }

    def build_activity_complex(self):
        """Build a relational complex from workspace activity.

        Users and documents are vertices. Interactions are edges.
        The resulting complex's structural character tells you
        what kind of work the team is doing:
            T-dominant = taxonomic/organizational activity
            G-dominant = deep analytical work
            F-dominant = adversarial review (finding contradictions)
            C-dominant = collaborative exploration
        """
        if not self.activity_edges:
            return None

        try:
            import numpy as np

            from rexgraph.graph import RexGraph

            # Build vertex set from users and targets
            labels = []
            label_idx = {}
            for user, _action, target, _ts in self.activity_edges:
                for entity in [user, target]:
                    if entity and entity not in label_idx:
                        label_idx[entity] = len(labels)
                        labels.append(entity)

            # Build edges from interactions
            sources = []
            targets_arr = []
            for user, _action, target, _ts in self.activity_edges:
                if user in label_idx and target in label_idx:
                    src = label_idx[user]
                    tgt = label_idx[target]
                    if src != tgt:
                        sources.append(src)
                        targets_arr.append(tgt)

            if not sources:
                return None

            # Deduplicate edges
            edge_set = {}
            for s, t in zip(sources, targets_arr, strict=False):
                key = (min(s, t), max(s, t))
                edge_set[key] = edge_set.get(key, 0) + 1

            src = np.array([k[0] for k in edge_set], dtype=np.int32)
            tgt = np.array([k[1] for k in edge_set], dtype=np.int32)

            rex = RexGraph(sources=src, targets=tgt)
            kappa = rex.coherence
            km = float(kappa.mean()) if kappa is not None and len(kappa) > 0 else 0.0
            return {
                "rex": rex,
                "labels": labels,
                "n_users": len(set(u for u, _, _, _ in self.activity_edges)),
                "betti": rex.betti,
                "kappa": km if not (isinstance(km, float) and km != km) else 0.0,
            }
        except Exception as e:
            logger.warning("Activity complex failed: %s", e)
            return None

    def detect_query_overlap(self) -> list:
        """Detect when multiple users query the same structural region.

        Compares query targets across users. If two users queried
        the same document entities, they're working on overlapping
        structure. Returns pairs of (user_a, user_b, shared_targets, count).
        """
        from collections import defaultdict

        # Group query targets by user
        user_queries = defaultdict(set)
        for user, action, target, _ts in self.activity_edges:
            if action == "query":
                # Each word in the query is a target entity
                words = set(target.lower().split())
                user_queries[user].update(words)

        # Find overlapping users
        overlaps = []
        users = sorted(user_queries.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                shared = user_queries[users[i]] & user_queries[users[j]]
                if shared:
                    overlaps.append({
                        "user_a": users[i],
                        "user_b": users[j],
                        "shared_terms": sorted(shared),
                        "overlap_count": len(shared),
                        "suggestion": "Users are querying overlapping regions - consider sharing findings",
                    })
        return overlaps


class AuthManager:
    """Manages tokens, workspaces, and access control.

    Supports three auth modes:
        bearer  - API key tokens (default)
        oidc    - OpenID Connect (Okta, Azure AD, Google)
        saml    - SAML 2.0 (enterprise)

    Set REXGRAPH_AUTH_MODE=oidc and configure OIDC_* env vars for SSO.
    """

    def __init__(self):
        self._tokens: dict[str, TokenEntry] = {}
        #: presented-token sha256 -> the entry it verified as, or False for a known-bad
        #: one. Holds no secret and is cleared whenever the token set changes.
        self._verify_cache: dict[str, object] = {}
        self._workspaces: dict[str, WorkspaceState] = {}
        self._auth_enabled = False
        self._recovery_hash: str = ""
        self._disable_passphrase_hash: str = ""
        self._config_existed = False
        self._auth_mode = os.environ.get("REXGRAPH_AUTH_MODE", "bearer")
        self._oidc_config = None
        self._load_config()
        if self._auth_mode == "oidc":
            self._init_oidc()

    def _load_config(self):
        """Load tokens from config file if it exists."""
        config_path = _CONFIG_DIR / "auth.json"
        if config_path.exists():
            self._config_existed = True
            try:
                data = json.loads(config_path.read_text())
                for entry in data.get("tokens", []):
                    te = TokenEntry(**entry)
                    self._tokens[te.token_hash] = te
                self._auth_enabled = data.get("enabled", False)
                self._recovery_hash = data.get("recovery_hash", "")
                self._disable_passphrase_hash = data.get("disable_passphrase_hash", "")
                logger.info("Loaded %d tokens, auth=%s, recovery=%s",
                            len(self._tokens), self._auth_enabled,
                            bool(self._recovery_hash))
            except Exception as e:
                logger.warning("Failed to load auth config: %s", e)

    def _save_config(self):
        """Save tokens to config file."""
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = _CONFIG_DIR / "auth.json"
        data = {
            "enabled": self._auth_enabled,
            "recovery_hash": self._recovery_hash,
            "disable_passphrase_hash": self._disable_passphrase_hash,
            "tokens": [
                {
                    "token_hash": te.token_hash,
                    "user_id": te.user_id,
                    "workspaces": te.workspaces,   # derived view, kept for older readers
                    "role": te.role,               # derived scalar, kept for older readers
                    "roles": te.roles,             # authoritative per-workspace map
                    "created": te.created,
                }
                for te in self._tokens.values()
            ],
        }
        config_path.write_text(json.dumps(data, indent=2))
        # Restrict permissions: owner read/write only
        with contextlib.suppress(OSError):
            os.chmod(str(config_path), 0o600)

    def _invalidate_verify_cache(self) -> None:
        """Forget every remembered verification. Called wherever the token set moves, so
        a revoked token stops working on the next request rather than at restart."""
        self._verify_cache = {}

    @staticmethod
    def _hash(token: str) -> str:
        import bcrypt
        return bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def _verify_hash(token: str, stored_hash: str) -> bool:
        import bcrypt
        try:
            return bcrypt.checkpw(token.encode(), stored_hash.encode())
        except Exception:
            return False

    def create_token(self, user_id: str, workspaces: list[str],
                     role: str = ROLE_USER) -> str:
        """Create a new API token granting `role` in each of `workspaces`. Returns the raw token."""
        return self._mint(user_id, {ws: role for ws in (workspaces or ["default"])})

    def _mint(self, user_id: str, roles: dict[str, str]) -> str:
        """Mint a fresh token for `user_id` with a per-workspace roles map. Returns the raw token."""
        import secrets
        raw = secrets.token_urlsafe(32)
        h = self._hash(raw)
        self._tokens[h] = TokenEntry(token_hash=h, user_id=user_id,
                                     roles={w: _norm_role(r) for w, r in roles.items() if w},
                                     created=time.time())
        self._invalidate_verify_cache()
        self._save_config()
        return raw

    #### member management (the shared-workspace roster, per workspace)
    def _tokens_of(self, user_id: str) -> list[TokenEntry]:
        return [te for te in self._tokens.values() if te.user_id == user_id]

    def _roles_of(self, user_id: str) -> dict[str, str]:
        """The member's merged per-workspace roles across any tokens they hold."""
        merged: dict[str, str] = {}
        for te in self._tokens_of(user_id):
            merged.update(te.roles)
        return merged

    def _admins_of(self, workspace: str) -> set:
        return {te.user_id for te in self._tokens.values() if te.is_admin_in(workspace)}

    def _n_admins(self, workspace: str = "default") -> int:
        return len(self._admins_of(workspace))

    def _revoke_user_unsaved(self, user_id: str) -> int:
        dead = [h for h, te in self._tokens.items() if te.user_id == user_id]
        for h in dead:
            del self._tokens[h]
            self._invalidate_verify_cache()
        return len(dead)

    def add_member(self, user_id: str, role: str = ROLE_USER,
                   workspace: str = "default") -> str | None:
        """Grant `user_id` the role in `workspace`. If they are new, mint their token (returned, shown
        once); if they already have one, update its roles IN PLACE (no rotation, so their existing token
        keeps working, so their access in other workspaces is untouched) and return None."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required")
        role = _norm_role(role)
        toks = self._tokens_of(user_id)
        if not toks:
            return self._mint(user_id, {workspace: role})
        keep = max(toks, key=lambda t: t.created)          # consolidate onto the newest token
        merged: dict[str, str] = {}
        for t in toks:
            merged.update(t.roles)
            if t is not keep:
                del self._tokens[t.token_hash]
                self._invalidate_verify_cache()
        merged[workspace] = role
        keep.roles = {w: _norm_role(r) for w, r in merged.items()}
        keep._resync()
        self._save_config()
        return None

    def revoke_member(self, user_id: str, workspace: str | None = None) -> int:
        """Revoke a member's access. With `workspace`, remove only that workspace's role (their token
        keeps working elsewhere); without it, remove the member entirely. Refuses to remove the last
        admin of any affected workspace. Returns how many members/roles were removed (0 if none)."""
        toks = self._tokens_of(user_id)
        if not toks:
            return 0
        merged: dict[str, str] = {}
        for t in toks:
            merged.update(t.roles)
        removing = set(merged) if workspace is None else ({workspace} if workspace in merged else set())
        if workspace is not None and not removing:
            return 0
        for ws in removing:
            if merged.get(ws) == ROLE_ADMIN and self._admins_of(ws) <= {user_id}:
                raise ValueError(f"cannot revoke the last admin of workspace '{ws}'")
        if workspace is None:
            n = self._revoke_user_unsaved(user_id)
            if n:
                self._save_config()
            return n
        keep = max(toks, key=lambda t: t.created)
        for t in toks:
            if t is not keep:
                del self._tokens[t.token_hash]
                self._invalidate_verify_cache()
        del merged[workspace]
        if merged:
            keep.roles = {w: _norm_role(r) for w, r in merged.items()}
            keep._resync()
        else:
            del self._tokens[keep.token_hash]
            self._invalidate_verify_cache()
        self._save_config()
        return 1

    def members(self, workspace: str | None = None) -> list[dict]:
        """The roster. With `workspace`, one row per member who has a role there ({user_id, role,
        workspace}); without it, one row per member with their full {user_id, roles} map."""
        users: dict[str, dict] = {}
        for te in self._tokens.values():
            u = users.setdefault(te.user_id, {"user_id": te.user_id, "roles": {}, "created": te.created})
            u["roles"].update(te.roles)
            u["created"] = min(u["created"], te.created)
        rows = list(users.values())
        if workspace is None:
            return sorted(rows, key=lambda m: m["user_id"])
        out = []
        for m in rows:
            r = m["roles"].get(workspace) or m["roles"].get("*")
            if r:
                out.append({"user_id": m["user_id"], "role": r, "workspace": workspace,
                            "created": m["created"]})
        return sorted(out, key=lambda m: (m["role"] != ROLE_ADMIN, m["user_id"]))

    def verify(self, token: str) -> TokenEntry | None:
        """Verify a bearer token or OIDC JWT. Returns the entry or None."""
        if not self._auth_enabled:
            return TokenEntry(
                token_hash="", user_id="local",
                roles={"*": ROLE_ADMIN},          # solo/local: admin in every workspace
            )
        if not token:
            return None

        # Try OIDC first if configured
        if self._auth_mode == "oidc" and self._oidc_config:
            result = self.verify_oidc_token(token)
            if result:
                return result

        # Fall back to bearer token.
        #
        # bcrypt is deliberately slow, and this scanned EVERY stored hash on EVERY
        # request: at cost 12 with five tokens, and the matching one last, a request that
        # returns 0.1 KB took 0.86s and the app took about 3s a page. The work was the
        # authentication, not the data, and it grew with the number of tokens on file.
        #
        # So verify once and remember the answer for the process. The key is a SHA-256 of
        # the presented token rather than the token itself, so the cache never holds the
        # secret; and a MISS is cached too, or an unauthenticated caller could still make
        # the server do five bcrypts per request by repeating one bad token. Cleared
        # whenever the token set changes, so a revoked token stops working immediately.
        key = hashlib.sha256(token.encode()).hexdigest()
        cached = self._verify_cache.get(key)
        if cached is not None:
            return cached or None
        for stored_hash, entry in self._tokens.items():
            if self._verify_hash(token, stored_hash):
                self._verify_cache[key] = entry
                return entry
        if len(self._verify_cache) < 4096:      # a bound, so a flood cannot grow it
            self._verify_cache[key] = False
        return None

    def enable_auth(self, *, persist: bool = True):
        """Turn auth on. Persisted unless `persist=False`.

        Enabling is the safe direction, so it needs no confirmation: the worst an
        accidental enable does is ask for a token that the caller already has on file.
        """
        self._auth_enabled = True
        if persist:
            self._save_config()

    def disable_auth(self, *, persist: bool = True, confirm: bool = False):
        """Turn auth off.

        Disabling is the UNSAFE direction and this used to persist unconditionally, so
        any in-process caller wrote `enabled: false` into the host's own auth.json. Six
        test fixtures did exactly that, which is how a test suite turned auth off on a
        live install and left it off. Two guards now, and they are separate on purpose:

            persist=False   flip the flag for this process only and never touch disk.
                            What a test wants: it needs the server object open, not the
                            host reconfigured.
            confirm=True    required before a disable is written to a config that HAS
                            TOKENS, because that is someone's live install. Missing it
                            raises rather than writing, so an accidental call is loud
                            instead of silent.

        The network path is stricter still and unchanged: POST
        /api/v1/admin/auth/disable additionally requires the request to originate from
        the server host, an admin token, and the disable passphrase.
        """
        self._auth_enabled = False
        if not persist:
            return
        if self._tokens and not confirm:
            self._auth_enabled = True
            raise PermissionError(
                "refusing to write auth off for a config that has "
                f"{len(self._tokens)} token(s). Pass confirm=True if that is meant, or "
                "persist=False to disable it for this process only.")
        logger.warning("authentication disabled and written to %s",
                       _CONFIG_DIR / "auth.json")
        self._save_config()

    # Step-up secret for turning auth OFF. A leaked or cached API token is
    # not enough to disable auth: the caller must also present this passphrase,
    # which is bcrypt-hashed here and never stored client-side.

    def set_disable_passphrase(self, passphrase: str):
        """Set or rotate the passphrase required to disable auth."""
        if not passphrase or len(passphrase) < 8:
            raise ValueError("disable passphrase must be at least 8 characters")
        self._disable_passphrase_hash = self._hash(passphrase)
        self._save_config()

    def verify_disable_passphrase(self, passphrase: str) -> bool:
        """True only if a passphrase is configured and matches."""
        if not self._disable_passphrase_hash:
            return False
        return self._verify_hash(passphrase or "", self._disable_passphrase_hash)

    @property
    def has_disable_passphrase(self) -> bool:
        return bool(self._disable_passphrase_hash)

    @property
    def is_fresh(self) -> bool:
        """True when no auth.json was found at load time (a fresh install)."""
        return not self._config_existed

    def bootstrap_admin(self, initial_token: str | None = None) -> str | None:
        """When auth is enabled with no tokens, mint the first admin token.

        Returns the raw token if one was created (to be shown once), else None.
        Pass initial_token (e.g. from REXGRAPH_ADMIN_TOKEN) to set a known
        value instead of a random one.
        """
        if not self._auth_enabled or self._tokens:
            return None
        if initial_token:
            h = self._hash(initial_token)
            self._tokens[h] = TokenEntry(
                token_hash=h, user_id="admin",
                workspaces=["default"], role="admin", created=time.time(),
            )
            self._invalidate_verify_cache()
            self._save_config()
            return initial_token
        return self.create_token("admin", ["default"], role="admin")

    def create_recovery_key(self) -> str:
        """Create a recovery key. Returns the raw key (shown once).

        The recovery key is the only way to regain access when all API
        tokens are lost.  It is bcrypt-hashed and stored in auth.json
        alongside the token hashes, so reading the config file does not
        reveal it.
        """
        import secrets
        raw = "rk_" + secrets.token_urlsafe(32)
        self._recovery_hash = self._hash(raw)
        self._save_config()
        logger.info("Recovery key created")
        return raw

    def verify_recovery(self, key: str) -> bool:
        """Verify a recovery key against the stored hash."""
        if not self._recovery_hash or not key:
            return False
        return self._verify_hash(key, self._recovery_hash)

    def recover(self, recovery_key: str) -> str | None:
        """Use recovery key to create a new admin token.

        Returns the raw token string, or None if the key is invalid.
        """
        if not self.verify_recovery(recovery_key):
            return None
        raw = self.create_token("admin-recovered", ["default"], "admin")
        logger.info("Access recovered via recovery key")
        return raw

    @property
    def has_recovery_key(self) -> bool:
        return bool(self._recovery_hash)

    @property
    def auth_enabled(self):
        return self._auth_enabled

    @property
    def auth_mode(self):
        return self._auth_mode

    def _init_oidc(self):
        """Initialize OIDC configuration from environment variables.

        Required env vars:
            OIDC_ISSUER         - e.g. https://accounts.google.com
            OIDC_CLIENT_ID      - your app's client ID
            OIDC_CLIENT_SECRET  - your app's client secret (optional for public clients)

        Optional:
            OIDC_AUDIENCE       - token audience (defaults to client ID)
            OIDC_WORKSPACE_CLAIM: JWT claim for workspace (default: "workspace")
            OIDC_ROLE_CLAIM     - JWT claim for role (default: "role")
        """
        issuer = os.environ.get("OIDC_ISSUER", "")
        client_id = os.environ.get("OIDC_CLIENT_ID", "")

        if not issuer or not client_id:
            logger.warning(
                "OIDC mode enabled but OIDC_ISSUER and OIDC_CLIENT_ID not set. "
                "Falling back to bearer token auth."
            )
            self._auth_mode = "bearer"
            return

        self._oidc_config = {
            "issuer": issuer,
            "client_id": client_id,
            "client_secret": os.environ.get("OIDC_CLIENT_SECRET", ""),
            "audience": os.environ.get("OIDC_AUDIENCE", client_id),
            "workspace_claim": os.environ.get("OIDC_WORKSPACE_CLAIM", "workspace"),
            "role_claim": os.environ.get("OIDC_ROLE_CLAIM", "role"),
            "jwks_uri": issuer.rstrip("/") + "/.well-known/jwks.json",
        }
        logger.info("OIDC configured: issuer=%s", issuer)

    def verify_oidc_token(self, token: str) -> TokenEntry | None:
        """Verify a JWT token from the OIDC provider.

        Requires: pip install python-jose[cryptography] requests
        """
        if not self._oidc_config:
            return None

        try:
            import requests
            from jose import jwt as jose_jwt

            # Fetch JWKS (cached in production; this is the simple version)
            jwks = requests.get(self._oidc_config["jwks_uri"], timeout=10).json()

            payload = jose_jwt.decode(
                token,
                jwks,
                algorithms=["RS256"],
                audience=self._oidc_config["audience"],
                issuer=self._oidc_config["issuer"],
            )

            user_id = payload.get("sub", payload.get("email", "unknown"))
            workspace = payload.get(
                self._oidc_config["workspace_claim"], "default"
            )
            role = payload.get(self._oidc_config["role_claim"], "write")

            if isinstance(workspace, str):
                workspace = [workspace]

            return TokenEntry(
                token_hash="oidc",
                user_id=user_id,
                workspaces=workspace,
                role=role,
                created=time.time(),
            )
        except ImportError:
            logger.error("OIDC requires: pip install python-jose[cryptography] requests")
            return None
        except Exception as e:
            logger.warning("OIDC verification failed: %s", e)
            return None

    def get_workspace(self, name: str) -> WorkspaceState:
        """Get or create a workspace."""
        # Validate workspace name
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', name):
            raise ValueError("Invalid workspace name: use alphanumeric, dash, underscore (max 64 chars)")
        if name not in self._workspaces:
            self._workspaces[name] = WorkspaceState(
                name=name, created=time.time(),
            )
        return self._workspaces[name]

    def list_workspaces(self) -> list[str]:
        return sorted(self._workspaces.keys())

    def list_tokens(self) -> list[dict]:
        return [
            {
                "user_id": te.user_id,
                "workspaces": te.workspaces,
                "role": te.role,
                "roles": te.roles,
                "created": te.created,
            }
            for te in self._tokens.values()
        ]


# Singleton
_manager = None

def get_auth_manager() -> AuthManager:
    global _manager
    if _manager is None:
        _manager = AuthManager()
    return _manager


def reset_auth_manager() -> None:
    """Drop the cached manager so the next call reloads from disk. For tests."""
    global _manager
    _manager = None


def identity_and_workspace(request) -> tuple[str, str]:
    """Who this request is and which workspace it names, off the raw request.

    The middlewares need this before any route dependency has resolved, and three of
    them needed it, so it lives here once rather than as three token parsers that can
    disagree about what counts as a bearer header.

    Returns `("local", "default")` when auth is off, which is the single-operator case
    and is not being distinguished from itself. An unverifiable token yields an empty
    identity: the auth middleware is what rejects it, and answering "anonymous" here
    would let it share one bucket with every other anonymous caller.
    """
    ws = (request.headers.get("X-Workspace")
          or request.query_params.get("workspace") or "")
    try:
        mgr = get_auth_manager()
        if not mgr.auth_enabled:
            return "local", (ws or "default")
        header = request.headers.get("Authorization", "")
        raw = header[7:].strip() if header[:7].lower() == "bearer " else ""
        entry = mgr.verify(raw) if raw else None
        if entry is None:
            return "", (ws or "default")
        if not ws:
            ws = entry.workspaces[0] if entry.workspaces else "default"
        return entry.user_id, ws
    except Exception:                            # noqa: BLE001 - never break a request
        return "", (ws or "default")


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> TokenEntry:
    """FastAPI dependency: verify bearer token.

    When auth is disabled, returns a default local user.
    When auth is enabled, requires a valid bearer token.
    """
    mgr = get_auth_manager()

    if not mgr.auth_enabled:
        return TokenEntry(
            token_hash="", user_id="local",
            roles={"*": ROLE_ADMIN},              # solo/local: admin in every workspace
        )

    if credentials is None:
        raise HTTPException(401, "Authentication required")

    entry = mgr.verify(credentials.credentials)
    if entry is None:
        raise HTTPException(401, "Invalid token")

    return entry


async def require_admin(token: TokenEntry = Depends(require_auth)) -> TokenEntry:
    """FastAPI dependency: require INSTANCE admin (admin of the root workspace 'default').

    Instance-level operations - enabling/disabling auth, recovery keys, the legacy token routes -
    are gated here. Per-workspace member management uses :func:`require_workspace_admin` instead.
    When auth is disabled, ``require_auth`` returns the local admin identity, so solo/local use is
    unaffected.
    """
    if not is_admin(token, "default"):
        raise HTTPException(403, "Admin role required")
    return token


def require_workspace(
    request: Request,
    token: TokenEntry = Depends(require_auth),
) -> WorkspaceState:
    """FastAPI dependency: get the workspace for this request.

    Workspace is determined by:
    1. X-Workspace header
    2. ?workspace= query parameter
    3. First workspace in the token's access list
    4. "default"
    """
    mgr = get_auth_manager()

    ws_name = (
        request.headers.get("X-Workspace")
        or request.query_params.get("workspace")
        or (token.workspaces[0] if token.workspaces else "default")
    )

    if mgr.auth_enabled and not token.can_access(ws_name):
        raise HTTPException(403, f"No access to workspace '{ws_name}'")

    ws = mgr.get_workspace(ws_name)
    return ws


async def require_workspace_admin(
    ws: WorkspaceState = Depends(require_workspace),
    token: TokenEntry = Depends(require_auth),
) -> WorkspaceState:
    """FastAPI dependency: require admin OF THE CURRENT workspace (from X-Workspace/?workspace).

    A user who is admin of workspace A but a plain user of workspace B can manage A and not B.
    Returns the workspace so routes can use its name.
    """
    if not is_admin(token, ws.name):
        raise HTTPException(403, f"Admin of workspace '{ws.name}' required")
    return ws
