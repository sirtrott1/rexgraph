"""
agent.server.routes.admin: workspace and token management.

    POST /api/v1/admin/token          create a token
    GET  /api/v1/admin/tokens         list tokens
    POST /api/v1/admin/auth/enable    enable auth
    POST /api/v1/admin/auth/disable   disable auth
    POST /api/v1/admin/recovery-key   create recovery key (admin, shown once)
    POST /api/v1/admin/recover        use recovery key to get new token (NO auth)
    GET  /api/v1/admin/workspaces     list workspaces
    GET  /api/v1/admin/workspace/activity  workspace activity summary
    GET  /api/v1/admin/workspace/complex   workspace activity as relational complex
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from agent.metrics import coherence_kappa
from agent.server.auth import (
    ROLE_USER,
    TokenEntry,
    WorkspaceState,
    get_auth_manager,
    is_admin,
    require_admin,
    require_auth,
    require_workspace,
    require_workspace_admin,
)

router = APIRouter(prefix="/v1/admin")

# One shared admin gate (in auth.py) so the same rule applies here and on the agent routes.
_require_admin = require_admin


_LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def _require_localhost(request: Request):
    """Reject anything that is not a direct loopback connection.

    Uses the raw socket peer (request.client), never the client-supplied
    X-Forwarded-For, and refuses requests that carry any forwarding header:
    a request that arrived through a proxy is treated as remote even when the
    proxy itself runs on the host. This is the gate for changing the auth
    posture, so it stays strict.
    """
    host = request.client.host if request.client else None
    forwarded = request.headers.get("X-Forwarded-For") or request.headers.get("Forwarded")
    if host not in _LOOPBACK or forwarded:
        raise HTTPException(
            403, "This operation must be performed from the server host "
                 "(direct loopback connection, no proxy).")


@router.get("/whoami")
async def whoami(token: TokenEntry = Depends(require_auth),
                 ws: WorkspaceState = Depends(require_workspace)):
    """The current identity: who you are, your role in EACH workspace, your role in the current
    workspace, and whether auth is on. The UI reads this to show the right controls per workspace
    (member-management appears only where you are an admin)."""
    mgr = get_auth_manager()
    return {"user_id": token.user_id, "roles": token.roles, "workspaces": token.workspaces,
            "role": token.role, "current_workspace": ws.name,
            "is_admin": is_admin(token, ws.name),           # admin of the CURRENT workspace
            "instance_admin": is_admin(token, "default"),   # admin of the root workspace
            "auth_enabled": mgr.auth_enabled}


@router.get("/members")
async def list_members(ws: WorkspaceState = Depends(require_workspace_admin)):
    """The roster for the CURRENT workspace: one row per member with their role there. Requires you to
    be an admin of that workspace (set X-Workspace / ?workspace to pick which one)."""
    mgr = get_auth_manager()
    return {"workspace": ws.name, "members": mgr.members(ws.name), "n_admins": mgr._n_admins(ws.name)}


@router.post("/members")
async def add_member(
    user_id: str = Body(..., embed=True),
    role: str = Body(ROLE_USER, embed=True),
    ws: WorkspaceState = Depends(require_workspace_admin),
    caller: TokenEntry = Depends(require_auth),
):
    """Grant `user_id` a role in the CURRENT workspace (admin of that workspace only). A new member
    gets a token, shown once; an existing member's token is updated in place (still valid). `role` is
    'admin' or 'user'."""
    mgr = get_auth_manager()
    try:
        raw = mgr.add_member(user_id, role=role, workspace=ws.name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    role_out = "admin" if role == "admin" else "user"
    from agent import activity as _activity
    _activity.record("user:" + caller.user_id, "member.add",
                     detail={"target": user_id, "role": role_out, "workspace": ws.name})
    if raw:
        return {"token": raw, "user_id": user_id, "role": role_out, "workspace": ws.name,
                "note": "Give this token to the member. Shown once."}
    return {"updated": True, "user_id": user_id, "role": role_out, "workspace": ws.name,
            "note": "Existing member updated; their token is unchanged."}


@router.delete("/members/{user_id}")
async def revoke_member(user_id: str, request: Request,
                        token: TokenEntry = Depends(require_auth),
                        ws: WorkspaceState = Depends(require_workspace)):
    """Revoke a member. By default removes their role in the CURRENT workspace (admin of that
    workspace). Pass ?all=true to remove the member entirely, which requires INSTANCE admin (admin of
    'default'). Refuses to remove the last admin of any affected workspace."""
    mgr = get_auth_manager()
    remove_all = str(request.query_params.get("all", "")).lower() in ("1", "true", "yes")
    if remove_all:
        if not is_admin(token, "default"):
            raise HTTPException(403, "Removing a member entirely requires instance admin")
        target = None
    else:
        if not is_admin(token, ws.name):
            raise HTTPException(403, f"Admin of workspace '{ws.name}' required")
        target = ws.name
    try:
        n = mgr.revoke_member(user_id, workspace=target)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if n == 0:
        raise HTTPException(404, f"No such member: {user_id}")
    from agent import activity as _activity
    _activity.record("user:" + token.user_id, "member.revoke",
                     detail={"target": user_id, "workspace": target or "(all)"})
    return {"revoked": user_id, "workspace": target or "(all)", "removed": n}


@router.post("/token")
async def create_token(
    user_id: str = Body(..., embed=True),
    workspaces: list = Body(["default"], embed=True),
    role: str = Body(ROLE_USER, embed=True),
    token: TokenEntry = Depends(_require_admin),
):
    """Create a new API token (low-level; `POST /members` is the managed path that rotates per user)."""
    mgr = get_auth_manager()
    raw = mgr.create_token(user_id, workspaces, role)
    return {
        "token": raw,
        "user_id": user_id,
        "workspaces": workspaces,
        "role": "admin" if role == "admin" else "user",
        "note": "Save this token. Shown once.",
    }


@router.get("/tokens")
async def list_tokens(token: TokenEntry = Depends(_require_admin)):
    """List all registered tokens (metadata only, never raw tokens)."""
    mgr = get_auth_manager()
    return {"tokens": mgr.list_tokens()}


@router.post("/auth/enable")
async def enable_auth(
    request: Request,
    passphrase: str = Body("", embed=True),
    token: TokenEntry = Depends(_require_admin),
):
    """Enable bearer token authentication (host-local only).

    Optionally set the disable passphrase at the same time by passing
    `passphrase`; a passphrase must be set before auth can later be disabled.
    """
    _require_localhost(request)
    mgr = get_auth_manager()
    mgr.enable_auth()
    if passphrase:
        try:
            mgr.set_disable_passphrase(passphrase)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    return {"auth_enabled": True, "disable_passphrase_set": mgr.has_disable_passphrase}


@router.post("/auth/passphrase")
async def set_disable_passphrase(
    request: Request,
    passphrase: str = Body(..., embed=True),
    token: TokenEntry = Depends(_require_admin),
):
    """Set or rotate the passphrase required to disable auth (host-local only)."""
    _require_localhost(request)
    mgr = get_auth_manager()
    try:
        mgr.set_disable_passphrase(passphrase)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"disable_passphrase_set": True}


@router.post("/auth/disable")
async def disable_auth(
    request: Request,
    passphrase: str = Body("", embed=True),
    token: TokenEntry = Depends(_require_admin),
):
    """Disable authentication (open access).

    Guarded three ways: the caller must be admin, the request must be a direct
    loopback connection from the server host, and it must carry the disable
    passphrase. A leaked or cached API token alone cannot turn auth off.
    """
    _require_localhost(request)
    mgr = get_auth_manager()
    if not mgr.has_disable_passphrase:
        raise HTTPException(
            403, "No disable passphrase is set. Set one first from the host: "
                 "POST /api/v1/admin/auth/passphrase (or `rexgraph-auth passphrase`).")
    if not mgr.verify_disable_passphrase(passphrase):
        raise HTTPException(403, "Invalid disable passphrase")
    mgr.disable_auth(confirm=True)   # localhost + admin token + passphrase already checked above
    return {"auth_enabled": False}


@router.post("/recovery-key")
async def create_recovery_key(token: TokenEntry = Depends(_require_admin)):
    """Create a recovery key (admin only). Returns the raw key once.

    The recovery key lets a locked-out admin obtain a new API token
    without filesystem access. Store it offline like a seed phrase.
    """
    mgr = get_auth_manager()
    raw = mgr.create_recovery_key()
    return {
        "recovery_key": raw,
        "note": "Store offline. Shown once.",
    }


@router.post("/recover")
async def recover_access(request: Request):
    """Use a recovery key to obtain a new admin token. No auth required.

    This is the only recovery path: there is no filesystem backdoor.
    The recovery key is verified against a bcrypt hash.
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, "JSON body required") from exc
    recovery_key = body.get("recovery_key", "")
    if not recovery_key:
        raise HTTPException(400, "recovery_key field required")
    mgr = get_auth_manager()
    if not mgr.has_recovery_key:
        raise HTTPException(404, "No recovery key configured for this instance")
    new_token = mgr.recover(recovery_key)
    if not new_token:
        raise HTTPException(401, "Invalid recovery key")
    return {"token": new_token, "status": "recovered"}


@router.get("/workspaces")
async def list_workspaces(token: TokenEntry = Depends(require_auth)):
    """List all workspaces the current user can access."""
    mgr = get_auth_manager()
    all_ws = mgr.list_workspaces()
    # `token.role` is the legacy scalar view, which _resync sets to admin when the token
    # is admin in ANY workspace, so an admin of one workspace was handed the whole
    # instance roster. The roster is an instance question, the same one /api/health
    # answers, so it takes the instance answer.
    if is_admin(token, "default"):
        return {"workspaces": all_ws}
    return {"workspaces": [w for w in all_ws if w in token.workspaces]}


@router.get("/workspace/activity")
async def workspace_activity(ws: WorkspaceState = Depends(require_workspace)):
    """Activity summary for the current workspace."""
    return ws.activity_summary()


@router.get("/workspace/complex")
async def workspace_complex(ws: WorkspaceState = Depends(require_workspace)):
    """Build the workspace activity as a relational complex."""
    result = ws.build_activity_complex()
    if result is None:
        return {"status": "no activity yet"}

    r = result["rex"]
    response = {
        "labels": result["labels"],
        "n_users": result["n_users"],
        "nV": r.nV,
        "nE": r.nE,
        "betti": result["betti"],
        "kappa": result["kappa"],
    }

    try:
        chi = r.structural_character
        if chi is not None:
            cm = chi.mean(axis=0)
            channels = ["T", "G", "F", "C"]
            n = min(4, len(cm))
            response["chi_mean"] = {channels[i]: float(cm[i]) for i in range(n)}
            response["dominant"] = channels[int(cm[:n].argmax())]
    except Exception:
        pass

    return response


@router.get("/workspace/overlap")
async def query_overlap(ws: WorkspaceState = Depends(require_workspace)):
    """Detect when multiple users are querying the same structural region."""
    overlaps = ws.detect_query_overlap()
    return {"overlaps": overlaps, "n_overlaps": len(overlaps)}


@router.get("/workspace/settings")
async def get_workspace_settings(
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """This workspace's settings, with the defaults it falls back to."""
    from agent import work_recorder as wr
    from agent.server.persistence import load_settings
    return {"workspace": ws.name, "settings": load_settings(ws.name),
            "defaults": {wr.SETTING: False, wr.KIND_SETTING: list(wr.KINDS)},
            "record_work_kinds_available": list(wr.KINDS)}


@router.post("/workspace/settings")
async def set_workspace_settings(
    body: dict = Body(...),
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """Merge changes into this workspace's settings. Sending {record_work: false}
    turns recording off; nothing already recorded is removed."""
    from agent import work_recorder as wr
    from agent.server.persistence import update_settings
    allowed = {wr.SETTING, wr.KIND_SETTING}
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise HTTPException(400, f"unknown setting(s): {', '.join(unknown)}. "
                                 f"Supported: {', '.join(sorted(allowed))}")
    kinds = body.get(wr.KIND_SETTING)
    if kinds is not None:
        if not isinstance(kinds, list) or any(k not in wr.KINDS for k in kinds):
            raise HTTPException(400, f"{wr.KIND_SETTING} must be a list drawn from: "
                                     f"{', '.join(wr.KINDS)}")
    ws.record_activity(token.user_id, "workspace_settings", ",".join(sorted(body)))
    return {"workspace": ws.name, "settings": update_settings(ws.name, body)}


@router.get("/workspace/files")
async def workspace_files(
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """List all documents in the workspace with metadata."""
    from agent.server.persistence import list_workspace_files
    return {"workspace": ws.name, "files": list_workspace_files(ws.name)}


@router.delete("/workspace/files/{doc_id}")
async def delete_workspace_file(
    doc_id: str,
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """Delete a document from the workspace."""
    import shutil

    # This rmtree's whatever the id resolves to, so a doc_id of ".." deleted the
    # workspace root rather than a document in it.
    from agent.server.persistence import doc_path as _doc_path
    try:
        candidates = [_doc_path(ws.name, doc_id), _doc_path(ws.name, doc_id, suffix="")]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    for target in candidates:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(str(target))
            else:
                target.unlink()
            return {"deleted": doc_id}
    raise HTTPException(404, f"Document not found: {doc_id}")


@router.get("/workspace/stats")
async def workspace_stats(
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """Aggregate statistics for the workspace."""
    from agent.server.persistence import get_workspace_stats
    return get_workspace_stats(ws.name)


@router.get("/workspace/doc/{doc_id}")
async def workspace_doc_detail(
    doc_id: str,
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """Load a document from the workspace and return structural analysis."""
    import numpy as np

    from agent.server.persistence import load_document_rex

    rex = load_document_rex(ws.name, doc_id)
    if rex is None:
        raise HTTPException(404, f"Document not found: {doc_id}")

    result = {
        "doc_id": doc_id,
        "nV": rex.nV,
        "nE": rex.nE,
        "nF": rex.nF,
        "betti": list(rex.betti),
        "euler": rex.euler_characteristic,
        "chain_valid": rex.chain_valid,
    }

    try:
        kappa = coherence_kappa(rex)
        if kappa is not None and len(kappa) > 0:
            result["kappa_mean"] = round(float(kappa.mean()), 4)
    except Exception:
        pass

    try:
        flow = np.ones(rex.nE, dtype=np.float64) if rex.nE > 0 else np.array([])
        if rex.nE > 0:
            h = rex.hodge_full(flow)
            result["hodge"] = {
                "gradient": float(h.get("pct_grad", 0)),
                "curl": float(h.get("pct_curl", 0)),
                "harmonic": float(h.get("pct_harm", 0)),
            }
    except Exception:
        pass

    try:
        vc = rex.void_complex
        result["n_voids"] = int(vc.get("n_voids", 0))
        result["n_potential"] = int(vc.get("n_potential", 0))
    except Exception:
        pass

    return result
