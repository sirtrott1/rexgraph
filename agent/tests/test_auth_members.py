"""Shared multi-user auth: per-workspace admin/user roles, member management, and enforcement.

The workspace is shared; roles are PER WORKSPACE. A member can be admin of one workspace and a plain
user of another. The root workspace 'default' is the instance - admin-of-default is the instance
administrator (enables auth, removes members entirely). A stored role that nothing checks is not a
role, so these cover the route-level gating too.
"""
import pytest
from fastapi.testclient import TestClient

from agent.server import auth
from agent.server.auth import get_auth_manager, ROLE_ADMIN, ROLE_USER


def _fresh():
    auth.reset_auth_manager()
    mgr = get_auth_manager()
    mgr.enable_auth()
    return mgr


def _client():
    from agent.server.app import app
    return TestClient(app)


# -- manager semantics --------------------------------------------------------

def test_role_normalization_and_roster():
    mgr = _fresh()
    ta = mgr.add_member("alice", role="admin", workspace="default")
    mgr.add_member("bob", role="write", workspace="default")       # legacy label folds to 'user'
    assert mgr.verify(ta).is_admin_in("default")
    roster = {m["user_id"]: m for m in mgr.members("default")}
    assert roster["alice"]["role"] == "admin" and roster["bob"]["role"] == "user"
    assert mgr.members("default")[0]["user_id"] == "alice"          # admins sort first


def test_add_member_updates_in_place_no_rotation():
    mgr = _fresh()
    t1 = mgr.add_member("bob", role="user", workspace="default")    # new -> mints a token
    assert t1
    assert mgr.add_member("bob", role="admin", workspace="default") is None   # existing -> in place
    assert mgr.verify(t1).is_admin_in("default")                   # SAME token, now upgraded
    assert len(mgr.members("default")) == 1


def test_per_workspace_roles_are_isolated():
    mgr = _fresh()
    ta = mgr.add_member("alice", role="admin", workspace="default")
    assert mgr.add_member("alice", role="user", workspace="proj") is None      # in place, keeps token
    tb = mgr.add_member("bob", role="admin", workspace="proj")
    al, bo = mgr.verify(ta), mgr.verify(tb)
    assert al.is_admin_in("default") and not al.is_admin_in("proj") and al.can_access("proj")
    assert bo.is_admin_in("proj") and not bo.can_access("default")
    assert {m["user_id"] for m in mgr.members("proj")} == {"alice", "bob"}
    assert [m["user_id"] for m in mgr.members("default")] == ["alice"]


def test_cannot_revoke_last_admin_of_a_workspace():
    mgr = _fresh()
    mgr.add_member("alice", role="admin", workspace="default")
    with pytest.raises(ValueError):
        mgr.revoke_member("alice")                                 # removing entirely orphans default
    with pytest.raises(ValueError):
        mgr.revoke_member("alice", "default")
    mgr.add_member("carol", role="admin", workspace="default")     # a second admin now exists
    assert mgr.revoke_member("alice", "default") == 1


def test_revoke_one_workspace_keeps_other_access():
    mgr = _fresh()
    ta = mgr.add_member("alice", role="admin", workspace="default")
    mgr.add_member("alice", role="user", workspace="proj")
    assert mgr.revoke_member("alice", "proj") == 1
    al = mgr.verify(ta)                                             # token still valid...
    assert al.is_admin_in("default") and not al.can_access("proj")  # ...just no longer in proj


# -- route enforcement --------------------------------------------------------

def test_routes_enforce_per_workspace_admin():
    mgr = _fresh()
    at = mgr.add_member("alice", role="admin", workspace="default")   # instance admin
    bt = mgr.add_member("bob", role="admin", workspace="proj")        # admin of proj only
    ut = mgr.add_member("carol", role="user", workspace="default")    # user of default
    c = _client()
    A = {"Authorization": "Bearer " + at}
    B = {"Authorization": "Bearer " + bt}
    U = {"Authorization": "Bearer " + ut}

    assert c.get("/api/v1/admin/whoami", headers=U).json()["is_admin"] is False
    assert c.get("/api/v1/admin/whoami", headers=A).json()["is_admin"] is True

    assert c.get("/api/v1/admin/members", headers=U).status_code == 403          # user of default
    assert c.get("/api/v1/admin/members", headers=A).status_code == 200          # admin of default
    assert c.get("/api/v1/admin/members", headers=B).status_code == 200          # bob's default ws = proj
    assert c.get("/api/v1/admin/members", headers=B,
                 params={"workspace": "default"}).status_code == 403             # bob has no default access

    # bob may add to proj, not to default
    r = c.post("/api/v1/admin/members", headers=B, params={"workspace": "proj"},
               json={"user_id": "dan", "role": "user"})
    assert r.status_code == 200 and r.json().get("token")
    assert c.post("/api/v1/admin/members", headers=B, params={"workspace": "default"},
                  json={"user_id": "dan", "role": "user"}).status_code == 403


def test_consequential_verb_requires_workspace_admin():
    mgr = _fresh()
    at = mgr.add_member("alice", role="admin", workspace="default")
    ut = mgr.add_member("carol", role="user", workspace="default")
    c = _client()
    A = {"Authorization": "Bearer " + at}
    U = {"Authorization": "Bearer " + ut}
    assert c.post("/api/v1/agents/command",
                  json={"command": "kill ghost", "scope": "hive"}, headers=U).status_code == 200
    assert c.post("/api/v1/agents/command",
                  json={"command": "kill ghost", "confirm": True, "scope": "hive"},
                  headers=U).status_code == 403
    assert c.post("/api/v1/agents/command",
                  json={"command": "kill ghost", "confirm": True, "scope": "hive"},
                  headers=A).status_code != 403


def test_revoke_entirely_requires_instance_admin():
    mgr = _fresh()
    at = mgr.add_member("alice", role="admin", workspace="default")
    bt = mgr.add_member("bob", role="admin", workspace="proj")
    mgr.add_member("dan", role="user", workspace="proj")
    c = _client()
    A = {"Authorization": "Bearer " + at}
    B = {"Authorization": "Bearer " + bt}
    # bob administers proj but is not the instance admin -> cannot remove a member entirely
    assert c.delete("/api/v1/admin/members/dan", headers=B,
                    params={"all": "true", "workspace": "proj"}).status_code == 403
    # alice (admin of default = instance) can
    assert c.delete("/api/v1/admin/members/dan", headers=A, params={"all": "true"}).status_code == 200
