"""agent.console: a game-like command surface over the hive network (RimWorld/Factorio for agents).

Chat with and command the hive at any scale - the whole network, one hive, a worker team, or a single
worker. Read-only verbs (status/monitor/dashboard) inspect; build verbs (require/forge/chat) act; and
CONSEQUENTIAL verbs (kill) are PROPOSED unless you pass confirm=True. The human is always the
governor: nothing destructive or outward-facing happens without an explicit confirm.

    console.command("status", scope="network")
    console.command("require review test", scope="hive")
    console.command("chat how do you handle a 503?", scope="worker:payments")
    console.command("kill rogue", scope="hive")             # -> proposal; add confirm=True to do it
"""
from __future__ import annotations

from typing import Any


class CommandConsole:
    """Scoped command + chat interface over a hive, its reactive layer, and its foundry."""

    def __init__(self, hive=None, *, reactive=None, foundry=None):
        if hive is None:
            from . import hive as hivemod
            hive = hivemod.get_hive()
        self.hive = hive
        self.reactive = reactive
        self.foundry = foundry

    def command(self, text: str, *, scope: str = "hive", confirm: bool = False) -> dict[str, Any]:
        verb, _, arg = (text or "").strip().partition(" ")
        handler = getattr(self, "_cmd_" + verb.lower(), None)
        if handler is None:
            return {"ok": False, "error": f"unknown command '{verb}'", "commands": self._verbs()}
        try:
            return handler(arg.strip(), scope=scope, confirm=confirm)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _verbs(self) -> list[str]:
        return sorted(n[5:] for n in dir(self) if n.startswith("_cmd_"))

    def _scope_target(self, scope: str) -> str | None:
        if scope and (scope.startswith("worker:") or scope.startswith("team:")):
            return scope.split(":", 1)[1]
        return None

    # -- read-only -------------------------------------------------------------

    def _cmd_help(self, arg, *, scope, confirm):
        return {"ok": True, "commands": self._verbs(),
                "scopes": ["network", "hive", "team:<name>", "worker:<name>"],
                "note": "consequential verbs (kill) require confirm=True"}

    def _cmd_status(self, arg, *, scope, confirm):
        return {"ok": True, "scope": scope, "status": self.hive.status()}

    def _cmd_monitor(self, arg, *, scope, confirm):
        return {"ok": True, "scope": scope, "monitor": self.hive.monitor()}

    def _cmd_dashboard(self, arg, *, scope, confirm):
        from .dashboard import hive_dashboard
        return {"ok": True, "dashboard": hive_dashboard(self.hive)}

    # -- build verbs (grow / drive the hive) -----------------------------------

    def _cmd_require(self, arg, *, scope, confirm):
        if self.reactive is None:
            return {"ok": False, "error": "no reactive layer attached (pass reactive=...)"}
        return {"ok": True, "deployed": self.reactive.require(*arg.split())}

    def _cmd_forge(self, arg, *, scope, confirm):
        if self.foundry is None:
            return {"ok": False, "error": "no foundry attached (pass foundry=...)"}
        parts = arg.split()
        if len(parts) < 2:
            return {"ok": False, "error": "usage: forge <name> <archetype>"}
        return {"ok": True, "forged": self.foundry.forge(parts[0], parts[1], steps=30)}

    def _cmd_chat(self, arg, *, scope, confirm):
        """Talk to a scope: a worker gets asked directly; a hive/network routes to the best bee."""
        target = self._scope_target(scope)
        if target and self.hive.get(target) is not None:
            return {"ok": True, "from": target, "reply": self.hive.ask(target, arg)}
        d = self.hive.dispatch(arg)
        return {"ok": True, "from": d.get("bee"), "reply": d.get("reply"), "routed": d.get("routed")}

    def _cmd_set(self, arg, *, scope, confirm):
        """Overwrite a worker's specialties: set <worker> <kw1> <kw2> ..."""
        parts = arg.split()
        name = parts[0] if parts else self._scope_target(scope)
        bee = self.hive.get(name) if name else None
        if bee is None:
            return {"ok": False, "error": f"no worker '{name}'"}
        bee.specialties = parts[1:]
        return {"ok": True, "worker": name, "specialties": bee.specialties}

    # -- consequential (governed) ----------------------------------------------

    def _cmd_kill(self, arg, *, scope, confirm):
        name = arg or self._scope_target(scope)
        if not name:
            return {"ok": False, "error": "usage: kill <worker>"}
        if not confirm:
            return {"ok": False, "governed": True, "proposed": f"remove worker '{name}'",
                    "confirm": "re-run the command with confirm=True to apply"}
        return {"ok": self.hive.remove(name), "removed": name}
