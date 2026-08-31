"""agent.console: a game-like command surface over the hive network (RimWorld/Factorio for agents).

Chat with and command the hive at any scale: the whole network, one hive, a worker team, or a single
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

    @staticmethod
    def _propose(what: str) -> dict[str, Any]:
        """The answer to a governed verb that was not confirmed.

        One form rather than the same block per handler: `kill` had it and `set`,
        `require` and `forge` did not, which meant the route's admin check never ran for
        them. That check only fires when confirm is true, so a handler that ignores
        confirm is a handler with no gate at all.
        """
        return {"ok": False, "governed": True, "proposed": what,
                "confirm": "re-run the command with confirm=True to apply"}

    def _verbs(self) -> list[str]:
        return sorted(n[5:] for n in dir(self) if n.startswith("_cmd_"))

    def _scope_target(self, scope: str) -> str | None:
        if scope and (scope.startswith("worker:") or scope.startswith("team:")):
            return scope.split(":", 1)[1]
        return None

    #### read-only
    def _cmd_help(self, arg, *, scope, confirm):
        return {"ok": True, "commands": self._verbs(),
                "scopes": ["network", "hive", "team:<name>", "worker:<name>"],
                "governed": ["require", "forge", "set", "kill"],
                "note": "governed verbs propose unless confirm=True, and confirming one "
                        "needs admin of the workspace"}

    def _cmd_status(self, arg, *, scope, confirm):
        return {"ok": True, "scope": scope, "status": self.hive.status()}

    def _cmd_monitor(self, arg, *, scope, confirm):
        return {"ok": True, "scope": scope, "monitor": self.hive.monitor()}

    def _cmd_dashboard(self, arg, *, scope, confirm):
        from .dashboard import hive_dashboard
        return {"ok": True, "dashboard": hive_dashboard(self.hive)}

    def _cmd_chat(self, arg, *, scope, confirm):
        """Talk to a scope: a worker gets asked directly; a hive/network routes to the best bee."""
        target = self._scope_target(scope)
        if target and self.hive.get(target) is not None:
            return {"ok": True, "from": target, "reply": self.hive.ask(target, arg)}
        d = self.hive.dispatch(arg)
        return {"ok": True, "from": d.get("bee"), "reply": d.get("reply"), "routed": d.get("routed")}

    #### governed: no effect unless confirm=True, and the route requires admin to confirm
    def _cmd_require(self, arg, *, scope, confirm):
        if self.reactive is None:
            return {"ok": False, "error": "no reactive layer attached (pass reactive=...)"}
        if not confirm:
            return self._propose(f"deploy workers for {arg or '(nothing named)'}")
        return {"ok": True, "deployed": self.reactive.require(*arg.split())}

    def _cmd_forge(self, arg, *, scope, confirm):
        if self.foundry is None:
            return {"ok": False, "error": "no foundry attached (pass foundry=...)"}
        parts = arg.split()
        if len(parts) < 2:
            return {"ok": False, "error": "usage: forge <name> <archetype>"}
        if not confirm:
            return self._propose(f"forge model '{parts[0]}' from archetype '{parts[1]}'")
        return {"ok": True, "forged": self.foundry.forge(parts[0], parts[1], steps=30)}

    def _cmd_set(self, arg, *, scope, confirm):
        """Overwrite a worker's specialties: set <worker> <kw1> <kw2> ..."""
        parts = arg.split()
        name = parts[0] if parts else self._scope_target(scope)
        # Governance is decided before existence, the rule the OCR path already follows:
        # answering "no worker 'w1'" to a caller who may not act tells them which workers
        # exist, which is a question they were never entitled to ask.
        if not confirm:
            return self._propose(f"overwrite the specialties of worker '{name}'")
        bee = self.hive.get(name) if name else None
        if bee is None:
            return {"ok": False, "error": f"no worker '{name}'"}
        bee.specialties = parts[1:]
        return {"ok": True, "worker": name, "specialties": bee.specialties}

    def _cmd_kill(self, arg, *, scope, confirm):
        name = arg or self._scope_target(scope)
        if not name:
            return {"ok": False, "error": "usage: kill <worker>"}
        if not confirm:
            return self._propose(f"remove worker '{name}'")
        return {"ok": self.hive.remove(name), "removed": name}
