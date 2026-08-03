"""
agent.hive_config: named, switchable, editable hive profiles.

A profile holds a whole setup in one object: how the swarm is composed (auto from disk, attach
live servers, or an explicit list of bees), the memory budget, and the engine preferences
(optimizer, attention, whether the monitor uses the semantic embedder, the routing policy). Pick
a built-in preset, edit it, save it as a user profile, switch between profiles, and apply one to
bring the hive up.

Layers:
  - BUILTIN_PROFILES: code-defined presets (read-only), always present.
  - user profiles: JSON under <REXGRAPH_CONFIG_DIR>/hive_profiles/, created by save().
  - active pointer: which profile is currently selected (<config>/hive_profiles/active.json).

`apply(id, hive)` reads a profile and stands the hive up accordingly, reusing the hive module's
own compose/attach/spawn. This module decides which bees; `hive` performs the spawn/attach.
"""
from __future__ import annotations

import builtins
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _config_dir() -> Path:
    base = Path(os.environ.get("REXGRAPH_CONFIG_DIR", Path.home() / ".config" / "rexgraph"))
    return base / "hive_profiles"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "profile"


@dataclass
class BeeSpec:
    """One bee in a profile. `source` decides where it comes from: 'auto' (chosen by the planner),
    'path' (spawn this GGUF), or 'attach' (reference a running url)."""
    name: str
    role: str = "worker"              # queen | worker | embedder
    source: str = "path"             # path | attach | auto
    model: str = ""                  # gguf path (source=path)
    url: str = ""                    # endpoint (source=attach)
    specialties: list[str] = field(default_factory=list)


@dataclass
class ComputeSpec:
    """Execution-layer config for a setup: the CPU parallel width and the preferred compute backend.
    Applied via rexgraph.compute before every operation runs (see lifecycle._execute)."""
    threads: int | None = None    # None -> all cores; an int caps the OpenMP / parallel_map width
    backend: str = "auto"            # auto (best available for the host, incl. GPU) | cpu |
                                     # openmp | cuda (also the value for ROCm/AMD hosts) | mps.
                                     # 'auto' now resolves to the host's recommended backend, so
                                     # a GPU host accelerates the eigen-free tower automatically.


@dataclass
class CoordinatorSpec:
    """Coordinator tuning for a setup: whether the coordinator dispatches hive fan-outs, its pool
    idle TTLs, worker-core affinity, and the user priority weights. All defaults are neutral, so
    behavior is unchanged until a user tunes them."""
    enabled: bool = True
    idle_ttl_proc: float = 30.0
    idle_ttl_thread: float = 120.0
    affinity: bool = False
    hive_shares: dict = field(default_factory=dict)     # hive name -> relative resource share
    task_weights: dict = field(default_factory=dict)    # task kind -> priority weight
    worker_weights: dict = field(default_factory=dict)  # worker name -> priority weight


@dataclass
class HiveProfile:
    """A complete, switchable hive setup."""
    id: str
    name: str
    description: str = ""
    builtin: bool = False
    # composition: how the swarm is brought up
    compose: str = "auto"            # auto | attach-live | manual | auto+attach
    budget_gb: float | None = None    # None -> detected budget
    max_workers: int = 4
    bees: list[BeeSpec] = field(default_factory=list)   # used when compose includes 'manual'
    # engine preferences (surfaced to the rest of the stack; see apply())
    optimizer: str = "auto"         # auto (routes per model: GreensCochain for cochain-native, else Adam; default) | hodge | adam | greens
    attention: str = "relational"    # relational (RexGraph-native, default) | standard
    monitor_embed: bool = True       # monitor uses the embedder bee for semantic alignment
    routing: str = "specialty+history"
    compute: ComputeSpec = field(default_factory=ComputeSpec)   # execution-layer tuning
    coordinator: CoordinatorSpec = field(default_factory=CoordinatorSpec)   # coordinator tuning
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> HiveProfile:
        bees = [BeeSpec(**b) if isinstance(b, dict) else b for b in d.get("bees", [])]
        comp = d.get("compute")
        compute = ComputeSpec(**comp) if isinstance(comp, dict) else (comp or ComputeSpec())
        coord = d.get("coordinator")
        coordinator = CoordinatorSpec(**coord) if isinstance(coord, dict) else (coord or CoordinatorSpec())
        known = {f for f in cls.__dataclass_fields__}
        rest = {k: v for k, v in d.items()
                if k in known and k not in ("bees", "compute", "coordinator")}
        return cls(**rest, bees=bees, compute=compute, coordinator=coordinator)


# built-in presets: always available, read-only

BUILTIN_PROFILES: list[HiveProfile] = [
    HiveProfile(
        id="solo", name="Solo driver", builtin=True,
        description="One capable queen, no workers. The simplest setup - fast to bring up, "
                    "minimal memory. Good for straight chat and single-model work.",
        compose="auto", max_workers=0, monitor_embed=False, tags=["minimal", "chat"]),
    HiveProfile(
        id="research", name="Research swarm", builtin=True,
        description="A queen plus a full set of worker bees and the embedder - the whole "
                    "relational monitor lit up (load-bearing, disagreement, alignment). Best for "
                    "multi-domain analysis where you want the swarm's structure visible.",
        compose="auto", max_workers=4, monitor_embed=True,
        tags=["swarm", "monitor", "analysis"]),
    HiveProfile(
        id="coding", name="Coding hive", builtin=True,
        description="Queen + code-focused workers + embedder, tuned for software tasks. Routes "
                    "code queries to the coder bees.",
        compose="auto", max_workers=3, monitor_embed=True, tags=["code"]),
    HiveProfile(
        id="lean", name="Lean / edge", builtin=True,
        description="Smallest models under a tight budget - for constrained machines or leaving "
                    "headroom for other work. A couple of light workers at most.",
        compose="auto", budget_gb=12.0, max_workers=2, monitor_embed=False,
        compute=ComputeSpec(threads=4, backend="cpu"), tags=["lightweight"]),
    HiveProfile(
        id="attach", name="Use what's running", builtin=True,
        description="Don't spawn anything - enroll the inference servers already running on this "
                    "host (Ollama / vLLM / llama.cpp / LM Studio) as bees. Zero startup cost.",
        compose="attach-live", monitor_embed=True, tags=["attach", "external"]),
    HiveProfile(
        id="max", name="Max quality", builtin=True,
        description="The largest queen that fits plus a broad worker set and the embedder - "
                    "throughput traded for capability. For the strongest answers this box can give.",
        compose="auto+attach", max_workers=4, monitor_embed=True, tags=["quality", "swarm"]),
]

_BUILTIN_BY_ID = {p.id: p for p in BUILTIN_PROFILES}


class ProfileStore:
    """Persistent registry: built-in presets + user-saved profiles + the active pointer."""

    def __init__(self, directory: Path | None = None):
        self.dir = directory or _config_dir()

    # listing / lookup
    def _user_files(self):
        if not self.dir.exists():
            return []
        return sorted(p for p in self.dir.glob("*.json") if p.name != "active.json")

    def user_profiles(self) -> builtins.list[HiveProfile]:
        out = []
        for f in self._user_files():
            try:
                out.append(HiveProfile.from_dict(json.loads(f.read_text())))
            except Exception:
                continue
        return out

    def list(self) -> builtins.list[HiveProfile]:
        """Built-ins first, then user profiles (user profiles with a built-in id override it)."""
        users = self.user_profiles()
        user_ids = {p.id for p in users}
        return [p for p in BUILTIN_PROFILES if p.id not in user_ids] + users

    def get(self, pid: str) -> HiveProfile | None:
        for f in self._user_files():
            if f.stem == pid:
                try:
                    return HiveProfile.from_dict(json.loads(f.read_text()))
                except Exception:
                    break
        return _BUILTIN_BY_ID.get(pid)

    # mutation (user profiles only)
    def save(self, profile: HiveProfile) -> HiveProfile:
        """Persist a user profile. A built-in is never overwritten in place: saving one clones it
        into a user profile with the same id that shadows the built-in (reset means delete it)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if not profile.id:
            profile.id = _slug(profile.name)
        profile.builtin = False
        (self.dir / f"{profile.id}.json").write_text(json.dumps(profile.to_dict(), indent=2))
        return profile

    def create(self, name: str, base: str | None = None, **overrides) -> HiveProfile:
        """New user profile, optionally cloned from an existing one (built-in or user)."""
        src = self.get(base) if base else None
        d = src.to_dict() if src else {}
        d.update(overrides)
        d["name"] = name
        d["id"] = _slug(name)
        d["builtin"] = False
        prof = HiveProfile.from_dict(d)
        return self.save(prof)

    def delete(self, pid: str) -> bool:
        """Remove a user profile (or a user override of a built-in). Built-ins themselves persist."""
        f = self.dir / f"{pid}.json"
        if f.exists():
            f.unlink()
            if self.active_id() == pid:
                self.set_active(None)
            return True
        return False

    # active pointer
    def _active_file(self) -> Path:
        return self.dir / "active.json"

    def active_id(self) -> str | None:
        f = self._active_file()
        if f.exists():
            try:
                return json.loads(f.read_text()).get("active")
            except Exception:
                return None
        return None

    def set_active(self, pid: str | None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._active_file().write_text(json.dumps({"active": pid}))

    def active(self) -> HiveProfile | None:
        pid = self.active_id()
        return self.get(pid) if pid else None

    # apply: stand the hive up from a profile
    def apply(self, pid: str, hive_obj=None, *, reset: bool = True, wait: float = 120.0) -> dict:
        """Bring the hive up per a profile. `reset` clears the current swarm first (switching
        setups). Sets this profile active. Returns what happened plus the resulting status."""
        prof = self.get(pid)
        if prof is None:
            raise KeyError(f"no profile {pid!r}")
        from agent import hive as H
        hive_obj = hive_obj or H.get_hive()
        if reset:
            hive_obj.stop_all()

        result = {"profile": prof.id, "compose": prof.compose, "spawned": [], "attached": []}
        # explicit manual bees first (path/attach), regardless of mode
        for b in prof.bees:
            try:
                if b.source == "attach" and b.url:
                    hive_obj.attach(b.name, b.url, role=b.role, model=b.model,
                                    specialties=b.specialties)
                    result["attached"].append(b.name)
                elif b.source == "path" and b.model:
                    hive_obj.spawn(b.name, b.model, role=b.role,
                                   specialties=b.specialties, wait=wait)
                    result["spawned"].append({"name": b.name, "ok": True})
            except Exception as ex:
                result["spawned"].append({"name": b.name, "ok": False, "error": str(ex)})

        if prof.compose in ("attach-live", "auto+attach"):
            result["attached"] += [x.name for x in hive_obj.attach_live()]
        if prof.compose in ("auto", "auto+attach"):
            auto = hive_obj.auto(prof.budget_gb, wait=wait, max_workers=prof.max_workers)
            result["plan"] = auto.get("plan")
            result["spawned"] += auto.get("spawned", [])

        self.set_active(prof.id)
        result["engine"] = {"optimizer": prof.optimizer, "attention": prof.attention,
                            "monitor_embed": prof.monitor_embed, "routing": prof.routing}
        result["status"] = hive_obj.status()
        return result


_STORE: ProfileStore | None = None


def get_store() -> ProfileStore:
    global _STORE
    if _STORE is None:
        _STORE = ProfileStore()
    return _STORE


def reset_store() -> None:
    global _STORE
    _STORE = None


def coordinator_settings() -> CoordinatorSpec:
    """The active setup's coordinator spec, or neutral defaults when no setup is active."""
    try:
        active = get_store().active()
        if active is not None:
            return active.coordinator
    except Exception:
        pass
    return CoordinatorSpec()
