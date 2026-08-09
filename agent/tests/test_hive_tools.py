"""Bees holding the stack's capabilities, and not holding more than their caller may.

A hive of chat bees can only be told answers. Registering the tool registry as workers
means a bee can compute one: the same registry a model driving MCP reads, so a name that
resolves for one resolves for the other and there is no second dispatcher to drift.

The half worth testing hardest is the boundary. A bee is a caller like any other, not a
trusted one, so an admin-only tool is not advertised to a non-admin context and a file
is a handle in that workspace or it is nothing. "No dangerous agents" is that sentence
being true rather than a policy written somewhere.
"""
from __future__ import annotations

import pytest

ONTOLOGY = """[Term]
id: GO:0000001
name: alpha

[Term]
id: GO:0000002
name: beta
is_a: GO:0000001

[Term]
id: GO:0000003
name: gamma
is_a: GO:0000001
"""


@pytest.fixture
def obo(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    p = tmp_path / "terms.obo"
    p.write_text(ONTOLOGY)
    return str(p)


@pytest.fixture
def hive(obo):
    from agent.hive import Hive
    h = Hive("test")
    h.add_tools()
    return h


def test_every_registered_tool_becomes_a_worker(hive):
    from agent.mcp_tools import TOOLS
    assert set(hive.providers("analyze")) == set(TOOLS)


def test_a_bee_computes_through_its_tool(hive, obo):
    out = hive.invoke("rexgraph_homology", {"files": [obo]})
    assert out["betti"][0] >= 1
    assert out["grades"], "the tool returned no per-grade reading"


def test_dispatch_routes_to_the_tool_the_hint_names(hive, obo):
    got = hive.dispatch_capability("analyze", {"files": [obo]},
                                   hint="homology betti ranks")
    assert got["worker"] == "rexgraph_homology"


def test_tools_join_the_worker_type_taxonomy(hive):
    """Typed `tool:<name>`, so they are routable and diagnosable like any member
    rather than a category the ontology cannot see."""
    built = hive.type_complex()
    assert built is not None
    rex, _meta = built
    assert rex.nE > 0


def test_an_invocation_is_recorded_in_the_complex(hive, obo):
    """A bee that computes without relaying is invisible to the monitor."""
    before = len(hive._complex.agents())
    hive.invoke("rexgraph_homology", {"files": [obo]})
    agents = hive._complex.agents()
    assert len(agents) > before, "the tool call left no trace in the agentic complex"
    assert "rexgraph_homology" in agents, f"the tool is not a participant: {agents}"


#### the boundary


def test_an_admin_tool_is_not_advertised_to_a_plain_bee(obo):
    from agent.hive import Hive
    from agent.mcp_tools import TOOLS, Context
    h = Hive("plain")
    added = h.add_tools(context=Context(is_admin=False, auth_enabled=True))
    admin_only = {n for n, t in TOOLS.items() if t.requires == "admin"}
    assert not (set(added) & admin_only)


def test_a_bee_cannot_read_a_path_outside_its_workspace(tmp_path, monkeypatch):
    """The registry's own gate, reached through the hive: a bee is not a way around
    the file boundary."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    secret = tmp_path / "secret.obo"
    secret.write_text(ONTOLOGY)

    from agent.hive import Hive
    from agent.mcp_tools import Context
    from agent.server.handles import HandleError
    h = Hive("scoped")
    h.add_tools(context=Context(workspace="beta", is_admin=False, auth_enabled=True))
    with pytest.raises(HandleError):
        h.invoke("rexgraph_join_sources", {"files": [str(secret)]})


def test_a_bee_reads_a_handle_its_workspace_holds(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    from agent.hive import Hive
    from agent.mcp_tools import Context
    from agent.server.handles import mint
    handle = mint("beta", ONTOLOGY.encode(), name="terms.obo")["handle"]
    h = Hive("scoped")
    h.add_tools(context=Context(workspace="beta", is_admin=False, auth_enabled=True))
    out = h.invoke("rexgraph_join_sources", {"files": [handle]})
    assert out["n_relations"] > 0


def test_selecting_a_subset_registers_only_that_subset(obo):
    from agent.hive import Hive
    h = Hive("subset")
    added = h.add_tools(names=["rexgraph_homology"])
    assert added == ["rexgraph_homology"]
    assert h.providers("analyze") == ["rexgraph_homology"]
