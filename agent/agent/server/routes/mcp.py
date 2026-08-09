"""
agent.server.routes.mcp: the tool registry in MCP's shape.

MCP is a compatibility layer, not the contract. What a tool IS, who may run it and
what it may read are settled in the registry and enforced on the way in by
`/rex/v1/call`; this route exists so a client that speaks MCP can reach the same
capabilities without either side learning the other's format.

That makes it a second door to one room, which is the arrangement worth being careful
about: a door with a different lock is a door with no lock. So it takes the same
identity, builds the same context, and dispatches through the same gate. The only
difference is the shape of the envelope.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from ..auth import TokenEntry, require_auth

router = APIRouter(prefix="/v1/mcp")


@router.get("/tools")
async def list_tools(request: Request, token: TokenEntry = Depends(require_auth)):
    """Every capability this caller may run, with its schema."""
    from agent.mcp_tools import definitions

    from .rex import _context
    return {"tools": definitions(_context(request, token))}


@router.post("/call")
async def call_tool(request: Request, body: dict = Body(...),
                    token: TokenEntry = Depends(require_auth)):
    """Run one tool. Body: {"name": ..., "arguments": {...}}.

    Delegates to the native route's handler so there is one implementation of
    admission, dispatch and recording rather than two that have to agree.
    """
    from .rex import call_tool as native
    return await native(request, body, token)


@router.get("/health")
async def health():
    """That the registry loads and every advertised name resolves.

    Unauthenticated on purpose: it reports nothing about the caller, the workspace or
    what is stored, only that the server is up and its registry is coherent.
    """
    try:
        from agent.mcp_tools import TOOLS
        broken = [n for n, t in TOOLS.items() if not callable(t.handler)]
    except Exception as e:                       # noqa: BLE001
        raise HTTPException(503, f"registry did not load: {e}") from e
    if broken:
        raise HTTPException(503, f"tools with no handler: {', '.join(broken)}")
    return {"ok": True, "n_tools": len(TOOLS)}
