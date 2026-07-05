"""Mnemo MCP server — exposes the self-managing memory core as MCP tools so any
MCP client (Claude Desktop, IDEs, other agents) gains persistent memory.

Run (stdio):   python src/mcp_server.py
Claude Desktop config:
    { "mcpServers": { "mnemo": { "command": "python",
        "args": ["/ABS/PATH/qwen-hackathon/src/mcp_server.py"] } } }
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from memory import MemoryCore

mcp = FastMCP("mnemo")
_core = MemoryCore()


@mcp.tool()
def remember(text: str, pinned: bool = False) -> str:
    """Store a memory (a fact, preference, or event) for later recall.
    Set pinned=true for durable identity facts that must never be forgotten.
    Near-duplicates are automatically consolidated."""
    mem_id = _core.store(text, pinned=pinned)
    return f"stored (id={mem_id}, pinned={pinned})"


@mcp.tool()
def recall(query: str, k: int = 5, char_budget: int | None = None) -> str:
    """Retrieve the most relevant memories for a query, ranked by semantic
    relevance weighted by how 'fresh' each memory is. Pass char_budget to cap
    total size when working under a limited context window."""
    hits = _core.recall(query, k=k, char_budget=char_budget)
    if not hits:
        return "(no relevant memories)"
    return "\n".join(
        f"[{m.score:.2f}{'📌' if m.pinned else ''}] {m.text}" for m in hits
    )


@mcp.tool()
def forget_stale() -> str:
    """Run the forgetting sweep: archive memories whose relevance has decayed
    below threshold (pinned memories are never forgotten). Returns how many were
    archived plus current store stats."""
    n = _core.forget_sweep()
    st = _core.stats()
    return f"archived {n} stale memories · live={st['live']} archived={st['archived']}"


@mcp.tool()
def memory_stats() -> str:
    """Report how many live vs archived memories are in the store."""
    st = _core.stats()
    return f"live={st['live']} archived={st['archived']}"


if __name__ == "__main__":
    mcp.run()
