"""Tenet MCP server — exposes the self-managing memory core as MCP tools so any
MCP client (Claude Desktop, IDEs, other agents) gains persistent memory.

Run (stdio):   tenet-mcp   (console script, after `pip install tenet-memory[mcp]`)
                or: python -m tenet.mcp_server
Claude Desktop config:
    { "mcpServers": { "tenet": { "command": "tenet-mcp" } } }
"""
from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP

from .core import Tenet

mcp = FastMCP("tenet")
_tenet = Tenet()
_core = _tenet.core


@mcp.tool()
def learn(message: str, pinned: bool = False) -> str:
    """Ingest a raw message/note: automatically distill it into atomic facts, then
    store each with supersession (a changed fact retires the old value, history kept).
    This is the main write path — prefer it over `remember` for conversational input."""
    ids = _tenet.ingest(message, pinned=pinned)
    return f"learned {len(ids)} fact(s)" if ids else "no durable fact found"


@mcp.tool()
def remember(text: str, pinned: bool = False) -> str:
    """Store a single pre-formed fact directly (skip distillation).
    Set pinned=true for durable identity facts that must never be forgotten."""
    mem_id = _core.store(text, pinned=pinned)
    return f"stored (id={mem_id}, pinned={pinned})"


@mcp.tool()
def recall(query: str, k: int = 5, char_budget: int | None = None) -> str:
    """Retrieve the most relevant memories for a query, ranked by semantic
    relevance weighted by how 'fresh' each memory is. Pass char_budget to cap
    total size when working under a limited context window. Keyed facts carry
    a learned p_valid (world-model confidence the fact is still current) when
    available — annotation only, it never changes ranking or which memories
    are returned; use it to decide whether to hedge or re-confirm a fact."""
    hits = _core.recall(query, k=k, char_budget=char_budget)
    if not hits:
        return "(no relevant memories)"
    lines = []
    for m in hits:
        line = f"[{m.score:.2f}{'📌' if m.pinned else ''}] {m.text}"
        if m.confidence is not None:
            line += f" (p_valid={m.confidence:.2f})"
        lines.append(line)
    return "\n".join(lines)


@mcp.tool()
def doubts(threshold: float = 0.5) -> str:
    """List current keyed facts the learned world model doubts (P(still valid)
    < threshold, from per-key-class survival fitted on this store's own
    supersession history) — worth proactively re-confirming with the user.
    Each row: key, current value, age, p_valid, and the key's typical
    (learned) lifetime. Most-doubted first."""
    rows = _core.uncertain_facts(threshold=threshold)
    if not rows:
        return f"no doubted facts (all current facts above P(valid) >= {threshold})"
    lines = []
    for d in rows:
        life = d["expected_lifetime_days"]
        life_s = "—" if life is None else f"{life:.0f}d"
        lines.append(f"[p={d['p_valid']:.2f}] {d['key']}: {d['text']}  "
                     f"(age {d['age_days']:.0f}d, typical lifetime {life_s})")
    return "\n".join(lines)


@mcp.tool()
def time_travel(query: str, as_of: str, k: int = 5) -> str:
    """Bi-temporal recall: what the user believed/said about `query` as of a
    given point in time (ISO date/datetime, e.g. "2026-03-01" or
    "2026-03-01T12:00:00") — the belief state at that instant, not the
    current one. Facts superseded after `as_of` are returned as they read
    then; facts learned after `as_of` are excluded."""
    try:
        ts = datetime.fromisoformat(as_of).timestamp()
    except ValueError:
        return f"invalid as_of {as_of!r} — expected ISO date/datetime, e.g. 2026-03-01"
    hits = _core.recall(query, k=k, as_of=ts)
    if not hits:
        return f"(no memories known as of {as_of})"
    return "\n".join(f"- {m.text}" for m in hits)


@mcp.tool()
def forget_stale() -> str:
    """Run the forgetting sweep: archive memories whose relevance has decayed
    below threshold (pinned memories are never forgotten). Returns how many were
    archived plus current store stats."""
    n = _core.forget_sweep()
    st = _core.stats()
    return (f"archived {n} stale memories · current={st['current']} "
            f"superseded={st['superseded']} archived={st['archived']}")


@mcp.tool()
def memory_stats() -> str:
    """Report the store's current / superseded (history) / archived counts."""
    st = _core.stats()
    return f"current={st['current']} superseded={st['superseded']} archived={st['archived']}"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
