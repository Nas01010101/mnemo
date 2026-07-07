"""Tenet — agent memory as a self-consistent belief state.

    from tenet import Tenet
    m = Tenet()
    m.ingest("I moved to Toronto last week.")   # distills -> atomic keyed facts -> store
    m.recall("where does the user live?")        # bi-temporal, forgetting-aware retrieval

`Memory` / `MemoryCore` (tenet.memory) are the lower-level bi-temporal store Tenet
wraps; most callers only need `Tenet`. Optional surfaces (HTTP API, MCP server,
Alibaba OSS snapshots, the reference assistant) live in their own submodules
(tenet.api, tenet.mcp_server, tenet.alicloud_oss, tenet.agent) and pull in extra
dependencies only when imported — see the `api`/`mcp`/`oss` extras.
"""
from __future__ import annotations

from .core import Tenet
from .memory import Memory, MemoryCore

__all__ = ["Tenet", "Memory", "MemoryCore"]

__version__ = "0.1.0"
