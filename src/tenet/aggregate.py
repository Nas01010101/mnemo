"""Read-time recency aggregation — an OPTIONAL de-conflicting pass over `recall()`'s
output pool. COMPOSES with the existing store; does not replace ingestion-time
supersession (docs/COMPARISON.md follow-up #1, "CAR-style read-time max(serial)
aggregation as an optional reader").

Motivation: CAR (arXiv:2606.01435) shows that even when a retrieval pool still
contains conflicting values, a deterministic max(serial) ("latest wins") pick over
LLM-extracted (serial, value) candidates recovers most of the accuracy a fully
pre-resolved store would give — CAR does this because ITS pool is raw, undated
lines with no structure. Tenet already resolves conflicts at INGESTION (keyed
supersession, `memory.py`'s `store()`/`_resolve_key_supersede`), so a correctly-keyed
store rarely needs this — but a residual duplicate-key group can still reach a
`recall()` pool when the distiller/heuristic key extraction didn't collide two
mentions of the same real-world attribute (missed the collision key-resolution is
FOR), or when multi-hop `expand`/`hops` retrieval pulls in evidence from several
sessions that individually look relevant.

This module is the SAME idea as CAR's max(serial), but LLM-free and in-code: Tenet's
memories already carry the serial-equivalent signal (`valid_at`, bi-temporal event
time) structurally, so no extraction call is needed — group the pool by `key`, keep
only the highest-`valid_at` (ties: highest-`created_at`) member of each group.

Default OFF (`recall(..., agg_reader=True)` or `TENET_AGG_READER=1`) until it's
measured to help — see docs/COMPARISON.md for the payoff (or lack of) on FC-MH and
LoCoMo. Ranking invariant preserved: this is a POOL FILTER (which members are
returned), never a re-ranker — the surviving members keep their original relative
order and relevance/decay scores untouched, exactly like the existing stale-echo/
consistency filters (`memory.py` `_STALE_ECHO`, `consistency.py`) already do for raw
slices.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import Memory


def aggregate_by_key(memories: "list[Memory]") -> "list[Memory]":
    """Collapse same-`key` groups to their highest-`valid_at` member. Unkeyed (raw
    or unkeyed-fact) entries are never grouped — passed through untouched. Preserves
    the original relative order of whatever survives; O(n), two passes, no
    re-scoring."""
    best_by_key = {}
    for m in memories:
        if not m.key:
            continue
        cur = best_by_key.get(m.key)
        if cur is None or (m.valid_at, m.created_at) > (cur.valid_at, cur.created_at):
            best_by_key[m.key] = m
    return [m for m in memories if not m.key or best_by_key.get(m.key) is m]
