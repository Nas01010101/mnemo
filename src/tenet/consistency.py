"""Read-time belief-evidence consistency (ChurnBench §9 fix, component 1).

The paper already claims "belief-evidence consistency retires raw evidence that
echoes a superseded belief" (memory.py's `_STALE_ECHO` global filter). ChurnBench
(§9) found that filter tuned for near-verbatim echoes only: a raw turn phrased as
"I got promoted, I'm now a principal analyst." doesn't clear a 0.80 cosine bar
against its distilled paraphrase "The user's job title is principal analyst." —
different enough wording that a *global* threshold can't be raised to catch it
without also risking cross-key false positives.

This module adds a narrower, KEY-SCOPED check: a raw slice is stale if it is
close to a superseded fact whose KEY already has a CURRENT fact in the candidate
pool. Scoping to "we already know the current value for this key" means the
threshold can be set lower (more sensitive) without touching unrelated keys —
the false-positive surface is just that one key's own value chain, not the
whole store.

Deterministic, embeddings-only (reuses vectors already computed at ingest), no
LLM. Split out of memory.py per the fix's own <=40-line budget for the change;
see BENCHMARK.md §9.1 for the measured threshold sweep and the regression-gate
outcome that decided the shipped default.
"""
from __future__ import annotations

import numpy as np


def stale_raw_ids(raw_rows, superseded_rows, pool_keys: set[str], threshold: float) -> set[int]:
    """Ids of `raw_rows` that echo a superseded fact whose key is in `pool_keys`,
    at cosine similarity >= threshold.

    raw_rows: sqlite3.Row iterable with `id`, `embedding` (kind='raw' candidates).
    superseded_rows: sqlite3.Row iterable with `id`, `skey`, `embedding` — every
        currently-superseded KEYED fact in the store (kind='fact', expired_at set).
    pool_keys: skeys of the CURRENT facts already surfaced for this query (the
        "already in the pool" condition — see module docstring).
    """
    scoped = [r for r in superseded_rows if r["skey"] in pool_keys]
    if not raw_rows or not scoped:
        return set()
    sup_mat = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in scoped])
    stale: set[int] = set()
    for row in raw_rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        if float(np.max(sup_mat @ emb)) >= threshold:
            stale.add(row["id"])
    return stale
