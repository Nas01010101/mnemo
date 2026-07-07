"""Tenet — the memory agent: distillation + bi-temporal store, one interface.

    m = Tenet()
    m.ingest("I moved to Toronto last week.")   # distills -> atomic keyed facts -> store
    m.recall("where does the user live?")        # bi-temporal, forgetting-aware retrieval

`ingest` is the write path (LLM distillation, one call per message). `recall` is the
read path (pure vector + decay, no LLM — frontier-correct for latency). The store
handles supersession/forgetting; the distiller supplies the keys that make it reliable.
"""
from __future__ import annotations

import time

from .distill import distill
from .memory import Memory, MemoryCore


class Tenet:
    def __init__(self, db_path=None, *, now=time.time, distill_model: str | None = None):
        self.core = MemoryCore(db_path, now=now) if db_path else MemoryCore(now=now)
        self._now = now
        self._distill_model = distill_model

    def ingest(self, message: str, *, pinned: bool = False) -> list[int]:
        """Distill a raw message into atomic facts and store each (with supersession).
        Returns the stored memory ids. Empty if nothing durable was found."""
        kw = {"model": self._distill_model} if self._distill_model else {}
        facts = distill(message, **kw)
        ids = []
        for f in facts:
            ids.append(self.core.store(
                f.statement, key=f.key, salience=f.salience, pinned=pinned,
            ))
        return ids

    def ingest_session(self, turns, *, source: str | None = None,
                       valid_at: float | None = None, pinned: bool = False,
                       surprise_gate: float | None = 0.97) -> dict:
        """Hybrid ingest of a conversation session (list of {role,content} or (role,content)).

        Stores BOTH:
          • distilled keyed facts  — for supersession + temporal consistency
          • raw verbatim turns      — so quantitative/specific detail (durations, numbers,
                                       names) survives; distillation alone flattens these.
        This mirrors SOTA (LongMemEval-V2: the raw slice pool matters for static questions).
        `source` (e.g. a session id) is stored as provenance for recall eval + demo.
        """
        norm = [(t["role"], t["content"]) if isinstance(t, dict) else t for t in turns]
        convo = "\n".join(f"{r}: {c}" for r, c in norm)
        kw = {"model": self._distill_model} if self._distill_model else {}
        fact_ids = [
            self.core.store(f.statement, key=f.key, salience=f.salience,
                            source=source, pinned=pinned)
            for f in distill(convo, **kw)
        ]
        raw_ids = []
        for role, content in norm:
            if not content.strip():
                continue
            # raw slices: retrievable, lower salience, never supersede each other;
            # surprise-gated so redundant observations aren't stored.
            rid = self.core.store(
                f"{role}: {content.strip()}", kind="raw", salience=0.35,
                source=source, valid_at=valid_at, surprise_gate=surprise_gate,
            )
            if rid != -1:
                raw_ids.append(rid)
        return {"facts": fact_ids, "raw": raw_ids}

    def store_fact(self, text: str, **kw) -> int:
        """Store a pre-formed fact directly (bypass distillation)."""
        return self.core.store(text, **kw)

    def recall(self, query: str, **kw) -> list[Memory]:
        return self.core.recall(query, **kw)

    def forget_sweep(self) -> int:
        return self.core.forget_sweep()

    def stats(self) -> dict:
        return self.core.stats()

    def close(self):
        self.core.close()
