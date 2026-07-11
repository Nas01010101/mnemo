"""Tenet — the memory agent: distillation + bi-temporal store, one interface.

    m = Tenet()
    m.ingest("I moved to Toronto last week.")   # distills -> atomic keyed facts -> store
    m.recall("where does the user live?")        # bi-temporal, forgetting-aware retrieval

`ingest` is the write path (LLM distillation, one call per message). `recall` is the
read path (pure vector + decay, no LLM — frontier-correct for latency). The store
handles supersession/forgetting; the distiller supplies the keys that make it reliable.
"""
from __future__ import annotations

import os
import time

from .distill import distill
from .memory import Memory, MemoryCore

# Retraction routing (docs/COMPARISON.md follow-up #3) — the DISTILLER always tags
# facts with action="retract" when it detects explicit forget-intent (harmless if
# ignored: an extra JSON field), but ingest() only ACTS on it (routes to
# core.retract() instead of core.store()) when this is on. Default OFF until
# measured to help; TENET_RETRACT=1 (or ingest(..., retract=True) per call) opts in.
_RETRACT_DEFAULT = os.environ.get("TENET_RETRACT", "").strip().lower() in ("1", "true", "on", "yes")


class Tenet:
    def __init__(self, db_path=None, *, now=time.time, distill_model: str | None = None):
        self.core = MemoryCore(db_path, now=now) if db_path else MemoryCore(now=now)
        self._now = now
        self._distill_model = distill_model

    def ingest(self, message: str, *, pinned: bool = False, retract: bool | None = None) -> list[int]:
        """Distill a raw message into atomic facts and store each (with supersession).
        Returns the stored memory ids. Empty if nothing durable was found.

        `retract` — if the distiller tags a fact action="retract" (explicit
        forget-intent, e.g. "forget my old address") AND this is enabled (default
        `_RETRACT_DEFAULT`, env `TENET_RETRACT=1`, or pass True/False per call), that
        fact is routed to `core.retract(key)` (a deletion) instead of `core.store()`
        — and does NOT get an id in the returned list (nothing was stored). When
        disabled (the default), retract-tagged facts are stored normally like any
        other fact — the flag gates the BEHAVIOR change, not the detection, so
        flipping it never needs a different distiller call."""
        kw = {"model": self._distill_model} if self._distill_model else {}
        facts = distill(message, **kw)
        use_retract = _RETRACT_DEFAULT if retract is None else retract
        ids = []
        for f in facts:
            if use_retract and f.action == "retract":
                self.core.retract(f.key)
                continue
            ids.append(self.core.store(
                f.statement, key=f.key, salience=f.salience, pinned=pinned,
            ))
        return ids

    def ingest_session(self, turns, *, source: str | None = None,
                       valid_at: float | None = None, pinned: bool = False,
                       surprise_gate: float | None = 0.97,
                       retract: bool | None = None) -> dict:
        """Hybrid ingest of a conversation session (list of {role,content} or (role,content)).

        Stores BOTH:
          • distilled keyed facts  — for supersession + temporal consistency
          • raw verbatim turns      — so quantitative/specific detail (durations, numbers,
                                       names) survives; distillation alone flattens these.
        This mirrors SOTA (LongMemEval-V2: the raw slice pool matters for static questions).
        `source` (e.g. a session id) is stored as provenance for recall eval + demo.

        `retract` — same routing as `ingest()` (docs/COMPARISON.md follow-up #3):
        a fact the distiller tags action="retract" is routed to `core.retract(key)`
        instead of `core.store()` when enabled (default `_RETRACT_DEFAULT`). Only
        applies to the distilled-facts half — raw verbatim turns have no key/action
        and are always stored as before.
        """
        norm = [(t["role"], t["content"]) if isinstance(t, dict) else t for t in turns]
        convo = "\n".join(f"{r}: {c}" for r, c in norm)
        kw = {"model": self._distill_model} if self._distill_model else {}
        use_retract = _RETRACT_DEFAULT if retract is None else retract
        fact_ids = []
        for f in distill(convo, **kw):
            if use_retract and f.action == "retract":
                self.core.retract(f.key)
                continue
            fact_ids.append(self.core.store(f.statement, key=f.key, salience=f.salience,
                                             source=source, pinned=pinned))
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

    def retract(self, key: str, **kw) -> int:
        """Explicit "forget X" — see MemoryCore.retract for the full docstring
        (a deletion, distinct from supersession's retire-then-replace)."""
        return self.core.retract(key, **kw)

    def recall(self, query: str, **kw) -> list[Memory]:
        return self.core.recall(query, **kw)

    def navigate(self, query: str, **kw):
        """Adaptive-depth, LLM-free associative recall — deepens hops only while
        new evidence clears a relevance-gain gate. See MemoryCore.navigate /
        tenet.navigate.navigate for the full docstring."""
        return self.core.navigate(query, **kw)

    def forget_sweep(self) -> int:
        return self.core.forget_sweep()

    def uncertain_facts(self, threshold: float = 0.5) -> list[dict]:
        """Current keyed facts the learned world model doubts — see MemoryCore
        for the full docstring."""
        return self.core.uncertain_facts(threshold=threshold)

    def stats(self) -> dict:
        return self.core.stats()

    def close(self):
        self.core.close()
