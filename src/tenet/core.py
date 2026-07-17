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

    def ingest(self, message: str, *, pinned: bool = False, retract: bool | None = None,
               key_prefix: str | None = None) -> list[int]:
        """Distill a raw message into atomic facts and store each (with supersession).
        Returns the stored memory ids. Empty if nothing durable was found.

        `retract` — if the distiller tags a fact action="retract" (explicit
        forget-intent, e.g. "forget my old address") AND this is enabled (default
        `_RETRACT_DEFAULT`, env `TENET_RETRACT=1`, or pass True/False per call), that
        fact is routed to `core.retract(key)` (a deletion) instead of `core.store()`
        — and does NOT get an id in the returned list (nothing was stored). When
        disabled (the default), retract-tagged facts are stored normally like any
        other fact — the flag gates the BEHAVIOR change, not the detection, so
        flipping it never needs a different distiller call.

        `key_prefix` — prepended to every distilled fact's key before storing
        (e.g. "alice::" turns "user::residence" into "alice::user::residence").
        `memory.py`'s `_split_key` treats everything before the LAST "::" as one
        subject, so this genuinely namespaces supersession (both the exact-key
        and embedding-based fuzzy-key paths are subject-scoped) — not just a
        read-time filter. None (default) leaves keys untouched; existing callers
        are unaffected. Used by `add()` for Mem0-style multi-user isolation."""
        kw = {"model": self._distill_model} if self._distill_model else {}
        facts = distill(message, **kw)
        use_retract = _RETRACT_DEFAULT if retract is None else retract
        ids = []
        for f in facts:
            key = f"{key_prefix}{f.key}" if key_prefix else f.key
            if use_retract and f.action == "retract":
                self.core.retract(key)
                continue
            ids.append(self.core.store(
                f.statement, key=key, salience=f.salience, pinned=pinned,
                scenario=f.scenario,
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
                                             source=source, pinned=pinned,
                                             scenario=f.scenario))
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
        """Current keyed facts the learned staleness model doubts — see MemoryCore
        for the full docstring."""
        return self.core.uncertain_facts(threshold=threshold)

    # ---- Mem0-compatible CRUD facade (add/search/get_all/delete/update) --------
    # Thin aliases over the methods above — no engine logic here, just the
    # ingest/recall/retract/list_beliefs calls a Mem0-style caller already knows
    # how to make. Purely additive: none of the methods above changed behavior,
    # and calling code that never touches add/search/get_all/delete/update is
    # completely unaffected. `user_id` isolation is real, not cosmetic — it
    # namespaces the SUBJECT of each fact's "subject::attribute" key (see
    # `ingest()`'s `key_prefix`), so two users' same-named facts (both saying "I
    # live in X") genuinely can't supersede or fuzzy-match each other; it isn't
    # just a read-time filter over a shared, ambiguous key space.
    def add(self, text: str, *, user_id: str | None = None, **kw) -> list[int]:
        """Mem0-compatible alias for `ingest()`. `user_id`, when given, scopes
        this write to that user's namespace (so it won't supersede or surface
        under a different user's `search()`/`get_all()`/`delete()`). A thin
        wrapper — same distillation + bi-temporal supersession as `ingest()`,
        nothing reimplemented."""
        prefix = f"{user_id}::" if user_id else None
        return self.ingest(text, key_prefix=prefix, **kw)

    def search(self, query: str, *, user_id: str | None = None, k: int = 5, **kw) -> list[Memory]:
        """Mem0-compatible alias for `recall()`. `user_id`, when given, scopes
        results to that user's namespace: over-fetches internally then filters
        by key prefix (a thin, read-only post-filter on top of `recall()` — no
        engine changes), so it may return fewer than `k` hits if that user has
        few eligible memories. Only "fact" (keyed) memories can be user-scoped
        this way — `recall()`'s raw-slice pool has no key and is excluded from
        a `user_id`-scoped search (use `search(..., user_id=None)` if you also
        want the raw pool)."""
        if user_id is None:
            return self.recall(query, k=k, **kw)
        prefix = f"{user_id}::"
        fetch_k = max(k * 4, k + 10)  # over-fetch to compensate for post-filtering
        hits = self.recall(query, k=fetch_k, **kw)
        return [m for m in hits if m.key and m.key.startswith(prefix)][:k]

    def get_all(self, *, user_id: str | None = None) -> list[dict]:
        """Mem0-compatible alias for `list_beliefs()` — the CURRENT belief state
        as clean, human-readable dicts: `{id, key, text, p_valid}`. This is a
        genuine differentiator, not just API parity: Mem0/Zep/Letta memories are
        opaque vector/graph entries; Tenet's are a literal
        subject::attribute -> value table, readable at a glance. `user_id`, when
        given, scopes to that user's namespace and the prefix is stripped from
        the returned `key` (the caller already knows which user they asked
        for)."""
        beliefs = [b for b in self.core.list_beliefs() if b["status"] == "current"]
        prefix = f"{user_id}::" if user_id else None
        out = []
        for b in beliefs:
            key = b["key"] or ""
            if prefix:
                if not key.startswith(prefix):
                    continue
                key = key[len(prefix):]
            out.append({"id": b["id"], "key": key, "text": b["text"], "p_valid": b["p_valid"]})
        return out

    def delete(self, key: str, *, user_id: str | None = None) -> int:
        """Mem0-compatible alias for `retract()` — a deletion (no replacement
        value), not a value update; see `MemoryCore.retract`. `user_id`, when
        given, is prepended to `key` to match how `add()` namespaced it."""
        full_key = f"{user_id}::{key}" if user_id else key
        return self.retract(full_key)

    def update(self, text: str, *, user_id: str | None = None, **kw) -> list[int]:
        """Mem0-compatible alias — but note: Tenet has no separate "update"
        operation to call. Bi-temporal supersession means calling `add()` again
        with a changed value automatically retires the old one and stores the
        new one (`memory.py` `store()`'s keyed path) — that IS the update, by
        design, not a special case needing its own method. This alias exists so
        Mem0-style code that calls `.update(new_text)` instead of
        `.add(new_text)` still works; it's a plain passthrough to `add()`."""
        return self.add(text, user_id=user_id, **kw)

    def stats(self) -> dict:
        return self.core.stats()

    def close(self):
        self.core.close()
