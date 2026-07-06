"""MemoryCore — self-managing, bi-temporal memory for LLM agents (Tenet, Track 1).

The three Track-1 asks, in one class:
  • efficient storage & retrieval  -> Qwen embeddings + sqlite + cosine (no LLM in read path)
  • timely forgetting              -> decay score(recency, uses, salience) + sweep
  • recall under limited context   -> budgeted top-k retrieval

Bi-temporal model (the biggest accuracy lever, per SOTA — see docs/SOTA.md):
  • valid_at / invalid_at   — when the fact is true *in the world* (event time)
  • created_at / expired_at — when the system *knew* it (transaction time)
Supersession INVALIDATES the old fact (sets invalid_at + expired_at) instead of
overwriting it, so history is preserved: "what did I believe in March" stays answerable
via recall(as_of=...), while default recall returns only currently-true facts.

Zero heavy deps: sqlite (stdlib) + numpy. Brute-force cosine — fine at hackathon scale
(<1e5 memories); swap in sqlite-vec later without touching the API.
"""
from __future__ import annotations

import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "tenet.db"

# Forgetting knobs
_HALFLIFE_S = 14 * 24 * 3600  # a memory's recency weight halves every 14 days
_FORGET_THRESHOLD = 0.15      # decay score below this -> archived by the sweep
_STALE_ECHO = 0.80            # a raw slice this similar to a superseded fact is stale


@dataclass
class Memory:
    id: int
    text: str
    score: float          # relevance × decay at query time
    created_at: float     # transaction time: when learned
    valid_at: float       # event time: when true in the world
    invalid_at: float | None
    expired_at: float | None
    last_access: float
    uses: int
    pinned: bool
    salience: float
    kind: str = "fact"
    source: str | None = None

    @property
    def is_current(self) -> bool:
        return self.expired_at is None


class MemoryCore:
    def __init__(self, db_path: Path | str = _DEFAULT_DB, *, now=time.time):
        self._now = now  # injectable clock so tests can simulate time passing
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock: FastAPI serves sync endpoints from a
        # threadpool, so the connection is touched from multiple threads.
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 text TEXT NOT NULL,
                 kind TEXT NOT NULL DEFAULT 'fact',  -- 'fact' (distilled, keyed) | 'raw' (verbatim slice)
                 source TEXT,                   -- provenance (e.g. session id) for recall eval + demo
                 skey TEXT,                     -- semantic key "subject::attribute" for supersession
                 embedding BLOB NOT NULL,
                 salience REAL NOT NULL DEFAULT 0.5,
                 valid_at REAL NOT NULL,        -- event time (true from)
                 invalid_at REAL,               -- event time (true until); NULL = still true
                 created_at REAL NOT NULL,      -- transaction time (learned at)
                 expired_at REAL,               -- transaction time (retired at); NULL = current
                 last_access REAL NOT NULL,
                 uses INTEGER NOT NULL DEFAULT 0,
                 pinned INTEGER NOT NULL DEFAULT 0,
                 archived INTEGER NOT NULL DEFAULT 0
               )"""
        )
        self.db.commit()
        self._client = None

    # ---- embedding -------------------------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    _EMBED_MAX_CHARS = 6000  # keep well under the model's token cap per input

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Unit-normalised embeddings via the active provider (Qwen Cloud by default,
        or a local model when EMBED_PROVIDER=local). Batching/truncation/fallback are
        handled in config.embed_texts."""
        return config.embed_texts(texts)

    # ---- store -----------------------------------------------------------
    def store(
        self,
        text: str,
        *,
        key: str | None = None,
        kind: str = "fact",
        source: str | None = None,
        pinned: bool = False,
        salience: float = 0.5,
        valid_at: float | None = None,
        supersede: float = 0.90,
        dedup: float = 0.985,
        surprise_gate: float | None = None,
        _vec: np.ndarray | None = None,
    ) -> int:
        """Add a memory.

        Supersession — how the old fact is retired (bi-temporal, history preserved):
        - If `key` (a "subject::attribute" semantic key from the distiller) is given,
          any current memory with the SAME key and DIFFERENT text is superseded. This
          is the reliable path: it catches value changes (14:20→09:45) and rephrasings
          that embedding similarity alone cannot separate from restatements.
        - Without a key, fall back to embedding proximity: >= dedup ⇒ restatement
          (refresh); supersede..dedup with differing text ⇒ supersede. Best-effort only.
        """
        text = text.strip()
        if not text:
            raise ValueError("empty memory")
        vec = _vec if _vec is not None else self._embed(text)  # network call — outside the lock
        t = self._now()
        va = valid_at if valid_at is not None else t
        with self._lock:
            if kind == "raw":
                # World-model efficiency (predictive-coding principle): only store a raw
                # observation the memory does NOT already predict. If it's near-identical
                # to an existing raw slice (cosine >= surprise_gate), it carries no new
                # information — skip it. Shrinks the store without losing novel detail.
                if surprise_gate is not None:
                    for r in self.db.execute(
                        "SELECT embedding FROM memories WHERE kind='raw' AND archived=0 "
                        "AND expired_at IS NULL"
                    ).fetchall():
                        if float(np.dot(vec, np.frombuffer(r["embedding"], dtype=np.float32))) >= surprise_gate:
                            return -1  # redundant observation, not stored
            elif key is not None:
                prior = self.db.execute(
                    "SELECT id, text, pinned, salience FROM memories "
                    "WHERE skey=? AND archived=0 AND expired_at IS NULL",
                    (key,),
                ).fetchall()
                for row in prior:
                    if row["text"] == text:
                        self._touch(row["id"])
                        return row["id"]          # exact restatement of a keyed fact
                for row in prior:                 # same key, new value ⇒ supersede all priors
                    # A pinned/high-salience fact-slot keeps those properties across
                    # value updates (pinning "residence" survives a move).
                    pinned = pinned or bool(row["pinned"])
                    salience = max(salience, row["salience"])
                    self.db.execute(
                        "UPDATE memories SET invalid_at=?, expired_at=? WHERE id=?",
                        (va, t, row["id"]),
                    )
            else:
                hit = self._nearest_current(vec)  # (id, sim, text)
                if hit and hit[1] >= dedup:
                    self._touch(hit[0])
                    return hit[0]
                if hit and hit[1] >= supersede and hit[2] != text:
                    self.db.execute(
                        "UPDATE memories SET invalid_at=?, expired_at=? WHERE id=?",
                        (va, t, hit[0]),
                    )
            cur = self.db.execute(
                "INSERT INTO memories(text, kind, source, skey, embedding, salience, valid_at, "
                "created_at, last_access, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (text, kind, source, key, vec.tobytes(), float(salience), va, t, t, int(pinned)),
            )
            self.db.commit()
            return cur.lastrowid

    # ---- recall ----------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        char_budget: int | None = None,
        as_of: float | None = None,
    ) -> list[Memory]:
        """Most relevant memories, ranked by relevance × decay.

        Default: only *currently-true* facts (expired_at IS NULL). Pass as_of=<ts>
        to time-travel — recall what was believed true at that timestamp.
        char_budget greedily caps total size (recall under a limited context window).
        """
        qv = self._embed(query)  # network call — kept outside the lock
        with self._lock:
            rows = self._rows_as_of(as_of)
            if not rows:
                return []
            scored = []
            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                relevance = float(np.dot(qv, emb))        # cosine (both unit)
                rank = relevance * self._decay(row)       # forgetting-aware rank
                scored.append((rank, relevance, row))
            scored.sort(key=lambda x: x[0], reverse=True)

            # Dual-pool selection: distilled facts win on consistency/temporal, raw
            # slices win on verbatim detail (durations, numbers). Guarantee raw slices
            # a share of the budget so distillation can't crowd out the exact answer —
            # but if one pool is empty, the other fills all k slots (no starvation).
            facts = [(rank, row) for rank, _rel, row in scored if row["kind"] != "raw"]
            raws = [(rel, row) for _rank, rel, row in scored if row["kind"] == "raw"]

            # World-model consistency: the current facts are the belief state. A raw
            # slice that echoes a SUPERSEDED belief (e.g. "I moved to Boston" after the
            # user moved on) is stale evidence — retire it from current recall so it
            # can't reintroduce an outdated value. (Only for current recall, not as_of.)
            if as_of is None and raws:
                expired = self._expired_fact_matrix()
                if expired is not None:
                    kept = []
                    for rel, row in raws:
                        emb = np.frombuffer(row["embedding"], dtype=np.float32)
                        if float(np.max(expired @ emb)) < _STALE_ECHO:
                            kept.append((rel, row))
                    raws = kept
            n_raw = min(len(raws), k // 2)
            picked = facts[: k - n_raw] + raws[:n_raw]
            if len(picked) < k:  # backfill from whatever remains, best rank first
                leftover = facts[k - n_raw:] + raws[n_raw:]
                leftover.sort(key=lambda x: x[0], reverse=True)
                picked += leftover[: k - len(picked)]

            out: list[Memory] = []
            used = 0
            for rank, row in sorted(picked, key=lambda x: x[0], reverse=True):
                if char_budget is not None and used + len(row["text"]) > char_budget and out:
                    continue
                if as_of is None:
                    self._touch(row["id"])
                used += len(row["text"])
                out.append(self._to_memory(row, rank))
            return out

    # ---- forgetting ------------------------------------------------------
    def _decay(self, row) -> float:
        """Score in (0,1]. Pinned never decays. Recency half-life, boosted by how
        often the memory is used and by its salience."""
        if row["pinned"]:
            return 1.0
        age = self._now() - row["last_access"]
        recency = math.pow(0.5, age / _HALFLIFE_S)
        use_boost = 1.0 + math.log1p(row["uses"]) * 0.15
        sal_boost = 0.6 + 0.8 * row["salience"]           # 0.6..1.4
        return min(1.0, recency * use_boost * sal_boost)

    def forget_sweep(self) -> int:
        """Archive current memories whose decay score fell below threshold
        (pinned never forgotten). Superseded/expired facts are left in place as
        history but excluded from current recall. Returns count archived."""
        n = 0
        with self._lock:
            for row in self._current_rows():
                if not row["pinned"] and self._decay(row) < _FORGET_THRESHOLD:
                    self.db.execute("UPDATE memories SET archived=1 WHERE id=?", (row["id"],))
                    n += 1
            self.db.commit()
        return n

    # ---- helpers ---------------------------------------------------------
    def _current_rows(self):
        """Live, currently-true memories (not archived, not superseded)."""
        return self.db.execute(
            "SELECT * FROM memories WHERE archived=0 AND expired_at IS NULL"
        ).fetchall()

    def _rows_as_of(self, as_of: float | None):
        if as_of is None:
            return self._current_rows()
        # time-travel: facts the system knew (created_at<=t) and hadn't retired
        # (expired_at IS NULL OR expired_at>t), that were true in the world then.
        return self.db.execute(
            "SELECT * FROM memories WHERE archived=0 AND created_at<=? "
            "AND (expired_at IS NULL OR expired_at>?) "
            "AND valid_at<=? AND (invalid_at IS NULL OR invalid_at>?)",
            (as_of, as_of, as_of, as_of),
        ).fetchall()

    def _expired_fact_matrix(self):
        """Embeddings of superseded (expired) facts — the retired belief values.
        Returns an (M×d) matrix or None."""
        rows = self.db.execute(
            "SELECT embedding FROM memories WHERE kind='fact' AND expired_at IS NOT NULL "
            "AND archived=0"
        ).fetchall()
        if not rows:
            return None
        return np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

    def _nearest_current(self, vec: np.ndarray):
        best = None
        for row in self._current_rows():
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            sim = float(np.dot(vec, emb))
            if best is None or sim > best[1]:
                best = (row["id"], sim, row["text"])
        return best

    def _touch(self, mem_id: int):
        self.db.execute(
            "UPDATE memories SET uses=uses+1, last_access=? WHERE id=?",
            (self._now(), mem_id),
        )
        self.db.commit()

    def _to_memory(self, row, score: float) -> Memory:
        return Memory(
            id=row["id"], text=row["text"], score=round(score, 4),
            created_at=row["created_at"], valid_at=row["valid_at"],
            invalid_at=row["invalid_at"], expired_at=row["expired_at"],
            last_access=row["last_access"], uses=row["uses"],
            pinned=bool(row["pinned"]), salience=row["salience"],
            kind=row["kind"], source=row["source"],
        )

    def stats(self) -> dict:
        c = self.db.execute(
            "SELECT "
            "SUM(CASE WHEN archived=0 AND expired_at IS NULL THEN 1 ELSE 0 END) current, "
            "SUM(CASE WHEN expired_at IS NOT NULL THEN 1 ELSE 0 END) superseded, "
            "SUM(CASE WHEN archived=1 THEN 1 ELSE 0 END) archived "
            "FROM memories"
        ).fetchone()
        return {
            "current": c["current"] or 0,
            "superseded": c["superseded"] or 0,
            "archived": c["archived"] or 0,
        }

    def close(self):
        self.db.close()
