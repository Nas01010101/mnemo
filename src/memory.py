"""MemoryCore — self-managing memory for LLM agents (Mnemo, Track 1).

The three Track-1 asks, in one class:
  • efficient storage & retrieval  -> Qwen embeddings + sqlite + cosine
  • timely forgetting              -> decay score(recency, uses, pinned) + sweep
  • recall under limited context   -> budgeted top-k retrieval

Zero heavy deps: sqlite (stdlib) + numpy. Vectors are brute-force cosine — fine at
hackathon scale (<1e5 memories); swap in sqlite-vec later without touching the API.
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

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "mnemo.db"

# Forgetting knobs
_HALFLIFE_S = 14 * 24 * 3600  # a memory's recency weight halves every 14 days
_FORGET_THRESHOLD = 0.15      # decay score below this -> archived by the sweep


@dataclass
class Memory:
    id: int
    text: str
    score: float          # live decay score at query time
    created_at: float
    last_access: float
    uses: int
    pinned: bool


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
                 embedding BLOB NOT NULL,
                 created_at REAL NOT NULL,
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
        if self._client is None:
            self._client = config.qwen_client()
        r = self._client.embeddings.create(model=config.QWEN_EMBED_MODEL, input=text)
        v = np.asarray(r.data[0].embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v  # unit-normalise -> cosine == dot product

    # ---- store -----------------------------------------------------------
    def store(self, text: str, *, pinned: bool = False, dedup: float = 0.93) -> int:
        """Add a memory. If a near-duplicate exists (cosine >= dedup), refresh it
        instead of inserting — cheap write-time consolidation."""
        text = text.strip()
        if not text:
            raise ValueError("empty memory")
        vec = self._embed(text)  # network call — kept outside the lock
        with self._lock:
            if dedup:
                hit = self._nearest(vec)
                if hit and hit[1] >= dedup:
                    # Near-duplicate: the NEWER fact supersedes the old one (e.g. a
                    # changed flight time). Overwrite text + embedding in place so
                    # the stale value retires without a conflicting duplicate.
                    self.db.execute(
                        "UPDATE memories SET text=?, embedding=?, last_access=?, uses=uses+1 WHERE id=?",
                        (text, vec.tobytes(), self._now(), hit[0]),
                    )
                    self.db.commit()
                    return hit[0]
            t = self._now()
            cur = self.db.execute(
                "INSERT INTO memories(text, embedding, created_at, last_access, pinned) "
                "VALUES (?,?,?,?,?)",
                (text, vec.tobytes(), t, t, int(pinned)),
            )
            self.db.commit()
            return cur.lastrowid

    # ---- recall ----------------------------------------------------------
    def recall(self, query: str, *, k: int = 5, char_budget: int | None = None) -> list[Memory]:
        """Return the most relevant *live* memories, ranked by relevance × decay.
        If char_budget is set, greedily fill up to that many characters (recall
        under a limited context window)."""
        qv = self._embed(query)  # network call — kept outside the lock
        with self._lock:
            rows = self._live_rows()
            if not rows:
                return []
            scored = []
            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                relevance = float(np.dot(qv, emb))        # cosine (both unit)
                rank = relevance * self._decay(row)       # forgetting-aware rank
                scored.append((rank, relevance, row))
            scored.sort(key=lambda x: x[0], reverse=True)

            out: list[Memory] = []
            used = 0
            for rank, _rel, row in scored:
                if len(out) >= k:
                    break
                if char_budget is not None and used + len(row["text"]) > char_budget and out:
                    continue  # skip ones that don't fit, keep filling with smaller ones
                self._touch(row["id"])
                used += len(row["text"])
                out.append(self._to_memory(row, rank))
            return out

    # ---- forgetting ------------------------------------------------------
    def _decay(self, row) -> float:
        """Score in (0,1]. Pinned never decays. Otherwise recency half-life plus a
        mild boost for frequently-accessed memories."""
        if row["pinned"]:
            return 1.0
        age = self._now() - row["last_access"]
        recency = math.pow(0.5, age / _HALFLIFE_S)
        use_boost = 1.0 + math.log1p(row["uses"]) * 0.15
        return min(1.0, recency * use_boost)

    def forget_sweep(self) -> int:
        """Archive memories whose decay score fell below threshold. Returns count."""
        n = 0
        with self._lock:
            for row in self._live_rows():
                if not row["pinned"] and self._decay(row) < _FORGET_THRESHOLD:
                    self.db.execute("UPDATE memories SET archived=1 WHERE id=?", (row["id"],))
                    n += 1
            self.db.commit()
        return n

    # ---- helpers ---------------------------------------------------------
    def _live_rows(self):
        return self.db.execute("SELECT * FROM memories WHERE archived=0").fetchall()

    def _nearest(self, vec: np.ndarray):
        best = None
        for row in self._live_rows():
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            sim = float(np.dot(vec, emb))
            if best is None or sim > best[1]:
                best = (row["id"], sim)
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
            created_at=row["created_at"], last_access=row["last_access"],
            uses=row["uses"], pinned=bool(row["pinned"]),
        )

    def stats(self) -> dict:
        live = self.db.execute("SELECT COUNT(*) c FROM memories WHERE archived=0").fetchone()["c"]
        arch = self.db.execute("SELECT COUNT(*) c FROM memories WHERE archived=1").fetchone()["c"]
        return {"live": live, "archived": arch}

    def close(self):
        self.db.close()
