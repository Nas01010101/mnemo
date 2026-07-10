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
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .consistency import stale_raw_ids
from .dynamics import Dynamics
from .navigate import navigate as _navigate

# TENET_DB_PATH lets callers (tests, mcp_server's module-level default Tenet())
# redirect the default db off data/ when that symlink target isn't reachable
# (e.g. an external volume TCC-blocks this process — see BENCHMARK.md §9).
_DEFAULT_DB = Path(os.environ["TENET_DB_PATH"]) if os.environ.get("TENET_DB_PATH") else (
    Path(__file__).resolve().parent.parent.parent / "data" / "tenet.db")

# Forgetting knobs
_HALFLIFE_S = 14 * 24 * 3600  # a memory's recency weight halves every 14 days
_FORGET_THRESHOLD = 0.15      # decay score below this -> archived by the sweep
_STALE_ECHO = 0.80            # a raw slice this similar to a superseded fact is stale
_REPLAY_LAMBDA = 0.5          # recursive recall: weight of the evidence pool in the cue
# Read-time belief-evidence consistency (consistency.py, ChurnBench §9 fix,
# component 1) — key-scoped, catches paraphrased echoes _STALE_ECHO misses.
# 0.70: swept on REAL ChurnBench U=2 data (0.60/0.70/0.80 — 100% stale-echo
# recall / 7% false-positive rate at 0.70, dominating both alternatives) and
# defaulted ON because ALL regression gates passed (BENCHMARK.md §9.1: 7
# deterministic suites, bench_horizon U=8 stayed 100%, FC sh_6k proved
# structurally invariant — that arm never stores kind='raw' rows). Override
# with TENET_CONSISTENCY_DEFAULT="off" (or any non-numeric value) to force it
# off if a future regression appears; pass consistency_threshold=None
# per-call to opt out for a single call.
_CONSISTENCY_ENV = os.environ.get("TENET_CONSISTENCY_DEFAULT")
if _CONSISTENCY_ENV is None:
    _CONSISTENCY_THRESHOLD_DEFAULT: float | None = 0.70
elif _CONSISTENCY_ENV.replace(".", "", 1).isdigit():
    _CONSISTENCY_THRESHOLD_DEFAULT = float(_CONSISTENCY_ENV)
else:
    _CONSISTENCY_THRESHOLD_DEFAULT = None


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
    key: str | None = None  # semantic "subject::attribute" key (fact rows only)
    confidence: float | None = None  # learned P(still valid) at query time (dynamics)

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
        # keyed-supersession lookups are per-insert; without this index they are
        # full scans, which turns bulk ingestion (benchmarks: ~10k facts/sequence)
        # into O(n^2). Harmless at conversation scale.
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_skey ON memories(skey) ")
        self.db.commit()
        self._client = None
        self._dyn: Dynamics | None = None   # learned fact dynamics (lazily fitted)
        self._dyn_dirty = True

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
            self._dyn_dirty = True  # ledger changed -> refit dynamics lazily
            return cur.lastrowid

    # ---- recall ----------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        char_budget: int | None = None,
        as_of: float | None = None,
        expand: int = 0,
        hops: int = 0,
        consistency_threshold: float | None = _CONSISTENCY_THRESHOLD_DEFAULT,
    ) -> list[Memory]:
        """Most relevant memories, ranked by relevance × decay.

        Default: only *currently-true* facts (expired_at IS NULL). Pass as_of=<ts>
        to time-travel — recall what was believed true at that timestamp.
        char_budget greedily caps total size (recall under a limited context window).

        `expand` — belief-anchored evidence expansion. The dual-pool top-k names the
        belief state and the sessions it came from; multi-hop/temporal answers, though,
        need the surrounding verbatim turns from *those same sessions*, which the k//2
        raw cap crowds out. When expand>0, after the top-k is chosen we pull up to
        `expand` extra query-relevant raw slices whose `source` matches a surfaced
        memory — closing the multi-session gap while staying well under a flat RAG's
        token count (evidence is anchored to already-relevant sessions, not the haystack).

        `hops` — recursive associative recall (ReContext-style replay, arXiv:2607.02509,
        with cosine standing in for attention as the cue–trace association). When hops>0,
        the `expand` slots are selected over `hops` rounds instead of one: after each
        round the cue is re-conditioned on the evidence gathered so far
        (cue ← normalize(q + λ·mean(pool))), and the WHOLE store is re-scored — so a
        later round can reach a session the raw query never surfaced (the associative
        hop a multi-session question needs). Selection stays read-only, LLM-free, and
        subject to the same stale-echo filter; callers keep the token budget cap.

        `consistency_threshold` — read-time belief-evidence consistency (component 1
        of the ChurnBench §9 fix, consistency.py). Defaults to
        `_CONSISTENCY_THRESHOLD_DEFAULT` (0.70, on — see that constant's comment for
        the regression-gate history); pass None to leave the raw pool as-is beyond
        the global `_STALE_ECHO` filter below. A float drops any raw slice whose
        embedding is close (cosine >= threshold) to a SUPERSEDED fact of a key whose
        CURRENT fact is already ranked in the top-k facts for this query — i.e. we
        already know the current value for that key, so a close paraphrase of an old
        one is confirmed-stale, not just embedding-adjacent. Only affects which raw
        slices are eligible; current-fact ranking is byte-identical with this on or off.
        """
        qv = self._embed(query)  # network call — kept outside the lock
        with self._lock:
            rows = self._rows_as_of(as_of)
            if not rows:
                return []
            # One BLAS matmul instead of a per-row python loop (docs/HARNESS.md §3:
            # 28-33× on the scoring step; deserialization done once, contiguously).
            mat = np.frombuffer(
                b"".join(row["embedding"] for row in rows), dtype=np.float32
            ).reshape(len(rows), -1)
            rels = mat @ qv                               # cosine (both unit)
            # World-model layer: annotate each keyed fact with the LEARNED probability
            # it is still the current truth (dynamics.py — per-key-class survival
            # fitted on this store's own supersession history). Deliberately NOT a
            # rank discount: a doubted fact is still the best known answer, and
            # demoting it re-creates the churn failure supersession prevents
            # (measured). Doubt is surfaced (Memory.confidence, uncertain_facts())
            # for the agent to caveat or re-verify with the user.
            dyn = self._dynamics() if as_of is None else None
            now = self._now()
            conf: dict[int, float] = {}
            if dyn is not None:
                for row in rows:
                    p = self._p_valid(row, dyn, now)
                    if p is not None:
                        conf[row["id"]] = p

            scored = [
                (float(rel) * self._decay(row), float(rel), row)
                for rel, row in zip(rels, rows)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)

            # World-model consistency: the current facts are the belief state. A raw
            # slice that echoes a SUPERSEDED belief (e.g. "I moved to Boston" after the
            # user moved on) is stale evidence — retire it from current recall so it
            # can't reintroduce an outdated value. (Only for current recall, not as_of.)
            expired = self._expired_fact_matrix() if as_of is None else None

            # Dual-pool selection: distilled facts win on consistency/temporal, raw
            # slices win on verbatim detail (durations, numbers). Guarantee raw slices
            # a share of the budget so distillation can't crowd out the exact answer —
            # but if one pool is empty, the other fills all k slots (no starvation).
            facts = [(rank, row) for rank, _rel, row in scored if row["kind"] != "raw"]

            # Read-time belief-evidence consistency (component 1, consistency.py):
            # key-scoped, computed only when requested (default off — see recall()'s
            # docstring). Does not touch `facts`/`scored`, so current-fact ranking is
            # unaffected either way; only raw-slice eligibility (_fresh) changes.
            stale_ids: set[int] = set()
            if consistency_threshold is not None and as_of is None:
                pool_keys = {row["skey"] for _rank, row in facts[:k] if row["skey"]}
                if pool_keys:
                    stale_ids = stale_raw_ids(
                        [row for row in rows if row["kind"] == "raw"],
                        self._expired_keyed_rows(), pool_keys, consistency_threshold,
                    )

            def _fresh(row) -> bool:
                if row["id"] in stale_ids:
                    return False
                if expired is None:
                    return True
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                return float(np.max(expired @ emb)) < _STALE_ECHO

            raws = [(rel, row) for _rank, rel, row in scored
                    if row["kind"] == "raw" and _fresh(row)]
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
                out.append(self._to_memory(row, rank, confidence=conf.get(row["id"])))

            # Belief-anchored evidence expansion: add query-relevant raw turns from the
            # sessions the top-k already surfaced. Ranked by pure relevance (detail, not
            # decay), fresh (not a stale echo), not already picked.
            if expand and as_of is None and hops <= 1:
                anchors = {m.source for m in out if m.source}
                if anchors:
                    have = {m.id for m in out}
                    cands = [(rel, row) for _r, rel, row in scored
                             if row["kind"] == "raw" and row["source"] in anchors
                             and row["id"] not in have and _fresh(row)]
                    cands.sort(key=lambda x: x[0], reverse=True)
                    for rel, row in cands[:expand]:
                        if char_budget is not None and used + len(row["text"]) > char_budget:
                            continue
                        self._touch(row["id"])
                        used += len(row["text"])
                        out.append(self._to_memory(row, rel))

            # Recursive associative recall: spend the expand slots over `hops` rounds,
            # re-conditioning the cue on the evidence pool after each (replay). Unlike
            # anchored expansion, candidates span the WHOLE store, so a hop can pull in
            # a session related to the evidence rather than to the raw query.
            elif expand and as_of is None and hops > 1:
                have = {m.id for m in out}
                pool_vecs = [np.frombuffer(row["embedding"], dtype=np.float32)
                             for _rank, row in picked if row["id"] in have]
                fresh_rows = [row for _r, _rel, row in scored
                              if row["id"] not in have and _fresh(row)]
                per_hop = max(1, expand // hops)
                taken = 0
                for _ in range(hops):
                    if taken >= expand or not fresh_rows:
                        break
                    if pool_vecs:  # replay: fold the pool into the cue
                        pool = np.mean(pool_vecs, axis=0)
                        cue = qv + _REPLAY_LAMBDA * pool
                        cue /= (np.linalg.norm(cue) or 1.0)
                    else:
                        cue = qv
                    rescored = sorted(
                        ((float(np.dot(cue, np.frombuffer(row["embedding"], dtype=np.float32))), row)
                         for row in fresh_rows),
                        key=lambda x: x[0], reverse=True)
                    for rel, row in rescored[:per_hop]:
                        if taken >= expand:
                            break
                        if char_budget is not None and used + len(row["text"]) > char_budget:
                            continue
                        self._touch(row["id"])
                        used += len(row["text"])
                        out.append(self._to_memory(row, rel))
                        have.add(row["id"])
                        pool_vecs.append(np.frombuffer(row["embedding"], dtype=np.float32))
                        taken += 1
                    fresh_rows = [row for row in fresh_rows if row["id"] not in have]
            return out

    # ---- adaptive navigation (LLM-free multi-hop) -------------------------
    def navigate(
        self,
        query: str,
        *,
        k: int = 10,
        max_hops: int = 4,
        tau_gain: float = 0.15,
        char_budget: int | None = None,
    ) -> tuple[list[Memory], list[dict]]:
        """Adaptive-depth associative recall: same read path as `recall`, but
        descends further hops only while new evidence clears a relevance-gain
        gate, stopping the moment returns diminish. LLM-free. Does not change
        default `recall()` behaviour — see tenet.navigate.navigate for the full
        escalation-schedule / adaptive-stop docstring and trace format."""
        return _navigate(self, query, k=k, max_hops=max_hops, tau_gain=tau_gain,
                          char_budget=char_budget)

    # ---- fact dynamics (the world-model layer) -----------------------------
    def _dynamics(self):
        """Learned lifetime model, refit lazily from the ledger when it changed.

        Default is the closed-form Gamma-exponential Dynamics (dynamics.py). Set env
        TENET_DYNAMICS=neural (+ TENET_NEURAL_NPZ=<path>) to swap in the trained GRU
        world model (dynamics_neural.py, numpy-only). The neural path needs per-event
        value embeddings, so it binds the ledger's stored embeddings — use it with the
        bge-small local embedder it was trained on (EMBED_PROVIDER=local, 384d)."""
        if self._dyn is not None and not self._dyn_dirty:
            return self._dyn
        rows = self.db.execute(
            "SELECT skey, valid_at, invalid_at, embedding FROM memories "
            "WHERE kind='fact' AND skey IS NOT NULL AND archived=0"
        ).fetchall()
        import os
        nd = None
        if os.environ.get("TENET_DYNAMICS", "").lower() == "neural":
            from .dynamics_neural import build_from_ledger  # numpy-only world model
            nd = build_from_ledger(rows, now=self._now())   # None on any failure
        self._dyn = nd if nd is not None else Dynamics.fit(rows, now=self._now())
        self._dyn_dirty = False
        return self._dyn

    def _p_valid(self, row, dyn: Dynamics, now: float) -> float | None:
        """Learned P(this fact is still the current truth); None for raw rows."""
        if row["kind"] != "fact" or row["skey"] is None:
            return None
        return dyn.p_valid(row["skey"], now - row["valid_at"], now=now)

    def uncertain_facts(self, threshold: float = 0.5) -> list[dict]:
        """Current keyed facts the dynamics model doubts (P(valid) < threshold) —
        the agent's 'worth re-verifying with the user' list. Sorted most-doubted
        first. Includes ripple effects (a correlated key changed recently)."""
        with self._lock:
            dyn = self._dynamics()
            now = self._now()
            out = []
            for r in self.db.execute(
                "SELECT * FROM memories WHERE kind='fact' AND skey IS NOT NULL "
                "AND archived=0 AND expired_at IS NULL"
            ).fetchall():
                p = dyn.p_valid(r["skey"], now - r["valid_at"], now=now)
                if p < threshold:
                    out.append({
                        "key": r["skey"], "text": r["text"], "p_valid": round(p, 3),
                        "age_days": round((now - r["valid_at"]) / 86400.0, 1),
                        "expected_lifetime_days": dyn.expected_lifetime_days(r["skey"]),
                    })
            out.sort(key=lambda d: d["p_valid"])
            return out

    # ---- belief state (demo UI) -------------------------------------------
    def list_beliefs(self, as_of: float | None = None) -> list[dict]:
        """Distilled facts as plain dicts, for a UI belief-state view (no LLM).

        as_of=None: EVERY unarchived fact (current + superseded) — the full
        per-key history a UI needs to render "current, with struck-through
        history below". as_of=<ts>: only facts true-in-the-world and
        known-to-the-system at that instant (time-travel snapshot, mirrors
        `recall`'s bi-temporal filter) — one entry per key, status "current"
        relative to that moment.
        """
        with self._lock:
            if as_of is None:
                rows = self.db.execute(
                    "SELECT * FROM memories WHERE kind='fact' AND archived=0 "
                    "ORDER BY skey, valid_at"
                ).fetchall()
                # UI-only doubt marker: same learned p_valid recall() attaches,
                # never filters/sorts here either — see _p_valid.
                dyn, now = self._dynamics(), self._now()
                out = []
                for r in rows:
                    cur = r["expired_at"] is None
                    p = self._p_valid(r, dyn, now) if cur else None
                    out.append({"id": r["id"], "key": r["skey"] or "(unkeyed)", "text": r["text"],
                                "valid_at": r["valid_at"], "expired_at": r["expired_at"],
                                "status": "current" if cur else "superseded",
                                "p_valid": None if p is None else round(p, 3)})
                return out
            rows = [r for r in self._rows_as_of(as_of) if r["kind"] == "fact"]
            rows.sort(key=lambda r: (r["skey"] or "(unkeyed)", r["valid_at"]))
            return [
                {"id": r["id"], "key": r["skey"] or "(unkeyed)", "text": r["text"],
                 "valid_at": r["valid_at"], "expired_at": r["expired_at"],
                 "status": "current"}
                for r in rows
            ]

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

    def _expired_keyed_rows(self):
        """Superseded KEYED facts (id, skey, embedding) — consistency.py's key-scoped
        input; a superset of `_expired_fact_matrix`'s rows filtered to skey IS NOT NULL,
        fetched separately since that helper only returns bare embeddings."""
        return self.db.execute(
            "SELECT id, skey, embedding FROM memories WHERE kind='fact' "
            "AND expired_at IS NOT NULL AND skey IS NOT NULL AND archived=0"
        ).fetchall()

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

    def _to_memory(self, row, score: float, confidence: float | None = None) -> Memory:
        return Memory(
            id=row["id"], text=row["text"], score=round(score, 4),
            created_at=row["created_at"], valid_at=row["valid_at"],
            invalid_at=row["invalid_at"], expired_at=row["expired_at"],
            last_access=row["last_access"], uses=row["uses"],
            pinned=bool(row["pinned"]), salience=row["salience"],
            kind=row["kind"], source=row["source"], key=row["skey"],
            confidence=None if confidence is None else round(confidence, 3),
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
