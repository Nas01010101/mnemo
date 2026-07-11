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

Scale (docs/SCALE.md): the embedding matrix + bi-temporal metadata are RESIDENT in
memory (`.index.ResidentIndex`), built once and updated incrementally on
store()/forget_sweep() — recall() no longer re-`SELECT`s and re-deserializes the whole
table on every call (measured: 356ms of a 1055ms call at 100k facts, gone). See
`index.py`'s module docstring for the full design (growth strategy, multi-process
staleness detection, what's still O(n) Python and why).
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .aggregate import aggregate_by_key
from .consistency import stale_raw_ids
from .dynamics import Dynamics
from .index import ResidentIndex
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

# Read-time recency aggregation (aggregate.py, docs/COMPARISON.md follow-up #1) —
# an OPTIONAL CAR-style max(serial)-equivalent pool de-conflicting pass, LLM-free
# (Tenet's `valid_at` already IS the serial-equivalent signal). Default OFF until
# measured to help; TENET_AGG_READER=1 (or recall(..., agg_reader=True) per call)
# opts in.
_AGG_READER_DEFAULT = os.environ.get("TENET_AGG_READER", "").strip().lower() in ("1", "true", "on", "yes")

# Raw-turn-favored recall (docs/COMPARISON.md follow-up #2) — an OPTIONAL dual-pool
# re-weighting for benchmarks where EXACT wording matters more than distilled
# consistency (LoCoMo verbatim recall: RAG beats Tenet 38.8 vs 33.8 because
# distillation paraphrases away the wording the answer key checks — docs/BENCHMARK.md
# §12). Default split caps raw slices at k//2 of the initial top-k, facts filling the
# rest; ON, raw slices get PRIORITY up to the full k budget (facts only fill what's
# left) — combines follow-up #2's "(a) raise the raw-slice budget" and "(b) raw-
# priority ordering" into one knob, since they're the same underlying change (a
# higher raw cap IS raw-first ordering once the cap can reach k). Default OFF until
# measured to help; TENET_RAW_RECALL=1 (or recall(..., raw_recall=True) per call)
# opts in. Does not touch ranking (still relevance x decay) or which pool a memory
# comes from — only how many raw-vs-fact slots the fixed k budget allocates.
_RAW_RECALL_DEFAULT = os.environ.get("TENET_RAW_RECALL", "").strip().lower() in ("1", "true", "on", "yes")


# --- Embedding-based key resolution (TENET_KEY_RESOLUTION) ------------------
# The per-message distiller keys the SAME real-world attribute inconsistently
# ("user::milk_preference" one turn, "user::milk" the next), so exact-skey
# collision misses most natural-language preference updates (measured true-fire
# 3.8% on PersonaMem-v2 retractions, BENCHMARK.md §13 — the reason we tied/lost
# on preference-drift while winning on stable-attribute churn). When ON, store()
# ALSO supersedes a current fact of the SAME subject whose attribute key is
# embedding-near this one (cosine >= _TAU_KEY) and value-compatible (not a
# sub-attribute of it). Fully LLM-free — reuses the fact embedder on the short
# attribute slug. Default is env-driven and runtime-overridable (the firing
# benchmark sweeps _TAU_KEY on a labeled set; see scripts/bench_supersession_firing.py).
def _env_off(name: str, default_on: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default_on
    return v not in ("0", "off", "false", "no")


# Default flipped ON 2026-07-10 after the firing-fix gates passed (see
# scripts/bench_supersession_firing.py + scratchpad/supersession_fix.md).
# Force off with TENET_KEY_RESOLUTION=off.
_KEY_RESOLUTION = _env_off("TENET_KEY_RESOLUTION", default_on=True)
_TAU_KEY = float(os.environ.get("TENET_KEY_RESOLUTION_TAU", "0.78"))  # swept: 80% true-fire
# _TEXT_FLOOR raised 0.35->0.66 (2026-07-10) after a shared-salient-word false-supersession
# ("user::surface_probe" vs "user::temporal_probe": both key AND text share "probe", clearing
# tau but NOT a 0.66 fact-text floor). Swept on the labeled set incl. 4 adversarial shared-word
# negatives: 0.66 gives 80% true-fire at 0% false-fire (robust across the 0.62-0.72 band; true-fire
# is unchanged from the old 0.35, so the floor costs nothing). The fact-text cosine is the clean
# separator — shared-word-but-different-fact pairs sit at <=0.70, real value-updates at >=0.72.
_TEXT_FLOOR = float(os.environ.get("TENET_KEY_RESOLUTION_TEXTFLOOR", "0.66"))
# Sub-attribute qualifiers: a key whose attribute is <another attribute> + one of
# these names a DIFFERENT property of the same object (pet vs pet_name, car vs
# car_color) — an update to one must never supersede the other. Meta qualifiers
# (preference/current/favorite/…) are deliberately NOT here, so "milk" and
# "milk_preference" DO collapse onto the same slot.
_SUB_ATTR_TOKENS = frozenset({
    "name", "brand", "color", "colour", "type", "model", "id", "number", "num",
    "count", "size", "age", "price", "cost", "date", "time", "year", "location",
    "city", "address", "owner", "maker", "author", "title", "breed", "make",
})


def _split_key(k: str) -> tuple[str, str]:
    # rpartition on the LAST "::" so multi-segment namespaces (the LangGraph store
    # joins namespace+key into one skey, e.g. "users::alex::prefs") keep the whole
    # namespace as the subject and only the final segment as the attribute — otherwise
    # "users::alex::prefs" and "users::sam::prefs" would share subject "users" and be
    # wrongly collapsed across users. For the distiller's 2-segment "user::milk" this
    # is identical to a first-"::" split.
    subj, _, attr = k.rpartition("::")
    return subj, attr


def _key_tokens(attr: str) -> set[str]:
    return {t for t in re.split(r"[_\W]+", attr.lower()) if t}


def _value_compatible(new_key: str, old_key: str) -> bool:
    """True if `new_key` may supersede `old_key`: SAME subject and NOT a
    sub-attribute refinement of the other. Blocks pet=dog vs pet_name=Rex and
    car vs car_color from collapsing, while keeping milk vs milk_preference
    compatible (the extra 'preference' token is not a sub-attribute qualifier)."""
    ns, na = _split_key(new_key)
    os_, oa = _split_key(old_key)
    if ns != os_ or not na or not oa:
        return False
    ta, tb = _key_tokens(na), _key_tokens(oa)
    if ta == tb:
        return True
    if ta < tb and (tb - ta) & _SUB_ATTR_TOKENS:
        return False
    if tb < ta and (ta - tb) & _SUB_ATTR_TOKENS:
        return False
    return True


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
        # WAL: commit is a WAL-file append instead of the rollback-journal's
        # create+write+fsync+delete dance — the leading hypothesis (docs/SCALE.md
        # "ingestion ceiling") for why per-insert commit() throughput collapsed with
        # store size. synchronous=NORMAL is the standard WAL pairing: still durable
        # across an app crash (committed transactions survive), only the last WAL
        # page is at risk on an OS crash/power loss — the accepted trade for a
        # per-call fsync that no longer scales with file size.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
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
        self._key_vec_cache: dict[str, np.ndarray] = {}  # attribute-slug embeddings (key resolution)
        self._index: ResidentIndex | None = None  # lazily built — see _ensure_index

    # ---- resident index (docs/SCALE.md) -----------------------------------
    def _ensure_index(self, dim_hint: int | None = None) -> None:
        """Build the resident index on first use, or refresh it if another process
        (a second HTTP/FC worker on the same db file) has written since our last
        build (`PRAGMA data_version` — cheap, O(1), never trips on OUR OWN writes
        since those are already applied incrementally by store()/forget_sweep()).

        `dim_hint`: the embedding dimensionality, needed to size a brand-new index
        when the table is still empty (nothing to infer it from) — store() always
        knows this from its own vector; recall()/_nearest_current() on a store that
        already has rows infer it from the first row instead and don't need the hint.
        """
        if self._index is not None:
            if self._index.is_stale(self.db):
                self._index.refresh(self.db)
            return
        row = self.db.execute("SELECT embedding FROM memories LIMIT 1").fetchone()
        if row is not None:
            d = len(row["embedding"]) // 4  # float32
        elif dim_hint is not None:
            d = dim_hint
        else:
            return  # nothing to build yet and no hint (empty store, read-only call)
        self._index = ResidentIndex.build(self.db, d)

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
        with self._lock:
            row_id = self._store_locked(
                text, key=key, kind=kind, source=source, pinned=pinned, salience=salience,
                valid_at=valid_at, supersede=supersede, dedup=dedup,
                surprise_gate=surprise_gate, vec=vec,
            )
            self.db.commit()
            self._dyn_dirty = True  # ledger changed -> refit dynamics lazily
            return row_id

    def store_many(self, items: list[dict]) -> list[int]:
        """Bulk insert with ONE commit for the whole batch, instead of store()'s one
        commit per item — the fix for per-call synchronous commit dominating
        ingestion throughput at scale (docs/SCALE.md "ingestion ceiling"; also see
        the WAL pragmas in `__init__`, which help every commit including this one).

        Each item is the same kwargs `store()` takes (`text` plus any of `key`,
        `kind`, `source`, `pinned`, `salience`, `valid_at`, `supersede`, `dedup`,
        `surprise_gate`, `_vec`). Per-item supersession/dedup/restatement semantics
        are UNCHANGED — this only changes when the transaction commits, not what
        each item does or its return value (a list of ids, same order as `items`,
        `-1` for a skipped surprise-gated raw observation, same as store()).

        Embeddings for items without `_vec` are computed via ONE batched
        `embed_batch()` call up front (outside the lock, same as store()'s single-
        item path) rather than one call per item — a real API-based embedder pays
        one round-trip instead of N.
        """
        if not items:
            return []
        need_embed = [i for i, it in enumerate(items) if it.get("_vec") is None]
        if need_embed:
            texts = [items[i]["text"].strip() for i in need_embed]
            vecs = self.embed_batch(texts)
            for i, v in zip(need_embed, vecs):
                items[i] = {**items[i], "_vec": v}
        ids: list[int] = []
        with self._lock:
            for it in items:
                text = it["text"].strip()
                if not text:
                    raise ValueError("empty memory")
                ids.append(self._store_locked(
                    text, key=it.get("key"), kind=it.get("kind", "fact"),
                    source=it.get("source"), pinned=it.get("pinned", False),
                    salience=it.get("salience", 0.5), valid_at=it.get("valid_at"),
                    supersede=it.get("supersede", 0.90), dedup=it.get("dedup", 0.985),
                    surprise_gate=it.get("surprise_gate"), vec=it["_vec"],
                ))
            self.db.commit()
            self._dyn_dirty = True
        return ids

    def _store_locked(self, text: str, *, key, kind, source, pinned, salience,
                       valid_at, supersede, dedup, surprise_gate, vec: np.ndarray) -> int:
        """Everything store()/store_many() do per item EXCEPT acquiring the lock and
        committing — so store_many() can run many of these in one transaction.
        Caller MUST hold self._lock. Mirrors every DB mutation into the resident
        index incrementally (append/mark_expired), so recall() never needs to
        re-query for what THIS process just wrote."""
        t = self._now()
        va = valid_at if valid_at is not None else t
        self._ensure_index(dim_hint=vec.shape[0])
        idx = self._index

        if kind == "raw":
            # Belief-store efficiency (predictive-coding principle): only store a raw
            # observation the memory does NOT already predict. If it's near-identical
            # to an existing raw slice (cosine >= surprise_gate), it carries no new
            # information — skip it. Shrinks the store without losing novel detail.
            # Vectorized (was a Python for-loop over every raw row — docs/SCALE.md).
            if surprise_gate is not None and idx is not None and idx.size:
                mask = idx.mask_raw_current()
                if mask.any():
                    sims = idx.matrix[: idx.size][mask] @ vec
                    if float(sims.max()) >= surprise_gate:
                        return -1  # redundant observation, not stored
        elif key is not None:
            prior = self.db.execute(
                "SELECT id, text, pinned, salience FROM memories "
                "WHERE skey=? AND archived=0 AND expired_at IS NULL",
                (key,),
            ).fetchall()
            for row in prior:
                if row["text"] == text:
                    self._touch_locked(row["id"])
                    return row["id"]          # exact restatement of a keyed fact
            superseded_ids = []
            for row in prior:                 # same key, new value ⇒ supersede all priors
                # A pinned/high-salience fact-slot keeps those properties across
                # value updates (pinning "residence" survives a move).
                pinned = pinned or bool(row["pinned"])
                salience = max(salience, row["salience"])
                self.db.execute(
                    "UPDATE memories SET invalid_at=?, expired_at=? WHERE id=?",
                    (va, t, row["id"]),
                )
                superseded_ids.append(row["id"])
            if superseded_ids and idx is not None:
                idx.mark_expired(superseded_ids, va, t)
            # Embedding-based key resolution: also retire same-subject current facts
            # whose attribute key the distiller spelled differently (milk vs
            # milk_preference) — the fix for the 3.8% NL-update fire rate. Guarded
            # (value_compatible + text floor) so distinct attributes never collapse.
            if _KEY_RESOLUTION:
                p2, s2, ids2 = self._resolve_key_supersede(key, vec, va, t)
                pinned = pinned or p2
                salience = max(salience, s2)
                if ids2 and idx is not None:
                    idx.mark_expired(ids2, va, t)
        else:
            hit = self._nearest_current(vec)  # (id, sim, text)
            if hit and hit[1] >= dedup:
                self._touch_locked(hit[0])
                return hit[0]
            if hit and hit[1] >= supersede and hit[2] != text:
                self.db.execute(
                    "UPDATE memories SET invalid_at=?, expired_at=? WHERE id=?",
                    (va, t, hit[0]),
                )
                if idx is not None:
                    idx.mark_expired([hit[0]], va, t)
        cur = self.db.execute(
            "INSERT INTO memories(text, kind, source, skey, embedding, salience, valid_at, "
            "created_at, last_access, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (text, kind, source, key, vec.tobytes(), float(salience), va, t, t, int(pinned)),
        )
        row_id = cur.lastrowid
        if idx is not None:
            idx.append(id=row_id, text=text, kind=kind, source=source, skey=key,
                       embedding=vec, salience=float(salience), valid_at=va,
                       created_at=t, last_access=t, pinned=bool(pinned))
        return row_id

    # ---- retraction / tombstone (docs/COMPARISON.md follow-up #3) ------------
    def retract(self, key: str, *, valid_at: float | None = None) -> int:
        """Explicit "forget X" — retire every CURRENT fact under `key` WITHOUT
        replacing it with a new value. Distinct from supersession (store()'s keyed
        path), which retires-then-REPLACES: a retraction is a pure deletion of the
        belief, not a value update. Reuses the same bi-temporal machinery
        (invalid_at/expired_at) supersession does, so the effect on recall is
        identical in kind — current recall() no longer returns it, `recall(as_of=t)`
        for t before the retraction still does (history isn't erased, matching
        every other "retire" path in this store) — but no new row is inserted, so
        there is no "current value" to report afterward; `uncertain_facts()` and
        `list_beliefs()` correctly show nothing current for this key post-retraction.

        Returns the number of rows retracted (0 if nothing was current under `key`).
        Pinned facts ARE retractable (an explicit "forget X" should win over a pin —
        unlike the forgetting SWEEP, which never touches pinned facts; a user asking
        to forget something is a stronger, explicit signal than decay)."""
        t = self._now()
        va = valid_at if valid_at is not None else t
        with self._lock:
            self._ensure_index()
            rows = self.db.execute(
                "SELECT id FROM memories WHERE skey=? AND archived=0 AND expired_at IS NULL",
                (key,),
            ).fetchall()
            if not rows:
                return 0
            ids = [r["id"] for r in rows]
            self.db.executemany(
                "UPDATE memories SET invalid_at=?, expired_at=? WHERE id=?",
                [(va, t, i) for i in ids],
            )
            self.db.commit()
            self._dyn_dirty = True
            if self._index is not None:
                self._index.mark_expired(ids, va, t)
            return len(ids)

    def _key_emb(self, key: str) -> np.ndarray:
        """Embedding of a key's readable ATTRIBUTE slug (subject stripped so the shared
        'user::' prefix doesn't inflate every pair's similarity). Cached per key."""
        v = self._key_vec_cache.get(key)
        if v is None:
            _subj, attr = _split_key(key)
            v = self._embed((attr.replace("_", " ") or key).strip())
            self._key_vec_cache[key] = v
        return v

    def _resolve_key_supersede(self, key: str, text_vec: np.ndarray,
                               va: float, t: float) -> tuple[bool, float, list[int]]:
        """Supersede current same-subject facts whose attribute key is embedding-near
        `key` (cosine >= _TAU_KEY) and value-compatible. Returns (any_pinned,
        max_salience, superseded_ids) — the third so the caller can mirror the
        UPDATE into the resident index (docs/SCALE.md) without this method needing
        to know about it. Runs inside store()'s lock. LLM-free.

        Candidate lookup is resident-index-backed (`ResidentIndex.rows_for_subject`,
        an incrementally-maintained subject->positions dict), NOT the SQL this used
        to run: `EXPLAIN QUERY PLAN` on the original `... skey LIKE ?` query showed
        `SCAN memories` — sqlite does NOT use the skey index for a LIKE prefix search
        combined with the surrounding non-indexed AND-conditions here, so this was a
        full-table scan on every keyed insert, confirmed as the dominant remaining
        cause of ingestion throughput degrading with store size after the resident
        matrix + WAL fixes (docs/SCALE.md "ingestion ceiling", before/after table)."""
        subj, _ = _split_key(key)
        if not subj:
            return False, 0.0, []
        self._ensure_index(dim_hint=text_vec.shape[0])
        if self._index is None:
            return False, 0.0, []
        rows = self._index.rows_for_subject(subj, exclude_skey=key)
        if not rows:
            return False, 0.0, []
        nk = self._key_emb(key)
        any_pinned, max_sal, superseded_ids = False, 0.0, []
        for row in rows:
            ok = row["skey"]
            if not _value_compatible(key, ok):
                continue
            if float(np.dot(nk, self._key_emb(ok))) < _TAU_KEY:
                continue
            # domain floor on the FACT texts: a spuriously high key-sim must not collapse
            # two facts that are actually about unrelated things.
            ov = np.frombuffer(row["embedding"], dtype=np.float32)
            if float(np.dot(text_vec, ov)) < _TEXT_FLOOR:
                continue
            any_pinned = any_pinned or bool(row["pinned"])
            max_sal = max(max_sal, float(row["salience"]))
            self.db.execute(
                "UPDATE memories SET invalid_at=?, expired_at=? WHERE id=?",
                (va, t, row["id"]),
            )
            superseded_ids.append(row["id"])
        return any_pinned, max_sal, superseded_ids

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
        agg_reader: bool = _AGG_READER_DEFAULT,
        raw_recall: bool = _RAW_RECALL_DEFAULT,
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

        `agg_reader` — OPTIONAL CAR-style read-time recency aggregation
        (aggregate.py, docs/COMPARISON.md follow-up #1): if the returned pool has
        more than one memory sharing the same `key`, keep only the highest-`valid_at`
        member of each group (LLM-free max(serial)-equivalent, since `valid_at` IS
        the serial-equivalent signal). Default OFF (`_AGG_READER_DEFAULT`,
        `TENET_AGG_READER=1` to opt in globally). Runs LAST, after expand/hops, so it
        sees the final pool; a pure filter (drops entries, never reorders/rescopes
        the ones that survive) — does not touch the annotation-only ranking
        invariant.

        `raw_recall` — OPTIONAL raw-turn-favored dual-pool split (docs/COMPARISON.md
        follow-up #2, for LoCoMo-style verbatim-recall regimes): the default split
        caps raw slices at k//2 of the initial top-k; ON, raw slices get PRIORITY up
        to the full k budget instead. Default OFF (`_RAW_RECALL_DEFAULT`,
        `TENET_RAW_RECALL=1` to opt in globally) — with it off this method is
        byte-identical to before this parameter existed. Only changes pool
        COMPOSITION (how many raw-vs-fact slots k allocates); still relevance x
        decay ranked within each pool, same annotation-only invariant.
        """
        qv = self._embed(query)  # network call — kept outside the lock
        with self._lock:
            rows, mat, decay_vec = self._rows_matrix_decay_as_of(as_of)
            if not rows:
                return []
            # One BLAS matmul against the RESIDENT matrix (docs/SCALE.md) — no sqlite
            # fetch and no per-row deserialization on the way in; this was 356ms of a
            # 1055ms call at 100k facts (4 hidden full-table SELECTs: this one, plus
            # the ones inside _dynamics/_expired_fact_matrix/_expired_keyed_rows
            # below, all now index-backed too), the matmul itself was always cheap
            # (docs/HARNESS.md §3: 2.6ms at 100k).
            rels = mat @ qv                               # cosine (both unit)
            # Drift model: annotate each keyed fact with the LEARNED probability
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
                    if row["kind"] == "raw" or row["skey"] is None:
                        continue  # _p_valid returns None for these anyway — skip the call
                    p = self._p_valid(row, dyn, now)
                    if p is not None:
                        conf[row["id"]] = p

            # decay_vec is vectorized (docs/SCALE.md — was a Python for-loop calling
            # _decay(row) once per row: math.pow/math.log1p each time, 94ms at 100k).
            scored = [
                (float(rel) * float(dec), float(rel), row)
                for rel, dec, row in zip(rels, decay_vec, rows)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)

            # Belief-store consistency: the current facts are the belief state. A raw
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
            if raw_recall:
                # Raw-turn-favored split (docs/COMPARISON.md follow-up #2): raw
                # slices may fill the WHOLE k budget, facts only get the leftover —
                # the opposite priority of the default split below.
                n_raw = min(len(raws), k)
                picked = raws[:n_raw] + facts[: k - n_raw]
                leftover_order = raws[n_raw:] + facts[k - n_raw:]
            else:
                n_raw = min(len(raws), k // 2)
                picked = facts[: k - n_raw] + raws[:n_raw]
                leftover_order = facts[k - n_raw:] + raws[n_raw:]
            if len(picked) < k:  # backfill from whatever remains, best rank first
                leftover_order.sort(key=lambda x: x[0], reverse=True)
                picked += leftover_order[: k - len(picked)]

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
            return aggregate_by_key(out) if agg_reader else out

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

    # ---- fact dynamics (staleness/confidence hints) -------------------------
    def _dynamics(self):
        """Learned lifetime model, refit lazily from the ledger when it changed.

        Default is the closed-form Gamma-exponential survival model (dynamics.py,
        numpy-only, no LLM) — per-key-class hazard rates fitted from this store's
        own supersession history, used to flag which current facts are probably
        stale and worth re-confirming (`uncertain_facts()`). Set env
        TENET_DYNAMICS=neural (+ TENET_NEURAL_NPZ=<path>) to swap in the trained
        drift model (dynamics_neural.py, numpy-only inference — opt-in, default
        off, see paper/tenet.md for the measured NLL/calibration results). The
        neural path needs per-event value embeddings, so it binds the ledger's
        stored embeddings — use it with the bge-small local embedder it was
        trained on (EMBED_PROVIDER=local, 384d)."""
        if self._dyn is not None and not self._dyn_dirty:
            return self._dyn
        # Index-backed (docs/SCALE.md) — was its own full-table SELECT on every
        # refit; Dynamics.fit()/build_from_ledger() themselves are UNCHANGED, only
        # their input's source moved from sqlite to the resident index.
        self._ensure_index()
        if self._index is not None:
            idx = self._index
            mask = (~idx.is_raw[: idx.size]) & idx.has_skey[: idx.size]
            rows = idx.rows_for_mask(mask)
        else:
            rows = []
        nd = None
        if os.environ.get("TENET_DYNAMICS", "").lower() == "neural":
            from .dynamics_neural import build_from_ledger  # numpy-only drift model
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
        history but excluded from current recall. Returns count archived.
        Vectorized (docs/SCALE.md) — was a Python for-loop calling `_decay(row)`
        once per current row; the resident index is also compacted (rows physically
        removed, not just flagged) so recall() never has to filter them again."""
        with self._lock:
            self._ensure_index()
            if self._index is None:
                return 0
            idx = self._index
            mask_cur = idx.mask_current()
            if not mask_cur.any():
                return 0
            decay = idx.decay(self._now(), _HALFLIFE_S)
            to_archive = mask_cur & (~idx.pinned[: idx.size]) & (decay < _FORGET_THRESHOLD)
            ids = [int(idx.ids[p]) for p in np.flatnonzero(to_archive)]
            if not ids:
                return 0
            self.db.executemany("UPDATE memories SET archived=1 WHERE id=?", [(i,) for i in ids])
            self.db.commit()
            idx.remove(ids)
            return len(ids)

    # ---- helpers -----------------------------------------------------------
    # All resident-index-backed now (docs/SCALE.md) — no SQL, no per-row
    # `np.frombuffer` deserialization; that used to be 4 separate full-table
    # `fetchall()`s hidden inside one `recall()` call (356ms of 1055ms at 100k).
    def _current_rows(self):
        """Live, currently-true memories (not archived, not superseded)."""
        self._ensure_index()
        if self._index is None:
            return []
        return self._index.rows_for_mask(self._index.mask_current())

    def _rows_as_of(self, as_of: float | None):
        if as_of is None:
            return self._current_rows()
        self._ensure_index()
        if self._index is None:
            return []
        return self._index.rows_for_mask(self._index.mask_as_of(as_of))

    def _rows_matrix_decay_as_of(self, as_of: float | None):
        """(rows, embedding submatrix, decay array) for recall()'s scoring pass, in
        one shot — the fetch+deserialize+decay-loop replacement. `rows`/`mat`/`decay`
        are index-aligned (same order, same length)."""
        self._ensure_index()
        if self._index is None:
            return [], None, None
        idx = self._index
        mask = idx.mask_current() if as_of is None else idx.mask_as_of(as_of)
        if not mask.any():
            return [], None, None
        rows, mat = idx.rows_and_matrix_for_mask(mask)
        decay = idx.decay(self._now(), _HALFLIFE_S)[mask]
        return rows, mat, decay

    def _expired_fact_matrix(self):
        """Embeddings of superseded (expired) facts — the retired belief values.
        Returns an (M×d) matrix or None."""
        self._ensure_index()
        if self._index is None:
            return None
        idx = self._index
        mask = idx.mask_expired_fact()
        if not mask.any():
            return None
        return idx.matrix[: idx.size][mask]

    def _expired_keyed_rows(self):
        """Superseded KEYED facts (id, skey, embedding) — consistency.py's key-scoped
        input; a superset of `_expired_fact_matrix`'s rows filtered to skey IS NOT NULL,
        fetched separately since that helper only returns bare embeddings."""
        self._ensure_index()
        if self._index is None:
            return []
        return self._index.rows_for_mask(self._index.mask_expired_keyed_fact())

    def _nearest_current(self, vec: np.ndarray):
        """Highest-cosine current row (unkeyed store()'s dedup/supersede path).
        Vectorized (docs/SCALE.md) — was a Python for-loop computing one dot
        product per current row (`_nearest_current`'s namesake unvectorized scan,
        confirmed O(n) per insert -> O(n^2) total for bulk unkeyed ingestion)."""
        self._ensure_index(dim_hint=vec.shape[0])
        idx = self._index
        if idx is None or idx.size == 0:
            return None
        mask = idx.mask_current()
        if not mask.any():
            return None
        n = idx.size
        sims = idx.matrix[:n] @ vec
        sims = np.where(mask, sims, -np.inf)
        best_pos = int(np.argmax(sims))  # first-encountered on ties, matches the old `sim > best[1]`
        return (int(idx.ids[best_pos]), float(sims[best_pos]), idx.text[best_pos])

    def _touch_locked(self, mem_id: int) -> None:
        """Bump uses/last_access, no commit — caller (store()/store_many() via
        _store_locked) commits once for the whole batch."""
        now = self._now()
        self.db.execute("UPDATE memories SET uses=uses+1, last_access=? WHERE id=?", (now, mem_id))
        if self._index is not None:
            self._index.touch(mem_id, now)

    def _touch(self, mem_id: int):
        """Bump uses/last_access and commit immediately — caller must already hold
        self._lock (recall() does). A single small UPDATE per touched result, not
        the bulk-insert case store_many() exists for."""
        self._touch_locked(mem_id)
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
