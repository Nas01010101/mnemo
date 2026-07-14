"""ResidentIndex — an in-memory mirror of MemoryCore's `memories` table (every
non-archived row: current AND superseded, needed for as_of/time-travel and the
stale-echo consistency check), built once and updated incrementally.

Why this exists (docs/SCALE.md, measured): `recall()` used to re-`SELECT` and
re-`np.frombuffer`-deserialize the WHOLE table on every single call — 327ms of a
1055ms call at 100k facts, plus THREE more full-table SQL queries hidden inside the
same call (`_dynamics()`'s refit input, `_expired_fact_matrix()`,
`_expired_keyed_rows()` — 356ms of `fetchall()` total across all four). None of that
scales with query relevance, only with store size — it's pure overhead. This module
removes it: embeddings + the metadata needed to mask/score/rank live in numpy arrays
here, kept warm across calls; `MemoryCore` mutates them incrementally (`append` on
`store()`, `mark_expired` on supersession, `remove` on `forget_sweep`) instead of
re-querying sqlite.

Growth: a classic dynamic array (capacity doubles when exceeded), so `append()` is
amortized O(1) — NOT a fresh allocation+copy per insert, which would silently
reintroduce an O(n^2) wall for exactly the kind of bulk ingestion this is meant to
help.

What's still O(n) per call, deliberately not addressed here: `dynamics.py`'s
per-key-class survival fit/lookup (`Dynamics.fit`, `.p_valid`) — that's a numerically
careful module this change doesn't touch; `_dynamics()` now feeds it resident rows
instead of a fresh SQL query (removes the query, not the fit itself). See
docs/SCALE.md for the full before/after breakdown.

Multi-process safety: this index has NO visibility into writes made by another
process (a second FC/HTTP worker, say) through a DIFFERENT sqlite connection.
`is_stale()` checks `PRAGMA data_version` — a counter sqlite bumps whenever any
OTHER connection commits, but never for commits made through this same connection
(exactly the signal needed: our own writes are already applied incrementally, we
only need to detect someone else's). `MemoryCore` checks this cheaply before trusting
the index and does one full `refresh()` if it's gone stale — correct under
multi-process writers, at the cost of one full rebuild the first time another
process's write is noticed (not on every call).
"""
from __future__ import annotations

import numpy as np

# Sentinel for "never" (invalid_at/expired_at IS NULL in sqlite — still true/still
# current). Using +inf instead of None/NaN lets every bi-temporal comparison below
# collapse into a single vectorized `>` — `NEVER > as_of` is always True, matching
# the SQL "expired_at IS NULL OR expired_at>?" without a separate null-check branch.
NEVER = float("inf")

_ROW_COLUMNS = (
    "id, text, kind, source, skey, embedding, salience, valid_at, "
    "invalid_at, created_at, expired_at, last_access, uses, pinned"
)


def _read_data_version(db) -> int:
    return db.execute("PRAGMA data_version").fetchone()[0]


def _subject(skey: str | None) -> str | None:
    """'user::residence' -> 'user' (rpartition on the LAST '::', matching
    memory.py's `_split_key` — duplicated here, not imported, to keep this module
    free of memory.py's key-semantics knowledge; it's 2 lines, not worth a
    circular import)."""
    if not skey:
        return None
    subj, _, _ = skey.rpartition("::")
    return subj or None


class ResidentIndex:
    def __init__(self, d: int, capacity: int = 256):
        self.d = d
        self._cap = max(capacity, 1)
        self.size = 0
        self.ids = np.empty(self._cap, dtype=np.int64)
        self.matrix = np.empty((self._cap, d), dtype=np.float32)
        self.valid_at = np.empty(self._cap, dtype=np.float64)
        self.invalid_at = np.empty(self._cap, dtype=np.float64)
        self.created_at = np.empty(self._cap, dtype=np.float64)
        self.expired_at = np.empty(self._cap, dtype=np.float64)
        self.last_access = np.empty(self._cap, dtype=np.float64)
        self.uses = np.empty(self._cap, dtype=np.int32)
        self.pinned = np.empty(self._cap, dtype=bool)
        self.salience = np.empty(self._cap, dtype=np.float32)
        self.is_raw = np.empty(self._cap, dtype=bool)
        self.has_skey = np.empty(self._cap, dtype=bool)
        self.text: list[str] = []
        self.skey: list[str | None] = []
        self.source: list[str | None] = []
        self._pos: dict[int, int] = {}  # row id -> array position
        self.by_subject: dict[str, list[int]] = {}  # skey subject -> positions
        self.data_version: int | None = None

    # ---- construction / refresh -------------------------------------------
    @classmethod
    def build(cls, db, d: int) -> "ResidentIndex":
        # _ROW_COLUMNS is a module-level constant (never user input); all user-supplied
        # values in this codebase go through parameterized queries.
        rows = db.execute(f"SELECT {_ROW_COLUMNS} FROM memories WHERE archived=0").fetchall()  # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query, python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        idx = cls(d, capacity=max(len(rows), 256))
        for row in rows:
            idx._append_row(row)
        idx.data_version = _read_data_version(db)
        return idx

    def is_stale(self, db) -> bool:
        """True if another connection has committed since our last (re)build. Our
        OWN writes (via append/mark_expired/remove) never trip this — only someone
        else's do, which is exactly the case a resident cache needs to detect."""
        return self.data_version is None or _read_data_version(db) != self.data_version

    def refresh(self, db) -> None:
        """Full rebuild in place (rare path: first call, or another process wrote)."""
        fresh = ResidentIndex.build(db, self.d)
        self.__dict__.update(fresh.__dict__)

    # ---- growth (amortized O(1) append) ------------------------------------
    def _ensure_capacity(self, extra: int) -> None:
        need = self.size + extra
        if need <= self._cap:
            return
        new_cap = max(need, self._cap * 2)
        for name in ("ids", "valid_at", "invalid_at", "created_at", "expired_at",
                     "last_access", "uses", "pinned", "salience", "is_raw", "has_skey"):
            old = getattr(self, name)
            grown = np.empty(new_cap, dtype=old.dtype)
            grown[: self.size] = old[: self.size]
            setattr(self, name, grown)
        grown_mat = np.empty((new_cap, self.d), dtype=np.float32)
        grown_mat[: self.size] = self.matrix[: self.size]
        self.matrix = grown_mat
        self._cap = new_cap

    def _append_row(self, row) -> None:
        i = self.size
        self.ids[i] = row["id"]
        self.matrix[i] = np.frombuffer(row["embedding"], dtype=np.float32)
        self.valid_at[i] = row["valid_at"]
        self.invalid_at[i] = row["invalid_at"] if row["invalid_at"] is not None else NEVER
        self.created_at[i] = row["created_at"]
        self.expired_at[i] = row["expired_at"] if row["expired_at"] is not None else NEVER
        self.last_access[i] = row["last_access"]
        self.uses[i] = row["uses"]
        self.pinned[i] = bool(row["pinned"])
        self.salience[i] = row["salience"]
        self.is_raw[i] = row["kind"] == "raw"
        self.has_skey[i] = row["skey"] is not None
        self.text.append(row["text"])
        self.skey.append(row["skey"])
        self.source.append(row["source"])
        self._pos[int(row["id"])] = i
        subj = _subject(row["skey"])
        if subj:
            self.by_subject.setdefault(subj, []).append(i)
        self.size += 1

    def append(self, *, id: int, text: str, kind: str, source: str | None,
               skey: str | None, embedding: np.ndarray, salience: float,
               valid_at: float, created_at: float, last_access: float,
               pinned: bool) -> None:
        """Mirror one freshly-INSERTed row. Amortized O(1)."""
        self._ensure_capacity(1)
        self._append_row({
            "id": id, "text": text, "kind": kind, "source": source, "skey": skey,
            "embedding": embedding.tobytes(), "salience": salience, "valid_at": valid_at,
            "invalid_at": None, "created_at": created_at, "expired_at": None,
            "last_access": last_access, "uses": 0, "pinned": pinned,
        })

    # ---- incremental updates (mirror an UPDATE, no rebuild) ----------------
    def mark_expired(self, ids, invalid_at: float, expired_at: float) -> None:
        for row_id in ids:
            pos = self._pos.get(int(row_id))
            if pos is not None:
                self.invalid_at[pos] = invalid_at
                self.expired_at[pos] = expired_at

    def touch(self, row_id: int, now: float) -> None:
        pos = self._pos.get(int(row_id))
        if pos is not None:
            self.uses[pos] += 1
            self.last_access[pos] = now

    def remove(self, ids) -> None:
        """Physically drop rows (forget_sweep's archive=1) — the only structural
        change that isn't O(1): compacts the arrays. Not on the recall()/store() hot
        path (forget_sweep runs occasionally, not per query/insert)."""
        ids = set(int(i) for i in ids)
        if not ids or self.size == 0:
            return
        keep = ~np.isin(self.ids[: self.size], np.fromiter(ids, dtype=np.int64))
        new_size = int(keep.sum())
        for name in ("ids", "valid_at", "invalid_at", "created_at", "expired_at",
                     "last_access", "uses", "pinned", "salience", "is_raw", "has_skey"):
            arr = getattr(self, name)
            arr[:new_size] = arr[: self.size][keep]
        self.matrix[:new_size] = self.matrix[: self.size][keep]
        self.text = [t for t, k in zip(self.text, keep) if k]
        self.skey = [s for s, k in zip(self.skey, keep) if k]
        self.source = [s for s, k in zip(self.source, keep) if k]
        self.size = new_size
        self._pos = {int(self.ids[p]): p for p in range(new_size)}
        self.by_subject = {}
        for p in range(new_size):
            subj = _subject(self.skey[p])
            if subj:
                self.by_subject.setdefault(subj, []).append(p)

    # ---- masks (vectorized — the bi-temporal filters _rows_as_of used to run in SQL) --
    def mask_current(self) -> np.ndarray:
        return self.expired_at[: self.size] == NEVER

    def mask_as_of(self, as_of: float) -> np.ndarray:
        n = self.size
        return (
            (self.created_at[:n] <= as_of)
            & (self.expired_at[:n] > as_of)
            & (self.valid_at[:n] <= as_of)
            & (self.invalid_at[:n] > as_of)
        )

    def mask_raw_current(self) -> np.ndarray:
        return self.is_raw[: self.size] & self.mask_current()

    def mask_expired_fact(self) -> np.ndarray:
        """Superseded (expired_at set) facts, any key or none — matches
        `_expired_fact_matrix`'s old SQL exactly (`kind='fact' AND expired_at IS
        NOT NULL`, no skey requirement)."""
        n = self.size
        return (~self.is_raw[:n]) & (self.expired_at[:n] != NEVER)

    def mask_expired_keyed_fact(self) -> np.ndarray:
        """Superseded KEYED facts — matches `_expired_keyed_rows`'s old SQL exactly
        (adds `skey IS NOT NULL` on top of `mask_expired_fact`). consistency.py's
        key-scoped input."""
        n = self.size
        return self.mask_expired_fact() & self.has_skey[:n]

    # ---- vectorized decay (was a Python for-loop calling _decay(row) per row) ----
    def decay(self, now: float, halflife_s: float) -> np.ndarray:
        n = self.size
        age = now - self.last_access[:n]
        recency = np.power(0.5, age / halflife_s)
        use_boost = 1.0 + np.log1p(self.uses[:n]) * 0.15
        sal_boost = 0.6 + 0.8 * self.salience[:n]
        raw = np.minimum(1.0, recency * use_boost * sal_boost)
        return np.where(self.pinned[:n], 1.0, raw)

    # ---- row reconstruction (ONLY for the small set of rows actually returned —
    # never called per-n; see memory.py's recall()) --------------------------------
    def row_dict(self, pos: int) -> dict:
        return {
            "id": int(self.ids[pos]),
            "text": self.text[pos],
            "kind": "raw" if self.is_raw[pos] else "fact",
            "source": self.source[pos],
            "skey": self.skey[pos],
            "embedding": self.matrix[pos].tobytes(),
            "salience": float(self.salience[pos]),
            "valid_at": float(self.valid_at[pos]),
            "invalid_at": None if self.invalid_at[pos] == NEVER else float(self.invalid_at[pos]),
            "created_at": float(self.created_at[pos]),
            "expired_at": None if self.expired_at[pos] == NEVER else float(self.expired_at[pos]),
            "last_access": float(self.last_access[pos]),
            "uses": int(self.uses[pos]),
            "pinned": bool(self.pinned[pos]),
        }

    def rows_for_subject(self, subject: str, exclude_skey: str | None = None) -> list[dict]:
        """Current, keyed FACT rows sharing this skey subject (e.g. all of "user"'s
        attributes) — memory.py's `_resolve_key_supersede` candidate set. Was a
        `skey LIKE ? || '::%'` SQL scan; confirmed via `EXPLAIN QUERY PLAN` to NOT
        use the skey index (LIKE + the surrounding non-indexed AND-conditions push
        sqlite's planner to a full `SCAN memories` — the dominant remaining cause of
        ingestion throughput degrading with store size, docs/SCALE.md). `by_subject`
        makes this O(1) average instead of O(n)."""
        out = []
        for p in self.by_subject.get(subject, ()):
            if self.is_raw[p] or self.expired_at[p] != NEVER:
                continue
            if exclude_skey is not None and self.skey[p] == exclude_skey:
                continue
            out.append(self.row_dict(p))
        return out

    def rows_for_mask(self, mask: np.ndarray) -> list[dict]:
        """Row dicts for a (typically small) masked subset — e.g. raw candidates or
        superseded keyed facts, never the full n."""
        return [self.row_dict(int(p)) for p in np.flatnonzero(mask)]

    def rows_and_matrix_for_mask(self, mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """Like `rows_for_mask`, but ALSO returns the matching (M×d) embedding
        submatrix directly from the resident matrix — no serialize-to-bytes +
        `np.frombuffer` round trip through `row_dict`'s `embedding` field. This is
        what `recall()` uses for its scoring matmul (memory.py); `rows_for_mask`
        alone is for callers that only need text/metadata (e.g. `list_beliefs`)."""
        positions = np.flatnonzero(mask)
        rows = [self.row_dict(int(p)) for p in positions]
        mat = self.matrix[: self.size][mask]
        return rows, mat
