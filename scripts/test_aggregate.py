"""Deterministic tests for aggregate.py's read-time recency aggregation
(docs/COMPARISON.md follow-up #1, "CAR-style read-time max(serial) aggregation").
Pure-function tests (no store, no LLM) plus one recall()-level integration check
that the flag is OFF by default and composes correctly when explicitly enabled.

Run: python scripts/test_aggregate.py
"""
import os
os.environ.setdefault("EMBED_PROVIDER", "local")  # before importing config

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet.aggregate import aggregate_by_key  # noqa: E402
from tenet.memory import Memory, MemoryCore  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    mark = "ok" if cond else "FAIL"
    print(f"  {mark} {name}  {detail}")
    if not cond:
        FAILS.append(name)


def mk(id, text, key, valid_at, created_at=None, kind="fact"):
    return Memory(id=id, text=text, score=0.5,
                  created_at=created_at if created_at is not None else valid_at,
                  valid_at=valid_at, invalid_at=None, expired_at=None,
                  last_access=valid_at, uses=0, pinned=False, salience=0.5,
                  kind=kind, source=None, key=key)


def test_pure_function():
    # same-key group -> keep only the highest valid_at
    pool = [mk(1, "old value", "user::residence", 100),
            mk(2, "new value", "user::residence", 200),
            mk(3, "unrelated", "user::job", 150)]
    out = aggregate_by_key(pool)
    check("collapses same-key group to latest valid_at", [m.id for m in out] == [2, 3],
          f"got ids={[m.id for m in out]}")

    # tie on valid_at -> highest created_at wins
    pool2 = [mk(1, "a", "user::x", 100, created_at=100), mk(2, "b", "user::x", 100, created_at=200)]
    out2 = aggregate_by_key(pool2)
    check("ties broken by created_at", [m.id for m in out2] == [2], f"got {[m.id for m in out2]}")

    # unkeyed entries always pass through untouched, even "duplicated"
    pool3 = [mk(1, "raw a", None, 100), mk(2, "raw b", None, 100)]
    out3 = aggregate_by_key(pool3)
    check("unkeyed entries never grouped", len(out3) == 2, f"got {len(out3)}")

    # preserves original relative order of survivors (not re-sorted by valid_at)
    pool4 = [mk(1, "x", "a::a", 50), mk(2, "y", "b::b", 999), mk(3, "z", "a::a", 200)]
    out4 = aggregate_by_key(pool4)
    check("survivor order matches original sequence, not re-sorted",
          [m.id for m in out4] == [2, 3], f"got {[m.id for m in out4]}")

    # no-op on an already-conflict-free pool
    pool5 = [mk(1, "x", "a::a", 50), mk(2, "y", "b::b", 999)]
    out5 = aggregate_by_key(pool5)
    check("no-op on conflict-free pool", out5 == pool5)

    # three-way group: highest wins, both losers dropped
    pool6 = [mk(1, "v1", "k::k", 10), mk(2, "v2", "k::k", 30), mk(3, "v3", "k::k", 20)]
    out6 = aggregate_by_key(pool6)
    check("three-way group keeps only the single latest", [m.id for m in out6] == [2],
          f"got {[m.id for m in out6]}")


def test_recall_integration():
    """agg_reader defaults OFF (unchanged behavior) and, when explicitly enabled,
    de-conflicts a pool that intentionally has two "current" rows sharing a key —
    a scenario normal store()-time supersession prevents, so this bypasses it via
    direct SQL (the same pattern bulk_seed()-style test helpers use elsewhere) to
    exercise the READ-time filter in isolation."""
    db = Path(tempfile.mkdtemp()) / "agg_test.db"
    core = MemoryCore(db)
    import numpy as np
    rng = np.random.default_rng(7)
    v1 = rng.standard_normal(384).astype(np.float32); v1 /= np.linalg.norm(v1)
    now = core._now()
    # two DIFFERENT vectors so query matching doesn't accidentally favor one, but
    # SAME skey and BOTH current (expired_at NULL) — the artificial-duplicate case.
    id1 = core.store("Alex lives in Montreal.", key="user::residence", valid_at=now - 100, _vec=v1)
    # bypass supersession: insert a second "current" row under the SAME key directly
    v2 = rng.standard_normal(384).astype(np.float32); v2 /= np.linalg.norm(v2)
    # blend toward v1 so both score similarly for the same query
    v2 = (v1 * 0.9 + v2 * 0.1); v2 /= np.linalg.norm(v2)
    with core._lock:
        cur = core.db.execute(
            "INSERT INTO memories(text, kind, source, skey, embedding, salience, valid_at, "
            "created_at, last_access, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("Alex lives in Toronto.", "fact", None, "user::residence", v2.tobytes(), 0.5,
             now, now, now, 0))
        core.db.commit()
        if core._index is not None:
            core._index.append(id=cur.lastrowid, text="Alex lives in Toronto.", kind="fact",
                                source=None, skey="user::residence", embedding=v2,
                                salience=0.5, valid_at=now, created_at=now, last_access=now,
                                pinned=False)
    id2 = cur.lastrowid

    off = core.recall("where does alex live", k=5)
    check("agg_reader default OFF: both same-key rows still in the pool",
          {m.id for m in off} >= {id1, id2}, f"ids={[m.id for m in off]}")

    on = core.recall("where does alex live", k=5, agg_reader=True)
    ids_on = {m.id for m in on}
    check("agg_reader=True: only the higher-valid_at row of the duplicate-key group survives",
          id2 in ids_on and id1 not in ids_on, f"ids={ids_on}, kept id2(Toronto, newer)={id2 in ids_on}")
    core.close()


def main() -> int:
    print("=== pure-function tests ===")
    test_pure_function()
    print("=== recall() integration ===")
    test_recall_integration()
    if FAILS:
        print("\nFAILURES:", FAILS)
        return 1
    print("\nALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
