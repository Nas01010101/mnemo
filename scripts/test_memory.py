"""Exercise the bi-temporal MemoryCore against the live Qwen embedding API.
Run: python scripts/test_memory.py   (uses a throwaway temp DB)
"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet import memory as M  # noqa: E402

clock = {"t": 1_000_000.0}          # controllable clock for forgetting/temporal tests
def now(): return clock["t"]


def main() -> int:
    db = Path(tempfile.mkdtemp()) / "test.db"
    core = M.MemoryCore(db, now=now)
    fails = []

    # --- store + semantic recall (keyed facts, as the distiller emits them) ---
    core.store("Alex lives in Montreal.", key="alex::residence", pinned=True)
    core.store("Alex prefers dark roast coffee, no sugar.", key="alex::coffee_pref")
    core.store("The Q3 budget review is scheduled for October 14.", key="q3_review::date")
    core.store("Alex is allergic to shellfish.", key="alex::allergy")

    hits = core.recall("What does he like to drink?", k=2)
    top = hits[0].text if hits else ""
    print(f"[recall] drink -> {top!r}")
    if "coffee" not in top.lower():
        fails.append("semantic recall missed the coffee memory")

    # --- bi-temporal supersession via semantic key -----------------------
    t_before_move = now()
    clock["t"] += 30 * 24 * 3600  # a month later he moves
    core.store("Alex lives in Toronto.", key="alex::residence")  # same key, new value
    cur = [m.text for m in core.recall("where does he live?", k=3)]
    print(f"[supersede] current top -> {cur[0] if cur else None!r}")
    if not any("Toronto" in t for t in cur[:1]):
        fails.append("supersession: new location did not rank first")
    st = core.stats()
    if st["superseded"] < 1:
        fails.append("supersession: old fact was not retired to history")
    if st["current"] != 4:  # 4 facts, one value updated in place (not a 5th current)
        fails.append(f"supersession: expected 4 current, got {st['current']}")

    # --- time-travel: what did we believe before the move? ---------------
    past = core.recall("where does he live?", k=3, as_of=t_before_move)
    print(f"[as_of] belief before move -> {past[0].text if past else None!r}")
    if not any("Montreal" in m.text for m in past):
        fails.append("time-travel: as_of did not return the historical belief")

    # --- context budget ---------------------------------------------------
    budgeted = core.recall("tell me about Alex", k=5, char_budget=60)
    used = sum(len(m.text) for m in budgeted)
    print(f"[budget] {len(budgeted)} memories, {used} chars (<=~60 + 1 overflow ok)")

    # --- forgetting: advance 120 days, low-salience unpinned facts decay ---
    clock["t"] += 120 * 24 * 3600
    archived = core.forget_sweep()
    st = core.stats()
    print(f"[forget] swept {archived}; current={st['current']} superseded={st['superseded']} archived={st['archived']}")
    survivors = [m.text for m in core.recall("who is he", k=10)]
    if not any("Alex" in s and ("Montreal" in s or "Toronto" in s) for s in survivors):
        # pinned identity fact mentions Montreal; must survive
        if not any("Alex" in s for s in survivors):
            fails.append("pinned identity memory was incorrectly forgotten")
    if archived == 0:
        fails.append("forgetting swept nothing after 120 days")

    core.close()
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
