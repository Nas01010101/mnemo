"""Exercise MemoryCore against the live Qwen embedding API.
Run: python scripts/test_memory.py   (uses a throwaway temp DB)
"""
import sys, tempfile, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import memory as M  # noqa: E402

clock = {"t": 1_000_000.0}          # controllable clock for forgetting tests
def now(): return clock["t"]


def main() -> int:
    db = Path(tempfile.mkdtemp()) / "test.db"
    core = M.MemoryCore(db, now=now)
    fails = []

    # --- store + semantic recall -----------------------------------------
    core.store("User's name is Wissem and he lives in Montreal.", pinned=True)
    core.store("Wissem prefers dark roast coffee, no sugar.")
    core.store("The Q3 budget review is scheduled for October 14.")
    core.store("Wissem is allergic to shellfish.")

    hits = core.recall("What does he like to drink?", k=2)
    top = hits[0].text if hits else ""
    print(f"[recall] drink -> {top!r}")
    if "coffee" not in top.lower():
        fails.append("semantic recall missed the coffee memory")

    # --- dedup / consolidation -------------------------------------------
    before = core.stats()["live"]
    core.store("Wissem likes dark roast coffee with no sugar.")  # near-dup
    after = core.stats()["live"]
    print(f"[dedup] live {before} -> {after} (expect unchanged)")
    if after != before:
        fails.append("dedup failed to consolidate a near-duplicate")

    # --- context budget ---------------------------------------------------
    budgeted = core.recall("tell me about Wissem", k=5, char_budget=60)
    used = sum(len(m.text) for m in budgeted)
    print(f"[budget] {len(budgeted)} memories, {used} chars (<=~60 + 1 overflow ok)")

    # --- forgetting: advance 90 days, unpinned low-use memories decay -----
    clock["t"] += 90 * 24 * 3600
    archived = core.forget_sweep()
    st = core.stats()
    print(f"[forget] swept {archived}; live={st['live']} archived={st['archived']}")
    # pinned identity memory must survive
    survivors = [m.text for m in core.recall("who is he", k=10)]
    if not any("Montreal" in s for s in survivors):
        fails.append("pinned memory was incorrectly forgotten")
    if archived == 0:
        fails.append("forgetting swept nothing after 90 days")

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
