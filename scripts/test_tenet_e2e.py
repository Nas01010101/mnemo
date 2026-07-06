"""End-to-end: raw messages -> distill -> bi-temporal store -> recall + supersession.
Run: python scripts/test_tenet_e2e.py
"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet import Tenet  # noqa: E402

clock = {"t": 1_000_000.0}
def now(): return clock["t"]


def main() -> int:
    db = Path(tempfile.mkdtemp()) / "e2e.db"
    m = Tenet(db, now=now)
    fails = []

    # Session 1
    for msg in [
        "Hi, I'm Alex. I live in Montreal and I'm vegetarian. Nice weather today!",
        "My manager is Dr. Adeel Khan.",
    ]:
        m.ingest(msg); clock["t"] += 3600

    # Session 2 (weeks later) — a move + a manager change: should SUPERSEDE, not duplicate
    clock["t"] += 20 * 24 * 3600
    m.ingest("Update: I moved to Toronto. Also my manager is now Sarah Chen.")

    st = m.stats()
    print("stats:", st)

    # currently-true recall
    live = {mm.text for mm in m.recall("where does the user live and who manages them?", k=6)}
    print("current:", live)
    if not any("Toronto" in t for t in live):
        fails.append("current residence should be Toronto")
    if any("Montreal" in t for t in live):
        fails.append("stale Montreal fact should not be in current recall")
    if not any("Sarah Chen" in t for t in live):
        fails.append("current manager should be Sarah Chen")
    if any("Adeel" in t for t in live):
        fails.append("stale manager (Adeel) should not be in current recall")

    if st["superseded"] < 2:
        fails.append(f"expected >=2 superseded (residence+manager), got {st['superseded']}")

    # diet preference survived untouched
    if not any("vegetarian" in t.lower() for t in live):
        # try a targeted query
        diet = [mm.text for mm in m.recall("dietary restriction", k=3)]
        if not any("vegetarian" in t.lower() for t in diet):
            fails.append("durable diet fact was lost")

    m.close()
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nE2E PASS ✅  (distillation-driven bi-temporal supersession works)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
