"""Deterministic tests for the fact-dynamics (world-model) layer.

Run: EMBED_PROVIDER=local python scripts/test_dynamics.py
No LLM; local embeddings only; simulated clock.
"""
import os
import sys
import tempfile
from pathlib import Path

# This suite exercises the closed-form Dynamics layer (including its internals);
# the neural model has its own verifier (scripts/verify_neural_mac.py).
os.environ["TENET_DYNAMICS"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tenet.memory import MemoryCore  # noqa: E402

DAY = 86400.0
FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


class Clock:
    def __init__(self, t=1_000_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def main():
    clock = Clock()
    db = Path(tempfile.mkdtemp()) / "dyn.db"
    m = MemoryCore(db, now=clock)

    # Stable key: set once, never changes.
    m.store("my name is Alex", key="user::name")

    # Churny key: changes every ~2 days, 5 times.
    moods = ["happy", "tired", "excited", "stressed", "calm", "focused"]
    m.store(f"mood is {moods[0]}", key="user::mood")
    for mood in moods[1:]:
        clock.t += 2 * DAY
        m.store(f"mood is {mood}", key="user::mood")

    # Correlated keys: residence + job change together, twice.
    m.store("lives in Boston", key="user::residence")
    m.store("works at Acme", key="user::job")
    for city, job in [("Seattle", "Initech"), ("Austin", "Globex")]:
        clock.t += 30 * DAY
        m.store(f"lives in {city}", key="user::residence")
        clock.t += 1 * DAY
        m.store(f"works at {job}", key="user::job")

    dyn = m._dynamics()

    # 1. Learned hazards: churny key doubted young; stable key trusted old.
    p_mood_2d = dyn.p_valid("user::mood", 2 * DAY)
    p_name_100d = dyn.p_valid("user::name", 100 * DAY)
    check("churny key doubted faster than stable key",
          p_mood_2d < p_name_100d, f"mood@2d={p_mood_2d:.2f} name@100d={p_name_100d:.2f}")
    check("stable key stays trusted at 100d", p_name_100d > 0.7, f"{p_name_100d:.2f}")

    # 2. uncertain_facts surfaces the stale-suspect fact.
    clock.t += 7 * DAY  # a week of silence: mood is now way past its usual lifetime
    doubts = m.uncertain_facts(threshold=0.5)
    check("week-old mood flagged as doubtful",
          any(d["key"] == "user::mood" for d in doubts),
          str([(d["key"], d["p_valid"]) for d in doubts]))
    check("name not flagged", all(d["key"] != "user::name" for d in doubts))

    # 3. Ripple: residence changed recently -> correlated job is doubted more.
    dyn2 = m._dynamics()
    clock2_now = clock.t
    m.store("lives in Denver", key="user::residence")  # third move, just now
    m._dyn_dirty = True
    dyn3 = m._dynamics()
    p_job_no_ripple = dyn2.p_valid("user::job", 8 * DAY, now=clock2_now)
    p_job_ripple = dyn3.p_valid("user::job", 8 * DAY, now=clock.t)
    check("residence->job ripple learned",
          "job" in dyn3._ripple.get("residence", {}),
          str(dyn3._ripple))
    check("job doubted more right after a move",
          p_job_ripple < p_job_no_ripple,
          f"ripple={p_job_ripple:.2f} < base={p_job_no_ripple:.2f}")

    # 4. Recall still returns the current fact, now with a confidence.
    got = m.recall("what is the user's mood", k=3)
    cur = [x for x in got if x.key == "user::mood" and x.is_current]
    check("current churny fact still retrievable", len(cur) == 1)
    check("recall attaches learned confidence",
          cur and cur[0].confidence is not None and 0.0 < cur[0].confidence < 1.0,
          f"confidence={cur[0].confidence if cur else None}")

    m.close()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("\nDYNAMICS ALL PASS ✅  (learned hazards, doubts, ripple, confident recall)")


if __name__ == "__main__":
    main()
