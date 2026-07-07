"""Honest recall benchmark for Tenet.

Plants known facts across a long, noisy multi-session history, then asks questions
and measures:
  • recall@k        — fraction of questions whose gold fact is in the top-k retrieved
  • context saved % — how much smaller the retrieved context is vs dumping full history
  • supersession    — does an updated fact override the stale one?

Baseline = "recency" (last-k memories, no semantic search) to show the embedding +
decay ranking actually earns its keep.

Run: python scripts/eval_recall.py
"""
import sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet import memory as M  # noqa: E402

clock = {"t": 1_000_000.0}
def now(): return clock["t"]

# (fact, question, gold-substring)
FACTS = [
    ("Alex's flight to Tokyo departs at 14:20 on March 3.", "When does my flight leave?", "14:20"),
    ("The office wifi password is blufin-2026.", "What's the wifi password?", "blufin-2026"),
    ("Alex's manager is Dr. Adeel Khan.", "Who is my manager?", "Adeel"),
    ("The prod database is hosted in the ap-southeast-1 region.", "Where is the prod DB?", "ap-southeast-1"),
    ("Alex is vegetarian and dislikes cilantro.", "What are my dietary restrictions?", "vegetarian"),
    ("The API rate limit for the free tier is 60 requests per minute.", "What's the free-tier rate limit?", "60"),
    ("Project Nimbus ships on the last Friday of Q3.", "When does Project Nimbus ship?", "Q3"),
    ("Alex's preferred IDE is Neovim with the tokyonight theme.", "Which editor do I use?", "Neovim"),
]

# distractor noise interleaved between the real facts (simulates a long history)
NOISE = [f"Random log line {i}: routine heartbeat ok, latency {20+i}ms." for i in range(40)]


def build(core):
    """Interleave facts + noise across simulated sessions spread over days."""
    seq = []
    fi = 0
    for i, n in enumerate(NOISE):
        seq.append(("noise", n))
        if i % 5 == 2 and fi < len(FACTS):
            seq.append(("fact", FACTS[fi][0])); fi += 1
    while fi < len(FACTS):
        seq.append(("fact", FACTS[fi][0])); fi += 1
    for kind, text in seq:
        core.store(text)
        clock["t"] += 3600  # an hour between events


def recency_baseline(core, k):
    """Naive: return the k most recently stored live memories (no semantic search)."""
    core.db.row_factory = __import__("sqlite3").Row
    rows = core.db.execute(
        "SELECT text FROM memories WHERE archived=0 ORDER BY created_at DESC LIMIT ?", (k,)
    ).fetchall()
    return [r["text"] for r in rows]


def main() -> int:
    db = Path(tempfile.mkdtemp()) / "eval.db"
    core = M.MemoryCore(db, now=now)
    build(core)

    total_hist_chars = sum(len(t) for _, t in
                           [("x", NOISE[i]) for i in range(len(NOISE))] + [("x", f[0]) for f in FACTS])

    k = 3
    sem_hits = base_hits = 0
    retrieved_chars = 0
    for _fact, q, gold in FACTS:
        sem = core.recall(q, k=k)
        sem_texts = [m.text for m in sem]
        retrieved_chars += sum(len(t) for t in sem_texts)
        if any(gold.lower() in t.lower() for t in sem_texts):
            sem_hits += 1
        if any(gold.lower() in t.lower() for t in recency_baseline(core, k)):
            base_hits += 1

    n = len(FACTS)
    avg_retrieved = retrieved_chars / n
    saved = 100 * (1 - avg_retrieved / total_hist_chars)

    print(f"History: {len(NOISE)} noise + {n} facts = {total_hist_chars} chars total")
    print(f"recall@{k}  Tenet (semantic+decay): {sem_hits}/{n} = {100*sem_hits/n:.0f}%")
    print(f"recall@{k}  recency baseline:        {base_hits}/{n} = {100*base_hits/n:.0f}%")
    print(f"context saved vs full history: {saved:.1f}%  (avg {avg_retrieved:.0f} vs {total_hist_chars} chars)")

    # supersession: update a fact, confirm the new value wins
    core.store("UPDATE: Alex's flight to Tokyo now departs at 09:45 on March 3.", pinned=False)
    top = core.recall("When does my flight leave?", k=2)
    new_wins = any("09:45" in m.text for m in top[:1])
    print(f"supersession (updated fact ranks first): {'PASS' if new_wins else 'FAIL'}")

    core.close()
    ok = sem_hits >= base_hits and sem_hits >= n - 1
    print("\n" + ("RESULT: Tenet >= baseline ✅" if ok else "RESULT: needs tuning ⚠️"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
