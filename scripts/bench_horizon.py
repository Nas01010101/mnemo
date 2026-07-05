"""Long-horizon knowledge-update stress test — where a memory structurally beats RAG.

As a fact is updated MORE times over a long history, naive-RAG's top-k fills with stale
versions of that fact and the reader must pick the latest from a noisy pile; accuracy
degrades. Mnemo supersedes each update, so current recall returns exactly ONE value
regardless of how many times it changed. We sweep the number of updates and plot both.

Run: python scripts/bench_horizon.py --principals 6 --k 6
"""
import argparse, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
import config       # noqa: E402
from mnemo import Mnemo  # noqa: E402

CITIES = ["Boston", "Denver", "Seattle", "Austin", "Chicago", "Miami",
          "Portland", "Atlanta", "Dallas", "Phoenix", "Nashville", "Raleigh"]
DISTRACTORS = [
    "The weather has been really nice this week.", "I watched a great documentary.",
    "Traffic was terrible this morning.", "I read an article about space travel.",
    "My neighbor got a new dog.", "I've been trying to drink more water.",
    "The coffee shop downtown changed its hours.", "I started a new book last night.",
    "Work has been busy lately.", "I need to renew my passport soon.",
    "I tried a new recipe yesterday.", "The gym was crowded today.",
]
Q = "Which city does the user currently live in?"


def answer(context):
    return config.chat(
        [{"role": "system", "content": "Answer using ONLY the memory provided. Give the "
          "user's CURRENT city. Reply with just the city name."},
         {"role": "user", "content": f"Memory:\n{context}\n\nQuestion: {Q}"}],
        qwen_default="qwen3.7-plus", max_tokens=16)


def one(principal, n_updates, k, n_distractors):
    rng = np.random.RandomState(principal * 97 + n_updates)
    cities = list(CITIES); rng.shuffle(cities)
    chain = cities[:n_updates]           # chronological moves
    latest = chain[-1]
    # build interleaved history
    events = [(i, f"User: I just moved to {chain[i]}.") for i in range(n_updates)]
    dd = list(DISTRACTORS); rng.shuffle(dd)
    for j, d in enumerate(dd[:n_distractors]):
        events.append((0.5 + j * 0.3, f"User: {d}"))
    events.sort(key=lambda x: x[0])
    turns = [t for _o, t in events]

    # embed once, shared
    m = Mnemo(Path(tempfile.mkdtemp()) / "h.db")
    vecs = m.core.embed_batch(turns)
    mat = np.array(vecs)
    qv = m.core.embed_batch([Q])[0]

    # RAG: top-k raw turns in chronological order
    top = np.argsort(-(mat @ qv))[:k]
    rag_ctx = "\n".join(turns[i] for i in sorted(top))
    rag_ok = latest.lower() in answer(rag_ctx).lower()

    # Mnemo: ingest chronologically (supersession), recall current
    clock = [1e6]
    m2 = Mnemo(Path(tempfile.mkdtemp()) / "h2.db", now=lambda: clock[0])
    for t in turns:
        role, content = t.split(": ", 1)
        m2.ingest_session([{"role": role, "content": content}], valid_at=clock[0])
        clock[0] += 3600
    mem_ctx = "\n".join(f"- {h.text}" for h in m2.core.recall(Q, k=k))
    mem_ok = latest.lower() in answer(mem_ctx).lower()
    m.close(); m2.close()
    return rag_ok, mem_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principals", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--distractors", type=int, default=12)
    ap.add_argument("--updates", default="2,4,6,8")
    args = ap.parse_args()
    sweep = [int(x) for x in args.updates.split(",")]

    t0 = time.time()
    print(f"long-horizon knowledge-update (k={args.k}, distractors={args.distractors}, "
          f"{args.principals} principals/point)\n")
    print(f"{'#updates':>9} | {'RAG acc':>8} | {'MNEMO acc':>9}")
    print("-" * 32)
    results = []
    for nu in sweep:
        rag = mem = 0
        for p in range(args.principals):
            r, m = one(p, nu, args.k, args.distractors)
            rag += r; mem += m
        n = args.principals
        results.append((nu, 100 * rag / n, 100 * mem / n))
        print(f"{nu:>9} | {100*rag/n:>7.0f}% | {100*mem/n:>8.0f}%")
    print(f"\nwall={time.time()-t0:.0f}s")
    # headline
    deg_rag = results[0][1] - results[-1][1]
    deg_mem = results[0][2] - results[-1][2]
    print(f"as updates {sweep[0]}→{sweep[-1]}: RAG {deg_rag:+.0f}pp, MNEMO {deg_mem:+.0f}pp")


if __name__ == "__main__":
    main()
