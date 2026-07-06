"""Controlled knowledge-update benchmark — the capability where bi-temporal
supersession structurally beats naive RAG.

Setup: one user whose facts CHANGE over time (residence, job, car, phone, gym),
each updated across sessions, interleaved with distractor sessions (noise). Then we
ask for the CURRENT value of each fact.

  • naive-RAG retrieves the top-k most similar turns — which include BOTH the stale
    and the current statements, so the reader sees conflicting values.
  • Tenet supersedes: the old value is retired (expired_at set), so current recall
    returns ONLY the latest value.

Metrics (per updated fact):
  • current-correct : answer contains the latest value
  • stale-leak      : answer contains an OUTDATED value (the failure RAG is prone to)

Run: python scripts/bench_knowledge_update.py --principals 4
"""
import argparse, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
import config       # noqa: E402
from tenet import Tenet  # noqa: E402

_usage = {"in": 0, "out": 0}

# attribute -> (question, [chronological values, oldest..newest])
ATTRS = {
    "residence": ("Which city does the user currently live in?",
                  ["Boston", "Denver", "Seattle"]),
    "job_title": ("What is the user's current job title?",
                  ["junior analyst", "senior analyst", "team lead"]),
    "car": ("What car does the user currently drive?",
            ["Honda Civic", "Toyota Camry", "Tesla Model 3"]),
    "phone": ("What phone does the user currently use?",
              ["iPhone 12", "iPhone 14", "Pixel 8"]),
    "gym": ("Which gym does the user currently go to?",
            ["Planet Fitness", "Equinox", "CrossFit Central"]),
}
UPDATE_TMPL = {
    "residence": "I just moved to {v}.",
    "job_title": "I got promoted — I'm now a {v}.",
    "car": "I bought a new car, a {v}.",
    "phone": "I switched phones, now using a {v}.",
    "gym": "I changed gyms, I go to {v} now.",
}
DISTRACTORS = [
    "The weather has been really nice this week.",
    "I watched a great documentary last night.",
    "I'm thinking about learning to cook Thai food.",
    "Traffic was terrible this morning.",
    "I read an interesting article about space travel.",
    "My neighbor got a new dog, it's adorable.",
    "I've been trying to drink more water lately.",
    "The coffee shop downtown changed its hours.",
]


def build_history(seed_offset):
    """Interleave update statements (oldest->newest per attr) with distractors.
    Returns (sessions, gold) where gold[attr] = (latest, [stale...])."""
    rng = np.random.RandomState(seed_offset)
    events = []  # (order, "User: text")
    gold = {}
    # rotate value order per principal so it's not always the same latest value
    for attr, (_q, vals) in ATTRS.items():
        rot = seed_offset % len(vals)
        vv = vals[rot:] + vals[:rot]
        gold[attr] = (vv[-1], vv[:-1])
        for i, v in enumerate(vv):
            events.append((i, f"User: {UPDATE_TMPL[attr].format(v=v)}"))
    # spread distractors across the timeline, with some verbatim repeats (users
    # repeat themselves) so surprise-gating has redundant observations to drop
    dpool = list(DISTRACTORS) + list(DISTRACTORS[:4])  # 4 exact repeats
    rng.shuffle(dpool)
    for j, d in enumerate(dpool):
        events.append((0.5 + j * 0.3, f"User: {d}"))
    # stable sort by order (update steps 0,1,2 keep chronological supersession)
    events.sort(key=lambda x: x[0])
    sessions = [[{"role": "User", "content": t.split(": ", 1)[1]}] for _o, t in events]
    return sessions, gold


def answer(context, question):
    return config.chat(
        [{"role": "system", "content":
          "Answer using ONLY the memory provided. Give the user's CURRENT value. "
          "Reply with just the value, nothing else."},
         {"role": "user", "content": f"Memory:\n{context}\n\nQuestion: {question}"}],
        qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"), max_tokens=32)


def score(ans, latest, stale):
    a = ans.lower()
    return latest.lower() in a, any(s.lower() in a for s in stale)


def run_principal(seed_offset, k):
    sessions, gold = build_history(seed_offset)
    turns = [t["content"] for s in sessions for t in s]

    # shared embeddings
    m = Tenet(Path(tempfile.mkdtemp()) / "ku.db")
    vecs = m.core.embed_batch([f"User: {t}" for t in turns])

    # naive-RAG store (just the raw turns + vectors)
    rag_texts = [f"User: {t}" for t in turns]
    rag_mat = np.array(vecs)

    # Tenet: ingest each session (distill+supersede) in chronological order
    clock = [1_000_000.0]
    m2 = Tenet(Path(tempfile.mkdtemp()) / "ku2.db", now=lambda: clock[0])
    raw_kept = 0
    for s in sessions:
        raw_kept += len(m2.ingest_session(s, valid_at=clock[0])["raw"]); clock[0] += 3600

    rows = []
    for attr, (q, _vals) in ATTRS.items():
        latest, stale = gold[attr]
        qv = m.core.embed_batch([q])[0]
        # RAG top-k
        sims = rag_mat @ qv
        top = np.argsort(-sims)[:k]
        rag_ctx = "\n".join(rag_texts[i] for i in sorted(top))
        rag_ans = answer(rag_ctx, q)
        # Tenet current recall
        mem_ctx = "\n".join(f"- {h.text}" for h in m2.core.recall(q, k=k))
        mem_ans = answer(mem_ctx, q)
        r_ok, r_leak = score(rag_ans, latest, stale)
        m_ok, m_leak = score(mem_ans, latest, stale)
        rows.append((attr, latest, rag_ans, mem_ans, r_ok, r_leak, m_ok, m_leak,
                     len(rag_ctx), len(mem_ctx)))
    stored = {"turns": len(turns), "raw_stored": raw_kept,
              "full_hist_chars": sum(len(t) for t in rag_texts)}
    m.close(); m2.close()
    return rows, stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principals", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    all_rows = []
    stores = []
    t0 = time.time()
    for p in range(args.principals):
        rows, stored = run_principal(p, args.k)
        stores.append(stored)
        for r in rows:
            all_rows.append(r)
            attr, latest, ra, ma, rok, rleak, mok, mleak, rcl, mcl = r
            print(f"[p{p}] {attr:10s} latest={latest:14s} | "
                  f"RAG={'✓' if rok else '✗'}{'⚠leak' if rleak else '':5s} '{ra[:22]}' | "
                  f"TENET={'✓' if mok else '✗'}{'⚠leak' if mleak else '':5s} '{ma[:22]}'")

    n = len(all_rows)
    def rate(idx): return 100 * sum(r[idx] for r in all_rows) / n
    avg = lambda idx: sum(r[idx] for r in all_rows) / n
    print(f"\n=== knowledge-update: current-value accuracy (n={n}, k={args.k}) ===")
    print(f"RAG    current-correct={rate(4):5.1f}%   stale-leak={rate(5):5.1f}%")
    print(f"TENET  current-correct={rate(6):5.1f}%   stale-leak={rate(7):5.1f}%")
    # efficiency (the world-model win): compact belief-state recall vs raw dump
    full_chars = sum(s["full_hist_chars"] for s in stores) / len(stores)
    print(f"\n=== efficiency ===")
    print(f"context chars fed to reader:  full-history≈{full_chars:.0f}  "
          f"RAG≈{avg(8):.0f}  TENET≈{avg(9):.0f}")
    print(f"  → TENET reads {100*(1-avg(9)/avg(8)):.0f}% less context than RAG, "
          f"{100*(1-avg(9)/full_chars):.0f}% less than full history")
    turns = sum(s["turns"] for s in stores); stored = sum(s["raw_stored"] for s in stores)
    print(f"surprise-gated storage: {stored} memories kept from {turns} turns "
          f"({100*(1-stored/turns):.0f}% of raw observations dropped as redundant)")
    print(f"\ntokens in/out: {_usage['in']:,}/{_usage['out']:,}  wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
