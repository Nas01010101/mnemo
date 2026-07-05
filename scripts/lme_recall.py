"""Retrieval-recall benchmark on the full LongMemEval_S haystack (the primary,
budget-cheap metric — no answerer/judge LLM calls).

Measures session-level recall@k: after ingesting a ~115k-token / ~50-session history,
does the memory surface a memory that came from an EVIDENCE session (answer_session_ids)?

Compares:
  • rag    — embed all turns, top-k cosine                         [baseline]
  • mnemo  — hybrid distilled-facts + raw-slices, dual-pool recall  [ours]

Shared, batched embeddings + parallel distillation keep it tractable.

Usage: python scripts/lme_recall.py --limit 30 --k 10 --seed 0
"""
import argparse, json, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
import config       # noqa: E402
from distill import distill  # noqa: E402
from mnemo import Mnemo       # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "lme" / "longmemeval_s.json"


def flatten(inst):
    """-> list of (session_id, 'Role: content') oldest-first, + question vec later."""
    rows = []
    for sid, sess in zip(inst["haystack_session_ids"], inst["haystack_sessions"]):
        for t in sess:
            if t["content"].strip():
                rows.append((sid, f"{t['role'].capitalize()}: {t['content'].strip()}"))
    return rows


def recall_hit(sources, evidence):
    return any(s in evidence for s in sources)


def eval_instance(inst, k, embedder):
    evidence = set(inst["answer_session_ids"])
    turns = flatten(inst)
    texts = [t for _, t in turns]
    sids = [s for s, _ in turns]

    # one shared embedding pass over all turns + the question
    all_vecs = embedder(texts + [inst["question"]])
    turn_vecs, qv = np.array(all_vecs[:-1]), all_vecs[-1]

    # --- naive RAG: top-k cosine over raw turns ---
    t0 = time.time()
    sims = turn_vecs @ qv
    top = np.argsort(-sims)[:k]
    rag_lat = time.time() - t0
    rag_ok = recall_hit([sids[i] for i in top], evidence)

    # --- mnemo: hybrid ingest (facts + raw), dual-pool recall ---
    db = Path(tempfile.mkdtemp()) / "r.db"
    m = Mnemo(db)
    # parallel distill per session
    sess_pairs = list(zip(inst["haystack_session_ids"], inst["haystack_sessions"]))
    def _distill(pair):
        sid, sess = pair
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in sess)
        try:
            return sid, distill(convo)
        except Exception:
            return sid, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        distilled = list(ex.map(_distill, sess_pairs))
    # batch-embed all fact statements
    facts = [(sid, f) for sid, fs in distilled for f in fs]
    if facts:
        fvecs = m.core.embed_batch([f.statement for _, f in facts])
        for (sid, f), fv in zip(facts, fvecs):
            m.core.store(f.statement, key=f.key, salience=f.salience,
                         source=sid, _vec=fv)
    # store raw slices with the embeddings we already computed
    for (sid, text), tv in zip(turns, turn_vecs):
        m.core.store(text, kind="raw", salience=0.35, source=sid, _vec=tv)

    t0 = time.time()
    hits = m.core.recall(inst["question"], k=k)
    mnemo_lat = time.time() - t0
    mnemo_ok = recall_hit([h.source for h in hits], evidence)
    m.close()
    return inst["question_type"], rag_ok, mnemo_ok, rag_lat, mnemo_lat, len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    data = [d for d in json.load(open(DATA)) if not d["question_id"].endswith("_abs")]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    core = Mnemo(Path(tempfile.mkdtemp()) / "emb.db")  # embedder host
    embedder = core.core.embed_batch

    agg = {"rag": [0, []], "mnemo": [0, []]}
    by_type = {}
    t_start = time.time()
    for i, inst in enumerate(data):
        qt, rok, mok, rl, ml, nt = eval_instance(inst, args.k, embedder)
        agg["rag"][0] += rok; agg["rag"][1].append(rl)
        agg["mnemo"][0] += mok; agg["mnemo"][1].append(ml)
        bt = by_type.setdefault(qt, [0, 0, 0]); bt[0] += rok; bt[1] += mok; bt[2] += 1
        print(f"[{i+1}/{len(data)}] {qt[:18]:18s} turns={nt:4d} | rag:{'✓' if rok else '✗'} mnemo:{'✓' if mok else '✗'}")

    n = len(data)
    print(f"\n=== session-level recall@{args.k} on LongMemEval_S (n={n}) ===")
    for name in ("rag", "mnemo"):
        c, lats = agg[name]
        med = sorted(lats)[len(lats) // 2] * 1000 if lats else 0
        print(f"{name:6s}  recall@{args.k}={100*c/n:5.1f}%  ({c}/{n})  retrieval_med={med:5.0f}ms")
    print("\nby question type (rag / mnemo):")
    for qt, (r, mm, tot) in sorted(by_type.items()):
        print(f"  {qt:24s} {100*r/tot:5.1f}% / {100*mm/tot:5.1f}%  (n={tot})")
    print(f"\nwall={time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
