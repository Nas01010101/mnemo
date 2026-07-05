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


ANSWER_MODEL = config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus")
_qa_usage = {"in": 0, "out": 0}


def _qa_chat(system, user, max_tokens=256):
    return config.chat([{"role": "system", "content": system},
                        {"role": "user", "content": user}],
                       qwen_default=ANSWER_MODEL, max_tokens=max_tokens)


_ANS_SYS = ("Answer the question using ONLY the provided memory about the user. Reason over "
            "the notes, then give a short direct answer. If absent, say 'I don't know'. "
            "End with 'ANSWER: <answer>'.")
_JUDGE_SYS = ("Grade whether the model answer matches the gold answer for the question. Be "
              "lenient about phrasing; judge semantic correctness. Reply exactly 'yes' or 'no'.")


def qa_answer(context, question, qdate):
    out = _qa_chat(_ANS_SYS, f"Question date: {qdate}\n\nMemory:\n{context}\n\nQuestion: {question}")
    return out.split("ANSWER:")[-1].strip() if "ANSWER:" in out else out


def qa_judge(question, gold, pred):
    return _qa_chat(_JUDGE_SYS,
                    f"Question: {question}\nGold: {gold}\nModel answer: {pred}\nCorrect?",
                    max_tokens=4).lower().startswith("y")


def eval_instance(inst, k, embedder, qa=False):
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
    rag_ctx = "\n".join(texts[i] for i in sorted(top))

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
    mnemo_ctx = "\n".join(f"- {h.text}" for h in hits)
    m.close()

    r = {"type": inst["question_type"], "rag_recall": rag_ok, "mnemo_recall": mnemo_ok,
         "rag_lat": rag_lat, "mnemo_lat": mnemo_lat, "turns": len(texts),
         "full_ctx_chars": sum(len(t) for t in texts),
         "rag_ctx_chars": len(rag_ctx), "mnemo_ctx_chars": len(mnemo_ctx)}
    if qa:
        rp = qa_answer(rag_ctx, inst["question"], inst["question_date"])
        mp = qa_answer(mnemo_ctx, inst["question"], inst["question_date"])
        r["rag_qa"] = qa_judge(inst["question"], inst["answer"], rp)
        r["mnemo_qa"] = qa_judge(inst["question"], inst["answer"], mp)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qa", action="store_true", help="also run answer-accuracy + context efficiency")
    ap.add_argument("--type", default="", help="filter to one question_type (e.g. knowledge-update)")
    args = ap.parse_args()

    import random
    data = [d for d in json.load(open(DATA)) if not d["question_id"].endswith("_abs")]
    if args.type:
        data = [d for d in data if d["question_type"] == args.type]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    core = Mnemo(Path(tempfile.mkdtemp()) / "emb.db")  # embedder host
    embedder = core.core.embed_batch

    rows = []
    t_start = time.time()
    for i, inst in enumerate(data):
        r = eval_instance(inst, args.k, embedder, qa=args.qa)
        rows.append(r)
        tail = ""
        if args.qa:
            tail = f" | QA rag:{'✓' if r['rag_qa'] else '✗'} mnemo:{'✓' if r['mnemo_qa'] else '✗'}"
        print(f"[{i+1}/{len(data)}] {r['type'][:18]:18s} turns={r['turns']:4d} | "
              f"recall rag:{'✓' if r['rag_recall'] else '✗'} mnemo:{'✓' if r['mnemo_recall'] else '✗'}{tail}")

    n = len(rows)
    def pct(key): return 100 * sum(r[key] for r in rows) / n
    print(f"\n=== LongMemEval_S (n={n}{', type='+args.type if args.type else ''}) ===")
    print(f"session-level recall@{args.k}:  rag={pct('rag_recall'):.1f}%  mnemo={pct('mnemo_recall'):.1f}%")
    if args.qa:
        print(f"answer accuracy (QA):       rag={pct('rag_qa'):.1f}%  mnemo={pct('mnemo_qa'):.1f}%")
    avg = lambda key: sum(r[key] for r in rows) / n
    print(f"context chars fed to reader: full≈{avg('full_ctx_chars'):.0f}  "
          f"rag≈{avg('rag_ctx_chars'):.0f}  mnemo≈{avg('mnemo_ctx_chars'):.0f}")
    print(f"  → mnemo uses {100*(1-avg('mnemo_ctx_chars')/avg('full_ctx_chars')):.1f}% less context than full history")
    # per-type
    print("\nby type (recall rag/mnemo" + (" · QA rag/mnemo" if args.qa else "") + "):")
    types = {}
    for r in rows:
        types.setdefault(r["type"], []).append(r)
    for qt, rs in sorted(types.items()):
        t = len(rs)
        line = f"  {qt:24s} {100*sum(x['rag_recall'] for x in rs)/t:5.1f}%/{100*sum(x['mnemo_recall'] for x in rs)/t:5.1f}%"
        if args.qa:
            line += f"  ·  {100*sum(x['rag_qa'] for x in rs)/t:5.1f}%/{100*sum(x['mnemo_qa'] for x in rs)/t:5.1f}%"
        print(line + f"  (n={t})")
    if args.qa:
        print(f"\nQA tokens: in={_qa_usage['in']:,} out={_qa_usage['out']:,}")
    print(f"wall={time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
