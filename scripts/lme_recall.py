"""Retrieval-recall benchmark on the full LongMemEval_S haystack (the primary,
budget-cheap metric — no answerer/judge LLM calls).

Measures session-level recall@k: after ingesting a ~115k-token / ~50-session history,
does the memory surface a memory that came from an EVIDENCE session (answer_session_ids)?

Compares:
  • rag    — embed all turns, top-k cosine                         [baseline]
  • tenet  — hybrid distilled-facts + raw-slices, dual-pool recall  [ours]

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
from tenet import Tenet       # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "lme" / "longmemeval_s.json"


def flatten(inst):
    """-> list of (session_id, '[date] Role: content') oldest-first.

    Session dates are load-bearing: temporal-reasoning is 27% of LongMemEval and
    unanswerable without them (LME paper: time-aware indexing = +11.3% recall).
    Both arms (RAG and Tenet) get the same dated turns — fairness preserved.
    """
    rows = []
    dates = inst.get("haystack_dates") or [""] * len(inst["haystack_session_ids"])
    for sid, date, sess in zip(inst["haystack_session_ids"], dates, inst["haystack_sessions"]):
        tag = f"[{date}] " if date else ""
        for t in sess:
            if t["content"].strip():
                rows.append((sid, f"{tag}{t['role'].capitalize()}: {t['content'].strip()}"))
    return rows


# Per-type reader budgets (agentmemory-style: multi-hop question types need more
# evidence). (k, expand, hops, char_budget). RAG stays the fixed k=10 baseline;
# Tenet's context size is REPORTED, so bigger budgets are visible, not hidden.
_TYPE_BUDGETS = {
    "multi-session":      (14, 60, 3, 22000),
    "temporal-reasoning":  (12, 40, 0, 16000),
    "knowledge-update":    (10, 24, 0, 10000),
}
_DEFAULT_BUDGET = (10, 20, 0, 9000)


def recall_hit(sources, evidence):
    return any(s in evidence for s in sources)


ANSWER_MODEL = config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus")
READER_MODEL = config.get("READER_MODEL", "")  # OpenRouter reader override (strong model)
_qa_usage = {"in": 0, "out": 0}


def _qa_chat(system, user, max_tokens=256):
    return config.chat([{"role": "system", "content": system},
                        {"role": "user", "content": user}],
                       qwen_default=ANSWER_MODEL, max_tokens=max_tokens,
                       or_model=READER_MODEL or None)


# Chain-of-Note reading (Yu 2023; LME paper: up to +10pp even under oracle retrieval):
# extract per-item notes first, then reason over the notes, then answer.
_ANS_SYS = ("Answer the question using ONLY the provided memory about the user.\n"
            "Step 1 — NOTES: for each memory item relevant to the question, write one short "
            "note quoting the key detail (keep dates!).\n"
            "Step 2 — REASON: combine the notes; resolve times relative to the question date; "
            "if a fact changed over time, the LATEST dated value is current.\n"
            "Step 3 — answer briefly. If the memory lacks the answer, say 'I don't know'.\n"
            "End with 'ANSWER: <answer>'.")
_JUDGE_SYS = ("Grade whether the model answer matches the gold answer for the question. Be "
              "lenient about phrasing; judge semantic correctness. Reply exactly 'yes' or 'no'.")


def qa_answer(context, question, qdate):
    out = _qa_chat(_ANS_SYS, f"Question date: {qdate}\n\nMemory:\n{context}\n\nQuestion: {question}",
                   max_tokens=800)  # CoN needs room for notes + reasoning
    return out.split("ANSWER:")[-1].strip() if "ANSWER:" in out else out


def qa_judge(question, gold, pred):
    out = _qa_chat(_JUDGE_SYS,
                   f"Question: {question}\nGold: {gold}\nModel answer: {pred}\nCorrect?",
                   max_tokens=4).lower()
    return out.startswith("y") if out else None  # None = judge API failure


def eval_instance(inst, k, embedder, qa=False, do_full=True, expand=0, hops=0,
                  cache_dir=None, type_budgets=False):
    evidence = set(inst["answer_session_ids"])
    qid = inst["question_id"]
    turns = flatten(inst)
    texts = [t for _, t in turns]
    sids = [s for s, _ in turns]

    # ingestion cache: turn embeddings + the fully-ingested tenet store are identical
    # across runs — persisting them turns a ~$0.5 iteration into a ~$0.05 one.
    npz = cache_dir / f"{qid}.npz" if cache_dir else None
    dbp = cache_dir / f"{qid}.db" if cache_dir else None
    if npz is not None and npz.exists():
        turn_vecs = np.load(npz)["v"]
        qv = np.asarray(embedder([inst["question"]])[0])
    else:
        all_vecs = embedder(texts + [inst["question"]])
        turn_vecs, qv = np.array(all_vecs[:-1]), all_vecs[-1]
        if npz is not None:
            np.savez_compressed(npz, v=turn_vecs)

    # --- naive RAG: top-k cosine over raw turns ---
    t0 = time.time()
    sims = turn_vecs @ qv
    top = np.argsort(-sims)[:k]
    rag_lat = time.time() - t0
    rag_ok = recall_hit([sids[i] for i in top], evidence)
    rag_ctx = "\n".join(texts[i] for i in sorted(top))

    # --- tenet: hybrid ingest (facts + raw), dual-pool recall ---
    if dbp is not None and dbp.exists():
        m = Tenet(dbp)                       # cached, fully-ingested store
    else:
        m = Tenet(dbp or Path(tempfile.mkdtemp()) / "r.db")
        # parallel distill per session (session date prepended so the distiller
        # grounds event times — dates are load-bearing for temporal questions)
        dates = inst.get("haystack_dates") or [""] * len(inst["haystack_session_ids"])
        sess_pairs = list(zip(inst["haystack_session_ids"], dates, inst["haystack_sessions"]))
        def _distill(pair):
            sid, date, sess = pair
            convo = "\n".join(f"{t['role']}: {t['content']}" for t in sess)
            if date:
                convo = f"Session date: {date}\n{convo}"
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
    if type_budgets:
        # max-accuracy operating point: per-question-type budgets (context size is
        # measured and reported — bigger budgets are visible, never hidden)
        tk, texp, thops, tbudget = _TYPE_BUDGETS.get(inst["question_type"], _DEFAULT_BUDGET)
    else:
        # efficiency/parity point: cap Tenet's context at RAG's own size so the
        # comparison is at EQUAL-OR-LOWER tokens
        tk, texp, thops, tbudget = k, expand, hops, (len(rag_ctx) if expand else None)
    hits = m.core.recall(inst["question"], k=tk, expand=texp, hops=thops, char_budget=tbudget)
    tenet_lat = time.time() - t0
    tenet_ok = recall_hit([h.source for h in hits], evidence)
    tenet_ctx = "\n".join(f"- {h.text}" for h in hits)
    m.close()

    r = {"type": inst["question_type"], "rag_recall": rag_ok, "tenet_recall": tenet_ok,
         "rag_lat": rag_lat, "tenet_lat": tenet_lat, "turns": len(texts),
         "full_ctx_chars": sum(len(t) for t in texts),
         "rag_ctx_chars": len(rag_ctx), "tenet_ctx_chars": len(tenet_ctx)}
    if qa:
        rp = qa_answer(rag_ctx, inst["question"], inst["question_date"])
        mp = qa_answer(tenet_ctx, inst["question"], inst["question_date"])
        rj = qa_judge(inst["question"], inst["answer"], rp) if rp.strip() else None
        mj = qa_judge(inst["question"], inst["answer"], mp) if mp.strip() else None
        if rj is None or mj is None:
            # An API failure (empty answer or judge) must not count as a wrong
            # answer for either arm — exclude the instance from QA scoring.
            r["qa_error"] = True
        else:
            r["rag_qa"], r["tenet_qa"] = rj, mj
            r["rag_pred"], r["tenet_pred"] = rp, mp
            r["qid"], r["question"], r["gold"] = qid, inst["question"], inst["answer"]
            r["tenet_ctx"] = tenet_ctx
        if do_full and not r.get("qa_error"):  # full-context ceiling (expensive)
            full_ctx = "\n".join(f"[{d}] {t}" for d, t in turns)
            fp = qa_answer(full_ctx, inst["question"], inst["question_date"])
            fj = qa_judge(inst["question"], inst["answer"], fp) if fp.strip() else None
            if fj is None:
                r["qa_error"] = True
                r.pop("rag_qa", None); r.pop("tenet_qa", None)
            else:
                r["full_qa"] = fj
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qa", action="store_true", help="also run answer-accuracy + context efficiency")
    ap.add_argument("--type", default="", help="filter to one question_type (e.g. knowledge-update)")
    ap.add_argument("--no-full", action="store_true", help="skip the costly full-context ceiling")
    ap.add_argument("--expand", type=int, default=0,
                    help="belief-anchored evidence expansion: extra query-relevant raw slices "
                         "from surfaced sessions (0=off)")
    ap.add_argument("--hops", type=int, default=0,
                    help="recursive associative recall: select the expand slots over this many "
                         "replay-conditioned rounds (ReContext-style; 0/1=single-shot anchored)")
    ap.add_argument("--type-budgets", action="store_true",
                    help="max-accuracy mode: per-question-type k/expand/hops/context budgets")
    ap.add_argument("--cache", default="",
                    help="ingestion-cache dir (embeddings + distilled store persist across runs)")
    ap.add_argument("--dump", default="", help="write per-instance misses to this JSONL")
    args = ap.parse_args()

    import random
    data = [d for d in json.load(open(DATA)) if not d["question_id"].endswith("_abs")]
    if args.type:
        data = [d for d in data if d["question_type"] == args.type]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    core = Tenet(Path(tempfile.mkdtemp()) / "emb.db")  # embedder host
    embedder = core.core.embed_batch

    cache_dir = None
    if args.cache:
        cache_dir = Path(args.cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
    dump_f = open(args.dump, "w") if args.dump else None

    rows = []
    t_start = time.time()
    for i, inst in enumerate(data):
        r = eval_instance(inst, args.k, embedder, qa=args.qa, do_full=not args.no_full,
                          expand=args.expand, hops=args.hops,
                          cache_dir=cache_dir, type_budgets=args.type_budgets)
        rows.append(r)
        if dump_f and args.qa and not r.get("qa_error") and not (r["rag_qa"] and r["tenet_qa"]):
            dump_f.write(json.dumps({k2: r.get(k2) for k2 in
                                     ("qid", "type", "question", "gold", "rag_qa", "tenet_qa",
                                      "rag_pred", "tenet_pred", "tenet_ctx")}) + "\n")
            dump_f.flush()
        tail = ""
        if args.qa:
            tail = (" | QA ERROR (excluded)" if r.get("qa_error") else
                    f" | QA rag:{'✓' if r['rag_qa'] else '✗'} tenet:{'✓' if r['tenet_qa'] else '✗'}")
        print(f"[{i+1}/{len(data)}] {r['type'][:18]:18s} turns={r['turns']:4d} | "
              f"recall rag:{'✓' if r['rag_recall'] else '✗'} tenet:{'✓' if r['tenet_recall'] else '✗'}{tail}")

    n = len(rows)
    def pct(key): return 100 * sum(r[key] for r in rows) / n
    print(f"\n=== LongMemEval_S (n={n}{', type='+args.type if args.type else ''}) ===")
    print(f"session-level recall@{args.k}:  rag={pct('rag_recall'):.1f}%  tenet={pct('tenet_recall'):.1f}%")
    avg = lambda key: sum(r[key] for r in rows) / n
    qa_rows = [r for r in rows if not r.get("qa_error")]
    nq = len(qa_rows)
    def pctq(key): return 100 * sum(r[key] for r in qa_rows) / max(nq, 1)
    if args.qa:
        if nq < n:
            print(f"  ⚠ {n-nq} instance(s) excluded from QA (API errors) — scored over n={nq}")
        full_str = f"full-context={pctq('full_qa'):.1f}%  " if not args.no_full else ""
        print(f"\nanswer accuracy (QA):  {full_str}"
              f"rag@{args.k}={pctq('rag_qa'):.1f}%  tenet={pctq('tenet_qa'):.1f}%")
        # the frontier: accuracy per unit of reader context (tokens ≈ chars/4)
        frontier = [("rag", "rag_qa", "rag_ctx_chars"), ("tenet", "tenet_qa", "tenet_ctx_chars")]
        if not args.no_full:
            frontier = [("full-context", "full_qa", "full_ctx_chars")] + frontier
        for name, acc_k, ctx_k in frontier:
            toks = avg(ctx_k) / 4
            print(f"  {name:13s} acc={pctq(acc_k):5.1f}%  ctx≈{toks:6.0f} tok  "
                  f"acc/1k-tok={pctq(acc_k)/max(toks/1000,1e-6):6.1f}")
    print(f"\ncontext chars fed to reader: full≈{avg('full_ctx_chars'):.0f}  "
          f"rag≈{avg('rag_ctx_chars'):.0f}  tenet≈{avg('tenet_ctx_chars'):.0f}")
    print(f"  → tenet uses {100*(1-avg('tenet_ctx_chars')/avg('full_ctx_chars')):.1f}% less context than full history")
    # per-type
    print("\nby type (recall rag/tenet" + (" · QA rag/tenet" if args.qa else "") + "):")
    types = {}
    for r in rows:
        types.setdefault(r["type"], []).append(r)
    for qt, rs in sorted(types.items()):
        t = len(rs)
        line = f"  {qt:24s} {100*sum(x['rag_recall'] for x in rs)/t:5.1f}%/{100*sum(x['tenet_recall'] for x in rs)/t:5.1f}%"
        if args.qa:
            qs = [x for x in rs if not x.get("qa_error")]
            tq = max(len(qs), 1)
            line += f"  ·  {100*sum(x['rag_qa'] for x in qs)/tq:5.1f}%/{100*sum(x['tenet_qa'] for x in qs)/tq:5.1f}%"
        print(line + f"  (n={t})")
    if args.qa:
        print(f"\nQA tokens: in={_qa_usage['in']:,} out={_qa_usage['out']:,}")
    print(f"wall={time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
