"""LongMemEval benchmark harness for Tenet (the committed validation).

Compares three read strategies on the same questions, LLM-judged (Qwen):
  • full   — dump all session text into the answerer (no memory)   [baseline]
  • rag    — embed turns, top-k cosine, no distill/bi-temporal      [baseline]
  • tenet  — distill -> bi-temporal store -> forgetting-aware recall [ours]

Reports overall + per-type accuracy, retrieval latency, and context tokens.
Honest protocol: Qwen is answerer AND judge (we're on Qwen Cloud), so numbers are
INDICATIVE and compared against our own baselines — not pasted onto the gpt-4o leaderboard.

Usage:
  python scripts/lme_bench.py --limit 5 --methods tenet,full,rag          # cost probe
  python scripts/lme_bench.py --limit 120 --methods tenet,rag --seed 0    # real run
"""
import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
from tenet import Tenet  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "lme" / "longmemeval_oracle.json"
ANSWER_MODEL = config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus")
JUDGE_MODEL = config.get("QWEN_JUDGE_MODEL", "qwen3.7-plus")

_client = config.qwen_client()
_usage = {"prompt": 0, "completion": 0}


def _chat(model, system, user, max_tokens=512):
    r = _client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0, max_tokens=max_tokens, extra_body={"enable_thinking": False},
    )
    _usage["prompt"] += r.usage.prompt_tokens
    _usage["completion"] += r.usage.completion_tokens
    return (r.choices[0].message.content or "").strip()


def _embed(texts):
    r = _client.embeddings.create(model=config.QWEN_EMBED_MODEL, input=texts)
    return [np.asarray(d.embedding, dtype=np.float32) for d in r.data]


ANSWER_SYS = (
    "You answer a question using ONLY the provided memory/context about the user. "
    "Think step by step over the notes (chain-of-note), then give a short, direct answer. "
    "If the context lacks the answer, say 'I don't know'. End with 'ANSWER: <answer>'."
)


def answer_from(context: str, question: str, qdate: str) -> str:
    user = f"Question date: {qdate}\n\nMemory/context:\n{context}\n\nQuestion: {question}"
    out = _chat(ANSWER_MODEL, ANSWER_SYS, user)
    return out.split("ANSWER:")[-1].strip() if "ANSWER:" in out else out


JUDGE_SYS = (
    "You are grading whether a model's answer matches the gold answer for a question. "
    "Be lenient about phrasing; judge semantic correctness. "
    "Reply with exactly 'yes' or 'no'."
)


def judge(question: str, gold: str, pred: str) -> bool:
    user = f"Question: {question}\nGold answer: {gold}\nModel answer: {pred}\nIs the model answer correct?"
    return _chat(JUDGE_MODEL, JUDGE_SYS, user, max_tokens=4).lower().startswith("y")


def sessions_as_turns(inst):
    """Flatten sessions to (date, 'User: ...'/'Assistant: ...') strings, oldest first."""
    turns = []
    for sess, date in zip(inst["haystack_sessions"], inst["haystack_dates"]):
        for turn in sess:
            role = turn["role"].capitalize()
            turns.append((date, f"{role}: {turn['content']}"))
    return turns


# ---- three read strategies -------------------------------------------------
def run_full(inst):
    turns = sessions_as_turns(inst)
    ctx = "\n".join(f"[{d}] {t}" for d, t in turns)
    t0 = time.time()
    ans = answer_from(ctx, inst["question"], inst["question_date"])
    return ans, time.time() - t0, len(ctx)


def run_rag(inst, k=10):
    turns = sessions_as_turns(inst)
    texts = [t for _, t in turns]
    t0 = time.time()
    embs = []
    for i in range(0, len(texts), 10):   # DashScope embedding batch limit = 10
        embs += _embed(texts[i:i + 10])
    embs = np.array([v / (np.linalg.norm(v) or 1) for v in embs])
    qv = _embed([inst["question"]])[0]; qv = qv / (np.linalg.norm(qv) or 1)
    sims = embs @ qv
    top = np.argsort(-sims)[:k]
    lat = time.time() - t0
    ctx = "\n".join(f"[{turns[i][0]}] {texts[i]}" for i in sorted(top))
    ans = answer_from(ctx, inst["question"], inst["question_date"])
    return ans, lat, len(ctx)


def run_tenet(inst, k=10):
    import tempfile
    db = Path(tempfile.mkdtemp()) / "b.db"
    m = Tenet(db)
    # hybrid ingest per session: distilled keyed facts + raw verbatim slices
    for sess, date in zip(inst["haystack_sessions"], inst["haystack_dates"]):
        try:
            m.ingest_session(sess)
        except Exception:
            pass
    t0 = time.time()
    hits = m.recall(inst["question"], k=k)
    lat = time.time() - t0
    ctx = "\n".join(f"- {h.text}" for h in hits)
    ans = answer_from(ctx, inst["question"], inst["question_date"])
    m.close()
    return ans, lat, len(ctx)


RUNNERS = {"full": run_full, "rag": run_rag, "tenet": run_tenet}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--methods", default="tenet,full,rag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-abstention", action="store_true", default=True)
    args = ap.parse_args()

    data = json.load(open(DATA))
    if args.skip_abstention:
        data = [d for d in data if not d["question_id"].endswith("_abs")]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]
    methods = args.methods.split(",")

    results = {m: {"correct": 0, "lat": [], "ctx": [], "by_type": {}} for m in methods}
    t_start = time.time()
    for i, inst in enumerate(data):
        line = f"[{i+1}/{len(data)}] {inst['question_type'][:18]:18s} "
        for meth in methods:
            try:
                ans, lat, ctxlen = RUNNERS[meth](inst)
                ok = judge(inst["question"], inst["answer"], ans)
            except Exception as e:
                print(f"  ! {meth} errored: {type(e).__name__}: {str(e)[:80]}")
                ok, lat, ctxlen = False, 0.0, 0
            r = results[meth]
            r["correct"] += ok; r["lat"].append(lat); r["ctx"].append(ctxlen)
            bt = r["by_type"].setdefault(inst["question_type"], [0, 0])
            bt[0] += ok; bt[1] += 1
            line += f"| {meth}:{'✓' if ok else '✗'} "
        print(line)

    n = len(data)
    print(f"\n=== Results (n={n}, answerer+judge={ANSWER_MODEL}, INDICATIVE Qwen-judged) ===")
    for meth in methods:
        r = results[meth]
        acc = 100 * r["correct"] / n
        med_lat = sorted(r["lat"])[len(r["lat"]) // 2] if r["lat"] else 0
        avg_ctx = sum(r["ctx"]) / n
        print(f"{meth:6s}  acc={acc:5.1f}%  ({r['correct']}/{n})  "
              f"retrieval_med={med_lat*1000:6.0f}ms  ctx≈{avg_ctx:6.0f} chars")
        for qt, (c, tot) in sorted(r["by_type"].items()):
            print(f"          {qt:24s} {100*c/tot:5.1f}% ({c}/{tot})")

    pt, ct = _usage["prompt"], _usage["completion"]
    print(f"\ntokens: prompt={pt:,} completion={ct:,}  | wall={time.time()-t_start:.0f}s")
    print(f"(rough per-question amortized: {(pt+ct)/n:,.0f} tokens)")


if __name__ == "__main__":
    main()
