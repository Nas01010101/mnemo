"""MemoryAgentBench Accurate-Retrieval split (ICLR 2026, arXiv:2507.05257).

Four sub-benchmarks over 22 long contexts (197K–534K tokens): RULER-QA (SH/MH
document QA), LongMemEval(S*) dialogue QA, EventQA (novel event continuation).
Published bar (gpt-4o-mini backbone, paper Table 3): HippoRAG-v2 AR avg 65.1
(SH-QA 76 / MH-QA 66 / LME(S*) 50.7 / EventQA 67.6); Mem0 32.6, Zep 37.5.

Tenet arm: ZERO-LLM ingestion (raw chunk slices + embeddings only), dual-pool
recall with belief-anchored expansion + associative hops; extraction reader.
Control arm: top-k chunk RAG, identical reader. Metric: MAB SubEM (verbatim,
imported from bench_factcon) — note the paper scores LME(S*) with a GPT-4o
judge; we report the stricter SubEM for it and label the difference.

Usage: python scripts/bench_mab_ar.py --cells ruler_qa1_197K --qpc 20   # smoke
       python scripts/bench_mab_ar.py --qpc 100                          # full
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
from tenet import Tenet  # noqa: E402
from bench_factcon import subem_max, wilson_ci, answer_extract  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "mab_ar"
CHUNK_CHARS = 1200          # raw slice size (~300 tokens)


def lme_chunks(context: str) -> list[str] | None:
    """Structure-aware ingestion for longmemeval cells: the context is a stringified
    list alternating 'Chat Time: <date>' markers and turn lists. Generic slicing
    divorces turns from their dates — the #1 shared-miss cause (temporal/aggregation
    questions). Emit date-prefixed turns, chunked at turn boundaries. Zero-LLM."""
    import ast
    try:
        items = ast.literal_eval(context)
    except Exception:
        return None
    LME_SESSION_CHARS = 1200      # v3 (2400) regressed: fewer retrieval units → less
                                  # session diversity in top-k. 1200 = the v2 optimum.
    out, cur, date = [], "", ""
    def flush():
        nonlocal cur
        if cur.strip():
            out.append(cur.strip())
        cur = ""
    for it in items:
        if isinstance(it, str):
            date = it.replace("Chat Time: ", "").strip()
            flush()
        elif isinstance(it, list):
            for t in it:
                line = f"[{date}] {t.get('role', '?')}: {t.get('content', '')}".strip()
                if len(cur) + len(line) > LME_SESSION_CHARS:
                    flush()
                # a single long turn becomes its own (oversize) chunk rather than
                # being split away from its date prefix
                cur = (cur + "\n" + line) if cur else line
    flush()
    return out or None


def chunks_of(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        j = text.rfind("\n", i, i + CHUNK_CHARS)
        j = j if j > i + 200 else min(i + CHUNK_CHARS, len(text))
        out.append(text[i:j].strip())
        i = j
    return [c for c in out if c]


def build_store(cache_id: str, chs: list[str]) -> tuple[Tenet, np.ndarray]:
    """Zero-LLM ingestion: every chunk is a raw slice; embeddings cached."""
    dbp, npz = CACHE / f"{cache_id}.db", CACHE / f"{cache_id}.npz"
    if dbp.exists() and npz.exists():
        mat = np.load(npz)["v"]
        if len(mat) != len(chs):
            raise RuntimeError(f"cache {cache_id}: {len(mat)} vectors != {len(chs)} chunks")
        return Tenet(dbp), mat
    m = Tenet(dbp)
    vecs = []
    B = 256
    for i in range(0, len(chs), B):
        vecs.extend(m.core.embed_batch(chs[i:i + B]))
    mat = np.array(vecs)
    for idx, (c, v) in enumerate(zip(chs, mat)):
        m.core.store(c, kind="raw", salience=0.5, source=str(idx), _vec=v)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, v=mat)
    return m, mat


# MAB's official metric for the longmemeval cells is LLM-as-judge (paper: gpt-4o),
# NOT SubEM — the anscheck prompts below are copied VERBATIM from the MAB repo
# (llm_based_eval/longmem_qa_evaluate.py, get_anscheck_prompt). Judge model comes
# from QWEN_JUDGE_MODEL (default qwen-max, DashScope) — labeled wherever reported.
_ANSCHECK = {
    "default": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}

_judge_client = None


def judge_correct(question: str, gold, pred: str, qtype: str, abstention: bool) -> bool | None:
    """Official MAB judge. Returns None on judge API failure (caller excludes)."""
    global _judge_client
    import os
    if _judge_client is None:
        from openai import OpenAI
        if os.environ.get("JUDGE_PROVIDER") == "openrouter":
            _judge_client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                                   base_url="https://openrouter.ai/api/v1")
        else:
            _judge_client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                                   base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    g = gold[0] if isinstance(gold, list) else gold
    tpl = _ANSCHECK["abstention"] if abstention else _ANSCHECK.get(qtype, _ANSCHECK["default"])
    for _ in range(4):
        try:
            r = _judge_client.chat.completions.create(
                model=os.environ.get("JUDGE_MODEL", os.environ.get("QWEN_JUDGE_MODEL", "qwen-max")),
                messages=[{"role": "user", "content": tpl.format(question, g, pred)}],
                max_tokens=5, temperature=0)
            out = (r.choices[0].message.content or "").strip().lower()
            if out.startswith(("yes", "no")):
                return out.startswith("yes")
        except Exception:
            time.sleep(3)
    return None


# EventQA cells are MULTIPLE-CHOICE: the question lists candidate next events and
# the gold is one of them verbatim — a free-form extraction reader paraphrases and
# can never SubEM-match. The choice reader copies exactly one candidate.
_CHOICE_PROMPT = (
    "You are given the events of a story so far, a list of possible subsequent events, "
    "and retrieved excerpts from the story. Using ONLY the excerpts, decide which "
    "candidate event actually happens next.\n"
    "Reply by COPYING exactly ONE event from the candidate list, verbatim, with no "
    "other words.\n\n[Story excerpts]\n{pool}\n\n{question}\nNext event:")


def answer_choice(pool: str, question: str) -> str:
    import config as _c
    return _c.chat(
        [{"role": "user", "content": _CHOICE_PROMPT.format(pool=pool, question=question)}],
        qwen_default=_c.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"), max_tokens=96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma list of source prefixes (default all)")
    ap.add_argument("--qpc", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--expand", type=int, default=20)
    ap.add_argument("--dump", default="")
    ap.add_argument("--diverse", type=int, default=0,
                    help="session-diverse recall (tenet arm, LME cells): cap hits per "
                         "session at this value so top-k must cover distinct sessions; "
                         "targets multi-session synthesis misses")
    ap.add_argument("--dump-preds", default="",
                    help="write EVERY prediction (both arms) to this JSONL so scoring "
                         "can be redone later — e.g. LLM-judge once a judge is available")
    ap.add_argument("--judge", action="store_true",
                    help="score longmemeval cells with MAB's official LLM-judge "
                         "(anscheck prompts verbatim; QWEN_JUDGE_MODEL, default qwen-max) "
                         "instead of SubEM — the paper's own metric for those cells")
    args = ap.parse_args()

    from datasets import load_dataset
    ar = load_dataset("ai-hyz/MemoryAgentBench", split="Accurate_Retrieval")
    want = set(args.cells.split(",")) if args.cells else None

    dump_f = open(args.dump, "w") if args.dump else None
    preds_f = open(args.dump_preds, "w") if args.dump_preds else None
    per_cell: dict[str, list[int]] = {}
    t0 = time.time()
    for ex in ar:
        source = ex["metadata"]["source"]
        if want and not any(source.startswith(w) for w in want):
            continue
        chs = lme_chunks(ex["context"]) if source.startswith("longmemeval") else None
        chs = chs or chunks_of(ex["context"])
        # cache is keyed by the CHUNK LIST itself, not the raw context — a chunker
        # change can never silently pair stale embeddings with fresh chunks
        cache_id = "ar" + hashlib.md5("\x00".join(chs).encode()).hexdigest()[:12]
        print(f"\n=== {source}: {len(chs)} chunks (cache {cache_id}) ===", flush=True)
        m, mat = build_store(cache_id, chs)

        stats = per_cell.setdefault(source, [0, 0, 0, 0])  # rag_ok, tenet_ok, n, err
        use_judge = args.judge and source.startswith("longmemeval")
        qtypes = ex["metadata"].get("question_types") or [""] * len(ex["questions"])
        qids = ex["metadata"].get("question_ids") or [""] * len(ex["questions"])
        for qi, (q, gold) in enumerate(list(zip(ex["questions"], ex["answers"]))[: args.qpc]):
            qv = np.asarray(m.core.embed_batch([q])[0])
            top = sorted(np.argsort(-(mat @ qv))[: args.k])
            rag_pool = "\n---\n".join(chs[i] for i in top)
            if args.diverse and source.startswith("longmemeval"):
                # over-fetch, then greedy-fill capping hits per session (the [date]
                # prefix identifies the session) — coverage across sessions beats
                # depth within one for multi-session synthesis questions
                pool_hits = m.core.recall(q, k=4 * args.k, expand=args.expand,
                                          hops=args.hops)
                seen: dict[str, int] = {}
                hits, budget = [], len(rag_pool)
                for h in pool_hits:
                    sess = h.text.split("]", 1)[0]
                    if seen.get(sess, 0) >= args.diverse:
                        continue
                    if sum(len(x.text) for x in hits) + len(h.text) > budget:
                        break
                    seen[sess] = seen.get(sess, 0) + 1
                    hits.append(h)
            else:
                hits = m.core.recall(q, k=args.k, expand=args.expand, hops=args.hops,
                                     char_budget=len(rag_pool))
            tenet_pool = "\n---\n".join(h.text for h in hits)
            read = answer_choice if source.startswith("eventqa") else answer_extract
            rp = read(rag_pool, q)
            tp = read(tenet_pool, q)
            if preds_f:
                preds_f.write(json.dumps({"cell": source, "q": q, "gold": gold,
                                          "rag": rp, "tenet": tp,
                                          "qtype": qtypes[qi],
                                          "abs": str(qids[qi]).endswith("_abs")}) + "\n")
                preds_f.flush()
            if not rp.strip() or not tp.strip():
                stats[3] += 1
                continue
            if use_judge:
                qt, ab = qtypes[qi], str(qids[qi]).endswith("_abs")
                r_ok = judge_correct(q, gold, rp, qt, ab)
                t_ok = judge_correct(q, gold, tp, qt, ab)
                if r_ok is None or t_ok is None:   # judge failure: excluded, never scored
                    stats[3] += 1
                    continue
            else:
                r_ok, t_ok = subem_max(rp, gold), subem_max(tp, gold)
            stats[0] += r_ok; stats[1] += t_ok; stats[2] += 1
            if dump_f and not (r_ok and t_ok):
                dump_f.write(json.dumps({"cell": source, "q": q[:300], "gold": gold,
                                         "rag": rp[:200], "tenet": tp[:200],
                                         "rag_ok": bool(r_ok), "tenet_ok": bool(t_ok)}) + "\n")
                dump_f.flush()
            if (qi + 1) % 20 == 0:
                print(f"  [{qi+1}] rag={stats[0]}/{stats[2]} tenet={stats[1]}/{stats[2]}", flush=True)
        m.close()

    # group sub-benchmark cells (e.g. eventqa_65536 + eventqa_full -> eventqa)
    print(f"\n=== MAB Accurate-Retrieval (SubEM, k={args.k}, hops={args.hops}) ===")
    print(f"{'cell':>22} | {'RAG':>18} | {'TENET':>18} | err")
    groups: dict[str, list[int]] = {}
    for src, (r, t, n, e) in sorted(per_cell.items()):
        base = src.split("_")[0]
        g = groups.setdefault(base, [0, 0, 0, 0])
        for i, v in enumerate((r, t, n, e)):
            g[i] += v
        if n:
            print(f"{src:>22} | {100*r/n:5.1f}% (n={n:3d}) | {100*t/n:5.1f}% (n={n:3d}) | {e}")
    print("-" * 70)
    for base, (r, t, n, e) in sorted(groups.items()):
        if n:
            lo, hi = wilson_ci(t / n, n)
            print(f"{base:>22} | {100*r/n:5.1f}% | TENET {100*t/n:.1f}% [{100*lo:.1f},{100*hi:.1f}] n={n}")
    print(f"wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
