"""EXPERIMENT — RULER multi-hop (MAB-AR qa2): BM25+dense hybrid retrieval with
reciprocal-rank fusion at the retrieval stage.

Hypothesis: Tenet's honest MAB-AR loss (RULER MH 45 vs HippoRAG-v2's 66,
BENCHMARK.md §7) is partly retrieval-bound — multi-hop bridge chunks are lexically
anchored (entity names) in ways pure dense cosine under-ranks. The same
BM25+dense+RRF change lifted LME-V2 gold recall 12%→59.7% (paper §4.7).

Two stages, cheap gate first:
  1. DETERMINISTIC gold-in-pool rate (no LLM): baseline tenet retrieval vs RRF
     hybrid, same char budget. If RRF lifts < +3pp, stop — negative, no reader time.
  2. Paired SubEM QA eval (only if stage 1 passes): identical reader/prompts/seed,
     baseline pool vs RRF pool, McNemar + Wilson CIs.

Reuses bench_mab_ar.py machinery verbatim (chunker, caches, reader, SubEM) — no
metric or prompt is reimplemented. Zero-cloud: reader routes to ollama
(LLM_PROVIDER=ollama, e.g. qwen2.5:7b on the RTX box); embeddings local bge-small.

Usage:
  EMBED_PROVIDER=local python scripts/exp_ruler_mh_rrf.py --stage 1
  LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b OLLAMA_BASE_URL=http://<rtx>:11434/v1 \
      EMBED_PROVIDER=local python scripts/exp_ruler_mh_rrf.py --stage 2 --qpc 100
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from bench_factcon import subem_max, wilson_ci, answer_extract  # noqa: E402
from bench_mab_ar import build_store, chunks_of  # noqa: E402
import hashlib  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs_scratch"


def _tok(s: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if w]


def rrf_pool(chs, mat, bm25, q, qv, budget: int, k_each: int = 30, c: int = 60) -> str:
    """BM25 + dense RRF over chunks, filled to the same char budget as baseline."""
    dense_rank = np.argsort(-(mat @ qv))
    bm_scores = bm25.get_scores(_tok(q))
    bm_rank = np.argsort(-bm_scores)
    score: dict[int, float] = {}
    for r, i in enumerate(dense_rank[:k_each]):
        score[int(i)] = score.get(int(i), 0.0) + 1.0 / (c + r + 1)
    for r, i in enumerate(bm_rank[:k_each]):
        score[int(i)] = score.get(int(i), 0.0) + 1.0 / (c + r + 1)
    fused = sorted(score, key=lambda i: -score[i])
    picked, used = [], 0
    for i in fused:
        if used + len(chs[i]) > budget and picked:
            continue
        picked.append(i)
        used += len(chs[i])
    return "\n---\n".join(chs[i] for i in sorted(picked))


def baseline_pool(m, chs, mat, q, qv, k: int, expand: int, hops: int) -> str:
    top = sorted(np.argsort(-(mat @ qv))[:k])
    rag_budget = len("\n---\n".join(chs[i] for i in top))
    hits = m.core.recall(q, k=k, expand=expand, hops=hops, char_budget=rag_budget)
    return "\n---\n".join(h.text for h in hits), rag_budget


def gold_in(pool: str, gold) -> bool:
    golds = gold if isinstance(gold, list) else [gold]
    return any(str(g).lower() in pool.lower() for g in golds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--cells", default="ruler_qa2", help="source prefix (MH = qa2)")
    ap.add_argument("--qpc", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--expand", type=int, default=20)
    args = ap.parse_args()

    from datasets import load_dataset
    ar = load_dataset("ai-hyz/MemoryAgentBench", split="Accurate_Retrieval")

    t0 = time.time()
    rows = []
    for ex in ar:
        source = ex["metadata"]["source"]
        if not source.startswith(args.cells):
            continue
        chs = chunks_of(ex["context"])
        cache_id = "ar" + hashlib.md5("\x00".join(chs).encode()).hexdigest()[:12]
        print(f"\n=== {source}: {len(chs)} chunks (cache {cache_id}) ===", flush=True)
        m, mat = build_store(cache_id, chs)
        bm25 = BM25Okapi([_tok(c) for c in chs])

        for qi, (q, gold) in enumerate(list(zip(ex["questions"], ex["answers"]))[: args.qpc]):
            qv = np.asarray(m.core.embed_batch([q])[0])
            base_pool, budget = baseline_pool(m, chs, mat, q, qv, args.k, args.expand, args.hops)
            hyb_pool = rrf_pool(chs, mat, bm25, q, qv, budget)
            row = {"cell": source, "qi": qi, "q": q, "gold": gold,
                   "base_gold_in": gold_in(base_pool, gold),
                   "rrf_gold_in": gold_in(hyb_pool, gold)}
            if args.stage == 2:
                bp, hp = answer_extract(base_pool, q), answer_extract(hyb_pool, q)
                if not bp.strip() or not hp.strip():
                    row["err"] = True
                else:
                    row.update(base_pred=bp, rrf_pred=hp,
                               base_ok=bool(subem_max(bp, gold)),
                               rrf_ok=bool(subem_max(hp, gold)))
            rows.append(row)
            if (qi + 1) % 20 == 0:
                b = sum(r["base_gold_in"] for r in rows)
                h = sum(r["rrf_gold_in"] for r in rows)
                print(f"  [{len(rows)}] gold-in-pool base={b} rrf={h}", flush=True)
        m.close()

    n = len(rows)
    b = sum(r["base_gold_in"] for r in rows)
    h = sum(r["rrf_gold_in"] for r in rows)
    print(f"\n=== stage-1 gold-in-pool (n={n}) ===")
    blo, bhi = wilson_ci(b / n, n)
    hlo, hhi = wilson_ci(h / n, n)
    print(f"  baseline: {100*b/n:.1f}% [{100*blo:.1f},{100*bhi:.1f}]")
    print(f"  rrf:      {100*h/n:.1f}% [{100*hlo:.1f},{100*hhi:.1f}]")

    if args.stage == 2:
        sc = [r for r in rows if "base_ok" in r]
        bok = sum(r["base_ok"] for r in sc)
        hok = sum(r["rrf_ok"] for r in sc)
        n01 = sum((not r["base_ok"]) and r["rrf_ok"] for r in sc)
        n10 = sum(r["base_ok"] and (not r["rrf_ok"]) for r in sc)
        # McNemar exact (binomial) on discordant pairs
        from math import comb
        nd = n01 + n10
        p = (sum(comb(nd, x) for x in range(min(n01, n10) + 1)) / 2 ** nd * 2) if nd else 1.0
        print(f"\n=== stage-2 SubEM (n={len(sc)}, excluded={len(rows)-len(sc)}) ===")
        blo, bhi = wilson_ci(bok / len(sc), len(sc))
        hlo, hhi = wilson_ci(hok / len(sc), len(sc))
        print(f"  baseline: {100*bok/len(sc):.1f}% [{100*blo:.1f},{100*bhi:.1f}]")
        print(f"  rrf:      {100*hok/len(sc):.1f}% [{100*hlo:.1f},{100*hhi:.1f}]")
        print(f"  McNemar: n01(rrf-only-right)={n01} n10(base-only-right)={n10} p={min(p,1):.3f}")

    OUT.mkdir(exist_ok=True)
    out = OUT / f"ruler_mh_rrf_stage{args.stage}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}  wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
