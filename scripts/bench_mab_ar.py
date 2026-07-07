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
import config       # noqa: E402
from tenet import Tenet  # noqa: E402
from bench_factcon import subem_max, wilson_ci, answer_extract  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "mab_ar"
CHUNK_CHARS = 1200          # raw slice size (~300 tokens)


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
        return Tenet(dbp), np.load(npz)["v"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma list of source prefixes (default all)")
    ap.add_argument("--qpc", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--expand", type=int, default=20)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    from datasets import load_dataset
    ar = load_dataset("ai-hyz/MemoryAgentBench", split="Accurate_Retrieval")
    want = set(args.cells.split(",")) if args.cells else None

    dump_f = open(args.dump, "w") if args.dump else None
    per_cell: dict[str, list[int]] = {}
    t0 = time.time()
    for ex in ar:
        source = ex["metadata"]["source"]
        if want and not any(source.startswith(w) for w in want):
            continue
        cache_id = "ar" + hashlib.md5(ex["context"].encode()).hexdigest()[:12]
        chs = chunks_of(ex["context"])
        print(f"\n=== {source}: {len(chs)} chunks (cache {cache_id}) ===", flush=True)
        m, mat = build_store(cache_id, chs)

        stats = per_cell.setdefault(source, [0, 0, 0, 0])  # rag_ok, tenet_ok, n, err
        for qi, (q, gold) in enumerate(list(zip(ex["questions"], ex["answers"]))[: args.qpc]):
            qv = np.asarray(m.core.embed_batch([q])[0])
            top = sorted(np.argsort(-(mat @ qv))[: args.k])
            rag_pool = "\n---\n".join(chs[i] for i in top)
            hits = m.core.recall(q, k=args.k, expand=args.expand, hops=args.hops,
                                 char_budget=len(rag_pool))
            tenet_pool = "\n---\n".join(h.text for h in hits)
            rp = answer_extract(rag_pool, q)
            tp = answer_extract(tenet_pool, q)
            if not rp.strip() or not tp.strip():
                stats[3] += 1
                continue
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
