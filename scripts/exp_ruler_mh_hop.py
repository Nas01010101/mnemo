"""EXPERIMENT — RULER multi-hop: Self-Ask per-hop query decomposition at pool
construction (the lever the RRF null pointed to, docs_scratch/ruler_mh_rrf.md).

Finding that motivates this: gold-in-pool 72% >> QA 45% on ruler_qa2, and RRF
re-ranking ties baseline exactly — the gap vs HippoRAG-v2 (66) is multi-hop
COMPOSITION: the bridge chunk for hop 2 is similar to the *intermediate* entity,
not to the original question, so query-similarity ranking never surfaces it.
HippoRAG reaches it by graph traversal; we reach it by asking the (local) reader
to name the sub-questions, then retrieving per sub-question.

Arm (decompose): one 7B call decomposes the question into <=2 sub-questions;
pool = dense top-k/2 per sub-question + top-k/2 for the original question,
deduped, capped at the SAME char budget as baseline. Reader identical. A failed
decomposition (bad JSON) falls back to the baseline pool — the arm can only add
retrieval directions, never lose the original ones.

Paired protocol: identical questions/reader/prompts/budget; SubEM; McNemar +
Wilson CIs; decomposition + reading on ollama (RTX box) — zero cloud spend.

Usage:
  LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b OLLAMA_BASE_URL=http://<rtx>:11434/v1 \
      EMBED_PROVIDER=local python scripts/exp_ruler_mh_hop.py --qpc 100
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from tenet import config  # noqa: E402
from bench_factcon import subem_max, wilson_ci, answer_extract  # noqa: E402
from bench_mab_ar import build_store, chunks_of  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs_scratch"

_DECOMP_PROMPT = (
    "Decompose the question into at most 2 simpler retrieval sub-questions that must "
    "each be answered to answer it. If the question is already simple, return it alone. "
    "Write the sub-questions in English.\n"
    'Reply ONLY with JSON: {{"subs": ["...", "..."]}}\n\nQuestion: {q}')


def decompose(q: str) -> list[str]:
    try:
        raw = config.chat(
            [{"role": "user", "content": _DECOMP_PROMPT.format(q=q)}],
            qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"),
            max_tokens=160, json_mode=True)
        subs = json.loads(raw).get("subs", [])
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()][:2]
        return subs
    except Exception:
        return []


def pool_from(chs, idxs, budget: int) -> str:
    picked, used = [], 0
    for i in idxs:
        if used + len(chs[i]) > budget and picked:
            continue
        picked.append(i)
        used += len(chs[i])
    return "\n---\n".join(chs[i] for i in sorted(picked))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="ruler_qa2")
    ap.add_argument("--qpc", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
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

        for qi, (q, gold) in enumerate(list(zip(ex["questions"], ex["answers"]))[: args.qpc]):
            qv = np.asarray(m.core.embed_batch([q])[0])
            base_idx = list(np.argsort(-(mat @ qv))[: args.k])
            budget = len("\n---\n".join(chs[i] for i in sorted(base_idx)))

            subs = decompose(q)
            if len(subs) >= 2:
                # ITERATIVE Self-Ask: answer hop 1 from its own small pool, then
                # ANCHOR hop 2's retrieval on that intermediate answer — the bridge
                # chunk is similar to the intermediate entity, not to the original
                # question (the composition gap the RRF null isolated).
                s1v = np.asarray(m.core.embed_batch([subs[0]])[0])
                s1_idx = list(np.argsort(-(mat @ s1v))[: max(args.k // 2, 4)])
                ans1 = answer_extract(pool_from(chs, s1_idx, budget // 2), subs[0])
                hop2_q = f"{subs[1]} {ans1.strip()}" if ans1.strip() else subs[1]
                s2v = np.asarray(m.core.embed_batch([hop2_q])[0])
                per = max(args.k // 3, 3)
                idxs, seen = [], set()
                for v in (qv, s1v, s2v):
                    for i in np.argsort(-(mat @ v))[:per]:
                        if int(i) not in seen:
                            seen.add(int(i))
                            idxs.append(int(i))
                hop_pool = pool_from(chs, idxs, budget)
                row_extra = {"ans1": ans1[:120], "hop2_q": hop2_q[:200]}
            elif subs:
                sv = np.asarray(m.core.embed_batch([subs[0]])[0])
                idxs, seen = [], set()
                for v in (qv, sv):
                    for i in np.argsort(-(mat @ v))[: args.k // 2]:
                        if int(i) not in seen:
                            seen.add(int(i))
                            idxs.append(int(i))
                hop_pool = pool_from(chs, idxs, budget)
                row_extra = {}
            else:
                hop_pool = pool_from(chs, base_idx, budget)
                row_extra = {}

            base_pool = pool_from(chs, base_idx, budget)
            bp, hp = answer_extract(base_pool, q), answer_extract(hop_pool, q)
            row = {"cell": source, "qi": qi, "q": q, "gold": gold, "subs": subs, **row_extra}
            if not bp.strip() or not hp.strip():
                row["err"] = True
            else:
                golds = gold if isinstance(gold, list) else [gold]
                row.update(base_pred=bp, hop_pred=hp,
                           base_gold_in=any(str(g).lower() in base_pool.lower() for g in golds),
                           hop_gold_in=any(str(g).lower() in hop_pool.lower() for g in golds),
                           base_ok=bool(subem_max(bp, gold)), hop_ok=bool(subem_max(hp, gold)))
            rows.append(row)
            if (qi + 1) % 10 == 0:
                sc = [r for r in rows if "base_ok" in r]
                print(f"  [{len(rows)}] base={sum(r['base_ok'] for r in sc)}/{len(sc)} "
                      f"hop={sum(r['hop_ok'] for r in sc)}/{len(sc)}", flush=True)
        m.close()

    sc = [r for r in rows if "base_ok" in r]
    n = len(sc)
    bok, hok = sum(r["base_ok"] for r in sc), sum(r["hop_ok"] for r in sc)
    n01 = sum((not r["base_ok"]) and r["hop_ok"] for r in sc)
    n10 = sum(r["base_ok"] and (not r["hop_ok"]) for r in sc)
    from math import comb
    nd = n01 + n10
    p = (sum(comb(nd, x) for x in range(min(n01, n10) + 1)) / 2 ** nd * 2) if nd else 1.0
    print(f"\n=== SubEM paired (n={n}, excluded={len(rows)-n}) ===")
    blo, bhi = wilson_ci(bok / n, n)
    hlo, hhi = wilson_ci(hok / n, n)
    print(f"  baseline:  {100*bok/n:.1f}% [{100*blo:.1f},{100*bhi:.1f}]")
    print(f"  decompose: {100*hok/n:.1f}% [{100*hlo:.1f},{100*hhi:.1f}]")
    print(f"  gold-in-pool: base={sum(r['base_gold_in'] for r in sc)} hop={sum(r['hop_gold_in'] for r in sc)}")
    print(f"  McNemar: n01(hop-only)={n01} n10(base-only)={n10} p={min(p,1):.3f}")

    OUT.mkdir(exist_ok=True)
    out = OUT / "ruler_mh_hop.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}  wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
