"""Probe 2 — MH decompose-reader (the probe Wissem insisted we not reject untested).

FC multi-hop is Tenet's weakest cell (published SOTA trick = CAR max(serial) aggregation
/ Self-Ask decomposition applied AFTER retrieval, arXiv:2606.01435). This probe adds a
`--decompose` arm to the FC-MH evaluation as a THIN WRAPPER over bench_factcon.py (kept
untouched) so the diff is surgical:

  baseline (tenet)  : single recall(q, k) -> full reader (MAB official max-serial prompt)
  decompose         : (1) ONE cheap LLM call decomposes q into sub-questions;
                      (2) recall per sub-question against the SAME Tenet store;
                      (3) UNION the recalled pools (dedupe by id, serial-ordered);
                      (4) the SAME full reader answers over the union.

Single-variable A/B: identical store, identical final reader (model + prompt); the ONLY
difference is retrieval-pool construction (single vs decomposed-union). Same n, same
questions, temp=0 (deterministic → "same seeds").

GATE (pre-registered): report the delta with Wilson CIs. Escalate to mh_32k n=50 ONLY if
the mh_6k smoke delta is >= +15pp (outside noise). If the delta is <= noise, the honest
verdict is "MH is reader-reasoning-bound even with decomposition on a clean store".

Usage:
  python scripts/bench_mh_decompose.py --cell mh_6k --limit 20
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Writable scratch cache (repo data/ symlink is TCC-blocked — see BENCHMARK.md §9).
_SCRATCH = Path(os.environ.get(
    "ROUTING_CACHE_DIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "tenet_routing_cache")))

from tenet import config  # noqa: E402
import bench_factcon as fc  # noqa: E402

fc.CACHE = _SCRATCH / "factcon"
fc.CACHE.mkdir(parents=True, exist_ok=True)

CHEAP_MODEL = os.environ.get("ROUTING_CHEAP_MODEL", "qwen3.6-flash")


def cheap_decompose(question: str) -> list[str]:
    """ONE cheap (flash) decomposition call. Same prompt contract as bench_factcon's
    decompose(), but on the cheap tier — the probe spends the strong model only on the
    final aggregated answer."""
    out = config.chat([{"role": "user", "content": fc._DECOMP_PROMPT.format(question=question)}],
                      qwen_default=CHEAP_MODEL, max_tokens=200, json_mode=True)
    try:
        import re
        hops = json.loads(re.search(r"\{.*\}", out, re.S).group(0))["hops"]
        hops = [str(h) for h in hops][:4]
        return hops or [question]
    except Exception:  # noqa: BLE001
        return [question]


def decompose_pool(m, question: str, k: int) -> tuple[str, int]:
    """Union of per-sub-question recalls over the SAME store, deduped by memory id,
    serial-ordered (so 'newer = larger serial' holds for the reader). Returns
    (pool_text, n_subqs)."""
    subqs = cheap_decompose(question)
    seen: dict[int, object] = {}
    for sq in subqs:
        for h in m.core.recall(sq, k=k):
            seen.setdefault(h.id, h)
    hits = sorted(seen.values(), key=lambda h: int(h.source) if str(h.source).isdigit() else 0)
    pool = "\n".join(f"{h.source}. {h.text}" for h in hits)
    return pool, len(subqs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="mh_6k")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keys", choices=["llm", "heuristic"], default="llm")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    fc.KEY_MODE = args.keys
    assert config.LLM_PROVIDER == "qwen", "run against the shipped qwen backbone"

    from datasets import load_dataset
    cr = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution")
    target = f"factconsolidation_{args.cell}"
    t0 = time.time()

    ex = next(e for e in cr if e["metadata"]["source"] == target)
    facts = fc.parse_facts(ex["context"])
    cache_id = "ctx" + hashlib.md5(ex["context"].encode()).hexdigest()[:12]
    print(f"=== {args.cell}: {len(facts)} facts (cache {cache_id}) ===", flush=True)
    m = fc.build_tenet(cache_id, facts)

    qs = list(zip(ex["questions"], ex["answers"]))[: args.limit]

    # Precompute both pools per question (recall + cheap decompose), then run the SAME
    # full reader over each pool concurrently.
    base_pools, dec_pools, n_subqs = [], [], []
    for q, _g in qs:
        base_pools.append("\n".join(f"{h.source}. {h.text}" for h in m.core.recall(q, k=args.k)))
        dp, ns = decompose_pool(m, q, args.k)
        dec_pools.append(dp); n_subqs.append(ns)

    def _answer(job):
        arm, qi, pool = job
        q = qs[qi][0]
        try:
            return arm, qi, fc.answer(pool, q)  # MAB official max-serial reader (full model)
        except config.ProviderError:
            raise
        except Exception:  # noqa: BLE001
            return arm, qi, ""

    jobs = ([("base", i, base_pools[i]) for i in range(len(qs))]
            + [("dec", i, dec_pools[i]) for i in range(len(qs))])
    preds = {"base": [""] * len(qs), "dec": [""] * len(qs)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool_ex:
        for arm, qi, pred in pool_ex.map(_answer, jobs):
            preds[arm][qi] = pred
    m.close()

    # Score — a question is scored only if BOTH arms produced a reader answer (paired,
    # API failures excluded not scored wrong).
    stats = {"base": [0, 0], "dec": [0, 0]}
    misses = []
    for i, (q, gold) in enumerate(qs):
        bp, dp = preds["base"][i], preds["dec"][i]
        if not bp.strip() or not dp.strip():
            continue
        b_ok = fc.subem_max(bp, gold); d_ok = fc.subem_max(dp, gold)
        stats["base"][0] += b_ok; stats["base"][1] += 1
        stats["dec"][0] += d_ok; stats["dec"][1] += 1
        if b_ok != d_ok:
            misses.append({"q": q, "gold": gold, "base": bp, "dec": dp,
                           "base_ok": bool(b_ok), "dec_ok": bool(d_ok),
                           "n_subqs": n_subqs[i]})

    # ---- report ----
    print(f"\n=== FC {args.cell} decompose A/B (SubEM, k={args.k}, full reader) ===")
    rows = {}
    for arm in ("base", "dec"):
        c, n = stats[arm]
        p = c / n if n else 0.0
        lo, hi = fc.wilson_ci(p, n)
        rows[arm] = (p, lo, hi, n)
        label = "tenet (single recall)" if arm == "base" else "decompose (union pools)"
        print(f"  {label:>26}: {100*p:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}]  n={n}")
    delta = 100 * (rows["dec"][0] - rows["base"][0])
    avg_subqs = sum(n_subqs) / len(n_subqs) if n_subqs else 0
    print(f"\n  delta (decompose - tenet): {delta:+.1f} pp   (avg {avg_subqs:.1f} sub-qs/question)")
    print(f"  base CI [{100*rows['base'][1]:.1f},{100*rows['base'][2]:.1f}]  "
          f"dec CI [{100*rows['dec'][1]:.1f},{100*rows['dec'][2]:.1f}]  "
          f"(CIs overlap => within noise)")

    # ---- gate ----
    print("\n=== PRE-REGISTERED GATE (escalate to mh_32k n=50 iff delta >= +15pp) ===")
    if delta >= 15.0:
        print(f"  ESCALATE — smoke delta {delta:+.1f}pp >= +15pp; run mh_32k n=50 to confirm.")
    else:
        print(f"  NO ESCALATION — smoke delta {delta:+.1f}pp < +15pp.")
        overlap = not (rows["dec"][1] > rows["base"][2] or rows["base"][1] > rows["dec"][2])
        if overlap:
            print("  VERDICT: MH is reader-reasoning-bound even with decomposition on a "
                  "clean (conflict-free) Tenet store — CIs overlap, decomposition does not "
                  "move the needle beyond noise.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "cell": args.cell, "stats": stats, "rows": {k: list(v) for k, v in rows.items()},
            "delta_pp": delta, "avg_subqs": avg_subqs, "misses": misses,
        }, indent=2))
        print(f"\nwrote {args.out}")
    print(f"wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
