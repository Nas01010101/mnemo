"""A/B harness for the ChurnBench §9 read-time fix (BENCHMARK.md §9.1).

ChurnBench (§9) falsified the pre-registered churn-half-life gate: Tenet was the
*worst* arm because stale raw turns (paraphrased, not templated) survive the
write-time `_STALE_ECHO` filter into the k=10 recall pool alongside the correct
current fact. Root cause + two independent, additive fixes:

  component 1 — read-time belief-evidence consistency (memory.py's
      `consistency_threshold` param / consistency.py): key-scoped, drops a raw
      slice close to a superseded fact whose key already has a current fact
      in the pool. Threshold chosen from a sweep over REAL ChurnBench U=2 data
      (see docs/churnbench_threshold_sweep.json) — 0.70 gives 100% recall on
      genuinely-stale raw echoes at 7% false-positive rate on legitimately
      current-value echoes (the FP cost is low: the distilled CURRENT fact is
      unaffected either way, only a supplementary raw duplicate is dropped).

  component 2 — currency-structured reader context (bench_churn.py's
      `format_tenet_context`): "Current beliefs:" then "Supporting raw
      context:", each dated, instead of an unordered flat list.

Four tenet variants, same cached stores (the fix is read-time only — the
store/ingestion is byte-identical across all four; see build_tenet_db reuse
below), swept over U in {2, 8, 32} at n=50 questions/point (10 principals x 5
facts), matching §9's exact dataset config (n_facts=5, distractor_sessions=4,
seed=1) so this table is directly comparable to the §9 baseline row.

Usage:
  CHURN_CACHE_DIR=<writable dir> python scripts/bench_churn_fix_ab.py \
      --updates 2,8,32 --principals 10 --out docs/churnbench_fix_ab.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_churn import CACHE as DEFAULT_CACHE  # noqa: E402
from bench_churn import answer_natural, build_dataset, build_tenet_db, format_tenet_context, score  # noqa: E402
from bench_factcon import wilson_ci  # noqa: E402
from tenet import Tenet  # noqa: E402

CACHE = Path(os.environ["CHURN_CACHE_DIR"]) if os.environ.get("CHURN_CACHE_DIR") else DEFAULT_CACHE

# (label, consistency_threshold, currency_context)
VARIANTS = [
    ("tenet-baseline", None, False),
    ("tenet+1", 0.70, False),
    ("tenet+2", None, True),
    ("tenet+1+2", 0.70, True),
]


def _build_one(args_tuple):
    cache_dir, n_facts, n_dist, seed, u, p = args_tuple
    ds = build_dataset(u, n_facts, n_dist, seed * 1000 + p)
    cache_id = "cfg" + __import__("hashlib").sha256(
        json.dumps(ds["config"], sort_keys=True).encode()).hexdigest()[:12]
    m = build_tenet_db(cache_dir, cache_id, ds["sessions"])
    return ds, m


def _read_job(job):
    """One reader call: (label, u, attr, question, gold, ctx) -> (label, u, attr, question, gold, pred)."""
    label, u, attr, question, gold, ctx = job
    try:
        pred = answer_natural(ctx, question)
    except Exception:  # noqa: BLE001 — transient reader hiccup, excluded not failed
        pred = ""
    return label, u, attr, question, gold, pred


def run_u(u: int, principals: int, n_facts: int, n_dist: int, seed: int,
          k: int, build_workers: int, read_workers: int) -> dict:
    cache_dir = CACHE / str(seed) / f"u{u}_f{n_facts}_d{n_dist}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(cache_dir, n_facts, n_dist, seed, u, p) for p in range(principals)]
    print(f"  building {principals} tenet stores for U={u} (real ingestion, "
          f"reused across all 4 variants)...", flush=True)
    with ThreadPoolExecutor(max_workers=build_workers) as ex:
        built = list(ex.map(_build_one, jobs))

    # recall() is pure vector math (LLM-free) — cheap to precompute for all 4
    # variants x every question up front, single-threaded; only the reader
    # calls (the actual API cost) need concurrency.
    read_jobs = []
    for ds, m in built:
        now = m.core._now()
        for q in ds["questions"]:
            question, gold, attr = q["question"], q["gold"], q["attr"]
            for label, thresh, currency in VARIANTS:
                hits = m.core.recall(question, k=k, consistency_threshold=thresh)
                ctx = format_tenet_context(hits, now, currency=currency)
                read_jobs.append((label, u, attr, question, gold, ctx))
    for _ds, m in built:
        m.close()

    print(f"  reading {len(read_jobs)} (question x variant) pairs "
          f"({read_workers} workers)...", flush=True)
    out = {v[0]: {"ok": 0, "n": 0} for v in VARIANTS}
    misses = []
    with ThreadPoolExecutor(max_workers=read_workers) as ex:
        for label, uu, attr, question, gold, pred in ex.map(_read_job, read_jobs):
            if not pred.strip():
                continue
            ok = score(pred, gold)
            out[label]["n"] += 1
            out[label]["ok"] += int(ok)
            if not ok:
                misses.append({"u": uu, "attr": attr, "variant": label,
                               "question": question, "gold": gold, "pred": pred})
    return out, misses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", default="2,8,32")
    ap.add_argument("--principals", type=int, default=10)
    ap.add_argument("--n-facts", type=int, default=5)
    ap.add_argument("--distractor-sessions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--build-workers", type=int, default=4)
    ap.add_argument("--read-workers", type=int, default=12,
                    help="reader concurrency (question x variant pairs are independent)")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    sweep = [int(x) for x in args.updates.split(",")]
    t0 = time.time()
    curve = {v[0]: {} for v in VARIANTS}
    all_misses = []
    for u in sweep:
        print(f"\n=== U={u} ===", flush=True)
        res, misses = run_u(u, args.principals, args.n_facts, args.distractor_sessions,
                            args.seed, args.k, args.build_workers, args.read_workers)
        all_misses.extend(misses)
        for label, _t, _c in VARIANTS:
            ok, n = res[label]["ok"], res[label]["n"]
            p = ok / n if n else 0.0
            lo, hi = wilson_ci(p, n)
            curve[label][u] = {"acc": p, "ci_lo": lo, "ci_hi": hi, "n": n}
            print(f"  {label:>15}: {100*p:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] n={n}")

    print(f"\n=== ChurnBench §9.1 fix A/B — {'/'.join(str(u) for u in sweep)} "
          f"(n={args.principals * args.n_facts}/point) ===")
    hdr = " | ".join(f"{v[0]:>26}" for v in VARIANTS)
    print(f"{'U':>4} | {hdr}")
    for u in sweep:
        row = " | ".join(
            f"{100*curve[v[0]][u]['acc']:5.1f}% [{100*curve[v[0]][u]['ci_lo']:4.1f},"
            f"{100*curve[v[0]][u]['ci_hi']:5.1f}]".rjust(26) for v in VARIANTS)
        print(f"{u:>4} | {row}")

    if args.dump:
        Path(args.dump).write_text("\n".join(json.dumps(m) for m in all_misses))
        print(f"\nwrote {args.dump}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": {"updates": sweep, "principals": args.principals,
                      "n_facts": args.n_facts, "distractor_sessions": args.distractor_sessions,
                      "seed": args.seed, "k": args.k,
                      "variants": [{"label": v[0], "consistency_threshold": v[1],
                                   "currency_context": v[2]} for v in VARIANTS]},
            "curve": curve,
        }, indent=2))
        print(f"wrote {args.out}")
    print(f"\nwall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
