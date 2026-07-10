"""Probe 1 — confidence-routed answering (hackathon final week, Build 2 graft).

Hypothesis: Tenet's per-fact confidence (p_valid, attached at recall by the world-model
layer) + the relevance margin (top1-top2 recall-score gap) can route READER spend per
query, without touching memory ranking. Three answer tiers:

  (a) EXTRACTIVE  — top hit is a current belief with high margin + high p_valid (and a
                    non-diffuse key pool): answer with the belief's value directly.
                    ZERO reader tokens (substring scoring rewards the belief text, which
                    already contains the current value).
  (b) CHEAP READER  — qwen3.6-flash over the same recalled pool (medium confidence).
  (c) FULL READER   — qwen3.7-plus over the same recalled pool (low confidence).

HARD INVARIANT (design_decision_churnbench_routing.md): routing gates compute spent
ANSWERING, never memory ranking. The recalled pool is identical across tiers.

The router lives HERE (a bench-side probe); we productize into tenet only if it wins.

Method (frugal): for every eval question we compute ALL THREE tier outcomes ONCE
(extractive pred is free; cheap + full readers are one call each), plus the recall
signals (margin, p_valid, n_keys). The threshold grid is then swept ANALYTICALLY over
those cached per-question outcomes — zero extra API calls per config. Baseline =
all-full-reader on the same questions. We report the (accuracy, reader-token) Pareto
per config with Wilson CIs.

PRE-REGISTERED GATE: ship-worthy = >=40% reader-token reduction at <=2pp accuracy loss.
If accuracy drops >5pp at any savings, calibration is insufficient for routing — that is
a publishable NEGATIVE, reported plainly.

Eval set: ChurnBench U in {2,8} (tenet arm, cached deterministic-gold questions) + FC
sh_6k n=20 smoke (~150 questions). Only the TENET arm is needed (routing is a Tenet
feature), so no RAG/mem0/hipporag stores are built here.

Usage:
  python scripts/bench_routing.py --churn-updates 2,8 --principals 10 --fc-limit 20
  python scripts/bench_routing.py --smoke        # tiny, ~30 q
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Route caches to a writable scratch dir BEFORE importing the bench modules — the repo
# data/ symlink points at a TCC-blocked external volume (see docs/BENCHMARK.md §9 note).
_SCRATCH = Path(os.environ.get(
    "ROUTING_CACHE_DIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "tenet_routing_cache")))
os.environ.setdefault("CHURN_CACHE_DIR", str(_SCRATCH / "churn"))

import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
import bench_churn as bc  # noqa: E402
import bench_factcon as fc  # noqa: E402

fc.CACHE = _SCRATCH / "factcon"
fc.CACHE.mkdir(parents=True, exist_ok=True)

FULL_MODEL = os.environ.get("ROUTING_FULL_MODEL", "qwen3.7-plus")
CHEAP_MODEL = os.environ.get("ROUTING_CHEAP_MODEL", "qwen3.6-flash")
# Illustrative price ratio for the SECONDARY cost-weighted metric only (Qwen list prices
# put the flash tier well below the plus tier). Raw-token accounting (the pre-registered
# gate) never uses this — it counts a token as a token. Labeled as an assumption.
COST_PER_TOK = {FULL_MODEL: 1.0, CHEAP_MODEL: 0.15}


# --------------------------------------------------------------------------
# Token-measured reader. Mirrors config.chat's qwen path but captures usage.
# Requires LLM_PROVIDER=qwen (the shipped backbone); token accounting is only
# meaningful with a single measurable provider.
# --------------------------------------------------------------------------
def _read(model: str, messages: list, max_tokens: int) -> tuple[str, int]:
    import time as _t
    cl = config.chat_client()
    kw = dict(model=model, messages=messages, temperature=0, max_tokens=max_tokens,
              extra_body={"enable_thinking": False})
    last = "no response"
    for attempt in range(5):
        try:
            r = cl.chat.completions.create(**kw)
            if r.choices:
                txt = (r.choices[0].message.content or "").strip()
                tok = int(r.usage.total_tokens) if getattr(r, "usage", None) else 0
                return txt, tok
        except Exception as e:  # noqa: BLE001
            msg = str(e); low = msg.lower()
            if any(m in low for m in config._PERMANENT_MARKERS):
                raise config.ProviderError(config.LLM_PROVIDER, model, msg) from e
            last = msg
            _t.sleep(1.5 * (attempt + 1) if ("429" in msg or "rate" in low) else 1)
    raise config.ProviderError(config.LLM_PROVIDER, model, last)


# Reader message builders — replicate the shipped prompts VERBATIM so the full-reader
# baseline equals the existing bench numbers.
def _churn_msgs(ctx: str, q: str) -> list:
    return [{"role": "system", "content": "Answer using ONLY the memory provided. Give the "
             "user's CURRENT value. Reply with just the value, nothing else."},
            {"role": "user", "content": f"Memory:\n{ctx}\n\nQuestion: {q}"}]


def _fc_msgs(pool: str, q: str) -> list:
    return [{"role": "user", "content": fc._EXTRACT_PROMPT.format(pool=pool, question=q)}]


# --------------------------------------------------------------------------
# Per-question record: all three tier outcomes + recall signals, computed once.
# --------------------------------------------------------------------------
@dataclass
class QRec:
    source: str
    margin: float
    p_valid: float
    n_keys: int
    top_is_fact: bool
    extr_ok: bool
    cheap_tok: int = 0
    cheap_ok: bool = False
    full_tok: int = 0
    full_ok: bool = False
    err: bool = False


def _signals(hits) -> tuple[float, float, int, bool, str]:
    """(margin, p_valid, n_keys, top_is_fact, extr_pred) from a recall hit list."""
    if not hits:
        return 0.0, 0.0, 0, False, ""
    top_is_fact = hits[0].kind != "raw"
    margin = float(hits[0].score) - (float(hits[1].score) if len(hits) > 1 else 0.0)
    p_valid = float(hits[0].confidence) if (top_is_fact and hits[0].confidence is not None) else 0.0
    n_keys = len({h.key for h in hits if h.kind != "raw" and h.key})
    extr_pred = hits[0].text if top_is_fact else ""
    return margin, p_valid, n_keys, top_is_fact, extr_pred


# --------------------------------------------------------------------------
# Build the eval question stubs (recall + signals + extractive), serial per store.
# Returns (stub_dicts, reader_jobs) where each job is (idx, tier, model, msgs, max_tok).
# --------------------------------------------------------------------------
def gather_churn(updates: list[int], principals: int, n_facts: int, distractors: int,
                 seed: int, k: int):
    recs, jobs, meta = [], [], []
    cache_root = Path(os.environ["CHURN_CACHE_DIR"])
    for u in updates:
        cache_dir = cache_root / str(seed) / f"u{u}_f{n_facts}_d{distractors}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for p in range(principals):
            ds = bc.build_dataset(u, n_facts, distractors, seed * 1000 + p)
            cache_id = "cfg" + __import__("hashlib").sha256(
                json.dumps(ds["config"], sort_keys=True).encode()).hexdigest()[:12]
            m = bc.build_tenet_db(cache_dir, cache_id, ds["sessions"])
            now = m.core._now()
            for q in ds["questions"]:
                hits = m.core.recall(q["question"], k=k)  # shipped defaults (consistency on)
                margin, p_valid, n_keys, tif, extr = _signals(hits)
                ctx = bc.format_tenet_context(hits, now, currency=True)
                gold = q["gold"]
                idx = len(recs)
                recs.append(QRec(f"churn_u{u}", margin, p_valid, n_keys, tif,
                                 extr_ok=bc.substring_exact_match_score(extr, gold)))
                meta.append(("churn", gold))
                jobs.append((idx, "full", FULL_MODEL, _churn_msgs(ctx, q["question"]), 32))
                jobs.append((idx, "cheap", CHEAP_MODEL, _churn_msgs(ctx, q["question"]), 32))
            m.close()
    return recs, jobs, meta


def gather_fc(cell: str, limit: int, k: int):
    from datasets import load_dataset
    cr = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution")
    recs, jobs, meta = [], [], []
    target = f"factconsolidation_{cell}"
    for ex in cr:
        if ex["metadata"]["source"] != target:
            continue
        facts = fc.parse_facts(ex["context"])
        import hashlib
        cache_id = "ctx" + hashlib.md5(ex["context"].encode()).hexdigest()[:12]
        m = fc.build_tenet(cache_id, facts)
        for q, gold in list(zip(ex["questions"], ex["answers"]))[:limit]:
            hits = m.core.recall(q, k=k)
            margin, p_valid, n_keys, tif, extr = _signals(hits)
            pool = "\n".join(f"{h.source}. {h.text}" for h in hits)
            idx = len(recs)
            recs.append(QRec(f"fc_{cell}", margin, p_valid, n_keys, tif,
                             extr_ok=fc.subem_max(extr, gold) if extr else False))
            meta.append(("fc", gold))
            jobs.append((idx, "full", FULL_MODEL, _fc_msgs(pool, q), 48))
            jobs.append((idx, "cheap", CHEAP_MODEL, _fc_msgs(pool, q), 48))
        m.close()
        break
    return recs, jobs, meta


def run_readers(recs, jobs, meta, workers: int):
    """Execute all cheap+full reader jobs concurrently, fill preds/tokens/ok on recs."""
    def _do(job):
        idx, tier, model, msgs, mt = job
        try:
            txt, tok = _read(model, msgs, mt)
        except config.ProviderError:
            raise
        except Exception:  # noqa: BLE001 — transient hiccup: excluded, not scored wrong
            txt, tok = "", 0
        return idx, tier, txt, tok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, tier, txt, tok in ex.map(_do, jobs):
            kind, gold = meta[idx]
            ok = (bc.substring_exact_match_score(txt, gold) if kind == "churn"
                  else fc.subem_max(txt, gold)) if txt.strip() else False
            r = recs[idx]
            if tier == "full":
                r.full_tok, r.full_ok = tok, ok
                if not txt.strip():
                    r.err = True
            else:
                r.cheap_tok, r.cheap_ok = tok, ok
                if not txt.strip():
                    r.err = True


# --------------------------------------------------------------------------
# Threshold sweep (analytic, no API) + reporting.
# --------------------------------------------------------------------------
def route(r: QRec, m_hi: float, pv: float, kt: int, m_lo: float) -> str:
    if r.top_is_fact and r.margin >= m_hi and r.p_valid >= pv and r.n_keys <= kt:
        return "extractive"
    if r.margin >= m_lo:
        return "cheap"
    return "full"


def eval_config(recs: list[QRec], cfg: tuple) -> dict:
    m_hi, pv, kt, m_lo = cfg
    n = ok = raw = 0.0
    cost = 0.0
    tiers = {"extractive": 0, "cheap": 0, "full": 0}
    for r in recs:
        if r.err:
            continue
        t = route(r, m_hi, pv, kt, m_lo)
        tiers[t] += 1
        n += 1
        if t == "extractive":
            ok += r.extr_ok
        elif t == "cheap":
            ok += r.cheap_ok; raw += r.cheap_tok; cost += r.cheap_tok * COST_PER_TOK[CHEAP_MODEL]
        else:
            ok += r.full_ok; raw += r.full_tok; cost += r.full_tok * COST_PER_TOK[FULL_MODEL]
    return {"cfg": cfg, "n": int(n), "acc": ok / n if n else 0.0,
            "raw_tok": int(raw), "cost": cost, "tiers": tiers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--churn-updates", default="2,8")
    ap.add_argument("--principals", type=int, default=10)
    ap.add_argument("--n-facts", type=int, default=5)
    ap.add_argument("--distractor-sessions", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fc-cell", default="sh_6k")
    ap.add_argument("--fc-limit", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.principals, args.churn_updates, args.fc_limit = 2, "2,8", 6

    assert config.LLM_PROVIDER == "qwen", "token accounting needs LLM_PROVIDER=qwen"
    t0 = time.time()
    updates = [int(x) for x in args.churn_updates.split(",")]

    print(f"gathering churn (U={updates}, {args.principals} principals) ...", flush=True)
    recs, jobs, meta = gather_churn(updates, args.principals, args.n_facts,
                                    args.distractor_sessions, args.seed, args.k)
    print(f"gathering FC {args.fc_cell} (n={args.fc_limit}) ...", flush=True)
    r2, j2, m2 = gather_fc(args.fc_cell, args.fc_limit, args.k)
    off = len(recs)
    recs += r2; meta += m2
    jobs += [(idx + off, t, mdl, msgs, mt) for (idx, t, mdl, msgs, mt) in j2]

    print(f"{len(recs)} questions, {len(jobs)} reader calls — running ...", flush=True)
    run_readers(recs, jobs, meta, args.workers)
    scored = [r for r in recs if not r.err]
    n_err = len(recs) - len(scored)
    print(f"scored={len(scored)} excluded(API-fail)={n_err}  wall={time.time()-t0:.0f}s")

    # ---- signal diagnostics ----
    def _pct(vals):
        a = np.array(vals, float)
        return f"min={a.min():.3f} p50={np.median(a):.3f} p90={np.quantile(a,.9):.3f} max={a.max():.3f}"
    print("\n=== recall-signal distributions (scored qs) ===")
    print(f"  margin : {_pct([r.margin for r in scored])}")
    print(f"  p_valid: {_pct([r.p_valid for r in scored])}")
    print(f"  n_keys : {_pct([r.n_keys for r in scored])}")
    print(f"  extractive-if-taken accuracy: "
          f"{sum(r.extr_ok for r in scored if r.top_is_fact)}/"
          f"{sum(1 for r in scored if r.top_is_fact)} (top-is-fact qs)")

    # ---- baseline: all-full-reader ----
    base_n = len(scored)
    base_ok = sum(r.full_ok for r in scored)
    base_tok = sum(r.full_tok for r in scored)
    base_acc = base_ok / base_n if base_n else 0.0
    blo, bhi = fc.wilson_ci(base_acc, base_n)
    print(f"\n=== BASELINE (all full reader, {FULL_MODEL}) ===")
    print(f"  acc={100*base_acc:.1f}% [{100*blo:.1f},{100*bhi:.1f}]  raw_tok={base_tok}  n={base_n}")

    # ---- threshold grid ----
    # Grid weighted toward the axes that actually carry signal (smoke showed margins are
    # tightly compressed, so the low-margin band + the p_valid axis are where routing
    # decisions live). kt=99 = key-count gate effectively off (n_keys was non-discriminative
    # for ChurnBench — every principal has exactly n_facts keys, all pulled into top-k);
    # kt=6 kept as a probe of whether limiting to few-key pools helps.
    grid = [(m_hi, pv, kt, m_lo)
            for m_hi in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
            for pv in (0.0, 0.5, 0.7, 0.9, 0.95, 0.99)
            for kt in (99, 6)
            for m_lo in (0.0,)]
    results = [eval_config(scored, c) for c in grid]

    # Pareto frontier on (acc high, raw_tok low)
    def dominated(a, b):  # b dominates a
        return b["acc"] >= a["acc"] and b["raw_tok"] <= a["raw_tok"] and (
            b["acc"] > a["acc"] or b["raw_tok"] < a["raw_tok"])
    pareto = [r for r in results if not any(dominated(r, o) for o in results if o is not r)]
    pareto.sort(key=lambda r: r["raw_tok"])

    print(f"\n=== Pareto frontier (accuracy vs raw reader tokens) — {len(pareto)} configs ===")
    print(f"{'m_hi':>5} {'pv':>4} {'kt':>3} {'m_lo':>5} | {'acc [95% CI]':>20} "
          f"{'Δacc':>6} | {'raw_tok':>8} {'save%':>6} | extr/cheap/full | cost%")
    for r in pareto:
        m_hi, pv, kt, m_lo = r["cfg"]
        lo, hi = fc.wilson_ci(r["acc"], r["n"])
        save = 100 * (1 - r["raw_tok"] / base_tok) if base_tok else 0.0
        dacc = 100 * (r["acc"] - base_acc)
        base_cost = base_tok * COST_PER_TOK[FULL_MODEL]
        costp = 100 * (1 - r["cost"] / base_cost) if base_cost else 0.0
        tg = r["tiers"]
        print(f"{m_hi:>5} {pv:>4} {kt:>3} {m_lo:>5} | {100*r['acc']:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] "
              f"{dacc:+5.1f} | {r['raw_tok']:>8} {save:5.1f} | "
              f"{tg['extractive']:>3}/{tg['cheap']:>3}/{tg['full']:>3} | {costp:5.1f}")

    # ---- gate verdict ----
    gate_ok = [r for r in results
               if base_tok and (1 - r["raw_tok"] / base_tok) >= 0.40
               and (base_acc - r["acc"]) <= 0.02]
    any_savings_bad = any(base_tok and r["raw_tok"] < base_tok and (base_acc - r["acc"]) > 0.05
                          for r in results)
    print("\n=== PRE-REGISTERED GATE (>=40% token cut at <=2pp acc loss) ===")
    if gate_ok:
        best = max(gate_ok, key=lambda r: (1 - r["raw_tok"] / base_tok))
        save = 100 * (1 - best["raw_tok"] / base_tok)
        print(f"  PASS — best qualifying config {best['cfg']}: {save:.1f}% token cut, "
              f"Δacc={100*(best['acc']-base_acc):+.1f}pp  (tiers {best['tiers']})")
    else:
        # best config with <=2pp loss regardless of savings, to characterize the frontier
        near = [r for r in results if (base_acc - r["acc"]) <= 0.02 and r["raw_tok"] < base_tok]
        best_near = max(near, key=lambda r: (1 - r["raw_tok"] / base_tok)) if near else None
        if best_near:
            s = 100 * (1 - best_near["raw_tok"] / base_tok)
            print(f"  NOT MET — max token cut at <=2pp loss is {s:.1f}% "
                  f"({best_near['cfg']}, Δacc={100*(best_near['acc']-base_acc):+.1f}pp)")
        else:
            print("  NOT MET — no config saves tokens within 2pp of baseline accuracy")
    if any_savings_bad:
        print("  NEGATIVE FINDING: some savings configs drop accuracy >5pp — "
              "confidence calibration is INSUFFICIENT to route safely at those thresholds.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "baseline": {"acc": base_acc, "raw_tok": base_tok, "n": base_n,
                         "ci": [blo, bhi]},
            "configs": [{**r, "tiers": r["tiers"]} for r in results],
            "pareto": [r["cfg"] for r in pareto],
            "excluded": n_err,
        }, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
