"""ChurnBench — parametric high-churn memory benchmark (Build 1, hackathon final week).

Generalizes bench_horizon.py's single-city churn primitive (kept as-is, still wired to
`tenet bench run churn` / BENCHMARK.md §3) into a parametric generator: N keyed
attributes, each updated `updates_per_fact` times across a simulated, distractor-laden
conversation history, then asked for the CURRENT value. Deterministic substring scoring
(no LLM judge) — see `score()`, which reuses MAB's verbatim SubEM from bench_factcon.py.

Design choice (see scripts/bench_horizon.py vs here): a NEW script rather than extending
bench_horizon.py in place. bench_horizon.py is the existing single-fact primitive already
referenced by `tenet bench run churn` and BENCHMARK.md §3/README Fig. 1 — rewriting it
in place to add multi-fact + baseline-arm support would churn (no pun intended) an already
-cited reproduction command. This script is registered separately as `churnbench`.

Arms:
  rag      — top-k cosine over raw numbered lines, official-style extraction reader.
  tenet    — full pipeline: Tenet.ingest_session (real distiller) chunked over the
             simulated history, then core.recall() for the current belief.        [ours]
  mem0     — REUSED verbatim from bench_baselines.py's build_mem0 (per-fact LLM
             ADD/UPDATE ops at ingestion) — it's generic over (serial, text) facts,
             not entangled with MAB's fact-parsing beyond that tuple shape, which our
             generator already produces. Only the read-time prompt is swapped (see
             `_EXTRACT_SERIAL_PROMPT_CHURN` below): bench_baselines' answer_extract_serial
             frames the pool as "FICTIONAL / contradicts the real world" (needed for MAB's
             counterfactual facts); ChurnBench's facts are plausible, non-fictional
             profile updates, so that framing is dropped while keeping the same
             serial-priority extraction contract. Everything else (ingestion ops, ranking)
             is unmodified.
  hipporag — REUSED verbatim from bench_baselines.py's build_hipporag/hippo_rank
             (OpenIE triples + entity graph + Personalized PageRank blended with dense).
             Same read-prompt swap as mem0.
CAR and MemAgent arms are NOT reproduced here — the design doc names only Tenet/RAG/
Mem0/HippoRAG for this build; adding them would be scope creep.

Headline metric: churn half-life = largest updates_per_fact U (from the swept grid)
with accuracy >= 90%, per arm.

Usage:
  python scripts/bench_churn.py --principals 2 --updates 4,16 --qpc-smoke   # smoke, ~10 q
  python scripts/bench_churn.py --principals 10 --updates 2,4,8,16,32       # full grid (n=50/pt)
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
from tenet import Tenet  # noqa: E402
import bench_baselines as bb  # noqa: E402 — mem0/hipporag ingestion+ranking, reused
from bench_factcon import normalize_answer, substring_exact_match_score, wilson_ci  # noqa: E402

# data/ is a symlink to an external volume; CHURN_CACHE_DIR overrides it (e.g. when that
# volume is temporarily unreadable — see BENCHMARK.md §9 reproduction note).
CACHE = Path(os.environ["CHURN_CACHE_DIR"]) if os.environ.get("CHURN_CACHE_DIR") else (
    Path(__file__).resolve().parent.parent / "data" / "cache" / "churn")

# Static generator data (attribute pools, templates, distractors) lives in
# churnbench_data.py — split out to keep this file under the 500-line limit.
from churnbench_data import ATTR_ORDER, ATTR_SPECS, CHUNK_SIZE, DISTRACTORS  # noqa: E402


def _assert_no_substring_collisions(pool: list[str]) -> None:
    """No value in `pool` may be a substring of a different value in the same pool —
    a scoring-integrity precondition (a stale answer must never accidentally
    substring-match the current gold, or vice versa). Checked in test_churnbench.py."""
    low = [normalize_answer(v) for v in pool]
    for i, a in enumerate(low):
        for j, b in enumerate(low):
            if i != j and a in b:
                raise AssertionError(f"pool substring collision: {pool[i]!r} in {pool[j]!r}")


# --------------------------------------------------------------------------
# Deterministic dataset generator — NO LLM/network calls, fully reproducible
# from (updates_per_fact, n_facts, n_distractor_sessions, seed).
# --------------------------------------------------------------------------
def build_dataset(updates_per_fact: int, n_facts: int, n_distractor_sessions: int,
                   seed: int) -> dict:
    if n_facts > len(ATTR_SPECS):
        raise ValueError(f"n_facts={n_facts} > {len(ATTR_SPECS)} available attributes")
    attrs = ATTR_ORDER[:n_facts]

    streams: dict[str, list[str]] = {}
    gold: dict[str, str] = {}
    for attr in attrs:
        spec = ATTR_SPECS[attr]
        salt = zlib.crc32(attr.encode()) % 100_000       # deterministic, no hash() randomization
        rng = np.random.RandomState((seed * 97 + updates_per_fact + salt) % (2**31 - 1))
        pool = list(spec["pool"])
        rng.shuffle(pool)
        if updates_per_fact > len(pool):
            raise ValueError(f"updates_per_fact={updates_per_fact} > pool size {len(pool)} "
                              f"for attribute {attr!r}")
        chain = pool[:updates_per_fact]
        gold[attr] = chain[-1]
        streams[attr] = [spec["update"].format(v=v) for v in chain]

    drng = np.random.RandomState((seed * 97 + updates_per_fact + 999_983) % (2**31 - 1))
    dpool = list(DISTRACTORS)
    blocks: list[list[str]] = []
    for _ in range(n_distractor_sessions):
        drng.shuffle(dpool)
        blocks.append(list(dpool[:4]))

    # Random merge that preserves each attribute's intra-chain chronological order
    # (required for Tenet supersession correctness) while treating each distractor
    # block as an atomic "session" inserted at a random point — realistic interleaving.
    merge_rng = np.random.RandomState((seed * 97 + updates_per_fact + 555_001) % (2**31 - 1))
    pending = {a: 0 for a in attrs}
    block_queue = list(range(len(blocks)))
    merged: list[tuple[str, str | None]] = []   # (text, attr or None for distractor lines)
    while any(pending[a] < len(streams[a]) for a in attrs) or block_queue:
        candidates = [a for a in attrs if pending[a] < len(streams[a])]
        if block_queue:
            candidates = candidates + ["__block__"]
        choice = candidates[merge_rng.randint(len(candidates))]
        if choice == "__block__":
            merged.extend((line, None) for line in blocks[block_queue.pop(0)])
        else:
            merged.append((streams[choice][pending[choice]], choice))
            pending[choice] += 1

    facts = [(i + 1, t) for i, (t, _a) in enumerate(merged)]
    # Session chunking: cap at CHUNK_SIZE lines, but never put two updates to the SAME
    # attribute in one chunk. Two conflicting statements about the same attribute in a
    # single ingest_session call ask one distill() call to resolve a same-key conflict
    # itself — measured to sometimes pick the FIRST (not final) value, corrupting
    # supersession (see BENCHMARK.md §9 grounding note / the "known trap" in the task
    # brief). One update per attribute per chunk makes that ambiguity structurally
    # impossible: supersession across chunks is handled by store()'s own same-key
    # retirement, which is reliable.
    sessions: list[list[str]] = []
    cur: list[str] = []
    cur_attrs: set[str] = set()
    for text, a in merged:
        if len(cur) >= CHUNK_SIZE or (a is not None and a in cur_attrs):
            sessions.append(cur)
            cur, cur_attrs = [], set()
        cur.append(text)
        if a is not None:
            cur_attrs.add(a)
    if cur:
        sessions.append(cur)
    questions = [{"attr": a, "question": ATTR_SPECS[a]["question"], "gold": gold[a]}
                 for a in attrs]

    cfg = {"updates_per_fact": updates_per_fact, "n_facts": n_facts,
           "n_distractor_sessions": n_distractor_sessions, "seed": seed}
    digest = hashlib.sha256(json.dumps(
        {"config": cfg, "facts": facts, "questions": questions}, sort_keys=True
    ).encode()).hexdigest()
    return {"config": cfg, "facts": facts, "sessions": sessions, "questions": questions,
            "hash": digest}


def score(prediction: str, gold: str) -> bool:
    """Deterministic substring scoring — MAB's SubEM (normalize + substring), no LLM judge."""
    return substring_exact_match_score(prediction, gold)


# --------------------------------------------------------------------------
# Read-time extraction prompt for the reused mem0/hipporag arms: same
# serial-priority contract as bench_baselines' answer_extract_serial, minus the
# "fictional, contradicts the real world" framing (irrelevant here — ChurnBench
# facts are plausible, non-fictional profile updates; that framing would only
# risk confusing the reader, not help it).
# --------------------------------------------------------------------------
_EXTRACT_SERIAL_PROMPT_CHURN = (
    "The numbered notes below are from a user's conversation history. Each note starts "
    "with a serial number; when notes conflict, the note with the LARGEST serial number "
    "is the current, correct one. Find the note that answers the question (newest if "
    "several conflict) and COPY its value verbatim. Reply with ONLY the value — a short "
    "phrase, never a full sentence.\n\n[Notes]\n{pool}\n\nQuestion: {question}\nCopied value:")


def answer_extract_serial_churn(pool: str, question: str) -> str:
    return config.chat(
        [{"role": "user", "content":
          _EXTRACT_SERIAL_PROMPT_CHURN.format(pool=pool, question=question)}],
        qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"), max_tokens=48)


# rag/tenet arms: same natural "give the current value" reader as bench_knowledge_update.
def answer_natural(context: str, question: str) -> str:
    return config.chat(
        [{"role": "system", "content": "Answer using ONLY the memory provided. Give the "
          "user's CURRENT value. Reply with just the value, nothing else."},
         {"role": "user", "content": f"Memory:\n{context}\n\nQuestion: {question}"}],
        qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"), max_tokens=32)


def _fmt_age(seconds: float) -> str:
    days = seconds / 86400.0
    return "<1d ago" if days < 1 else f"{days:.0f}d ago"


def format_tenet_context(hits, now: float, currency: bool = False) -> str:
    """Render tenet's recalled memories as reader context (component 2 of the
    ChurnBench §9 fix).

    currency=False (default, unchanged behaviour): flat bullet list, no
    ordering/recency cue — this is what produced the §9 falsification.

    currency=True: split into "Current beliefs:" (distilled, current facts)
    then "Supporting raw context:" (raw verbatim turns), each dated. Tenet's
    store KNOWS which facts are current and when each was learned/valid
    (is_current, valid_at); surfacing that structure to the reader is
    product-faithful — the agent surface already dates/caveats facts the same
    way (tenet/agent.py's _format_memories + _fmt_age) — not benchmark-tuning.
    Other arms (RAG/mem0/hipporag) have no such metadata to present, so their
    prompts are deliberately left untouched — that asymmetry IS the design
    difference §9 measures.
    """
    if not currency:
        return "\n".join(f"- {h.text}" for h in hits)
    beliefs = [h for h in hits if h.kind != "raw"]
    raw = [h for h in hits if h.kind == "raw"]
    parts = []
    if beliefs:
        lines = "\n".join(f"- {h.text} ({_fmt_age(now - h.valid_at)})" for h in beliefs)
        parts.append(f"Current beliefs:\n{lines}")
    if raw:
        lines = "\n".join(f"- {h.text} ({_fmt_age(now - h.valid_at)})" for h in raw)
        parts.append(f"Supporting raw context:\n{lines}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Ingestion / index builders (cached under data/cache/churn/<seed>/)
# --------------------------------------------------------------------------
def _embed_lines(cache_dir: Path, cache_id: str, texts: list[str], embedder) -> np.ndarray:
    npz = cache_dir / f"{cache_id}.rag.npz"
    if npz.exists():
        return np.load(npz)["v"]
    v = np.array(embedder(texts))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, v=v)
    return v


def build_tenet_db(cache_dir: Path, cache_id: str, sessions: list[list[str]]):
    """Real product path: Tenet.ingest_session (distiller + bi-temporal store), chunked
    over the simulated history. Cached to disk; `now` is a controlled clock so cache
    reuse across runs doesn't skew recency decay (see module docstring / grounding read
    of bench_horizon.py's now=lambda pattern)."""
    dbp = cache_dir / f"{cache_id}.tenet.db"
    was_cached = dbp.exists()
    clock = [1_000_000.0]
    m = Tenet(dbp, now=lambda: clock[0])
    if not was_cached:
        total_lines = sum(len(chunk) for chunk in sessions)
        for i, chunk in enumerate(sessions):
            turns = [{"role": "User", "content": line} for line in chunk]
            m.ingest_session(turns, source=f"chunk{i}", valid_at=clock[0])
            clock[0] += 3600
        # Fail loud, not silent: raw-turn storage is unconditional except surprise-gating
        # (threshold 0.97, rarely triggered on diverse content), so a healthy ingest should
        # retain most input lines. A near-empty store after real input is the "weak
        # distiller / dead endpoint" trap (see module docstring) — surface it immediately
        # instead of letting every downstream question silently answer "Unknown".
        st = m.stats()
        stored = st["current"] + st["superseded"] + st["archived"]
        if total_lines > 0 and stored < 0.5 * total_lines:
            raise RuntimeError(
                f"tenet ingestion degraded: {stored} memories stored from {total_lines} "
                f"input lines ({dbp.name}) — check the distiller/embedder output before "
                f"trusting downstream tenet-arm answers")
    clock[0] += 3600  # advance "now" just past ingestion for recall-time decay
    return m


def _job(job):
    """One reader call. job = (arm, qi, kwargs-for-that-arm)."""
    arm, qi, question, gold, ctx_or_pool = job
    try:
        pred = (answer_natural(ctx_or_pool, question) if arm in ("rag", "tenet")
                else answer_extract_serial_churn(ctx_or_pool, question))
    except config.ProviderError:
        raise
    except Exception:  # noqa: BLE001 — transient reader hiccup, excluded not failed
        pred = ""
    return arm, qi, pred, gold


def run_dataset(ds: dict, arms: list[str], k: int, cache_dir: Path, host_embedder,
                 tenet_workers: int, consistency_threshold: float | None = None,
                 currency_context: bool = False) -> tuple[dict, list[dict]]:
    """Build every requested arm's index for one dataset, answer every question,
    return {arm: [(ok:bool|None, pred, gold, question)]} and miss records."""
    facts = ds["facts"]
    texts = [f"{s}. {t}" for s, t in facts]
    serials = [s for s, _ in facts]
    cache_id = "cfg" + hashlib.sha256(json.dumps(ds["config"], sort_keys=True)
                                       .encode()).hexdigest()[:12]
    line_vecs = _embed_lines(cache_dir, cache_id, texts, host_embedder)

    m_tenet = None
    if "tenet" in arms:
        m_tenet = build_tenet_db(cache_dir, cache_id, ds["sessions"])
    ing_mem0 = ing_hippo = None
    # bb.BCACHE is set ONCE per U-value by the caller (main()), not per-call here — it's
    # constant for the whole U-value's processing, so this function is safe to call
    # concurrently across principals (no shared-state mutation left in this function).
    if "mem0" in arms:
        ing_mem0 = bb.build_mem0(cache_id, facts, host_embedder)
    if "hipporag" in arms:
        # smaller batch than bench_baselines' MAB-tuned default (40): ChurnBench's
        # natural-language update sentences are wordier per-triple than MAB's templated
        # fact lines, so 40/batch can truncate max_tokens=1600 mid-JSON and lose the
        # whole batch to the fallback path (measured: 40/40 fallback -> the >50%
        # integrity raise). 15/batch stays well under the token cap.
        ing_hippo = bb.build_hipporag(cache_id, facts, host_embedder, batch_size=15)

    jobs = []
    for qi, q in enumerate(ds["questions"]):
        question, gold = q["question"], q["gold"]
        qv = np.asarray(host_embedder([question])[0])
        if "rag" in arms:
            top = sorted(np.argsort(-(line_vecs @ qv))[:k])
            pool = "\n".join(texts[i] for i in top)
            jobs.append(("rag", qi, question, gold, pool))
        if "tenet" in arms:
            hits = m_tenet.core.recall(question, k=k, consistency_threshold=consistency_threshold)
            ctx = format_tenet_context(hits, m_tenet.core._now(), currency=currency_context)
            jobs.append(("tenet", qi, question, gold, ctx))
        if "mem0" in arms:
            mems, mvecs = ing_mem0
            top = np.argsort(-(mvecs @ qv))[:k] if len(mems) else []
            pool = "\n".join(f'{mems[i]["serial"]}. {mems[i]["text"]}'
                             for i in sorted(top, key=lambda i: mems[i]["serial"]))
            jobs.append(("mem0", qi, question, gold, pool))
        if "hipporag" in arms:
            pool = bb.hippo_rank(ing_hippo, host_embedder, question, line_vecs, texts, serials, k)
            jobs.append(("hipporag", qi, question, gold, pool))

    results: dict[str, list] = {a: [None] * len(ds["questions"]) for a in arms}
    misses = []
    with ThreadPoolExecutor(max_workers=tenet_workers) as ex:
        for arm, qi, pred, gold in ex.map(_job, jobs):
            question = ds["questions"][qi]["question"]
            if not pred.strip():
                results[arm][qi] = None  # API/pipeline failure — excluded, never wrong
                continue
            ok = score(pred, gold)
            results[arm][qi] = ok
            if not ok:
                misses.append({"attr": ds["questions"][qi]["attr"], "arm": arm,
                               "question": question, "gold": gold, "pred": pred,
                               "config": ds["config"]})
    if m_tenet is not None:
        m_tenet.close()
    return results, misses


def churn_half_life(acc_by_u: dict[int, float], sweep: list[int]) -> int | str:
    passing = [u for u in sweep if acc_by_u.get(u, 0.0) >= 0.90]
    return max(passing) if passing else f"<{min(sweep)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", default="2,4,8,16,32", help="comma list of U values")
    ap.add_argument("--principals", type=int, default=10)
    ap.add_argument("--n-facts", type=int, default=5)
    ap.add_argument("--distractor-sessions", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--arms", default="tenet,rag,mem0,hipporag")
    ap.add_argument("--dump", default="", help="misses JSONL path")
    ap.add_argument("--out", default="", help="results JSON path")
    ap.add_argument("--workers", type=int, default=6, help="reader concurrency per principal dataset")
    ap.add_argument("--principal-workers", type=int, default=4,
                    help="how many principals' datasets to build+read concurrently "
                         "(total in-flight API calls ~= this * --workers)")
    ap.add_argument("--consistency-threshold", type=float, default=None,
                    help="tenet arm only: component 1 of the §9 fix (consistency.py). "
                         "None (default) = off, matching the §9-measured baseline.")
    ap.add_argument("--currency-context", action="store_true",
                    help="tenet arm only: component 2 of the §9 fix — structure the "
                         "reader context as 'Current beliefs:' / 'Supporting raw "
                         "context:' instead of a flat unordered list.")
    args = ap.parse_args()

    sweep = [int(x) for x in args.updates.split(",")]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    dump_f = open(args.dump, "w") if args.dump else None
    t0 = time.time()

    host = Tenet(CACHE / "emb_host.db")
    host_embedder = host.core.embed_batch

    acc: dict[str, dict[int, tuple[int, int]]] = {a: {} for a in arms}  # arm -> U -> (ok,n)
    for u in sweep:
        cache_dir = CACHE / str(args.seed) / f"u{u}_f{args.n_facts}_d{args.distractor_sessions}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        datasets = [build_dataset(u, args.n_facts, args.distractor_sessions,
                                  args.seed * 1000 + p) for p in range(args.principals)]
        print(f"\n=== U={u}: {args.principals} principals x {args.n_facts} facts "
              f"= {args.principals * args.n_facts} questions/arm ===", flush=True)

        # mem0/hipporag ingestion caches under this U-value's baselines/ dir. Set ONCE
        # here (constant for the rest of this U-value's processing) rather than per-call
        # inside run_dataset, so run_dataset has no shared-state mutation left and is
        # safe to run concurrently across principals below.
        bb.BCACHE = cache_dir / "baselines"

        ok_n = {a: [0, 0] for a in arms}
        with ThreadPoolExecutor(max_workers=args.principal_workers) as ex:
            for pi, (results, misses) in enumerate(
                    ex.map(lambda ds: run_dataset(ds, arms, args.k, cache_dir,
                                                  host_embedder, args.workers,
                                                  args.consistency_threshold,
                                                  args.currency_context), datasets)):
                for a in arms:
                    for r in results[a]:
                        if r is None:
                            continue
                        ok_n[a][1] += 1
                        ok_n[a][0] += int(r)
                if dump_f:
                    for rec in misses:
                        dump_f.write(json.dumps(rec) + "\n")
                    dump_f.flush()
                print(f"  [{pi+1}/{args.principals}] " +
                      " ".join(f"{a}={ok_n[a][0]}/{ok_n[a][1]}" for a in arms), flush=True)

        for a in arms:
            acc[a][u] = tuple(ok_n[a])

    host.close()
    if dump_f:
        dump_f.close()

    # ---- report ----
    print(f"\n=== ChurnBench: accuracy vs updates-per-fact (k={args.k}, "
          f"n_facts={args.n_facts}, principals={args.principals}) ===")
    hdr = " | ".join(f"{a.upper():>26}" for a in arms)
    print(f"{'U':>4} | {hdr}")
    curve = {a: {} for a in arms}
    for u in sweep:
        row = []
        for a in arms:
            ok, n = acc[a][u]
            p = ok / n if n else 0.0
            lo, hi = wilson_ci(p, n)
            curve[a][u] = {"acc": p, "ci_lo": lo, "ci_hi": hi, "n": n}
            row.append(f"{100*p:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] n={n:>3}")
        print(f"{u:>4} | {' | '.join(f'{x:>26}' for x in row)}")

    half_lives = {a: churn_half_life({u: curve[a][u]["acc"] for u in sweep}, sweep)
                 for a in arms}
    print("\nchurn half-life (largest U with acc>=90%):")
    for a in arms:
        print(f"  {a:>9}: {half_lives[a]}")

    if "tenet" in arms:
        base_arms = [a for a in arms if a != "tenet"]
        best_baseline = None
        if base_arms:
            def _hl_num(a):
                v = half_lives[a]
                return 0 if isinstance(v, str) else v
            best_baseline = max(base_arms, key=_hl_num)
            t_hl, b_hl = half_lives["tenet"], half_lives[best_baseline]
            t_num, b_num = _hl_num("tenet"), _hl_num(best_baseline)
            u8 = 8 if 8 in sweep else None
            gate_pass = (u8 is not None and t_num >= 2 * max(b_num, 1)
                        and curve["tenet"][u8]["ci_lo"] > curve[best_baseline][u8]["ci_hi"])
            all_flat = all(curve[a].get(sweep[-1], {}).get("acc", 0) >= 0.90 for a in arms)
            print(f"\nship gate (Tenet half-life >= 2x best baseline, CI-separated at U=8):")
            print(f"  tenet={t_hl}  best_baseline={best_baseline}({b_hl})")
            if all_flat:
                verdict = "FALSIFIED — every arm stays flat at U=8..%d; no structural claim" % sweep[-1]
            else:
                verdict = "PASS" if gate_pass else "PARTIAL/NOT MET"
            print(f"  verdict: {verdict}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "config": {"updates": sweep, "principals": args.principals,
                      "n_facts": args.n_facts, "distractor_sessions": args.distractor_sessions,
                      "seed": args.seed, "k": args.k, "arms": arms},
            "curve": curve, "half_life": half_lives,
        }, indent=2))
        print(f"\nwrote {out_path}")

    print(f"\nwall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
