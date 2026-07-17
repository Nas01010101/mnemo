"""ReMe head-to-head — Tenet vs Alibaba Tongyi Lab's ReMe (agentscope-ai/ReMe,
arXiv:2512.10696) on the FULL LongMemEval_S haystack, matched Qwen backbone.

Four arms, ALL scored by the SAME qa_answer/qa_judge (scripts/lme_recall.py) so
the memory system is the only variable (design doc §2.1):
  blind : raw question, zero context                          [no-memory control]
  rag   : top-k cosine over raw turns          (lme_recall.eval_instance's rag arm)
  reme  : ReMe's own ingest (auto_memory) + retrieve (bm25_search/search), driven
          as an isolated-venv black box exactly like bench_mem0_h2h.py drove real
          mem0ai — see scripts/reme_h2h_config.yaml for the model/endpoint pins.
  tenet : distill -> bi-temporal store -> dual-pool recall  (lme_recall.eval_instance)

Dataset: data/lme/longmemeval_s.json (full haystack, n=500 official). NEVER
longmemeval_oracle.json — that's the pre-filtered evidence-only variant and
silently downgrades the task; refuse_if_oracle() hard-refuses it.

Design doc: see the "Tenet vs. Qwen/Alibaba In-House Memory" design note, Part 2
(§2.1 protocol, §2.3 feasibility) — carried in this session's handoff, not the repo.

Usage (from repo root, once DashScope quota is confirmed):
  set -a; . ./.env; set +a
  python scripts/bench_reme_h2h.py --n 100 --arms blind,rag,reme,tenet \\
      --reme-venv /path/to/reme-venv --budget-cap 15

Dry run (no network, no reme-venv needed — stubbed LLM client + stubbed ReMe
subprocess call, exercises the full pipeline end-to-end):
  EMBED_PROVIDER=local python scripts/bench_reme_h2h.py --n 2 --dry-run

Token/cost projection (NO API calls — computed from the dataset + the measured
rag/tenet context ratios in docs/lme_qwen_n100_result.txt):
  python scripts/bench_reme_h2h.py --project 25,50,100
"""
from __future__ import annotations

import argparse, json, os, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tenet import config  # noqa: E402
from tenet import Tenet  # noqa: E402
from bench_factcon import wilson_ci  # noqa: E402
from reme_h2h_driver import reme_env as _reme_env, reme_ingest_and_search  # noqa: E402
# Reusing lme_recall's exact machinery — the memory system is the only variable
# that should differ between arms (design doc §2.1). `_ANS_SYS`/`_JUDGE_SYS` are
# module-private by convention only; we import them purely to size cost
# estimates on the SAME prompts eval_instance already sends, not to re-send them.
from lme_recall import (  # noqa: E402
    DATA, flatten, eval_instance, qa_answer, qa_judge, ANSWER_MODEL,
    _ANS_SYS, _JUDGE_SYS,
)

ARMS = ("blind", "rag", "reme", "tenet")
DISTILL_MODEL_TENET = config.get("QWEN_DISTILL_MODEL", "qwen3.6-flash")
DISTILL_MODEL_REME = os.environ.get("LME_DISTILL_MODEL", "qwen-flash")
EMBED_MODEL = config.get("QWEN_EMBED_MODEL", "text-embedding-v4")
REME_CONFIG_DEFAULT = Path(__file__).resolve().parent / "reme_h2h_config.yaml"

# ---------------------------------------------------------------------------
# Oracle guard — MUST run on data/lme/longmemeval_s.json (full haystack), never
# the pre-filtered evidence-only longmemeval_oracle.json (see design doc §2.1
# caveat: ReMe's own eval script defaults to oracle.json, which would silently
# downgrade the task if we copied that default instead of catching it).
# ---------------------------------------------------------------------------


def guard_not_oracle(path) -> None:
    if "oracle" in Path(path).name.lower():
        raise ValueError(
            f"refusing to run on {path} — this is the evidence-only LongMemEval "
            "variant (pre-filtered to ~2 sessions/question). Use longmemeval_s.json "
            "(full haystack) or the comparison is silently easier than Tenet's own "
            "reported numbers. See design doc §2.1."
        )


# ---------------------------------------------------------------------------
# Cost tracking — chars/4 token heuristic (design doc §2.3 sanctions this for
# projections; here it also backs the LIVE --budget-cap hard-stop, sized off the
# actual prompt/context/prediction text lengths we already have in hand, not a
# second network round-trip to ask the provider for usage).
# ---------------------------------------------------------------------------

# $ per 1M tokens (in, out) — DashScope intl list prices, cost-matched flash
# tiers per design doc §2.1 ("Alibaba's two cheapest models... deliberately not
# picking a weaker extractor for ReMe"). Embeddings are input-only pricing.
PRICING = {
    "qwen3.7-plus": (0.25, 1.50),        # reader + judge, ALL arms
    "qwen3.6-flash": (0.05, 0.40),       # Tenet distiller
    "qwen-flash": (0.05, 0.40),          # ReMe distiller (our cost-matched pin)
    "text-embedding-v4": (0.05, 0.0),    # RAG/Tenet embedder; reme's default
                                          # BM25 retrieval issues ZERO of these.
}


def est_tokens(chars: float) -> float:
    return chars / 4.0


def est_cost(model: str, in_chars: float, out_chars: float = 0.0) -> float:
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    return est_tokens(in_chars) / 1e6 * in_price + est_tokens(out_chars) / 1e6 * out_price


class CostTracker:
    """Cumulative $ spend by (arm, model) — "per-arm" is load-bearing: it's the
    only way to see which arm is actually expensive, not just which model is.
    Optional hard-stop cap on the grand total."""

    def __init__(self, cap: float | None):
        self.cap = cap
        self.usd: dict[tuple[str, str], float] = {}
        self.tok_in: dict[tuple[str, str], float] = {}
        self.tok_out: dict[tuple[str, str], float] = {}

    def add(self, arm: str, model: str, in_chars: float, out_chars: float = 0.0) -> float:
        key = (arm, model)
        c = est_cost(model, in_chars, out_chars)
        self.usd[key] = self.usd.get(key, 0.0) + c
        self.tok_in[key] = self.tok_in.get(key, 0.0) + est_tokens(in_chars)
        self.tok_out[key] = self.tok_out.get(key, 0.0) + est_tokens(out_chars)
        return c

    def total(self, arm: str | None = None) -> float:
        return sum(v for (a, _), v in self.usd.items() if arm is None or a == arm)

    def over_cap(self) -> bool:
        return self.cap is not None and self.total() >= self.cap

    def report(self) -> str:
        lines = []
        for arm in sorted({a for a, _ in self.usd}):
            lines.append(f"  [{arm}] subtotal=${self.total(arm):.3f}")
            for (a, m), usd in sorted(self.usd.items()):
                if a == arm:
                    lines.append(f"    {m:20s} in={self.tok_in[(a,m)]:>10,.0f} tok  "
                                 f"out={self.tok_out[(a,m)]:>8,.0f} tok  ${usd:.3f}")
        lines.append(f"  {'TOTAL':20s} ${self.total():.3f}")
        return "\n".join(lines)


def _track_answer_cost(cost: CostTracker, arm: str, model: str, ctx_chars: float,
                        inst: dict, pred: str) -> None:
    cost.add(arm, model, len(_ANS_SYS) + ctx_chars + len(inst["question"]), len(pred))


def _track_judge_cost(cost: CostTracker, arm: str, model: str, inst: dict, pred: str) -> None:
    cost.add(arm, model, len(_JUDGE_SYS) + len(inst["question"]) +
             len(str(inst["answer"])) + len(pred), 2)


def qa_score(context: str, inst: dict, cost: CostTracker, arm: str, model: str = ANSWER_MODEL):
    """qa_answer + qa_judge on `context`; track approx cost; (ok, pred) or
    (None, pred_or_None) on API failure — an API failure must never count as a
    wrong answer for any arm (same contract as lme_recall.eval_instance). No
    judge cost is tracked when no judge call happened (empty prediction)."""
    pred = qa_answer(context, inst["question"], inst["question_date"])
    _track_answer_cost(cost, arm, model, len(context), inst, pred)
    if not pred.strip():
        return None, None
    ok = qa_judge(inst["question"], inst["answer"], pred)
    _track_judge_cost(cost, arm, model, inst, pred)
    return ok, pred


# ---------------------------------------------------------------------------
# Per-instance, all-arms evaluation
# ---------------------------------------------------------------------------


def eval_all_arms(inst: dict, *, k: int, embedder, arms: tuple[str, ...],
                   cost: CostTracker, cache_dir: Path | None, type_budgets: bool,
                   reme_kwargs: dict, dry_run: bool, dry_script: list | None) -> dict:
    row = {"qid": inst["question_id"], "type": inst["question_type"]}
    turns = flatten(inst)
    full_chars = sum(len(t) for _, t in turns)

    if "rag" in arms or "tenet" in arms:
        r = eval_instance(inst, k, embedder, qa=True, do_full=False,
                          cache_dir=cache_dir, type_budgets=type_budgets)
        if r.get("qa_error"):
            row["qa_error"] = True
            return row
        if "rag" in arms:
            row["rag_ok"], row["rag_pred"] = r["rag_qa"], r["rag_pred"]
            _track_answer_cost(cost, "rag", ANSWER_MODEL, r["rag_ctx_chars"], inst, r["rag_pred"])
            _track_judge_cost(cost, "rag", ANSWER_MODEL, inst, r["rag_pred"])
        if "tenet" in arms:
            row["tenet_ok"], row["tenet_pred"] = r["tenet_qa"], r["tenet_pred"]
            _track_answer_cost(cost, "tenet", ANSWER_MODEL, r["tenet_ctx_chars"], inst, r["tenet_pred"])
            _track_judge_cost(cost, "tenet", ANSWER_MODEL, inst, r["tenet_pred"])
            # Tenet distills every session in the full haystack once (design
            # doc §2.3): one qwen3.6-flash call/session, input ~= that
            # session's raw text. Output is unmeasured without executing —
            # assumed ~12% of input chars (compact fact/JSON extraction).
            cost.add("tenet", DISTILL_MODEL_TENET, full_chars, full_chars * 0.12)

    if "blind" in arms:
        ok, pred = qa_score("", inst, cost, "blind")
        row["blind_ok"], row["blind_pred"] = ok, pred

    if "reme" in arms:
        try:
            context = reme_ingest_and_search(
                inst, dry_run=dry_run, dry_script=dry_script, **reme_kwargs, k=k)
        except Exception as e:  # noqa: BLE001 — one flaky subprocess must not
            # kill an unattended multi-hour run. Same contract as qa_score:
            # an infrastructure failure is EXCLUDED (ok=None), never scored
            # wrong. The row still writes, so this qid won't re-try the reme
            # arm on resume — visible in the JSONL via reme_error.
            print(f"  reme arm failed for {inst['question_id']}: {str(e)[:200]}",
                  flush=True)
            row["reme_ok"], row["reme_pred"] = None, None
            row["reme_error"] = str(e)[:500]
        else:
            # ReMe's auto_memory distills every session too (reme/steps/benchmark/
            # lme/auto_memory.py) — same full-haystack input volume as Tenet's
            # distiller, at the qwen-flash tier we pinned in reme_h2h_config.yaml.
            cost.add("reme", DISTILL_MODEL_REME, full_chars, full_chars * 0.12)
            ok, pred = qa_score(context, inst, cost, "reme", model=ANSWER_MODEL)
            row["reme_ok"], row["reme_pred"] = ok, pred

    return row


# ---------------------------------------------------------------------------
# Stats — Wilson CI per arm, McNemar pairwise (tenet-vs-reme is the primary win
# condition per design doc §2.1; tenet-vs-blind and reme-vs-blind substantiate
# that each arm's improvement over the no-memory control is itself real).
# ---------------------------------------------------------------------------


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    """pairs = [(a_ok, b_ok), ...] over jointly-scored questions. Returns
    (a_only_right, b_only_right, two-sided exact p-value)."""
    from math import comb
    n_a = sum(a and not b for a, b in pairs)
    n_b = sum(b and not a for a, b in pairs)
    nd = n_a + n_b
    if nd == 0:
        return n_a, n_b, 1.0
    p = sum(comb(nd, x) for x in range(min(n_a, n_b) + 1)) / 2 ** nd * 2
    return n_a, n_b, min(p, 1.0)


def summarize(rows: list[dict], arms: tuple[str, ...]) -> None:
    scored = {a: [(r[f"{a}_ok"], r) for r in rows
                  if not r.get("qa_error") and r.get(f"{a}_ok") is not None]
              for a in arms}
    print(f"\n=== ReMe head-to-head — LongMemEval_S full haystack "
          f"(n={len(rows)}, arms={','.join(arms)}, qwen3.7-plus reader+judge) ===")
    accs = {}
    for a in arms:
        oks = [ok for ok, _ in scored[a]]
        n = len(oks)
        acc = sum(oks) / max(n, 1)
        lo, hi = wilson_ci(acc, max(n, 1))
        accs[a] = acc
        print(f"{a:>6} | {100*acc:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] n={n:3d}")
    if "blind" in accs:
        for a in arms:
            if a != "blind":
                print(f"  Δ({a} vs blind) = {100*(accs[a]-accs['blind']):+.1f}pp")

    def paired(a: str, b: str):
        qa = {r["qid"]: ok for ok, r in scored.get(a, [])}
        qb = {r["qid"]: ok for ok, r in scored.get(b, [])}
        shared = sorted(set(qa) & set(qb))
        return [(qa[q], qb[q]) for q in shared]

    for a, b in (("tenet", "reme"), ("tenet", "blind"), ("reme", "blind")):
        if a in arms and b in arms:
            p = paired(a, b)
            if not p:
                continue
            n_a, n_b, pval = mcnemar(p)
            print(f"McNemar {a}-vs-{b}: {a}-only-right={n_a}  {b}-only-right={n_b}  "
                  f"p={pval:.4f}  (paired n={len(p)})")


# ---------------------------------------------------------------------------
# Projections — NO API calls, computed straight from the dataset. Uses the
# MEASURED rag_ctx/full_ctx and tenet_ctx/full_ctx ratios from the real n=100
# run in docs/lme_qwen_n100_result.txt (rag: 7,689/505,619=1.521%; tenet:
# 7,643/505,619=1.512%) as the reader-context sizing model; ReMe's context size
# is UNMEASURED (flagged) and assumed to sit at the same order as RAG (both are
# top-k lexical/vector retrieval over a comparable pool size).
# ---------------------------------------------------------------------------

_MEASURED_RAG_CTX_RATIO = 7689 / 505619
_MEASURED_TENET_CTX_RATIO = 7643 / 505619
_ASSUMED_REME_CTX_RATIO = _MEASURED_RAG_CTX_RATIO  # unmeasured — flagged in report
_ASSUMED_READER_OUT_CHARS = 600     # chain-of-note answer, ~150 tok — unmeasured
_ASSUMED_DISTILL_OUT_FRAC = 0.12    # distiller output as a fraction of input chars


def project(ns: list[int], seed: int = 0) -> None:
    import random
    guard_not_oracle(DATA)
    data = [d for d in json.load(open(DATA)) if not d["question_id"].endswith("_abs")]
    random.Random(seed).shuffle(data)

    print("=== Token/cost projection — NO API calls, computed from the dataset ===")
    print(f"(source: {DATA.name}, {len(data)} instances total; reader-context sizing "
          f"uses the measured n=100 rag/tenet ratios from docs/lme_qwen_n100_result.txt; "
          f"ReMe context size and all distiller/reader OUTPUT sizes are UNMEASURED "
          f"assumptions, flagged below)\n")

    for n in ns:
        sample = data[:n]
        full_chars = [sum(len(t) for _, t in flatten(d)) for d in sample]
        total_full = sum(full_chars)

        tracker = CostTracker(cap=None)
        for fc, d in zip(full_chars, sample):
            q_chars = len(d["question"])
            a_chars = len(str(d["answer"]))
            # blind: reader sees only the question, no context.
            tracker.add("blind", ANSWER_MODEL, len(_ANS_SYS) + q_chars, _ASSUMED_READER_OUT_CHARS)
            tracker.add("blind", ANSWER_MODEL, len(_JUDGE_SYS) + q_chars + a_chars +
                       _ASSUMED_READER_OUT_CHARS, 2)
            # rag / reme: reader sees a top-k context sized off the measured ratio.
            for arm, ratio in (("rag", _MEASURED_RAG_CTX_RATIO), ("reme", _ASSUMED_REME_CTX_RATIO)):
                ctx = fc * ratio
                tracker.add(arm, ANSWER_MODEL, len(_ANS_SYS) + ctx + q_chars, _ASSUMED_READER_OUT_CHARS)
                tracker.add(arm, ANSWER_MODEL, len(_JUDGE_SYS) + q_chars + a_chars +
                           _ASSUMED_READER_OUT_CHARS, 2)
            tracker.add("reme", DISTILL_MODEL_REME, fc, fc * _ASSUMED_DISTILL_OUT_FRAC)
            tracker.add("rag", EMBED_MODEL, fc, 0.0)   # rag's embedding pass
            # tenet: reader sees the smaller tenet-budget context + its own distiller pass.
            ctx = fc * _MEASURED_TENET_CTX_RATIO
            tracker.add("tenet", ANSWER_MODEL, len(_ANS_SYS) + ctx + q_chars, _ASSUMED_READER_OUT_CHARS)
            tracker.add("tenet", ANSWER_MODEL, len(_JUDGE_SYS) + q_chars + a_chars +
                       _ASSUMED_READER_OUT_CHARS, 2)
            tracker.add("tenet", DISTILL_MODEL_TENET, fc, fc * _ASSUMED_DISTILL_OUT_FRAC)
            # tenet's embedding pass (raw turns + distilled facts, qwen text-embedding-v4);
            # reme's BM25-default retrieval issues zero embedding calls.
            tracker.add("tenet", EMBED_MODEL, fc, 0.0)

        print(f"--- n={n} (full haystack ≈{total_full:,} chars, "
              f"≈{est_tokens(total_full):,.0f} tok total) ---")
        print(tracker.report())
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(args) -> None:
    guard_not_oracle(DATA)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {ARMS}")

    dry_run = args.dry_run
    if not dry_run and "reme" in arms and not args.reme_venv:
        raise SystemExit("--reme-venv is required for the reme arm outside --dry-run")

    if dry_run:
        _install_dry_run_stubs()

    import random
    data = [d for d in json.load(open(DATA)) if not d["question_id"].endswith("_abs")]
    if args.type:
        data = [d for d in data if d["question_type"] == args.type]
    random.Random(args.seed).shuffle(data)
    data = data[:args.n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    # qa_error rows are transient API failures with NO arm
                    # results; reme_error rows have every arm EXCEPT reme.
                    # Leaving either in `done` would exclude those questions
                    # (or that arm) forever on every resume — and if failures
                    # correlate with harder questions, that biases the reme
                    # arm downward. Re-run both.
                    if not d.get("qa_error") and not d.get("reme_error"):
                        done.add(d["qid"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"resume: {len(done)} question(s) already done in {out_path}")

    core = Tenet(Path(tempfile.mkdtemp()) / "emb.db")
    embedder = core.core.embed_batch
    cache_dir = Path(args.cache) if args.cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = Path(args.workspace_root or tempfile.mkdtemp(prefix="reme_h2h_ws_"))
    workspace_root.mkdir(parents=True, exist_ok=True)

    reme_bin = str(Path(args.reme_venv) / "bin" / "reme") if args.reme_venv else ""
    reme_kwargs = dict(reme_bin=reme_bin, reme_config=args.reme_config,
                       workspace_root=workspace_root,
                       env=_reme_env(embed_model=EMBED_MODEL, answer_model=ANSWER_MODEL,
                                     distill_model=DISTILL_MODEL_REME),
                       retrieval_job=args.retrieval_job)

    cost = CostTracker(cap=args.budget_cap)
    rows = []
    out_f = open(out_path, "a")
    t0 = time.time()
    stopped_early = False
    for i, inst in enumerate(data):
        if inst["question_id"] in done:
            continue
        row = eval_all_arms(inst, k=args.k, embedder=embedder, arms=arms, cost=cost,
                            cache_dir=cache_dir, type_budgets=args.type_budgets,
                            reme_kwargs=reme_kwargs, dry_run=dry_run,
                            dry_script=list(args.dry_script) if dry_run else None)
        rows.append(row)
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        print(f"[{i+1}/{len(data)}] {row.get('type','')[:18]:18s} "
              f"{'QA ERROR' if row.get('qa_error') else ''} "
              f"spend=${cost.total():.3f}", flush=True)
        if cost.over_cap():
            print(f"\nBUDGET CAP HIT (${args.budget_cap}) — stopping after {i+1} question(s).")
            stopped_early = True
            break
    out_f.close()

    for db_close in (core,):
        db_close.close()

    # replay the full resumed set (this run's rows + any prior completed rows)
    # for the summary, so a resumed run reports over everything done so far.
    all_rows = rows
    if done:
        # Last row per qid wins: a re-run after a qa_error leaves both the
        # error row and the fresh row in the file; only the fresh one counts.
        by_qid = {}
        for line in out_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                by_qid[d["qid"]] = d
        all_rows = list(by_qid.values())
    summarize(all_rows, arms)
    print(f"\nspend this run:\n{cost.report()}")
    print(f"wall={time.time()-t0:.0f}s" + ("  (stopped early: budget cap)" if stopped_early else ""))


def _install_dry_run_stubs() -> None:
    """No-network stand-ins for --dry-run: a contextual fake chat client (JSON
    facts for distill(), 'ANSWER: ...' for qa_answer, 'no' for qa_judge) and a
    deterministic fake embedder. Same monkeypatch pattern as
    scripts/test_usage_recall.py / scripts/test_retract.py."""
    import numpy as np

    class _FakeResp:
        def __init__(self, content):
            msg = type("Msg", (), {"content": content})()
            choice = type("Choice", (), {"message": msg})()
            self.choices = [choice]

    class _FakeCompletions:
        def create(self, **kw):
            messages = kw.get("messages", [])
            sysmsg = messages[0]["content"] if messages else ""
            if kw.get("response_format") or "STRICT JSON" in sysmsg:
                return _FakeResp(json.dumps({"facts": [
                    {"statement": "dry-run fact", "key": "dryrun::fact",
                     "salience": 0.5, "valid_at": None, "action": "remember", "scenario": ""},
                ]}))
            if "Grade whether" in sysmsg:
                return _FakeResp("no")
            return _FakeResp("NOTES: dry-run.\nREASON: dry-run.\nANSWER: dry-run-answer")

    class _FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    config.chat_client = lambda: _FakeClient()

    _rng = np.random.default_rng(0)

    def _fake_embed(texts):
        return [_rng.standard_normal(384).astype(np.float32) for _ in texts]

    config.embed_texts = _fake_embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--type", default="", help="filter to one question_type")
    ap.add_argument("--type-budgets", action="store_true")
    ap.add_argument("--cache", default="", help="ingestion-cache dir (rag/tenet embeddings)")
    ap.add_argument("--out", default="docs_scratch/reme_h2h_raw.jsonl",
                    help="resumable per-question raw JSONL")
    ap.add_argument("--budget-cap", type=float, default=None,
                    help="hard-stop $ spend estimate (chars/4 heuristic)")
    ap.add_argument("--reme-venv", default="", help="path to the isolated reme-ai venv")
    ap.add_argument("--reme-config", default=str(REME_CONFIG_DEFAULT))
    ap.add_argument("--retrieval-job", default="bm25_search",
                    choices=("bm25_search", "search", "vector_search"))
    ap.add_argument("--workspace-root", default="", help="dir for per-question ReMe workspaces")
    ap.add_argument("--dry-run", action="store_true",
                    help="stub LLM client + ReMe subprocess, no network, no reme-venv needed")
    ap.add_argument("--dry-script", nargs="*", default=["[dry-run] reme context stub"],
                    help="canned reme-arm context strings for --dry-run, consumed in order")
    ap.add_argument("--project", default="",
                    help="comma-separated n values; print the token/cost PROJECTION "
                         "table (no API calls) and exit, ignoring all other run flags")
    args = ap.parse_args()

    if args.project:
        project([int(x) for x in args.project.split(",")], seed=args.seed)
        return
    run(args)


if __name__ == "__main__":
    main()
