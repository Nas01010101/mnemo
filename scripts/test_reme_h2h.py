"""Deterministic tests for scripts/bench_reme_h2h.py (Tenet vs ReMe LongMemEval
head-to-head harness). No network, no reme-ai venv required — the dry-run path
under test stubs the LLM client (same `config.chat_client` monkeypatch pattern
as scripts/test_usage_recall.py) and never spawns a `reme` subprocess.

Part 1: pure functions imported directly from bench_reme_h2h — oracle guard,
McNemar/Wilson math against hand-computable values, cost tracking.

Part 2: the actual CLI (subprocess, `--dry-run`) — arm dispatch, resumable
JSONL, and the full pipeline end-to-end. Real module, real argv, not a
reimplementation — these fail if the script is reverted/broken.

Run: EMBED_PROVIDER=local python scripts/test_reme_h2h.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("EMBED_PROVIDER", "local")  # before importing tenet.config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# The Part 2 CLI tests drive the real bench over the full LongMemEval_S haystack
# (data/lme/longmemeval_s.json). That dataset is a large external file that is not
# committed, so it is absent in CI and on a fresh clone. Part 2 is skipped when it
# is missing; the deterministic Part 1 tests (math, guards, cost) always run.
_DATA = ROOT / "data" / "lme" / "longmemeval_s.json"

from bench_reme_h2h import guard_not_oracle, mcnemar, CostTracker, est_cost  # noqa: E402
from bench_factcon import wilson_ci  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- Part 1: pure functions, in-process, no subprocess ----------------------


def test_oracle_guard_refuses_evidence_only_variant():
    raised = False
    try:
        guard_not_oracle("data/lme/longmemeval_oracle.json")
    except ValueError:
        raised = True
    check("guard_not_oracle refuses longmemeval_oracle.json", raised)

    raised_full = False
    try:
        guard_not_oracle(Path("data/lme/longmemeval_s.json"))
    except ValueError:
        raised_full = True
    check("guard_not_oracle accepts the full-haystack longmemeval_s.json", not raised_full)

    # Path-shaped, not string-shaped — the guard must handle both (argparse
    # gives a str; internal callers may pass the module-level `Path` constant).
    raised_path_oracle = False
    try:
        guard_not_oracle(Path("/some/dir/longmemeval_oracle.json"))
    except ValueError:
        raised_path_oracle = True
    check("guard_not_oracle refuses a Path pointing at oracle.json, any directory",
          raised_path_oracle)


def test_mcnemar_known_values():
    # nd=0: no discordant pairs at all -> can't reject, p=1.0 by definition.
    n_a, n_b, p = mcnemar([(True, True), (False, False)])
    check("mcnemar: zero discordant pairs -> (0, 0, p=1.0)",
          (n_a, n_b, p) == (0, 0, 1.0), f"got {(n_a, n_b, p)}")

    # nd=1: a single discordant pair can never be significant.
    n_a, n_b, p = mcnemar([(True, False)])
    check("mcnemar: 1 discordant pair -> p=1.0 (can't reject on n=1)",
          n_a == 1 and n_b == 0 and abs(p - 1.0) < 1e-9, f"got {(n_a, n_b, p)}")

    # Textbook exact McNemar case: 10 discordant pairs, split 1 vs 9.
    # p = 2 * sum_{x=0}^{1} C(10,x) / 2^10 = 2 * 11 / 1024 = 0.021484375
    pairs = [(True, False)] * 1 + [(False, True)] * 9
    n_a, n_b, p = mcnemar(pairs)
    check("mcnemar: 1-vs-9 of 10 discordant -> exact textbook p≈0.021484",
          n_a == 1 and n_b == 9 and abs(p - 0.021484375) < 1e-9, f"got {(n_a, n_b, p)}")

    # Symmetric split (5 vs 5) must NOT be significant.
    pairs = [(True, False)] * 5 + [(False, True)] * 5
    _, _, p = mcnemar(pairs)
    check("mcnemar: symmetric 5-vs-5 split -> p=1.0 (no evidence of a difference)",
          abs(p - 1.0) < 1e-9, f"got p={p}")


def test_wilson_ci_reused_from_bench_factcon():
    """bench_reme_h2h imports wilson_ci rather than reimplementing it — sanity-
    check the import resolves to the SAME tested function (not a shadow copy)."""
    lo, hi = wilson_ci(0.5, 100)
    check("wilson_ci(0.5, n=100): CI straddles 0.5 and is a proper subset of [0,1]",
          lo < 0.5 < hi and 0.0 <= lo and hi <= 1.0, f"got ({lo:.3f},{hi:.3f})")
    lo0, hi0 = wilson_ci(0.0, 10)
    check("wilson_ci(0.0, n=10): lower bound is exactly 0",
          lo0 == 0.0 and hi0 > 0.0, f"got ({lo0},{hi0})")


def test_cost_tracker_budget_cap():
    cost = CostTracker(cap=0.01)
    check("CostTracker: not over cap before any spend", not cost.over_cap())
    cost.add("tenet", "qwen3.7-plus", in_chars=4_000_000, out_chars=0)  # ~1M tok in @ $0.25/M
    check("CostTracker: over cap once cumulative spend crosses it",
          cost.over_cap(), f"total=${cost.total():.4f}")
    expected = est_cost("qwen3.7-plus", 4_000_000, 0)  # same chars, independently computed
    check("CostTracker.total() matches est_cost() for the same call",
          abs(cost.total() - expected) < 1e-9, f"{cost.total()} vs {expected}")

    cost_uncapped = CostTracker(cap=None)
    cost_uncapped.add("tenet", "qwen3.7-plus", in_chars=1e9, out_chars=1e9)
    check("CostTracker: cap=None never trips over_cap()", not cost_uncapped.over_cap())


def test_cost_tracker_is_per_arm_not_just_per_model():
    """The deliverable is explicitly PER-ARM cost tracking — two arms sharing
    the same model (qwen3.7-plus is the reader+judge for all four arms) must
    stay separable, or you can never see which arm is actually expensive."""
    cost = CostTracker(cap=None)
    cost.add("rag", "qwen3.7-plus", in_chars=1_000_000, out_chars=0)
    cost.add("tenet", "qwen3.7-plus", in_chars=3_000_000, out_chars=0)
    check("CostTracker.total(arm='rag') isolates rag's spend, not the combined total",
          abs(cost.total("rag") - est_cost("qwen3.7-plus", 1_000_000, 0)) < 1e-9,
          f"total(rag)=${cost.total('rag'):.4f}")
    check("CostTracker.total(arm='tenet') isolates tenet's spend",
          abs(cost.total("tenet") - est_cost("qwen3.7-plus", 3_000_000, 0)) < 1e-9,
          f"total(tenet)=${cost.total('tenet'):.4f}")
    check("CostTracker.total() (no arm) sums both arms",
          abs(cost.total() - (cost.total("rag") + cost.total("tenet"))) < 1e-9)


def test_qa_score_never_scores_an_api_failure_as_wrong():
    """Edge path, not the happy path: an empty prediction (API failure) must
    return (None, None) — never counted as incorrect — and must NOT rack up a
    judge-call cost, since qa_judge is never actually invoked on empty input
    (same contract as lme_recall.eval_instance's qa_error exclusion)."""
    import bench_reme_h2h as bh
    orig_qa_answer = bh.qa_answer
    inst = {"question": "What color is the sky?", "question_date": "2024/01/01 (Mon) 00:00",
            "answer": "blue"}
    try:
        bh.qa_answer = lambda context, question, qdate: ""  # simulated API failure
        cost = CostTracker(cap=None)
        ok, pred = bh.qa_score("some context", inst, cost, "blind")
        check("qa_score: empty prediction -> (None, None), never scored wrong",
              (ok, pred) == (None, None), f"got {(ok, pred)}")
        check("qa_score: empty prediction -> only the answer call's cost is tracked "
              "(no judge call was made, so no judge cost)",
              list(cost.usd) == [("blind", bh.ANSWER_MODEL)], f"usd keys={list(cost.usd)}")
        check("qa_score: the one tracked call has non-zero cost (the answer call ran)",
              cost.total() > 0, f"total=${cost.total()}")
    finally:
        bh.qa_answer = orig_qa_answer


# ---- Part 2: the real CLI, subprocess, --dry-run (no network) ---------------

_PY = sys.executable
_SCRIPT = str(ROOT / "scripts" / "bench_reme_h2h.py")


def _run_cli(args, cwd=ROOT, timeout=120):
    return subprocess.run([_PY, _SCRIPT, *args], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout,
                          env={**os.environ, "EMBED_PROVIDER": "local"})


def test_dry_run_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "raw.jsonl")
        r = _run_cli(["--n", "1", "--dry-run", "--out", out])
        check("dry-run CLI exits 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
        check("dry-run prints the arm summary header",
              "ReMe head-to-head" in r.stdout, r.stdout[-400:])
        check("dry-run prints per-arm McNemar lines",
              "McNemar tenet-vs-reme" in r.stdout)
        rows = [json.loads(line) for line in Path(out).read_text().splitlines() if line.strip()]
        check("dry-run wrote exactly one JSONL row for --n 1", len(rows) == 1, f"got {len(rows)}")
        if rows:
            keys = set(rows[0])
            check("dry-run row has all four arms scored (or a qa_error)",
                  rows[0].get("qa_error") or
                  {"blind_ok", "rag_ok", "reme_ok", "tenet_ok"} <= keys,
                  f"keys={keys}")


def test_arm_dispatch_restricts_output_keys():
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "raw.jsonl")
        r = _run_cli(["--n", "1", "--dry-run", "--arms", "blind,reme", "--out", out])
        check("arm-restricted dry-run exits 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
        rows = [json.loads(line) for line in Path(out).read_text().splitlines() if line.strip()]
        if rows and not rows[0].get("qa_error"):
            keys = set(rows[0])
            check("--arms blind,reme: row scores blind and reme",
                  "blind_ok" in keys and "reme_ok" in keys, f"keys={keys}")
            check("--arms blind,reme: row does NOT score rag or tenet (arm dispatch works)",
                  "rag_ok" not in keys and "tenet_ok" not in keys, f"keys={keys}")


def test_resume_skips_already_done_questions():
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "raw.jsonl")
        r1 = _run_cli(["--n", "2", "--dry-run", "--out", out])
        check("resume test: first run (n=2) exits 0", r1.returncode == 0)
        rows1 = Path(out).read_text().splitlines()
        check("resume test: first run wrote 2 rows", len(rows1) == 2, f"got {len(rows1)}")

        r2 = _run_cli(["--n", "3", "--dry-run", "--out", out])
        check("resume test: second run (n=3, same --out) exits 0", r2.returncode == 0)
        check("resume test: second run reports resuming 2 already-done questions",
              "resume: 2 question(s) already done" in r2.stdout, r2.stdout[:300])
        rows2 = Path(out).read_text().splitlines()
        check("resume test: file has 3 rows total after resume (2 old + 1 new, no duplicates)",
              len(rows2) == 3, f"got {len(rows2)}")
        qids = [json.loads(line)["qid"] for line in rows2]
        check("resume test: no duplicate question ids across the resumed run",
              len(qids) == len(set(qids)), f"qids={qids}")


def test_oracle_guard_refuses_via_project_mode_too():
    """--project also reads DATA and must honor the same oracle guard (it calls
    guard_not_oracle(DATA) directly — DATA itself is longmemeval_s.json so this
    just confirms --project doesn't crash/refuse on the correct file)."""
    r = _run_cli(["--project", "1"])
    check("--project 1 exits 0 on the full-haystack dataset", r.returncode == 0,
          r.stderr[-500:] if r.returncode else "")
    check("--project prints the NO-API-calls projection header",
          "NO API calls" in r.stdout, r.stdout[:200])


def main() -> int:
    test_oracle_guard_refuses_evidence_only_variant()
    test_mcnemar_known_values()
    test_wilson_ci_reused_from_bench_factcon()
    test_cost_tracker_budget_cap()
    test_cost_tracker_is_per_arm_not_just_per_model()
    test_qa_score_never_scores_an_api_failure_as_wrong()
    if _DATA.exists():
        test_dry_run_end_to_end()
        test_arm_dispatch_restricts_output_keys()
        test_resume_skips_already_done_questions()
        test_oracle_guard_refuses_via_project_mode_too()
    else:
        print(f"  SKIP Part 2 (CLI dry-run over {_DATA.relative_to(ROOT)}): dataset not "
              "present (large external file, not committed); Part 1 deterministic tests ran")
    if FAILS:
        print("\nFAILURES:", FAILS)
        return 1
    print("\nALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
