"""Reader-generality arm: Gemini reader driven through the local `agy` CLI
(subscription serving, $0 API spend) — one call per item, never batched.

Protocol mirrors the un-batched multireader row of record (docs/BENCHMARK.md §2,
docs/lme_multireader_results.json): LME_S seed=0, k=10, bge-small embedder,
qwen3.6-flash distiller, qwen3.7-plus judge; only the READER is swapped for
`agy -p` (headless). Reader repair path identical to lme_recall.qa_answer.

Repro:
  set -a; source .env; set +a
  EMBED_PROVIDER=local python scripts/lme_reader_gemini_cli.py \
      --limit 40 --seed 0 --k 10 --cache docs_scratch/lme40_cache \
      --out docs_scratch/lme_geminicli_eff.jsonl
"""
import argparse, json, random, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lme_recall  # noqa: E402
from tenet import Tenet  # noqa: E402

AGY_MODEL_DEFAULT = "Gemini 3.5 Flash (High)"
_orig_qa_chat = lme_recall._qa_chat
_agy_usage = {"calls": 0, "retries": 0}


def _agy(prompt, model, timeout=300):
    for attempt in (1, 2, 3):
        _agy_usage["calls"] += 1
        try:
            p = subprocess.run(["agy", "-p", prompt, "--model", model, "--sandbox"],
                               capture_output=True, text=True, timeout=timeout)
            out = (p.stdout or "").strip()
            if p.returncode == 0 and out:
                return out
        except subprocess.TimeoutExpired:
            pass
        _agy_usage["retries"] += 1
        time.sleep(2 * attempt)
    return ""


def make_router(model):
    def routed(system, user, max_tokens=256):
        # judge stays on the qwen judge for protocol parity; everything else
        # (CoN read + bounded repair) goes through the Gemini CLI reader
        if system == lme_recall._JUDGE_SYS:
            return _orig_qa_chat(system, user, max_tokens)
        return _agy(f"{system}\n\n{user}", model)
    return routed


def wilson(c, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = c / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    hw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(100 * (ctr - hw), 1), round(100 * (ctr + hw), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--expand", type=int, default=0)
    ap.add_argument("--model", default=AGY_MODEL_DEFAULT)
    ap.add_argument("--cache", default="docs_scratch/lme40_cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lme_recall._qa_chat = make_router(args.model)

    data = [d for d in json.load(open(lme_recall.DATA)) if not d["question_id"].endswith("_abs")]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    core = Tenet(Path(tempfile.mkdtemp()) / "emb.db")
    embedder = core.core.embed_batch
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass
    outf = out_path.open("a")

    t0 = time.time()
    for i, inst in enumerate(data):
        if inst["question_id"] in done:
            continue
        r = lme_recall.eval_instance(inst, args.k, embedder, qa=True, do_full=False,
                                     expand=args.expand, hops=0, cache_dir=cache_dir)
        row = {k: r.get(k) for k in ("qid", "type", "question", "gold",
                                     "rag_qa", "tenet_qa", "rag_pred", "tenet_pred",
                                     "rag_ctx_chars", "tenet_ctx_chars", "qa_error")}
        outf.write(json.dumps(row) + "\n")
        outf.flush()
        print(f"[{i+1}/{len(data)}] {r['type'][:18]:18s} QA rag:{'✓' if r.get('rag_qa') else '✗'} "
              f"tenet:{'✓' if r.get('tenet_qa') else '✗'}  ({time.time()-t0:.0f}s, "
              f"agy calls={_agy_usage['calls']} retries={_agy_usage['retries']})", flush=True)
    outf.close()

    rows = [json.loads(line) for line in out_path.open()]
    rows = [r for r in rows if not r.get("qa_error")]
    n = len(rows)
    summary = {"protocol": {"n": n, "seed": args.seed, "k": args.k, "expand": args.expand,
                            "reader": f"{args.model} via agy CLI, un-batched, one call per item",
                            "judge": "qwen3.7-plus", "embedder": "bge-small (local)",
                            "distiller": "qwen3.6-flash"},
               "agy": dict(_agy_usage)}
    for arm in ("rag", "tenet"):
        c = sum(1 for r in rows if r[f"{arm}_qa"])
        per_type = {}
        for r in rows:
            per_type.setdefault(r["type"], [0, 0])
            per_type[r["type"]][1] += 1
            per_type[r["type"]][0] += 1 if r[f"{arm}_qa"] else 0
        summary[arm] = {"correct": c, "n": n, "acc": round(100 * c / n, 1),
                        "wilson95": wilson(c, n),
                        "per_type": {t: f"{a}/{b}" for t, (a, b) in sorted(per_type.items())}}
    print(json.dumps(summary, indent=1))
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
