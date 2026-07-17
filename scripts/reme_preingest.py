"""Parallel ReMe workspace pre-ingest for scripts/bench_reme_h2h.py.

ReMe's auto_memory job makes one LLM call per haystack session (~50/question),
so serial per-question ingest inside the bench loop takes ~15 min/question and
times out. This script front-loads that step with a thread pool over questions
(the bench then skips ingest via the .ingested marker — see reme_h2h_driver).

Must select the SAME instances as bench_reme_h2h.py: identical filter, seeded
shuffle, and slice — both sides import DATA and default seed=0.

Usage:
  python scripts/reme_preingest.py --n 100 --reme-venv scratchpad/reme-venv \
      --workspace-root docs_scratch/reme_ws --workers 6
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bench_reme_h2h import ANSWER_MODEL, DATA, DISTILL_MODEL_REME, EMBED_MODEL, guard_not_oracle
from reme_h2h_driver import reme_env, reme_run_job, reme_write_workspace


def load_instances(n: int, seed: int, qtype: str) -> list[dict]:
    import random
    data = [d for d in json.load(open(DATA)) if not d["question_id"].endswith("_abs")]
    if qtype:
        data = [d for d in data if d["question_type"] == qtype]
    random.Random(seed).shuffle(data)
    return data[:n]


def ingest_one(inst: dict, *, reme_bin: str, config: str, root: Path, env: dict,
               timeout: int) -> tuple[str, float, str]:
    ws = reme_write_workspace(inst, root)
    marker = ws / ".ingested"
    # Marker is only trusted when notes exist — auto_memory reports success
    # even at 0 notes written (see reme_h2h_driver for the same guard).
    has_notes = any((ws / "daily").rglob("*.md")) if (ws / "daily").is_dir() else False
    if marker.exists() and has_notes:
        return inst["question_id"], 0.0, "cached"
    t0 = time.time()
    reme_run_job(reme_bin, config, "auto_memory", ws, env, timeout=timeout)
    if not any((ws / "daily").rglob("*.md")):
        raise RuntimeError("auto_memory wrote no notes")
    marker.touch()
    return inst["question_id"], time.time() - t0, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--type", default="")
    ap.add_argument("--reme-venv", required=True)
    ap.add_argument("--reme-config", default=str(Path(__file__).parent / "reme_h2h_config.yaml"))
    ap.add_argument("--workspace-root", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    guard_not_oracle(DATA)
    instances = load_instances(args.n, args.seed, args.type)
    root = Path(args.workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    reme_bin = str(Path(args.reme_venv) / "bin" / "reme")
    env = reme_env(embed_model=EMBED_MODEL, answer_model=ANSWER_MODEL,
                   distill_model=DISTILL_MODEL_REME)

    t0, ok, cached, failed = time.time(), 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(ingest_one, inst, reme_bin=reme_bin, config=args.reme_config,
                            root=root, env=env, timeout=args.timeout): inst["question_id"]
                for inst in instances}
        for fut in as_completed(futs):
            qid = futs[fut]
            try:
                _, dt, status = fut.result()
                cached += status == "cached"
                ok += status == "ok"
                print(f"[{ok + cached + len(failed)}/{len(instances)}] {qid} {status} "
                      f"{dt:.0f}s  elapsed={time.time() - t0:.0f}s", flush=True)
            except Exception as e:  # keep going; report failures at the end
                failed.append(qid)
                print(f"[{ok + cached + len(failed)}/{len(instances)}] {qid} "
                      f"FAILED: {str(e)[:200]}", flush=True)

    print(f"\ndone: ok={ok} cached={cached} failed={len(failed)} "
          f"wall={time.time() - t0:.0f}s")
    if failed:
        print("failed qids:", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
