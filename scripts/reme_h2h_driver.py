"""ReMe subprocess driver for scripts/bench_reme_h2h.py — isolated venv, driven
as a black box via `reme start job=...` (service.backend: cli in
reme_h2h_config.yaml -> runs one job, prints its Response.answer, exits).
Workspace layout (query.json + session/*.json) is ReMe's own LongMemEval eval
format — see reme/steps/benchmark/lme/auto_memory.py in the installed
reme-ai package for the exact schema this mirrors.

Split out of bench_reme_h2h.py to keep that file under the repo's 500-line cap
and because this is independently testable/importable (no LLM/tenet deps here).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


def reme_env(base_env: dict | None = None, *, embed_model: str, answer_model: str,
             distill_model: str) -> dict:
    """Env vars for the `reme` subprocess — secrets flow via env, never argv.
    Maps this repo's DashScope credentials (src/tenet/config.py's QWEN_BASE_URL)
    onto ReMe's own `${LLM_*}`/`${EMBEDDING_*}` config placeholders."""
    env = dict(base_env if base_env is not None else os.environ)
    key = env.get("DASHSCOPE_API_KEY", "")
    base_url = env.get("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    env["LLM_API_KEY"] = key
    env["LLM_BASE_URL"] = base_url
    env["EMBEDDING_API_KEY"] = key
    env["EMBEDDING_BASE_URL"] = base_url
    env.setdefault("EMBEDDING_MODEL_NAME", embed_model)
    env.setdefault("LLM_MODEL_NAME", answer_model)
    env.setdefault("LME_DISTILL_MODEL", distill_model)
    return env


def reme_write_workspace(inst: dict, root: Path) -> Path:
    """query.json + session/*.json, ReMe's own LME workspace layout."""
    ws = root / inst["question_id"]
    (ws / "session").mkdir(parents=True, exist_ok=True)
    (ws / "query.json").write_text(json.dumps({
        "question_id": inst["question_id"], "question": inst["question"],
        "question_date": inst["question_date"], "answer": inst["answer"],
    }))
    dates = inst.get("haystack_dates") or [""] * len(inst["haystack_session_ids"])
    for sid, date, sess in zip(inst["haystack_session_ids"], dates, inst["haystack_sessions"]):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(sid))[:120]
        payload = {"haystack_session_id": sid, "haystack_date": date, "messages": sess}
        (ws / "session" / f"{safe}.json").write_text(json.dumps(payload))
    return ws


def reme_run_job(reme_bin: str, config_path: str, job: str, workspace: Path,
                  env: dict, job_args: dict | None = None, timeout: int = 2400) -> str:
    args = [reme_bin, "start", f"job={job}", f"config={config_path}",
            f"workspace_dir={workspace}"]
    for k, v in (job_args or {}).items():
        args.append(f"{k}={v}")
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"reme job={job} failed (rc={r.returncode}): {r.stderr[-2000:]}")
    return r.stdout.strip()


def reme_ingest_and_search(inst: dict, *, reme_bin: str, reme_config: str,
                            workspace_root: Path, k: int, env: dict,
                            retrieval_job: str = "bm25_search",
                            dry_run: bool = False, dry_script: list | None = None) -> str:
    """Ingest the full haystack via ReMe's own auto_memory job, then retrieve at
    question time. HARD CONSTRAINT (bench_reme_h2h.py's task): only ever called
    with dry_run=True in this repo today — a real call makes LLM/embedding
    requests through the reme-venv."""
    if dry_run:
        script = dry_script if dry_script is not None else []
        return script.pop(0) if script else "[dry-run] reme context stub"
    ws = reme_write_workspace(inst, workspace_root)
    marker = ws / ".ingested"
    # The marker alone isn't proof of a usable ingest: auto_memory reports
    # success even when every extraction failed and 0 notes were written,
    # which would silently score the reme arm at blind level. Trust the
    # marker only when notes actually exist; otherwise self-heal by
    # re-ingesting (covers markers touched by older preingest code too).
    has_notes = any((ws / "daily").rglob("*.md")) if (ws / "daily").is_dir() else False
    if not marker.exists() or not has_notes:
        # auto_memory = one LLM call per haystack session (~50/question) — the
        # slow step. scripts/reme_preingest.py runs it in parallel ahead of the
        # bench; the marker makes ingest idempotent across the two entry points.
        reme_run_job(reme_bin, reme_config, "auto_memory", ws, env)
        if not any((ws / "daily").rglob("*.md")):
            raise RuntimeError(f"auto_memory wrote no notes for {inst['question_id']}")
        marker.touch()
    # bm25_search reads the file_store keyword index, which auto_memory does
    # NOT build — update_index does (clear+rebuild over daily notes, local-only,
    # ~seconds). Run it at question time so retrieval never sees an empty index
    # regardless of which entry point ingested this workspace. Found live
    # 2026-07-17: bm25_search returned 0 bytes on a fully ingested workspace,
    # which would have silently scored the reme arm at blind level.
    reme_run_job(reme_bin, reme_config, "update_index", ws, env)
    return reme_run_job(reme_bin, reme_config, retrieval_job, ws, env,
                        job_args={"query": inst["question"], "limit": k})
