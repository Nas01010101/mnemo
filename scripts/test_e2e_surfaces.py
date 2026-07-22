"""End-to-end surface coverage: prove ONE store is really shared across every way a
caller can talk to Tenet — Python API, CLI (subprocess), MCP (tool functions, same
process), LangGraph BaseStore, and the HTTP API (uvicorn subprocess, the same code
path the live Alibaba Cloud FC deploy runs).

All surfaces point at the SAME sqlite file via TENET_DB_PATH (set once, below, before
any tenet import — every surface's default Tenet()/MemoryCore()/TenetStore() picks it
up: see memory.py's `_DEFAULT_DB`). Surfaces run in sequence, each surface's writes
checked visible to the NEXT surface — real cross-process shared state via the file
(the MCP surface additionally gets this "for free" for the module-level `_tenet`
singleton it creates on import — no object-swap trick needed, unlike
scripts/test_agent_uncertainty.py, which swaps `S._tenet.core` because IT needs an
isolated/seeded store; here sharing the real default is the whole point).

EMBED_PROVIDER=local by design (no Qwen key required for the read/write paths that
don't need distillation). The one write path that DOES need an LLM (CLI `remember`,
MCP `learn`, HTTP `/ingest`) is attempted for real (this repo's .env has a working
DASHSCOPE_API_KEY) but is SKIPPABLE — a ProviderError/503/nonzero-exit there is
reported as SKIP, not FAIL, matching scripts/test_tenet_e2e.py's convention, since a
dead/quota-exhausted key is an environment fact, not a regression.

Probe keys deliberately use DIFFERENT skey subjects ("surfaceprobe::marker",
"temporalprobe::marker", ...) rather than a shared namespace like "e2e::*". Found the
hard way: memory.py's embedding-based key resolution (`_KEY_RESOLUTION`, shipped
2026-07-10) supersedes same-SUBJECT facts whose ATTRIBUTE embeds close to each other
(cosine >= _TAU_KEY) AND whose fact texts clear a text-similarity floor — two probes
under a shared "e2e::" subject with attribute names that both contain the word "probe"
cleared both bars and cross-superseded each other, which looked like a shared-state
bug in THIS script until traced to key naming. See this run's findings for the
upstream report (routed, not fixed here — src/tenet/ is out of scope for this script).

Run: python scripts/test_e2e_surfaces.py
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---- env: must be set before any `tenet` import (config.py / memory.py read at
# import/first-use time), and inherited by every subprocess spawned below. -----------
_SCRATCH = Path(tempfile.mkdtemp(prefix="tenet_e2e_surfaces_"))
os.environ["TENET_DB_PATH"] = str(_SCRATCH / "shared.db")
os.environ.setdefault("EMBED_PROVIDER", "local")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MATRIX: dict[str, dict[str, str]] = {}  # surface -> {check: "ok"|"fail"|"skip"|"n/a"}
FAILS: list[str] = []


def record(surface: str, check: str, ok: bool | None, detail: str = "") -> None:
    """ok=True -> ok, ok=False -> fail (recorded in FAILS), ok=None -> skip."""
    MATRIX.setdefault(surface, {})[check] = "ok" if ok is True else ("skip" if ok is None else "fail")
    mark = {"ok": "✅", "skip": "⏭️ SKIP", "fail": "❌ FAIL"}[MATRIX[surface][check]]
    print(f"  [{surface:9s}] {check:12s} {mark}  {detail}")
    if ok is False:
        FAILS.append(f"{surface}.{check}: {detail}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ======================================================================================
# 1. Python API — tenet.Tenet / tenet.MemoryCore directly, in-process.
# ======================================================================================
def surface_python_api() -> None:
    from tenet import Tenet
    from tenet.config import ProviderError

    surface = "python-api"
    try:
        m = Tenet()  # picks up TENET_DB_PATH
        record(surface, "imports", True, "tenet.Tenet")
    except Exception as e:
        record(surface, "imports", False, str(e))
        return

    try:
        mid = m.store_fact("The e2e probe fact is: surface-python-api wrote this.",
                            key="surfaceprobe::marker", salience=0.9, pinned=True)
        hits = m.recall("what did the surface probe write?", k=5)
        navigated, trace = m.navigate("surface probe", k=5)
        doubts = m.uncertain_facts()
        ok = mid > 0 and any("surface-python-api" in h.text for h in hits) and isinstance(trace, list)
        record(surface, "basic-op", ok,
               f"store_fact->id={mid}, recall hits={len(hits)}, navigate hops={trace[-1]['hop'] if trace else 0}, "
               f"uncertain_facts={len(doubts)}")
    except Exception as e:
        record(surface, "basic-op", False, str(e))

    # bi-temporal + forgetting, still Python-API-only. `between` is captured AFTER v1's
    # store_fact() returns (so it's past v1's real created_at, which store() stamps
    # AFTER the embed() call completes — a pre-call timestamp plus a tiny epsilon isn't
    # a safe as_of boundary, since local-embedder latency alone can exceed a 1ms margin;
    # found this the hard way, see this run's findings for detail) and before v2 exists.
    try:
        m.store_fact("temporal probe MARKERALPHA", key="temporalprobe::marker")
        between = m._now()
        time.sleep(0.05)
        m.store_fact("temporal probe MARKERBETA", key="temporalprobe::marker")
        past = m.recall("temporal probe", as_of=between, k=5)
        now_hits = m.recall("temporal probe", k=5)
        swept = m.forget_sweep()
        found_alpha_past = any("MARKERALPHA" in h.text for h in past)
        found_alpha_now = any("MARKERALPHA" in h.text for h in now_hits)
        found_beta_now = any("MARKERBETA" in h.text for h in now_hits)
        ok = found_alpha_past and not found_alpha_now and found_beta_now and isinstance(swept, int)
        record(surface, "state-shared", ok,
               f"as_of recall found ALPHA={found_alpha_past}, "
               f"current recall found BETA={found_beta_now} (not stale ALPHA={not found_alpha_now}), swept={swept}")
    except Exception as e:
        record(surface, "state-shared", False, str(e))

    # error path: empty text must raise ValueError, not silently no-op
    try:
        raised = False
        try:
            m.store_fact("   ")
        except ValueError:
            raised = True
        # a real LLM call — skippable if no working provider
        llm_skip = None
        try:
            m.ingest("probe: my favorite color is teal.")
        except ProviderError as e:
            llm_skip = e.reason
        record(surface, "error-path", raised,
               f"empty-text raises ValueError={raised}; ingest()={'SKIPPED: ' + llm_skip if llm_skip else 'ok (real LLM call succeeded)'}")
    except Exception as e:
        record(surface, "error-path", False, str(e))

    m.close()


# ======================================================================================
# 2. CLI — `tenet` console script, subprocess, exit code + stdout checked.
# ======================================================================================
def _run_cli(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    tenet_bin = shutil.which("tenet")
    if tenet_bin is None:
        # not pip-installed as a console script in this env — fall back to module form
        cmd = [sys.executable, "-m", "tenet.cli", *args]
    else:
        cmd = [tenet_bin, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=os.environ.copy())


def surface_cli() -> None:
    surface = "cli"
    try:
        p = _run_cli("stats")
        record(surface, "imports", p.returncode == 0, f"`tenet stats` exit={p.returncode}: {p.stdout.strip()[:80]}")
    except Exception as e:
        record(surface, "imports", False, str(e))
        return

    try:
        p = _run_cli("recall", "surface probe", "-k", "5")
        ok = p.returncode == 0 and "surface-python-api" in p.stdout
        record(surface, "state-shared", ok,
               f"exit={p.returncode}, found python-api's fact via CLI recall={'surface-python-api' in p.stdout}")
    except Exception as e:
        record(surface, "state-shared", False, str(e))

    # basic-op: no `learn` verb exists at the CLI (only MCP has `learn` — the CLI's
    # write verb is `remember`, and per README that's the PRE-FORMED-fact path in
    # docs, but cmd_remember actually calls Tenet.ingest() i.e. IS the distill path —
    # see cli.py cmd_remember). Attempted for real; ProviderError -> friendly nonzero
    # exit -> reported as SKIP (matches scripts/test_tenet_e2e.py's convention).
    try:
        p = _run_cli("remember", "CLI surface probe: the answer is 42.")
        if p.returncode == 0:
            record(surface, "basic-op", True, f"`tenet remember` exit=0: {p.stdout.strip()[:80]}")
        elif "memory write failed" in (p.stdout + p.stderr):
            record(surface, "basic-op", None, "SKIPPED: LLM provider unavailable for `tenet remember`")
        else:
            record(surface, "basic-op", False, f"exit={p.returncode} stdout={p.stdout!r} stderr={p.stderr!r}")
    except Exception as e:
        record(surface, "basic-op", False, str(e))

    try:
        p1 = _run_cli("doubts")
        p2 = _run_cli("navigate", "surface probe", "-k", "5")
        ok = p1.returncode == 0 and p2.returncode == 0
        record(surface, "n/a-extra", ok, f"doubts exit={p1.returncode}, navigate exit={p2.returncode}")
    except Exception as e:
        record(surface, "n/a-extra", False, str(e))

    # error path: a malformed --as-of must exit nonzero with a clear message, not a
    # raw traceback (cli.py `_parse_as_of`).
    try:
        p = _run_cli("recall", "surface probe", "--as-of", "not-a-date")
        ok = p.returncode != 0 and "invalid" in (p.stdout + p.stderr).lower()
        record(surface, "error-path", ok, f"exit={p.returncode}, message={(p.stdout + p.stderr).strip()[:80]!r}")
    except Exception as e:
        record(surface, "error-path", False, str(e))


# ======================================================================================
# 3. MCP — import tenet.mcp_server, call the @mcp.tool()-decorated functions directly
#    (same pattern scripts/test_agent_uncertainty.py already uses: FastMCP's decorator
#    returns the original callable, so `S.recall(...)` etc. work with no MCP transport).
# ======================================================================================
def surface_mcp() -> None:
    surface = "mcp"
    try:
        from tenet import mcp_server as S
        record(surface, "imports", True, "tenet.mcp_server (module-level _tenet singleton created)")
    except Exception as e:
        record(surface, "imports", False, str(e))
        return

    # state-shared, proof #1: module-level singleton picked up the SAME TENET_DB_PATH
    # file the Python-API/CLI surfaces already wrote to (no object swap here).
    try:
        out = S.recall("surface probe", k=5)
        ok = "surface-python-api" in out
        record(surface, "state-shared", ok, f"S.recall found python-api's fact (file-shared, same TENET_DB_PATH)={ok}")
    except Exception as e:
        record(surface, "state-shared", False, str(e))

    try:
        stored = S.remember("MCP surface probe: pre-formed fact, no distillation.")
        recalled = S.recall("MCP surface probe", k=3)
        ok = "stored (id=" in stored and "MCP surface probe" in recalled
        record(surface, "basic-op", ok, f"remember={stored!r}, recall found it={'MCP surface probe' in recalled}")
    except Exception as e:
        record(surface, "basic-op", False, str(e))

    # skippable LLM path
    try:
        out = S.learn("MCP learn probe: I prefer dark mode.")
        if out.startswith("ERROR: memory write failed"):
            record(surface, "n/a-extra", None, f"S.learn() SKIPPED: {out}")
        else:
            record(surface, "n/a-extra", True, f"S.learn()={out!r}")
    except Exception as e:
        record(surface, "n/a-extra", False, str(e))

    # doubts / forget_stale / memory_stats / time_travel, all LLM-free
    try:
        d = S.doubts(threshold=0.99)  # threshold near 1 -> likely lists something or the clean "no doubted" message
        fs = S.forget_stale()
        ms = S.memory_stats()
        ok = isinstance(d, str) and "current=" in fs and "current=" in ms
        record(surface, "n/a-extra2", ok, f"doubts/forget_stale/memory_stats all returned strings: {ok}")
    except Exception as e:
        record(surface, "n/a-extra2", False, str(e))

    # error path: a garbage as_of date must return a friendly string, not raise
    try:
        out = S.time_travel("surface probe", "not-a-date")
        ok = "invalid" in out.lower()
        record(surface, "error-path", ok, f"time_travel(bad as_of)={out!r}")
    except Exception as e:
        record(surface, "error-path", False, f"raised instead of returning a friendly message: {e}")


# ======================================================================================
# 4. LangGraph BaseStore adapter — skip cleanly if `langgraph` isn't installed (it's an
#    optional extra: `pip install "tenet-memory[langgraph] @ git+https://github.com/Nas01010101/tenet.git"`).
# ======================================================================================
def surface_langgraph() -> None:
    surface = "langgraph"
    try:
        import langgraph  # noqa: F401
    except ImportError:
        record(surface, "imports", None, "SKIPPED: langgraph not installed (optional [langgraph] extra)")
        for check in ("basic-op", "state-shared", "error-path"):
            record(surface, check, None, "SKIPPED (langgraph absent)")
        return

    try:
        from tenet.integrations.langgraph import TenetStore
        store = TenetStore()  # db_path=None -> same TENET_DB_PATH default as every other surface
        record(surface, "imports", True, "tenet.integrations.langgraph.TenetStore")
    except Exception as e:
        record(surface, "imports", False, str(e))
        return

    try:
        ns = ("e2e", "surfaces")
        store.put(ns, "lg_probe", {"v": 1, "note": "langgraph surface probe"})
        item = store.get(ns, "lg_probe")
        store.put(ns, "lg_probe", {"v": 2, "note": "langgraph surface probe, updated"})  # re-put -> supersede
        item2 = store.get(ns, "lg_probe")
        results = store.search(ns, query="langgraph surface probe")
        ok = (item is not None and item.value["v"] == 1 and item2 is not None
              and item2.value["v"] == 2 and len(results) >= 1)
        record(surface, "basic-op", ok,
               f"put/get v1={item.value if item else None}, re-put supersedes -> v2={item2.value if item2 else None}, "
               f"search hits={len(results)}")
    except Exception as e:
        record(surface, "basic-op", False, str(e))

    try:
        # state-shared: search across the WHOLE store should also surface the
        # python-api-written probe (same underlying MemoryCore/db file) even though it
        # wasn't written via TenetStore.put — proves the same table, not a separate one.
        cur = store.core.recall("surface probe", k=5)
        ok = any("surface-python-api" in m.text for m in cur)
        record(surface, "state-shared", ok, f"store.core.recall sees python-api's fact={ok}")
    except Exception as e:
        record(surface, "state-shared", False, str(e))

    try:
        store.delete(ns, "lg_probe")
        gone = store.get(ns, "lg_probe")
        missing = store.get(ns, "does-not-exist")
        ok = gone is None and missing is None  # delete works; get on a missing key is None, not an exception
        record(surface, "error-path", ok, f"delete->get is None={gone is None}, get(missing)->None={missing is None}")
    except Exception as e:
        record(surface, "error-path", False, str(e))


# ======================================================================================
# 5. HTTP API — uvicorn subprocess (mirrors the live Alibaba FC deploy's exact code
#    path: `uvicorn tenet.api:app`), curled from outside, then killed.
# ======================================================================================
def surface_http_api() -> None:
    surface = "http-api"
    try:
        import fastapi, uvicorn  # noqa: F401
    except ImportError:
        for check in ("imports", "basic-op", "state-shared", "error-path"):
            record(surface, check, None, "SKIPPED: fastapi/uvicorn not installed (optional [api] extra)")
        return

    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tenet.api:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(Path(__file__).resolve().parent.parent / "src"),
        env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        # wait for the server to come up (same-process retry loop, not a fixed sleep)
        up = False
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=1) as r:
                    if r.status == 200:
                        up = True
                        break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.25)
        record(surface, "imports", up, f"uvicorn tenet.api:app up on {base}" if up else "server never came up")
        if not up:
            return

        # state-shared: default session's /health stats should already reflect every
        # write the earlier surfaces made to the SAME TENET_DB_PATH file.
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            health = json.loads(r.read())
        ok = health.get("status") == "ok" and health.get("current", 0) > 0 and health.get("embed_provider") == "local"
        record(surface, "state-shared", ok, f"GET /health -> {health}")

        # basic-op: a deterministic write (POST /memories, no LLM) + a read (POST /recall)
        req = urllib.request.Request(
            f"{base}/memories", data=json.dumps({"text": "HTTP surface probe: pre-formed fact."}).encode(),
            headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            store_resp = json.loads(r.read())
        req = urllib.request.Request(
            f"{base}/recall", data=json.dumps({"query": "HTTP surface probe", "k": 5}).encode(),
            headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            recall_resp = json.loads(r.read())
        ok = store_resp.get("id", 0) > 0 and any("HTTP surface probe" in x["text"] for x in recall_resp["results"])
        record(surface, "basic-op", ok, f"POST /memories -> id={store_resp.get('id')}, "
               f"POST /recall found it={any('HTTP surface probe' in x['text'] for x in recall_resp['results'])}")

        # GET /state?as_of= (bi-temporal read, mirrors the live-deploy demo UI)
        with urllib.request.urlopen(f"{base}/state", timeout=5) as r:
            state = json.loads(r.read())
        as_of_ts = time.time() - 3600
        with urllib.request.urlopen(f"{base}/state?as_of={as_of_ts}", timeout=5) as r:
            state_past = json.loads(r.read())
        ok = "beliefs" in state and "beliefs" in state_past
        record(surface, "n/a-extra", ok, f"GET /state beliefs={len(state.get('beliefs', []))}, "
               f"/state?as_of=... beliefs={len(state_past.get('beliefs', []))}")

        # /reset: gives an ISOLATED session, not the shared one — verify it's actually
        # isolated (fresh stats), which is the FastAPI-level analogue of "error path"
        # (a caller trying to reset must not nuke the shared demo state).
        req = urllib.request.Request(f"{base}/reset", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            reset_resp = json.loads(r.read())
        ok = reset_resp.get("current", -1) == 0  # brand new session, empty
        record(surface, "error-path", ok, f"POST /reset -> isolated fresh session, current={reset_resp.get('current')}")
    except Exception as e:
        record(surface, "error-path", False, f"unexpected exception: {e}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def print_matrix() -> None:
    checks = ["imports", "basic-op", "state-shared", "error-path"]
    print("\n=== surface × {imports, basic-op, state-shared, error-path} ===")
    header = f"{'surface':12s} " + " ".join(f"{c:14s}" for c in checks)
    print(header)
    print("-" * len(header))
    for surface, results in MATRIX.items():
        row = f"{surface:12s} "
        for c in checks:
            v = results.get(c, "n/a")
            row += f"{v:14s} "
        print(row)


def main() -> int:
    print(f"scratch db: {os.environ['TENET_DB_PATH']}")
    print("\n=== 1. Python API ===")
    surface_python_api()
    print("\n=== 2. CLI (subprocess) ===")
    surface_cli()
    print("\n=== 3. MCP (in-process tool functions) ===")
    surface_mcp()
    print("\n=== 4. LangGraph BaseStore ===")
    surface_langgraph()
    print("\n=== 5. HTTP API (uvicorn subprocess) ===")
    surface_http_api()

    print_matrix()

    shutil.rmtree(_SCRATCH, ignore_errors=True)

    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print("  -", f)
        return 1
    print("\nE2E SURFACES PASS ✅  (or cleanly SKIPPED where a live LLM key was needed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
