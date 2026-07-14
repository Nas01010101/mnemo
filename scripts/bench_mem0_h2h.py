"""Real Mem0 head-to-head — Tenet vs the actual `mem0ai` package (not our faithful
reimpl arm) on the knowledge-churn task, identical reader, identical embedder.

Three arms over the SAME per-principal churn history (reused verbatim from
bench_knowledge_update: several attributes each updated oldest→newest amid
distractors) and the SAME `qwen3.7-plus` reader + `score()`:
  - naive-RAG   : top-k raw turns
  - Tenet       : bi-temporal belief-state recall (qwen3.6-flash distill)
  - mem0 (real) : mem0ai 2.x — add() per turn, search() per question. Its internal
                  fact-extraction/consolidation LLM is qwen3.7-plus (STRONGER than
                  Tenet's flash distiller — deliberately generous to Mem0), bge-small
                  embedder (same as Tenet/RAG), chroma vector store.

Metric per arm: current-value accuracy (reader names the LATEST value) and
stale-leak rate (reader names a superseded value). Wilson 95% CIs; McNemar
paired test Tenet-vs-Mem0. API failures excluded, never scored wrong.

Runs in the isolated mem0 venv (scratchpad/mem0-venv) so mem0's heavy deps never
touch the product env. Zero new infra: chroma is on-disk, embeds are local.

Usage (from repo root):
  set -a; . ./.env; set +a
  scratchpad/mem0-venv/bin/python scripts/bench_mem0_h2h.py --principals 6 --k 8
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings, tempfile
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("EMBED_PROVIDER", "local")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from tenet import Tenet, config  # noqa: E402
from bench_knowledge_update import build_history, answer, score, ATTRS  # noqa: E402
from bench_factcon import wilson_ci  # noqa: E402


def make_mem0():
    from mem0 import Memory
    coll = "h2h_" + str(int(time.time() * 1000) % 1_000_000)
    cfg = {
        "llm": {"provider": "openai", "config": {
            "model": config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"),
            "api_key": os.environ["DASHSCOPE_API_KEY"],
            "openai_base_url": os.environ.get("QWEN_BASE_URL",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")}},
        "embedder": {"provider": "huggingface", "config": {"model": "BAAI/bge-small-en-v1.5"}},
        "vector_store": {"provider": "chroma", "config": {
            "collection_name": coll, "path": f"scratchpad/mem0_chroma/{coll}"}},
    }
    return Memory.from_config(cfg)


def mem0_search(m, q, uid, k):
    r = m.search(q, filters={"user_id": uid}, limit=k)
    res = r.get("results", r) if isinstance(r, dict) else r
    return "\n".join(f"- {it.get('memory')}" if isinstance(it, dict) else f"- {it}" for it in res)


def run(principals: int, k: int):
    arms = ("rag", "tenet", "mem0")
    ok = {a: [] for a in arms}       # per-question correctness (current value)
    leak = {a: [] for a in arms}     # per-question stale-leak
    paired = []                       # (tenet_ok, mem0_ok) for McNemar
    misses = []
    t0 = time.time()

    for p in range(principals):
        sessions, gold = build_history(p)
        turns = [t["content"] for s in sessions for t in s]
        uid = f"p{p}"

        # FRESH mem0 store per principal (isolated, like Tenet's fresh DB) — a shared
        # collection makes mem0's per-add dedup scan a growing store and slow to a crawl,
        # and would leak facts across principals. One user, one clean store.
        m0 = make_mem0()

        # RAG store (raw turns + local embeds)
        host = Tenet(Path(tempfile.mkdtemp()) / "h.db")
        rag_texts = [f"User: {t}" for t in turns]
        rag_mat = np.array(host.core.embed_batch(rag_texts))

        # Tenet: chronological distill+supersede
        clock = [1_000_000.0]
        tn = Tenet(Path(tempfile.mkdtemp()) / "t.db", now=lambda: clock[0])
        for s in sessions:
            tn.ingest_session(s, valid_at=clock[0]); clock[0] += 3600

        # real Mem0: add each turn online (its own extraction/consolidation LLM)
        for t in turns:
            try:
                m0.add(f"User: {t}", user_id=uid)
            except Exception as e:
                print(f"  [mem0 add err p{p}] {e}", flush=True)

        for attr, (q, _vals) in ATTRS.items():
            latest, stale = gold[attr]
            qv = host.core.embed_batch([q])[0]
            top = np.argsort(-(rag_mat @ qv))[:k]
            ctx = {
                "rag": "\n".join(rag_texts[i] for i in sorted(top)),
                "tenet": "\n".join(f"- {h.text}" for h in tn.core.recall(q, k=k)),
                "mem0": mem0_search(m0, q, uid, k),
            }
            row = {"p": p, "attr": attr, "latest": latest}
            a_ok = {}
            for a in arms:
                ans = answer(ctx[a], q)
                if not ans.strip():
                    a_ok[a] = None; continue
                c, lk = score(ans, latest, stale)
                a_ok[a] = c
                ok[a].append(c); leak[a].append(lk)
                row[f"{a}_ans"] = ans[:60]; row[f"{a}_ok"] = c
            if a_ok.get("tenet") is not None and a_ok.get("mem0") is not None:
                paired.append((a_ok["tenet"], a_ok["mem0"]))
            if not (a_ok.get("tenet") and a_ok.get("mem0")):
                misses.append(row)
        host.close(); tn.close()
        print(f"  [{p+1}/{principals}] "
              + "  ".join(f"{a}={sum(ok[a])}/{len(ok[a])}" for a in arms), flush=True)

    print(f"\n=== Mem0 head-to-head — knowledge churn (k={k}, {principals} principals, "
          f"qwen3.7-plus reader) ===")
    print(f"{'arm':>8} | {'current-value acc':>26} | stale-leak")
    for a in arms:
        n = len(ok[a]); acc = sum(ok[a]) / max(n, 1); lo, hi = wilson_ci(acc, max(n, 1))
        lkr = sum(leak[a]) / max(n, 1)
        print(f"{a:>8} | {100*acc:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] n={n:3d} | {100*lkr:4.1f}%")

    # McNemar Tenet vs Mem0
    n01 = sum((not t) and mm for t, mm in paired)   # mem0-only right
    n10 = sum(t and (not mm) for t, mm in paired)    # tenet-only right
    from math import comb
    nd = n01 + n10
    pval = (sum(comb(nd, x) for x in range(min(n01, n10) + 1)) / 2 ** nd * 2) if nd else 1.0
    print(f"\nMcNemar (Tenet vs Mem0): tenet-only-right={n10}  mem0-only-right={n01}  "
          f"p={min(pval,1):.4f}  (paired n={len(paired)})")
    print(f"wall={time.time()-t0:.0f}s")

    out = {"config": {"principals": principals, "k": k, "reader": "qwen3.7-plus",
                      "mem0_llm": "qwen3.7-plus", "embedder": "bge-small"},
           "arms": {a: {"acc": sum(ok[a]) / max(len(ok[a]), 1), "n": len(ok[a]),
                        "stale_leak": sum(leak[a]) / max(len(leak[a]), 1)} for a in arms},
           "mcnemar": {"tenet_only": n10, "mem0_only": n01, "p": min(pval, 1)}}
    Path("scratchpad/mem0_h2h.json").write_text(json.dumps(out, indent=2))
    Path("scratchpad/mem0_h2h_misses.jsonl").write_text(
        "\n".join(json.dumps(m) for m in misses))
    print("wrote scratchpad/mem0_h2h.json + _misses.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--principals", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    run(a.principals, a.k)
