"""Stage 3 — the ungameable verdict harness for local fact-distiller candidates.

For each candidate distiller (ollama tag, or the cloud reference) it measures, on a
held-out set the model never trained on:

  (a) JSON validity, key=value pathology, precision/recall/F1 vs reference, fabrication
  (b) key-consistency across paraphrases  (the supersession-critical property)
  (c) THE END-TO-END GATE: raw messages -> distill(local) -> bi-temporal store, then
      * scripts/test_tenet_e2e.py  (move + manager change must supersede)
      * a churn-style supersession probe: U updates per attribute; the current value
        must be the LAST one and exactly U-1 prior values must be superseded.

Distillation is routed at the candidate via LLM_PROVIDER=ollama + OLLAMA_MODEL; embeddings
are held constant (qwen) so the ONLY variable is the distiller. A candidate is ship-worthy
if it passes e2e AND the churn probe at >= the reference distiller's behavior on the same
messages.

Usage:
  python scripts/eval_local_distiller.py --candidates qwen2.5:0.5b-instruct tenet-distiller-0.5b
  python scripts/eval_local_distiller.py --candidates ... --box http://100.88.179.78:11434 --gate-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "distiller_lora"))

# churn-style update chains (attr -> [message sequence], gold = last value substring)
CHURN_CHAINS = {
    "residence": (["I just moved to Denver.", "I just moved to Austin.",
                   "I just moved to Seattle.", "I just moved to Portland."], "Portland"),
    "car": (["I bought a new car, a Honda Accord.", "I bought a new car, a Toyota Prius.",
             "I bought a new car, a Tesla Model 3."], "Tesla Model 3"),
    "job_title": (["I got promoted, I'm now a Data Scientist.",
                   "I got promoted, I'm now a Staff Engineer.",
                   "I got promoted, I'm now a Director."], "Director"),
}


def run_e2e(env) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "test_tenet_e2e.py")],
                       env=env, capture_output=True, text=True, timeout=600)
    out = (r.stdout + r.stderr).strip().splitlines()
    tail = out[-1] if out else ""
    ok = r.returncode == 0 and "SKIPPED" not in (r.stdout + r.stderr)
    return ok, tail


def run_churn_probe(model, box, embed_provider="qwen") -> dict:
    """In-process: feed each chain through a fresh Tenet with the local distiller,
    check the current value is the last and superseded == sum(len-1)."""
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["OLLAMA_BASE_URL"] = box.rstrip("/") + "/v1"
    os.environ["OLLAMA_MODEL"] = model
    os.environ["EMBED_PROVIDER"] = embed_provider
    # fresh import so module-level provider constants pick up the env
    for m in [k for k in list(sys.modules) if k.startswith("tenet")]:
        del sys.modules[m]
    from tenet import Tenet
    from tenet.config import ProviderError

    clock = {"t": 1_000_000.0}
    db = Path(tempfile.mkdtemp()) / "churn.db"
    m = Tenet(db, now=lambda: clock["t"])
    expected_superseded = sum(len(seq) - 1 for seq, _ in CHURN_CHAINS.values())
    correct = 0
    try:
        for attr, (seq, gold) in CHURN_CHAINS.items():
            for msg in seq:
                m.ingest(msg); clock["t"] += 3600
        for attr, (seq, gold) in CHURN_CHAINS.items():
            hits = {mm.text for mm in m.recall(f"what is the user's current {attr}?", k=6)}
            cur_ok = any(gold.lower() in t.lower() for t in hits)
            stale = any(other.lower() in t.lower() for t in hits
                        for oseq, _ in [CHURN_CHAINS[attr]] for other in
                        [s.split()[-1].strip('.') for s in seq[:-1]])
            correct += int(cur_ok and not stale)
    except ProviderError as e:
        m.close()
        return {"error": f"provider: {e.reason}"}
    st = m.stats()
    m.close()
    return {"attrs_correct": correct, "attrs_total": len(CHURN_CHAINS),
            "superseded": st.get("superseded"), "expected_superseded": expected_superseded,
            "current_total": st.get("current", st.get("live"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--box", default="http://100.88.179.78:11434")
    ap.add_argument("--embed", default="qwen")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--metrics-only", action="store_true")
    args = ap.parse_args()

    results = []
    for cand in args.candidates:
        print(f"\n===== CANDIDATE: {cand} =====", flush=True)
        row = {"candidate": cand}

        if not args.gate_only:
            from harness import ollama_endpoint
            from run_stage0 import score_endpoint
            DATA = ROOT / "scripts" / "distiller_lora" / "data"
            messages = [json.loads(l) for l in (DATA / "eval_messages.jsonl").open()]
            groups = json.loads((DATA / "paraphrase_groups.json").read_text())
            summary, _, _ = score_endpoint(ollama_endpoint(cand, args.box), messages, groups)
            row["metrics"] = summary
            print("metrics:", json.dumps(summary), flush=True)

        if not args.metrics_only:
            env = dict(os.environ)
            env.update(LLM_PROVIDER="ollama", OLLAMA_BASE_URL=args.box.rstrip("/") + "/v1",
                       OLLAMA_MODEL=cand, EMBED_PROVIDER=args.embed)
            e2e_ok, e2e_tail = run_e2e(env)
            row["e2e_pass"] = e2e_ok
            row["e2e_tail"] = e2e_tail
            print(f"e2e: {'PASS' if e2e_ok else 'FAIL'} | {e2e_tail}", flush=True)
            churn = run_churn_probe(cand, args.box, args.embed)
            row["churn"] = churn
            print("churn:", json.dumps(churn), flush=True)

        results.append(row)

    out = ROOT / "scripts" / "distiller_lora" / "data" / "stage3_verdict.json"
    out.write_text(json.dumps(results, indent=2))
    print("\n=== wrote", out)


if __name__ == "__main__":
    main()
