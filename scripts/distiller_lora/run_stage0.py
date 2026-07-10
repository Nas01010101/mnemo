"""Stage 0 / Stage 3 metric runner. Scores any set of distiller endpoints on the
held-out eval set, on the axes that actually govern memory correctness:

  json_valid     — fraction of messages whose raw completion is valid JSON envelope
  kv_pathology   — fraction with a 'key::attr=value' statement (the small-model failure mode)
  P / R / F1     — fuzzy statement match vs the qwen3.7-plus reference labels
  fabrication    — mean facts emitted on QUESTION+CHITCHAT msgs (must be ~0; these have no fact)
  key_consist    — mean within-group key identity across paraphrases of ONE fixed fact
                   (this is THE supersession-critical property)

Usage: python3 run_stage0.py [--cloud] cand1 cand2 ...   (candidate = ollama model tag)
       --cloud additionally scores the qwen3.7-plus reference as the quality ceiling.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from harness import (ollama_endpoint, qwen_endpoint, raw_json_valid,
                     raw_keyvalue_pathology, prf)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def score_endpoint(ep, messages, groups):
    per_msg = []
    n_valid = n_kv = 0
    all_p = all_r = all_f = 0.0
    n_scored = 0
    fabricated = []
    for m in messages:
        facts, raw = ep.distill(m["text"])
        valid = raw_json_valid(raw)
        kv = raw_keyvalue_pathology(raw)
        n_valid += valid
        n_kv += kv
        cand_stmts = [f.statement for f in facts]
        rec = {"id": m["id"], "cat": m["category"], "n": len(facts),
               "valid": valid, "kv": kv, "keys": [f.key for f in facts]}
        if m["category"] in ("fact", "churn"):
            p, r, f1 = prf(cand_stmts, [rf["statement"] for rf in m["ref_facts"]])
            all_p += p; all_r += r; all_f += f1; n_scored += 1
            rec.update(p=round(p, 3), r=round(r, 3), f1=round(f1, 3))
        else:  # question / chitchat → expect zero facts
            fabricated.append(len(facts))
        per_msg.append(rec)

    # key-consistency: for each group, run all paraphrases, take the dominant key share
    kc_scores = []
    group_detail = []
    for g in groups:
        keys = []
        for ptext in g["paraphrases"]:
            facts, _ = ep.distill(ptext)
            # the key of the single most salient fact (the attribute update)
            if facts:
                keys.append(max(facts, key=lambda f: f.salience).key)
        if keys:
            mode, cnt = Counter(keys).most_common(1)[0]
            kc_scores.append(cnt / len(g["paraphrases"]))
            group_detail.append({"hint": g["gold_key_hint"], "mode_key": mode,
                                 "share": round(cnt / len(g["paraphrases"]), 2),
                                 "keys": keys})
        else:
            kc_scores.append(0.0)
            group_detail.append({"hint": g["gold_key_hint"], "mode_key": None,
                                 "share": 0.0, "keys": []})

    n = len(messages)
    summary = {
        "candidate": ep.name,
        "json_valid": round(n_valid / n, 3),
        "kv_pathology": round(n_kv / n, 3),
        "precision": round(all_p / n_scored, 3) if n_scored else None,
        "recall": round(all_r / n_scored, 3) if n_scored else None,
        "f1": round(all_f / n_scored, 3) if n_scored else None,
        "fabrication": round(sum(fabricated) / len(fabricated), 3) if fabricated else None,
        "key_consist": round(sum(kc_scores) / len(kc_scores), 3) if kc_scores else None,
    }
    return summary, per_msg, group_detail


def main():
    args = [a for a in sys.argv[1:] if a != "--cloud"]
    do_cloud = "--cloud" in sys.argv
    if not args:
        args = ["qwen2.5:0.5b-instruct", "qwen2.5:1.5b-instruct", "qwen2.5:7b"]

    messages = [json.loads(l) for l in (DATA / "eval_messages.jsonl").open()]
    groups = json.loads((DATA / "paraphrase_groups.json").read_text())

    endpoints = [ollama_endpoint(m) for m in args]
    if do_cloud:
        endpoints.append(qwen_endpoint("qwen3.7-plus"))

    summaries = []
    for ep in endpoints:
        print(f"\n>>> scoring {ep.name} ...", flush=True)
        summary, per_msg, groupd = score_endpoint(ep, messages, groups)
        summaries.append(summary)
        tag = ep.name.replace("/", "_").replace(":", "_")
        (DATA / f"stage0_{tag}.json").write_text(json.dumps(
            {"summary": summary, "per_msg": per_msg, "groups": groupd}, indent=2))
        print(json.dumps(summary, indent=2), flush=True)

    # verdict table
    cols = ["candidate", "json_valid", "kv_pathology", "precision", "recall",
            "f1", "fabrication", "key_consist"]
    print("\n\n=== STAGE 0 VERDICT TABLE ===")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for s in summaries:
        print("| " + " | ".join(str(s.get(c)) for c in cols) + " |")
    (DATA / "stage0_verdict.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
