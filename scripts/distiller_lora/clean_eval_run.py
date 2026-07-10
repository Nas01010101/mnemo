"""Score candidates on the DECONTAMINATED eval set (values + phrasings disjoint from
train) — the honest generalization number — plus a decontaminated churn supersession
gate that uses held-out values the tuned model never saw."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from harness import ollama_endpoint
from run_stage0 import score_endpoint

DATA = Path(__file__).resolve().parent / "data"
BOX = os.environ.get("BOX_OLLAMA", "http://100.88.179.78:11434")

# held-out churn chains: NONE of these values/phrasings are in train
CLEAN_CHAINS = {
    "residence": (["Lisbon is home for me now.", "Kyoto is home for me now.",
                   "Reykjavik is home for me now."], "Reykjavik"),
    "car": (["My current set of wheels is a Polestar 2.",
             "My current set of wheels is a Lucid Air.",
             "My current set of wheels is a Genesis GV70."], "Genesis GV70"),
    "employer": (["My paychecks come from Palantir these days.",
                  "My paychecks come from Databricks these days.",
                  "My paychecks come from Instacart these days."], "Instacart"),
}


def clean_metrics(cand):
    messages = [json.loads(l) for l in (DATA / "eval_messages_clean.jsonl").open()]
    groups = json.loads((DATA / "paraphrase_groups_clean.json").read_text())
    summary, _, _ = score_endpoint(ollama_endpoint(cand, BOX), messages, groups)
    return summary


def clean_churn(cand, embed="qwen"):
    os.environ.update(LLM_PROVIDER="ollama", OLLAMA_BASE_URL=BOX.rstrip("/") + "/v1",
                      OLLAMA_MODEL=cand, EMBED_PROVIDER=embed)
    for m in [k for k in list(sys.modules) if k.startswith("tenet")]:
        del sys.modules[m]
    from tenet import Tenet
    from tenet.config import ProviderError
    clock = {"t": 1_000_000.0}
    db = Path(tempfile.mkdtemp()) / "c.db"
    m = Tenet(db, now=lambda: clock["t"])
    exp = sum(len(s) - 1 for s, _ in CLEAN_CHAINS.values())
    correct = 0
    try:
        for _, (seq, _g) in CLEAN_CHAINS.items():
            for msg in seq:
                m.ingest(msg); clock["t"] += 3600
        for attr, (seq, gold) in CLEAN_CHAINS.items():
            hits = {mm.text for mm in m.recall(f"what is the user's current {attr}?", k=6)}
            correct += int(any(gold.lower() in t.lower() for t in hits))
    except ProviderError as e:
        m.close(); return {"error": e.reason}
    st = m.stats(); m.close()
    return {"attrs_correct": correct, "attrs_total": len(CLEAN_CHAINS),
            "superseded": st.get("superseded"), "expected_superseded": exp}


if __name__ == "__main__":
    cands = sys.argv[1:] or ["tenet-distiller-0.5b-v2"]
    out = []
    for c in cands:
        print(f"\n>>> {c}", flush=True)
        met = clean_metrics(c)
        chn = clean_churn(c)
        print("clean_metrics:", json.dumps(met), flush=True)
        print("clean_churn:", json.dumps(chn), flush=True)
        out.append({"candidate": c, "clean_metrics": met, "clean_churn": chn})
    (DATA / "clean_verdict.json").write_text(json.dumps(out, indent=2))
    print("\nwrote clean_verdict.json")
