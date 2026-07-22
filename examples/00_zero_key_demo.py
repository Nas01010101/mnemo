"""Zero-key demo — Tenet's core belief-state loop with NO API key, NO network.

The "wow in 60 seconds" script: no DASHSCOPE_API_KEY, no OpenAI key, nothing to
sign up for. `EMBED_PROVIDER=local` is forced below (sentence-transformers,
CPU) BEFORE tenet is imported, so this runs on a fresh clone the moment
`pip install -e ".[local]"` (or `pip install "tenet-memory[local] @ git+https://github.com/Nas01010101/tenet.git"`)
finishes.

This intentionally bypasses `Tenet.ingest()` (which distills raw text via one
LLM call — see distill.py) and calls `store_fact()` / `core.store()` directly
with pre-formed (text, key) pairs instead. That's the honest tradeoff of a
zero-key demo: write-time distillation needs a model; **recall, supersession,
time-travel, and doubts do not** — they're pure vector-similarity + sqlite +
a closed-form statistical fit over the ledger, by design (see memory.py's
module docstring: "recall... no LLM in the read path"). A real deployment
still wants `ingest()` for free-form conversation; this demo shows the part
of Tenet that works without ever calling out to a model.

Run (from a clone):
    pip install -e ".[local]"
    python examples/00_zero_key_demo.py
"""
from __future__ import annotations

import os

os.environ["EMBED_PROVIDER"] = "local"  # forced: this demo never touches an LLM/cloud API

import tempfile
import time
from pathlib import Path

from tenet import Tenet

DAY = 86400.0


def main() -> None:
    db_path = Path(tempfile.mkdtemp()) / "zero_key_demo.db"
    # A simulated, injectable clock (same pattern as scripts/test_dynamics.py)
    # instead of wall time: "12 months of history" plays out in under a second,
    # the doubts section below is reproducible, and — important for time-travel
    # below — created_at (transaction time) and valid_at (event time) both
    # derive from the SAME clock instead of drifting apart.
    clock = {"t": time.time() - 400 * DAY}

    def now() -> float:
        return clock["t"]

    m = Tenet(db_path, now=now)

    def store(text: str, *, key: str, salience: float = 0.6, pinned: bool = False) -> None:
        m.store_fact(text, key=key, salience=salience, pinned=pinned)

    print("=== seeding a belief history (no LLM — pre-formed key/value facts) ===\n")

    # pinned=True on a key's FIRST store() call disables its recency decay
    # (`Memory._decay`: "Pinned never decays") and propagates through every
    # future supersession of that key (memory.py: `pinned = pinned or
    # bool(row["pinned"])`) — the right call for durable identity facts,
    # and it keeps ranking below governed by relevance, not by which fact
    # happened to be (re)stored more recently in the simulated timeline.
    store("Alex's employer is Initech, working as a QA engineer.", key="alex::employer", pinned=True)
    store("Alex's home city is Montreal.", key="alex::residence", pinned=True)
    t_before_move = clock["t"]
    clock["t"] += 90 * DAY  # 3 months later
    store("Alex's home city is now Toronto.", key="alex::residence")

    clock["t"] += 60 * DAY  # +2 months
    store("Alex's employer is now Doctave, working as a senior QA engineer.", key="alex::employer")

    # A CHURNY key: weekly status updates. This is what teaches the world
    # model that "status" changes often — a fast-learned hazard, unlike the
    # slow-changing residence/employer keys above (dynamics.py fits this
    # per key-class from the ledger's own supersession history).
    for status in ("heads-down on the Q3 launch", "on vacation",
                    "back and catching up", "ramping a new project"):
        clock["t"] += 7 * DAY
        store(f"Alex's current status: {status}", key="alex::status")

    clock["t"] += 200 * DAY  # +~7 months of silence: nothing updated since
    store("Alex's team at Doctave is using the new docs pipeline.", key="alex::project")

    print("=== supersession: current belief wins, old value is retired (not deleted) ===")
    cur = m.recall("where does Alex live?", k=1)
    print(f"current residence -> {cur[0].text!r}")
    assert "Toronto" in cur[0].text

    job = m.recall("where does Alex work?", k=1)
    print(f"current employer  -> {job[0].text!r}")
    assert "Doctave" in job[0].text
    print(f"store stats: {m.stats()}  (superseded facts kept as history, not gone)\n")

    print("=== time-travel: recall(as_of=...) answers what was true THEN ===")
    past = m.recall("where does Alex live?", k=1, as_of=t_before_move)
    print(f"residence before the move -> {past[0].text!r}")
    assert "Montreal" in past[0].text
    print()

    print("=== doubts: the learned world model flags facts worth re-verifying ===")
    print("(a fact's P(still true) decays with age since its last confirmation —")
    print(" dynamics.py fits this per key-class from the ledger's own supersession")
    print(" history: keys that churn often decay fast, stable ones decay slowly.)\n")
    doubts = m.uncertain_facts(threshold=0.5)
    for d in doubts:
        print(f"  doubt: {d['text']!r}  p_valid={d['p_valid']}  age={d['age_days']}d")
    # "status" changed 4x in a month then went quiet for 7 months -> the learned
    # hazard for that key class doubts it long before the (never-updated)
    # residence/employer facts do.
    assert any(d["key"] == "alex::status" for d in doubts)
    print()

    print("Recall, supersession, time-travel, and doubts above never called an LLM —")
    print("pure vector similarity (local embedder) + sqlite + a closed-form fit.")
    print("The one thing that DOES need a model is turning free-form conversation")
    print("into keyed facts (Tenet.ingest()) — see examples/01_quickstart.py.")
    m.close()


if __name__ == "__main__":
    main()
