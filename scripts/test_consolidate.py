"""Deterministic tests for write-time consolidation (TENET_CONSOLIDATE — the
opt-in Mem0-parity knob for extreme churn, ChurnBench §9 follow-up).

Semantics under test:
  OFF (default) — behavior byte-identical to before the flag existed: superseded
      facts leave current recall (bi-temporal), raw echoes of the old value are
      still subject only to the read-time filters.
  ON  — at supersession time, current raw slices echoing the retired value
      (cosine >= _TAU_CONSOLIDATE against the retired fact) are ARCHIVED: gone
      from every recall path including `as_of` (same semantics as the forget
      sweep; rows stay in the sqlite ledger for audit). The belief layer keeps
      full bi-temporality either way — fact time-travel still works.

No LLM, local embedder. Run: python scripts/test_consolidate.py
"""
import os
os.environ.setdefault("EMBED_PROVIDER", "local")  # before importing config

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet.memory import MemoryCore  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def fresh_core(clock):
    return MemoryCore(Path(tempfile.mkdtemp()) / "consolidate.db", now=lambda: clock["t"])


def seed_churn(core, clock, consolidate=None):
    """One fact churned once, with a raw turn echoing the OLD value."""
    core.store("The user lives in Boston.", key="user::residence")
    raw_id = core.store("user: I live in Boston, moved here last fall.",
                        kind="raw", salience=0.35)
    clock["t"] += 100
    t_before_update = clock["t"] - 1
    clock["t"] += 100
    new_id = core.store("The user lives in Seattle.", key="user::residence",
                        consolidate=consolidate)
    return raw_id, new_id, t_before_update


# ---- OFF (default): unchanged behavior --------------------------------------

def test_off_is_unchanged():
    clock = {"t": 1_000_000.0}
    core = fresh_core(clock)
    raw_id, _, t_past = seed_churn(core, clock)  # consolidate unset -> default OFF

    row = core.db.execute("SELECT archived FROM memories WHERE id=?", (raw_id,)).fetchone()
    check("OFF: raw echo of the old value is NOT archived", row["archived"] == 0)

    past = core.recall("where does the user live", as_of=t_past, k=10,
                       consistency_threshold=None)
    check("OFF: as_of time-travel still reaches the raw echo",
          any(m.id == raw_id for m in past), f"ids={[m.id for m in past]}")


# ---- ON (per-call kwarg): raw echoes archived --------------------------------

def test_on_archives_raw_echoes():
    clock = {"t": 1_000_000.0}
    core = fresh_core(clock)
    raw_id, new_id, t_past = seed_churn(core, clock, consolidate=True)

    row = core.db.execute("SELECT archived FROM memories WHERE id=?", (raw_id,)).fetchone()
    check("ON: raw echo of the superseded value IS archived (ledger row kept)",
          row is not None and row["archived"] == 1)

    cur = core.recall("where does the user live", k=10, consistency_threshold=None)
    check("ON: current recall never returns the archived raw echo",
          not any(m.id == raw_id for m in cur), f"ids={[m.id for m in cur]}")
    check("ON: current recall returns the NEW value",
          any(m.id == new_id for m in cur))

    past = core.recall("where does the user live", as_of=t_past, k=10,
                       consistency_threshold=None)
    check("ON: archived raw echo is gone from as_of too (documented trade-off)",
          not any(m.id == raw_id for m in past))
    # belief layer stays bi-temporal: the OLD FACT is still visible via as_of
    check("ON: fact-layer time-travel still shows the superseded BELIEF",
          any("Boston" in m.text and m.key == "user::residence" for m in past),
          f"past={[m.text for m in past]}")


# ---- ON: unrelated raws survive ----------------------------------------------

def test_on_spares_unrelated_raws():
    clock = {"t": 1_000_000.0}
    core = fresh_core(clock)
    core.store("The user lives in Boston.", key="user::residence")
    other = core.store("user: my dentist appointment is on March 14 at 9am.",
                       kind="raw", salience=0.35)
    clock["t"] += 100
    core.store("The user lives in Seattle.", key="user::residence", consolidate=True)

    row = core.db.execute("SELECT archived FROM memories WHERE id=?", (other,)).fetchone()
    check("ON: an unrelated raw slice is NOT archived", row["archived"] == 0)
    cur = core.recall("when is the dentist appointment", k=5, consistency_threshold=None)
    check("ON: unrelated raw still recallable", any(m.id == other for m in cur))


# ---- ON via env flag ----------------------------------------------------------

def test_env_flag():
    clock = {"t": 1_000_000.0}
    core = fresh_core(clock)
    os.environ["TENET_CONSOLIDATE"] = "1"
    try:
        raw_id, _, _ = seed_churn(core, clock)  # consolidate=None -> env decides
    finally:
        del os.environ["TENET_CONSOLIDATE"]
    row = core.db.execute("SELECT archived FROM memories WHERE id=?", (raw_id,)).fetchone()
    check("env TENET_CONSOLIDATE=1: raw echo archived without per-call kwarg",
          row["archived"] == 1)


# ---- ON: repeated churn, every stale echo swept -------------------------------

def test_extreme_churn_no_stale_leak():
    clock = {"t": 1_000_000.0}
    core = fresh_core(clock)
    cities = ["Boston", "Seattle", "Denver", "Austin", "Miami", "Chicago", "Portland"]
    raw_ids = []
    for c in cities:
        core.store(f"The user lives in {c}.", key="user::residence", consolidate=True)
        raw_ids.append(core.store(f"user: I just moved to {c}, loving it so far.",
                                  kind="raw", salience=0.35))
        clock["t"] += 100
    # every raw echo except the CURRENT city's should be archived
    stale = raw_ids[:-1]
    archived = [core.db.execute("SELECT archived FROM memories WHERE id=?", (i,)).fetchone()["archived"]
                for i in stale]
    check("extreme churn: every stale raw echo archived", all(a == 1 for a in archived),
          f"archived={archived}")
    last = core.db.execute("SELECT archived FROM memories WHERE id=?", (raw_ids[-1],)).fetchone()
    check("extreme churn: the CURRENT city's raw echo survives", last["archived"] == 0)
    cur = core.recall("where does the user live", k=10, consistency_threshold=None)
    texts = " | ".join(m.text for m in cur)
    check("extreme churn: no superseded city text in current recall",
          not any(c in texts for c in cities[:-1]), texts[:160])


# ---- retroactive sweep on a store built with the flag OFF ---------------------

def test_consolidate_sweep_retroactive():
    clock = {"t": 1_000_000.0}
    core = fresh_core(clock)
    raw_id, _, _ = seed_churn(core, clock)          # flag OFF: echo survives
    other = core.store("user: my dentist appointment is on March 14 at 9am.",
                       kind="raw", salience=0.35)
    n = core.consolidate_sweep()
    check("sweep: archives exactly the stale echo", n == 1, f"archived {n}")
    row = core.db.execute("SELECT archived FROM memories WHERE id=?", (raw_id,)).fetchone()
    check("sweep: the stale echo is archived", row["archived"] == 1)
    row2 = core.db.execute("SELECT archived FROM memories WHERE id=?", (other,)).fetchone()
    check("sweep: unrelated raw untouched", row2["archived"] == 0)
    check("sweep: idempotent", core.consolidate_sweep() == 0)


if __name__ == "__main__":
    test_off_is_unchanged()
    test_on_archives_raw_echoes()
    test_on_spares_unrelated_raws()
    test_env_flag()
    test_extreme_churn_no_stale_leak()
    test_consolidate_sweep_retroactive()
    print()
    if FAILS:
        print(f"CONSOLIDATE FAILURES: {FAILS}")
        sys.exit(1)
    print("ALL PASS ✅  (TENET_CONSOLIDATE: opt-in write-time consolidation)")
