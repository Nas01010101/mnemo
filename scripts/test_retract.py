"""Deterministic tests for the retraction/tombstone op (docs/COMPARISON.md
follow-up #3, "ask_to_forget" — a deletion, distinct from value-replacement
supersession).

Part 1: MemoryCore.retract() directly — store -> retract -> recall no longer
returns it; recall(as_of=<before retraction>) still shows it existed. No LLM.

Part 2: Tenet.ingest()'s retract-routing flag, with distill()'s LLM call STUBBED
(same `config.chat_client` monkeypatch pattern as scripts/test_errors.py — no
network) — confirms the flag actually gates behavior: OFF (default) stores a
retract-tagged fact normally (unchanged behavior); ON routes it to retract()
instead.

Run: EMBED_PROVIDER=local python scripts/test_retract.py
"""
import os
os.environ.setdefault("EMBED_PROVIDER", "local")  # before importing config

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet import config  # noqa: E402
from tenet.core import Tenet  # noqa: E402
from tenet.memory import MemoryCore  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- Part 1: MemoryCore.retract() directly, no LLM --------------------------

def test_retract_core():
    clock = {"t": 1_000_000.0}
    core = MemoryCore(Path(tempfile.mkdtemp()) / "retract.db", now=lambda: clock["t"])

    mid = core.store("Alex's favorite color is teal.", key="user::favorite_color")
    before_retraction = clock["t"] + 0.5
    clock["t"] += 10
    n = core.retract("user::favorite_color")
    check("retract() reports 1 row retracted", n == 1, f"got {n}")

    current = core.recall("what is the favorite color", k=5)
    check("current recall() no longer returns the retracted fact",
          not any(m.id == mid for m in current), f"ids={[m.id for m in current]}")

    past = core.recall("what is the favorite color", as_of=before_retraction, k=5)
    check("recall(as_of=<before retraction>) still shows it existed",
          any(m.id == mid for m in past), f"ids={[m.id for m in past]}")

    # retracting a key with nothing current is a no-op, not an error
    n2 = core.retract("user::favorite_color")
    check("retracting an already-retracted key is a no-op (0, no error)", n2 == 0, f"got {n2}")

    n3 = core.retract("user::nonexistent_key")
    check("retracting a never-stored key is a no-op (0, no error)", n3 == 0, f"got {n3}")

    # pinned facts ARE retractable (an explicit "forget X" outranks a pin)
    pid = core.store("Alex's callsign is Falcon.", key="user::callsign", pinned=True)
    n4 = core.retract("user::callsign")
    check("pinned facts are retractable (explicit forget beats pin)", n4 == 1, f"got {n4}")
    still_current = core.recall("callsign", k=5)
    check("retracted pinned fact no longer current",
          not any(m.id == pid for m in still_current), f"ids={[m.id for m in still_current]}")

    core.close()


# ---- Part 2: Tenet.ingest()'s retract-routing flag, distill() stubbed -------

class _FakeResp:
    def __init__(self, content):
        msg = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": msg})()
        self.choices = [choice]


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)

    def create(self, **_kw):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResp(item)


class _FakeClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(script)})()


def _stub_distill(*json_strings):
    fake = _FakeClient(list(json_strings))
    config.chat_client = lambda: fake
    return fake


_RETRACT_JSON = json.dumps({"facts": [
    {"statement": "forgot Alex's old phone number", "key": "user::phone",
     "salience": 0.5, "valid_at": None, "action": "retract"},
]})


def test_ingest_retract_flag_off_by_default():
    """Flag OFF (the default): a retract-tagged fact is routed through the exact
    same code path as before this feature existed — plain `core.store()`, action
    field ignored entirely. NOTE: since the distilled fact happens to share the
    prior fact's key, ORDINARY same-key supersession still retires "Pixel" here —
    that's pre-existing store() behavior, unrelated to retract() and expected
    either way; what this test actually verifies is that the id IS returned (it
    was stored, not diverted to retract()) and the new text is what's current —
    i.e. behaviorally identical to calling ingest() before this feature shipped."""
    orig_client, orig_sleep = config.chat_client, time.sleep
    time.sleep = lambda *_a, **_kw: None
    try:
        m = Tenet(Path(tempfile.mkdtemp()) / "ingest_off.db")
        m.core.store("Alex's phone is a Pixel.", key="user::phone")
        _stub_distill(_RETRACT_JSON)
        ids = m.ingest("forget my old phone number")  # retract=None -> _RETRACT_DEFAULT (False)
        check("flag OFF: retract-tagged fact IS stored (not routed to retract())",
              len(ids) == 1, f"ids={ids}")
        hits = m.recall("phone", k=5)
        check("flag OFF: the distilled statement is what's current (plain store() semantics)",
              any("forgot Alex's old phone number" in h.text for h in hits),
              f"hits={[h.text for h in hits]}")
        m.close()
    finally:
        config.chat_client = orig_client
        time.sleep = orig_sleep


def test_ingest_retract_flag_on():
    """Flag ON (per-call override): a retract-tagged fact is routed to
    core.retract(key) instead of being stored, and retires the prior current fact
    under that key."""
    orig_client, orig_sleep = config.chat_client, time.sleep
    time.sleep = lambda *_a, **_kw: None
    try:
        m = Tenet(Path(tempfile.mkdtemp()) / "ingest_on.db")
        mid = m.store_fact("Alex's phone is a Pixel.", key="user::phone")
        _stub_distill(_RETRACT_JSON)
        ids = m.ingest("forget my old phone number", retract=True)
        check("flag ON: retract-tagged fact gets NO stored id",
              ids == [], f"ids={ids}")
        hits = m.recall("phone", k=5)
        check("flag ON: the prior fact under that key is now retracted",
              not any(h.id == mid for h in hits), f"hits={[h.text for h in hits]}")
        m.close()
    finally:
        config.chat_client = orig_client
        time.sleep = orig_sleep


def test_ingest_retract_flag_on_mixed_batch():
    """A distill() response with BOTH a remember and a retract fact: only the
    retract one is diverted; the remember one is stored as normal."""
    orig_client, orig_sleep = config.chat_client, time.sleep
    time.sleep = lambda *_a, **_kw: None
    try:
        mixed = json.dumps({"facts": [
            {"statement": "Alex's job is now a pilot.", "key": "user::job",
             "salience": 0.8, "valid_at": None, "action": "remember"},
            {"statement": "forgot Alex's old phone number", "key": "user::phone",
             "salience": 0.5, "valid_at": None, "action": "retract"},
        ]})
        m = Tenet(Path(tempfile.mkdtemp()) / "ingest_mixed.db")
        m.store_fact("Alex's phone is a Pixel.", key="user::phone")
        _stub_distill(mixed)
        ids = m.ingest("I'm a pilot now, also forget my old phone number", retract=True)
        check("mixed batch: exactly 1 id returned (the remember fact only)",
              len(ids) == 1, f"ids={ids}")
        job_hits = m.recall("job", k=5)
        check("mixed batch: the remember fact WAS stored",
              any("pilot" in h.text for h in job_hits), f"hits={[h.text for h in job_hits]}")
        phone_hits = m.recall("phone", k=5)
        check("mixed batch: the retract fact's target is gone from current recall",
              not any("Pixel" in h.text for h in phone_hits), f"hits={[h.text for h in phone_hits]}")
        m.close()
    finally:
        config.chat_client = orig_client
        time.sleep = orig_sleep


def test_ingest_session_retract_flag_on():
    """ingest_session() (the chunked path bench_persona.py/bench_locomo.py use)
    gets the SAME retract routing as ingest() — only the distilled-facts half;
    raw verbatim turns are unaffected."""
    orig_client, orig_sleep = config.chat_client, time.sleep
    time.sleep = lambda *_a, **_kw: None
    try:
        m = Tenet(Path(tempfile.mkdtemp()) / "ingest_session_on.db")
        mid = m.store_fact("Alex's phone is a Pixel.", key="user::phone")
        _stub_distill(_RETRACT_JSON)
        out = m.ingest_session([("user", "forget my old phone number")],
                               source="s0", retract=True, surprise_gate=None)
        check("ingest_session flag ON: no fact id for the retracted entry",
              out["facts"] == [], f"facts={out['facts']}")
        check("ingest_session flag ON: the raw turn is still stored as usual",
              len(out["raw"]) == 1, f"raw={out['raw']}")
        hits = m.recall("phone", k=5)
        check("ingest_session flag ON: the prior fact under that key is retracted",
              not any(h.id == mid for h in hits), f"hits={[h.text for h in hits]}")
        m.close()
    finally:
        config.chat_client = orig_client
        time.sleep = orig_sleep


def main() -> int:
    print("=== Part 1: MemoryCore.retract() ===")
    test_retract_core()
    print("=== Part 2: Tenet.ingest() retract routing (distill stubbed) ===")
    test_ingest_retract_flag_off_by_default()
    test_ingest_retract_flag_on()
    test_ingest_retract_flag_on_mixed_batch()
    print("=== Part 3: Tenet.ingest_session() retract routing ===")
    test_ingest_session_retract_flag_on()
    if FAILS:
        print("\nFAILURES:", FAILS)
        return 1
    print("\nALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
