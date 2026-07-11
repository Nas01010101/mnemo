"""Deterministic tests for the Mem0-compatible CRUD facade (`Tenet.add/search/
get_all/delete/update`) — thin aliases over ingest/recall/retract/list_beliefs,
purely additive. distill()'s LLM call is STUBBED (same `config.chat_client`
monkeypatch pattern as scripts/test_retract.py — no network), so this is fully
deterministic.

Covers: add -> search -> get_all (readable dicts) -> delete, the update()
passthrough (supersession-as-update), and multi-user isolation — the key thing
this facade must get right: two users' same-named facts (both distilled to
"user::residence") must NOT supersede or search-leak into each other, because
`add()` namespaces the key SUBJECT via `key_prefix`, not just a read-time filter.

Run: EMBED_PROVIDER=local python scripts/test_mem0_api.py
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

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- stub plumbing (mirrors scripts/test_retract.py) -------------------------

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


def _stub(*json_strings):
    fake = _FakeClient(list(json_strings))
    config.chat_client = lambda: fake
    return fake


def _facts_json(*facts):
    """facts: (statement, key, salience) tuples -> the {"facts":[...]} distill() expects."""
    return json.dumps({"facts": [
        {"statement": s, "key": k, "salience": sal, "valid_at": None, "action": "remember"}
        for s, k, sal in facts
    ]})


def main() -> int:
    orig_client, orig_sleep = config.chat_client, time.sleep
    time.sleep = lambda *_a, **_kw: None
    try:
        m = Tenet(Path(tempfile.mkdtemp()) / "mem0_api.db")

        # ---- add(): both users' distiller output uses the SAME nominal key
        # ("user::residence") — that's the realistic case (the distiller always
        # subjects first-person statements to "user"), and exactly what add()'s
        # key_prefix namespacing must handle correctly.
        _stub(_facts_json(("Alex lives in Toronto.", "user::residence", 0.8)))
        alice_ids = m.add("I live in Toronto.", user_id="alice")
        check("add() returns stored ids", len(alice_ids) == 1, f"ids={alice_ids}")

        _stub(_facts_json(("Bob lives in Boston.", "user::residence", 0.8)))
        bob_ids = m.add("I live in Boston.", user_id="bob")
        check("add() (2nd user, same nominal key) returns stored ids",
              len(bob_ids) == 1, f"ids={bob_ids}")

        # ---- no cross-user supersession: alice's fact must still be CURRENT
        # after bob's add() with the same nominal key — the whole point of
        # namespacing by user_id via key_prefix, not just filtering at read time.
        alice_all = m.get_all(user_id="alice")
        bob_all = m.get_all(user_id="bob")
        check("no cross-user supersession: alice's residence still current after bob's add()",
              any("Toronto" in b["text"] for b in alice_all),
              f"alice_all={alice_all}")
        check("bob's own fact is current too",
              any("Boston" in b["text"] for b in bob_all), f"bob_all={bob_all}")

        # ---- get_all(): clean, readable dicts, prefix stripped
        check("get_all() returns clean dict shape {id,key,text,p_valid}",
              alice_all and set(alice_all[0].keys()) == {"id", "key", "text", "p_valid"},
              f"keys={list(alice_all[0].keys()) if alice_all else None}")
        check("get_all(user_id=...) strips the namespace prefix from key",
              alice_all[0]["key"] == "user::residence", f"key={alice_all[0]['key']!r}")

        # ---- get_all() unscoped sees BOTH users (raw key still has the prefix)
        everyone = m.get_all()
        check("get_all() with no user_id sees both users' facts",
              sum(1 for b in everyone if "residence" in (b["key"] or "")) >= 2,
              f"n={len(everyone)}")
        check("get_all() unscoped keeps the raw namespaced key",
              any(b["key"] == "alice::user::residence" for b in everyone) and
              any(b["key"] == "bob::user::residence" for b in everyone),
              f"keys={[b['key'] for b in everyone]}")

        # ---- search(): user_id-scoped search doesn't leak across users
        alice_hits = m.search("where do they live", user_id="alice", k=5)
        bob_hits = m.search("where do they live", user_id="bob", k=5)
        check("search(user_id='alice') finds alice's fact",
              any("Toronto" in h.text for h in alice_hits), f"hits={[h.text for h in alice_hits]}")
        check("search(user_id='alice') does NOT leak bob's fact",
              not any("Boston" in h.text for h in alice_hits), f"hits={[h.text for h in alice_hits]}")
        check("search(user_id='bob') finds bob's fact, not alice's",
              any("Boston" in h.text for h in bob_hits)
              and not any("Toronto" in h.text for h in bob_hits),
              f"hits={[h.text for h in bob_hits]}")

        unscoped_hits = m.search("where do they live", k=10)
        check("search() with no user_id can see both users",
              any("Toronto" in h.text for h in unscoped_hits)
              and any("Boston" in h.text for h in unscoped_hits),
              f"hits={[h.text for h in unscoped_hits]}")

        # ---- update(): passthrough to add() — supersession IS the update
        _stub(_facts_json(("Alex lives in Denver.", "user::residence", 0.8)))
        m.update("I moved to Denver.", user_id="alice")
        alice_all2 = m.get_all(user_id="alice")
        check("update() (passthrough to add()) supersedes the prior value",
              any("Denver" in b["text"] for b in alice_all2)
              and not any("Toronto" in b["text"] for b in alice_all2),
              f"alice_all={alice_all2}")
        bob_all2 = m.get_all(user_id="bob")
        check("update() for alice does not touch bob's fact",
              any("Boston" in b["text"] for b in bob_all2), f"bob_all={bob_all2}")

        # ---- delete(): a deletion (retract), not a value replacement
        n = m.delete("user::residence", user_id="alice")
        check("delete() reports 1 row retracted", n == 1, f"got {n}")
        alice_all3 = m.get_all(user_id="alice")
        check("delete()'d fact is gone from get_all()",
              not any("residence" in (b["key"] or "") for b in alice_all3),
              f"alice_all={alice_all3}")
        bob_all3 = m.get_all(user_id="bob")
        check("delete() for alice does not touch bob's fact",
              any("Boston" in b["text"] for b in bob_all3), f"bob_all={bob_all3}")

        m.close()
    finally:
        config.chat_client = orig_client
        time.sleep = orig_sleep

    if FAILS:
        print("\nFAILURES:", FAILS)
        return 1
    print("\nALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
