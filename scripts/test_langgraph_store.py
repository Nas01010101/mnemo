"""Deterministic tests for TenetStore (src/tenet/integrations/langgraph.py) —
the LangGraph BaseStore adapter.

No LLM calls, no network: EMBED_PROVIDER=local forced below, and the langgraph
`Op` types are constructed/dispatched directly (no langgraph runtime needed
beyond importing `langgraph.store.base`).

Skips cleanly (exit 0) if `langgraph` isn't installed — CI installs the
`langgraph` extra separately; this suite shouldn't fail a base install.

Run: python scripts/test_langgraph_store.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("EMBED_PROVIDER", "local")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from langgraph.store.base import GetOp, ListNamespacesOp, MatchCondition, PutOp, SearchOp
except ImportError:
    print("langgraph not installed — skipping (pip install 'tenet-memory[langgraph] @ git+https://github.com/Nas01010101/tenet.git')")
    raise SystemExit(0)

from tenet.integrations.langgraph import TenetStore  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    clock = {"t": 1_000_000.0}
    db = Path(tempfile.mkdtemp()) / "langgraph_store.db"
    store = TenetStore(db_path=str(db), now=lambda: clock["t"])

    ns = ("users", "alex")

    # --- put + get (batch dispatch over PutOp/GetOp directly) --------------
    r_put = store.batch([PutOp(ns, "prefs", {"theme": "dark", "notify": True}, index=None, ttl=None)])
    check("put via batch(PutOp) returns None", r_put == [None])

    got = store.batch([GetOp(ns, "prefs", refresh_ttl=True)])[0]
    check("get via batch(GetOp) round-trips value",
          got is not None and got.value == {"theme": "dark", "notify": True},
          detail=str(got.value if got else None))
    check("get: namespace/key reconstructed from skey",
          got is not None and got.namespace == ns and got.key == "prefs")

    # Convenience methods (get/put/search/delete/list_namespaces) built on batch().
    check("convenience .get matches batch(GetOp)",
          store.get(ns, "prefs").value == {"theme": "dark", "notify": True})

    # --- supersession on re-put (the differentiator) ------------------------
    clock["t"] += 3600
    store.put(ns, "prefs", {"theme": "light", "notify": True})
    check("re-put with new value supersedes (get returns new value)",
          store.get(ns, "prefs").value == {"theme": "light", "notify": True})
    # the OLD value is retired, not gone — visible directly in the underlying
    # ledger (BaseStore's synchronous-snapshot API has no time-travel verb).
    old_rows = store.core.db.execute(
        "SELECT text FROM memories WHERE skey=? AND expired_at IS NOT NULL", ("users::alex::prefs",)
    ).fetchall()
    check("old value kept in history (bi-temporal, not overwritten)",
          len(old_rows) == 1 and '"dark"' in old_rows[0]["text"])

    # --- a second item + confidence metadata on search ----------------------
    store.put(ns, "diet", {"restriction": "vegetarian"})
    store.put(("users", "sam"), "prefs", {"theme": "dark"})  # different namespace

    hits = store.search(("users", "alex"), query="ui preferences", limit=5)
    check("search(query=...) scoped to namespace prefix",
          all(h.namespace[:2] == ns for h in hits) and len(hits) >= 1,
          detail=str([(h.namespace, h.key) for h in hits]))
    check("search hit is the theme fact, top-ranked",
          hits and hits[0].key == "prefs" and hits[0].value.get("theme") == "light")
    check("search does not leak the other namespace",
          all(h.key != "prefs" or h.namespace == ns for h in hits) and
          not any(h.namespace == ("users", "sam") for h in hits))

    # --- filter operator ($eq passthrough via plain equality) --------------
    filtered = store.search(ns, filter={"theme": "light"})
    check("search(filter=...) matches on value field", any(h.key == "prefs" for h in filtered))
    filtered_out = store.search(ns, filter={"theme": "dark"})
    check("search(filter=...) excludes non-matching value (superseded theme)",
          not any(h.key == "prefs" for h in filtered_out))

    # --- delete -> archive, not a hard delete -------------------------------
    store.delete(ns, "diet")
    check("delete: get() no longer returns it", store.get(ns, "diet") is None)
    archived_row = store.core.db.execute(
        "SELECT archived FROM memories WHERE skey=?", ("users::alex::diet",)
    ).fetchone()
    check("delete: soft-deleted (archived=1), row still present",
          archived_row is not None and archived_row["archived"] == 1)

    # --- list_namespaces -----------------------------------------------------
    all_ns = store.list_namespaces()
    check("list_namespaces includes both users",
          ("users", "alex") in all_ns and ("users", "sam") in all_ns,
          detail=str(all_ns))
    prefixed = store.list_namespaces(prefix=("users", "alex"))
    check("list_namespaces(prefix=...) scopes correctly",
          prefixed == [("users", "alex")])

    store.close()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("\nLANGGRAPH STORE ALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
