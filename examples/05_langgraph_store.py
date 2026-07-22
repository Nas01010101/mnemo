"""Tenet as a LangGraph `BaseStore` — the drop-in long-term-memory interface
LangGraph agents expect (`store=` on `StateGraph.compile(...)`), backed by
Tenet's bi-temporal MemoryCore instead of an in-memory dict or Postgres.

The differentiator vs every other BaseStore implementation: `put()` doesn't
overwrite. Re-putting the same (namespace, key) SUPERSEDES the old value —
retired to history, not deleted — the identical mechanism `Tenet.ingest()`
uses for conversational facts. `search(query=...)` rides Tenet's learned
"is this still true?" confidence (`p_valid`) on each hit.

Run:
    pip install "tenet-memory[langgraph] @ git+https://github.com/Nas01010101/tenet.git"
    export DASHSCOPE_API_KEY=sk-...      # or: EMBED_PROVIDER=local (offline)
    python examples/05_langgraph_store.py

Uses a throwaway on-disk DB (a tempdir) so it's safe to re-run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from tenet.integrations.langgraph import TenetStore


def main() -> None:
    db_path = Path(tempfile.mkdtemp()) / "langgraph_store.db"
    store = TenetStore(db_path=str(db_path))

    user = ("memories", "alex")  # LangGraph's own convention: (namespace, user_id)-style tuples

    # 1. put() — an agent remembering something about the user.
    store.put(user, "residence", {"city": "Montreal"})
    print("residence:", store.get(user, "residence").value)  # -> {'city': 'Montreal'}

    # 2. The fact changes. Same (namespace, key) -> SUPERSEDED, not duplicated.
    store.put(user, "residence", {"city": "Toronto"})
    print("residence after move:", store.get(user, "residence").value)  # -> {'city': 'Toronto'}

    # 3. A few more facts, then semantic search across the namespace — this is
    #    what an agent calls instead of a hand-written get() when it doesn't
    #    know the exact key (LangMem's `manage_memory`/`search_memory` tools
    #    both compile down to exactly this call).
    store.put(user, "diet", {"restriction": "vegetarian"})
    store.put(user, "coffee_pref", {"roast": "dark", "sugar": False})
    hits = store.search(user, query="what does the user like to drink?", limit=3)
    for h in hits:
        # _tenet_p_valid: the learned-dynamics confidence this fact is still
        # current (see dynamics.py) — SearchItem has no separate metadata
        # slot in the real BaseStore contract, so it rides in `value`.
        print(f"search hit: {h.key} -> {h.value}  (score={h.score:.3f})")

    # 4. list_namespaces() — what LangGraph calls to enumerate what's stored,
    #    e.g. for a memory-browsing UI or a nightly consolidation pass.
    print("namespaces:", store.list_namespaces(prefix=("memories",)))

    # 5. delete() — soft-deleted (Tenet archives rather than hard-deletes),
    #    but gone from every read path immediately.
    store.delete(user, "diet")
    print("diet after delete:", store.get(user, "diet"))  # -> None

    store.close()

    # Wiring into an actual graph is a one-line `store=` on compile:
    #
    #   from langgraph.graph import StateGraph
    #   graph = StateGraph(State).compile(store=TenetStore(db_path="data/agent.db"))
    #
    # then any node can call `store.get(...)`/`store.put(...)`/`store.search(...)`
    # from its `config["store"]` — no code beyond this file needed on Tenet's side.


if __name__ == "__main__":
    main()
