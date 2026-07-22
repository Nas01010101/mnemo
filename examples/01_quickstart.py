"""Quickstart: remembering facts, updating them, and time-travel recall.

Shows the core Tenet loop:
  1. ingest()  — turn a raw message into distilled, keyed facts (one LLM call)
  2. recall()  — the read path: pure vector similarity + decay, NO LLM call
  3. a fact changing — same semantic key ("user::residence") means the new
     value SUPERSEDES the old one instead of duplicating it
  4. time-travel — recall(as_of=<timestamp>) answers what was true THEN, not
     now; history is preserved, never deleted

Run:
    pip install git+https://github.com/Nas01010101/tenet.git
    export DASHSCOPE_API_KEY=sk-...      # or: LLM_PROVIDER=openrouter EMBED_PROVIDER=local
    python examples/01_quickstart.py

Uses a throwaway on-disk DB (a tempdir) so it's safe to re-run.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from tenet import Tenet


def main() -> None:
    db_path = Path(tempfile.mkdtemp()) / "quickstart.db"
    m = Tenet(db_path)

    # 1. Remember something. ingest() distills the raw message into atomic
    #    facts with a semantic key like "user::residence" — the key is what
    #    makes step 3 (supersession) reliable instead of best-effort.
    m.ingest("Hi, I'm Alex. I live in Montreal and I'm vegetarian.")

    # 2. recall() is the read path: no LLM call, just embeddings + decay.
    hits = m.recall("where does the user live?")
    print("current residence:", hits[0].text)  # -> "... Montreal ..."

    # Snapshot "now" so we can time-travel back to it after the fact changes.
    t_before_move = time.time()

    # 3. The fact changes. Same key -> the old value is SUPERSEDED (marked
    #    invalid, not deleted) rather than stored as a second, competing fact.
    m.ingest("Update: I moved to Toronto last week.")

    hits = m.recall("where does the user live?")
    print("current residence:", hits[0].text)  # -> "... Toronto ..." (current wins)

    # 4. Time-travel: what did we believe BEFORE the move?
    past = m.recall("where does the user live?", as_of=t_before_move)
    print("residence as of before the move:", past[0].text)  # -> "... Montreal ..."

    # The vegetarian fact never changed, so it's untouched by the supersession.
    diet = m.recall("dietary restriction")
    print("diet:", diet[0].text)

    print("\nstore stats:", m.stats())  # {'current': N, 'superseded': 1, 'archived': 0}
    m.close()


if __name__ == "__main__":
    main()
