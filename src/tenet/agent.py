"""Tenet Assistant — a personal agent with self-managing memory (Track 1).

A conversational assistant that remembers you across sessions, stays correct when your
facts CHANGE (supersession), can recall what you *used* to say (time-travel), and forgets
stale trivia. This is the agent Track 1 asks for; Tenet (tenet/core.py) is its memory.

Run the interactive assistant:   python -m tenet.agent   (or the `tenet-agent` console script)
Powered by Qwen Cloud (config.py provider layer).
"""
from __future__ import annotations

import time

from . import config
from .core import Tenet

_SYS = """You are a warm, concise personal assistant with long-term memory of the user.
Use ONLY the remembered facts below to personalise your reply; never invent facts about
them. If a fact recently changed, acknowledge the current value naturally.

What you remember about the user:
{memories}"""


def _is_pure_question(msg: str) -> bool:
    """True if every sentence in the turn is interrogative — nothing to learn from.

    Guards the store against small distillers hallucinating facts out of questions
    ("Where do I live?" -> fabricated residence) which would then supersede the
    real current value. Mixed turns ("I moved to Lisbon. Where do I live?") still
    ingest normally.
    """
    parts = [p.strip() for p in msg.replace("!", ".").split(".") if p.strip()]
    if not parts:
        return msg.strip().endswith("?")
    return all(p.endswith("?") for p in parts)


class MemoryAgent:
    def __init__(self, db_path=None, *, now=time.time):
        self.m = Tenet(db_path, now=now) if db_path else Tenet(now=now)

    def respond(self, user_msg: str, *, k: int = 8) -> dict:
        """Recall relevant memory → answer with Qwen → learn from what the user said."""
        mems = self.m.recall(user_msg, k=k, expand=4)
        ctx = "\n".join(f"- {x.text}" for x in mems) or "(nothing yet)"
        reply = config.chat(
            [{"role": "system", "content": _SYS.format(memories=ctx)},
             {"role": "user", "content": user_msg}],
            qwen_default=config.QWEN_MODEL, max_tokens=300,
        )
        before = self.m.stats()["superseded"]
        # extract + store facts (with supersession); pure questions carry no facts
        learned = [] if _is_pure_question(user_msg) else self.m.ingest(user_msg)
        superseded = self.m.stats()["superseded"] - before
        return {"reply": reply, "learned": len(learned), "superseded": superseded,
                "recalled": [x.text for x in mems]}

    def recall_history(self, topic: str, as_of: float | None = None):
        """Time-travel: what the user believed/said about `topic`, optionally as-of a time."""
        return [x.text for x in self.m.recall(topic, k=10, as_of=as_of)]

    def forget_stale(self) -> int:
        return self.m.forget_sweep()

    def stats(self) -> dict:
        return self.m.stats()


def main():  # simple REPL
    agent = MemoryAgent()
    print("Tenet Assistant — I remember across sessions. (Ctrl-C to exit)")
    print(f"currently holding: {agent.stats()}")
    while True:
        try:
            msg = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye — I'll remember this next time.")
            break
        if not msg:
            continue
        if msg.startswith("/history "):
            for t in agent.recall_history(msg[9:]):
                print("   •", t)
            continue
        out = agent.respond(msg)
        print(f"\nassistant › {out['reply']}")
        if out["learned"]:
            print(f"   [remembered {out['learned']} new fact(s)]")


if __name__ == "__main__":
    main()
