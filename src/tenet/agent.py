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
them. If a fact recently changed, acknowledge the current value naturally. A fact marked
with a confidence caveat may be stale — hedge naturally or ask to re-confirm instead of
stating it as certain.

What you remember about the user:
{memories}{doubts}"""

# Fact dynamics (dynamics.py) attaches a learned P(still valid) staleness/confidence
# hint to keyed facts. ANNOTATION ONLY below — never used to filter/reorder recall
# (measured: rank-demoting doubted facts broke the churn benchmark 100%->33%). We only
# decide how to *word* a fact that's already been recalled on its normal relevance rank.
_CONF_THRESHOLD = 0.6   # below this, caveat the fact in the prompt
_DOUBT_THRESHOLD = 0.5  # uncertain_facts() cutoff for the session-start doubts line
_DOUBT_TOP_N = 3


def _fmt_age(seconds: float) -> str:
    days = seconds / 86400.0
    return "<1d ago" if days < 1 else f"{days:.0f}d ago"


def _format_memories(mems, *, now: float) -> str:
    """Render recalled memories as bullet lines. Facts below _CONF_THRESHOLD get a
    compact, token-cheap caveat (confidence + age); confident facts stay clean."""
    if not mems:
        return "(nothing yet)"
    lines = []
    for m in mems:
        line = f"- {m.text}"
        if m.confidence is not None and m.confidence < _CONF_THRESHOLD:
            line += (f"  [confidence {m.confidence:.2f} — last confirmed "
                     f"{_fmt_age(now - m.valid_at)}; may be stale, hedge or re-confirm]")
        lines.append(line)
    return "\n".join(lines)


def _format_doubts_line(doubts: list[dict]) -> str:
    """One line naming the most-doubted current beliefs (top _DOUBT_TOP_N by lowest
    P(valid)), for the agent to proactively re-confirm when contextually relevant —
    not a forced interrogation every turn. Empty doubts -> empty string, zero overhead."""
    if not doubts:
        return ""
    top = ", ".join(f"{d['key']} (P={d['p_valid']:.2f})" for d in doubts[:_DOUBT_TOP_N])
    return ("\n\nBeliefs likely stale — proactively re-confirm naturally when "
            f"contextually relevant (don't interrogate): {top}")


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
        # Anticipatory verification: session-start check of which stored facts the
        # staleness model doubts, cached once (not recomputed per turn). Empty
        # store/no doubts -> empty line, no crash (uncertain_facts is pure SQL +
        # closed-form fit).
        self._doubts_line = _format_doubts_line(self.m.uncertain_facts(threshold=_DOUBT_THRESHOLD))

    def respond(self, user_msg: str, *, k: int = 8) -> dict:
        """Recall relevant memory → answer with Qwen → learn from what the user said."""
        mems = self.m.recall(user_msg, k=k, expand=4)
        ctx = _format_memories(mems, now=self.m._now())
        reply = config.chat(
            [{"role": "system", "content": _SYS.format(memories=ctx, doubts=self._doubts_line)},
             {"role": "user", "content": user_msg}],
            qwen_default=config.QWEN_MODEL, max_tokens=300,
        )
        before = self.m.stats()["superseded"]
        # extract + store facts (with supersession); pure questions carry no facts
        warning = None
        learned: list[int] = []
        if not _is_pure_question(user_msg):
            try:
                learned = self.m.ingest(user_msg)
            except config.ProviderError as e:
                # Answering still worked (recall is LLM-free) — don't pretend the
                # write also succeeded. Surface it instead of silently losing the turn.
                warning = f"memory write failed: {e.reason} — this turn was not memorized"
        superseded = self.m.stats()["superseded"] - before
        out = {"reply": reply, "learned": len(learned), "superseded": superseded,
               "recalled": [x.text for x in mems]}
        if warning:
            out["warning"] = warning
            out["reply"] = f"{reply}\n\n[{warning}]" if reply else f"[{warning}]"
        return out

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
