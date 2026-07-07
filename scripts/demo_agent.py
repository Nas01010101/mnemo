"""Scripted demo of the Tenet Assistant — the story for the 3-min video.

Shows an assistant that (1) learns facts, (2) recalls them personally, (3) stays correct
when a fact CHANGES, (4) can time-travel to what you used to say, (5) forgets trivia.
Run: python scripts/demo_agent.py
"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet.agent import MemoryAgent  # noqa: E402

clock = {"t": 1_000_000.0}
def now(): return clock["t"]
DAY = 86400


def say(agent, msg, *, days=0):
    clock["t"] += days * DAY
    out = agent.respond(msg)
    print(f"\n\033[36myou ›\033[0m {msg}")
    print(f"\033[32massistant ›\033[0m {out['reply']}")
    if out["learned"]:
        print(f"   \033[90m[remembered {out['learned']} new fact(s)]\033[0m")


def main():
    agent = MemoryAgent(Path(tempfile.mkdtemp()) / "demo.db", now=now)

    print("=" * 70)
    print("  SESSION 1 — getting to know you")
    print("=" * 70)
    say(agent, "Hi! I'm Alex. I live in Montreal and I work as a data analyst.")
    say(agent, "I'm vegetarian, and I really love dark roast coffee.")
    say(agent, "What do you know about me so far?")

    print("\n" + "=" * 70)
    print("  SESSION 2 — three weeks later, things changed")
    print("=" * 70)
    say(agent, "Big news — I moved to Toronto and got promoted to senior analyst!", days=21)
    say(agent, "Where do I live and what's my job now?")

    print("\n" + "=" * 70)
    print("  The payoff — memory that stays correct under change")
    print("=" * 70)
    say(agent, "Remind me what I told you about my coffee preference.")

    print("\n\033[33m# time-travel: what did I believe BEFORE the move?\033[0m")
    past = agent.recall_history("where does the user live", as_of=1_000_000.0 + 5 * DAY)
    print("   as-of week 1 →", [p for p in past if "Montreal" in p or "Toronto" in p][:1])
    now_belief = agent.recall_history("where does the user live")
    print("   now         →", [p for p in now_belief if "Montreal" in p or "Toronto" in p][:1])

    print(f"\n\033[90mstore: {agent.stats()}  "
          f"(old values retired to history, not deleted)\033[0m")
    print("\n\033[1mTenet: an assistant whose memory stays true as your life changes.\033[0m")


if __name__ == "__main__":
    main()
