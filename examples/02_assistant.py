"""Minimal chat-loop assistant wired with Tenet as its memory — condensed
from Tenet's own reference assistant (src/agent.py in the repo).

Pattern: recall() before every turn to ground the system prompt in what's
remembered -> call YOUR LLM -> ingest() the user's turn so new/changed facts
are learned for next time. Tenet doesn't care which model answers, only what
it remembers — swap call_llm() for your own OpenAI/Anthropic/etc. client.

Run:
    pip install tenet-memory openai
    export DASHSCOPE_API_KEY=sk-...   # or point call_llm() at your own provider
    python examples/02_assistant.py
"""
from __future__ import annotations

from tenet import Tenet

_SYSTEM = """You are a warm, concise personal assistant with long-term memory
of the user. Use ONLY the remembered facts below to personalise your reply;
never invent facts about them.

What you remember about the user:
{memories}"""


def call_llm(messages: list[dict]) -> str:
    """Swap this for your own client. Shown here with the OpenAI SDK pointed
    at Qwen Cloud, matching the shipped Tenet product (see src/config.py)."""
    import os

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    r = client.chat.completions.create(
        model="qwen3.7-plus", messages=messages, max_tokens=300,
    )
    return (r.choices[0].message.content or "").strip()


class Assistant:
    def __init__(self, db_path: str | None = None):
        self.memory = Tenet(db_path) if db_path else Tenet()

    def respond(self, user_msg: str, *, k: int = 8) -> str:
        # Read path: pure vector recall, no LLM call.
        mems = self.memory.recall(user_msg, k=k, expand=4)
        ctx = "\n".join(f"- {mem.text}" for mem in mems) or "(nothing yet)"
        reply = call_llm([
            {"role": "system", "content": _SYSTEM.format(memories=ctx)},
            {"role": "user", "content": user_msg},
        ])
        # Write path: one LLM call distills the turn into atomic, keyed facts
        # and stores them (superseding any that changed).
        self.memory.ingest(user_msg)
        return reply


def main() -> None:
    assistant = Assistant()
    print("Tenet assistant — remembers across runs. Ctrl-C to exit.")
    while True:
        try:
            msg = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye — I'll remember this next time.")
            break
        if not msg:
            continue
        print(f"assistant > {assistant.respond(msg)}")


if __name__ == "__main__":
    main()
