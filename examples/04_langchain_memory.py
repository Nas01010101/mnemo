"""Tenet as a memory backend inside a LangChain-style agent loop.

`TenetMemory` mirrors the shape LangChain's `BaseMemory` expects
(`load_memory_variables`, `save_context`, `memory_variables`) without
subclassing it and without importing langchain — so this adapter has zero
langchain dependency. Only `main()`, which builds a demo chain, needs
`langchain-core`, and that import is guarded with a helpful error.

Run:
    pip install git+https://github.com/Nas01010101/tenet.git langchain-core   # langchain-core only for the demo loop
    export DASHSCOPE_API_KEY=sk-...
    python examples/04_langchain_memory.py
"""
from __future__ import annotations

from tenet import Tenet


class TenetMemory:
    """Drop-in-shaped memory adapter: recall() on read, ingest() on write.

    Use it anywhere a framework expects `load_memory_variables(inputs) -> dict`
    and `save_context(inputs, outputs) -> None` — not just LangChain.
    """

    memory_variables = ["history"]

    def __init__(self, db_path: str | None = None, *, k: int = 6, input_key: str = "input"):
        self.tenet = Tenet(db_path) if db_path else Tenet()
        self.k = k
        self.input_key = input_key

    def load_memory_variables(self, inputs: dict) -> dict:
        """Read path: recall facts relevant to the incoming input. No LLM call."""
        query = inputs.get(self.input_key, "")
        mems = self.tenet.recall(query, k=self.k) if query else []
        return {"history": "\n".join(f"- {m.text}" for m in mems)}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Write path: distill the user turn into durable, keyed facts
        (supersession-aware — a changed fact replaces the old value)."""
        user_msg = inputs.get(self.input_key, "")
        if user_msg:
            self.tenet.ingest(user_msg)

    def clear(self) -> None:
        # Tenet is durable-by-design (bi-temporal history, not a scratch buffer).
        # Use forget_sweep() for principled decay instead of a hard wipe.
        raise NotImplementedError("use self.tenet.forget_sweep() instead of clear()")


def _require_langchain():
    try:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as e:
        raise ImportError(
            "This demo loop needs langchain-core: pip install langchain-core\n"
            "(TenetMemory itself has no langchain dependency — it works with any "
            "framework expecting load_memory_variables/save_context.)"
        ) from e
    return FakeListChatModel, ChatPromptTemplate


def main() -> None:
    FakeListChatModel, ChatPromptTemplate = _require_langchain()

    memory = TenetMemory()
    llm = FakeListChatModel(responses=["Got it, I'll remember that.", "Noted — updated."])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Known facts about the user:\n{history}"),
        ("human", "{input}"),
    ])
    chain = prompt | llm

    for turn in ["I'm Alex and I work at Acme.", "Actually, I just left Acme."]:
        vars = memory.load_memory_variables({"input": turn})
        reply = chain.invoke({"input": turn, "history": vars["history"]})
        print(f"you: {turn}\nbot: {reply.content}\n")
        memory.save_context({"input": turn}, {"output": reply.content})


if __name__ == "__main__":
    main()
