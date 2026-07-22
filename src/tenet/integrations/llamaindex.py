"""TenetMemoryBlock — a LlamaIndex long-term-memory block backed by Tenet's
bi-temporal MemoryCore.

LlamaIndex's memory API (llama_index.core.memory.Memory) composes long-term
memory out of `BaseMemoryBlock` subclasses — the shipped peers are
`StaticMemoryBlock`, `FactExtractionMemoryBlock`, and `VectorMemoryBlock`
(verified against the installed `llama-index-core` 0.14.23 source, not from
memory: a block implements async `_aget`/`_aput`, may implement `atruncate`,
and `Memory` injects `aget`'s string into the system template while flushing
ejected short-term messages through `aput`).

What this block does that the shipped peers don't:

* `FactExtractionMemoryBlock` extracts facts into an append-only list — a
  changed fact yields two contradictory list entries. `TenetMemoryBlock`
  routes puts through `Tenet.ingest()`, so a changed fact SUPERSEDES the old
  value (retired to history, never deleted) and `aget` only ever surfaces
  what is currently believed true.
* Reads are LLM-free: `aget` is `Tenet.recall()` — embeddings + closed-form
  decay math, low-milliseconds — so the block adds no model call to the
  agent's context-assembly path. Puts are the one place a model runs
  (write-time distillation into keyed atomic facts).
* `atruncate` drops the LOWEST-ranked recall lines first (recall returns
  relevance×decay-ranked facts), instead of discarding the whole block.

Mapping:
  aput(messages)      -> Tenet.ingest(text) per user/assistant message
                          (distill -> keyed facts -> supersession on collision)
  aget(messages)      -> Tenet.recall(last user text, k=..., char_budget=...)
                          formatted as "- <fact>" lines under a small header
  atruncate(content)  -> drop trailing (lowest-ranked) lines to fit

Install: pip install 'tenet-memory[llamaindex] @ git+https://github.com/Nas01010101/tenet.git'
Example: examples/06_llamaindex_memory.py
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

try:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.memory import BaseMemoryBlock
except ImportError as e:  # pragma: no cover - exercised via scripts/test_llamaindex_block.py
    raise ImportError(
        "TenetMemoryBlock needs llama-index-core: pip install 'tenet-memory[llamaindex] @ git+https://github.com/Nas01010101/tenet.git'"
    ) from e

from pydantic import Field

from ..core import Tenet

_HEADER = "Long-term memory (current beliefs; superseded history not shown):"

# ~4 chars/token is the same coarse estimate llama-index's own blocks accept for
# truncation hints; atruncate is a hint API, not an exact contract.
_CHARS_PER_TOKEN = 4


class TenetMemoryBlock(BaseMemoryBlock[str]):
    """Bi-temporal long-term memory for a LlamaIndex `Memory` block stack."""

    name: str = "tenet_memory"
    description: Optional[str] = (
        "Bi-temporal belief store: facts supersede on change, recall is LLM-free."
    )
    tenet: Any = Field(default=None, description="A tenet.Tenet instance (created if omitted).")
    k: int = Field(default=8, description="Max facts recalled per aget.")
    char_budget: Optional[int] = Field(
        default=1200, description="Greedy char cap on recalled facts (recall under a budget)."
    )
    ingest_roles: tuple = Field(
        default=(MessageRole.USER, MessageRole.ASSISTANT),
        description="Message roles worth remembering.",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.tenet is None:
            self.tenet = Tenet()

    @staticmethod
    def _text(msg: ChatMessage) -> str:
        # ChatMessage.content renders the message's content blocks to plain text
        # (None when the message is non-textual, e.g. pure image blocks).
        return (msg.content or "").strip()

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        query = ""
        for msg in reversed(messages or []):
            if msg.role == MessageRole.USER and self._text(msg):
                query = self._text(msg)
                break
        if not query:  # nothing to anchor recall on -> contribute nothing
            return ""
        hits = await asyncio.to_thread(
            lambda: self.tenet.recall(query, k=self.k, char_budget=self.char_budget)
        )
        if not hits:
            return ""
        return _HEADER + "\n" + "\n".join(f"- {h.text}" for h in hits)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        for msg in messages:
            text = self._text(msg)
            if msg.role in self.ingest_roles and text:
                # distill -> keyed atomic facts; a changed fact collides on its
                # key and supersedes (bi-temporal retire, not overwrite)
                await asyncio.to_thread(self.tenet.ingest, text)

    async def atruncate(self, content: str, tokens_to_truncate: int) -> Optional[str]:
        lines = content.splitlines()
        if len(lines) <= 1:  # header only / empty
            return None
        drop_chars = tokens_to_truncate * _CHARS_PER_TOKEN
        while len(lines) > 1 and drop_chars > 0:
            drop_chars -= len(lines.pop())  # facts are ranked: drop worst-ranked (last) first
        return "\n".join(lines) if len(lines) > 1 else None
