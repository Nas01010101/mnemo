"""Deterministic tests for TenetMemoryBlock (src/tenet/integrations/llamaindex.py) —
the LlamaIndex `BaseMemoryBlock` adapter.

No LLM calls, no network: EMBED_PROVIDER=local forced below; puts are verified by
stubbing `Tenet.ingest` (distillation is the one LLM step and is covered by its own
suite), reads run against pre-formed facts written with `store_fact` (the same
zero-key path examples/00_zero_key_demo.py uses).

Skips cleanly (exit 0) if `llama-index-core` isn't installed — CI installs the
`llamaindex` extra separately; this suite shouldn't fail a base install.

Run: python scripts/test_llamaindex_block.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("EMBED_PROVIDER", "local")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
except ImportError:
    print("llama-index-core not installed — skipping (pip install 'tenet-memory[llamaindex] @ git+https://github.com/Nas01010101/tenet.git')")
    raise SystemExit(0)

from tenet.core import Tenet  # noqa: E402
from tenet.integrations.llamaindex import TenetMemoryBlock  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    mem = Tenet(tmp / "li.db")
    mem.store_fact("Alex's home city is Montreal.", key="alex::residence", valid_at=1_000.0)
    mem.store_fact("Alex's home city is Toronto.", key="alex::residence", valid_at=2_000.0)
    mem.store_fact("Alex works as a data engineer.", key="alex::job", valid_at=1_500.0)

    block = TenetMemoryBlock(tenet=mem, k=5)

    # --- aget: LLM-free recall, current beliefs only (supersession respected)
    out = asyncio.run(block.aget([ChatMessage(role=MessageRole.USER, content="Where does Alex live?")]))
    check("aget returns the current belief", "Toronto" in out, out[:80])
    check("aget hides the superseded value", "Montreal" not in out)
    check("aget carries the block header", out.startswith("Long-term memory"))

    # --- aget with no user message contributes nothing
    out2 = asyncio.run(block.aget([ChatMessage(role=MessageRole.SYSTEM, content="sys prompt")]))
    check("aget without a user message is empty", out2 == "")

    # --- aput routes user/assistant text through Tenet.ingest, skips system/empty
    seen = []
    block.tenet.ingest = lambda text, **kw: seen.append(text)  # stub the LLM step
    asyncio.run(block.aput([
        ChatMessage(role=MessageRole.USER, content="I adopted a cat named Miso."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Noted — Miso!"),
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content="   "),
    ]))
    check("aput ingests user+assistant text", seen == ["I adopted a cat named Miso.", "Noted — Miso!"],
          repr(seen))

    # --- accept_short_term_memory gating comes from the base class
    seen.clear()
    block.accept_short_term_memory = False
    asyncio.run(block.aput([ChatMessage(role=MessageRole.USER, content="ephemeral")],
                           from_short_term_memory=True))
    check("short-term flush respected when opted out", seen == [])
    block.accept_short_term_memory = True

    # --- atruncate drops lowest-ranked (trailing) lines, keeps the header
    content = "Long-term memory:\n- best fact\n- ok fact\n- worst fact"
    cut = asyncio.run(block.atruncate(content, tokens_to_truncate=3))
    check("atruncate drops the worst-ranked line first",
          cut is not None and "worst" not in cut and "best fact" in cut, repr(cut))
    check("atruncate to nothing returns None",
          asyncio.run(block.atruncate("Long-term memory:", 10)) is None)

    print(("PASS — all checks green" if not FAILS else f"FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
