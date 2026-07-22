# Examples

- [`01_quickstart.py`](01_quickstart.py) — remember, update, and time-travel recall in ~50 lines.
- [`02_assistant.py`](02_assistant.py) — a minimal chat-loop assistant wired with Tenet as memory.
- [`03_mcp_client.md`](03_mcp_client.md) — point Claude Desktop (or any MCP client) at Tenet's MCP server.
- [`04_langchain_memory.py`](04_langchain_memory.py) — a thin `TenetMemory` adapter for LangChain-style agent loops.
- [`06_llamaindex_memory.py`](06_llamaindex_memory.py) — `TenetMemoryBlock` in LlamaIndex's `Memory` block stack (`pip install "tenet-memory[llamaindex] @ git+https://github.com/Nas01010101/tenet.git"`); changed facts supersede instead of contradicting.

All examples import `from tenet import Tenet` (`pip install git+https://github.com/Nas01010101/tenet.git`) and need at least one working provider — set `DASHSCOPE_API_KEY`, or `LLM_PROVIDER=openrouter` + `EMBED_PROVIDER=local` (no API key, local embeddings). See `src/tenet/config.py` for the full provider matrix.
