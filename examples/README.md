# Examples

- [`01_quickstart.py`](01_quickstart.py) — remember, update, and time-travel recall in ~50 lines.
- [`02_assistant.py`](02_assistant.py) — a minimal chat-loop assistant wired with Tenet as memory.
- [`03_mcp_client.md`](03_mcp_client.md) — point Claude Desktop (or any MCP client) at Tenet's MCP server.
- [`04_langchain_memory.py`](04_langchain_memory.py) — a thin `TenetMemory` adapter for LangChain-style agent loops.

All examples import `from tenet import Tenet` (`pip install tenet-memory`) and need at least one working provider — set `DASHSCOPE_API_KEY`, or `LLM_PROVIDER=openrouter` + `EMBED_PROVIDER=local` (no API key, local embeddings). See `src/config.py` for the full provider matrix.
