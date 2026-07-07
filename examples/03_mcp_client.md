# Using Tenet as an MCP server

Tenet ships an MCP server (`src/mcp_server.py`) that exposes the memory core as
tools, so any MCP client — Claude Desktop, an IDE, another agent — gets
persistent, self-managing memory without writing any glue code. This is a doc
instead of a runnable script because the useful part is the config snippet and
the launch command, not Python.

## 1. Launch the server

The server speaks stdio MCP. After `pip install tenet-memory[mcp]` there's a
console script:

```bash
tenet-mcp
```

Running from a repo checkout instead (no install)? Use module form:

```bash
python -m tenet.mcp_server
```

It needs the same environment as the rest of Tenet: `DASHSCOPE_API_KEY` (or
`LLM_PROVIDER=openrouter` + `EMBED_PROVIDER=local`), read from `.env` in the
repo root. See `src/tenet/config.py` for the full provider matrix.

## 2. Point Claude Desktop at it

Add this to Claude Desktop's MCP config
(`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "tenet": {
      "command": "tenet-mcp"
    }
  }
}
```

Restart Claude Desktop. It will pick up 5 tools (see below).

## 3. Or via the `claude mcp` CLI

```bash
claude mcp add tenet -- tenet-mcp
```

## Available tools

| tool | what it does |
|---|---|
| `learn(message, pinned=False)` | Distill a raw message into atomic facts and store each (supersession-aware). The main write path — prefer this over `remember` for conversational input. |
| `remember(text, pinned=False)` | Store one pre-formed fact directly, skipping distillation. `pinned=true` for identity facts that must never be forgotten. |
| `recall(query, k=5, char_budget=None)` | Retrieve the most relevant current memories for a query, ranked by relevance × freshness. |
| `forget_stale()` | Run the forgetting sweep (archives decayed, unpinned memories) and report store stats. |
| `memory_stats()` | Report current / superseded / archived counts. |

## Why this matters

Any agent talking to this MCP server gets bi-temporal, supersession-aware
memory for free — including time-travel recall, which the current MCP tool
surface doesn't expose directly (`recall` only returns current facts). If you
need `as_of` from an MCP client, call `Tenet.recall(query, as_of=...)`
directly in Python instead (see `examples/01_quickstart.py`), or add a thin
`recall_as_of` tool to `src/tenet/mcp_server.py` — it's a one-line wrapper
around `core.recall(query, as_of=t)`.
