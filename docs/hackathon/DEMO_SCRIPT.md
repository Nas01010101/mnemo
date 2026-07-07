# 3-minute demo video script

Goal: show Tenet *doing the hard things* (supersession, forgetting, time-travel, MCP),
not just "it stores text". Record screen + voice; upload public to YouTube.

**0:00 – 0:20 — Hook / problem**
> "LLM agents forget between sessions, and most memory tools just append and retrieve.
> The hard parts are facts that *change*, forgetting what's *stale*, and recalling under
> a small context window. That's Tenet."

**0:20 – 0:50 — Supersession (the money shot)**
Terminal, run the e2e test or a live snippet:
```python
m.ingest("Hi, I'm Alex, I live in Montreal and I'm vegetarian.")
m.ingest("Update: I moved to Toronto. Also my manager is now Sarah Chen.")
m.recall("where does the user live and who manages them?")
```
> "Watch — I told it Montreal, then Toronto. It doesn't keep both. The old value is
> *superseded* — retired to history — so it answers Toronto, only. Same for the manager."
Show `stats()`: current=3, superseded=2.

**0:50 – 1:20 — Time-travel + forgetting**
```python
m.recall("where did the user live?", as_of=<before the move>)   # → Montreal
m.forget_sweep()                                                 # stale low-value archived
```
> "History isn't lost — I can ask what it believed *before* the move. And stale,
> low-value memories get forgotten automatically; pinned identity facts never do."

**1:20 – 2:00 — Bi-temporal + how it works**
Show `docs/architecture.svg` on screen.
> "Every fact has event time and transaction time. Write-time distillation turns raw
> messages into atomic keyed facts so updates supersede reliably. The read path has no
> LLM — pure vector plus decay — so recall is fast."

**2:00 – 2:35 — MCP + Qwen Cloud**
Show Claude Desktop (or the MCP tool list) using `learn`/`recall`; show `smoke_test.py`.
> "It's MCP-native — drop it into any MCP client and your agent has persistent memory.
> All powered by Qwen Cloud: distillation on qwen3.6-flash, retrieval on
> text-embedding-v4, reading on qwen3.7-plus."

**2:35 – 3:00 — Honest results + close**
Show the `docs/BENCHMARK.md` table.
> "On LongMemEval it's competitive with strong RAG on raw recall — but its real edge is
> answering with the *current* value of a changed fact, forgetting, and time-travel,
> which retrieval alone can't do. That's Tenet: memory that manages itself."
