# Design — Track 1: MemoryAgent

**Track chosen: Track 1 (MemoryAgent).** Rationale: the top-weighted criterion
(Innovation 30%) explicitly rewards **MCP integrations**, and a memory service is
the highest-leverage thing to expose over MCP. Plays to real strength; fastest path
to a *production-grade* (not toy) demo before the Jul 9 deadline.

## Product: **Tenet** — a self-managing memory service, exposed over MCP

One-liner: *Give any LLM client persistent memory that stores, forgets, and recalls
on its own — plug it into Claude Desktop / any MCP client, or call it over HTTP.*

The three things Track 1 explicitly asks for, and how we answer each:

| Track ask | Our mechanism |
|---|---|
| **Efficient storage & retrieval** | Qwen embeddings (`text-embedding-v3`) + sqlite-vec vector search; write-time dedup/consolidation |
| **Timely forgetting of outdated info** | Decay score = f(recency, access-frequency, pinned); background sweep archives low-score memories; supersession links so new facts retire old ones |
| **Recall of critical memories under limited context** | Context-budget-aware retrieval: top-k by relevance, then Qwen-based compression to fit a token budget the caller specifies |

### Why this wins points
- **Innovation (30%)** — exposed as a real **MCP server** (`store` / `recall` / `forget` tools); self-managing forgetting is the novel component vs mem0/Zep which mostly append.
- **Technical Depth (30%)** — clean layered architecture (MCP + HTTP API over one memory core), embeddings, vector index, decay algorithm, error handling.
- **Problem Value (25%)** — memory is *the* unsolved LLM-agent pain point; productizable + OSS-community-friendly.
- **Presentation (15%)** — architecture diagram + a benchmark number (recall@k, and "context saved %") + a crisp 3-min demo plugging it into a live MCP client.

## Architecture

```
                        ┌──────────────────────────────┐
   MCP clients ────────▶│  MCP server (stdio)          │
   (Claude Desktop,     │  tools: store/recall/forget  │
    any MCP host)       └───────────────┬──────────────┘
                                         │  (same core)
   HTTP clients ───────▶┌───────────────▼──────────────┐
   (curl / web demo)    │  FastAPI  (Alibaba Cloud)     │◀── Proof-of-deploy
                        └───────────────┬──────────────┘
                                         │
                        ┌────────────────▼─────────────┐
                        │  MemoryCore                  │
                        │  • embed (Qwen text-embed)   │
                        │  • sqlite-vec store          │
                        │  • decay / forget sweep      │
                        │  • budgeted recall + Qwen     │
                        │    compression               │
                        └───────────────┬──────────────┘
                                         │
                        ┌────────────────▼─────────────┐
                        │  Qwen Cloud (DashScope-intl) │
                        │  qwen3.7-plus + text-embed-v3│
                        └──────────────────────────────┘
```

## Deploy target (satisfies mandatory proof-of-deploy)
FastAPI backend on **Alibaba Cloud** — Function Compute (serverless, simplest) or a small
ECS instance. The `store`/`recall` endpoints run there; [`src/tenet/alicloud_oss.py`](../src/tenet/alicloud_oss.py)
is the linkable "code file that uses Alibaba Cloud services/APIs".

## Scope for 4 days (ponytail — least code that wins)
1. `MemoryCore` — embed, store, semantic recall, decay/forget. **(core, day 1)**
2. Tiny eval harness — recall@k on a synthetic multi-session set + "context saved %". **(day 1-2)**
3. MCP server wrapping the core. **(day 2)**
4. FastAPI + Alibaba Cloud deploy + proof recording. **(day 2-3)**
5. Architecture diagram, README, 3-min demo video, blog post. **(day 3-4)**

## Build order status
- [x] Track locked
- [x] MemoryCore — store/recall/forget/dedup/supersession, thread-safe (`src/tenet/memory.py`)
- [x] eval harness — 100% recall@3 vs 38% baseline, 94.2% ctx saved (`scripts/eval_recall.py`)
- [x] MCP server — remember/recall/forget_stale/memory_stats (`src/tenet/mcp_server.py`)
- [x] FastAPI backend built + tested (`src/tenet/api.py`) — **deploy to AliCloud still pending (needs AccessKey)**
- [ ] Alibaba Cloud deploy + proof recording  ← **blocked on user's AliCloud AccessKey**
- [ ] architecture diagram (render docs/DESIGN.md ascii → image)
- [ ] 3-min demo video + blog post
- [ ] Devpost submission (web form — manual/browser)
