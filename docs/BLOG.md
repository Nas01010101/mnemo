# Building Tenet: memory that forgets, on purpose

*My build journey for the Global AI Hackathon with Qwen Cloud (Track 1: MemoryAgent).*

Every "AI memory" demo shows the same trick: save a fact, retrieve it later. But that's
the easy 20%. The hard 80% is what happens *over time* — when a fact **changes**, when
old information should be **forgotten**, and when you have to **recall** the right thing
into a context window that's too small to hold everything. Track 1 asked for exactly
those three, so I built **Tenet** around them.

## The insight that shaped it
I started by reading the field: Mem0, Zep/Graphiti, Letta, the LongMemEval benchmark,
and 2026 results like Mastra's Observational Memory. One thing jumped out — from the Zep
work especially: **the benchmark number is downstream of the data model.** Systems that
track *when a fact is true* (a bi-temporal model with validity windows) beat vector-only
stores on temporal questions by ~15 points. So Tenet's core is bi-temporal: every fact
has **event time** (`valid_at`/`invalid_at`) and **transaction time**
(`created_at`/`expired_at`). When a fact changes, the old value isn't overwritten — it's
**superseded**, retired to history. Current recall returns only what's true now; a
`recall(as_of=…)` can still tell you what it believed last month.

## The bug that taught me the most
My first supersession used embedding similarity to detect "same fact, new value." Then I
measured it: a value change like "flight at 14:20" → "flight at 09:45" has **0.988**
cosine similarity — nearly identical — while a rephrasing of the *same* fact can drop to
**0.79**. Embeddings literally can't separate "restated" from "changed" from "rephrased."
That's *why* every serious system uses LLM extraction. So I added **write-time
distillation**: Qwen turns each message into atomic facts with a stable
`subject::attribute` key, and supersession keys on that. Reliable at last.

## The result that kept me honest
I ran the real LongMemEval_S benchmark expecting to show Tenet crushing naive RAG. It
didn't — on raw *retrieval recall*, a good embedding RAG is a strong baseline and edged
me out. Instead of hiding that, I dug in: retrieval recall is the wrong metric for a
*memory* system. RAG retrieves the top-k similar turns — which include both the stale and
the new value of a changed fact — and hands the reader a contradiction. Tenet supersedes,
so it serves only the current value. That's the capability that matters, and it's where
Tenet is *structurally* ahead.

## Built on Qwen Cloud
Distillation on `qwen3.6-flash`, retrieval on `text-embedding-v4`, reading on
`qwen3.7-plus` — all through the OpenAI-compatible DashScope API, which made wiring it up
a five-minute job. The read path has **no LLM call** (pure vector + decay), so recall
stays fast. And it's **MCP-native**: drop it into Claude Desktop and any agent gets
persistent, self-managing memory.

Memory that manages itself — stores what matters, retires what changed, forgets what's
stale. That's Tenet.

*Code: https://github.com/Nas01010101/tenet · Built for [Qwen Cloud Hackathon](https://qwencloud-hackathon.devpost.com).*
