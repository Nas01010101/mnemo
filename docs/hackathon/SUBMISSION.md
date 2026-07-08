# Devpost submission package

Copy-paste ready. Fill the two bracketed links after you push the repo + upload the video.

---

## Track
**Track 1: MemoryAgent**

## Project name
**Tenet — self-managing bi-temporal memory for AI agents**

## Elevator pitch (≤ 200 chars)
A personal AI assistant on Qwen Cloud whose memory stays true as your life changes — it
supersedes facts, time-travels to what you used to say, and forgets stale trivia.

## Text description
**Tenet is a personal AI assistant** (`src/tenet/agent.py`) that remembers you across sessions
and — unlike a chatbot bolted onto RAG — **stays correct when your facts change**. Move
cities, change jobs, update a preference, and it tracks the *current* truth; ask what you
used to say and it recalls the history; it forgets stale trivia on its own. It's powered
by **Tenet**, a memory engine built for the parts append-and-retrieve systems get wrong.

**The problem.** LLM agents forget between sessions, and memory layers bolted on top mostly
*append and retrieve* — they don't handle a fact that **changes** over time, **forgetting**
what's gone stale, or **recalling** under a small context window. Track 1 asks for exactly
those three.

**Tenet** is a memory service built around them:
- **Bi-temporal model** — every fact carries *event time* (`valid_at`/`invalid_at`) and
  *transaction time* (`created_at`/`expired_at`). When a fact changes, the old value is
  **superseded** (retired to history), not overwritten — so current recall returns only
  the truth *now*, while `recall(as_of=…)` can still answer "what did I believe in March".
- **Write-time distillation** — Qwen (`qwen3.6-flash`) turns raw messages into atomic
  facts with a stable `subject::attribute` key, so later updates reliably collide and
  supersede. A **hybrid index** also keeps raw verbatim slices so specific detail
  (durations, numbers) survives.
- **Timely forgetting** — salience-weighted recency decay; a sweep archives stale,
  low-value memories while pinned identity facts are never forgotten.
- **Recall under a budget** — `recall(char_budget=N)` fills to a token budget; the read
  path is pure vector + decay with **no LLM call**, so retrieval is fast.
- **MCP-native** — exposes `learn` / `recall` / `forget_stale` / `stats` so any MCP
  client (Claude Desktop, IDEs, other agents) gains persistent memory; also a FastAPI
  HTTP API.

**Built on Qwen Cloud** end-to-end: `text-embedding-v4` for retrieval, `qwen3.6-flash`
for distillation, `qwen3.7-plus` for reading — all via the OpenAI-compatible DashScope
(Alibaba Cloud Model Studio) API. Optional Alibaba Cloud OSS snapshots for durability.

**Evaluation — beats published SOTA on the standardized benchmark.** On
**MemoryAgentBench** (ICLR 2026) **FactConsolidation** — the conflict-resolution axis where
famous systems collapse (Zep 7%, Mem0 18%, MemGPT 28%) — Tenet scores **86.5% single-hop,
above the published state of the art (78.0)**, and ties multi-hop SOTA (30.2), using a
*weaker* backbone and zero-LLM ingestion (official metric + prompt verbatim, all 800
questions, Wilson CIs). On MAB **Accurate-Retrieval** it averages **59.3 — 2nd of all
published systems** (20+ points above Mem0/Zep/MemGPT) and **beats the field on EventQA
(70.7 vs 67.6)**. We also reimplemented four rival paper methods (Mem0, CAR, HippoRAG-v2,
MemAgent) in the same harness: **Tenet leads every arm on both axes**. On our controlled
knowledge-churn benchmark, **RAG collapses 100%→50% while Tenet holds 100%**; on
LongMemEval_S Tenet has the best accuracy-per-token (49.2 vs RAG 27.4 per 1k tokens).
Honest weak spots — multi-session synthesis and multi-hop chaining — are reported, not
hidden. Every number reproduces from one documented command: `docs/BENCHMARK.md`.

**What's novel.** Memory as a *self-consistent belief state* instead of a document log:
ingestion-time bi-temporal supersession, a belief–evidence consistency rule (stale raw
evidence of a superseded belief is retired — no prior system does this), surprise-gated
writes, and an LLM-free read path — shipped as a pip package (`pip install tenet-memory`),
a polished CLI (`tenet chat/remember/recall/stats`), an MCP server, and an HTTP API,
with a 2-page paper + full preprint in `paper/`.

## Built with
`Qwen Cloud` (qwen3.7-plus, qwen3.6-flash, text-embedding-v4) · `Model Context Protocol` ·
`FastAPI` · `sqlite` · `NumPy` · `Alibaba Cloud OSS` · `Python`

## Links (fill in)
- **Code repository:** https://github.com/Nas01010101/tenet (public, MIT license visible in About)
- **Demo video (≤3 min):** [YOUTUBE URL]
- **Architecture diagram:** `docs/architecture.svg` in the repo
- **Proof of Alibaba Cloud services/APIs:** `src/tenet/config.py` + `src/tenet/distill.py` +
  `src/tenet/memory.py` call `dashscope-intl.aliyuncs.com` (Alibaba Cloud Model Studio);
  optional OSS: `src/tenet/alicloud_oss.py`
- **Blog post (optional, Blog Post Prize):** [BLOG URL]

## Submission checklist
- [ ] Public repo + LICENSE visible in About section
- [ ] Alibaba Cloud services used (DashScope) + proof file linked
- [ ] Architecture diagram (`docs/architecture.svg`)
- [ ] ≤3-min demo video on YouTube (public)
- [ ] Text description (above)
- [ ] Track identified (Track 1)
- [ ] (optional) blog/social post linked
- [ ] (optional, for full "runs on Alibaba Cloud" credit) backend deployed to ECS/FC —
      see `docs/DEPLOY.md`
