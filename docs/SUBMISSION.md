# Devpost submission package

Copy-paste ready. Fill the two bracketed links after you push the repo + upload the video.

---

## Track
**Track 1: MemoryAgent**

## Project name
**Mnemo — self-managing bi-temporal memory for AI agents**

## Elevator pitch (≤ 200 chars)
A personal AI assistant on Qwen Cloud whose memory stays true as your life changes — it
supersedes facts, time-travels to what you used to say, and forgets stale trivia.

## Text description
**Mnemo is a personal AI assistant** (`src/agent.py`) that remembers you across sessions
and — unlike a chatbot bolted onto RAG — **stays correct when your facts change**. Move
cities, change jobs, update a preference, and it tracks the *current* truth; ask what you
used to say and it recalls the history; it forgets stale trivia on its own. It's powered
by **Mnemo**, a memory engine built for the parts append-and-retrieve systems get wrong.

**The problem.** LLM agents forget between sessions, and memory layers bolted on top mostly
*append and retrieve* — they don't handle a fact that **changes** over time, **forgetting**
what's gone stale, or **recalling** under a small context window. Track 1 asks for exactly
those three.

**Mnemo** is a memory service built around them:
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

**Honest evaluation (LongMemEval_S).** Recall@10 is on par with strong RAG (95%=95%).
Mnemo delivers the **best accuracy-per-token** (41.2 vs RAG 27.4 vs full-context 0.5 acc
per 1k reader-tokens) — matching most of RAG's answer quality on half the context. And in
the **long-horizon regime** — a fact updated many times — **RAG collapses (100%→50%) while
Mnemo holds 100%**, because supersession keeps exactly one current value instead of a pile
of stale ones. We're explicit about the weak spot: multi-hop temporal synthesis, where
compression costs accuracy. Full numbers in `docs/BENCHMARK.md`.

**What's novel.** A memory system engineered for the *accuracy-per-token* frontier and
*long-horizon robustness* rather than one-shot recall: bi-temporal supersession, a
world-model consistency rule (stale evidence of a superseded belief is retired), and
surprise-gated writes (predictive-coding efficiency) — exposed over MCP.

## Built with
`Qwen Cloud` (qwen3.7-plus, qwen3.6-flash, text-embedding-v4) · `Model Context Protocol` ·
`FastAPI` · `sqlite` · `NumPy` · `Alibaba Cloud OSS` · `Python`

## Links (fill in)
- **Code repository:** https://github.com/Nas01010101/mnemo (public, MIT license visible in About)
- **Demo video (≤3 min):** [YOUTUBE URL]
- **Architecture diagram:** `docs/architecture.svg` in the repo
- **Proof of Alibaba Cloud services/APIs:** `src/config.py` + `src/distill.py` +
  `src/memory.py` call `dashscope-intl.aliyuncs.com` (Alibaba Cloud Model Studio);
  optional OSS: `src/alicloud_oss.py`
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
