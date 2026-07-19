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
**MemoryAgentBench** (arXiv:2507.05257) **FactConsolidation** — the conflict-resolution axis where
famous systems collapse (Zep 7%, Mem0 18%, MemGPT 28%) — Tenet scores **86.5% single-hop,
above the published state of the art (78.0)**, and ties multi-hop SOTA (30.2), using a
*weaker* backbone and zero-LLM ingestion (official metric + prompt verbatim, all 800
questions, Wilson CIs). On MAB **Accurate-Retrieval** it averages **59.3 — 2nd among the
published memory frameworks we compare to** (behind HippoRAG-v2's 65.1; 20+ points above
Mem0/Zep/MemGPT) and **beats every published memory framework on EventQA (70.7 vs 67.6; long-context baselines reach 82.6)**. We also reimplemented
four rival paper methods (Mem0, CAR, HippoRAG-v2, MemAgent) in the same harness: **Tenet
leads every arm on both axes**. On our controlled knowledge-churn *primitive* (one templated
attribute), **RAG collapses 100%→50% while Tenet holds 100%** — while on the harsher
*paraphrased* multi-attribute ChurnBench a read-time consistency fix (now default-on) lifts
Tenet from worst-arm to 98/92/82 at U=2/8/32, with delete-outright consolidation still
leading at extreme churn (`docs/BENCHMARK.md` §9). On
LongMemEval_S Tenet has the best accuracy-per-token (49.2 vs RAG 27.4 per 1k tokens),
and on Qwen Cloud's own product reader (`qwen3.7-plus`, n=100) reaches **81.0% vs matched
RAG's 79.0%** at 100% recall@10 and 98.5% less context than full history — winning the
multi-session (75.0 vs 54.2) and temporal-reasoning (80.0 vs 73.3) categories.
And in a direct head-to-head against **ReMe — Alibaba's own agent-memory framework, run
as a black box through its own released pipeline** — Tenet scores **67% vs ReMe's 34%**
on LongMemEval_S n=100 (same Qwen reader/judge for every arm, McNemar p ≈ 2×10⁻⁶,
Tenet ahead on every question type; a supplementary arm running ReMe's own answering
agent as-shipped lands at a statistically indistinguishable 37.0%, p=0.55 — the gap is
not a protocol artifact; `docs/BENCHMARK.md` §15).
Honest weak spots — multi-session synthesis and multi-hop chaining — are reported, not
hidden. Every number reproduces from one documented command: `docs/BENCHMARK.md`.

**What's novel.** Memory as a *self-consistent belief state* instead of a document log:
ingestion-time bi-temporal supersession, a belief–evidence consistency rule (stale raw
evidence of a superseded belief is retired — none of the systems we compare to documents this mechanism), surprise-gated
writes, and an LLM-free read path — shipped as a pip package (`pip install tenet-memory`),
a polished CLI (`tenet chat/remember/recall/stats`), an MCP server, and an HTTP API,
with a 2-page paper + full preprint in `paper/`.

## Built with
`Qwen Cloud` (qwen3.7-plus, qwen3.6-flash, text-embedding-v4) · `Model Context Protocol` ·
`FastAPI` · `sqlite` · `NumPy` · `Alibaba Cloud OSS` · `Python`

## Judging criteria mapping

### Technical Depth (30%) — sophisticated Qwen Cloud API + MCP integration
- **Three distinct Qwen Cloud APIs, each for the task it's best at**: `text-embedding-v4`
  (retrieval, `config.embed_texts`), `qwen3.6-flash` (write-time distillation into atomic
  `subject::attribute` facts, `distill.py`), `qwen3.7-plus` (the assistant's reader,
  `agent.py`) — all through one fail-loud provider layer (`config.py`) that swaps
  Qwen/OpenRouter/Ollama by env var with zero code change.
- **MCP server exposes the full bi-temporal surface**, not just
  store/recall: `learn`, `remember`, `recall` (annotated with learned `p_valid`),
  `doubts` (staleness/confidence table), `time_travel` (bi-temporal read — recall as
  of an arbitrary past instant), `forget_stale`, `memory_stats` — `src/tenet/mcp_server.py`.
- **A learned drift model, not a heuristic**: `dynamics.py` fits a closed-form
  Gamma-Lomax survival model *per key class* from the ledger's own supersession history
  (no hardcoded half-lives), plus a ripple term for correlated fact change; an opt-in
  neural GRU temporal-point-process (`dynamics_neural.py`, numpy-only inference, no
  torch dependency at runtime) swaps in via `TENET_DYNAMICS=neural`.
- **One BLAS matmul read path, zero LLM calls at query time** (`memory.py:recall`,
  see `docs/HARNESS.md` §3 for the 28–33× speedup over a per-row loop).
- **A fully local, air-gapped stack — including our own LoRA-tuned distiller.** Beyond
  swapping providers by env var, the write path's one LLM call (distillation) can run
  on `tenet-distiller-1.5b-v2`, a Qwen2.5-1.5B model we LoRA-fine-tuned on an RTX 3080
  to reproduce bi-temporal supersession offline: on a decontaminated held-out eval it
  hits 6/6 clean-churn supersessions, 0.0 fabrication, and 0.775 key-consistency —
  *beating* the cloud reference's own 0.707 (`docs/BENCHMARK.md` §10,
  `scripts/distiller_lora/`). These are deterministic point estimates on a small probe
  set (n=26 messages / 8 paraphrase groups, no CIs), not a production SLA. Paired with
  local embeddings, `learn`→`supersede`→
  `doubts`→time-travel runs with zero cloud calls, network off included.

### Innovation (30%) — non-trivial logic, modularity, error handling
- **Memory as a self-consistent belief state, not a document log**: bi-temporal
  supersession (`memory.py:store`), belief–evidence consistency (a raw slice echoing a
  *superseded* belief is retired from recall, `_STALE_ECHO`), and surprise-gated writes
  (predictive-coding: redundant observations aren't stored) — see `docs/ARCHITECTURE.md`.
- **Annotation-only confidence, a real invariant, enforced and regression-tested.**
  Confidence never reorders or filters recall — only how a fact is *worded*. The story:
  rank-demoting doubted facts broke the knowledge-churn benchmark **100% → 33%**;
  restoring annotation-only brought it back to **100%**. Enforced in
  `memory.py`/`agent.py`/`mcp_server.py` and asserted by
  `scripts/test_agent_uncertainty.py::test_ranking_invariant` (monkeypatches `p_valid`
  to two wildly different regimes, asserts identical `recall()` order).
- **Anticipatory verification, not forced interrogation**: at session start the agent
  checks `uncertain_facts()` and injects one soft system-prompt line naming the
  top-3 doubted beliefs to *proactively* re-confirm "when contextually relevant" —
  never a scripted per-turn challenge. Empty store → zero-overhead no-op, verified by
  `test_agent_uncertainty.py::test_agent_construction_guard`.
- **Modularity**: one `Tenet()` core shared verbatim by the CLI, the MCP server, the
  HTTP API, and the assistant — every surface is a thin adapter, no duplicated logic.
- **Error handling**: embedding failures raise (never silently degrade to a zero vector
  — a past incident scored a run at 5% that way, see `config.py`); Qwen chat calls
  back off and retry on rate limits; `mcp_server.py` imports cleanly and serves against
  an empty/nonexistent DB (`python -c "import tenet.mcp_server"`, item 8 sanity check).
- **Fail-loud on permanent provider errors, not silent zero-fact writes**:
  `config.chat` used to retry every exception then return `""`, so a bad key/quota
  outage looked identical to "the model said nothing" and `ingest()` silently learned
  0 facts. Now permanent errors (401/402/403, quota) raise `ProviderError` immediately;
  transient ones (429, upstream hiccups) still retry; every write surface
  (agent/CLI/MCP/API) surfaces the failure instead of pretending success — regression-
  tested end to end in `scripts/test_errors.py`.
- **Adaptive multi-hop recall (`navigate()`), staying LLM-free.** The Qwen paper *From
  Passive Retrieval to Active Memory Navigation* (NapMem, arXiv:2607.05794) proposes
  navigating memory as a structured action space, stopped by a GRPO-trained 9B policy
  — training + serving Tenet can't ship, and it puts an LLM in the read path. `navigate()`
  is the deterministic instantiation: it reuses `recall`'s existing belief-anchored
  expansion + associative hops, and replaces the learned stop policy with an
  embedding-based relevance-gain gate — deepen only while a hop surfaces genuinely new,
  relevant evidence, stop the moment it saturates. Wired into `MemoryCore.navigate` /
  `Tenet.navigate`, `tenet navigate`, and the MCP `navigate` tool tonight
  (`src/tenet/navigate.py`, `scripts/test_navigate.py`). Honest: mechanism-validated on
  a controlled bridge/saturation fixture, not yet benchmarked end-to-end — no numbers
  claimed here.

### Impact (25%)
- **Beats published SOTA on the standardized benchmark**: MemoryAgentBench (arXiv:2507.05257)
  FactConsolidation single-hop **86.5% pooled**, above published mini-tier SOTA (78.0),
  on a *weaker* local-7B backbone with zero-LLM ingestion — `docs/BENCHMARK.md` §6.
- **Same-harness reproduction of four published methods** (CAR, Mem0-style,
  HippoRAG-v2-style, MemAgent-style) — Tenet leads every arm on both single- and
  multi-hop axes, closest rival CAR at 87.5/33.0 vs Tenet 90.0/36.0 — `docs/BENCHMARK.md` §6.1.
- **The regime RAG structurally can't scale to**: on the templated churn primitive
  (a fact updated 2→12 times) — naive-RAG collapses **100% → 50%** past 8 updates,
  **Tenet holds 100%** throughout (`docs/BENCHMARK.md` §3, `scripts/bench_horizon.py`). On
  the harsher paraphrased ChurnBench (§9) that lead reverses to a falsified gate, recovered
  to 98/92/82 by the default-on read-time consistency fix — reported, not hidden.
- **LongMemEval-V2 case study** (web-agent trajectory memory, up to 115M-token
  haystacks): three LLM-free retrieval changes lifted gold-evidence recall from a naive
  port's ~12% to **59.7%** @48K budget, reader-gated rather than retrieval-gated —
  `paper/tenet.md` §4.7.
- **Ships as a real product**: `pip install tenet-memory`, a polished CLI
  (`tenet chat/remember/recall/navigate/stats/doubts/sweep`), an MCP server any client
  can plug into today, an HTTP API + belief-ledger web demo, and a 2-page paper + full
  preprint.
- **Drops into the LangGraph ecosystem, not just its own CLI**: `TenetStore`
  (`src/tenet/integrations/langgraph.py`) implements LangGraph's real `BaseStore`
  `batch`/`abatch` Op contract, so `StateGraph.compile(store=TenetStore(...))` gives any
  LangGraph agent bi-temporal supersession — a re-`put()` of the same `(namespace, key)`
  retires the old value to history instead of overwriting it — for free. Optional extra
  `pip install tenet-memory[langgraph]`, tested end to end in
  `scripts/test_langgraph_store.py`.

### Presentation (15%)
- **One-page architecture doc** with a Mermaid component diagram, the drift-model
  equations, and the annotation-only invariant story: `docs/ARCHITECTURE.md`.
- **Belief-ledger demo UI** (`src/tenet/static/index.html`) — chat on the left, the
  live belief state on the right with struck-through superseded history, a time-travel
  scrubber, and a faint dotted-underline "doubt" marker (hover for `p_valid`) on beliefs
  the drift model flags as likely stale.
- **`tenet doubts` CLI** renders the same staleness/confidence table as a Rich
  table (or plain-text fallback) for a live terminal demo.
- Every benchmark number reproduces from one documented CLI command
  (`tenet bench run <name>`, `docs/BENCHMARK.md`); honest weak spots (multi-session
  synthesis, multi-hop chaining) are reported, not hidden.
- **A 60-second, zero-API-key first run**: `pip install tenet-memory[local]` then
  `python examples/00_zero_key_demo.py` walks the whole LLM-free read path
  (supersession, time-travel, learned-dynamics doubts) with no signup and no network
  call — the lowest-friction way a judge can see the belief-state mechanism work.
- **`CHANGELOG.md`** (Keep a Changelog format) tracks every notable change since
  0.1.0; **`docs/BLOG.md`** (Blog Post Prize candidate) is the honest origin story —
  auditing LoCoMo's ground truth ourselves before trusting a leaderboard number, and
  why that pushed Tenet toward deterministic, reproducible evaluation.

## Links (fill in)
- **Code repository:** https://github.com/Nas01010101/tenet (public, MIT license visible in About)
- **Demo video (≤3 min):** [YOUTUBE URL]
- **Architecture diagram:** `docs/architecture.svg` in the repo
- **Proof of Alibaba Cloud services/APIs:** `src/tenet/config.py` + `src/tenet/distill.py` +
  `src/tenet/memory.py` call `dashscope-intl.aliyuncs.com` (Alibaba Cloud Model Studio);
  optional OSS: `src/tenet/alicloud_oss.py`
- **Live backend on Alibaba Cloud (bonus):** https://tenet-demo-wrenarokun.ap-southeast-1.fcapp.run
  (Function Compute, `ap-southeast-1`) — `curl .../health`, `docs/DEPLOY.md`
- **Blog post (optional, Blog Post Prize):** [BLOG URL]

## Submission checklist
- [ ] Public repo + LICENSE visible in About section
- [ ] Alibaba Cloud services used (DashScope) + proof file linked
- [ ] Architecture diagram (`docs/architecture.svg`)
- [ ] ≤3-min demo video on YouTube (public)
- [ ] Text description (above)
- [ ] Track identified (Track 1)
- [ ] (optional) blog/social post linked
- [x] (optional, for full "runs on Alibaba Cloud" credit) backend deployed to ECS/FC —
      live at https://tenet-demo-wrenarokun.ap-southeast-1.fcapp.run, see `docs/DEPLOY.md`

## Devpost form draft
Ready-to-paste content for every Devpost form field (tagline, inspiration,
what-it-does, how-we-built-it, challenges, accomplishments, what's-next, built-with):
[`docs/hackathon/DEVPOST.md`](DEVPOST.md).

## Deploy status (updated 2026-07-10)
**Live on Alibaba Cloud Function Compute** (`ap-southeast-1`), verified working:
**https://tenet-demo-wrenarokun.ap-southeast-1.fcapp.run** — `/health` returns 200, a
real `POST /ingest` distills via `qwen3.6-flash` and stores successfully, and `GET /`
serves the belief-ledger web demo UI, all live on Alibaba Cloud compute. Full
deploy method, the two runtime-version pitfalls hit along the way
(`custom.debian10`/`11`'s system Python is 3.7/3.9, too old for our deps —
`custom.debian12` ships 3.11 and works), and the one known caveat (ephemeral `/tmp`,
no OSS-backed persistence yet — OSS itself isn't activated for this account) are
documented in `docs/DEPLOY.md` "Current status". The "uses Alibaba Cloud services/APIs"
proof requirement (satisfied independently via DashScope) and this deployed-backend
bonus credit are now both covered.
