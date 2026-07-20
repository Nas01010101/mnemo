# Devpost form content — ready to paste

Assembled from `docs/hackathon/SUBMISSION.md` (the judging-criteria mapping, source of
truth for claims) and `README.md`. This file maps 1:1 onto Devpost's submission form
fields so you can copy each section into the matching box. Fill the two bracketed links
(video, and blog if entering the Blog Post Prize) after upload.

---

## Project name
Tenet

## Tagline (<60 chars)
Agent memory that stays true as your life changes.

*(58 chars — swap for "Bi-temporal agent memory that never goes stale." (49 chars) if
Devpost's field is stricter.)*

## Track
Track 1: MemoryAgent

---

## Inspiration
Every LLM agent with "memory" today is really just RAG over a transcript log —
append-and-retrieve. That's fine until a fact *changes*: you move cities, change jobs,
update a preference, and the agent still has both the old and new statement sitting in
its retrieval pool, sometimes surfacing the stale one. Track 1 names the three things
retrieval memory structurally doesn't solve — facts that change, forgetting what's gone
stale, and recalling under a small context window — and that's exactly the gap Tenet is
built to close. The name comes from the idea of memory as something that has to stay
*internally consistent through time*, the way a bi-temporal ledger does: what's true
*now* vs. what we *used to believe*, kept distinct and both queryable.

## What it does
Tenet is a memory engine — and a personal assistant built on it (`src/tenet/agent.py`) —
that remembers you across sessions and stays correct when your facts change:

- **Bi-temporal supersession.** Every fact carries event time (`valid_at`/`invalid_at`)
  and transaction time (`created_at`/`expired_at`). When a fact changes, the old value is
  *superseded* — retired to history, not overwritten — so current recall returns only
  the truth *now*, while `recall(as_of=…)` answers "what did I believe in March."
- **Write-time distillation.** Qwen (`qwen3.6-flash`) turns raw messages into atomic
  facts with a stable `subject::attribute` key so later updates reliably collide and
  supersede, instead of piling up as near-duplicate passages.
- **Timely forgetting.** A salience-weighted decay sweep archives stale, low-value
  memories while pinned identity facts are never forgotten.
- **Recall under a budget, with zero LLM calls on read.** `recall(char_budget=N)` fills
  to a token budget with pure vector similarity + closed-form decay math — no model call,
  low-millisecond latency — plus an adaptive multi-hop `navigate()` on the same read path.
- **MCP-native.** `learn` / `remember` / `recall` / `doubts` / `time_travel` /
  `forget_stale` / `memory_stats` — any MCP client (Claude Desktop, an IDE, another
  agent) gets persistent, self-managing memory with no glue code
  (`src/tenet/mcp_server.py`).

**How it's different — and what's honestly shared.** Bi-temporal validity is not our
invention: Zep/Graphiti (arXiv:2501.13956) and Engram (arXiv:2606.09900) also keep
valid-time + transaction-time and invalidate rather than delete. Tenet's originality is
in what it *removes* to deliver the same temporal correctness:
- **No graph database.** Zep/Graphiti runs on Neo4j/FalkorDB; Tenet is `pip install` —
  SQLite + numpy, embedded in your process.
- **No LLM in the supersession path.** Zep detects contradictions with an LLM judgment
  over candidate edges at ingest; Tenet's distiller emits a stable `subject::attribute`
  key once, and supersession is a deterministic key collision — $0, auditable,
  same-input-same-output.
- **No LLM anywhere on the read path** — recall, time-travel, budget-fill, forgetting
  and multi-hop `navigate()` are embeddings + closed-form math, low-milliseconds.
- **A belief state you can open and read** — `get_all()` returns
  `subject::attribute → value`, current vs. superseded — not opaque vectors or graph
  nodes — plus an evaluation discipline (Wilson CIs on every number, self-falsified
  claims, four features shipped default-OFF because we measured them as no-benefit)
  that the 2026 reproduction audits show is the field's actual gap.

One line: **Zep's temporal correctness, Mem0's drop-in API, and a belief state you can
actually read — with zero infrastructure.** Full honest matrix incl. Engram:
[`docs/COMPARISON.md`](../COMPARISON.md).

And the design isn't waiting for a user: our Track 3 entry **Majalis** (agent society,
github.com/Nas01010101/majalis) runs its shared belief board on Tenet's exact supersession
design — same keyed `entity::attribute` collision semantics, retire-never-delete, as an
independent in-process implementation — with four heterogeneous Qwen agents writing through
it in a live, deployed product. The mechanism is proven under real multi-agent load, not
just benchmarks.

## How we built it
Built on **Qwen Cloud end to end**, three distinct Qwen Cloud APIs each doing the job
it's best at, through one fail-loud provider layer (`src/tenet/config.py`) that swaps
Qwen/OpenRouter/Ollama by env var with zero code change:

- **`text-embedding-v4`** — retrieval (`config.embed_texts`).
- **`qwen3.6-flash`** — write-time distillation of raw messages into atomic
  `subject::attribute` facts (`src/tenet/distill.py`).
- **`qwen3.7-plus`** — the assistant's reader (`src/tenet/agent.py`).
- **MCP server** exposing the full bi-temporal surface, not just store/recall —
  including `doubts` (a staleness/confidence table over current facts) and `time_travel`
  (bi-temporal read as of an arbitrary past instant) — `src/tenet/mcp_server.py`.
- **A learned drift model, not a heuristic.** `dynamics.py` fits a closed-form
  Gamma-Lomax survival model *per key class* from the ledger's own supersession
  history (no hardcoded half-lives), plus a ripple term for correlated fact change.
- **Our own RTX-trained LoRA distiller**, for a fully local, air-gapped stack. Beyond
  swapping providers by env var, the write path's one LLM call (distillation) can run
  entirely offline on `tenet-distiller-1.5b-v2` — a Qwen2.5-1.5B model we LoRA-fine-tuned
  on an RTX 3080 to reproduce bi-temporal supersession with zero cloud calls
  (`scripts/distiller_lora/`). On a decontaminated held-out eval it hits 6/6 clean-churn
  supersessions, 0.0 fabrication, and 0.775 key-consistency — *beating* the cloud
  reference's own 0.707 (`docs/BENCHMARK.md` §10).
- Ships as a real product: `pip install tenet-memory` (once published), a CLI
  (`tenet chat/remember/recall/navigate/stats/doubts/sweep/timeline/export`), an HTTP
  API + a belief-ledger web demo (`src/tenet/static/index.html`), LangGraph `BaseStore`
  + LlamaIndex `BaseMemoryBlock` adapters, and a 2-page paper + full preprint in `paper/`.

## Challenges we ran into
**The falsified-gate story.** We pre-registered success gates for two ideas before
running them, specifically so a negative result couldn't get quietly reframed as a win
afterward — and both gates failed, honestly reported rather than hidden:

- **Confidence-routed reader spend (gate not met).** The hypothesis: per-fact `p_valid` +
  a relevance margin could route reader spend across cheap/expensive tiers and cut token
  cost. An 84-config threshold sweep over 120 questions found *no* configuration saving
  tokens within 2pp of baseline; reaching the pre-registered ≥40%-savings cut cost
  −9 to −12pp accuracy, past the −5pp kill trigger. Diagnosis: `p_valid` is a **currency**
  signal (is this fact still fresh) — orthogonal to the *relevance* errors that actually
  sink extractive routing (the rank-1 belief is often the wrong attribute, not a stale
  one). Lesson kept and enforced in code: **confidence stays annotation-only**, never
  reordering or filtering recall — a real invariant, regression-tested
  (`scripts/test_agent_uncertainty.py::test_ranking_invariant`) after an earlier version
  of us tried rank-demoting doubted facts and watched a benchmark collapse 100% → 33%.
- **Multi-hop decomposition (clean null).** A Self-Ask-style query decomposition arm
  scored *identically* to the baseline on FactConsolidation multi-hop (25.0 vs 25.0,
  two questions flipped and offset each other) — below the pre-registered +15pp bar, so
  we didn't spend the larger confirmatory run. Combined with an earlier `navigate()` A/B
  null, this triangulates to an honest conclusion: at this backbone tier,
  FactConsolidation multi-hop is **reader-reasoning-bound**, not retrieval-pool-bound —
  no amount of cleverer retrieval construction moves that number; the bottleneck is hop
  composition in the reader itself.

**Stale raw-turn leakage under real conversational churn.** Our first high-churn stress
test (ChurnBench, several attributes each updated many times via paraphrased,
non-templated statements) found already-superseded raw turns slipping past the
write-time filter and outvoting the current fact at read time. Fixed with a read-time,
key-scoped consistency check plus a currency-structured reader context — churn half-life
rose from <2 to 8 — reported as a partial, honest fix (still short of a flat-32
half-life), not a full close of the gap (`docs/BENCHMARK.md` §9–9.1).

## Accomplishments that we're proud of
- **Numbers you can actually reproduce — in a field where that's rare.** Independent 2026
  audits (Maximem, Bench'd) found the headline agent-memory scores don't survive
  reproduction: Mem0 claims 93.4% on LongMemEval but reproduces at **73.8%** (57.5% pre-update; ~49% effective in production), and LoCoMo's own answer key is 6.4% wrong with a judge that accepts up to 63% of
  wrong answers. Tenet is built the opposite way: **every number carries a Wilson 95% CI**,
  we **ship four flags default-OFF because we measured them as no-benefit** (`RAW_RECALL`,
  `AGG_READER`, `RETRACT`, `CONSOLIDATE`), and we **publicly falsified our own pre-registered
  churn claim** before fixing it. Every result reproduces from one command.
- **Beats published SOTA on the standardized benchmark.** MemoryAgentBench (arXiv:2507.05257)
  FactConsolidation single-hop: **97.0%** [94.8, 98.3] — above even the published
  *gpt-4o-tier* pooled result (94.8), CI excludes the mini-tier SOTA (78.0) by 17 points —
  and multi-hop **45.8%** [40.9, 50.6], **1.5× the published SOTA** (30.2; every published
  memory system ≤7) — on a *weaker* local-7B backbone with zero-LLM, deterministic
  ingestion. All 800 questions, official metric + prompt verbatim, Wilson CIs. These
  numbers follow an ingestion-keyer fix that **our own miss-file audit exposed** (pre-fix
  86.5/30.0; both runs preserved in `docs/factcon_results.json`).
- **Same-harness reproduction of four rival methods** (Mem0, CAR, HippoRAG-v2, MemAgent
  style) — Tenet leads every arm on both single- and multi-hop axes.
- **The regime RAG structurally can't scale to.** On a controlled knowledge-churn
  benchmark (a fact updated 2→12 times), naive RAG collapses 100%→50% past 8 updates;
  Tenet holds 100% throughout.
- **MAB Accurate-Retrieval: 59.3 average, 2nd of all published systems** (20+ points
  above Mem0/Zep/MemGPT), and beats every published memory framework on EventQA (70.7 vs 67.6; long-context baselines reach 82.6).
- **MAB Test-Time Learning: 77.2 average (n=500)** — above BM25 (75.4) and every
  published memory system (Zep 62.8, MemGPT 67.6, Mem0 32.4) on a *weaker*, $0 local
  7B reader; 3 of MAB's 4 competencies now covered.
- Best accuracy-per-token on LongMemEval_S (49.2 vs RAG's 27.4 per 1k tokens); on Qwen
  Cloud's own product reader (`qwen3.7-plus`, n=100) Tenet reaches **81.0% vs matched
  RAG's 79.0%** at 100% recall@10 and 98.5% less context than full history.
- **Beats ReMe — Alibaba's own agent-memory framework — head-to-head, 67% vs 34%** on
  LongMemEval_S n=100, running ReMe's released pipeline end-to-end as a black box
  (its own `auto_memory` ingest + BM25 retrieval) with the identical Qwen reader/judge
  for every arm; McNemar p ≈ 2×10⁻⁶, Tenet ahead on every question type. A supplementary
  arm running ReMe's own answering agent as-shipped lands at a statistically
  indistinguishable 37.0% (p=0.55) — the gap is not a protocol artifact
  (`docs/BENCHMARK.md` §15).
- We report the honest losses too: multi-session synthesis and multi-hop chaining are
  documented weak spots, not hidden — every number reproduces from one CLI command
  (`tenet bench run <name>`, `docs/BENCHMARK.md`).

## What we learned
- **A well-calibrated signal isn't automatically a *useful* one for the thing you want to
  use it for.** `p_valid` is a genuinely good currency estimate, but currency and
  relevance are different axes of error — conflating them (as our routing attempt did)
  burns accuracy for a savings that never materializes.
- **Pre-registering the gate *before* running the experiment is what makes a negative
  result trustworthy** — and worth reporting instead of quietly shelving.
- **Annotation-only confidence is a real architectural invariant, not a nice-to-have** —
  we broke it once (rank-demoting doubted facts), watched accuracy fall off a cliff, and
  it's now enforced by a monkeypatch-based regression test, not just a docstring.
- **A zero-LLM read path pays off exactly where retrieval systems fail hardest** — the
  high-churn regime — because the mechanism keeping answers correct (bi-temporal
  supersession) is deterministic bookkeeping, not something an LLM has to get right on
  every read.

## What's next
- **Close the multi-session synthesis gap** (still 42.9 vs RAG's 57.1) with
  session-diverse retrieval — guaranteeing evidence coverage across distinct sessions
  instead of only deepening the sessions the top-*k* already surfaced.
- **Publish to PyPI** (`tenet-memory` — currently install-from-source only) and wire the
  README badge from placeholder to live.
- **Wider-N validation of the local LoRA distiller** — current numbers are a small,
  deterministic probe (n=26 messages / 8 churn groups), directionally strong but not yet
  a production SLA.
- **A learned graph traversal for multi-hop chaining.** We tried the cheap read-time levers
  for the RULER multi-hop loss and both were measured negatives — BM25+dense RRF ties baseline
  gold-in-pool exactly, and Self-Ask query decomposition *hurts* on a strong reader (error
  propagation). The gap is genuine graph-traversal territory (HippoRAG-style PPR), a different
  architecture — and, notably, mostly a weak-reader artifact: on the shipped `qwen3.7-plus`
  reader baseline RULER-MH is already ~60.6%, CI-overlapping the graph leader's 66.

## Built with
Qwen Cloud (`qwen3.7-plus`, `qwen3.6-flash`, `text-embedding-v4`) · Model Context
Protocol · FastAPI · SQLite · NumPy · Alibaba Cloud OSS · LangGraph · Python

## Try it out
- **Code repository:** https://github.com/Nas01010101/tenet (public, MIT license visible
  in About)
- **Live demo, running on Alibaba Cloud:** https://tenet-demo-wrenarokun.ap-southeast-1.fcapp.run
  (Function Compute; belief-ledger UI at `/`, `curl .../health`)
- **Zero-key demo (one command, no API key):** `git clone https://github.com/Nas01010101/tenet
  && cd tenet && pip install -e ".[local]" && python examples/00_zero_key_demo.py` —
  measured 94 s clone-to-output on a warm pip cache (first install pulls ~1 GB of wheels)
- **MCP config:** `examples/03_mcp_client.md`

## Links
- **Demo video (≤3 min):** [YOUTUBE URL]
- **Architecture diagram:** `docs/architecture.svg` in the repo
- **Proof of Alibaba Cloud services/APIs:** `src/tenet/config.py` + `src/tenet/distill.py`
  + `src/tenet/memory.py` call `dashscope-intl.aliyuncs.com` (Alibaba Cloud Model
  Studio); `src/tenet/alicloud_oss.py` is the optional OSS proof file.
- **Live backend on Alibaba Cloud:** https://tenet-demo-wrenarokun.ap-southeast-1.fcapp.run
  — full deploy method + caveats: `docs/DEPLOY.md`.
- **Blog post (optional, Blog Post Prize):** [BLOG URL] — candidate: `docs/BLOG.md`
