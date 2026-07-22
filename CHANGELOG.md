# Changelog

All notable changes to Tenet are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project is
pre-1.0, so minor versions may include breaking changes.

## [Unreleased]

### Added
- **LangGraph `BaseStore` adapter** (`src/tenet/integrations/langgraph.py`,
  `TenetStore`) — drop Tenet into any LangGraph agent as `store=`. Implements
  the real `batch`/`abatch` Op contract (`GetOp`/`PutOp`/`SearchOp`/
  `ListNamespacesOp`), with the same supersede-don't-overwrite semantics as
  `Tenet.ingest()`: a re-`put()` of the same key retires the old value to
  history instead of clobbering it, and `search(query=...)` carries the
  learned-dynamics confidence (`p_valid`) on each hit. Optional extra:
  `pip install "tenet-memory[langgraph] @ git+https://github.com/Nas01010101/tenet.git"`. See `examples/05_langgraph_store.py`.
- **Zero-key quickstart** (`examples/00_zero_key_demo.py`) — the full
  supersession / time-travel / doubts loop with no API key and no network,
  forcing the local embedder. The one thing that still needs a model is
  turning free-form conversation into keyed facts (`Tenet.ingest()`).
- **Adaptive memory navigation** (`navigate()`) — LLM-free, per-query-depth
  instantiation of active memory navigation (NapMem, arXiv:2607.05794):
  reuses belief-anchored expansion + associative hops, replacing a learned
  stop policy with an embedding-based saturation gate, so simple queries
  don't over-fetch and multi-hop queries aren't capped at a fixed depth.
- **ChurnBench** (`scripts/bench_churn.py`, `tenet bench run churnbench`) —
  a parametric high-churn stress test (5 keyed attributes, U∈{2,4,8,16,32}
  paraphrased updates) that **falsifies** §3's single-attribute churn result:
  on paraphrased multi-attribute churn Tenet was initially the worst arm
  (`docs/BENCHMARK.md` §9). Reported in full, gate marked falsified.
- **Local distiller** (`scripts/distiller_lora/`, opt-in
  `LLM_PROVIDER=ollama OLLAMA_MODEL=tenet-distiller-1.5b-v2`) — a LoRA-tuned
  Qwen2.5-1.5B that reproduces bi-temporal supersession fully offline (6/6
  clean-churn, 0.0 fabrication, 0.775 key-consistency on a small decontaminated
  probe; point estimates, no CIs). Default provider stays Qwen Cloud.

### Changed
- **`recall()` now defaults to read-time belief-evidence consistency**
  (`consistency_threshold=0.70`, `src/tenet/consistency.py`) — drops a raw
  slice close to a superseded fact whose key already has a current fact in the
  pool. Fixes ChurnBench stale-raw leakage: churn half-life <2 → 8 (`docs/BENCHMARK.md`
  §9.1). Opt out with `consistency_threshold=None` or `TENET_CONSISTENCY_DEFAULT=off`;
  all 7 deterministic regression suites pass with it on.

## [0.1.0] - 2026-07-09

Initial build for the Global AI Hackathon with Qwen Cloud (Track 1:
MemoryAgent) — a bi-temporal, self-managing memory core plus the surfaces
and evaluation needed to defend it honestly.

### Core memory (bi-temporal store, distillation, forgetting)
- `MemoryCore`: sqlite + numpy vector store with store/recall/forget_sweep,
  budgeted (`char_budget`) top-k retrieval, and dedup on near-identical
  restatements.
- **Bi-temporal fact model**: `valid_at`/`invalid_at` (event time) alongside
  `created_at`/`expired_at` (transaction time); `recall(as_of=...)` answers
  "what did we believe then," never destructively overwritten.
- **Keyed supersession**: write-time distillation (one LLM call) extracts
  atomic facts with a stable `subject::attribute` key, so a changed value
  reliably supersedes the old one instead of relying on embedding
  similarity alone (measured: a value change can score *higher* cosine
  similarity than a rephrasing of the same fact).
- **Hybrid index**: distilled facts (consistency, supersession) alongside
  raw verbatim slices (numeric/quantitative detail), with world-model
  stale-echo suppression so a raw slice can't reintroduce a superseded value.
- **Belief-anchored evidence expansion** (`expand=`) and **recursive
  associative recall** (`hops=`, ReContext-style replay): close the
  multi-session accuracy gap against flat RAG without flattening the
  accuracy-per-token frontier.
- **Surprise-gated writes**: a raw observation the store already predicts
  (cosine ≥ threshold) is skipped — measured 15% of turns dropped as
  redundant, no accuracy loss.
- Vectorized recall scoring (one BLAS matmul instead of a per-row Python
  loop; ~28-33x on the scoring step, docs/HARNESS.md §3).

### World model (learned fact dynamics)
- **Fact dynamics** (`dynamics.py`): closed-form Gamma-exponential survival
  model fit per key-class from the ledger's own supersession history, plus
  a co-supersession ripple matrix (when key A changes, correlated key B's
  confidence drops too). Fully closed-form, LLM-free, refit lazily.
- `Memory.confidence` / `uncertain_facts()` — `p_valid`, the learned
  probability a current fact is still true, surfaced on recall (annotation
  only, never a rank discount) and as a "worth re-verifying" doubts list.
- **Neural world model** (`dynamics_neural.py`, opt-in via
  `TENET_DYNAMICS=neural`): a 276k-parameter GRU temporal point process
  (Weibull hazard + next-key + contrastive next-value), trained on an RTX
  box, numpy-only inference (1MB npz, ~106us/query).
- Uncertainty-aware agent prompting and anticipatory verification
  (`agent.py`); doubts and time-travel exposed through the MCP surface and
  a faint doubt marker in the web demo.

### Benchmarks
- **LongMemEval_S**: retrieval recall parity with strong RAG (97.5% vs
  95%); an accuracy-per-token frontier (`--expand`) from 1.6x RAG's
  accuracy/token at half the context, up to parity with RAG's one-shot
  accuracy at equal-or-fewer tokens.
- **Long-horizon knowledge churn** (`bench_horizon.py`, templated
  single-attribute primitive): RAG collapses 100%→50% as a fact is updated
  more times than its retrieval budget can hold; Tenet holds 100% —
  supersession keeps exactly one current value regardless of churn. (The
  harsher paraphrased ChurnBench, §9, falsifies then partially recovers this;
  see [Unreleased].)
- **MemoryAgentBench (ICLR 2026) FactConsolidation** (`bench_factcon.py`):
  official SubEM metric + reader prompt, zero-LLM ingestion, weak local-7B
  reader. Single-hop 86.5% pooled — above the published mini-tier SOTA
  (78.0); multi-hop 30.2% pooled — ties it.
- **Same-harness reproductions** of four published memory mechanisms (CAR,
  Mem0-style, HippoRAG-v2-style, MemAgent-style) at matched backbone
  (`bench_baselines.py`) — Tenet leads every arm on both single- and
  multi-hop.
- **MemoryAgentBench Accurate-Retrieval** (`bench_mab_ar.py`): protocol-
  faithful per sub-benchmark against a gpt-4o-mini reader; EventQA beaten,
  RULER single-hop at parity, RULER multi-hop the one honest loss against
  HippoRAG-v2's graph-based chaining.
- Honest-limitations section (multi-session synthesis still behind RAG;
  reader-noise caveats) kept alongside every win. `tenet bench` CLI: one
  command per published number, logged to `data/bench_runs.jsonl`.

### Surfaces
- `Tenet` / `MemoryCore` Python API; `MemoryAgent` reference assistant
  (`agent.py`) with `/chat` and a demo loop.
- MCP server (`mcp_server.py`) exposing recall/ingest/doubts/time-travel as
  tools for Claude Desktop and other MCP clients.
- HTTP API (`api.py`, FastAPI) and a belief-ledger web demo (paper/ink
  editorial redesign, red-ink supersession stamps, time-travel scrubber).
- `tenet` CLI (`chat`/`remember`/`recall`/`stats`/`sweep`/`doubts`/
  `serve-*`/`bench`), rich output with a plain-text fallback.

### Packaging
- Restructured to a `src/`-layout pip package (`tenet-memory`), with
  optional extras (`api`, `mcp`, `oss`, `local`, `cli`) so a bare install
  stays dependency-light.
- Provider abstraction: Qwen Cloud (shipped default) / OpenRouter (chat
  fallback) / local `sentence-transformers` embeddings, so the same code
  runs off-Qwen for benchmark validation.
- Alibaba Cloud deploy artifacts + OSS (object storage) integration for the
  DashScope/AliCloud track proof.
- MIT license, community files (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY),
  submission package (README, ARCHITECTURE.md, BENCHMARK.md, COMPARISON.md,
  judging-criteria mapping), and both a 2-page and a full-length paper
  (`paper/tenet.pdf`, `paper/tenet_full.pdf`).
