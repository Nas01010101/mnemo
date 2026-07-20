# Tenet: Agent Memory as a Self-Consistent Belief State — Extended Abstract

**Anas** · Global AI Hackathon with Qwen Cloud (Track 1: MemoryAgent) · 2026
**Code:** https://github.com/Nas01010101/tenet · **License:** MIT · Full paper: [`paper/tenet.md`](tenet.md) / [`tenet.pdf`](tenet.pdf)

---

## Problem: memory-as-retrieval fails when facts change

Long-term memory for LLM agents is almost universally *retrieval over a growing log of
past turns*. That abstraction works for one-shot recall of static facts and fails,
silently, when facts **change**. We name this failure **knowledge churn**: as a fact is
updated over a long interaction, stale versions crowd the retrieval budget *k*, and once
updates exceed *k* the current value may not even be retrieved. On a controlled benchmark,
a strong RAG memory falls from **100% to 50%** current-value accuracy as one fact is
updated 2→12 times — and the collapse is identical under `gpt-4o-mini` and `gpt-4o`
readers, so it is *structural*, not reader weakness. The 2026 field agrees this axis is
where famous systems break: on MemoryAgentBench's conflict-resolution split, the original
table reports **Zep 7%, Mem0 18%, MemGPT 28%** single-hop, and <=7% multi-hop for every memory system in the table (long-context reasoning baselines reach 28) evaluated.

## Approach: a bi-temporal belief state, with no LLM in the read path

**Tenet** reframes agent memory as a **self-consistent belief state** — a compact,
supersession-aware set of current facts — over a plain sqlite + numpy substrate (no graph
database, no vector-DB service). Five mechanisms:

1. **Distillation into keyed beliefs** (write time, the one LLM call): each turn becomes
   atomic facts with a stable key `subject::attribute`. Keys, not similarity, drive
   supersession — we measure a value change ("14:20"→"09:45") at cosine **0.99**, *higher*
   than a mere rephrasing at 0.79, so no similarity threshold can separate the two.
2. **Bi-temporal supersession**: every fact carries event time (`valid_at`/`invalid_at`)
   and transaction time (`created_at`/`expired_at`). A changed value *supersedes* its
   predecessor — retired to history, not overwritten — giving time-travel
   (`recall(as_of=t)`) for free.
3. **Belief–evidence consistency**: raw turns echoing a *superseded* belief are retired
   from recall. This single rule lifts current-value accuracy **55% → 100%** in ablation.
4. **Surprise-gated writes** (predictive coding): observations the store already predicts
   are discarded — 15% of writes dropped with no accuracy change.
5. **Belief-anchored evidence expansion + saturation-gated `navigate()`**: spare context
   budget is filled with raw turns *only from sessions the belief state already surfaced*;
   multi-hop deepening continues only while newly reached evidence clears an
   embedding-only relevance-gain gate. Reads — `recall`, `doubts`, time-travel,
   `navigate` — are embeddings + closed-form math: **~11 ms, flat from 1k to 100k facts**,
   zero LLM calls.

The ledger doubles as training data for **fact dynamics**: a closed-form Gamma–exponential
survival model per key class (`residence` learns a slow hazard, `mood` a fast one) surfaces
per-fact confidence and an `uncertain_facts()` re-verification list; an opt-in 276k-param
GRU temporal-point-process (1MB numpy artifact, 106µs/query) beats the closed form on
planted non-memoryless structure (NLL 2.76→1.99, 5 seeds).

## Results (all reproducible: one CLI command per number)

**Standardized conflict resolution — MemoryAgentBench FactConsolidation (arXiv:2507.05257), all
800 questions, official SubEM + prompt, Wilson 95% CIs.** With deterministic zero-LLM keys
and a deliberately weak local 7B backbone: single-hop **97.0** [94.8, 98.3] — above even
the published gpt-4o-tier pooled result (94.8) — and multi-hop **45.8**, 1.5× the published
SOTA (30.2, CI excludes); same-harness naive-RAG scores 47.8 / 4.5 (2026-07-19 run after
an ingestion-keyer fix our own miss-file audit exposed; pre-fix 86.5/30.0 preserved in the
evidence artifact). In a backbone-matched reimplementation of four
published mechanisms (CAR, Mem0-style, HippoRAG-v2-style, MemAgent-style), Tenet leads
every arm on both axes. Head-to-head against a released framework run black-box through
its own pipeline (ReMe, reme-ai 0.4.1.1; identical Qwen reader/judge): **Tenet 67.0% vs
ReMe 34.0%** on LongMemEval_S n=100, McNemar $p \approx 2\times10^{-6}$ (matched RAG 64.0, blind 0.0).

**MemoryAgentBench Accurate-Retrieval** (~2,000 questions, 197K–534K-token contexts,
official metrics, matched reader): average **59.3** — second only to HippoRAG-v2 (65.1,
which runs LLM OpenIE over every token; Tenet's ingestion never calls an LLM), 20+ points
above Mem0 (32.6) and Zep (37.5), and ahead of the field on EventQA (70.7 vs 67.6, CI
excludes). RULER multi-hop is the honest loss (45 vs 66). On MAB **Test-Time Learning**
(n=500, official scoring) Tenet pools **77.2** — above BM25 (75.4) and every published
memory system (Zep 62.8, Mem0 32.4) on a weaker, zero-cost local 7B reader — covering
three of MAB's four competencies.

**LongMemEval_S, on the shipped Qwen Cloud stack** (`qwen3.7-plus` reader, n=100):
Tenet **81.0%** vs matched RAG 79.0%, at **100% recall@10** and **98.5% less context**
than full history — winning multi-session (75.0 vs 54.2) and temporal reasoning (80.0 vs
73.3). Frontier off-Qwen readers agree directionally (gpt-5.5: 77.5 vs 75.0;
Gemini-3.5-flash: 75.0 vs 70.0). Retrieval is saturated (recall@10 = 97.5–100%), so
absolute accuracy tracks reader strength; under a frozen weak reader Tenet still traces
the best accuracy-per-token frontier (49.2 acc/1k tok at half of RAG's context, 1.6×).

**Knowledge churn (reported without a strawman)**: on the single-attribute primitive Tenet
holds 100% vs RAG's 50% — but that primitive is pre-registered to favor Tenet, so we also run
the harder multi-fact **ChurnBench**, where our own claim was *falsified* then recovered to
**half-life 32** (~82% at U=32). We state the loss plainly: an *idealized* delete-outright
Mem0-style arm stays flat 100 there — **Tenet does not beat it on raw churn accuracy.** Tenet's
real edge over Mem0 is that (a) the *real* `mem0ai` package doesn't delete-outright — it
accumulates stale copies and loses a live head-to-head — and (b) Tenet keeps a queryable belief
history + time-travel that delete-outright discards. Porting delete-outright into Tenet
(`TENET_CONSOLIDATE`) was a measured **no-benefit** (default-OFF, a fourth measured-negative).

**Local write path**: a LoRA-tuned Qwen2.5-1.5B distiller runs the full
learn→supersede→doubt loop with zero cloud calls — 6/6 clean-churn supersessions, 0.0
fabrication, and key-consistency **0.775 > the cloud reference's 0.707** on a
decontaminated eval (probe-scale n, reported as such).

## Honesty ledger

Falsified-then-fixed (ChurnBench), measured nulls (confidence-routed reader compute;
adaptive navigation at the strong-reader tier), and standing losses (LoCoMo verbatim
recall 33.8 vs 38.8; RULER multi-hop; extreme-churn consolidation) are all reported with
CIs, not hidden. Three default-OFF flags exist because we measured them as negative.

## Why it matters for the Qwen ecosystem

Tenet is the complement to Alibaba's own 2026 memory line (QwenLong-L1.5, AgeMem, ActMem),
which spends heavily at training, ingestion, or read time: Tenet gets its structural wins —
conflict resolution, churn robustness, auditable human-readable beliefs, ~2K read tokens
per query — with **zero-LLM reads and a `pip install` substrate**, shipped as a
Mem0-compatible API (`add`/`search`/`get_all`/`delete`), an MCP server, a LangGraph
`BaseStore`, an HTTP API, and a Qwen-Cloud-powered assistant.

---

<sub>Every number above links to a reproduction command in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md); runs are logged with git-sha to
`data/bench_runs.jsonl` (`tenet bench run <name>`).</sub>
