# Benchmarks & honest evaluation

Tenet is evaluated on the standard **LongMemEval_S** benchmark (500 questions,
~115k-token multi-session histories) plus controlled capability tests. Every number is
**honest and reproducible** from `scripts/`. Where a strong baseline beats us, we say so.

> **Off-Qwen validation protocol.** The Qwen free-trial quota was exhausted mid-evaluation,
> so these runs use a local embedder (`bge-small-en-v1.5`) + `gpt-4o-mini` reader via
> OpenRouter. This validates the **architecture** (Tenet vs baselines under identical
> settings); the shipped product uses Qwen Cloud (`text-embedding-v4` + `qwen3.7-plus`)
> via `config.py` — swap by env, no code change. Numbers are indicative, compared only
> against baselines we run ourselves — never pasted onto the public gpt-4o leaderboard.

## TL;DR
- **Retrieval recall: on par with strong RAG** (95% = 95% @k=10).
- **Best accuracy-per-token on LongMemEval** — Tenet matches most of RAG's answer quality
  on **half the reader context** and **99% less than full-context**.
- **Dominates the long-horizon regime**: as a fact is updated many times, RAG collapses
  (100%→50%); **Tenet holds 100%**. This is the regime long-term memory is *for*.
- **Honest weakness**: multi-hop temporal synthesis — compression currently costs
  accuracy vs RAG's raw turns. Documented as future work.

## 1. Retrieval recall — LongMemEval_S (`scripts/lme_recall.py`)
Session-level recall@10 over the full ~50-session haystack (n=20):

| System | recall@10 |
|---|---|
| naive-RAG | 95% |
| **Tenet** | **95%** |

Parity. (An earlier design scored 37% here; the hybrid index + stale-echo fixes below
closed the gap.)

## 2. QA answer-accuracy frontier (`scripts/lme_recall.py --qa`)
The metric production actually cares about — answer accuracy **per token of reader
context** (LongMemEval-V2's accuracy/latency direction). n=20:

| System | QA acc | reader tokens | **acc / 1k tokens** |
|---|---:|---:|---:|
| full-context (no memory) | 65% | 123,773 | 0.5 |
| RAG @k=10 | 60% | 2,193 | 27.4 |
| **Tenet** | 45% | **1,092** | **41.2** ← best |

Tenet delivers the **highest accuracy per token** — half of RAG's context, 99% less than
full history. On **raw** accuracy it trails RAG (45 vs 60); the gap is entirely in two
categories:

| question type | RAG QA | Tenet QA |
|---|---|---|
| single-session-user | 100% | 100% |
| knowledge-update | 67% | 67% |
| multi-session | 75% | 50% |
| **temporal-reasoning** | 33% | **0%** |

Recall is 100% on those types (the evidence *is* retrieved) — Tenet's compressed context
loses the fine detail multi-hop/temporal answers need. Honest limitation, §6.

## 3. Long-horizon knowledge churn — where memory structurally wins (`scripts/bench_horizon.py`)
A fact updated N times over a long history, retrieval budget k=6, 15 distractors,
12 independent principals per point:

| # updates | naive-RAG | **Tenet** |
|---:|---:|---:|
| 2 | 100% | 100% |
| 4 | 100% | 100% |
| 6 | 100% | 100% |
| 8 | 67% | **100%** |
| 10 | 58% | **100%** |
| 12 | 50% | **100%** |

**RAG loses 50 points; Tenet loses 0.** Once the number of stale versions exceeds the
retrieval budget, RAG's top-k can't hold them all and the reader picks a wrong (old)
value; Tenet's bi-temporal supersession keeps exactly **one** current value regardless of
how many times the fact changed. This is the long-term-memory regime RAG cannot scale to.

## 4. Knowledge-update + the world-model mechanisms (`scripts/bench_knowledge_update.py`)
The first version of this test *refuted* the naive design (55% correct, 45% stale-leak vs
RAG 95%): the hybrid raw-slice pool reintroduced values the fact layer had retired. The
fix is a **world-model consistency rule** — the current facts are the belief state; a raw
slice echoing a *superseded* belief is stale evidence and is retired from current recall
(`_STALE_ECHO`). That took Tenet 55% → **100%**, matching RAG (0% stale-leak).

**World-model memory efficiency** — **surprise-gated writes** (predictive-coding
principle): an observation the store already predicts (cosine ≥ 0.97) carries no
information and isn't stored. Measured: **15% of turns dropped as redundant, no accuracy
loss.** RAG stores everything.

## 5. Capabilities proven deterministically (`scripts/test_memory.py`, `test_tenet_e2e.py`)
Supersession · time-travel (`recall(as_of=…)`) · forgetting sweep · context-budget recall
· distillation-driven consistent keys — all pass without any benchmark, demonstrating the
core value directly.

## 6. Honest limitations
- **Not a better general retriever.** For one-shot factual retrieval, a well-tuned
  embedding RAG matches or beats Tenet on raw accuracy. Tenet's edge is efficiency
  (acc/token), long-horizon robustness, and capabilities RAG lacks.
- **Multi-hop temporal synthesis** is Tenet's weakest category (0% on LME temporal) —
  distillation compresses away the detail these need even though recall is 100%. Future
  work: query-aware raw expansion for temporal questions.
- QA numbers are off-Qwen (gpt-4o-mini reader); re-running on Qwen Cloud is a config flip.

## Reproduce
```bash
python scripts/test_memory.py ; python scripts/test_tenet_e2e.py     # capabilities
python scripts/lme_recall.py --limit 20 --k 10 --qa --seed 2         # recall + QA frontier
python scripts/bench_horizon.py --principals 12 --k 6 --updates 2,4,6,8,10,12   # long-horizon
python scripts/bench_knowledge_update.py --principals 4              # supersession + efficiency
# off-Qwen: prefix with  LLM_PROVIDER=openrouter EMBED_PROVIDER=local OPENROUTER_MODEL=openai/gpt-4o-mini
```
