# Tenet vs the field (honest positioning)

How Tenet compares to the leading agent-memory systems (2026). Sourced from
`docs/SOTA.md`. **Read the benchmark caveat first — accuracy numbers are NOT
apples-to-apples.**

## ⚠️ Benchmark comparability caveat
The LongMemEval numbers below come from each project's own runs with **strong readers**
(Mem0/Zep: gpt-4o; Mastra: gpt-5-mini; agentmemory: Claude Opus 4.6) and their own
judge/protocol. Tenet's numbers are **off-Qwen validation with `gpt-4o-mini`** (the Qwen
quota was exhausted). A weaker reader alone moves LongMemEval scores by tens of points, so
**Tenet's 45% QA is not directly below their 90%+** — different reader, different protocol.
We therefore compare on **architecture and capabilities** (directly comparable) and are
explicit that we do **not** claim an accuracy win over these systems.

## Feature / capability matrix
| | **Tenet** | Mem0 | Zep / Graphiti | Letta (MemGPT) | Mastra OM | agentmemory |
|---|---|---|---|---|---|---|
| Substrate | vector + bi-temporal | vector + entity-link | **bi-temporal graph** | tiered (core/archival) | observational notes | retrieval ensemble |
| Bi-temporal (valid + txn time) | ✅ | ❌ (create ts only) | ✅ | ❌ | ❌ | ❌ |
| Supersession / auto-invalidation | ✅ | partial (LLM update) | ✅ | agent-managed | ❌ (append) | ❌ (append) |
| **Principled forgetting** | ✅ decay sweep | ❌ | ❌ | evict on overflow | ❌ | ❌ |
| **World-model efficiency** (surprise-gated writes) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| No LLM in read path | ✅ | ✅ | ✅ | ✅ (tools) | ✅ | ❌ (rerank) |
| MCP-native | ✅ | partial | ✅ | ❌ | ❌ | ❌ |
| Graph infra required | ❌ (light) | ❌ (removed theirs) | ✅ (heavy writes) | ❌ | ❌ | ❌ |
| **Long-horizon churn tested** | ✅ (100% @12 updates) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Time-travel (`as_of`) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| LongMemEval_S (own reader) | 45%¹ (gpt-4o-mini) | 94.4% (gpt-4o) | 71.2% (gpt-4o) | n/a | 94.9% (gpt-5-mini) | 96.2% (Opus 4.6) |

¹ off-Qwen validation, weak reader — see caveat. Recall@10 is 95% (= our RAG baseline).

## Where Tenet sits
- **Architecturally closest to Zep/Graphiti** (bi-temporal + supersession + MCP + no-LLM
  reads) but **lighter**: no knowledge graph. Mem0 *removed* its graph after finding it
  ran 3× slower / 2× tokens for a thin gain, so Tenet's vector+bi-temporal substrate is a
  deliberate, evidence-backed choice.
- **Adds two things none of them have:** (1) **principled forgetting** (decay sweep +
  surprise-gated writes — a bounded, self-pruning store, not append-forever), and
  (2) a **world-model efficiency** framing (predictive-coding: store only what isn't
  already predicted).
- **Tests a regime none of them report:** long-horizon knowledge churn, where Tenet holds
  100% and RAG-style retrieval collapses to 50%.
- **Optimises the frontier the leaderboard ignores:** accuracy *per token*. Tenet gives
  the best acc/1k-tokens of the approaches we ran (41 vs RAG 27 vs full-context 0.5).

## What Tenet does NOT claim
- Not SOTA on raw LongMemEval accuracy — agentmemory (96.2%), Mastra (94.9%) and Mem0
  (94.4%) lead there with far stronger readers; we can't match that on gpt-4o-mini and
  don't pretend to.
- Not better than a strong RAG at one-shot factual retrieval.
- Weak on multi-hop temporal synthesis (documented, `docs/BENCHMARK.md` §6).

## The one-line positioning
> Zep's bi-temporal correctness + Mem0's lightweight vector substrate + **forgetting and
> world-model efficiency neither has**, MCP-native — tuned for accuracy-per-token and
> long-horizon robustness rather than one-shot leaderboard accuracy.
