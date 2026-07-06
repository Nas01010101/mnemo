# Tenet vs the field (honest positioning)

How Tenet compares to the leading agent-memory systems (2026). Sourced from
`docs/SOTA.md`. **Read the benchmark caveat first — accuracy numbers are NOT
apples-to-apples.**

## ⚠️ Benchmark comparability caveat
The LongMemEval numbers below come from each project's own runs with **strong readers**
(Mem0/Zep: gpt-4o; Mastra: gpt-5-mini; agentmemory: Claude Opus 4.6) and their own
judge/protocol. Tenet's numbers use a `gpt-4o` reader (matching them) but a **small local `bge-small` embedder
and an unoptimized harness** — so our whole pipeline scores lower: our own **RAG baseline gets
only 65%**, far below their 90%+ pipelines. The gap is embedder/harness, **not** the memory
design. We therefore compare Tenet only to **our own RAG under identical settings**, and on
**architecture/capabilities**; we do **not** claim an accuracy win over these systems.

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
| LongMemEval_S | 52.5%¹ | 94.4% | 71.2% | n/a | 94.9% | 96.2% |

¹ gpt-4o reader + bge-small embedder; our RAG baseline scores 65% on the same setup, so
absolute numbers reflect our light embedder/harness, not the memory design. Recall@10 = 97.5%.

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
- Not SOTA on raw LongMemEval accuracy — agentmemory (96.2%), Mastra (94.9%), Mem0 (94.4%)
  lead with heavily-tuned retrieval pipelines; on our light bge-small harness even RAG only
  reaches 65%, and we don't pretend otherwise.
- Not better than a strong RAG at one-shot factual retrieval.
- Weak on multi-hop temporal synthesis (documented, `docs/BENCHMARK.md` §6).

## The one-line positioning
> Zep's bi-temporal correctness + Mem0's lightweight vector substrate + **forgetting and
> world-model efficiency neither has**, MCP-native — tuned for accuracy-per-token and
> long-horizon robustness rather than one-shot leaderboard accuracy.
