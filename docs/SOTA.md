# Track 1 SOTA landscape + how we win (research memo)

> Compiled 2026-07-05 from primary sources (arXiv, official leaderboards, vendor
> research pages). Goal: understand the frontier, place our current build on it,
> and design the winning Track-1 entry.

## 1. The benchmarks that matter
| Benchmark | What it tests | Status |
|---|---|---|
| **LongMemEval_S / _M** (ICLR'25, `2410.10813`) | 500 Qs, ~115k-token histories, 5 memory abilities incl. knowledge-update + temporal | **The standard.** Public dataset + harness. |
| **LongMemEval-V2** (2026, `2605.12493`) | Web-agent trajectories, **25M–115M tokens**, 451 Qs, **latency-aware LAFS metric** | The new hard frontier; rewards accuracy **without making the agent wait**. |
| **LoCoMo** | 16–26k-token dialogues | Older, **saturated + criticised** (fits in context; scores don't transfer). |
| MemoryArena / Locomo-Plus / Hindsight / BEAM | newer variants | field still fragmenting, no consolidated standard. |

## 2. Current SOTA — LongMemEval_S, real retrieval (no oracle)
| Rank | System | Score | How |
|---|---|---|---|
| 🥇 | **agentmemory V4** (JordanMcCann) | **96.2%** (481/500) | Claude Opus 4.6, single deterministic pass. Solo, 16 days, $1k. |
| 2 | Chronos High (PwC) | 95.6% | enhanced config |
| 3 | **Mastra Observational Memory** | 94.87% (gpt-5-mini) | **stable, cacheable context — beats the oracle**, single pass, no rerank |
| 4 | Mem0 | 94.4% (vendor) | write-time LLM fact extraction, vector-first |
| 5 | OMEGA | 93.2% | — |
| 6 | Hindsight | 91.4% | 4 parallel retrieval strategies + neural rerank |
| — | Supermemory | 85.9% | single-pass |
| — | AutoMem | 87.0% | graph+vector, recall@5 97% |
| — | **Zep / Graphiti** | 71.2% | **bi-temporal knowledge graph — wins temporal by ~15pts** |
| — | Full context (no memory) | 60.2% | baseline |

## 3. What actually drives the score (the causal map)
1. **Temporal / bi-temporal data model is the single biggest lever.** Zep beats Mem0 by ~15pts on temporal purely because it stores fact-validity windows (`valid_at`/`invalid_at`) + transaction time (`created_at`/`expired_at`) and auto-invalidates contradicted facts. "The benchmark number is downstream of the data model."
2. **Write-time distillation** (Mem0): an LLM extracts atomic salient facts on every write → clean memories, cheap reads.
3. **Observational Memory** (Mastra, the most interesting 2026 result): maintain a **stable, prompt-cacheable working context** instead of injecting fresh retrieved chunks every turn. Result: beats the *oracle*, single pass, no reranking, and preserves KV-cache (others invalidate it every turn).
4. **Graph memory is losing its shine.** A-Mem/HippoRAG-2/Graphiti add associativity + multi-hop, but **Mem0 removed its own graph** — it lost single/multi-hop recall, ran 3× slower, cost 2× tokens for a thin temporal gain. Graph = expensive write, slow read, marginal.
5. **Latency is now first-class** (LME-V2 LAFS). No-LLM-in-the-read-path (Zep, Mastra) is the frontier-correct choice.

## 4. Where OUR current build sits
`src/memory.py` today = **vector (Qwen embed) + cosine + decay-forgetting + overwrite-with-newer supersession + budgeted recall**, over MCP+HTTP.
- Honest placement: architecturally ~**Supermemory / early-Mem0 tier** (vector-first, single timestamp). Our "100% on 8 synthetic facts" is a **toy** number — NOT comparable to LongMemEval and must not be presented as SOTA.
- **Gaps vs frontier:** no bi-temporal model, no write-time distillation, no stable-cacheable working set, no real-benchmark number, no latency measurement.
- **What we already do that the leaderboard leaders DON'T:** principled **forgetting** + **context-budget recall**. Most SOTA systems are append-only recall-maximisers.

## 5. The strategic gap → our thesis
**The LongMemEval leaderboard optimises pure recall accuracy. Track 1's rubric explicitly asks for three things the leaderboard ignores: efficient retrieval, *timely forgetting of outdated information*, and *recall under a limited context window*.** The frontier (LME-V2) is now moving toward exactly this — accuracy **per unit latency/context**. So the win is not to out-accuracy a 16-day Opus-4.6 effort; it's to own the axis the leaders neglect.

### Thesis: **frontier-optimal, self-managing memory**
> The first memory system engineered for the **accuracy–latency–forgetting** frontier: SOTA-informed recall, **no LLM in the read path**, principled forgetting + supersession, and a **stable cacheable working set** — benchmarked honestly on LongMemEval.

## 6. Winning architecture (upgrade path from what we have)
Keep the clean vector core; add the four proven levers + our differentiators:

| Upgrade | Source idea | Effort | Why it wins |
|---|---|---|---|
| **A. Bi-temporal fact model** — add `valid_at`/`invalid_at`/`created_at`/`expired_at`; supersession invalidates instead of overwriting (keeps history) | Zep/Graphiti | S (columns + logic) | biggest accuracy lever, esp. knowledge-update + temporal categories |
| **B. Write-time distillation** — Qwen extracts atomic facts + salience + a `valid_at` from raw turns before storing | Mem0 | M (1 Qwen call/write) | clean atomic memories; salience feeds forgetting |
| **C. Observational working set** — maintain a compact, stable, cacheable digest of high-salience live memories; served as a prefix so downstream LLM keeps KV-cache | Mastra OM | M | latency + the "beats oracle" effect; unique vs per-turn injection |
| **D. No-LLM read path + latency metric** — pure vector+recency scoring at read, measure P50/P95 retrieval latency | Zep, LME-V2 | S | frontier-correct; a number no toy demo reports |
| **E. Principled forgetting + a forgetting metric** (ours, extended) | Track-1 spec | S | **novel** — nobody benchmarks forgetting; we define + measure it |
| **F. MCP-native** (done) | hackathon | ✅ | Innovation-30% |

### Novel contribution (defensible, maybe paper-able)
1. **A forgetting benchmark.** The field has no standard forgetting metric. We define one (does the system drop *superseded/stale* facts while keeping live ones? measured as stale-recall↓ + live-recall↑) and report it. This is genuinely new and exactly the Track-1 ask.
2. **Frontier framing** — plot accuracy vs retrieval-latency vs context-tokens and show we dominate naive RAG / full-context on all three, the LME-V2 direction, with a self-managing (forgetting) store the leaders lack.

## 7. Validation plan (honest, budget-aware: $40 Qwen credit)
- Wire the **public LongMemEval_S** dataset + harness. Answerer + judge = Qwen (`qwen3.7-plus`); **report the protocol honestly** (Qwen-as-judge ≠ the gpt-4o leaderboard protocol, so our number is *indicative*, positioned against our own full-context + naive-RAG baselines, not pasted onto their leaderboard).
- Run a **budget subset** (100–150 Qs) first to fit credits; scale if credits allow.
- Report: overall accuracy, per-category (esp. knowledge-update, temporal), **retrieval P50/P95 latency**, **context tokens used**, and our **forgetting metric** — vs full-context and naive-RAG baselines we run ourselves.
- Target: clearly beat naive RAG + full-context, land credibly in the 80s on LME_S with Qwen, and be the only entry showing the forgetting + latency + context frontier.

## 8. Build order (revised, 4 days)
1. **A + B** (bi-temporal + distillation) — the accuracy core. *day 1–2*
2. **LongMemEval harness + baselines + first number.** *day 2*
3. **C + D** (observational working set + latency metric). *day 2–3*
4. **E** (forgetting metric) + final benchmark table. *day 3*
5. MCP polish, Alibaba Cloud deploy + proof, architecture diagram, 3-min demo, blog, Devpost. *day 3–4*

## Sources
LongMemEval `2410.10813` · LongMemEval-V2 `2605.12493` (LAFS) · Mem0 `2504.19413` · Zep `2501.13956` · A-Mem `2502.12110` · HippoRAG-2 `2502.14802` · MemoryOS `2506.06326` · "Memory is Reconstructed, Not Retrieved" `2606.06036` · Mastra OM (mastra.ai/research/observational-memory) · agentmemory V4 (github.com/JordanMcCann/agentmemory) · framework comparisons (jatinbansal.com, particula.tech, automem.ai — 2026).
