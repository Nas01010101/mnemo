# Tenet vs the field (honest positioning)

How Tenet compares to the leading agent-memory systems (2026). Sourced from
`docs/SOTA.md`. **Read the benchmark caveat first — accuracy numbers are NOT
apples-to-apples.**

## ⚠️ Benchmark comparability caveat
The LongMemEval numbers below come from each project's own runs with **strong readers**
(Mem0/Zep: gpt-4o; Mastra: gpt-5-mini; agentmemory: Claude Opus 4.6) and their own
judge/protocol. Tenet's numbers use a `gpt-4o` reader (matching them) but a **small local `bge-small` embedder
and an unoptimized harness** — so our whole pipeline scores lower: our own **RAG baseline gets
only ~57%**, far below their 90%+ pipelines. The gap is embedder/harness, **not** the memory
design. We therefore compare Tenet only to **our own RAG under identical settings**, and on
**architecture/capabilities**; we do **not** claim an accuracy win over these systems.

## ⚠️ Frontier reality check — the 2026 reproduction crisis (verified 2026-07-14)
The 90%+ vendor numbers in the LongMemEval row below are **self-reported and do not
survive independent reproduction.** Two independent harnesses (Maximem, Bench'd, 2026)
found:
- **Mem0** claims **93.4%** LongMemEval; reproduces at **73.8% hosted / 32.4% OSS** — a
  20–60-point gap.
- **LoCoMo**'s answer key is **6.4% wrong** and its LLM judge **accepts up to 63% of
  deliberately incorrect answers** — so LoCoMo scores are structurally unreliable.
- A **no-memory** `gpt-4o-mini` baseline scores **57.6%** — several "memory systems" barely
  clear it.
- Honest, reproduced frontier: **Engram 83.6** (official category judge, bi-temporal +
  facts-plus-raw, arXiv:2606.09900), AutoMem 87, Supermemory 81.6.

**This is Tenet's positioning, not a footnote.** In a field where the headline number is
routinely inflated 20–60 points, Tenet reports **every** result with Wilson 95% CIs, ships
**four default-OFF flags that we measured as negative** (`RAW_RECALL`, `AGG_READER`,
`RETRACT`, `CONSOLIDATE`), and **falsified its own pre-registered churn claim in public**
(§4.8) before fixing it. We do not claim a cross-protocol LongMemEval win; we claim
**standardized, apples-to-apples wins** (MemoryAgentBench FactConsolidation **86.5 SH**,
official SubEM + prompt, *above* the published mini-tier SOTA 78.0; MAB-AR **59.3**, 2nd of
all published systems) plus a real head-to-head where we control every variable (§A). The
brand is: **the memory system whose numbers you can actually reproduce.**

## Feature / capability matrix
| | **Tenet** | Mem0 | Zep / Graphiti | Letta (MemGPT) | Mastra OM | agentmemory |
|---|---|---|---|---|---|---|
| Substrate | vector + bi-temporal | vector + entity-link | **bi-temporal graph** | tiered (core/archival) | observational notes | retrieval ensemble |
| Bi-temporal (valid + txn time) | ✅ | ❌ (create ts only) | ✅ | ❌ | ❌ | ❌ |
| Supersession / auto-invalidation | ✅ | partial (LLM update) | ✅ | agent-managed | ❌ (append) | ❌ (append) |
| **Principled forgetting** | ✅ decay sweep | ❌ | ❌ | evict on overflow | ❌ | ❌ |
| **Surprise-gated writes** (bounded store) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| No LLM in read path | ✅ | ✅ | ✅ | ✅ (tools) | ✅ | ❌ (rerank) |
| **Human-readable memory** (open & audit what it knows) | ✅ `get_all()` belief state | ❌ opaque vectors | ❌ graph nodes | ❌ state blocks | partial (notes) | ❌ |
| **Mem0-compatible API** (`add`/`search`/`get_all`/`delete`) | ✅ | ✅ (native) | ❌ (graph API) | ❌ (runtime) | ❌ | ❌ |
| MCP-native | ✅ | partial | ✅ | ❌ | ❌ | ❌ |
| **Infra to run** | **none (`pip`: sqlite+numpy)** | vector DB | **graph DB (Neo4j/FalkorDB)** | agent server + Postgres | service | service |
| **Long-horizon churn tested** | ✅ (**100% at U=2/8/32**, tied-for-first with Mem0-style, dominates RAG/HippoRAG at U=32 — see head-to-head §A) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Time-travel (`as_of`) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| LongMemEval_S | **81.0%**¹ | 94.4% | 71.2% | n/a | 94.9% | 96.2% |

¹ **81.0%** = `qwen3.7-plus` reader (the shipped Qwen-Cloud stack), n=100, ≥ matched RAG (79.0%),
100% recall@10, 98.5% less context than full — `docs/lme_qwen_n100_result.txt`. The older
**57.5%** figure was a *deliberately weak-reader* efficiency operating point (gpt-4o reader +
bge-small embedder), where our own RAG baseline also scores 57.5% — i.e. absolute numbers there
reflect the light embedder/harness, not the memory
design. Recall@10 = 97.5%. Efficiency point = 52.5% at half the tokens (best acc/token).

## Where Tenet sits
- **Architecturally closest to Zep/Graphiti** (bi-temporal + supersession + MCP + no-LLM
  reads) but **lighter**: no knowledge graph. Mem0 *removed* its graph after finding it
  ran 3× slower / 2× tokens for a thin gain, so Tenet's vector+bi-temporal substrate is a
  deliberate, evidence-backed choice.
- **Zep-without-the-graph-DB.** Zep/Graphiti needs a graph database (Neo4j / FalkorDB / Kuzu)
  running to get bi-temporal correctness; Tenet gets the same temporal correctness from
  `sqlite + numpy` — **zero infrastructure, `pip install`**. For most teams the operational
  overhead of a graph DB *is* the deciding factor, and Tenet removes it.
- **Human-readable memory — the thing none of the big three have.** Mem0 stores opaque vectors,
  Zep stores graph nodes, Letta stores agent state blocks: in every case you cannot open a file
  and read what the agent believes. Tenet's memory is `subject::attribute → value` with explicit
  current/superseded status (`get_all()` / `list_beliefs()`) — directly auditable.
- **Drop-in for Mem0 users:** a Mem0-compatible CRUD surface (`add`/`search`/`get_all`/`delete`,
  optional `user_id` scoping) means Tenet slots in where Mem0 would, then adds the temporal
  correctness Mem0's flat create-timestamp store lacks.
- **Adds what none of them have:** **principled forgetting** (decay sweep + surprise-gated
  writes — a bounded, self-pruning store, not append-forever), plus optional LLM-free
  **staleness hints** (`tenet doubts` — learned P(still-valid) per attribute; annotation-only,
  never re-ranks).
- **Tests a regime none of them report:** long-horizon knowledge churn. On the harsher
  paraphrased, multi-attribute ChurnBench (`docs/BENCHMARK.md` §9/§14), current Tenet holds
  **100% at U=2/8/32** — tied-for-first with Mem0-style and dominating RAG/HippoRAG-v2 (which
  collapse to 30% at U=32). This **reverses the earlier §9 falsification** (pre-fix Tenet was
  46% / half-life <2): the 2026-07-10 supersession-firing fix closed it. See the controlled
  head-to-head table below (§A). The full falsification → fix history is reported, not hidden.
- **Optimises the frontier the leaderboard ignores:** accuracy *per token*. Tenet gives
  the best acc/1k-tokens of the approaches we ran (49 vs RAG 27 vs full-context 0.5) — and,
  via belief-anchored expansion, can spend that headroom to reach one-shot **accuracy parity
  with RAG at equal tokens**, so it dominates the frontier rather than trading off.

## Head-to-head benchmark results (controlled, same-harness) — 2026-07-10

The capability matrix above is architectural. This section is the **measured** cross-system
head-to-head: Tenet vs the reproduced competitor **methods** (Mem0-style, HippoRAG-v2-style,
MemAgent-style, CAR) as arms of the SAME harness — same reader, same embedder, same SubEM /
judge, same seeds, Wilson 95% CIs, API-failures-excluded. Method is the only variable, so
these ARE apples-to-apples (unlike the vendor leaderboard numbers, which are not — see the
"published" column further down). Reproductions live in `scripts/bench_baselines.py`,
`scripts/bench_churn.py`.

### A. ChurnBench — long-horizon churn (our thesis regime) · reader qwen3.7-plus, n=30/pt
A fact's value is updated `U` times across a distractor-laden history; ask the CURRENT value.

| U (updates/fact) | **Tenet** | RAG | Mem0-style | HippoRAG-v2-style |
|---|---:|---:|---:|---:|
| 2 | **100.0** [88.6,100] | 96.7 [83.3,99.4] | 90.0 [74.4,96.5] | **100.0** [88.6,100] |
| 8 | **100.0** [88.6,100] | **100.0** [88.6,100] | **100.0** [88.6,100] | **100.0** [88.6,100] |
| 32 | **100.0** [88.6,100] | 30.0 [16.7,47.9] | **100.0** [88.6,100] | 30.0 [16.7,47.9] |
| churn half-life | **32** | 8 | **32** | 8 |

**Tenet is tied-for-best (with Mem0-style) and dominates the retrieval baselines at extreme
churn** (100% vs 30% at U=32; CI-separated). Two things make this Tenet's regime: (1) it
**reverses the earlier §9 falsification** — pre-fix Tenet scored 46% / half-life <2 here; the
2026-07-10 supersession-firing fix (concrete-key distiller + embedding key-resolution) plus
the §9.1 read-time consistency fix is what closed it to 100%; (2) Tenet reaches parity with
Mem0-style while doing **LLM-free reads** — Mem0-style pays an LLM ADD/UPDATE call *per fact
at write*, Tenet resolves supersession at distill time and reads with pure vector + decay.
RAG and HippoRAG-v2 collapse at U=32 because the top-k physically fills with stale versions —
a failure no reader strength or graph can fix.

### B. MAB FactConsolidation — supersession under counterfactual updates · matched 7B, n=200 pooled
(qwen2.5:7b backbone for ALL arms — the matched-method table; SubEM, official prompt.)

| system | FC-SH | FC-MH |
|---|---:|---:|
| **Tenet** | **90.0** [85.1,93.4] | **36.0** [29.7,42.9] |
| CAR (published FC-SOTA, arXiv:2606.01435) | 87.5 [82.2,91.4] | 33.0 [26.9,39.8] |
| Mem0-style | 81.0 | 12.0 |
| HippoRAG-v2-style | 66.0 | 9.0 |
| MemAgent-style | 44.0 (n=25) | 16.0 (n=25) |

**Tenet leads every arm on both axes** — including CAR, the published FC-SOTA recipe (CIs
overlap on SH; Tenet edges MH). Ingestion-time supersession matches or beats the best
assembly-time aggregation while doing the work once at write, not every read.

### C. LoCoMo — verbatim multi-session recall (a regime we LOSE) · qwen-judged, n=500
| system | overall | temporal | single-hop |
|---|---:|---:|---:|
| RAG | **38.8** [34.6,43.1] | **28.8** | **51.6** |
| Tenet | 33.8 [29.8,38.1] | 18.3 | 46.2 |

Naive RAG **beats** Tenet here (paired McNemar p=0.031): LoCoMo rewards *verbatim* recall and
Tenet's distillation paraphrases the exact wording the answer key needs. Reported plainly.
Mem0/HippoRAG were **not** run same-harness on LoCoMo — their per-turn LLM ingestion over
LoCoMo's long histories is thousands of calls, outside this pass's budget (noted, not hidden).

### D. PersonaMem-v2 — implicit-persona MC with a blind control · n=485
| arm | overall | updated=True (retraction) |
|---|---:|---:|
| Tenet (fix ON) | 53.4 [49.0,57.8] | 71.0 [62.4,78.2] |
| RAG | 50.9 [46.5,55.4] | 74.2 [65.8,81.1] |
| BLIND (no memory) | 34.4 [30.3,38.8] | 35.5 [27.6,44.2] |

Memory is load-bearing (both arms ≫ blind's 34.4), and Tenet ≈ RAG overall (tie). The
supersession fix lifted Tenet's overall from 50.5→53.4; the retraction subset is flat because
`ask_to_forget` is a *deletion*, not a value-replacement.

### Published, NOT same-harness (vendor / paper numbers — different backbone, judge, embedder)
Shown side-by-side so readers see both our controlled reproduction AND the marketing numbers.
**Do not compare these columns to ours** — Mem0's LoCoMo 92.5 is gpt-4o-mini-judged, the
LongMemEval numbers use gpt-4o/gpt-5 readers, etc. Our matched-harness reproductions above are
the fair comparison.

| system | LongMemEval | LoCoMo | MAB FC-SH (paper) | MAB-AR | judge/backbone |
|---|---:|---:|---:|---:|---|
| Mem0 | 94.4 | **92.5** | 18 | — | gpt-4o / gpt-4o-mini judge |
| Zep / Graphiti | 71.2 | — | 7 | — | gpt-4o; bi-temporal graph |
| HippoRAG-v2 | — | — | 54 | 65.1 | graph + PPR |
| CAR (FC-SOTA) | — | — | 78.0 / 94.8 | — | gpt-4o-mini / gpt-4o |
| agentmemory | 96.2 | — | — | — | Claude Opus |
| Mastra OM | 94.9 | — | — | — | gpt-5-mini |
| **Tenet (ours)** | 57.5¹ | 33.8² | 90.0³ | — | ¹bge-small harness ²qwen-judged ³matched 7B same-harness |

### Where each system wins / loses (honest map)
- **Tenet WINS (same-harness):** MAB FactConsolidation SH **and** MH — beats all four
  competitor methods including published-SOTA CAR. Long-horizon churn — **tied-for-first**
  with Mem0-style (both 100% at U=32) while dominating RAG/HippoRAG, and doing it with
  **LLM-free reads**. Plus capabilities none report: bi-temporal `as_of`, principled
  forgetting, per-token efficiency.
- **Tenet TIES:** PersonaMem-v2 overall (≈ RAG); ChurnBench (= Mem0-style).
- **Tenet LOSES:** LoCoMo verbatim recall (RAG > Tenet, p=0.031); multi-hop chaining; the
  vendor LongMemEval leaderboard on *absolute* accuracy under a weak reader.

**On the two losses — where the ceiling actually is (measured, not spun):**
- **Multi-hop is a *reader* ceiling, not a memory one.** We measured it: `navigate()` already
  retrieves the facts a multi-hop question needs across hops (retrieval pool is not the
  bottleneck — BENCHMARK.md §684); the loss is the *reader composing* those facts into a chained
  answer. A graph (Zep's approach) does not fix a reader-composition limit, and it would cost the
  graph-DB infra Tenet deliberately avoids — so we do **not** build one. A stronger reader closes
  this; an opt-in `reason`/decompose mode (Self-Ask over the belief state) is the in-scope lever.
- **The "~57%" absolute is a *weak-reader* artifact, not the memory design.** Recall@10 is already
  **97.5%** — the right facts are in the context. With a **frontier reader** (gpt-5.5 / Gemini-3.5,
  clean un-batched), Tenet reaches **75–77.5%** (≥ matched RAG), and the 57.5% is a deliberately
  weak-reader *efficiency* operating point, not Tenet's accuracy ceiling. See BENCHMARK.md
  §"Reader-generality".

### Improvement follow-ups — three implemented and measured (all kept OFF), one open
The top three EV follow-ups were built (behind default-off flags) and measured; all are
**honest negatives**, and the negatives are themselves informative:

1. **CAR-style read-time `max(serial)` aggregation** (`src/tenet/aggregate.py`,
   `TENET_AGG_READER`, default OFF) — **clean null**: FC-MH 15.0 → 15.0 (n=20), LoCoMo
   29.0 → 28.0 (n=100). *Why:* Tenet's store is already conflict-free by construction —
   ingestion-time supersession keeps one current value per key, so there is rarely a
   duplicate-key group left for a read-time aggregator to collapse. Aggregation is
   **redundant with the belief-state design**, not additive; LoCoMo's loss is verbatim
   paraphrasing, which this does not touch. Kept OFF.
2. **Retraction / tombstone op** (`MemoryCore.retract()`, `TENET_RETRACT`, default OFF) —
   **measured regression** on the PersonaMem retraction subset: 67.7 → 50.8 (n=124, CIs
   nearly disjoint). *Why:* removing a forgotten fact strips the context the 4-way MC
   reader needs to recognize *that a retraction happened* vs. distractor options that
   assume a prior value. A semantically-correct "delete with no replacement" is not what
   this benchmark rewards. Deterministically correct (7 tests), kept OFF, flagged.

3. **Raw-turn-favored recall mode** (`MemoryCore.recall(raw_recall=True)`, `TENET_RAW_RECALL`,
   default OFF) — **measured regression** on LoCoMo, the one verbatim regime we lose. ON gives
   the raw source-turn pool priority up to the full top-`k` budget (facts backfill the
   remainder) instead of the default `k//2` cap. Result at n=100 (qwen reader, seed 0, only
   the flag differs): overall 28.0 → 24.0, temporal 9.5 → **0.0** (collapsed), single-hop
   40.0 → 38.2, multi-hop 16.7 → 11.1; paired McNemar OFF wins the discordant pairs 7–3
   (p=0.34, not significant alone but every category points the same way). Per the stop-gate,
   not escalated to n=500. *Why:* with a fixed `k`, forcing the shared pool raw-heavy
   **displaces** the distiller's session/date-carrying facts rather than adding capacity —
   sampled temporal misses under ON all returned "No information available." RAG's edge on
   LoCoMo is not "more raw text" but an **unshared** pool that is *entirely* raw with the full
   budget to itself; Tenet's shared belief+raw pool cannot replicate that structure by
   re-weighting without giving up the evidence that was pulling weight. Deterministically
   correct (5 tests), byte-identical default-OFF path, kept OFF, flagged.

Still open (not yet built): **hard-delete for extreme-churn keys** (low priority — we already
tie Mem0-style and it risks the `as_of` history win). The verbatim-recall gap is now
characterized as a **structural** RAG advantage (unshared full-budget raw pool), not a
tunable one — closing it would require a separate raw-only retrieval path, which trades away
the belief-state design that wins FactConsolidation.

## What Tenet does NOT claim
- Not SOTA on raw LongMemEval accuracy — agentmemory (96.2%), Mastra (94.9%), Mem0 (94.4%)
  lead with heavily-tuned retrieval pipelines; on our light bge-small harness even RAG only
  reaches ~57%, and we don't pretend otherwise.
- Not an accuracy *win* over a strong RAG at one-shot retrieval — with belief-anchored
  expansion Tenet reaches **parity** at equal tokens (57.5% = 57.5%), reported as parity given
  ≈±5–7pp reader noise, not a headline win.
- Multi-session synthesis is the one category still behind RAG (43 vs 57; documented, `docs/BENCHMARK.md` §6).

## The one-line positioning
> **Zep's bi-temporal correctness and Mem0's drop-in API — with zero infrastructure**
> (`pip install`, sqlite+numpy, no graph DB), a **human-readable** belief state you can open and
> audit, LLM-free reads, principled forgetting, and MCP-native — tuned for accuracy-per-token and
> long-horizon robustness rather than one-shot leaderboard accuracy.
