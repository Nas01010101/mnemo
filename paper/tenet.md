# Tenet: Agent Memory as a Self-Consistent World Model

**Anas** · Global AI Hackathon with Qwen Cloud (Track 1) · 2026

> **Code:** https://github.com/Nas01010101/tenet   **License:** MIT

---

### Abstract

Long-term memory for LLM agents is almost universally implemented as *retrieval over a
growing log of past turns* — a document-retrieval abstraction. We argue this is the wrong
abstraction for an agent that must model a changing world. We introduce **knowledge
churn** — the repeated updating of a fact over a long interaction — and show that
retrieval-augmented memory (RAG-memory) *silently degrades* under it: as the number of
stale versions of a fact exceeds the retrieval budget *k*, the reader is handed
conflicting values and answers incorrectly. On a controlled benchmark, a strong RAG-memory
falls from 100% to 50% current-value accuracy as a fact is updated 2→12 times.

We propose **Tenet**, which reframes memory as a **self-consistent belief state** — a
compact *world model of the user* — rather than a document store. Tenet (i) distills raw
turns into atomic, keyed facts; (ii) maintains a **bi-temporal** record so a changed fact
*supersedes* its predecessor (retired to history, not overwritten); (iii) enforces
**belief–evidence consistency** by retiring raw evidence that echoes a superseded belief;
(iv) applies a **predictive-coding write policy** — surprise-gating — that stores only
observations the model cannot already predict; and (v) closes the accuracy gap to raw
retrieval with **belief-anchored evidence expansion** — spending spare context on
query-relevant turns from the sessions the belief state already surfaced. Tenet holds
**100% current-value accuracy across all levels of the templated single-attribute churn
primitive** (where a strong RAG-memory falls to 50%; on the harder *paraphrased*
multi-attribute ChurnBench, §4.8, this claim was falsified, then fully recovered to
100/100/100 — half-life 32, tied-for-first with delete-outright Mem0-style, LLM-free —
by a read-time consistency rule plus write-side key-resolution, §4.9/§14), matches strong RAG on retrieval recall (95–97.5%), and — with expansion — **matches its
one-shot answer accuracy at equal-or-lower token budget** (57.5% vs 57.5% under a gpt-4o
reader) while retaining a high-efficiency operating point at **half the context** and the
**best accuracy-per-token** of the systems we evaluate. Tenet thus traces an
accuracy–efficiency *frontier* that meets RAG at its budget and beats it at every lower one.
We release all code and benchmarks.

---

## 1. Introduction

An agent that talks to a user for months does not need a transcript; it needs a **model of
the user** that stays true as the user changes. Yet the dominant memory design —
retrieval-augmented generation over stored conversation turns [Mem0; LongMemEval] — treats
memory as a document index: embed every turn, retrieve the top-*k* most similar at query
time, let the reader sort it out. This works well for one-shot recall of a *static* fact.
It fails, quietly, when facts **change**.

We formalize this failure as **knowledge churn**. Consider a user who moves cities several
times over a long relationship with an assistant. Each "I moved to *X*" turn is stored;
all are similar to the query "where do I live?", so the top-*k* fills with *stale* versions.
Once the number of updates exceeds *k*, the correct (latest) value may not even be
retrieved, and even when it is, the reader must infer recency from a pile of contradictory
statements. Accuracy collapses (§4.2).

The root problem is abstraction. A document store has no notion that "I live in Boston" and
"I live in Seattle" are the *same fact* with a *changed value*; both are just passages. We
argue memory should instead be a **belief state**: a compact set of *current* beliefs about
the world, each with a temporal extent, updated by observation, kept internally consistent,
and queryable across time. This is the stance world-model and predictive-coding accounts
take toward perception [Friston]; we bring it to agent memory.

**Contributions.**
1. We identify and name **knowledge churn**, a failure mode of retrieval-augmented memory,
   and give a controlled benchmark that exhibits it (RAG: 100%→50% as a fact is updated
   2→12×; §4.2).
2. We present **Tenet**, a memory that is a *self-consistent belief state*: bi-temporal
   supersession, a **belief–evidence consistency rule** (retire raw evidence of superseded
   beliefs), and a **surprise-gated (predictive-coding) write policy**.
3. We evaluate on LongMemEval_S and controlled tests: Tenet is **churn-robust (100% at all
   levels of the templated churn primitive; §4.2)** — a claim we then falsify and partially
   fix on a harder paraphrased ChurnBench (§4.8–4.9) — on par with RAG on recall (95%),
   **best-in-class on accuracy-per-token**, and —
   with belief-anchored evidence expansion — **at parity with strong RAG on one-shot accuracy
   at equal token budget**, closing a gap earlier belief-only compression left open.
4. On the standardized **MemoryAgentBench FactConsolidation** benchmark, ingestion-time
   supersession with zero-LLM deterministic keys **exceeds the published single-hop SOTA at
   the gpt-4o-mini tier (86.5 vs 78.0 pooled) and ties the multi-hop tier (30.2)** — using
   only a local 7B backbone, where the original benchmark's 22 systems score ≤60 / ≤7.

## 2. Related work

**Retrieval memory.** Mem0 [Chhikara 2025] distills salient facts at write time over a
vector store with entity links; it attaches only a *creation* timestamp and, notably,
*removed* its graph variant after finding it 3× slower / 2× tokens for a thin gain —
evidence we take seriously in choosing a light vector substrate. LongMemEval [Wu 2024]
is the standard long-horizon benchmark; its V2 [Wu 2026] adds a *latency-aware* metric,
signalling a field shift toward accuracy *per cost*, which our per-token results target.

**Temporal knowledge graphs.** Zep/Graphiti [Rasmussen 2025] maintain a *bi-temporal*
knowledge graph (valid + transaction time) with automatic invalidation — but pay heavy
per-write extraction and require graph infrastructure. Tenet keeps the bi-temporal
semantics without the graph.

**The 2026 bi-temporal convergence.** Concurrently with this work, several systems adopted
bi-temporal supersession: MemStrata [MemStrata 2026] applies a deterministic
(subject, relation, object) supersession rule over a bi-temporal ledger with no LLM in the
read path — and shows *similarity-threshold* supersession leaks stale values where
deterministic keying does not, independently corroborating our keyed design; Engram
[Engram 2026] pairs a bi-temporal knowledge graph with a hybrid facts-plus-raw-chunks read
path (converging on our dual-pool finding) and reaches 83.6% on LongMemEval_S under the
official judge; TOKI [TOKI 2026] gives contradiction resolution a formal bitemporal
operator algebra. On conflict-resolution benchmarks, [Freshness 2026] shows the assembly
step dominates: deterministic max(serial) aggregation after retrieval sets the current
SOTA on MemoryAgentBench FactConsolidation. Tenet differs from all of these in *where*
consistency is enforced — at ingestion (the store never contains a stale current value)
and at recall (belief–evidence consistency retires stale raw evidence) — and in coupling
the belief state to a write policy (surprise gating) and a budget-bounded evidence
expansion, evaluated on the knowledge-churn axis none of them report.

**Memory in the Qwen ecosystem.** Alibaba's own 2026 memory line takes the opposite
design point from Tenet: QwenLong-L1.5 [Shen 2026] reaches 76.4 on LongMemEval by
RL-training a 30B reasoner that reads up to 128K tokens per query; AgeMem [Yu 2026]
RL-trains the memory policy itself (Qwen2.5-7B backbone); ActMem [Zhang 2026]
(Alibaba-co-authored) reaches 75.6 with LLM-built causal graphs at ingestion. All three
spend heavily — at training time, ingestion time, or read time. Tenet reaches 60.0 on the
same benchmark at the same reader tier with **zero-LLM ingestion and ~2K read tokens per
query** (~79% of QwenLong-L1.5's score at under 2% of its read budget), and its structural
wins (knowledge churn, conflict resolution) survive on a 7B backbone.

**Active memory navigation.** A concurrent 2026 line treats memory access itself as an
agentic, learned process. NapMem [NapMem 2026] exposes a four-level memory pyramid (raw
conversation / typed records / topic tracks / user profile) as a structured action space
and trains a GRPO policy that descends the pyramid and stops once evidence is
sufficient — a trained 9B policy beats an untrained 397B model on memory benchmarks.
QwenLong-L1.5 [Shen 2026] additionally plans a navigational path over memory chunks,
lifting LongMemEval by +15.6 points. MemFlow [MemFlow 2026] is training-free: an LLM
router selects a memory tier per query, and an LLM Validator retries with a heavier tier
on a low-confidence read. MemReader [MemReader 2026] moves the same active-navigation
idea to the write side, extracting memory rather than passively logging it. AgeMem
[Yu 2026] and Mem-α [Mem-alpha 2026] RL-train the read/construction policy itself;
MemCog [MemCog 2026] reports a similar cognitive-navigation frame (affiliation
unverified). All of these make the *read* path agentic — an LLM or an RL policy decides
how deep to go and when to stop. Tenet takes the opposite bet on the read side: it keeps
recall LLM-free and gets adaptive depth from an embedding saturation gate (§3.8) instead
of a learned or LLM-mediated stop.

**OS-style and observational memory.** MemGPT/Letta [Packer 2023] page memory between a
context "RAM" and archival "disk", agent-managed. Mastra's Observational Memory maintains a
stable, cacheable summary. Both are largely append-oriented and do not model fact
supersession or forgetting as first-class operations.

**What is missing.** No prior system combines (a) bi-temporal supersession, (b) explicit
**belief–evidence consistency**, (c) **predictive-coding write-gating**, (d) principled
forgetting, and (e) **belief-anchored evidence expansion** in a light, graph-free store —
nor does any report the **knowledge-churn** regime.

## 3. Method

Tenet stores two layers over one bi-temporal table: a **belief layer** of distilled,
keyed facts, and an **evidence layer** of raw turns. Reads never call an LLM.

**3.1 Distillation into keyed beliefs.** Each turn is distilled by a small LLM into atomic
facts, each with a stable semantic key *κ = subject∷attribute* (e.g. `user∷residence`), a
salience *s ∈ [0,1]*, and an event time. The key is what makes supersession reliable:
embedding similarity cannot separate a *restated* fact from a *value-changed* one (we
measure a flight-time value-change "14:20→09:45" at cosine 0.99 — *higher* than a plain
rephrasing of the same fact at 0.79, so no similarity threshold distinguishes a changed
value from a restatement), but a shared key can.

**3.2 Bi-temporal supersession.** Every memory carries event time (`valid_at`,
`invalid_at`) and transaction time (`created_at`, `expired_at`). Storing a fact with key
*κ* whose value differs from the current fact at *κ* **supersedes** it: the old fact's
`invalid_at`/`expired_at` are set; it leaves the current set but remains in history.
Current recall filters `expired_at IS NULL`; `recall(as_of=t)` reconstructs the belief set
as of any past *t* (time-travel).

**3.3 Belief–evidence consistency (the key rule).** The evidence layer is what lets the
reader answer detail questions, but it is also where stale values hide: a raw turn "I moved
to Boston" survives even after the belief `user∷residence` moves on. We therefore retire,
from current recall, any raw slice *e* whose embedding is close to a *superseded* belief:

  exclude *e* if  max₍f ∈ expired beliefs₎ cos(e, f) ≥ τ_stale   (τ_stale = 0.80).

This single rule is what turns supersession from a fact-layer nicety into end-to-end
correctness: it took current-value accuracy from 55% to 100% (§4.3).

**3.4 Predictive-coding write policy (surprise-gating).** A world model stores *prediction
error*, not everything. On write, a raw observation *e* is discarded if the store already
predicts it — i.e. it is near-identical to an existing slice:

  store *e*  ⇔  max₍e' ∈ store₎ cos(e, e') < g_surprise   (g_surprise = 0.97).

This bounds the store and drops redundant repetition (§4.4) with no accuracy loss.

**3.5 Forgetting.** Each memory's rank is relevance × a decay factor
*d = 2^(−Δt/h) · (1 + log(1+uses)·β) · (0.6 + 0.8s)* (half-life *h* = 14 d); a sweep
archives current, unpinned memories with *d* below a threshold. Pinned identity facts never
decay. Retrieval is a **dual pool** — beliefs for consistency, evidence for verbatim detail
— guaranteeing each a share of the budget.

**3.6 Fact dynamics — learning how facts change.** The ledger doubles as training
data for drift: superseded facts are observed lifetimes, current facts censored ones.
Per key class, a conjugate Gamma–exponential model gives closed-form survival
P(still valid | Δt) = (β₀+T / (β₀+T+Δt))^(α₀+n) — "residence" learns a slow hazard,
"mood" a fast one, per user, no LLM. Co-supersession statistics add *ripple*: when a
correlated key changes, neighbors' survival is discounted, so one observation updates
beliefs about unobserved attributes. Both surface as a per-fact `confidence` and an
`uncertain_facts()` re-verification list. Deliberately annotation-only: rank-demoting
doubted facts measurably re-created the churn failure (§4.2) before we reverted it.
A pre-registered follow-up asked whether confidence could also route *reader compute*
(extractive/cheap/full tiers): it cannot — on 120 questions no threshold configuration
in an 84-point sweep saved reader tokens within 2pp of the 91.7% all-full-reader
baseline; confidence is a *currency* signal, orthogonal to the relevance errors that
dominate extractive mistakes. Calibration suffices for caveats and doubt surfacing, not
for compute routing. We report the null rather than shipping it.
A trained alternative — a 276k-param GRU temporal point process (Weibull hazard +
next-key + contrastive next-value heads) — beats the closed form decisively on
planted non-memoryless structure (NLL 2.76→1.99, CI excludes 0; next-value 0.997
recall@5 vs 0.05 chance; 5 seeds) but is honestly mixed on sparse real chains;
it ships as a 1MB numpy artifact (106µs/query), opt-in, defaults unchanged.

**3.7 Belief-anchored evidence expansion.** Compressing a session into a few keyed beliefs
is what wins churn and efficiency, but it can drop the fine detail a multi-hop question
needs, even when the right session *is* retrieved (recall is 95–100%, §4.1). We recover it
without reverting to flat retrieval: the top-*k* dual-pool result names both the belief state
*and the sessions it came from* (via each memory's `source`). Given spare context budget *B*,
Tenet fills it with up to *m* additional query-relevant **raw turns drawn only from those
already-surfaced sessions** — evidence anchored to the belief state, not the whole haystack —
subject to the same belief–evidence consistency filter (§3.3) so no stale value re-enters.
*B* is set to the baseline RAG budget, so expansion never spends *more* context than flat
retrieval; it is a knob (m=0 → the efficiency point; m large under budget *B* → the parity
point) that lets one system trace an accuracy–efficiency frontier rather than sit at a point.

**3.8 Saturation-gated navigation.** Belief-anchored expansion (§3.7) draws raw evidence
from a session boundary; associative multi-hop recall goes further by re-conditioning the
cue on retrieved evidence and re-scoring the store, but at a caller-fixed depth
`hops = N`. A fixed *N* over-fetches simple queries (extra hops add distractors that hurt
the reader) and under-fetches genuinely multi-hop ones. `navigate()` replaces the fixed
depth with an iterative, budget-bounded hop-deepening loop that reuses belief-anchored
expansion at every hop and adopts hop *d+1* only while its newly-reached evidence is
still informative:

  adopt hop *d+1*  ⇔  max₍e ∈ new at d+1₎ score(e) ≥ τ_gain   (τ_gain = 0.15),

where "new at *d+1*" is evidence reached at depth *d+1* but not at depth *d*. When no hop
clears the gate, or the hop budget (≤ 4, mirroring NapMem's tool-call budget) is
exhausted, navigation stops and returns the last adopted pool, dropping the marginal
hop's distractors rather than keeping them. The gate is an embedding-only
marginal-relevance test over scores Tenet already computes for recall, so it adds no LLM
call and requires no training: it is the training-free counterpart to NapMem's learned
stop and to MemFlow's LLM-Validator retry, running at embedding-search latency (~ms)
instead of an LLM round trip.

We validate the mechanism, not a benchmark number. On a deterministic, torch-free
synthetic store (hashed bag-of-words embeddings, no network call), `navigate()`
(a) reaches a bridge fact invisible to broad top-*k* retrieval via an associative hop,
and (b) stops early (hop 2) on a saturated single-hop query while continuing to hop 3 on
a genuinely multi-hop one, confirming the gate behaves as intended (multi-hop reach,
single-hop early-stop). A subsequent seeded A/B on FactConsolidation 6k cells at the
`qwen3.7-plus` tier (n=20 per cell, same reader, prompts, and seed; retrieval-pool
construction the only variable) measured *no benefit*: accuracy identical to default
recall on both cells, and −5 to −10pp against a fixed-4-hop arm, all within Wilson CIs.
Dumped trajectories confirm the mechanism executed as designed (pools differed from
baseline in 21/21 rows, adaptive depth 2–3): a sufficiently strong reader is robust to
the larger fixed pool, so adaptive early-stopping buys no answer precision at this tier.
We therefore report navigation as a mechanism contribution only, with a measured null
at the tested tier.

## 4. Experiments

**Protocol.** LongMemEval_S (500 questions, ~115k-token histories). We use a **`gpt-4o`
reader** (the same reader Mem0 and Zep report against), a local embedder
(`bge-small-en-v1.5`), and a cheap `gpt-4o-mini` distiller; the shipped system runs the same
code on Qwen Cloud (`text-embedding-v4`, `qwen3.7-plus`) by a config flip. Numbers are
compared only to baselines we run under identical settings. Baselines: **RAG** (top-*k* raw
turns) and **full-context** (entire history).

**4.1 Recall, and the accuracy–efficiency frontier** (n=40, k=10, gpt-4o reader).

| System | mode | recall@10 | QA acc | reader tokens | **acc / 1k tok** |
|---|---|---:|---:|---:|---:|
| full-context | — | — | ~65%* | ~124,000 | 0.5* |
| RAG | top-*k* turns | 95% | 57.5% | 2,101 | 27.4 |
| **Tenet** | efficiency (*m*=0) | **97.5%** | 52.5% | **1,067** | **49.2** |
| **Tenet** | parity (expansion) | **97.5%** | **57.5%** | 2,083 | 27.6 |

Tenet is a **frontier, not a point** (§3.6). At its **efficiency** operating point it answers
at **half the context** (1,067 tok) for the **best accuracy-per-token** of any system — 49.2,
1.6× RAG's 27.4 and ~100× full-context — at recall parity. Turning up belief-anchored
expansion, and capping context at RAG's own budget, brings raw QA accuracy to **parity with a
strong RAG (57.5% = 57.5%) at fewer tokens (2,083 vs 2,101)** — closing the one-shot gap that
belief-only compression left open. Tenet thus **meets RAG's accuracy at its budget and beats
it at every lower budget**; the one category still behind is *multi-session* synthesis
(42.9 vs 57.1, up from 28.6), where a question needs several evidence sessions but only some
are surfaced (§5). *(\*full-context under a weaker reader — 100× the tokens for no gain over
RAG; retrieval memory is essential.)*

The finding is **reader-robust**. On a cheaper `gpt-4o-mini` reader the parity point edges
ahead (Tenet 60.0 vs RAG 55.0 QA at the same budgets); the efficiency point's per-token
dominance holds across the `gpt-4o-mini` and `gpt-4o` readers we ran (≈1.6×).

**Reader-generality at the frontier tier (2026-07-11, un-batched of record).** The
identical captured task set (fresh n=40, seed 0, Qwen distiller) was run through 2026
frontier readers under one cross-family judge (`qwen3.7-plus`). A first pass batched 10
questions per CLI call; a methodology audit flagged that as contaminable, so we re-ran two
scriptable readers **one call per item** (`gpt-5.5`, `gemini-3.5-flash`) and re-judged —
those are the numbers of record. Batching had inflated absolute accuracy ~2–17pp; the
directional claims survive clean: Tenet ≥ RAG at both operating points on both readers
(77.5/77.5 vs 75.0 GPT-5.5; 75.0/72.5 vs 70.0 Gemini), ≈1.9× accuracy-per-token at half
context, and the multi-session weakness still *inverts* un-batched (RAG 50.0 vs Tenet
62.5 on both). Per-cell CIs overlap at n=40; the evidence is the directional replication
across two independently-isolated reader families. (Sonnet-5 ran via the batched subagent
harness only — kept for completeness, not load-bearing.) Artifacts: `docs/lme_unbatched_*.jsonl`.

**4.2 Knowledge churn (headline).** One fact updated *N* times amid distractors, k=6, 12
principals/point:

| updates N | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| RAG | 100 | 100 | 100 | 67 | 58 | **50** |
| **Tenet** | 100 | 100 | 100 | **100** | **100** | **100** |

RAG degrades monotonically once *N > k* (−50 pp); Tenet is flat at 100%. Supersession keeps
exactly one current value regardless of churn — the property a *belief state* has and a
document index cannot. **The curves are identical under a `gpt-4o-mini` and a `gpt-4o`
reader**: the failure is *structural* (once $N>k$ the latest value is not reliably retrieved),
so a stronger reader cannot rescue RAG — it is not an artifact of reader quality.

**4.3 Ablation — belief–evidence consistency.** On a controlled knowledge-update set,
removing the §3.3 rule drops Tenet to **55%** current-value accuracy with a **45%
stale-leak** (it answers with an outdated value); adding it restores **100%** / 0% leak.
This is the single most important mechanism.

**4.4 Efficiency — surprise-gating.** On histories with repeated statements, the §3.4 policy
discards **15% of observations** as redundant with **no accuracy change**, yielding a
bounded store where RAG grows unboundedly.

**4.5 Standardized conflict resolution — MemoryAgentBench FactConsolidation.** On the
ICLR 2026 conflict-resolution benchmark [Hu 2026] (SubEM metric and official reader prompt
verbatim; all 800 questions; Wilson 95% CIs), ingestion-time supersession with **fully
deterministic, zero-LLM keys** and a deliberately weak **local 7B backbone** scores:

| pooled (4 lengths, 6K–262K) | naive-RAG (same reader) | **Tenet** | published SOTA (mini / gpt-4o) [Freshness 2026] |
|---|---:|---:|---:|
| FC-SH (n=400) | 47.8 | **86.5** [82.8, 89.5] | 78.0 / 94.8 |
| FC-MH (n=400) | 4.5 | **30.2** [26.0, 34.9] | 30.2 / 51.5 |

Single-hop **exceeds the published gpt-4o-mini-tier SOTA (78.0; our CI excludes it) on a
weaker backbone**, and every system in the original benchmark table scores ≤60 (Zep 7,
Mem0 18, MemGPT 28); multi-hop exactly ties the mini-tier SOTA, where the original table's
best is ≤7. Accuracy barely degrades with haystack length (SH 89→81 from 6K→262K) because
the store is conflict-resolved at ingestion — the property assembly-time aggregation must
re-derive at every read. Multi-hop still degrades with length (42→20); reported honestly.

To remove the backbone confound entirely, we reimplemented four published memory
mechanisms as arms of the *same* harness (same 7B reader, embedder, SubEM, prompt;
6K+32K cells, n=200/axis): CAR [Freshness 2026] 87.5 SH / 33.0 MH, Mem0-style
consolidation 81.0 / 12.0, HippoRAG-v2-style OpenIE+PPR graph 66.0 / 9.0,
MemAgent-style overwrite memory 44.0 / 16.0 (n=25). **Tenet leads every arm on both
axes (90.0 / 36.0 on the same cells)** — ingestion-time supersession matches or beats
the best assembly-time aggregation while doing the consistency work once at write time.

**4.6 Accurate retrieval at zero ingestion cost — MemoryAgentBench AR.** On MAB's other
core competency (~2,000 questions, 197K–534K-token contexts; official per-benchmark
metrics: SubEM for RULER-QA, the official LLM-judge for LongMemEval(S*), choice accuracy
for EventQA; matched gpt-4o-mini reader), Tenet with **zero-LLM ingestion** (date-aware
structured chunks + embeddings only) averages **59.3** — second only to HippoRAG-v2
[Gutiérrez 2025] (65.1), which runs LLM OpenIE over every context token at ingestion,
and 20+ points above Mem0 (32.6), Zep (37.5) and MemGPT. Per sub-benchmark: EventQA
**70.7** (n=1,500; Wilson CI [68.3, 72.9] excludes HippoRAG-v2's 67.6), RULER SH-QA 75.0
(parity with 76), LME(S*) 46.3 vs 50.7, RULER MH-QA 45.0 vs 66 — the honest loss:
Personalized-PageRank graph traversal is genuinely stronger at multi-hop chaining over
narrative text. Together with §4.5, Tenet leads or ties the published field on two of
MAB's four competencies while being the only published memory framework in this comparison
whose ingestion never calls an LLM (the naive-RAG control aside).

**4.7 Case study — web-agent trajectory memory (LongMemEval-V2).** Adapting Tenet's
ingestion to LME-V2's web-agent trajectory haystacks [Wu 2026] (DOM states, actions;
up to 115M tokens), three LLM-free changes — structure-aware chunking, query cleaning,
and Qwen3-Embedding + BM25 hybrid retrieval (RRF) — lifted gold-evidence recall from a
naive port's ~12% to **59.7%** @48K budget (63.3% @72K; corpus ceiling ~92%) at ~0.25s
query latency. End-to-end accuracy is reader-gated, not retrieval-gated — and the gate is
the *reasoning mode*, not precision: 4-bit local 45.1% [39.5, 50.8], 8-bit 40.3%,
full-precision hosted *without* extended thinking 40.7% (over-abstains), all near the
same P(correct|gold)≈0.5–0.6 extraction ceiling; the leaderboard-default extended
thinking (~8K reasoning tokens/question) is what converts retrieval headroom, and lies
outside our compute budget. The substrate transfers; the binding constraint is
measurable and external to the memory.

**4.8 ChurnBench — a parametric stress test that falsifies our own churn claim under
natural updates.** §4.2's churn result uses one templated fact; ChurnBench sweeps
updates-per-fact U∈{2,4,8,16,32} over 5 keyed attributes updated via *paraphrased*
first-person conversation, n=50 questions/point, Wilson 95% CIs, real
`Tenet.ingest_session` (no hand-tuned keys). Pre-registered gate: Tenet half-life
(largest U with accuracy ≥90%) ≥2× the best baseline. **Result: falsified.** Tenet is
the *worst* of four arms at every U (half-life <2, vs RAG 8, HippoRAG-v2-style 8,
Mem0-style 32) — the opposite of the pre-registered hypothesis. Diagnosis: raw turns
for already-superseded values survive the write-time stale-echo filter when phrased
differently from their distilled paraphrase (cosine falls under the filter's
near-verbatim threshold), so the reader sees several conflicting statements with no
recency cue and often answers from a stale one — even when the correct fact is ranked
first. Mem0-style (which *deletes* superseded memories outright, leaving nothing to
leak) is immune to this failure mode and stays flat at 100% through U=32. Reported in
full in Appendix / `docs/BENCHMARK.md` §9. The fix and its measured effect: §4.9.

**4.9 The fix — read-time belief-evidence consistency + currency-structured context.**
Two additive, LLM-free, read-time changes (store/ingestion untouched, every ChurnBench
cache reused byte-for-byte). **(1)** A narrower, key-scoped consistency check
(`consistency.py`, `recall(consistency_threshold=...)`): drop a raw slice close to a
superseded fact whose *key already has a current fact in the pool* — more sensitive
than the global `_STALE_ECHO` filter without opening a cross-key false-positive
surface; current-fact ranking is unaffected either way. Threshold swept on real U=2
data (ground truth via exact substring match against the deterministic attribute
pools): 0.70 gives 100% stale-echo recall at a 7% false-positive rate (0.60: 81% FPR;
0.80, the original global threshold: only 20.9% recall — why it missed this case).
**(2)** Currency-structured reader context (`bench_churn.py`): "Current beliefs:"
(dated) then "Supporting raw context:" (dated) instead of one flat list —
product-faithful, since the store already knows which facts are current.

| U | tenet-baseline | tenet+1 | tenet+2 | **tenet+1+2** |
|---:|---:|---:|---:|---:|
| 2  | 60.0 | 90.0 | 82.0 | **98.0** |
| 8  | 42.0 | 84.0 | 82.0 | **92.0** |
| 32 | 36.0 | 80.0 | 70.0 | **82.0** |

Same cached stores, live qwen3.7-plus reader, n=50/point, Wilson 95% CIs
(`docs/BENCHMARK.md` §9.1). **Result: partial, pre-registered gate.** tenet+1+2 clears
≥90% at both U=2 (98.0) and U=8 (92.0) but falls short of Mem0-style's flat 100 at
U=32 (82.0) — the gate's "ship the fix, report half-life honestly" branch. Churn
half-life: tenet+1+2 = **8**, up from <2, now tied with RAG/HippoRAG-v2-style, still
behind Mem0-style's 32. All regression gates pass with the fix defaulted on (7
deterministic suites; §4.2's U=8 stays 100%; FactConsolidation sh_6k is provably
invariant — that arm never stores raw slices), so `recall()`'s
`consistency_threshold` now defaults to 0.70 in shipped code.

**4.10 Closing the write-path dependency: a local distiller.** Every result above distills
with Qwen Cloud (`qwen3.7-plus`) — the write path is Tenet's only remaining LLM dependency
(reads are already LLM-free). We probe whether a small model, LoRA-tuned and served fully
offline (ollama, one RTX 3080), can close it: LoRA SFT (bf16, rank 16, ~490 synthetic pairs)
of Qwen2.5-0.5B/1.5B-Instruct, labels from the cloud reference with keys
**force-canonicalized** per known attribute. An early candidate's fabrication rate of 1.0
traced to a data-balance bug (6% empty-target training examples, not a capacity limit);
rebalancing to ~22% empty-target dropped it to 0.08–0.17 with F1/key-consistency unchanged.
Evaluated on a **decontaminated** held-out set (novel values+phrasings, 0/66 overlap with
train — an earlier contaminated screen inflated key-consistency by ~0.17 and hid this
result):

| candidate | F1 | fabrication | key-consist. | clean churn (superseded/6) |
|---|---:|---:|---:|---:|
| qwen2.5-0.5b-instruct (untuned) | 0.295 | 0.333 | 0.30 | 0/6 |
| qwen2.5-1.5b-instruct (untuned) | 0.405 | 1.0 | 0.40 | 5/6 |
| tenet-distiller-0.5b-v2 (LoRA) | 0.867 | 0.167 | 0.75 | 3/6 |
| **tenet-distiller-1.5b-v2 (LoRA)** | 0.652 | **0.0** | **0.775** | **6/6** |
| qwen3.7-plus (cloud ref.)\* | 1.0 | 0.0 | 0.707 | — |

\*measured on the contaminated screen, not the decontaminated set — context, not
apples-to-apples. **Finding: key-consistency, not F1, is the load-bearing axis for
supersession.** The untuned baselines cannot supersede reliably despite non-trivial F1; the
0.5B LoRA variant has *higher* F1 than the 1.5B one (0.867 vs 0.652) but lower
key-consistency (0.75 vs 0.775) and only 3/6 clean-churn supersessions — a 0.5B-specific
out-of-distribution gap the contaminated screen hid. The 1.5B model reproduces the
reference's supersession behavior fully offline (6/6, zero fabrication) and its
canonical-key training **exceeds the cloud reference's own key-consistency** (0.775 vs
0.707), since forcing one key per attribute is a training-time constraint ad hoc cloud
prompting doesn't get for free. It passes an end-to-end supersession gate the untuned
baselines fail and ships as an opt-in `LLM_PROVIDER=ollama` path — the full
learn→supersede→doubt loop runs with zero cloud calls. Reported as a probe, not a
production claim: `n`=26 messages / 8 paraphrase groups, deterministic point estimates
without confidence intervals. Full tables and pipeline: `docs/BENCHMARK.md` §10,
`scripts/distiller_lora/`.

## 5. Limitations

- **Verbatim-recall benchmarks favor raw storage.** On LoCoMo-10 (n=500 stratified,
  qwen-judged, same reader/judge both arms) naive RAG beats Tenet 38.8 vs 33.8
  (McNemar p=0.031): LoCoMo's gold answers often reward the *exact wording* of a raw
  turn, and distillation paraphrases that wording away. LoCoMo stresses verbatim
  multi-session recall, not knowledge churn — reported plainly (BENCHMARK.md §12).
  Related audit finding: the LoCoMo audit's 156 entries correct evidence *citations*,
  not gold answers — an "audit-corrected" key moves QA accuracy by exactly 0.0pp.
- **Natural-language updates under-fire keyed supersession.** On PersonaMem-v2 (NapMem's
  benchmark; 4-way MC, n=485) a blind no-memory control scores 34.4% vs both memory arms
  at ~50.5% — memory is load-bearing, comparison valid — but Tenet ties RAG (50.5 vs 50.9,
  p=0.92), and on the retraction subset we predicted Tenet would win, RAG leads (74.2 vs
  67.7, ns). Only 3.8% of turns trigger keyed supersession because conversational "I've
  switched to X" rarely yields a distilled key that collides with the prior belief. **We
  then fixed this** (BENCHMARK.md §13.1): LLM-free embedding key-resolution (text-floor
  hardened to 0.66 to reject shared-word coincidences) + a vague-key-free distiller raise
  true-fire 40→80% at 0% false-fire on an expanded adversarial set, lift real PersonaMem
  firing 3.8→~7% and overall 50.5→53.4%, passing every regression gate (default on). It
  does *not* fully address explicit retractions ("forget X" = deletion, not replacement —
  a tombstone is the next step). So the churn claim holds for stable-attribute updates
  (§4.2, §6) and now degrades gracefully on free-form drift. (We repurposed a full-context
  task as retrieval, so absolute scores aren't vendor-comparable.)
- **Multi-session synthesis** is the one category where RAG still leads (42.9 vs 57.1).
  Belief-anchored expansion (§3.6) lifted it from 28.6 but does not close it: these questions
  need evidence from *several* sessions, and expansion only deepens the sessions the top-*k*
  already surfaced — if a required session is not among them, its detail is still missing.
  Session-diverse retrieval (guaranteeing coverage across distinct evidence sessions) is the
  natural next step. Elsewhere Tenet is at or above RAG.
- **The frontier is a knob, not free lunch.** Parity accuracy costs RAG-equal tokens; the big
  per-token win (1.6×) is at the efficiency point, which trades ~5 pp of raw accuracy. One
  system spans both, but no single setting is best on every axis at once.
- **Evaluation.** n=40, off-Qwen (gpt-4o / gpt-4o-mini readers, local embedder), one seed;
  reader stochasticity is ≈±5–7 pp, so the one-shot result is reported as *parity*, not a win.
  The shipped system uses Qwen Cloud; relative comparisons hold, as all systems share the reader.
- **Stale raw-turn leakage under natural conversational churn** (§4.8): with several
  attributes each updated many times via paraphrased (not templated) statements, the
  write-time stale-echo filter can miss a raw turn whose wording diverges from its
  distilled paraphrase, letting it reach the same recall window as the current fact and
  sometimes outvote it. **Partially fixed (§4.9)**: a read-time, key-scoped consistency
  check plus a currency-structured reader context raise churn half-life from <2 to 8
  (both defaulted on, all regression gates clean) — real progress, but still short of
  Mem0-style's write-time-consolidation immunity at U=32.

## 6. Conclusion

Treating agent memory as a **self-consistent belief state** rather than a document index
makes it robust to the way real knowledge behaves: it changes. Tenet stays correct under
knowledge churn where retrieval memory collapses, matches a strong RAG's one-shot accuracy at
equal token budget while offering the best accuracy-per-token of the systems we tested, using
a light graph-free substrate and no LLM in the read path. The
belief-state view also yields time-travel and principled forgetting for free. We hope
*knowledge churn* becomes a standard axis for evaluating agent memory.

## References

[Chhikara 2025] Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.
[Wu 2024] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. arXiv:2410.10813.
[Wu 2026] LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues. arXiv:2605.12493.
[Rasmussen 2025] Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956.
[Packer 2023] MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
[Xu 2025] A-MEM: Agentic Memory for LLM Agents. arXiv:2502.12110.
[Hu 2026] MemoryAgentBench: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions. arXiv:2507.05257 (ICLR 2026).

[Gutiérrez 2025] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2). arXiv:2502.14802.

[Shen 2026] Shen et al. QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management. arXiv:2512.12967 (Alibaba Tongyi).

[Yu 2026] Yu et al. Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for LLM Agents. arXiv:2601.01885 (ACL 2026, Alibaba).

[Zhang 2026] Zhang et al. ActMem: Bridging the Gap Between Memory Retrieval and Reasoning in LLM Agents. arXiv:2603.00026.
[Freshness 2026] Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution. arXiv:2606.01435.
[MemStrata 2026] Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge. arXiv:2606.26511.
[Engram 2026] Less Context, More Accuracy: A Bi-Temporal Memory Engine for LLM Agents. arXiv:2606.09900.
[TOKI 2026] TOKI: A Bitemporal Operator Algebra for Contradiction Resolution in LLM-Agent Persistent Memory. arXiv:2606.06240.
[NapMem 2026] NapMem: Memory as a Structured Action Space for Long-Horizon Agents. arXiv:2607.05794 (Qwen Large Model Application Team, Alibaba).
[MemFlow 2026] MemFlow: Training-Free Intent-Routed Memory Orchestration with LLM Validation. arXiv:2605.03312.
[MemReader 2026] MemReader: Active Extraction for Agent Memory Construction. arXiv:2604.07877.
[Mem-alpha 2026] Mem-α: Reinforcement-Learned Memory Construction for LLM Agents. OpenReview (ICLR 2026).
[MemCog 2026] MemCog: Cognitively-Inspired Active Navigation of Agent Memory. arXiv:2605.28046. (Affiliation unverified.)
[Friston] The free-energy principle: a unified brain theory? Nat. Rev. Neurosci., 2010.

*Reproduce every number: see `docs/BENCHMARK.md` and `scripts/`.*
