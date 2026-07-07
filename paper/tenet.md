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
**100% current-value accuracy across all churn levels** (where a strong RAG-memory falls to
50%), matches strong RAG on retrieval recall (95–97.5%), and — with expansion — **matches its
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
   levels)**, on par with RAG on recall (95%), **best-in-class on accuracy-per-token**, and —
   with belief-anchored evidence expansion — **at parity with strong RAG on one-shot accuracy
   at equal token budget**, closing a gap earlier belief-only compression left open.

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
measure the residence value-change "14:20→09:45" at cosine 0.99, indistinguishable from a
paraphrase), but a shared key can.

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

**3.6 Belief-anchored evidence expansion.** Compressing a session into a few keyed beliefs
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
dominance holds across `gpt-4o-mini`, `gpt-4o`, and `claude-opus-4.8` readers (≈1.6–1.7×).

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

## 5. Limitations

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
[Freshness 2026] Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution. arXiv:2606.01435.
[MemStrata 2026] Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge. arXiv:2606.26511.
[Engram 2026] Less Context, More Accuracy: A Bi-Temporal Memory Engine for LLM Agents. arXiv:2606.09900.
[TOKI 2026] TOKI: A Bitemporal Operator Algebra for Contradiction Resolution in LLM-Agent Persistent Memory. arXiv:2606.06240.
[Friston] The free-energy principle: a unified brain theory? Nat. Rev. Neurosci., 2010.

*Reproduce every number: see `docs/BENCHMARK.md` and `scripts/`.*
