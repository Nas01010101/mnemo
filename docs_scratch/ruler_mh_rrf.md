# RULER-MH RRF experiment — measured NULL (2026-07-13)

**Hypothesis** (from LME-V2's 12%→59.7% recall lift, paper §4.7): the MAB-AR RULER
multi-hop loss (Tenet 45 vs HippoRAG-v2 66, BENCHMARK.md §7) is partly retrieval-bound —
BM25+dense reciprocal-rank fusion at the retrieval stage should lift gold recall.

**Protocol** (`scripts/exp_ruler_mh_rrf.py`, pre-registered gate): stage 1 = deterministic
gold-in-pool rate at identical char budget, baseline tenet retrieval (k=10, expand=20,
hops=2) vs BM25+dense RRF (k_each=30, c=60); proceed to the paired SubEM reader eval only
if RRF lifts ≥ +3pp. Cell: `ruler_qa2_421K` (the only MH cell; n=100, all questions).
Embeddings bge-small local, cached store reused; zero LLM calls in stage 1.

**Result: exact tie.**

| arm | gold-in-pool (n=100) |
|---|---|
| baseline | 72.0% [62.5, 79.9] |
| BM25+dense RRF | 72.0% [62.5, 79.9] |

(20-question smoke read +5pp for RRF; the full run erased it — small-n noise.)

**Verdict: NULL — gate not met, reader eval correctly skipped.** RULER-MH retrieval is
NOT lexically bound; the LME-V2 fix does not transfer. Two implications:

1. Gold-in-pool (72%) far exceeds the measured QA score (45%): the binding constraint is
   multi-hop *composition* — chaining bridge evidence — not surfacing the answer string.
   That is consistent with HippoRAG-v2's edge being Personalized-PageRank *traversal*
   (reaching bridge chunks by graph adjacency, not query similarity), which neither dense
   nor BM25 ranking replicates. Caveat: gold-in-pool is a weak MH proxy (the answer entity
   can appear without the bridge evidence needed to justify it).
2. Next credible lever is not retrieval scoring but pool *construction*: query
   decomposition (Self-Ask per-hop recall, as in the FactCon `tenet+hop` arm) or
   entity-bridging expansion. Both are reader-or-structure work, out of scope tonight.

Raw per-question rows: `docs_scratch/ruler_mh_rrf_stage1.jsonl`.
