# RULER-MH Self-Ask chain — measured NEGATIVE, + a reader-sensitivity finding (2026-07-14)

Follow-up to the RRF null (`ruler_mh_rrf.md`), which isolated the RULER multi-hop gap
(Tenet 45 vs HippoRAG-v2 66 in §7, gpt-4o-mini) as **composition**-bound, not retrieval-
bound. This arm attacks composition directly: decompose the question (one LLM call), answer
hop 1 from its own pool, then **anchor hop 2's retrieval on the intermediate answer** — the
bridge-reaching trick HippoRAG gets from PPR graph traversal. Fallback to the baseline pool
on failed decomposition, identical char budget, paired SubEM + McNemar.

**Reader/decomposer: `qwen3.7-plus` (Qwen Cloud)** — a genuinely strong reader, the shipped
stack, and a much better decomposer than the RTX-7B this was first stuck behind.

## Result — decomposition HURTS (n=99, ruler_qa2_421K, all questions)

| arm | SubEM |
|---|---|
| baseline (top-k) | **60.6%** [50.8, 69.7] |
| Self-Ask chain | 53.5% [43.8, 63.0] |

- **gold-in-pool identical: 72 = 72** — decomposition changed *which* chunks, not whether the
  answer is present. So this is not a retrieval effect.
- **McNemar: chain wins 3, loses 10 (net −7, p=0.092)** — directionally clear loss.

### Why it hurts (from the 10 base-right / chain-wrong misses)
1. **Hop-2 retrieval loses the chunk baseline had.** "Arena the Lewiston Maineiacs played in,
   seating capacity?" → hop-1 correctly returns *Androscoggin Bank Colisée*, but the hop-2 pool
   anchored on that name no longer contains the capacity line straight top-k surfaced → reader
   answers "pool does not contain it."
2. **Decomposition breaks the answer format.** "Are Local H and For Against both from the US?"
   → decomposed into two fact lookups → reader answers "American rock band" instead of "yes."
3. **Hop-1 error propagation.** A wrong or off-target intermediate entity poisons hop-2's cue.

All three are the documented Self-Ask failure modes; a reader strong enough to chain the hops
*internally* does better on the straight top-k pool than on a decomposed, error-propagating one.

## The valuable side finding (reader sensitivity)
Baseline RULER-MH on `qwen3.7-plus` is **60.6% [50.8, 69.7]** — vs §7's published **45**
(gpt-4o-mini, protocol-matched to the leaderboard). Same benchmark cell, same baseline
retrieval, only the reader swapped. The CI **includes HippoRAG-v2's 66**. So the RULER-MH
"loss" is largely a **weak-reader artifact**: the gap nearly closes with reader strength alone,
no new memory mechanism required — the same reader-tracks-accuracy story as LME (§1–2).

**Caveat, stated plainly:** this is *off-protocol* — the MAB-AR leaderboard is scored at the
gpt-4o-mini tier, so 60.6 does **not** change Tenet's published #2 ranking or the §7 numbers.
It's a reader-sensitivity data point, one seed, n=99, single reader.

## Verdicts
- **Self-Ask decomposition: do NOT adopt** — measured-negative on a strong reader (−7 net).
- **The leaderboard-#1-via-decomposition attempt: failed.** The RULER-MH gap is real only at
  the weak-reader tier; at the shipped tier it's already CI-overlapping with the leader, and
  no cheap composition trick beats plain top-k there.
- Next credible lever would be a *learned* traversal (HippoRAG-style PPR over an entity graph),
  which is a different architecture and out of scope — and, per this finding, only worth it for
  the weak-reader-tier leaderboard number, not the shipped product.

Raw rows: `docs_scratch/ruler_mh_hop.jsonl`. Reproduce: `LLM_PROVIDER=qwen EMBED_PROVIDER=local
python scripts/exp_ruler_mh_hop.py --qpc 100`.
