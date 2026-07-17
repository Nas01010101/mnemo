"""Usage-scenario retrieval — an OPTIONAL recall knob inspired by Tongyi Lab's ReMe
(arXiv:2512.10696, agentscope-ai/ReMe): key a memory not just by CONTENT similarity
but by the USAGE SCENARIO — "when would this memory be used" — a query resembles.

ReMe's own retrieval indexes an LLM-generated usage-scenario description alongside
each memory and matches queries against it directly. Tenet's version reuses the
EXISTING distill() call (distill.py's `scenario` field, one extra JSON key — no
second LLM call) rather than a dedicated indexing pass: at write time the distiller
already tags each fact with a one-line "when would this be useful" description;
`memory.py`'s `store()` embeds it alongside the fact's content embedding when the
knob is on.

At read time, this module RRF-merges two independently-ranked lists — content
similarity (the existing `recall()` ranking) and scenario similarity (query vs each
fact's usage-scenario embedding) — so a query that paraphrases the SITUATION
("what should I cook for the user") rather than the fact's CONTENT ("user is allergic
to peanuts") can still surface it, without content similarity ranking (correctness
of the churn/consistency machinery downstream) ever being replaced, only re-ordered.

UNLIKE `aggregate.py`/`raw_recall`'s pool FILTERS (drop-only, never reorder survivors),
RRF fusion is explicitly a RE-RANKER — that's the whole point of a second retrieval
signal. Default OFF (`recall(..., usage_recall=True)` or `TENET_USAGE_RECALL=1`);
with it off, `recall()`'s ranking is byte-identical to before this module existed
(scenario embeddings are never even fetched from sqlite, let alone merged).
"""
from __future__ import annotations

# Standard RRF smoothing constant (Cormack & Clarke 2009 default; also what the
# existing exp_ruler_mh_rrf.py hybrid-retrieval experiment uses).
RRF_K = 60.0


def rrf_merge(scored: list, scenario_sims: dict[int, float], rrf_k: float = RRF_K) -> list:
    """Re-rank `scored` (a list of (content_score, rel, row) tuples, already sorted
    descending by content_score — `memory.py` recall()'s `scored` variable) by fusing
    its rank with a SECOND ranking derived from `scenario_sims` ({row_id: cosine
    similarity against the query's usage-scenario embedding}).

    Rows missing from `scenario_sims` (no usage-scenario embedding — raw slices, or
    facts stored before/without the knob) contribute 0 to the scenario arm and are
    ranked purely on their content-arm RRF term, so they never disappear from the
    pool — they just can't out-rank a row that scores on BOTH arms.

    Returns a NEW list, same (content_score, rel, row) shape so callers can slot it
    straight back into `recall()`'s existing dual-pool/expand/hops pipeline — only
    the ORDER changes, the tuple contents (rel, row) are untouched, so every
    downstream consumer of `rel`/`row` keeps working unmodified."""
    if not scenario_sims:
        return scored
    content_rank = {row["id"]: i for i, (_score, _rel, row) in enumerate(scored)}
    scenario_rank = {
        row_id: i for i, (row_id, _sim) in
        enumerate(sorted(scenario_sims.items(), key=lambda kv: kv[1], reverse=True))
    }

    def _combined(row_id: int) -> float:
        c = 1.0 / (rrf_k + content_rank[row_id])
        s = 1.0 / (rrf_k + scenario_rank[row_id]) if row_id in scenario_rank else 0.0
        return c + s

    merged = [(_combined(row["id"]), rel, row) for _score, rel, row in scored]
    merged.sort(key=lambda x: x[0], reverse=True)
    return merged
