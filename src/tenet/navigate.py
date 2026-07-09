"""navigate — LLM-free adaptive-depth associative memory navigation (Tenet).

The Qwen paper *From Passive Retrieval to Active Memory Navigation* (NapMem,
arXiv:2607.05794) argues memory should move from passively-retrieved context to a
structured action space an agent *navigates*: start broad, drill down to evidence,
and STOP once enough has been gathered. NapMem gets the stopping behaviour from a
GRPO-trained 9B policy. That needs training + serving we cannot ship, and it puts an
LLM in the read path — which Tenet's core product claim (LLM-free read) forbids.

`navigate()` is the *deterministic, LLM-free* instantiation of the same behaviour.
It reuses Tenet's existing recall primitives (belief-anchored `expand`, associative
`hops`) but replaces NapMem's fixed / learned depth with an ADAPTIVE stop: keep
descending only while each deeper hop still surfaces relevant NEW evidence, and stop
the moment returns diminish. This is the piece Tenet lacks today — `recall(hops=N)`
takes a caller-fixed N, which over-fetches simple queries (distractors that hurt the
downstream reader) and under-fetches multi-hop ones. Per-query adaptive depth fixes
both ends.

Positioning (honest): the closest published relative is MemFlow (arXiv:2605.03312),
a *training-free* Qwen3-1.7B orchestrator whose Validator retries with a heavier
memory tier when an answer is ungrounded. navigate() is that escalation loop with an
EMBEDDING-based sufficiency gate instead of an LLM validator call — so the whole read
path stays LLM-free and ~milliseconds.

Read-only: navigate never writes, never mutates the store beyond recall's own
access-touch bookkeeping, and imports nothing heavy (no torch, no network).
"""
from __future__ import annotations

from typing import Any, Protocol


class _Recallable(Protocol):
    def recall(self, query: str, *, k: int = ..., char_budget: int | None = ...,
               expand: int = ..., hops: int = ...) -> list[Any]: ...


def navigate(
    core: _Recallable,
    query: str,
    *,
    k: int = 10,
    max_hops: int = 4,
    tau_gain: float = 0.15,
    char_budget: int | None = None,
) -> tuple[list[Any], list[dict]]:
    """Adaptively-deep, LLM-free navigation over Tenet's belief pyramid.

    Escalation schedule (each rung reuses an existing recall mode):
      hop 1 — broad belief recall + belief-anchored raw expansion
              (`recall(expand=k, hops=1)`): the records level plus the verbatim
              turns from the sessions those records came from (NapMem "descend to
              raw evidence").
      hop h>1 — associative replay (`recall(expand=k, hops=h)`): re-condition the
              cue on the evidence gathered so far and re-score the WHOLE store, so a
              hop can reach a session the raw query never surfaced (the bridge a
              multi-hop question needs).

    Adaptive stop (the novel part): after each deeper hop, measure the best relevance
    among the memories it added that the previous rung did NOT have. Stop — and keep
    the *previous* pool — as soon as that marginal gain falls below ``tau_gain`` (or a
    hop adds nothing). Low-relevance new items are treated as distractors and dropped,
    which is what protects precision on simple queries; genuinely relevant bridge
    evidence clears the gate and is adopted, which is what helps multi-hop recall.

    Args:
      core: any object exposing Tenet's ``recall(query, k, char_budget, expand, hops)``
            (a ``tenet.memory.MemoryCore``, or ``Tenet.core``).
      query: the user question.
      k: base top-k width per hop.
      max_hops: hard budget on descent depth (bounds cost; NapMem uses 4).
      tau_gain: cosine-relevance floor a hop's best new item must clear to be adopted.
      char_budget: optional total-character cap forwarded to recall.

    Returns:
      ``(memories, trace)`` — the selected memories (Tenet ``Memory`` objects) and a
      per-hop trace of dicts (hop index, size, new-item count, marginal gain, whether
      the rung was adopted or triggered the stop) for logging / demo / ablation.
    """
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")

    pool = core.recall(query, k=k, char_budget=char_budget, expand=k, hops=1)
    ids = {m.id for m in pool}
    trace: list[dict] = [{"hop": 1, "n": len(pool), "adopted": True}]

    for h in range(2, max_hops + 1):
        deeper = core.recall(query, k=k, char_budget=char_budget, expand=k, hops=h)
        new = [m for m in deeper if m.id not in ids]
        gain = max((m.score for m in new), default=0.0)
        if not new or gain < tau_gain:
            trace.append({"hop": h, "new": len(new), "gain": round(float(gain), 4),
                          "adopted": False, "stop": "saturated"})
            break
        pool = deeper
        ids = {m.id for m in deeper}
        trace.append({"hop": h, "n": len(deeper), "new": len(new),
                      "gain": round(float(gain), 4), "adopted": True})

    return pool, trace
