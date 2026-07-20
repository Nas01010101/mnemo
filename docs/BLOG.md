# Your memory benchmark is grading on a broken curve

*Building Tenet for the Global AI Hackathon with Qwen Cloud (Track 1: MemoryAgent).*

Every "AI memory" leaderboard number you've seen — Mem0's 92.5 on LoCoMo included —
is downstream of a benchmark whose ground truth we haven't actually checked. We
started this project the normal way: read the field (Mem0, Zep/Graphiti, Letta,
LongMemEval), pick a benchmark, beat a baseline. Partway through we went and read the
one artifact almost nobody reads — the benchmark's own ground-truth data — and the
number underneath the number didn't hold up. That finding reshaped what we built and
how we chose to evaluate it.

## What the LoCoMo audit actually found

LoCoMo (long-context conversational memory) is the benchmark Mem0 markets its
headline 92.5 score against. An independent audit of its ground truth
([github.com/dial481/locomo-audit](https://github.com/dial481/locomo-audit),
`AUDIT_REPORT.md`) found **99 of 1,540 QA pairs (6.4%) have score-corrupting
ground-truth errors** — wrong dates, wrong answers, category mislabels — which caps
the benchmark's own **theoretical ceiling at 93.6%**. A perfect memory system,
graded against LoCoMo's actual labels, cannot score above that. Mem0's reported 92.5
is a hair under a ceiling created by the errors, not the model.

It gets worse on the grading side. LoCoMo's official evaluation uses a
`gpt-4o-mini` LLM judge to decide if a generated answer matches the reference. The
same audit (reported in
[penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer),
Apr 2026) constructed deliberately wrong-but-topically-adjacent answers and found the
judge **accepted 62.8% of them** as correct. A judge that lenient inflates every
system it grades, and it inflates systems with more verbose or more topically-broad
retrieval *more* than terser, precise ones — which is exactly the shape of a systemic
bias, not noise that washes out. Zep raised an earlier, independent version of this
concern about Mem0's LoCoMo methodology in
[blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
(May 2025) — three separate write-ups, over a year apart, converging on the same
conclusion: the number is real, the curve it's graded on isn't.

We're not naming this to dunk on Mem0 — every memory system that reports a LoCoMo
number, including early drafts of this one, inherits the same broken curve. The
honest response isn't to stop citing LoCoMo; it's to stop *trusting a single
LLM-judged benchmark as the only evidence*, and to build evaluation where a broken
curve isn't possible in the first place.

## Why we lean on deterministic evaluation instead

Two properties make a benchmark hard to grade-inflate: the correctness check is a
program, not a language model with its own biases, and the uncertainty on the
resulting number is reported, not implied.

Our **long-horizon knowledge-churn** test (`bench_horizon.py`) is exact-match against
the one currently-true value — no judge, nothing to lean on. As a fact gets updated
more times than a retrieval budget can hold (k=6, up to 12 updates), naive RAG's
top-k can't hold every stale version and starts serving old answers: it drops from
100% to 50% accuracy. Tenet holds 100% at every point on this templated single-attribute
primitive, because supersession keeps exactly one current value per key regardless of how
many times it changed — the regime long-term memory exists for. (We later stress-tested
that claim on a harsher *paraphrased*, multi-attribute variant — ChurnBench, `docs/BENCHMARK.md`
§9 — where it initially *failed*; a default-on read-time consistency fix recovers it to
98/92/82 at U=2/8/32. We report the falsification and the partial fix in full.)

For a standardized comparison against the published field we used
**MemoryAgentBench's FactConsolidation** (arXiv:2507.05257): the official
**SubEM** metric (deterministic substring match, not an LLM judge), the official
reader prompt copied verbatim, and **Wilson 95% confidence intervals** on every
score — all 800 questions, zero exclusions. Single-hop: **97.0% pooled, CI [94.8,
98.3]**, above even the published gpt-4o-tier pooled result (94.8), on a deliberately
weaker backbone (local qwen2.5:7b vs. the published gpt-4o-mini/gpt-4o tiers). Multi-hop:
**45.8%**, 1.5× the published SOTA of 30.2. (These follow an ingestion-keyer fix our own
miss-file audit exposed — pre-fix 86.5/30.0, both runs in the evidence artifact.)

To rule out "you just picked a better reader," we reimplemented four published memory
mechanisms — CAR (candidate extraction + max-serial aggregation, the published-SOTA
recipe), Mem0-style batched ADD/UPDATE consolidation, HippoRAG-v2-style OpenIE +
Personalized-PageRank, and MemAgent-style rolling overwrite memory — as arms of the
*same* harness: same reader, same embedder, same questions, same metric. Tenet led
every arm on both axes (SH 90.0 [85.1, 93.4] vs. CAR's 87.5 [82.2, 91.4] as the
closest competitor; MH 36.0 [29.7, 42.9] vs. CAR's 33.0). The full tables, every
reproduce command, and the categories where we lose (RULER multi-hop against
HippoRAG-v2's graph chaining, most notably) are in `docs/BENCHMARK.md` — losses
included, because a benchmark you only publish when you win is the same failure mode
as LoCoMo's.

## The architecture that follows from this

None of the above works without a data model built for it. Tenet is bi-temporal:
every fact carries **event time** (`valid_at`/`invalid_at` — when it's true in the
world) and **transaction time** (`created_at`/`expired_at` — when the system learned
it). At write time, one LLM call distills a message into atomic facts with a stable
`subject::attribute` key (`user::residence`, `project_nimbus::ship_date`); a later
message with the *same* key **supersedes** — invalidates and retires, never
overwrites — instead of adding a second, contradictory row. That key is load-bearing:
we measured a value change ("flight at 14:20" → "09:45") scoring **0.988** cosine
similarity, higher than a plain rephrasing of the same fact (0.79) — embeddings alone
cannot distinguish "changed" from "restated," which is why every serious system needs
extraction at write time, not just retrieval at read time.

Reads pay none of that cost. `recall()` is pure vector similarity plus a recency/
salience decay — **no LLM in the read path** — so latency stays flat regardless of
distillation cost. And because supersession leaves a full history instead of deleting
it, we get a second capability almost for free: a closed-form **fact-dynamics** model
(`dynamics.py`) fits a per-key survival curve from the ledger's own supersession
history — "residence" learns a slow hazard, "mood" a fast one — and surfaces
`uncertain_facts()`: a "worth re-verifying" doubt list, purely statistical, no model
call, refit lazily as the ledger grows.

## Honest limitations

Two are worth stating plainly. First, **the distiller is a quality gate on
everything downstream** — supersession, dynamics, and doubts are only as reliable as
the keys the write-time LLM call assigns; a mis-keyed fact either fails to supersede
or wrongly clobbers something unrelated. We haven't stress-tested distillation
quality at scale, only at benchmark scale. Second, **this is validated at
single-user, hackathon scale** — brute-force cosine over sqlite, fine under 1e5
memories, with no load-testing at multi-tenant volume. The interface is designed to
swap in `sqlite-vec` or Postgres without touching callers, but that swap hasn't been
measured.

## Built on Qwen Cloud

Distillation on `qwen3.6-flash`, retrieval on `text-embedding-v4`, reading on
`qwen3.7-plus` — all through the OpenAI-compatible DashScope API. The read path stays
LLM-free by design, and Tenet is MCP-native: drop it into Claude Desktop, or now a
LangGraph agent via `TenetStore(BaseStore)`, and it gets persistent,
self-consistent memory instead of a flat vector dump.

*Code: https://github.com/Nas01010101/tenet · Built for [Qwen Cloud Hackathon](https://qwencloud-hackathon.devpost.com).*
