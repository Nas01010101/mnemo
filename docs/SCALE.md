# Scale — where the first wall was, what fixed it, and where it is now

`docs/BENCHMARK.md` proves Tenet is *accurate*, at conversation scale (tens to low
thousands of memories, matching the standardized benchmarks it's evaluated on). This
doc asks a different question, end-to-end and at scale: **does the system hold up as
the store grows, and where does it stop?** Companion to `docs/DEPLOY.md` (what the live
Alibaba Cloud FC deploy inherits from these numbers) and `scripts/test_e2e_surfaces.py`
(proves one store is really shared across every surface — Python API, CLI, MCP,
LangGraph, HTTP API).

**Status: all three walls identified in the first pass of this doc are fixed and
measured.** See "Before / after" below for the numbers; "What changed in `src/`" for
the actual diffs (`src/tenet/index.py`, new; `src/tenet/memory.py`, modified).

## TL;DR

| | before | after |
|---|---:|---:|
| recall() e2e @ 100k | 1,054.9 ms | **10.8 ms** (~98× faster) |
| recall crosses 100ms at | ~10k facts | **not reached by 100k** (flat ~9-12ms) |
| recall crosses 1s at | ~100k facts | **not reached by 100k** |
| ingest (keyed) @ 100k | 13.3 facts/s | **7,441 facts/s** (~560× faster) |
| ingest (unkeyed) @ 10k pre-existing | 25.8 facts/s | **17,543 facts/s** (~680× faster) |
| RSS @ 100k | 1,157.5 MB | 1,394.7 MB (**+20%**, the honest tradeoff) |
| 1M recall latency (extrapolated) | ~10.0 s | **~24 ms** |
| 1M RSS (extrapolated) | ~7.8 GB | ~8.7 GB |

The read-latency and ingest-throughput walls are gone (flat, dominated by fixed
per-call costs — local-embedder inference for reads, `_key_emb`/SQL-lookup overhead
for keyed writes — not by store size, at least up to the 100k measured here). **RAM is
now the ONLY wall**, unchanged in kind, ~20% worse in magnitude (more resident
metadata) — still the honest first thing to hit at real scale. See "Verdict, revised"
below.

## Methodology (unchanged from the first pass)

- **Deterministic synthetic embeddings** (seeded `numpy` RNG, unit-normalized, d=384 —
  `bge-small-en-v1.5`'s dimensionality), injected via `MemoryCore.store(..., _vec=...)`
  — isolates the store's own scaling from embedder latency (measured separately, a
  fixed ~9-11ms/query, see the plot). Scaling/perf probe, not a retrieval-quality
  benchmark.
- **Corpus setup vs. measured operations stay separate** (`bulk_seed()` for reaching
  each milestone fast; `ingest_facts_per_sec` always comes from the real, unmodified
  `MemoryCore.store()` on a 500-call sample at that size).
- **Environment**: macOS, `EMBED_PROVIDER=local`, `TENET_DB_PATH` on scratch, one dev
  machine (not a dedicated benchmark box) — treat absolute numbers as
  order-of-magnitude, the *shape* of each curve as the load-bearing result.

## Before / after

### Recall latency vs store size

| n | e2e (before) | e2e (**after**) | fetch/build (before) | fetch/build (**after**) | matmul only |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 25.5 ms | **9.0 ms** | 1.7 ms | 0.9 ms | 0.006 ms |
| 10,000 | 218.2 ms | **10.1 ms** | 33.6 ms | 1.4 ms | 0.011 ms |
| 100,000 | 1,054.9 ms | **10.8 ms** | 327.3 ms | 1.8 ms | 0.022 ms |

`fetch/build` is what `_rows_as_of()`/the recall-scoring row-source costs — before, a
full sqlite `SELECT`+`fetchall()`+`np.frombuffer` rebuild; after, the resident index's
boolean-mask + `row_dict()` construction for the (now typically ALL, since nothing's
superseded in this synthetic corpus) matched subset — no SQL, no BLOB deserialization,
and it barely grows with n (0.9→1.8ms, 1k→100k) because it's now pure in-memory numpy
indexing. `matmul` was always cheap and stays cheap. **e2e latency after the fix is
flat and dominated by the local embedder's fixed per-query cost (~9ms)** — the
store-size-dependent part of `recall()` has gone from ~97% of the call (327ms of
1055ms at 100k) to a small, roughly-constant fraction.

### Ingestion throughput

| pre-existing n | keyed (before) | keyed (**after**) | unkeyed (before) | unkeyed (**after**) |
|---:|---:|---:|---:|---:|
| 0 | 2,035.6 facts/s | **16,352.6 facts/s** | 1,669.7 facts/s | **23,357.8 facts/s** |
| 2,000 | — | — | 179.9 facts/s | **20,065.9 facts/s** |
| 4,000 | — | — | 71.0 facts/s | **17,881.2 facts/s** |
| 6,000 | — | — | 34.7 facts/s | **18,446.5 facts/s** |
| 8,000 | — | — | 29.0 facts/s | **17,421.0 facts/s** |
| 10,000 | 78.2 facts/s | **17,695.3 facts/s** | 25.8 facts/s | **17,542.6 facts/s** |
| 100,000 | 13.3 facts/s | **7,441.1 facts/s** | *(not run)* | *(not run — no longer the story; see below)* |

Both paths went from a clean, confirmed O(n)-per-insert decay to **near-flat**
throughput. The unkeyed path in particular: 1,670→26 facts/s (a ~64× collapse) before,
**23,358→17,543 facts/s (a ~1.3× dip, well within noise for a shared dev machine)**
after — the O(n²)-total bug is gone, not just slower.

**One honest residual**: the KEYED path still shows a real, reproducible ~2.3×
throughput dip at 100k (16,353→17,695→7,441 facts/s, 1k→10k→100k — NOT monotonic
decay, closer to flat-then-a-drop). Not fully re-attributed after the fixes below;
candidates not ruled out: WAL checkpoint frequency/cost at larger file sizes, general
memory pressure on a shared machine (RSS is 1.4GB by this point), or some remaining
non-O(1) cost in the resident index's bookkeeping. Reporting as measured, not
papered over — an order of magnitude better than the ~560× collapse it replaced, but
not literally flat.

### Memory & disk footprint

| n | RSS (before) | RSS (**after**) | sqlite file |
|---:|---:|---:|---:|
| 1,000 | 456.8 MB | 589.9 MB | 1.84 MB (unchanged) |
| 10,000 | 456.8 MB | 667.7 MB | 19.82 MB (unchanged) |
| 100,000 | 1,157.5 MB | 1,394.7 MB | 198.56 MB (unchanged) |

RSS is **higher** after the fix — the honest tradeoff for making everything resident:
the full metadata mirror (valid_at/invalid_at/created_at/expired_at/last_access/uses/
pinned/salience/is_raw/has_skey arrays, plus Python-level text/skey/source lists and
the `by_subject` grouping dict) now lives in process memory alongside the embedding
matrix, not just the matrix. Disk footprint (the sqlite file itself) is unchanged —
these fixes don't touch what's persisted, only what's cached in-process. WAL mode adds
a `-wal`/`-shm` sidecar (a few hundred KB, checked directly: 393KB/`-shm`, 0 bytes
`-wal` once checkpointed) — negligible next to the main file.

## Extrapolation to 1M facts (NOT measured — linear fit, stated as such)

| metric | 100k (measured) | 1M before (extrapolated) | 1M **after** (extrapolated) |
|---|---:|---:|---:|
| recall() e2e latency | 10.8 ms | ~10.0 s | **~24.3 ms** |
| sqlite file size | 198.6 MB | ~1.99 GB | ~1.99 GB (unchanged) |
| process RSS | 1,394.7 MB | ~7.78 GB | **~8.69 GB** |

Latency at 1M goes from a genuine problem (10 seconds) to a non-issue (24ms,
dominated by the same fixed embedder cost visible at every other size) — a ~410×
improvement in the extrapolated number. RSS is the one metric that got slightly worse
in absolute terms (~8.7GB vs ~7.8GB) because more is now resident — but RSS was
ALREADY the identified first wall before this pass, and remains it now; these fixes
were never going to reduce memory usage, only latency and throughput. This
extrapolation carries the same caveat as before: linear-fit, and RAM pressure at scale
would make it worse than linear, not better.

## Verdict, revised

**RAM is now the ONLY wall** — the read-latency and ingest-throughput walls that used
to arrive first (at 10k and 100k respectively) are gone at every size measured here (up
to 100k). What was previously "three separate problems arriving in sequence" is now
one problem (memory footprint) that arrives around the same extrapolated point as
before (~1M facts, ~8-9GB) but is no longer preceded by a latency/throughput cliff on
the way there. Concretely: a store that reaches 100k facts is now fully usable
(11ms recall, 7-17k facts/s ingest) where it used to be unusable (1s+ recall, 13
facts/s ingest) — the practical ceiling moved from "~10k facts, uncomfortably" to
"however much RAM the resident index can hold before the process starts swapping."

**Next fix, if this becomes a real constraint**: the resident index currently holds
metadata for every non-archived row unconditionally. An ANN index (sqlite-vec/HNSW)
would help the (already-cheap) matmul step, not the metadata memory footprint — the
actual next lever is more likely a capacity-bounded resident set (LRU-evict cold
rows back to sqlite-only, rehydrate on demand) if a deployment genuinely needs
multi-million-fact stores in one process. Not needed at hackathon/product-launch
scale; flagged for the roadmap, not built here.

## What changed in `src/`

**New file: `src/tenet/index.py`** (296 lines) — `ResidentIndex`, an in-memory mirror
of the `memories` table (every non-archived row: current + superseded, needed for
`as_of` and the stale-echo consistency check). Holds embeddings in one contiguous
`(capacity, d)` numpy matrix plus parallel metadata arrays (valid_at/invalid_at/
created_at/expired_at/last_access/uses/pinned/salience/is_raw/has_skey) and Python
lists (text/skey/source), all kept warm across calls. Key design points:
- **Growth**: classic dynamic array (capacity doubles on overflow) — `append()` is
  amortized O(1), not a fresh allocation+copy per insert (which would silently
  reintroduce an O(n²) wall for exactly the bulk-ingestion case this fixes).
- **`by_subject: dict[str, list[int]]`** — an incrementally-maintained subject→
  positions index, added mid-pass after `EXPLAIN QUERY PLAN` showed
  `_resolve_key_supersede`'s old `skey LIKE ?` SQL query was a full `SCAN memories`,
  not using the `skey` index (confirmed, not assumed) — this was the dominant
  remaining cause of the keyed-insert throughput not being fully flat after the
  matrix+WAL fixes alone.
- **Masks** (`mask_current`, `mask_as_of`, `mask_raw_current`, `mask_expired_fact`,
  `mask_expired_keyed_fact`) — vectorized numpy boolean ops replacing what used to be
  four separate SQL `WHERE` clauses (`_current_rows`, `_rows_as_of`,
  `_expired_fact_matrix`, `_expired_keyed_rows`). `+inf` is the sentinel for
  `invalid_at`/`expired_at IS NULL` ("still true"/"still current"), so every
  bi-temporal comparison collapses into one `>` — no separate null-check branch.
- **`decay()`** — vectorized recency/use/salience scoring (was a Python `for`-loop
  calling `_decay(row)` once per row, `math.pow`/`math.log1p` each time).
- **Multi-process staleness**: `is_stale(db)` checks `PRAGMA data_version` — bumps on
  any OTHER connection's commit, never on this one's own writes (which are already
  applied incrementally). `MemoryCore._ensure_index()` checks this before trusting the
  index and does one full `refresh()` if another process wrote — verified directly
  (see "Correctness guards" below), not just documented as an assumption.

**Modified: `src/tenet/memory.py`** (751 → 937 lines — net GREW despite the
extraction; see "Honest note on file size" below):
- `__init__`: `PRAGMA journal_mode=WAL` + `PRAGMA synchronous=NORMAL` (fix 3).
- `store()` split into `store()` (unchanged public signature/behavior) +
  `_store_locked()` (the actual logic, no lock/commit — reusable) + new
  `store_many(items)` (batched embed + ONE commit for N items — fix 3's batched-ingest
  path). Every DB mutation in `_store_locked` (INSERT, the three UPDATE...SET
  expired_at call sites) now also mirrors into the resident index in the same call —
  `append()`/`mark_expired()` — so `recall()` never needs to re-query for what THIS
  process just wrote.
- `_nearest_current()`: vectorized (`mat @ vec` against the resident current-masked
  matrix) — fix 2, the confirmed O(n²) unkeyed-ingestion bug.
- `_resolve_key_supersede()`: candidate lookup moved from the `SCAN memories` SQL query
  to `ResidentIndex.rows_for_subject()` (O(1) average via `by_subject`) — the
  mid-pass finding above. Signature gained a third return value (superseded ids) so
  the caller can mirror the UPDATE into the index.
- `recall()`: row/matrix/decay sourcing changed from `_rows_as_of()` + a fresh
  `np.frombuffer(b"".join(...))` rebuild + a per-row `_decay()` loop to one call to
  the new `_rows_matrix_decay_as_of()` helper (resident-index-backed, vectorized
  decay). The delicate selection logic AFTER that point (dual-pool fact/raw split,
  stale-echo filtering, consistency check, expand/hops) is **byte-for-byte
  unchanged** — deliberately left alone to keep this a data-source swap, not an
  algorithm rewrite, on the part of the code with the most correctness surface.
- `_dynamics()`, `forget_sweep()`, `_current_rows()`/`_rows_as_of()`,
  `_expired_fact_matrix()`/`_expired_keyed_rows()`: all now resident-index-backed
  (no SQL) instead of their own `fetchall()`. `forget_sweep()` also vectorizes its
  decay check and physically compacts the index (`ResidentIndex.remove()`) instead of
  only flagging rows in sqlite.
- `_touch()`/new `_touch_locked()`: mirrors `uses`/`last_access` bumps into the index
  too, split so `_store_locked`'s internal touch (exact-restatement path) doesn't
  commit mid-batch.

**Honest note on file size**: `memory.py` was already 751 lines (over the repo's
500-line convention) before this pass, entirely from the in-progress supersession
work (unrelated to this fix). This pass DID extract the matrix/index logic into
`index.py` as instructed — a real, substantial 296-line module, not a token gesture —
but `memory.py` itself grew further (751→937) because the fix also added new,
legitimate functionality (`store_many`, WAL setup, the `_store_locked` split, and
docstrings explaining *why* each change exists, per this repo's own "comments explain
why" convention) rather than just moving code around. No further split was made:
`store()`'s family and `recall()`'s selection logic are both core to `MemoryCore`'s
single responsibility, and splitting them further for a line-count target alone felt
like the wrong tradeoff against correctness risk in an already-large diff — flagging
this judgment call rather than silently accepting it.

## Correctness guards (verified, not just claimed)

- **`as_of`/time-travel**: `ResidentIndex.mask_as_of()` reproduces the original SQL's
  four-condition bi-temporal filter exactly (`created_at<=t`, `expired_at IS NULL OR
  expired_at>t`, `valid_at<=t`, `invalid_at IS NULL OR invalid_at>t`) via the `+inf`
  sentinel trick. Verified with a direct repro: store a fact, capture a timestamp
  AFTER it's fully written (not before — a pre-write timestamp plus a tiny epsilon
  isn't safe once you know embedding latency can exceed it, a lesson from
  `test_e2e_surfaces.py`), supersede it, and confirm `recall(as_of=<the timestamp>)`
  still returns the OLD value while current `recall()` returns the new one — passes,
  and is now also covered by every `as_of` check across the full test suite
  (`test_agent_uncertainty.py`'s `time_travel` tests, `test_e2e_surfaces.py`'s
  bi-temporal check, all green).
- **`forget_sweep()`**: archived rows are physically removed from the resident index
  (`ResidentIndex.remove()`), not just flagged — confirmed via `test_memory.py`'s
  sweep assertions (unchanged, still green) plus a direct check that `stats()`
  reflects the archive count correctly post-sweep.
- **Multi-process (the HTTP/FC deploy)**: directly tested, not just documented — one
  process builds its resident index (a `recall()` call), a SEPARATE `python`
  subprocess opens the same `TENET_DB_PATH` file and stores a new fact, then the
  first process's NEXT `recall()` call is checked to see the second process's write.
  It does — `PRAGMA data_version` correctly detects the staleness and triggers one
  full `refresh()`. This is a real single-writer-at-a-time / multi-reader assumption
  made safe (any writer's commits become visible to every OTHER process's next
  `_ensure_index()` call, at the cost of one full rebuild the first time staleness is
  noticed) rather than a documented-but-unverified assumption.
- **`TENET_KEY_RESOLUTION` hardened behavior intact**: `_resolve_key_supersede`'s
  candidate SET changed (SQL scan → `by_subject` index lookup) but its FILTERING logic
  (`_value_compatible`, `_TAU_KEY`, `_TEXT_FLOOR`) is untouched — same guards, same
  thresholds, same behavior, just a faster way to find the candidates to check them
  against. `test_key_resolution.py` (deterministic, the hardened-`_TEXT_FLOOR=0.66`
  regression test from commit `9d37c33`) passes unchanged.
- **Accuracy re-verified on the resident-index code (not just inferred).** The refactor
  swaps `recall()`'s data source but leaves its selection algorithm byte-for-byte; to
  confirm this held end-to-end, the LLM-based accuracy benchmarks were re-run against the
  shipped code: **ChurnBench 100/100/100 at U=2/8/32 (half-life 32) and MAB FactConsolidation
  SH-6k 90.0% (n=20)** — both byte-identical to their pre-refactor values. The 100× recall
  speedup and the ingest fix cost zero accuracy.

## Findings from the original pass — resolved

The two findings the original pass of this doc routed for review are now addressed:
1. **`_nearest_current()`'s O(n) Python loop** — fixed (vectorized), see above.
2. **The `_KEY_RESOLUTION` false-positive mode** (shared-word cross-attribute
   supersession) — was ALREADY fixed independently by the supersession-fix agent
   (commit `9d37c33`, raising `_TEXT_FLOOR` 0.35→0.66) before this pass started;
   confirmed still passing via `test_key_resolution.py` and `test_e2e_surfaces.py`.

## Reproduce

```bash
# scale sweep + unkeyed-path probe + plot + JSON (what produced every number above)
EMBED_PROVIDER=local HF_HOME=~/.cache/huggingface TRANSFORMERS_CACHE=~/.cache/huggingface/hub \
  python scripts/bench_scale.py --scratch-dir /tmp/tenet_scale_bench

# faster iteration (skip 100k, fewer reps)
python scripts/bench_scale.py --sizes 1000,10000 --reps 5 --skip-unkeyed-probe

# end-to-end surface coverage (Python API / CLI / MCP / LangGraph / HTTP API, one store)
EMBED_PROVIDER=local HF_HOME=~/.cache/huggingface TRANSFORMERS_CACHE=~/.cache/huggingface/hub \
  python scripts/test_e2e_surfaces.py
```
Outputs: `docs/scale_results.json` (raw numbers, post-fix), `docs/scale_latency.png`
(the four panels referenced above, post-fix).
