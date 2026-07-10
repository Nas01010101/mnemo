# Scale — where the first wall is, and what causes it

`docs/BENCHMARK.md` proves Tenet is *accurate*, at conversation scale (tens to low
thousands of memories, matching the standardized benchmarks it's evaluated on). This
doc asks a different question, end-to-end and at scale: **does the system hold up as
the store grows, and where does it stop?** Companion to `docs/DEPLOY.md` (what the live
Alibaba Cloud FC deploy inherits from these numbers) and `scripts/test_e2e_surfaces.py`
(proves one store is really shared across every surface — Python API, CLI, MCP,
LangGraph, HTTP API).

**TL;DR — the honest ceiling.** Recall crosses **100ms at ~10k facts** and **1s at
~100k facts** on this machine, driven almost entirely by re-fetching and
re-deserializing the *entire* embedding matrix from sqlite on every single call — Tenet
has **zero read-side caching**, by design today (see "cold vs warm" below), not by
oversight. Ingestion throughput through the real `store()` API collapses from **~2000
facts/s at 1k to ~13 facts/s at 100k** — worse than the read path, and the dominant
practical constraint long before 1M facts are reachable. The **first wall is RAM**, not
CPU: resident memory is already 1.15GB at 100k and extrapolates to **~7.8GB at 1M** —
on a 16GB dev machine already running other processes, that's swap territory, and
recall latency would very likely be *worse* than the linear extrapolation once
swapping starts (the extrapolation below is optimistic on that front, and says so).

## Methodology

- **Deterministic synthetic embeddings** (seeded `numpy` RNG, unit-normalized, d=384 —
  `bge-small-en-v1.5`'s dimensionality, the local embedder this repo ships), injected
  via `MemoryCore.store(..., _vec=...)` — an existing parameter, not something added
  for this probe. This is deliberate: a real embedder call costs a roughly constant
  ~10-30ms regardless of store size (see `recall.e2e_ms` vs `recall.fetch_deserialize_ms`
  below for exactly how much) — mixing that back in would conflate embedder latency
  with the store's OWN algorithmic scaling, which is what this probe isolates. This is
  a scaling/perf probe, not a retrieval-quality benchmark; no accuracy numbers here.
- **Corpus setup vs. measured operations are DELIBERATELY SEPARATE.** Reaching each
  milestone (1k/10k/100k) is done via `bulk_seed()` — a direct, single-transaction bulk
  `INSERT` against the *same* schema `store()` writes, used ONLY to build the corpus
  fast. The numbers reported as "ingestion throughput" always come from the REAL,
  unmodified `MemoryCore.store()` — including its real per-call `db.commit()` — run on
  a separate 500-call sample at each milestone. Why this split exists, found the hard
  way: a first version of this script seeded the whole corpus through real `store()`
  calls and was still running after 30 minutes before being killed — every commit on
  this machine pays a real fsync-backed transaction. That cost is exactly what's
  measured (see "ingestion ceiling" below); paying it 100,000 times just to build a
  test corpus wastes wall-clock without adding information.
- **Environment**: macOS, `EMBED_PROVIDER=local` (`sentence-transformers`/bge-small;
  torch happened to already be installed in this Python environment, so no fallback
  was needed), `HF_HOME=~/.cache/huggingface`, `TENET_DB_PATH` pointed at a scratch
  directory (this repo's `data/` symlink is TCC-blocked in the sandboxed shell this ran
  in — see `memory.py`'s own comment on `_DEFAULT_DB`/`TENET_DB_PATH`). This is one
  developer machine, not a dedicated benchmark box — treat absolute numbers as
  order-of-magnitude, the *shape* of each curve as the load-bearing result.

## Recall latency vs store size

| n | recall e2e p50 | recall e2e p95 | DB fetch+deserialize | BLAS matmul only |
|---:|---:|---:|---:|---:|
| 1,000 | 25.5 ms | 28.8 ms | 1.7 ms | 0.02 ms |
| 10,000 | 218.2 ms | 836.8 ms | 33.6 ms | 0.37 ms |
| 100,000 | 1,054.9 ms | 1,244.4 ms | 327.3 ms | 2.64 ms |

**Reads cross 100ms between 1k and 10k, and 1s between 10k and 100k.**

The breakdown is the important part: **`matmul` (the actual linear-algebra scoring
step, `mat @ qv`) stays tiny** — 0.02→0.37→2.64ms, exactly the O(n·d) BLAS cost the
architecture doc claims, and genuinely cheap even at 100k. The **DB fetch+deserialize**
step (`_rows_as_of()`'s SQL read + `np.frombuffer(b"".join(...))` reshape) is 15-100×
larger than the matmul at every size and grows faster — it, not the linear algebra, is
what a caller actually pays for at scale. **e2e latency is itself larger than
fetch+matmul combined** at every size (25.5 vs 1.7ms at n=1k; the gap is the local
embedder's per-query inference cost, ~24ms, genuinely constant — see the plot).

## Cold vs warm — does Tenet cache anything?

**No.** `recall()` re-runs `_rows_as_of()` (a full SQL read of every current row) and
rebuilds the embedding matrix from scratch on *every single call* — there is no
resident/cached matrix, no query-plan reuse, nothing kept warm between calls. Ten
back-to-back identical `recall()` calls on the same unchanged store, at n=1,000 and
n=100,000, show **no meaningful speedup** (ratios 1.06× and 0.87× — noise, not
caching). At n=10,000 a 2.25× speedup DID appear — almost certainly the OS filesystem
page cache (the sqlite file's pages staying resident after the first read), not
anything Tenet does — and it shrinks back to nothing at 100k once the working set
(~150MB of embedding BLOBs) is large enough that OS-level caching stops fully covering
it under normal memory pressure. **This is the single highest-leverage fix available**:
an in-process resident matrix, invalidated only on write, would turn every `recall()`
after the first at a given store size from an O(n) sqlite read into an O(1) lookup —
the BLAS step alone (2.64ms at 100k) is what recall could cost if the DB round-trip
were removed.

## Ingestion throughput — the real ceiling, and where it comes from

| pre-existing n | keyed insert (indexed, real `store()`) | unkeyed insert (`_nearest_current` scan) |
|---:|---:|---:|
| 0 | 2,035.6 facts/s | 1,669.7 facts/s |
| 1,000 | *(see 1k row)* | — |
| 2,000 | — | 179.9 facts/s |
| 4,000 | — | 71.0 facts/s |
| 6,000 | — | 34.7 facts/s |
| 8,000 | — | 29.0 facts/s |
| 10,000 | 78.2 facts/s | 25.8 facts/s |
| 100,000 | 13.3 facts/s | *(not run — see below)* |

Two **different** effects are visible here, and they matter differently:

**1. The unkeyed insert path is a genuine O(n)-per-insert algorithmic bug** (not
src-edited here — flagging for routing, see "Findings for routing" below).
`MemoryCore.store()`'s unkeyed branch (`key=None`) calls `_nearest_current(vec)`, which
is an **unvectorized Python `for` loop over every current row** (`memory.py`, the
`_nearest_current` method) — unlike `recall()`'s BLAS matmul, or the keyed path's
indexed `SELECT`, this does one Python-level dot product per row, per insert. The probe
above isolates exactly this: pre-seed the store to each size via the fast keyed path,
then time a *fixed* batch of 200 unkeyed inserts from there. Throughput drops
smoothly and monotonically as pre-existing size grows — 1,670 → 26 facts/s from 0 to
10,000 pre-existing facts, a clean O(1/n) decay consistent with O(n) cost per insert
(→ **O(n²) total** for bulk unkeyed ingestion). The same unvectorized-scan pattern
exists in the `kind="raw"` + `surprise_gate` path (memory.py, the redundant-observation
check) — same class of bug, not separately re-measured here. **This genuinely was not
run to 100k**: at the observed decay rate, a 100k-pre-existing unkeyed insert would
individually cost several seconds, making a representative batch there impractical
within this pass — the trend from 0→10k already establishes the O(n) shape
unambiguously; extrapolating the fit to 100k/1M would only restate "don't do this."

**2. The KEYED (indexed) insert path also visibly degrades with n — 2,036 → 78 → 13.3
facts/s, 1k→10k→100k — despite the `skey` index bounding its own `SELECT` to O(1)-ish.**
An isolated microbenchmark of the two candidate substeps (a bare indexed `SELECT` on
`skey`, and a bare `INSERT`+`commit()`) at matched store sizes did NOT reproduce
anywhere near this magnitude of slowdown (both stayed under ~0.6ms per call up to
n=20,000 in isolation) — so this real, measured, reproducible throughput collapse is
**not fully attributed** within this pass. The leading hypothesis, consistent with
every piece of evidence gathered: `store()` does a synchronous `db.commit()` on
**every single insert** (no batching, no WAL mode — sqlite's default rollback-journal
mode), and per-commit fsync cost is known to scale with total database file size/page
count under that journal mode; separately, this machine was running other memory- and
CPU-heavy processes throughout this measurement (RSS reached 1.15GB for this process
alone by n=100,000), which is a real confound for a shared dev box. **The actionable
fix is the same regardless of the exact split between these two causes**: never do a
synchronous per-call commit at scale — batch writes and commit periodically, or switch
to WAL mode (`PRAGMA journal_mode=WAL`), either of which is a `src/tenet/memory.py`
change and out of scope for this pass (flagged below for routing, not edited here).

## Memory & disk footprint

| n | sqlite file (disk) | resident matrix *if* cached (n×384×4 bytes) | process RSS |
|---:|---:|---:|---:|
| 1,000 | 2.0 MB | 1.5 MB | 456.8 MB |
| 10,000 | 19.9 MB | 14.7 MB | 456.8 MB |
| 100,000 | 198.6 MB | 146.5 MB | 1,157.5 MB |

DB-file-size scales linearly with n as expected (~2KB/fact, dominated by the 1536-byte
float32 embedding BLOB). RSS stays flat 1k→10k (both comfortably inside the ~450MB
baseline — mostly the loaded embedder model + Python/numpy/torch runtime, a fixed cost
independent of store size) then jumps sharply to 1.15GB at 100k, consistent with
repeated ~150MB matrix allocations during the timed `recall()` reps (garbage-collected
between calls but visible in peak RSS) plus general working-set growth.

## Extrapolation to 1M facts (NOT measured — linear fit, stated as such)

| metric | 100k (measured) | 1M (**extrapolated**, linear fit) |
|---|---:|---:|
| recall() e2e latency | 1,054.9 ms | **~10.0 s** |
| sqlite file size | 198.6 MB | **~1.99 GB** |
| process RSS | 1,157.5 MB | **~7.78 GB** |

**This is an optimistic extrapolation.** The read path is architecturally O(n·d)
(documented, and the matmul-only column above confirms it empirically), so a linear
fit for latency and disk are defensible first-order models. RSS crossing ~7.8GB on a
machine with other load is very plausibly where things get *worse* than linear — once
the OS starts swapping, every operation (sqlite I/O included) slows down
non-linearly, the same effect plausibly already visible (but not conclusively
isolated) in the keyed-ingestion-throughput collapse at 100k above. Read "10 seconds
at 1M" as a floor, not a ceiling.

## Verdict: first wall, and the fix

**The wall is RAM, arriving before either the read-latency wall or a hard sqlite
limit.** In order of when a real deployment would hit each:

1. **~10k facts: recall crosses 100ms.** Still usable for an interactive agent, but
   noticeably slower than the low-single-digit-ms numbers `docs/HARNESS.md` reports at
   conversation scale.
2. **~100k facts: recall crosses 1s; ingestion is down to ~13 facts/s.** An agent
   ingesting a real conversation history (hundreds of facts) would already feel this on
   the write path; a read-heavy workload would feel it on every call.
3. **~1M facts (extrapolated): ~8GB RSS, ~10s recall.** On the 16GB dev machine this
   ran on, this is swap territory before it's a raw compute problem — the process alone
   would want half the machine's RAM, with the OS, other processes, and (on a real
   deploy) the embedder model still needing their share.

**Fixes, ranked by leverage** (none applied here — `src/tenet/memory.py` is out of
scope for this pass; routed below):
1. **A resident, incrementally-updated embedding matrix** (append on insert, drop rows
   on supersession/archive, kept in Python-process memory) instead of rebuilding it
   from a full sqlite read on every `recall()` call. This alone would remove the
   dominant cost at every measured size (fetch+deserialize is 15-100× the matmul cost).
2. **Batch writes / WAL mode** for `store()`'s commit — turns the per-call fsync tax
   (the leading hypothesis for the keyed-path throughput collapse) into an amortized
   cost.
3. **Vectorize `_nearest_current`** (the unkeyed store path) — replace the Python
   `for`-loop dot-product scan with the same `mat @ vec` BLAS pattern `recall()`
   already uses. Fixes the confirmed O(n²) unkeyed-ingestion bug directly.
4. **An ANN index (e.g. sqlite-vec, HNSW)** once brute-force linear scan itself (not
   just the DB round-trip) becomes the bottleneck — per the matmul-only column above,
   that point is comfortably past 1M facts on this hardware; premature before fixes
   1-3 land.

## Findings for routing (src/tenet/ out of scope for this pass — not edited)

**1. `_nearest_current()` is an unvectorized O(n) Python loop (the confirmed
O(n²)-total bug).** `memory.py`'s unkeyed `store()` path (also the `kind="raw"` +
`surprise_gate` redundant-observation check) scans every current row with a per-row
Python `float(np.dot(...))` call instead of the single `mat @ vec` BLAS call `recall()`
already uses for the identical operation. Empirically confirmed above (1,670→26
facts/s as pre-existing store size grows 0→10,000, clean O(1/n) throughput decay). Fix
#3 above.

**2. The just-shipped embedding-based key resolution (`_KEY_RESOLUTION`, commit
`5694afd`, default-on as of 2026-07-10) has a plausible false-positive mode worth a
second look — found via `scripts/test_e2e_surfaces.py`, not this scale probe.**
Two UNRELATED facts stored under keys sharing the same *subject* (e.g.
`"e2e::surface_probe"` and `"e2e::temporal_probe"`) with attribute names that both
happen to contain a common word (here, "probe") cross-superseded each other:
`_resolve_key_supersede`'s guards (`_TAU_KEY=0.78` attribute-embedding similarity AND
`_TEXT_FLOOR=0.35` fact-text similarity) both cleared for this pair on the local
`bge-small` embedder, which is weaker than the Qwen Cloud embedder the shipped product
uses and may show this pattern more or less depending on embedder choice — not verified
against `EMBED_PROVIDER=qwen` in this pass. Minimal repro:
```python
core.store("The e2e probe fact is: X wrote this.", key="e2e::surface_probe", pinned=True)
core.store("temporal probe v1", key="e2e::temporal_probe")
# -> the FIRST fact (different subject-attribute pair, unrelated content) is now
#    superseded (expired_at set), and the second inherits its `pinned` flag.
```
This is very plausibly within the false-positive rate the feature's own commit message
already reports as measured/accepted (not a regression claim — the sweep that set
`_TAU_KEY=0.78` was described as "0% false-fire" on its own labeled set, so this may
simply be an example outside that set's coverage) — reporting for whoever owns that
fix to assess against their own eval, not asserting it's wrong. This script's own probe
keys were changed to non-colliding subjects once traced (see
`scripts/test_e2e_surfaces.py`'s module docstring for the full account); no `src/`
changes were made.

**3. `src/tenet/memory.py` is 751 lines**, over the repo's own 500-line file-size
convention (noted in passing while reading the current file for this pass; a
consequence of the same in-progress supersession work above, not something to act on
here).

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
Outputs: `docs/scale_results.json` (raw numbers), `docs/scale_latency.png` (the four
panels referenced above).
