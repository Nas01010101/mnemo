"""Scale probe — drive MemoryCore to 1k/10k/100k facts and measure the SYSTEM,
not the embedder.

Track 1 asks "efficient storage & retrieval... at scale". BENCHMARK.md's numbers are
all at conversation scale (tens to low thousands of memories); this script answers a
different question: where's the first wall as the store grows, and what causes it.

Methodology (read this before trusting the numbers):
  - Embeddings are DETERMINISTIC SYNTHETIC unit vectors (seeded RNG, d=384 — bge-small's
    dimensionality), injected via `MemoryCore.store(..., _vec=...)`, which is a real,
    already-existing parameter (memory.py: "network call — outside the lock", i.e. it
    exists precisely so callers can bypass the embedding call). This is deliberate: a
    real local/cloud embedder call is O(1) per text and constant w.r.t. store size — that
    cost has been measured elsewhere (BENCHMARK.md, HARNESS.md) and stays fixed at scale.
    Mixing it back in here would conflate embedder latency with the store's OWN
    algorithmic scaling, which is what this script isolates. The one place embedder cost
    DOES matter (a live recall() query) is measured separately and reported as a fixed
    per-query constant, added on top of the store-scaling numbers below.
  - This is a scaling/perf probe, not a retrieval-quality benchmark — no accuracy numbers
    here; see BENCHMARK.md for those.

Run: python scripts/bench_scale.py                      # 1k/10k/100k + unkeyed-path probe
     python scripts/bench_scale.py --sizes 1000,10000    # skip 100k (faster iteration)
     python scripts/bench_scale.py --skip-unkeyed-probe  # skip the O(n^2) side-experiment
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

# ---- env (must be set before importing tenet — config.py reads at import time) ------
os.environ.setdefault("EMBED_PROVIDER", "local")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
from tenet.memory import MemoryCore  # noqa: E402

D = 384  # bge-small-en-v1.5 dimensionality — what a real deploy actually stores


def unit_vecs(rng: np.random.Generator, n: int, d: int = D) -> np.ndarray:
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def rss_mb() -> float:
    """Peak RSS so far, in MB. macOS ru_maxrss is bytes; Linux is KB — normalize."""
    kb_or_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_b / (1024 * 1024) if sys.platform == "darwin" else kb_or_b / 1024


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    idx = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return xs[idx]


# ---- ingestion --------------------------------------------------------------------
def ingest_batch(core: MemoryCore, rng: np.random.Generator, start: int, n: int,
                  *, keyed: bool = True, raw_surprise_gate: float | None = None) -> float:
    """Insert n synthetic facts (ids start..start+n) through the REAL, unmodified
    `MemoryCore.store()` API — including its per-call `db.commit()` (memory.py
    ~line 217). This is what a real deploy's ingestion throughput actually pays;
    it is the number reported as "facts/sec" everywhere in this script. Returns
    elapsed seconds."""
    vecs = unit_vecs(rng, n)
    t0 = time.perf_counter()
    for i in range(n):
        idx = start + i
        text = f"synthetic fact #{idx}: entity-{idx} attribute value {rng.integers(0, 1_000_000)}"
        if keyed:
            core.store(text, key=f"entity{idx}::attr{idx % 7}", _vec=vecs[i])
        else:
            core.store(text, kind="raw" if raw_surprise_gate is not None else "fact",
                       surprise_gate=raw_surprise_gate, _vec=vecs[i])
    return time.perf_counter() - t0


def bulk_seed(core: MemoryCore, rng: np.random.Generator, start: int, n: int) -> float:
    """Fast corpus seeding: a direct, single-transaction bulk INSERT against the exact
    same `memories` table schema `store()` writes to (same columns, same encoding —
    `np.ndarray.tobytes()`, the identical format `store()`/`recall()` read back with
    `np.frombuffer`) — but skipping store()'s per-row Python work and, critically, its
    per-call `db.commit()`.

    This is NOT the measured code path — it exists ONLY to make reaching 100k tractable
    in a benchmark's setup phase. store()'s real per-call commit() cost (the dominant
    ingestion-throughput bottleneck measured by this script — see SCALE.md) is measured
    separately via `ingest_batch` (the real, unmodified API) on a small representative
    sample at each milestone. Discovered the hard way: a first version of this script
    seeded ALL of 1k/10k/100k through real store() calls and took >30 minutes before
    being killed — every commit() on this machine's filesystem does a full fsync-backed
    transaction, ~15-20ms each, so 100k serial commits alone is ~25-30 minutes wall
    time. That IS a real, reportable finding (see SCALE.md "ingestion ceiling"); it is
    not something this benchmark should pay 100,000 times over just to build a corpus.
    """
    vecs = unit_vecs(rng, n)
    now = core._now()
    rows = [
        (f"synthetic fact #{start + i}: entity-{start + i} attribute value {rng.integers(0, 1_000_000)}",
         "fact", None, f"entity{start + i}::attr{(start + i) % 7}", vecs[i].tobytes(),
         0.5, now, now, now, 0)
        for i in range(n)
    ]
    t0 = time.perf_counter()
    with core._lock:
        core.db.executemany(
            "INSERT INTO memories(text, kind, source, skey, embedding, salience, valid_at, "
            "created_at, last_access, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        core.db.commit()
        core._dyn_dirty = True
    return time.perf_counter() - t0


# ---- recall timing ------------------------------------------------------------------
def time_recall(core: MemoryCore, query_vec: np.ndarray, reps: int) -> dict:
    """Times core.recall() end-to-end (real path, incl. a live embed() call for the
    query text) AND the store-internal deserialize+matmul substep in isolation (same
    private accessors recall() itself uses — read-only, no src edits), so embedder
    cost (fixed, ~constant w.r.t. n) can be told apart from the store's own scaling."""
    # end-to-end (includes one real local-embedder call per rep — the query text is
    # fixed so the embedder's own cost is ~constant across reps/sizes)
    e2e_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        core.recall("what is the value of entity-42 attribute 3?", k=10)
        e2e_ms.append((time.perf_counter() - t0) * 1000)

    # isolated substep: DB fetch+deserialize, and the BLAS matmul, timed separately —
    # this is exactly what recall()'s body does (memory.py `_rows_as_of` + the
    # `np.frombuffer(b"".join(...))` reshape + `mat @ qv`), called here read-only from
    # outside to attribute where end-to-end time goes. Not a copy of the algorithm,
    # a timing harness around the same public/private methods recall() itself calls.
    fetch_ms, matmul_ms = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        rows = core._rows_as_of(None)
        mat = np.frombuffer(b"".join(row["embedding"] for row in rows), dtype=np.float32).reshape(len(rows), -1)
        t1 = time.perf_counter()
        _ = mat @ query_vec
        t2 = time.perf_counter()
        fetch_ms.append((t1 - t0) * 1000)
        matmul_ms.append((t2 - t1) * 1000)

    return {
        "e2e_ms_p50": round(percentile(e2e_ms, 50), 3),
        "e2e_ms_p95": round(percentile(e2e_ms, 95), 3),
        "fetch_deserialize_ms_p50": round(percentile(fetch_ms, 50), 3),
        "matmul_ms_p50": round(percentile(matmul_ms, 50), 3),
    }


def cold_vs_warm(core: MemoryCore) -> dict:
    """Does back-to-back recall() get faster the 2nd time (a resident/cached matrix)
    or not (full reload every call)? Ten calls, same query, same store — no writes
    in between."""
    ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        core.recall("what is the value of entity-42 attribute 3?", k=10)
        ms.append((time.perf_counter() - t0) * 1000)
    first, rest = ms[0], ms[1:]
    return {
        "first_call_ms": round(first, 3),
        "subsequent_calls_ms_mean": round(sum(rest) / len(rest), 3),
        "speedup_ratio": round(first / (sum(rest) / len(rest)), 2) if rest else None,
        "verdict": ("NO warm-cache speedup — every call re-fetches+re-deserializes the "
                     "full matrix from sqlite (confirms: cold every query)"
                     if rest and abs(first / (sum(rest) / len(rest)) - 1) < 0.30
                     else "some speedup observed"),
    }


# ---- main scale sweep ---------------------------------------------------------------
def run_scale_sweep(sizes: list[int], db_path: Path, reps: int, measured_sample: int) -> list[dict]:
    """Builds the corpus to each milestone in `sizes` via `bulk_seed` (fast, see its
    docstring for why), then measures REAL `store()` ingestion throughput on
    `measured_sample` more real calls at that store size (i.e. throughput AT n facts
    already present, the number that matters — does per-insert cost grow with store
    size?), then measures recall latency, RSS, and DB size at n = target."""
    core = MemoryCore(db_path)
    rng = np.random.default_rng(1234)
    results = []
    done = 0
    for target in sizes:
        to_seed = max(0, target - done - measured_sample)
        if to_seed:
            seed_s = bulk_seed(core, rng, done, to_seed)
            done += to_seed
            print(f"  (bulk-seeded {to_seed} facts in {seed_s:.2f}s — corpus setup, not measured ingestion)")

        measured_n = min(measured_sample, target - done)
        ingest_s = ingest_batch(core, rng, done, measured_n, keyed=True) if measured_n > 0 else 0.0
        done += measured_n
        gc.collect()

        qvec = unit_vecs(np.random.default_rng(999), 1)[0]
        recall_stats = time_recall(core, qvec, reps)
        cw = cold_vs_warm(core)

        db_bytes = db_path.stat().st_size
        matrix_bytes = done * D * 4  # float32 — size of the transient matmul allocation

        row = {
            "n": done,
            "ingest_measured_facts": measured_n,
            "ingest_measured_seconds": round(ingest_s, 3),
            "ingest_facts_per_sec": round(measured_n / ingest_s, 1) if ingest_s > 0 else None,
            "recall": recall_stats,
            "cold_vs_warm": cw,
            "db_disk_mb": round(db_bytes / (1024 * 1024), 2),
            "resident_matrix_mb_if_cached": round(matrix_bytes / (1024 * 1024), 2),
            "rss_mb": round(rss_mb(), 1),
        }
        print(f"n={done:>7}  ingest(real store())={row['ingest_facts_per_sec']:>7} facts/s  "
              f"recall_e2e_p50={recall_stats['e2e_ms_p50']:>8}ms  "
              f"fetch={recall_stats['fetch_deserialize_ms_p50']:>7}ms  "
              f"matmul={recall_stats['matmul_ms_p50']:>6}ms  "
              f"db={row['db_disk_mb']:>7}MB  rss={row['rss_mb']:>7}MB")
        results.append(row)
    core.close()
    return results


# ---- unkeyed-path O(n) per-insert probe (the store()-side scale bug) -----------------
def run_unkeyed_probe(pre_sizes: list[int], db_path: Path, batch: int) -> list[dict]:
    """MemoryCore.store()'s unkeyed path (key=None) calls _nearest_current(), an
    unvectorized Python for-loop over EVERY current row (memory.py ~line 581-588) — O(n)
    per insert, unlike the keyed path (indexed SELECT, O(1)-ish regardless of store
    size). This directly times that divergence: seed the store to each `pre_sizes`
    level with FAST keyed inserts, then time a small `batch` of UNKEYED inserts from
    there, to show unkeyed throughput degrading as the store grows (confirms O(n) per
    insert -> O(n^2) total for bulk unkeyed ingestion)."""
    core = MemoryCore(db_path)
    rng = np.random.default_rng(4242)
    results = []
    done = 0
    for target in pre_sizes:
        if target > done:
            bulk_seed(core, rng, done, target - done)  # fast corpus setup, see bulk_seed docstring
            done = target
        elapsed = ingest_batch(core, rng, done + 1_000_000, batch, keyed=False)
        rate = batch / elapsed if elapsed > 0 else None
        print(f"  pre-existing n={done:>6}  unkeyed insert of {batch}: "
              f"{elapsed:.3f}s  ({rate:.1f} facts/s)" if rate else "n/a")
        results.append({"pre_existing_n": done, "unkeyed_batch": batch,
                         "elapsed_s": round(elapsed, 3),
                         "facts_per_sec": round(rate, 1) if rate else None})
    core.close()
    return results


def fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope, intercept — stdlib-free, no scipy needed."""
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="1000,10000,100000")
    ap.add_argument("--reps", type=int, default=15, help="recall() repetitions per size")
    ap.add_argument("--measured-sample", type=int, default=300,
                     help="real store() calls measured at each milestone (rest of the "
                          "corpus is bulk-seeded — see bulk_seed docstring)")
    ap.add_argument("--unkeyed-probe-sizes", default="0,2000,4000,6000,8000,10000")
    ap.add_argument("--unkeyed-probe-batch", type=int, default=200)
    ap.add_argument("--skip-unkeyed-probe", action="store_true")
    ap.add_argument("--scratch-dir", default=os.environ.get(
        "TENET_SCALE_SCRATCH", "/tmp/tenet_scale_bench"))
    ap.add_argument("--out-json", default=str(
        Path(__file__).resolve().parent.parent / "docs" / "scale_results.json"))
    ap.add_argument("--out-plot", default=str(
        Path(__file__).resolve().parent.parent / "docs" / "scale_latency.png"))
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"=== scale sweep: sizes={sizes} (deterministic synthetic embeddings, "
          f"EMBED_PROVIDER={os.environ.get('EMBED_PROVIDER')}) ===")
    sweep_db = scratch / "scale_sweep.db"
    sweep_db.unlink(missing_ok=True)
    scale_results = run_scale_sweep(sizes, sweep_db, args.reps, args.measured_sample)

    unkeyed_results = []
    if not args.skip_unkeyed_probe:
        print("\n=== unkeyed-store-path probe (the O(n) per-insert suspect) ===")
        probe_sizes = [int(s) for s in args.unkeyed_probe_sizes.split(",")]
        probe_db = scratch / "scale_unkeyed_probe.db"
        probe_db.unlink(missing_ok=True)
        unkeyed_results = run_unkeyed_probe(probe_sizes, probe_db, args.unkeyed_probe_batch)

    # ---- extrapolation to 1M (linear fit on the measured e2e recall latency; the
    # store's read path is documented O(n*d) linear-scan, so a linear fit is the
    # honest model — NOT measured at 1M, clearly labeled as extrapolation) --------
    ns = [r["n"] for r in scale_results]
    e2e = [r["recall"]["e2e_ms_p50"] for r in scale_results]
    extrapolation = {}
    if len(ns) >= 2:
        slope, intercept = fit_linear(ns, e2e)
        pred_1m = slope * 1_000_000 + intercept
        extrapolation = {
            "recall_ms_per_fact_slope": round(slope, 6),
            "recall_ms_intercept": round(intercept, 3),
            "predicted_e2e_recall_ms_at_1M": round(pred_1m, 1),
            "method": "linear fit on measured n vs e2e_ms_p50 (recall is documented O(n*d))",
        }
        db_slope, db_int = fit_linear(ns, [r["db_disk_mb"] for r in scale_results])
        extrapolation["predicted_db_disk_mb_at_1M"] = round(db_slope * 1_000_000 + db_int, 1)
        rss_slope, rss_int = fit_linear(ns, [r["rss_mb"] for r in scale_results])
        extrapolation["predicted_rss_mb_at_1M"] = round(rss_slope * 1_000_000 + rss_int, 1)

    out = {
        "d": D, "reps": args.reps, "sizes": sizes,
        "scale_sweep": scale_results,
        "unkeyed_probe": unkeyed_results,
        "extrapolation_to_1M": extrapolation,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")

    _plot(scale_results, unkeyed_results, extrapolation, args.out_plot)
    print(f"wrote {args.out_plot}")

    # ---- verdict -----------------------------------------------------------------
    print("\n=== verdict ===")
    for r in scale_results:
        wall = []
        if r["recall"]["e2e_ms_p50"] >= 1000:
            wall.append(">1s")
        elif r["recall"]["e2e_ms_p50"] >= 100:
            wall.append(">100ms")
        if wall:
            print(f"  recall crosses {', '.join(wall)} at n={r['n']} "
                  f"(e2e_ms_p50={r['recall']['e2e_ms_p50']})")
    return 0


def _plot(scale_results, unkeyed_results, extrapolation, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless, no torch/GUI backend needed
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in scale_results]
    e2e = [r["recall"]["e2e_ms_p50"] for r in scale_results]
    fetch = [r["recall"]["fetch_deserialize_ms_p50"] for r in scale_results]
    matmul = [r["recall"]["matmul_ms_p50"] for r in scale_results]
    db_mb = [r["db_disk_mb"] for r in scale_results]
    rss = [r["rss_mb"] for r in scale_results]
    ingest_rate = [r["ingest_facts_per_sec"] for r in scale_results]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0][0]
    ax.plot(ns, e2e, "o-", label="recall() end-to-end", color="#8b7cf8")
    ax.plot(ns, fetch, "s--", label="DB fetch+deserialize", color="#6a5acd")
    ax.plot(ns, matmul, "^--", label="BLAS matmul only", color="#b31b1b")
    if extrapolation:
        x1m = 1_000_000
        y1m = extrapolation["recall_ms_per_fact_slope"] * x1m + extrapolation["recall_ms_intercept"]
        ax.plot([ns[-1], x1m], [e2e[-1], y1m], ":", color="gray", label="linear extrapolation")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("store size (facts)"); ax.set_ylabel("latency (ms)")
    ax.set_title("recall() latency vs store size")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    # log-y + log-x: two DIFFERENT experiments share this axis (main sweep's keyed
    # points at n=1k/10k/100k; the separate unkeyed probe at pre-existing 0..10k) —
    # log scale keeps both legible instead of the unkeyed curve's steep early drop
    # swamping the keyed curve's later, smaller-magnitude decline.
    ax.plot(ns, ingest_rate, "o-", color="#8b7cf8", label="keyed insert (indexed, main sweep)")
    if unkeyed_results:
        ux = [r["pre_existing_n"] + 1 for r in unkeyed_results]  # +1: log-x can't plot 0
        uy = [r["facts_per_sec"] for r in unkeyed_results if r["facts_per_sec"]]
        if len(uy) == len(ux):
            ax.plot(ux, uy, "s--", color="#b31b1b", label="unkeyed insert (_nearest_current scan, separate probe)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("pre-existing store size (facts)"); ax.set_ylabel("insert throughput (facts/s, log)")
    ax.set_title("ingestion throughput: keyed vs unkeyed insert path", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(ns, db_mb, "o-", color="#8b7cf8")
    ax.set_xlabel("store size (facts)"); ax.set_ylabel("DB file size (MB)")
    ax.set_title("sqlite file size vs store size")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(ns, rss, "o-", color="#8b7cf8")
    ax.set_xlabel("store size (facts)"); ax.set_ylabel("process RSS (MB)")
    ax.set_title("resident memory vs store size")
    ax.grid(alpha=0.3)

    fig.suptitle("Tenet MemoryCore — scale probe (synthetic embeddings, d=384)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
