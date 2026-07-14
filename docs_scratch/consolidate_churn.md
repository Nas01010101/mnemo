# TENET_CONSOLIDATE — ChurnBench measurement (PAUSED: RTX busy, 2026-07-13)

The flag itself is DONE and shipped (commit 4a7397f): opt-in write-time consolidation
(`TENET_CONSOLIDATE=1` / `store(..., consolidate=True)` / retroactive
`consolidate_sweep()`), default OFF, 16/16 suites green incl. `scripts/test_consolidate.py`.

What's measured so far (RTX `qwen2.5:7b` reader+distiller, paired arms sharing
ingestion byte-for-byte via the sqlite backup-API snapshot):

- Smoke U=2, n=10: tenet 40.0% vs tenet_consolidate 20.0% — CIs fully overlap; noise.
  NOTE the 7B tier is a weak distiller (one smoke store had ZERO supersessions, so
  consolidation had nothing to act on) — absolute numbers here are far below §9's
  qwen3.7-plus tier; only the PAIRED comparison is meaningful.
- Full U=2/8/32 n=50/point run was killed mid-U=8 (RTX needed for other work).
  U=2 and U=8 ingestion caches survive under `data/cache/churn/1/` — a relaunch
  skips straight to U=32 ingestion (~30-45 min saved).

## Relaunch (when the RTX is free)

```bash
env LLM_PROVIDER=ollama OLLAMA_BASE_URL=http://100.88.179.78:11434/v1 \
    OLLAMA_MODEL=qwen2.5:7b EMBED_PROVIDER=local \
    python3 scripts/bench_churn.py --updates 2,8,32 --principals 10 \
    --arms tenet,tenet_consolidate --consistency-threshold 0.70 --currency-context \
    --workers 4 --principal-workers 2 \
    --dump scratchpad/churn_consolidate_misses.jsonl \
    --out scratchpad/churn_consolidate_7b.json
```

Success bar (pre-registered): consolidate arm ≥ tenet+1+2 arm at U=32, ideally ~flat
(Mem0-parity, which would make ChurnBench half-life 32, tied-for-first, while keeping
belief history — Mem0-style deletes it).

## RULER-MH chain eval relaunch (same box, run after)

```bash
env LLM_PROVIDER=ollama OLLAMA_BASE_URL=http://100.88.179.78:11434/v1 \
    OLLAMA_MODEL=qwen2.5:7b EMBED_PROVIDER=local TOKENIZERS_PARALLELISM=false \
    python3 scripts/exp_ruler_mh_hop.py --qpc 100
```

Iterative Self-Ask arm (answer hop 1 → anchor hop 2's retrieval on the intermediate
answer), paired SubEM + McNemar vs baseline at identical char budget. Single-shot
(non-iterative) smoke at n=10 read 40 vs 50 baseline — noise-level, and it predates
the iterative chaining + English-only decomposition fixes. Goal: move RULER-MH from
45 toward HippoRAG-v2's 66; every +4pp on this cell is ~+1pp on the MAB-AR average
(Tenet 59.3 vs leader 65.1).
