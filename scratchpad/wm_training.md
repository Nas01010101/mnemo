# Neural World-Model (Dynamics) for Tenet — Training & Honest Evaluation

A GRU **neural temporal point process** over fact-update events, trained to replace/augment
Tenet's closed-form Gamma-exponential belief-state dynamics (`src/tenet/dynamics.py`).
Rigorously benchmarked against that shipped baseline and a marginal-rate baseline on a
**held-out, leakage-safe** split. Trained on the RTX 3080 (16 GB) over SSH; the Mac never
loaded torch. All numbers below are **mean ± std over 5 training seeds on ONE fixed test
split** (`--split-seed 0`), so dispersion is training variance and the closed-form
baselines are deterministic constants on that split.

## TL;DR verdict
- **Synthetic (planted structure): neural wins decisively and robustly.** NLL 1.99 vs Gamma
  2.76 (paired-bootstrap gain **+0.808 [+0.697, +0.926]** nats/event, CI excludes 0),
  calibration KS 0.26 vs 0.56, next-key 0.63 vs 0.39, next-value recall@5 **0.997** vs 0.05
  chance. This is the whole point: the planted periodic / bursty / cascade / value structure
  is *provably* outside the closed-form model's constant-per-key-hazard family, and the
  neural model captures it.
- **MAB FactConsolidation (real): mixed / honestly negative on the aggregate.** The heuristic
  key space is 99% right-censored singletons, so the degenerate marginal (near-zero hazard)
  "wins" the all-record NLL; neural is **worse** there (gain −0.805). On the *observed*
  supersessions neural is clearly better (8.92 vs Gamma 15.90). Net: **neural does not beat
  the closed-form model on the real MAB chains as an aggregate lifetime model — reported, not
  hidden.**
- **LongMemEval (real): suggestive but underpowered.** Only n=21 held-out lifetimes; neural
  NLL is much lower but negative (peaked density on a tiny sample) and next-value recall is at
  chance. Not a conclusion, a data-starved hint.

## Architecture (small & exportable by design)
GRU (not a transformer) so the product inference is a **pure-numpy forward pass** — the
`tenet` package keeps zero torch dependency.

- **Event input** (per fact-update, concatenated, `in_dim = 111`):
  key-class embedding (32) | source embedding (8) | bge-small value embedding projected
  384→64 | Δt-since-previous-event-on-this-key features `[log1p(Δt/unit), is_first]` (2) |
  wall-clock time features `[sin/cos hour, sin/cos day-of-week, tmean_flag]` (5, zeroed when
  timestamps are not meaningful, e.g. MAB serials).
- **Backbone**: 1-layer GRU, hidden 192. **276,417 params.** (Key-class vocab capped:
  classes with <20 events → `<other>`, so MAB's ~15k singleton heuristic keys don't bloat the
  heads. Un-capped it was 3.7M params / 14 MB; capped it is 276k / **1.0 MB**.)
- **Heads** on each step's hidden state:
  1. **HAZARD** — Weibull(scale, shape) for time-to-next-change on that event's key. Chosen
     over full Neural Hawkes because it is **closed-form integrable** (exact NLL) *and* its
     shape `k` makes the hazard **increasing (k>1, aging/periodic-ish)** or **decreasing
     (k<1, bursty clustering)** — exactly the structure a per-key **constant-hazard**
     Gamma-exponential (memoryless Lomax survival) provably cannot represent. NLL:
     observed `-(log k − log s + (k−1)(log t − log s) − (t/s)^k)`; censored `(t/s)^k = −log S`.
     Survival for `p_valid` is smooth at any age: `S(t)=exp(-(t/s)^k)`.
  2. **NEXT-KEY** — softmax over key-classes for the next event in the interleaved stream
     (the learned "ripple"/cascade). Trained with cross-entropy.
  3. **NEXT-VALUE** — predicted 384-d embedding of the key's next value, InfoNCE contrastive
     vs in-batch true-next embeddings (τ=0.07).
- Loss `= haz_nll + 1.0·nextkey_ce + 0.5·nextvalue_infonce`. Adam lr 2e-3, grad-clip 5, 15
  epochs, pooled over all sources with a source embedding. Deterministic per seed.

The GRU runs over the **global time-ordered event stream** of each window (multi-key), with
per-key Δt as an input feature — one model that serves all three heads and sees cross-key
cascades. Windows (≤512 events) are the leakage-safe unit: whole windows go to train XOR
test; per-key lifetimes are computed within a window (a key's last event is right-censored).

## Data (all three built; `scripts/wm_data.py`, torch-free, deterministic)
| source | events | seqs | key-classes | supersessions | uniq values | time | note |
|---|---|---|---|---|---|---|---|
| **synthetic** | 50,000 | 101 | 6 | 49,398 | 48 | wall-clock | planted structure (below) |
| **MAB FactConsolidation** | 25,677 | 4 ctx | 15,342 | 4,306 | 4,452 | serial | real counterfactual chains; sparse, 99% censored |
| **LongMemEval_S** | 93 | 45 | 27 | 48 | 70 | wall-clock | regex-mined cross-session updates; tiny/noisy |

**Synthetic planted structure** (each item is *outside* the Gamma family):
(a) **periodic** keys (mood, energy) change ~daily on weekdays near a per-actor preferred hour
→ time-of-day/day-of-week-modulated hazard; (b) **bursty** keys (task_status, checkin) change
in tight bursts then go quiet → decreasing within-burst hazard (Weibull k<1); (c) **cascade**
residence→job within days (correlated cross-key); (d) **value** structure: job titles walk a
ladder, cities drawn from a geography graph (→ learnable next-value).

MAB and LME are the honest reality: MAB parses fine but its heuristic key space is dominated
by singleton facts (per-key lifetime modeling is ~degenerate); LME's regex distillation yields
only 93 events. Both are reported as-is, not cherry-picked.

## Results (mean ± std, 5 seeds, fixed test split; Gamma/marginal deterministic)
### synthetic  (n_test = 10,180 lifetimes, 2% censored)
| metric | **neural** | Gamma (closed-form) | marginal |
|---|---|---|---|
| NLL, all (↓) | **1.988 ± 0.067** | 2.764 | 3.258 |
| NLL, observed (↓) | **1.746 ± 0.103** | 2.148 | 2.621 |
| calibration KS (↓) | **0.263 ± 0.018** | 0.555 | 0.691 |
| next-key acc (↑) | **0.628 ± 0.001** | — | 0.390 |
| next-value recall@5 (↑) | **0.997 ± 0.001** | — | 0.050 (chance) |
| paired-bootstrap NLL gain vs Gamma | **+0.808 [+0.697, +0.926]** | | |

### MAB FactConsolidation  (n_test = 4,870, **99% censored**)
| metric | **neural** | Gamma | marginal |
|---|---|---|---|
| NLL, all (↓) | 0.949 ± 0.004 | 0.142 | **0.135** |
| NLL, observed (↓) | **8.918 ± 0.001** | 15.903 | 13.551 |
| calibration KS (↓) | **0.450 ± 0.002** | 1.000 | 0.999 |
| next-value recall@5 (↑) | 0.056 ± 0.035 | — | 0.050 (chance) |
| paired-bootstrap NLL gain vs Gamma | −0.805 [−0.824, −0.784] | | |

MAB read honestly: the all-record NLL is dominated by censoring, where predicting a
near-zero hazard (marginal) is optimal — so neural loses there. But on the actual observed
supersessions and on calibration, neural is clearly better. **Bottom line: not a win on the
aggregate real-MAB metric.** (next-key is a trivial 1.0 for both because after capping,
MAB's rare classes collapse to `<other>`.)

### LongMemEval_S  (n_test = **21** — underpowered)
| metric | **neural** | Gamma | marginal |
|---|---|---|---|
| NLL, all (↓) | −2.572 ± 0.027 | 2.839 | 3.713 |
| NLL, observed (↓) | −5.083 ± 0.056 | 4.734 | 6.326 |
| calibration KS (↓) | **0.428 ± 0.031** | 0.896 | 0.913 |
| next-value recall@5 | 0.000 (n=12) | — | 0.050 |

Much lower NLL, but n=21 and negative NLL means very peaked densities on a handful of points;
next-value at 0/12. **Treat as a hint, not a result** — LME needs an LLM distiller to yield
enough clean update chains.

### Systems
- **Params 276,417** · train time **7.9 s** (15 epochs, RTX 3080) · weights **1.0 MB**
  (`data/dynamics_neural.npz`).
- **numpy inference 106 µs / `p_valid` call.**
- **numpy-vs-torch Weibull rel-drift 9.1e-08** — the exported pure-numpy forward reproduces
  the torch model to float precision (the export is trustworthy).

## Integration & non-regression (the load-bearing safety property)
- `MemoryCore._dynamics()` gains an env hook: `TENET_DYNAMICS=neural` (+ optional
  `TENET_NEURAL_NPZ`) swaps in `NeuralDynamics` (numpy-only, same
  `p_valid(skey, age, now)` / `expected_lifetime_days` interface as `Dynamics`, plus
  `predict_next_key` / `predict_next_value_embedding`). **Default is unchanged closed-form**;
  a missing/dim-mismatched npz **falls back to closed-form** so the world model can never
  break recall. The neural path needs the bge-small (384-d) local embedder it was trained on.
- **Green on RTX with a local bge embedder (no paid API):** `test_dynamics.py` ALL PASS,
  `test_memory.py` ALL PASS, and an **LLM-free churn-integrity guard** (a key updated 12×)
  keeps the latest value ranked **first** with one current fact under **both** default and
  neural dynamics — confirming the annotation-only invariant (lesson 452ae5fd: confidence must
  never rank-demote a doubted-but-current fact). Torch-free Mac-side check
  (`scripts/verify_neural_mac.py`) additionally proves survival monotonicity, softmax
  normalization, deterministic inference, and graceful fallback.

## Paper-ready paragraph
> We replace Tenet's closed-form Gamma-exponential belief-state dynamics with a compact
> (276k-param, 1-layer GRU) neural temporal point process over fact-update events, whose
> Weibull hazard head admits non-constant (increasing or decreasing) per-key hazards and whose
> auxiliary heads predict the next key to change (learned supersession ripple) and the next
> value (contrastively). On a controlled synthetic corpus with planted periodic, bursty, and
> cascading update structure — provably outside the closed-form model's memoryless family —
> the neural model reduces held-out negative log-likelihood from 2.76 to 1.99 nats/event
> (paired-bootstrap gain 0.81, 95% CI [0.70, 0.93]), roughly halves the survival
> mis-calibration (KS 0.56→0.26), and predicts the next value at 0.997 recall@5 versus 0.05
> chance, all stable across five training seeds on a fixed held-out split. On real
> counterfactual-update chains the story is deliberately mixed: the model improves calibration
> and observed-lifetime likelihood on MemoryAgentBench's FactConsolidation but loses the
> censoring-dominated aggregate NLL to a trivial marginal-rate baseline, and LongMemEval yields
> too few clean update chains (n=21) to conclude. The exported model is a single 1 MB numpy
> artifact evaluated at 106 µs per query, drops into the memory core behind an env flag without
> altering default behavior, and passes the store's churn-integrity guard — a doubted-but-current
> fact is annotated with lower confidence but never demoted below stale distractors.

## Reproduce (on the RTX box, dsocr venv: torch 2.6+cu124, transformers 4.46)
```
# data (torch-free, deterministic)
python scripts/wm_data.py --out data/wm --synthetic 50000 --mab --lme --mab-limit 40
# single run  ->  data/dynamics_neural.npz + train_log.json + results.json
python scripts/train_dynamics.py --data data/wm --epochs 15 --seed 0 --split-seed 0 \
    --out data/dynamics_neural.npz --results data/wm/results.json
# 5-seed sweep on the fixed split
python scripts/wm_sweep.py --seeds 5 --epochs 15 --python <venv>/bin/python
# non-regression (local bge embedder, no paid API)
PYTHONPATH=src python scripts/run_tests_rtx.py --npz data/dynamics_neural.npz
# torch-free Mac-side safety check
python scripts/verify_neural_mac.py
```

## Honest limitations / what would change the verdict
- Weibull captures *monotone* hazards; true multi-modal periodicity is only partially captured
  via time-feature covariates shifting the scale, not a multi-modal hazard shape. Discretized
  hazard bins would model periodicity more directly (future ablation).
- The synthetic win is on **planted** structure by construction; it demonstrates the neural
  model *can* capture what the closed-form cannot, but the magnitude is a property of the
  generator, not of real user data.
- The real-data signal is weak (MAB sparse/censored, LME tiny). A production verdict needs a
  larger real corpus with an LLM distiller producing clean, multi-event per-key chains.
- Next-key on MAB is uninformative (classes collapse to `<other>`); the ripple result is only
  meaningful on synthetic where cross-key cascades are planted.
```
Artifacts: data/dynamics_neural.npz (weights, 1.0 MB) · data/wm/{summary,results,sweep,train_log}.json
RTX working copy: anas@100.88.179.78:~/tenet-wm  (seed npz: ~/tenet-wm/data/dynamics_neural.npz)
```
