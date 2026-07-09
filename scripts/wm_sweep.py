"""Run the neural world-model over N training seeds on ONE fixed test split, then
aggregate the per-source metrics as mean ± std. The split is fixed (--split-seed 0) so
the held-out set is identical across seeds — dispersion here is TRAINING variance, and
Gamma/marginal are deterministic on that split (their numbers are constant).

Usage (on RTX):  python scripts/wm_sweep.py --seeds 5 --epochs 15
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SOURCES = ["synthetic", "mab", "lme"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wm")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="data/wm/sweep.json")
    args = ap.parse_args()

    runs = []
    for seed in range(args.seeds):
        res_path = Path(args.data) / f"results_seed{seed}.json"
        npz = Path(args.data) / ("dynamics_neural.npz" if seed == 0 else f"dyn_seed{seed}.npz")
        cmd = [args.python, "scripts/train_dynamics.py", "--data", args.data,
               "--epochs", str(args.epochs), "--seed", str(seed),
               "--split-seed", str(args.split_seed), "--out", str(npz),
               "--log", str(Path(args.data) / f"log_seed{seed}.json"),
               "--results", str(res_path)]
        print(f"\n===== seed {seed} =====", flush=True)
        subprocess.run(cmd, check=True)
        runs.append(json.loads(res_path.read_text()))

    # aggregate
    agg = {}
    metrics = ["neural_nll", "neural_nll_obs", "neural_ks", "nextkey_neural_acc",
               "nextval_recall@5"]
    const = ["gamma_nll", "gamma_nll_obs", "gamma_ks", "marginal_nll", "marginal_ks",
             "nextkey_marginal_acc", "n_test_life", "cens_frac", "nextval_chance"]
    for s in SOURCES:
        if s not in runs[0]:
            continue
        row = {"seeds": args.seeds}
        for m in metrics:
            vals = np.array([r[s][m] for r in runs], dtype=float)
            row[m] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        for c in const:
            row[c] = runs[0][s][c]
        # bootstrap CI (from seed 0 — split-fixed, representative)
        row["nll_gain_vs_gamma"] = runs[0][s]["nll_gain_vs_gamma"]
        row["nll_gain_vs_marginal"] = runs[0][s]["nll_gain_vs_marginal"]
        agg[s] = row
    agg["_infer_us"] = runs[0].get("_infer_us")
    agg["_numpy_torch_reldrift"] = runs[0].get("_numpy_torch_reldrift")
    Path(args.out).write_text(json.dumps(agg, indent=2))

    # report
    print("\n\n================ SWEEP SUMMARY (mean±std over "
          f"{args.seeds} init seeds, fixed test split {args.split_seed}) ================")
    for s in SOURCES:
        if s not in agg:
            continue
        r = agg[s]
        print(f"\n[{s}]  n_test={r['n_test_life']}  cens_frac={r['cens_frac']:.2f}")
        print(f"  NLL(all)   neural {r['neural_nll']['mean']:.3f}±{r['neural_nll']['std']:.3f}"
              f"  | gamma {r['gamma_nll']:.3f}  | marginal {r['marginal_nll']:.3f}")
        print(f"  NLL(obs)   neural {r['neural_nll_obs']['mean']:.3f}±{r['neural_nll_obs']['std']:.3f}"
              f"  | gamma {r['gamma_nll_obs']:.3f}  | marginal {r['marginal_nll']:.3f}")
        print(f"  calib KS   neural {r['neural_ks']['mean']:.3f}±{r['neural_ks']['std']:.3f}"
              f"  | gamma {r['gamma_ks']:.3f}  | marginal {r['marginal_ks']:.3f}")
        print(f"  next-key   neural {r['nextkey_neural_acc']['mean']:.3f}±{r['nextkey_neural_acc']['std']:.3f}"
              f"  | marginal {r['nextkey_marginal_acc']:.3f}")
        print(f"  next-val R@5 neural {r['nextval_recall@5']['mean']:.3f}±{r['nextval_recall@5']['std']:.3f}"
              f"  | chance {r['nextval_chance']:.3f}")
        g = r["nll_gain_vs_gamma"]
        print(f"  NLL gain vs Gamma (paired bootstrap): {g['mean']:+.3f} "
              f"[{g['lo']:+.3f}, {g['hi']:+.3f}]  (positive = neural better)")
    print(f"\ninference {agg['_infer_us']:.1f} µs/call  numpy-torch drift {agg['_numpy_torch_reldrift']:.1e}")


if __name__ == "__main__":
    main()
