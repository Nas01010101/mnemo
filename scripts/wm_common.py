"""Shared, TORCH-FREE core for the neural world-model: sequence building,
leakage-safe splitting, closed-form baselines (Gamma from dynamics.py + marginal
exponential), and the survival/calibration metrics. Imported by both the Mac-side
sanity check and the RTX training script, so the baselines and metrics are IDENTICAL
across them.

Time-to-event convention (per source, in that source's native unit):
  - a per-key chain e_0..e_{m-1} sorted by t gives OBSERVED lifetimes
    L_i = t_{i+1}-t_i for i<m-1 (a value was superseded after L_i), and one
    RIGHT-CENSORED lifetime for e_{m-1}: C = T_obs - t_{m-1} (still current at horizon).
  - T_obs = max event t within the source (the observation horizon).
The split unit is the (seq,key) CHAIN: a whole per-key lifetime history goes to
train XOR test — the correct leakage boundary for a per-key temporal point process
(never let a key's future sit in test while its past trains). Never random-k-fold.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# import the SHIPPED closed-form model — the baseline we must beat/measure against
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet.dynamics import Dynamics, _A0, _B0  # noqa: E402

DAY = 86400.0


# ---------------------------------------------------------------------------
# load + chain building
# ---------------------------------------------------------------------------
def load_events(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def time_unit(events: list[dict]) -> float:
    """Native->model time unit divisor: wall-clock sources -> days; serial -> 1."""
    return DAY if events and events[0].get("tmean", 0) else 1.0


def make_windows(events: list[dict], max_len: int = 512) -> list[list[dict]]:
    """Split events into WINDOWS — the leakage-safe unit shared by the neural model
    (a window = one GRU sequence) and the baselines. A window is a time-ordered,
    multi-key slice of ONE seq; long seqs (e.g. MAB's few giant contexts) are chunked
    into consecutive windows so (a) the GRU length is bounded and (b) there are enough
    units to split 80/20. Within a window, per-key lifetimes are computed independently
    (a key's last event in the window is censored), so windows never leak across the
    train/test boundary."""
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_seq[e["seq"]].append(e)
    windows: list[list[dict]] = []
    for evs in by_seq.values():
        evs.sort(key=lambda e: e["t"])
        for i in range(0, len(evs), max_len):
            windows.append(evs[i:i + max_len])
    return windows


def split_windows(windows: list[list[dict]], frac_test: float = 0.2,
                  seed: int = 0) -> tuple[list, list]:
    """Leakage-safe: whole windows to train XOR test, deterministic by seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(windows))
    n_test = max(1, int(len(windows) * frac_test))
    test_idx = set(perm[:n_test].tolist())
    train = [w for i, w in enumerate(windows) if i not in test_idx]
    test = [w for i, w in enumerate(windows) if i in test_idx]
    return train, test


def lifetimes(windows: list[list[dict]], t_obs: float, unit: float,
              min_life: float = 1e-6) -> list[dict]:
    """Per-key lifetime records, computed WITHIN each window (leakage-safe):
      {kclass, key, life (model units), censored 0/1, t_event, event, hist, win_pos}
    For a key's events inside a window, consecutive gaps are OBSERVED lifetimes and the
    last is RIGHT-CENSORED at t_obs. hist = same-key events up to & incl this one."""
    out = []
    for win in windows:
        by_key: dict[str, list[dict]] = defaultdict(list)
        for e in win:
            by_key[e["key"]].append(e)
        for chain in by_key.values():
            chain.sort(key=lambda e: e["t"])
            for i, e in enumerate(chain):
                if i < len(chain) - 1:
                    life = (chain[i + 1]["t"] - e["t"]) / unit
                    cens = 0
                else:
                    life = (t_obs - e["t"]) / unit
                    cens = 1
                if life < min_life:
                    continue
                out.append({"kclass": e["kclass"], "key": e["key"], "life": float(life),
                            "censored": cens, "t_event": e["t"], "event": e,
                            "hist": chain[: i + 1]})
    return out


# ---------------------------------------------------------------------------
# baseline 1 — closed-form Gamma-exponential (the SHIPPED Dynamics model)
# ---------------------------------------------------------------------------
def fit_gamma(train_windows: list[list[dict]], t_obs: float) -> Dynamics:
    """Fit the shipped Dynamics on train windows. Per-key within each window: observed
    lifetimes -> invalid_at set, the key's last event -> censored (None). Dynamics fits
    in SECONDS (its _B0 prior is seconds); wall-clock sources are consistent, serial
    sources are mis-scaled by that prior (reported honestly)."""
    rows = []
    for win in train_windows:
        by_key: dict[str, list[dict]] = defaultdict(list)
        for e in win:
            by_key[e["key"]].append(e)
        for chain in by_key.values():
            chain.sort(key=lambda e: e["t"])
            for i, e in enumerate(chain):
                inv = chain[i + 1]["t"] if i < len(chain) - 1 else None
                rows.append({"skey": e["key"], "valid_at": e["t"], "invalid_at": inv})
    return Dynamics.fit(rows, now=t_obs)


def gamma_nll(dyn: Dynamics, recs: list[dict], unit: float) -> np.ndarray:
    """Per-record NLL under the Lomax posterior-predictive, in the SAME model units
    (days/serials) as the neural model, so NLLs are directly comparable.
    S(t)=(b/(b+t))^a ; f(t)=a b^a/(b+t)^(a+1). We convert (a,b) to model units by
    scaling b (a scale param) by 1/unit."""
    nlls = []
    for r in recs:
        a, b = dyn._post.get(r["kclass"], (_A0, _B0))
        b = b / unit                                  # seconds-scale b -> model units
        t = r["life"]
        if r["censored"]:
            nll = -a * (math.log(b) - math.log(b + t))            # -log S
        else:
            nll = -(math.log(a) + a * math.log(b) - (a + 1) * math.log(b + t))  # -log f
        nlls.append(nll)
    return np.array(nlls)


def gamma_pit(dyn: Dynamics, recs: list[dict], unit: float) -> np.ndarray:
    """PIT values u=F(life)=1-S for OBSERVED lifetimes (calibration check)."""
    u = []
    for r in recs:
        if r["censored"]:
            continue
        a, b = dyn._post.get(r["kclass"], (_A0, _B0))
        b = b / unit
        u.append(1.0 - (b / (b + r["life"])) ** a)
    return np.array(u)


# ---------------------------------------------------------------------------
# baseline 2 — single global marginal exponential rate
# ---------------------------------------------------------------------------
def fit_marginal(train_recs: list[dict]) -> float:
    """MLE exponential rate over ALL train lifetimes (observed + censored exposure)."""
    n_obs = sum(1 for r in train_recs if not r["censored"])
    total = sum(r["life"] for r in train_recs)
    return n_obs / total if total > 0 else 1.0


def marginal_nll(rate: float, recs: list[dict]) -> np.ndarray:
    nlls = []
    for r in recs:
        t = r["life"]
        nlls.append(rate * t if r["censored"] else rate * t - math.log(rate))
    return np.array(nlls)


def marginal_pit(rate: float, recs: list[dict]) -> np.ndarray:
    return np.array([1.0 - math.exp(-rate * r["life"]) for r in recs if not r["censored"]])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def ks_uniform(u: np.ndarray) -> float:
    """KS distance of PIT values from Uniform(0,1). 0 = perfectly calibrated."""
    if len(u) == 0:
        return float("nan")
    u = np.sort(np.clip(u, 0, 1))
    n = len(u)
    cdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(cdf - u)))


def calibration_bins(u: np.ndarray, bins: int = 10) -> list[float]:
    if len(u) == 0:
        return []
    return list(np.histogram(np.clip(u, 0, 1), bins=bins, range=(0, 1))[0])


def next_key_marginal_acc(train_windows, test_windows) -> tuple[float, float]:
    """Next-key baseline: predict the next event's kclass (in time order within a
    window) as the globally most-frequent class. Returns (marginal_acc, same) over test
    transitions — the frequency baseline the neural next-key head must beat."""
    freq: dict[str, int] = defaultdict(int)
    for win in train_windows:
        for e in win:
            freq[e["kclass"]] += 1
    top = max(freq, key=freq.get) if freq else None

    def transitions(windows):
        pairs = []
        for win in windows:
            evs = sorted(win, key=lambda e: e["t"])
            for a, b in zip(evs, evs[1:]):
                pairs.append((a["kclass"], b["kclass"]))
        return pairs
    test_pairs = transitions(test_windows)
    if not test_pairs:
        return float("nan"), float("nan")
    acc = sum(1 for _, nxt in test_pairs if nxt == top) / len(test_pairs)
    return acc, acc


def summarize_nll(name: str, nll: np.ndarray) -> str:
    return (f"{name:>10}: NLL {nll.mean():.4f} ± {nll.std()/math.sqrt(len(nll)):.4f} "
            f"(n={len(nll)})")


# ---------------------------------------------------------------------------
# convenience: prepare a source end-to-end for baselines (Mac-runnable)
# ---------------------------------------------------------------------------
def prepare_source(path: str | Path, frac_test: float = 0.2, seed: int = 0,
                   max_len: int = 512):
    events = load_events(path)
    unit = time_unit(events)
    t_obs = max(e["t"] for e in events)
    windows = make_windows(events, max_len)
    train_c, test_c = split_windows(windows, frac_test, seed)
    train_r = lifetimes(train_c, t_obs, unit)
    test_r = lifetimes(test_c, t_obs, unit)
    return dict(events=events, unit=unit, t_obs=t_obs, train_c=train_c, test_c=test_c,
                train_r=train_r, test_r=test_r)


def run_baselines(path: str | Path, seed: int = 0) -> dict:
    """Fit + eval BOTH closed-form baselines on a source. Torch-free; runs on the Mac."""
    P = prepare_source(path, seed=seed)
    dyn = fit_gamma(P["train_c"], P["t_obs"])
    rate = fit_marginal(P["train_r"])
    obs = [r for r in P["test_r"] if not r["censored"]]
    g_nll = gamma_nll(dyn, P["test_r"], P["unit"])
    m_nll = marginal_nll(rate, P["test_r"])
    g_nll_o = gamma_nll(dyn, obs, P["unit"]) if obs else np.array([float("nan")])
    m_nll_o = marginal_nll(rate, obs) if obs else np.array([float("nan")])
    nk_acc, _ = next_key_marginal_acc(P["train_c"], P["test_c"])
    return {
        "n_train_life": len(P["train_r"]), "n_test_life": len(P["test_r"]),
        "cens_frac": float(np.mean([r["censored"] for r in P["test_r"]])) if P["test_r"] else float("nan"),
        "gamma_nll": float(g_nll.mean()), "gamma_nll_obs": float(g_nll_o.mean()),
        "gamma_ks": ks_uniform(gamma_pit(dyn, P["test_r"], P["unit"])),
        "marginal_nll": float(m_nll.mean()), "marginal_nll_obs": float(m_nll_o.mean()),
        "marginal_ks": ks_uniform(marginal_pit(rate, P["test_r"])),
        "marginal_rate": rate, "nextkey_marginal_acc": nk_acc,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mac-runnable closed-form baseline sanity check")
    ap.add_argument("--data", default="data/wm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"=== closed-form baselines (seed {args.seed}) ===")
    for src in ("synthetic", "mab", "lme"):
        p = Path(args.data) / f"{src}.jsonl"
        if not p.exists():
            continue
        r = run_baselines(p, seed=args.seed)
        print(f"\n[{src}]  train_life={r['n_train_life']} test_life={r['n_test_life']} "
              f"cens_frac={r['cens_frac']:.2f}")
        print(f"   Gamma    NLL(all)={r['gamma_nll']:.4f} NLL(obs)={r['gamma_nll_obs']:.4f}  KS={r['gamma_ks']:.3f}")
        print(f"   Marginal NLL(all)={r['marginal_nll']:.4f} NLL(obs)={r['marginal_nll_obs']:.4f}  KS={r['marginal_ks']:.3f}  rate={r['marginal_rate']:.4g}")
        print(f"   next-key marginal acc={r['nextkey_marginal_acc']:.3f}")
