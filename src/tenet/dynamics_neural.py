"""Neural fact-dynamics — NUMPY-ONLY inference for the GRU temporal-point-process world
model trained by scripts/train_dynamics.py. No torch: the tenet package stays a pure
sqlite+numpy dependency, so a trained drift model ships as one .npz.

Selected at runtime via env TENET_DYNAMICS=neural (default stays the closed-form
Dynamics in dynamics.py — this module changes NO default behaviour). It mirrors the
Dynamics interface used by MemoryCore:
    p_valid(skey, age_s, now) -> P(fact still current) via Weibull survival
    expected_lifetime_days(skey)
plus the drift-model extras:
    predict_next_key(...)             -> ranked next key-classes (learned ripple)
    predict_next_value_embedding(skey)-> predicted 384d embedding of the next value

The Weibull head gives survival S(t)=exp(-(t/scale)^k) with a per-event (scale,shape)
read off the GRU hidden state — so hazard can be increasing (k>1) or decreasing (k<1),
unlike the closed-form model's per-key CONSTANT hazard.

This module owns the feature functions (time_feats/dt_feats) so the trainer and the
inference path are guaranteed identical.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DAY = 86400.0
N_DT, N_TF = 2, 5


def key_class(skey: str | None) -> str | None:
    if not skey:
        return None
    return skey.rsplit("::", 1)[-1].strip().lower() or None


# --- feature functions (SINGLE source of truth; trainer imports these) ---------
def time_feats(t: float, tmean: int) -> list[float]:
    if not tmean:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    hour = (t / 3600.0) % 24.0
    dow = ((t / DAY) + 4.0) % 7.0
    return [math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7), 1.0]


def dt_feats(dt_prev: float, unit: float) -> list[float]:
    if dt_prev < 0:
        return [0.0, 1.0]
    return [math.log1p(max(0.0, dt_prev) / unit), 0.0]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_from_ledger(rows, now: float, npz_path: str | None = None):
    """MemoryCore hook: build a NeuralDynamics bound to the ledger from sqlite rows with
    (skey, valid_at, embedding). Returns None on ANY failure (missing weights, dim
    mismatch, load error) so recall silently falls back to the closed-form model — the
    drift model must never break the read path. Weights path: TENET_NEURAL_NPZ env, else
    <repo>/data/dynamics_neural.npz."""
    import os
    import sys
    path = npz_path or os.environ.get(
        "TENET_NEURAL_NPZ",
        str(Path(__file__).resolve().parent.parent.parent / "data" / "dynamics_neural.npz"))
    try:
        model = NeuralDynamics.load(path)
        edim = model.cfg["edim"]
        recs = []
        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape[0] != edim:
                raise ValueError(f"embedding dim {v.shape[0]} != model edim {edim} "
                                 "(neural dynamics needs the bge-small local embedder)")
            recs.append({"skey": r["skey"], "valid_at": r["valid_at"], "vemb": v})
        return NeuralDynamics.fit_store(model, recs, now=now)
    except Exception as e:  # noqa: BLE001
        print(f"[tenet] neural dynamics unavailable ({e}); using closed-form", file=sys.stderr)
        return None


class NeuralDynamics:
    """Loaded from an exported .npz. Stateless model; history is supplied per query."""

    def __init__(self, W: dict, config: dict, kclasses: list[str], sources: list[str]):
        self.W = W
        self.cfg = config
        self.kclasses = list(kclasses)
        self.kc_id = {k: i for i, k in enumerate(self.kclasses)}
        self.sources = list(sources)
        self.src_id = {s: i for i, s in enumerate(self.sources)}
        self.H = config["hidden"]
        # cache per-(skey) history built via fit(), for the MemoryCore interface
        self._hist: dict[str, list[dict]] = {}
        self._unit = DAY
        self._source = self.sources[0] if self.sources else "synthetic"
        self._now = 0.0

    # ---- load ----
    @classmethod
    def load(cls, path: str | Path) -> "NeuralDynamics":
        # Opt-in path (TENET_DYNAMICS=neural) loading a LOCAL, user-supplied trained
        # artifact; the npz stores object arrays (kclasses/sources/config) so pickle is
        # required. Never loads from the network.
        d = np.load(path, allow_pickle=True)  # nosemgrep: trailofbits.python.pickles-in-numpy.pickles-in-numpy
        cfg = json.loads(str(d["config"]))
        W = {k: d[k] for k in d.files if k not in ("kclasses", "sources", "config")}
        return cls(W, cfg, list(d["kclasses"]), list(d["sources"]))

    # ---- numpy GRU (1 layer, PyTorch batch_first equations) ----
    def _gru_last(self, X: np.ndarray) -> np.ndarray:
        """X: (L, in_dim) -> final hidden (H,). Chunks ordered [r, z, n] per PyTorch."""
        Wih, Whh = self.W["gru.weight_ih_l0"], self.W["gru.weight_hh_l0"]
        bih, bhh = self.W["gru.bias_ih_l0"], self.W["gru.bias_hh_l0"]
        H = self.H
        h = np.zeros(H, dtype=np.float32)
        for t in range(X.shape[0]):
            gi = Wih @ X[t] + bih
            gh = Whh @ h + bhh
            r = _sigmoid(gi[:H] + gh[:H])
            z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h = (1.0 - z) * n + z * h
        return h

    def _encode(self, hist: list[dict], source: str, unit: float) -> np.ndarray:
        """hist: list of {kclass, vemb(np edim), dt_prev, t, tmean}. -> (L, in_dim)."""
        kemb, semb = self.W["kemb.weight"], self.W["semb.weight"]
        Wv, bv = self.W["vproj.weight"], self.W["vproj.bias"]
        sid = self.src_id.get(source, 0)
        rows = []
        for e in hist:
            ke = kemb[self.kc_id.get(e["kclass"], 0)]
            se = semb[sid]
            vp = Wv @ e["vemb"].astype(np.float32) + bv
            df = np.array(dt_feats(e["dt_prev"], unit), dtype=np.float32)
            tf = np.array(time_feats(e["t"], e.get("tmean", 1)), dtype=np.float32)
            rows.append(np.concatenate([ke, se, vp, df, tf]))
        return np.stack(rows).astype(np.float32)

    def _weibull(self, h: np.ndarray) -> tuple[float, float]:
        o = self.W["haz.weight"] @ h + self.W["haz.bias"]
        scale = math.exp(float(np.clip(o[0], -8, 12)))
        k = math.exp(float(np.clip(o[1], -3, 3)))
        return scale, k

    # ---- low-level drift-model queries ----
    def survival_hist(self, hist: list[dict], age: float, source: str | None = None,
                      unit: float | None = None) -> float:
        """S(age) for a key whose event history is `hist` (age in the SAME unit)."""
        if not hist:
            return 1.0
        h = self._gru_last(self._encode(hist, source or self._source, unit or self._unit))
        scale, k = self._weibull(h)
        return float(math.exp(-((max(age, 0.0) / scale) ** k)))

    def p_valid_hist(self, hist: list[dict], age: float, source: str | None = None,
                     unit: float | None = None) -> float:
        return self.survival_hist(hist, age, source, unit)

    def next_key_logits(self, hist: list[dict], source: str | None = None,
                        unit: float | None = None) -> np.ndarray:
        h = self._gru_last(self._encode(hist, source or self._source, unit or self._unit))
        return self.W["nk.weight"] @ h + self.W["nk.bias"]

    def next_value_embedding_hist(self, hist: list[dict], source: str | None = None,
                                  unit: float | None = None) -> np.ndarray:
        h = self._gru_last(self._encode(hist, source or self._source, unit or self._unit))
        v = self.W["nv.weight"] @ h + self.W["nv.bias"]
        n = np.linalg.norm(v)
        return v / n if n else v

    # ---- MemoryCore-compatible interface (mirrors dynamics.Dynamics) ----
    @classmethod
    def fit_store(cls, model: "NeuralDynamics", rows, now: float,
                  source: str = "synthetic", unit: float = DAY) -> "NeuralDynamics":
        """Bind the model to a ledger snapshot. rows: dicts with skey, valid_at, and
        'vemb' (the stored value embedding). Builds per-skey time-sorted histories with
        dt_prev so p_valid(skey, age, now) works with the Dynamics signature."""
        model._now = now
        model._source = source
        model._unit = unit
        by_key: dict[str, list[dict]] = {}
        for r in rows:
            by_key.setdefault(r["skey"], []).append(r)
        hist: dict[str, list[dict]] = {}
        for skey, evs in by_key.items():
            evs.sort(key=lambda r: r["valid_at"])
            seq, last = [], None
            kc = key_class(skey)
            for r in evs:
                dt = (r["valid_at"] - last) if last is not None else -1.0
                seq.append({"kclass": kc, "vemb": r["vemb"], "dt_prev": dt,
                            "t": r["valid_at"], "tmean": 1})
                last = r["valid_at"]
            hist[skey] = seq
        model._hist = hist
        return model

    def p_valid(self, skey: str | None, age_s: float, now: float | None = None) -> float:
        """Drop-in for Dynamics.p_valid: age_s in SECONDS (tenet clock). Converts to the
        model's unit (days). Falls back to 1.0 if the key has no bound history."""
        hist = self._hist.get(skey)
        if not hist:
            return 1.0
        return self.survival_hist(hist, age_s / self._unit, self._source, self._unit)

    def expected_lifetime_days(self, skey: str | None) -> float | None:
        hist = self._hist.get(skey)
        if not hist:
            return None
        h = self._gru_last(self._encode(hist, self._source, self._unit))
        scale, k = self._weibull(h)                      # scale in days already
        return float(scale * math.gamma(1.0 + 1.0 / k))  # Weibull mean

    def predict_next_key(self, skey: str | None = None, top: int = 3) -> list[tuple[str, float]]:
        """Ranked next key-classes given a key's history (or the most-recently-updated
        key if skey is None). Softmax over the learned next-key head (the ripple)."""
        if skey is None:
            if not self._hist:
                return []
            skey = max(self._hist, key=lambda k: self._hist[k][-1]["t"])
        hist = self._hist.get(skey)
        if not hist:
            return []
        logits = self.next_key_logits(hist, self._source, self._unit)
        p = np.exp(logits - logits.max()); p /= p.sum()
        idx = np.argsort(-p)[:top]
        return [(self.kclasses[i], float(p[i])) for i in idx]

    def predict_next_value_embedding(self, skey: str | None) -> np.ndarray | None:
        hist = self._hist.get(skey)
        if not hist:
            return None
        return self.next_value_embedding_hist(hist, self._source, self._unit)
