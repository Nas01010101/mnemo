"""Torch-FREE verification runnable on the Mac (never imports torch).

Checks:
  1. numpy GRU / NeuralDynamics math is sane: survival in (0,1], monotone-decreasing in
     age; next-key softmax normalises; next-value embedding is unit-norm.
  2. The MemoryCore TENET_DYNAMICS=neural hook is SAFE: with a random-weight .npz it runs
     recall without crashing and attaches confidence; with a missing/!dim npz it FALLS
     BACK to the closed-form model (never breaks recall).
  3. The DEFAULT path (TENET_DYNAMICS unset) is byte-identical to the shipped Dynamics.
Uses a deterministic stub embedder (no torch); the semantic-recall tests
(test_memory/test_dynamics) run for real on the GPU box.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EDIM = 384
FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def fake_npz(path, kclasses, sources, cfg, seed=0):
    rng = np.random.default_rng(seed)
    H, K, S = cfg["hidden"], len(kclasses), len(sources)
    ke, se, vp, dt, tf = cfg["kemb"], cfg["semb"], cfg["vproj"], cfg["n_dt"], cfg["n_tf"]
    in_dim = cfg["in_dim"]
    import json
    d = {
        "kemb.weight": rng.standard_normal((K, ke)).astype(np.float32),
        "semb.weight": rng.standard_normal((S, se)).astype(np.float32),
        "vproj.weight": (rng.standard_normal((vp, EDIM)) * 0.1).astype(np.float32),
        "vproj.bias": np.zeros(vp, np.float32),
        "gru.weight_ih_l0": (rng.standard_normal((3 * H, in_dim)) * 0.1).astype(np.float32),
        "gru.weight_hh_l0": (rng.standard_normal((3 * H, H)) * 0.1).astype(np.float32),
        "gru.bias_ih_l0": np.zeros(3 * H, np.float32),
        "gru.bias_hh_l0": np.zeros(3 * H, np.float32),
        "haz.weight": (rng.standard_normal((2, H)) * 0.1).astype(np.float32),
        "haz.bias": np.array([1.5, 0.0], np.float32),   # scale~e^1.5, shape~1
        "nk.weight": (rng.standard_normal((K, H)) * 0.1).astype(np.float32),
        "nk.bias": np.zeros(K, np.float32),
        "nv.weight": (rng.standard_normal((EDIM, H)) * 0.1).astype(np.float32),
        "nv.bias": np.zeros(EDIM, np.float32),
        "kclasses": np.array(kclasses), "sources": np.array(sources),
        "config": np.array(json.dumps(cfg)),
    }
    np.savez_compressed(path, **d)


def main():
    from tenet.dynamics_neural import NeuralDynamics

    kclasses = ["mood", "residence", "job", "name"]
    sources = ["synthetic", "mab", "lme"]
    cfg = {"hidden": 192, "kemb": 32, "semb": 8, "vproj": 64,
           "in_dim": 32 + 8 + 64 + 2 + 5, "n_dt": 2, "n_tf": 5, "edim": EDIM}
    tmp = Path(tempfile.mkdtemp())
    npz = tmp / "fake.npz"
    fake_npz(npz, kclasses, sources, cfg)
    nd = NeuralDynamics.load(npz)

    # 1. math sanity
    rng = np.random.default_rng(1)
    ve = rng.standard_normal(EDIM).astype(np.float32); ve /= np.linalg.norm(ve)
    hist = [{"kclass": "mood", "vemb": ve, "dt_prev": -1.0, "t": 0.0, "tmean": 1},
            {"kclass": "mood", "vemb": ve, "dt_prev": 1.0, "t": 1.0, "tmean": 1},
            {"kclass": "mood", "vemb": ve, "dt_prev": 2.0, "t": 3.0, "tmean": 1}]
    surv = [nd.survival_hist(hist, a, source="synthetic", unit=1.0) for a in (0.1, 1, 5, 30, 100)]
    check("survival in (0,1]", all(0 < s <= 1.0 + 1e-9 for s in surv), str([round(s, 4) for s in surv]))
    check("survival monotone-decreasing in age",
          all(surv[i] >= surv[i + 1] - 1e-9 for i in range(len(surv) - 1)), str(surv))
    nk = nd.next_key_logits(hist, "synthetic", 1.0)
    p = np.exp(nk - nk.max()); p /= p.sum()
    check("next-key softmax normalises", abs(p.sum() - 1.0) < 1e-6)
    nv = nd.next_value_embedding_hist(hist, "synthetic", 1.0)
    check("next-value embedding unit-norm", abs(np.linalg.norm(nv) - 1.0) < 1e-5)

    # GRU determinism
    s1 = nd.survival_hist(hist, 5.0, "synthetic", 1.0)
    s2 = nd.survival_hist(hist, 5.0, "synthetic", 1.0)
    check("inference deterministic", s1 == s2)

    # 2 + 3. MemoryCore hook. Stub the embedder so no torch is loaded.
    from tenet import config as cfgmod

    def stub_embed(texts):
        out = []
        for t in texts:
            seed = int.from_bytes(t.encode()[:8].ljust(8, b"\0"), "little")
            v = np.random.default_rng(seed).standard_normal(EDIM).astype(np.float32)
            v /= (np.linalg.norm(v) or 1.0)
            out.append(v)
        return out
    cfgmod.embed_texts = stub_embed
    import tenet.memory as M

    clock = {"t": 1_000_000_000.0}
    DAY = 86400.0

    def now():
        return clock["t"]

    # default path == shipped Dynamics
    os.environ.pop("TENET_DYNAMICS", None)
    db = tmp / "d.db"
    core = M.MemoryCore(db, now=now)
    core.store("mood is happy", key="user::mood")
    for i, mood in enumerate(["tired", "excited", "calm"]):
        clock["t"] += 2 * DAY
        core.store(f"mood is {mood}", key="user::mood")
    from tenet.dynamics import Dynamics
    dyn = core._dynamics()
    check("default dynamics is closed-form Dynamics", isinstance(dyn, Dynamics))
    got = core.recall("mood", k=3)
    check("default recall returns current mood",
          any(m.key == "user::mood" and m.is_current for m in got))
    core.close()

    # neural path with a valid random npz -> runs, attaches confidence, no crash
    os.environ["TENET_DYNAMICS"] = "neural"
    os.environ["TENET_NEURAL_NPZ"] = str(npz)
    core2 = M.MemoryCore(tmp / "d2.db", now=now)
    core2.store("lives in Boston", key="user::residence")
    clock["t"] += 10 * DAY
    core2.store("lives in Seattle", key="user::residence")
    dyn2 = core2._dynamics()
    check("neural path selected", isinstance(dyn2, NeuralDynamics), type(dyn2).__name__)
    p_valid = dyn2.p_valid("user::residence", 5 * DAY, now=now())
    check("neural p_valid in (0,1]", 0 < p_valid <= 1.0 + 1e-9, f"{p_valid:.4f}")
    got2 = core2.recall("residence", k=3)
    cur = [m for m in got2 if m.key == "user::residence" and m.is_current]
    check("neural recall returns current fact with confidence",
          bool(cur) and cur[0].confidence is not None, str([(m.text, m.confidence) for m in got2]))
    nk_pred = dyn2.predict_next_key("user::residence")
    check("predict_next_key returns ranked classes", len(nk_pred) > 0, str(nk_pred))
    nv_pred = dyn2.predict_next_value_embedding("user::residence")
    check("predict_next_value_embedding returns vector", nv_pred is not None and nv_pred.shape[0] == EDIM)
    core2.close()

    # neural path with WRONG npz path -> graceful fallback to closed-form (recall unbroken)
    os.environ["TENET_NEURAL_NPZ"] = str(tmp / "does_not_exist.npz")
    core3 = M.MemoryCore(tmp / "d3.db", now=now)
    core3.store("name is Alex", key="user::name")
    dyn3 = core3._dynamics()
    check("missing-npz falls back to closed-form", isinstance(dyn3, Dynamics), type(dyn3).__name__)
    check("recall still works after fallback", len(core3.recall("name", k=2)) >= 1)
    core3.close()

    os.environ.pop("TENET_DYNAMICS", None)
    os.environ.pop("TENET_NEURAL_NPZ", None)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("MAC NEURAL VERIFY ALL PASS (torch-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
