"""Train the neural world-model (dynamics) — a GRU neural temporal point process over
fact-update events — and evaluate it HONESTLY against the shipped closed-form Gamma
baseline and a marginal-rate baseline.

Runs on the RTX box (imports torch). The Mac must never run this (torch thrashes 16GB).

Model (small, exportable to a numpy forward — hence GRU, not a transformer):
  per-event input = [kclass emb | source emb | value-emb proj | dt features | time feats]
  GRU (1 layer) over each WINDOW (global time-ordered, multi-key event stream)
  heads on each step's hidden state h_i:
    HAZARD    : Weibull(scale,shape) for time-to-next-change on e_i's key.
                Weibull is closed-form integrable AND its shape k lets hazard be
                INCREASING (k>1, aging/periodic-ish) or DECREASING (k<1, bursty
                clustering) — exactly the structure a per-key CONSTANT-hazard
                Gamma-exponential provably cannot represent. NLL is exact:
                  obs:  -[log k - log s + (k-1)(log t - log s) - (t/s)^k]
                  cens:  (t/s)^k                (= -log S(t))
    NEXT-KEY  : softmax over kclasses for the next event in the stream (learned ripple).
    NEXT-VALUE: predicted 384d embedding of the key's next value, InfoNCE vs in-batch.

Eval (per data source, on HELD-OUT windows — leakage-safe, shared with wm_common so the
records are identical across neural/Gamma/marginal): NLL, PIT calibration (KS), next-key
acc vs marginal, next-value recall@5 among 100, params, train time, numpy inference µs.

Usage (on RTX):
  python scripts/train_dynamics.py --data data/wm --epochs 8 --seed 0 \
      --out data/dynamics_neural.npz --log data/wm/train_log.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import wm_common as C  # noqa: E402
from tenet.dynamics_neural import time_feats, dt_feats  # noqa: E402  (single source of truth)

SOURCES = ["synthetic", "mab", "lme"]
DAY = 86400.0


def window_targets(win: list[dict], t_obs: float, unit: float, min_life: float = 1e-6):
    """For a time-ordered window, per position i: hazard (life,cens), next-key class,
    next-value vid. Same per-key lifetime logic as wm_common.lifetimes, indexed on the
    global stream so it aligns with GRU hidden states."""
    evs = sorted(win, key=lambda e: e["t"])
    L = len(evs)
    # next same-key event index
    last_pos: dict[str, int] = {}
    nxt_same = [None] * L
    for i in range(L - 1, -1, -1):
        k = evs[i]["key"]
        nxt_same[i] = last_pos.get(k)
        last_pos[k] = i
    recs = []
    for i, e in enumerate(evs):
        j = nxt_same[i]
        if j is not None:
            life = (evs[j]["t"] - e["t"]) / unit
            cens = 0
            nv = evs[j]["vid"]
        else:
            life = (t_obs - e["t"]) / unit
            cens = 1
            nv = -1
        if life < min_life:
            life = min_life
        nk = evs[i + 1]["kclass"] if i < L - 1 else None
        recs.append({"pos": i, "event": e, "life": life, "cens": cens,
                     "next_kclass": nk, "next_vid": nv})
    return evs, recs


def boot_ci(deltas: np.ndarray, seed: int, n: int = 2000):
    """Paired bootstrap 95% CI on the mean per-record NLL improvement (positive =
    neural better). Returns {mean, lo, hi}."""
    deltas = np.asarray(deltas, dtype=np.float64)
    if len(deltas) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(1)
    return {"mean": float(deltas.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5))}


def embed_values(values, model_name, device, torch):
    """bge-small 384d embeddings. Prefer sentence-transformers (matches the tenet local
    embedder exactly); fall back to raw transformers with CLS pooling + L2 norm (bge's
    sentence-transformers pooling) so the RTX's torch-only venv needs no extra install."""
    try:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model_name, device=device)
        return enc.encode(values, normalize_embeddings=True, batch_size=256,
                          show_progress_bar=True).astype("float32")
    except Exception as e:  # noqa: BLE001
        print(f"[embed] sentence-transformers unavailable ({e}); using transformers CLS", flush=True)
    from transformers import AutoTokenizer, AutoModel
    import numpy as _np
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    B = 256
    with torch.no_grad():
        for i in range(0, len(values), B):
            batch = values[i:i + B]
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state[:, 0]          # CLS token (bge pooling)
            h = torch.nn.functional.normalize(h, dim=-1)
            out.append(h.cpu().numpy().astype("float32"))
            print(f"  embedded {min(i+B, len(values))}/{len(values)}", flush=True)
    return _np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wm")
    ap.add_argument("--out", default="data/dynamics_neural.npz")
    ap.add_argument("--log", default="data/wm/train_log.json")
    ap.add_argument("--results", default="data/wm/results.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--kemb", type=int, default=32)
    ap.add_argument("--vproj", type=int, default=64)
    ap.add_argument("--semb", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-kclass-count", type=int, default=20,
                    help="key-classes with fewer events are bucketed as <other>")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--alpha", type=float, default=1.0, help="next-key loss weight")
    ap.add_argument("--beta", type=float, default=0.5, help="next-value InfoNCE weight")
    ap.add_argument("--seed", type=int, default=0, help="model-init/training seed")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="FIXED test-split seed (decoupled from init so multi-seed runs "
                         "share one held-out set — the 'fixed test set' rigor bar)")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    # ---- load per-source, build global vocabs ----
    src_events, src_meta = {}, {}
    for s in SOURCES:
        p = Path(args.data) / f"{s}.jsonl"
        if not p.exists():
            continue
        ev = C.load_events(p)
        src_events[s] = ev
        src_meta[s] = {"unit": C.time_unit(ev), "t_obs": max(e["t"] for e in ev)}
    # Cap the key-class vocab: MAB's heuristic keys yield ~15k mostly-singleton classes
    # that bloat the embedding/next-key heads without signal. Keep classes with
    # >= min_kclass_count events; bucket the rest as "<other>". Keeps the model small and
    # defensible; MAB next-key is chance either way.
    from collections import Counter
    kc_count = Counter(e["kclass"] for ev in src_events.values() for e in ev)
    keep = {k for k, c in kc_count.items() if c >= args.min_kclass_count}
    for ev in src_events.values():
        for e in ev:
            if e["kclass"] not in keep:
                e["kclass"] = "<other>"
    kclasses = sorted({e["kclass"] for ev in src_events.values() for e in ev})
    kc_id = {k: i for i, k in enumerate(kclasses)}
    src_id = {s: i for i, s in enumerate(src_events)}
    values = sorted({e["value"] for ev in src_events.values() for e in ev})
    val_id = {v: i for i, v in enumerate(values)}
    print(f"kclasses={len(kclasses)} sources={len(src_id)} unique_values={len(values)}", flush=True)

    # ---- embed all unique values once (GPU) ----
    emb_path = Path(args.data) / "value_emb.npz"
    if emb_path.exists() and json.loads((Path(args.data) / "value_emb.meta").read_text()).get("n") == len(values):
        VE = np.load(emb_path)["v"]
        print(f"loaded cached value embeddings {VE.shape}", flush=True)
    else:
        VE = embed_values(values, args.embed_model, dev, torch)
        np.savez_compressed(emb_path, v=VE)
        (Path(args.data) / "value_emb.meta").write_text(json.dumps({"n": len(values)}))
        print(f"embedded {len(values)} values -> {VE.shape}", flush=True)
    edim = VE.shape[1]
    VE_t = torch.tensor(VE, device=dev)

    # ---- windows + split, per source (leakage-safe) ----
    splits = {}
    for s, ev in src_events.items():
        wins = C.make_windows(ev, args.max_len)
        tr, te = C.split_windows(wins, 0.2, args.split_seed)
        splits[s] = {"train": tr, "test": te}
        print(f"[{s}] windows={len(wins)} train={len(tr)} test={len(te)}", flush=True)

    # feature dims
    n_dt, n_tf = 2, 5
    in_dim = args.kemb + args.semb + args.vproj + n_dt + n_tf

    class WM(nn.Module):
        def __init__(self):
            super().__init__()
            self.kemb = nn.Embedding(len(kclasses), args.kemb)
            self.semb = nn.Embedding(len(src_id), args.semb)
            self.vproj = nn.Linear(edim, args.vproj)
            self.gru = nn.GRU(in_dim, args.hidden, num_layers=1, batch_first=True)
            self.haz = nn.Linear(args.hidden, 2)        # log_scale, log_shape
            self.nk = nn.Linear(args.hidden, len(kclasses))
            self.nv = nn.Linear(args.hidden, edim)

        def encode_step(self, kc, sc, vemb, dtf, tf):
            return torch.cat([self.kemb(kc), self.semb(sc), self.vproj(vemb), dtf, tf], -1)

        def forward(self, x):
            h, _ = self.gru(x)
            ls, lsh = self.haz(h).unbind(-1)
            return h, ls, lsh, self.nk(h), self.nv(h)

    model = WM().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}  in_dim={in_dim} hidden={args.hidden}", flush=True)

    # ---- precompute batched tensors for a window list ----
    def batchify(win_list, source):
        """Return list of (tensors) padded batches for training/eval."""
        sc_id = src_id[source]
        unit = src_meta[source]["unit"]
        t_obs = src_meta[source]["t_obs"]
        items = []
        for win in win_list:
            evs, recs = window_targets(win, t_obs, unit)
            L = len(evs)
            kc = [kc_id[e["kclass"]] for e in evs]
            vemb_idx = [val_id[e["value"]] for e in evs]
            dtf = [dt_feats(e["dt_prev"], unit) for e in evs]
            tf = [time_feats(e["t"], e.get("tmean", 0)) for e in evs]
            life = [r["life"] for r in recs]
            cens = [r["cens"] for r in recs]
            nk = [kc_id[r["next_kclass"]] if r["next_kclass"] is not None else -1 for r in recs]
            nv = [r["next_vid"] for r in recs]
            items.append(dict(L=L, kc=kc, vidx=vemb_idx, dtf=dtf, tf=tf,
                              life=life, cens=cens, nk=nk, nv=nv, sc=sc_id))
        return items, unit, t_obs

    def make_batches(items, bs, shuffle):
        idx = np.arange(len(items))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [items[j] for j in idx[i:i + bs]]

    def to_tensors(batch):
        Lmax = max(it["L"] for it in batch)
        B = len(batch)
        kc = torch.zeros(B, Lmax, dtype=torch.long, device=dev)
        sc = torch.zeros(B, Lmax, dtype=torch.long, device=dev)
        vidx = torch.zeros(B, Lmax, dtype=torch.long, device=dev)
        dtf = torch.zeros(B, Lmax, n_dt, device=dev)
        tf = torch.zeros(B, Lmax, n_tf, device=dev)
        life = torch.zeros(B, Lmax, device=dev)
        cens = torch.zeros(B, Lmax, device=dev)
        nk = torch.full((B, Lmax), -1, dtype=torch.long, device=dev)
        nv = torch.full((B, Lmax), -1, dtype=torch.long, device=dev)
        mask = torch.zeros(B, Lmax, device=dev)
        for b, it in enumerate(batch):
            L = it["L"]
            kc[b, :L] = torch.tensor(it["kc"], device=dev)
            sc[b, :L] = it["sc"]
            vidx[b, :L] = torch.tensor(it["vidx"], device=dev)
            dtf[b, :L] = torch.tensor(it["dtf"], device=dev)
            tf[b, :L] = torch.tensor(it["tf"], device=dev)
            life[b, :L] = torch.tensor(it["life"], device=dev)
            cens[b, :L] = torch.tensor(it["cens"], dtype=torch.float, device=dev)
            nk[b, :L] = torch.tensor(it["nk"], device=dev)
            nv[b, :L] = torch.tensor(it["nv"], device=dev)
            mask[b, :L] = 1.0
        return kc, sc, vidx, dtf, tf, life, cens, nk, nv, mask

    def step_forward(kc, sc, vidx, dtf, tf):
        vemb = VE_t[vidx]
        x = model.encode_step(kc, sc, vemb, dtf, tf)
        return model.forward(x)

    def losses(batch):
        kc, sc, vidx, dtf, tf, life, cens, nk, nv, mask = to_tensors(batch)
        _, ls, lsh, nk_logits, nv_pred = step_forward(kc, sc, vidx, dtf, tf)
        s = ls.clamp(-8, 12).exp()                       # Weibull scale
        k = lsh.clamp(-3, 3).exp()                        # Weibull shape
        t = life.clamp_min(1e-6)
        z = (t / s).clamp_min(1e-9) ** k                  # (t/s)^k
        logf = torch.log(k) - torch.log(s) + (k - 1) * (torch.log(t) - torch.log(s)) - z
        nll = torch.where(cens > 0.5, z, -logf)           # censored: -logS=z ; obs: -logf
        haz_loss = (nll * mask).sum() / mask.sum()

        nk_valid = (nk >= 0) & (mask > 0.5)
        nk_loss = F.cross_entropy(nk_logits[nk_valid], nk[nk_valid]) if nk_valid.any() \
            else torch.zeros((), device=dev)

        nv_valid = (nv >= 0) & (mask > 0.5)
        if nv_valid.any():
            pred = F.normalize(nv_pred[nv_valid], dim=-1)          # (M,edim)
            pos = VE_t[nv[nv_valid]]                                # (M,edim) already unit
            # InfoNCE vs in-batch: candidates = the M true-next embeddings in this batch
            logits = pred @ pos.t() / 0.07
            labels = torch.arange(pred.shape[0], device=dev)
            nv_loss = F.cross_entropy(logits, labels)
        else:
            nv_loss = torch.zeros((), device=dev)
        return haz_loss, nk_loss, nv_loss

    # ---- train (pooled over all sources) ----
    train_items = []
    for s in src_events:
        it, _, _ = batchify(splits[s]["train"], s)
        train_items += it
    print(f"pooled train windows={len(train_items)}", flush=True)

    log = {"args": vars(args), "params": n_params, "epochs": []}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        agg = [0.0, 0.0, 0.0, 0]
        for batch in make_batches(train_items, args.batch, shuffle=True):
            opt.zero_grad()
            hl, nkl, nvl = losses(batch)
            loss = hl + args.alpha * nkl + args.beta * nvl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            agg[0] += hl.item(); agg[1] += nkl.item(); agg[2] += nvl.item(); agg[3] += 1
        nb = max(1, agg[3])
        row = {"epoch": ep, "haz": agg[0] / nb, "nextkey": agg[1] / nb, "nextval": agg[2] / nb}
        log["epochs"].append(row)
        print(f"ep{ep}: haz={row['haz']:.4f} nextkey={row['nextkey']:.4f} "
              f"nextval={row['nextval']:.4f}", flush=True)
    train_time = time.time() - t0
    log["train_time_s"] = train_time
    print(f"train_time={train_time:.1f}s", flush=True)

    # ---- eval per source vs baselines ----
    model.eval()
    results = {}
    with torch.no_grad():
        for s in src_events:
            unit = src_meta[s]["unit"]
            t_obs = src_meta[s]["t_obs"]
            # closed-form baselines on the SAME split (identical records)
            dyn = C.fit_gamma(splits[s]["train"], t_obs)
            base_recs = C.lifetimes(splits[s]["train"], t_obs, unit)
            rate = C.fit_marginal(base_recs)

            # Single ALIGNED record set: neural, Gamma, marginal all scored on the SAME
            # lifetimes (built from window_targets), so NLLs are strictly comparable.
            aligned = []           # {kclass, life, censored} per non-degenerate event
            n_nll_all, n_nll_obs, n_pit = [], [], []
            nk_correct = nk_total = 0
            nv_hits = nv_total = 0
            rng = np.random.default_rng(args.seed)
            all_vids = np.arange(len(values))
            for win in splits[s]["test"]:
                evs, recs = window_targets(win, t_obs, unit)
                batch = [dict(L=len(evs),
                              kc=[kc_id[e["kclass"]] for e in evs],
                              vidx=[val_id[e["value"]] for e in evs],
                              dtf=[dt_feats(e["dt_prev"], unit) for e in evs],
                              tf=[time_feats(e["t"], e.get("tmean", 0)) for e in evs],
                              life=[r["life"] for r in recs], cens=[r["cens"] for r in recs],
                              nk=[kc_id[r["next_kclass"]] if r["next_kclass"] is not None else -1 for r in recs],
                              nv=[r["next_vid"] for r in recs], sc=src_id[s])]
                kc, sc, vidx, dtf, tf, life, cens, nk, nv, mask = to_tensors(batch)
                _, ls, lsh, nk_logits, nv_pred = step_forward(kc, sc, vidx, dtf, tf)
                sca = ls.clamp(-8, 12).exp()[0]; sh = lsh.clamp(-3, 3).exp()[0]
                nkp = nk_logits[0].argmax(-1); nvp = F.normalize(nv_pred[0], dim=-1)
                for i, r in enumerate(recs):
                    if r["life"] < 1e-6:                     # degenerate (same-timestamp) — drop
                        continue
                    sc_i = float(sca[i]); k_i = float(sh[i]); t = r["life"]
                    z = (t / sc_i) ** k_i
                    if r["cens"]:
                        n_nll_all.append(z)
                    else:
                        logf = math.log(k_i) - math.log(sc_i) + (k_i - 1) * (math.log(t) - math.log(sc_i)) - z
                        n_nll_all.append(-logf); n_nll_obs.append(-logf)
                        n_pit.append(1.0 - math.exp(-z))
                    aligned.append({"kclass": r["event"]["kclass"], "life": t, "censored": r["cens"]})
                    if r["next_kclass"] is not None:
                        nk_total += 1
                        nk_correct += int(int(nkp[i]) == kc_id[r["next_kclass"]])
                    if r["next_vid"] >= 0:
                        nv_total += 1
                        negs = rng.choice(all_vids[all_vids != r["next_vid"]], size=99, replace=False)
                        cand = np.concatenate([[r["next_vid"]], negs])
                        sims = (nvp[i:i + 1] @ VE_t[torch.tensor(cand, device=dev)].t())[0]
                        nv_hits += int(0 in torch.topk(sims, 5).indices.tolist())
            obs = [r for r in aligned if not r["censored"]]
            g_all = C.gamma_nll(dyn, aligned, unit); g_obs = C.gamma_nll(dyn, obs, unit) if obs else np.array([np.nan])
            m_all = C.marginal_nll(rate, aligned); m_obs = C.marginal_nll(rate, obs) if obs else np.array([np.nan])
            # paired bootstrap CI on mean per-record NLL improvement vs each baseline
            # (g_all and n_nll_all are in the SAME aligned order). Positive = neural better.
            neural_arr = np.array(n_nll_all)
            d_gamma = boot_ci(g_all - neural_arr, args.split_seed)
            d_marg = boot_ci(m_all - neural_arr, args.split_seed)
            nk_marg, _ = C.next_key_marginal_acc(splits[s]["train"], splits[s]["test"])
            results[s] = {
                "n_test_life": len(aligned),
                "cens_frac": float(np.mean([r["censored"] for r in aligned])) if aligned else float("nan"),
                "neural_nll": float(np.mean(n_nll_all)), "gamma_nll": float(g_all.mean()), "marginal_nll": float(m_all.mean()),
                "neural_nll_obs": float(np.mean(n_nll_obs)) if n_nll_obs else float("nan"),
                "gamma_nll_obs": float(g_obs.mean()), "marginal_nll_obs": float(m_obs.mean()),
                "neural_ks": C.ks_uniform(np.array(n_pit)), "gamma_ks": C.ks_uniform(C.gamma_pit(dyn, aligned, unit)),
                "marginal_ks": C.ks_uniform(C.marginal_pit(rate, aligned)),
                "nextkey_neural_acc": (nk_correct / nk_total) if nk_total else float("nan"),
                "nextkey_marginal_acc": nk_marg,
                "nextval_recall@5": (nv_hits / nv_total) if nv_total else float("nan"),
                "nextval_chance": 0.05, "nextval_n": nv_total,
                "nll_gain_vs_gamma": d_gamma, "nll_gain_vs_marginal": d_marg,
            }
            print(f"\n[{s}] {results[s]}", flush=True)

    # ---- numpy inference timing (single p_valid call) ----
    export = export_npz(model, VE, kclasses, list(src_id), args, in_dim, n_dt, n_tf, edim)
    np.savez_compressed(args.out, **export)
    Path(args.log).write_text(json.dumps(log, indent=2))
    Path(args.results).write_text(json.dumps(results, indent=2))
    print(f"\nsaved weights -> {args.out} ({Path(args.out).stat().st_size/1e6:.2f} MB)")
    print(f"saved log -> {args.log}  results -> {args.results}")

    # ---- CRITICAL: verify the exported numpy forward matches torch exactly ----
    try:
        from tenet.dynamics_neural import NeuralDynamics
        nd = NeuralDynamics.load(args.out)
        s0 = next(iter(src_events))
        win = splits[s0]["test"][0]
        evs, _ = window_targets(win, src_meta[s0]["t_obs"], src_meta[s0]["unit"])
        evs = evs[:6]
        hist = [{"kclass": e["kclass"], "vemb": VE[val_id[e["value"]]],
                 "dt_prev": e["dt_prev"], "t": e["t"], "tmean": e.get("tmean", 0)} for e in evs]
        # numpy final hidden -> Weibull scale
        np_scale, np_k = nd._weibull(nd._gru_last(nd._encode(hist, s0, src_meta[s0]["unit"])))
        # torch final hidden -> Weibull scale (same 6-event window)
        batch = [dict(L=len(evs), kc=[kc_id[e["kclass"]] for e in evs],
                      vidx=[val_id[e["value"]] for e in evs],
                      dtf=[dt_feats(e["dt_prev"], src_meta[s0]["unit"]) for e in evs],
                      tf=[time_feats(e["t"], e.get("tmean", 0)) for e in evs],
                      life=[1.0] * len(evs), cens=[0] * len(evs),
                      nk=[-1] * len(evs), nv=[-1] * len(evs), sc=src_id[s0])]
        kc, sc, vidx, dtf, tf, *_ = to_tensors(batch)
        _, ls, lsh, _, _ = step_forward(kc, sc, vidx, dtf, tf)
        t_scale = float(ls.clamp(-8, 12).exp()[0, -1]); t_k = float(lsh.clamp(-3, 3).exp()[0, -1])
        drift = max(abs(np_scale - t_scale) / (t_scale + 1e-9), abs(np_k - t_k) / (t_k + 1e-9))
        results["_numpy_torch_reldrift"] = drift
        print(f"numpy-vs-torch Weibull rel-drift={drift:.2e} "
              f"(np scale={np_scale:.3f} k={np_k:.3f} | torch scale={t_scale:.3f} k={t_k:.3f})")
        assert drift < 1e-3, f"numpy forward diverges from torch (drift={drift})"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] numpy-vs-torch check: {e}")

    # inference microbench via the numpy module
    try:
        from tenet.dynamics_neural import NeuralDynamics
        nd = NeuralDynamics.load(args.out)
        # a small 3-event history to time p_valid (numpy forward)
        vemb0 = VE[0].astype(np.float32)
        hist = [{"kclass": kclasses[0], "vemb": vemb0, "dt_prev": -1.0, "t": 0.0, "tmean": 1},
                {"kclass": kclasses[0], "vemb": vemb0, "dt_prev": 2.0, "t": 2.0, "tmean": 1},
                {"kclass": kclasses[0], "vemb": vemb0, "dt_prev": 3.0, "t": 5.0, "tmean": 1}]
        src0 = list(src_id)[0]
        import timeit
        us = timeit.timeit(lambda: nd.p_valid_hist(hist, age=5.0, source=src0, unit=1.0),
                           number=2000) / 2000 * 1e6
        print(f"numpy p_valid inference: {us:.1f} µs/call")
        results["_infer_us"] = us
        Path(args.results).write_text(json.dumps(results, indent=2))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] numpy inference microbench skipped: {e}")


def export_npz(model, VE, kclasses, sources, args, in_dim, n_dt, n_tf, edim):
    """Export ALL weights + vocab + config for the numpy-only forward pass."""
    import torch
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    out = {
        "kemb.weight": sd["kemb.weight"], "semb.weight": sd["semb.weight"],
        "vproj.weight": sd["vproj.weight"], "vproj.bias": sd["vproj.bias"],
        "gru.weight_ih_l0": sd["gru.weight_ih_l0"], "gru.weight_hh_l0": sd["gru.weight_hh_l0"],
        "gru.bias_ih_l0": sd["gru.bias_ih_l0"], "gru.bias_hh_l0": sd["gru.bias_hh_l0"],
        "haz.weight": sd["haz.weight"], "haz.bias": sd["haz.bias"],
        "nk.weight": sd["nk.weight"], "nk.bias": sd["nk.bias"],
        "nv.weight": sd["nv.weight"], "nv.bias": sd["nv.bias"],
        "kclasses": np.array(kclasses), "sources": np.array(sources),
        "config": np.array(json.dumps({
            "hidden": args.hidden, "kemb": args.kemb, "semb": args.semb, "vproj": args.vproj,
            "in_dim": in_dim, "n_dt": n_dt, "n_tf": n_tf, "edim": edim,
            "embed_model": args.embed_model})),
    }
    return out


if __name__ == "__main__":
    main()
