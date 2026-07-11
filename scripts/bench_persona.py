"""PersonaMem-v2 (Jiang et al., arXiv:2512.06688; bowen-upenn/PersonaMem-v2, CC-BY-4.0) —
the NapMem/Qwen-team-adjacent LLM-personalization benchmark whose question regime fits
Tenet's mechanism. Each item: a long (32k-token) implicit-persona chat history + a user
message + a 4-way MULTIPLE-CHOICE set of candidate assistant responses (1 correct + 3
distractors). The model must pick the response that matches the user's CURRENT preferences.

Why this benchmark (vs LoCoMo §12, which we lost): a large slice (`updated=True`, ~21%) are
preference *updates / retractions* — the user changed or asked to forget a preference, and a
distractor typically encodes the STALE value. Honoring the latest state is exactly what
Tenet's ingestion-time keyed supersession does. Hypothesis: Tenet wins its regime
(`updated=True`); we report the rest (`updated=False`, plain recall) honestly — where a raw
RAG baseline may match or win, as on LoCoMo.

SCORING IS DETERMINISTIC (official format is multiple-choice) — we pick the option letter and
exact-match the correct one. NO LLM judge, so NO judge-comparability caveat (unlike §12). The
absolute numbers are still not a vendor leaderboard entry (different reader model, retrieval-
compressed context vs their full 32k), but the within-harness arm comparison is clean.

Arms (matched backbone, identical reader prompt + top-k):
  rag    — top-k cosine over raw dated turns                          [baseline]
  tenet  — Tenet.ingest_session per chunk (keyed supersession), k=10  [ours]

Mirrors scripts/bench_locomo.py: scratch caches (repo data/ symlink is TCC-blocked), seeded
sampling, API failures excluded (never scored wrong), miss dumps, Wilson CIs + paired McNemar.

Usage:
  python scripts/bench_persona.py --csv <scratch>/benchmark.csv --cache <scratch>/persona_cache \
      --personas 20 --seed 0 --out <scratch>/personamem_run.json
  python scripts/bench_persona.py ... --smoke        # 1 persona
"""
from __future__ import annotations

import argparse, ast, json, os, random, re, sys, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
from tenet import Tenet  # noqa: E402
from bench_factcon import wilson_ci  # noqa: E402

HF_BASE = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main/"
ANSWER_MODEL = config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus")
CHUNK = 8  # turns per ingest_session chunk (keeps same-preference conflicts out of one chunk)
LETTERS = "ABCD"

_MC_SYS = ("You are a personalized AI assistant. Using ONLY the memory about this user below, "
           "choose the ONE response option that best matches the user's CURRENT preferences and "
           "situation. If the user updated or asked to forget a preference, honor the most "
           "RECENT state (ignore the outdated one). Reply with ONLY the letter of the best "
           "option — a single character A, B, C, or D, nothing else.")


def mc_pick(ctx: str, query: str, options: list[str]) -> tuple[str, int]:
    """Return (raw_reply, parsed_index or -1). Deterministic scoring — no judge."""
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    out = config.chat(
        [{"role": "system", "content": _MC_SYS},
         {"role": "user", "content": f"User memory:\n{ctx}\n\nUser's message: {query}\n\n"
          f"Response options:\n{body}\n\nBest option letter:"}],
        qwen_default=ANSWER_MODEL, max_tokens=4)
    m = re.search(r"[A-D]", out.upper())
    return out, (LETTERS.index(m.group(0)) if m else -1)


# --------------------------------------------------------------------------
# Data: chat-history download (cached) + flattening.
# --------------------------------------------------------------------------
def fetch_history(link: str, cache_dir: Path) -> list[dict]:
    fn = cache_dir / Path(link).name
    if not fn.exists():
        urllib.request.urlretrieve(HF_BASE + link, fn)
    d = json.load(open(fn))
    return d["chat_history"] if isinstance(d, dict) else d


def flatten_history(chat: list[dict]):
    """-> turn_rows [(idx, 'Role: content')] for user/assistant turns (system persona card
    dropped — the benchmark's signal is the IMPLICITLY-revealed prefs in conversation).
    Chronological order = supersession order (later turns are more recent)."""
    rows = []
    for t in chat:
        if t.get("role") in ("user", "assistant") and str(t.get("content", "")).strip():
            rows.append((len(rows), f"{t['role'].capitalize()}: {t['content'].strip()}"))
    return rows


def build_store(chat: list[dict], cache_dir: Path, cid: str, embedder, retract: bool = False):
    """Ingest one persona's history into Tenet via ingest_session per chunk (keyed
    supersession, monotonic clock so later chunks are more recent). Cache store + turn
    embeddings. Returns (Tenet, turn_vecs, turn_rows).

    `retract` — docs/COMPARISON.md follow-up #3: route distiller-tagged
    action="retract" facts (explicit "forget X") to core.retract() instead of
    core.store(). Changes INGESTION, so a cache built with retract=False cannot be
    reused for a retract=True run (or vice versa) — the cache_dir must differ."""
    turn_rows = flatten_history(chat)
    texts = [t for _, t in turn_rows]
    npz, dbp = cache_dir / f"{cid}.npz", cache_dir / f"{cid}.db"
    turn_vecs = np.load(npz)["v"] if npz.exists() else np.array(embedder(texts))
    if not npz.exists():
        np.savez_compressed(npz, v=turn_vecs)
    if dbp.exists():
        return Tenet(dbp), turn_vecs, turn_rows

    clock = [1_000_000.0]
    m = Tenet(dbp, now=lambda: clock[0])
    umsgs = [t for t in chat if t.get("role") in ("user", "assistant")
             and str(t.get("content", "")).strip()]
    for i in range(0, len(umsgs), CHUNK):
        turns = [{"role": t["role"], "content": t["content"].strip()} for t in umsgs[i:i + CHUNK]]
        m.ingest_session(turns, source=f"c{i//CHUNK}", valid_at=clock[0], retract=retract)
        clock[0] += 3600
    clock[0] += 3600  # advance now just past ingestion for recall-time decay
    st = m.stats()
    stored = st["current"] + st["superseded"] + st["archived"]
    if len(texts) and stored < 0.3 * len(texts):
        raise RuntimeError(f"persona ingest degraded: {stored} stored from {len(texts)} turns "
                           f"({cid}) — check distiller/embedder before trusting answers")
    return m, turn_vecs, turn_rows


# --------------------------------------------------------------------------
# Sampling: pick N personas (seeded), keep all their queries.
# --------------------------------------------------------------------------
def load_rows(csv_path: Path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        try:
            q = ast.literal_eval(r["user_query"])
            q = q["content"] if isinstance(q, dict) else str(q)
        except Exception:
            q = str(r["user_query"])
        try:
            incorrect = list(ast.literal_eval(r["incorrect_answers"]))
        except Exception:
            continue
        try:
            sp = ast.literal_eval(r["short_persona"])
            profile = sp.get("persona", "") if isinstance(sp, dict) else str(sp)
        except Exception:
            profile = str(r.get("short_persona", ""))
        rows.append({"pid": int(r["persona_id"]), "link": r["chat_history_32k_link"],
                     "query": q, "correct": str(r["correct_answer"]), "incorrect": incorrect,
                     "updated": bool(r["updated"]), "pref_type": str(r["pref_type"]),
                     "who": str(r["who"]), "profile": profile})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--personas", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--expand", type=int, default=20)
    ap.add_argument("--arms", default="tenet,rag")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dump", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tenet-retract", action="store_true",
                    help="route distiller-tagged action=retract facts to core.retract() "
                         "at ingestion (docs/COMPARISON.md follow-up #3); default off. "
                         "Changes ingestion — use a fresh --cache dir vs a non-retract run.")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    assert config.LLM_PROVIDER == "qwen", "run against the shipped qwen backbone"

    cache_dir = Path(args.cache); cache_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = cache_dir / "histories"; hist_dir.mkdir(exist_ok=True)
    rows = load_rows(Path(args.csv))
    pids = sorted({r["pid"] for r in rows})
    rng = random.Random(args.seed); rng.shuffle(pids)
    keep = set(pids[: (1 if args.smoke else args.personas)])
    rows = [r for r in rows if r["pid"] in keep]
    if args.smoke:
        rows = rows[:12]
    t0 = time.time()

    host = Tenet(cache_dir / "emb_host.db")
    embedder = host.core.embed_batch
    print(f"selected {len(rows)} queries over {len(keep)} personas "
          f"(updated=True: {sum(r['updated'] for r in rows)}) — ingesting ...", flush=True)

    stores: dict[int, tuple] = {}
    sup_fracs = []
    for pid in sorted({r["pid"] for r in rows}):
        link = next(r["link"] for r in rows if r["pid"] == pid)
        chat = fetch_history(link, hist_dir)
        stores[pid] = build_store(chat, cache_dir, f"p{pid}", embedder, retract=args.tenet_retract)
        st = stores[pid][0].stats()
        sf = st["superseded"] / max(1, st["current"] + st["superseded"])
        sup_fracs.append(sf)
        print(f"  ingested persona {pid} ({len(stores[pid][2])} turns, "
              f"superseded={st['superseded']}/{st['current']+st['superseded']})", flush=True)
    # Diagnostic for the supersession thesis: if ~0, ingestion-time keyed supersession is
    # NOT catching the natural-language preference updates, so tenet has no structural edge
    # on the updated=True regime (report this honestly whatever the accuracy shows).
    print(f"mean superseded fraction across personas: {np.mean(sup_fracs):.3f}", flush=True)

    # Precompute per-question contexts (recall) + shuffled options, then run MC readers.
    recs = []
    for r in rows:
        m, turn_vecs, turn_rows = stores[r["pid"]]
        qv = np.asarray(embedder([r["query"]])[0])
        opts = [r["correct"]] + r["incorrect"]
        order = list(range(len(opts)))
        random.Random(hash((r["pid"], r["query"])) & 0xFFFFFFFF).shuffle(order)
        shuffled = [opts[i] for i in order]
        correct_idx = order.index(0)
        # Static profile one-liner prepended to BOTH arms (fair; the vendor full-context
        # setup keeps the persona card in view — this restores the coarse static profile
        # that top-k retrieval otherwise drops, without giving either arm an edge).
        prof = f"User profile: {r['profile']}\n\n" if r.get("profile") else ""
        ctx = {}
        if "blind" in arms:
            # no-memory control: profile line only, ZERO retrieved turns. If this scores
            # near the memory arms, the MC is answerable from option-plausibility + the
            # static profile and memory is NOT load-bearing (a benchmark-validity check).
            ctx["blind"] = prof.strip() or "(no memory available)"
        if "rag" in arms:
            top = np.argsort(-(turn_vecs @ qv))[: args.k]
            ctx["rag"] = prof + "\n".join(turn_rows[i][1] for i in sorted(top))
        if "tenet" in arms:
            hits = m.core.recall(r["query"], k=args.k, expand=args.expand)
            ctx["tenet"] = prof + "\n".join(h.text for h in hits)
        recs.append({**r, "opts": shuffled, "correct_idx": correct_idx, "ctx": ctx,
                     "pick": {}, "ok": {}, "err": {}})

    jobs = [(i, a) for i in range(len(recs)) for a in arms]

    def _run(job):
        i, a = job
        r = recs[i]
        try:
            raw, idx = mc_pick(r["ctx"][a], r["query"], r["opts"])
        except config.ProviderError:
            raise
        except Exception:  # noqa: BLE001
            return i, a, "", -1
        return i, a, raw, idx

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, a, raw, idx in ex.map(_run, jobs):
            r = recs[i]
            if idx < 0:
                r["err"][a] = True
            else:
                r["pick"][a] = idx
                r["ok"][a] = (idx == r["correct_idx"])
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} MC reader calls", flush=True)
    for _pid, (m, _tv, _tr) in stores.items():
        m.close()
    host.close()

    # ---- aggregate: overall + by regime (updated) + by pref_type ----
    def bucket(pred):
        out = {a: [0, 0] for a in arms}
        for r in recs:
            if not pred(r):
                continue
            for a in arms:
                if a in r["err"] or a not in r["ok"]:
                    continue
                out[a][1] += 1; out[a][0] += int(r["ok"][a])
        return out

    def fmt(ok, n):
        p = ok / n if n else 0.0
        lo, hi = wilson_ci(p, n)
        return f"{100*p:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] n={n:>3}"

    n_excl = sum(1 for r in recs for a in arms if a in r["err"])
    print(f"\n=== PersonaMem-v2 (4-way MC, deterministic; random=25%) — reader {ANSWER_MODEL}, "
          f"k={args.k} ===")
    print(f"excluded (API fail): {n_excl} arm-questions   wall={time.time()-t0:.0f}s\n")

    segments = [("OVERALL", lambda r: True),
                ("updated=True  (SUPERSESSION regime)", lambda r: r["updated"]),
                ("updated=False (plain recall)", lambda r: not r["updated"])]
    hdr = "  ".join(f"{a.upper():>26}" for a in arms)
    print(f"{'segment':>36} | {hdr}")
    seg_tables = {}
    for name, pred in segments:
        t = bucket(pred); seg_tables[name] = t
        cells = "  ".join(f"{fmt(*t[a]):>26}" for a in arms)
        print(f"{name:>36} | {cells}")
    print("\n--- by pref_type ---")
    for pt in sorted({r["pref_type"] for r in recs}):
        t = bucket(lambda r, pt=pt: r["pref_type"] == pt)
        cells = "  ".join(f"{fmt(*t[a]):>26}" for a in arms)
        print(f"{pt:>36} | {cells}")

    # ---- paired McNemar (tenet vs rag), overall + updated=True ----
    def mcnemar(pred):
        b = c = 0
        for r in recs:
            if not pred(r) or "tenet" not in r["ok"] or "rag" not in r["ok"]:
                continue
            t, g = r["ok"]["tenet"], r["ok"]["rag"]
            if t and not g:
                b += 1
            elif g and not t:
                c += 1
        n = b + c
        from math import comb
        kmin = min(b, c)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(kmin + 1)) / 2**n) if n else 1.0
        return b, c, p
    if set(arms) >= {"tenet", "rag"}:
        print("\n--- paired McNemar (tenet vs rag) ---")
        for name, pred in (("overall", lambda r: True),
                           ("updated=True", lambda r: r["updated"]),
                           ("updated=False", lambda r: not r["updated"])):
            b, c, p = mcnemar(pred)
            adv = "tenet" if b > c else ("rag" if c > b else "tie")
            print(f"  {name:>14}: tenet-adv={b} rag-adv={c}  p={p:.4f}  -> {adv}")

    if args.dump:
        with open(args.dump, "w") as f:
            for r in recs:
                if r["err"] or any(a in r["ok"] and not r["ok"][a] for a in arms):
                    f.write(json.dumps({k: r[k] for k in
                            ("pid", "query", "updated", "pref_type", "who", "correct_idx",
                             "pick", "ok", "err", "opts")}, default=str) + "\n")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": {"personas": len(keep), "queries": len(recs), "seed": args.seed,
                       "k": args.k, "expand": args.expand, "arms": arms, "reader": ANSWER_MODEL},
            "segments": {name: {a: seg_tables[name][a] for a in arms} for name, _ in segments},
            "excluded": n_excl,
        }, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
