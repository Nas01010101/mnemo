"""LoCoMo-10 (Maharana et al., ACL 2024) — the field's most-marketed long-conversation
memory benchmark (Mem0 reports 92.5 here; also used by Zep, MemMachine, EverMemOS, and the
Qwen team's NapMem, arXiv:2607.05794). We ran it WITH the community answer-key audit
(github.com/dial481/locomo-audit) as a feature: every score is reported BOTH against the
original key and the audit-corrected key — a data point no vendor publishes.

Mirrors scripts/lme_recall.py's multi-session ingest pattern: per conversation, distill each
session into Tenet (dual-pool: distilled facts + dated raw turns), cache the ingested store,
then answer each question over that conversation's store.

Arms (matched backbone, identical reader + judge prompt):
  rag    — top-k cosine over raw dated turns                      [baseline]
  tenet  — dual-pool recall (distilled facts + raw slices), k=10  [ours]
(mem0/hipporag are NOT run here: mem0's per-turn LLM ADD/UPDATE ingestion is thousands of
calls over LoCoMo's long histories — not the "cheap generalization" the plan gated on. Add
later behind a flag if wanted.)

Scoring: LLM-judge accuracy per category (1=multi-hop, 2=temporal, 3=open-domain,
4=single-hop; category 5/adversarial skipped, as all published LoCoMo results do), Wilson
CIs, overall. Reported for BOTH answer keys.

JUDGE CAVEAT (load-bearing): published LoCoMo numbers use a gpt-4o-mini judge; we judge with
qwen3.7-plus and CANNOT replicate their judge. Every number is therefore labeled
"qwen-judged; not directly comparable to vendor gpt-4o-mini-judged numbers." The WITHIN-
harness arm comparison (tenet vs rag, same judge) IS valid.

Usage:
  python scripts/bench_locomo.py --data <scratch>/locomo10.json --audit <scratch>/audit_errors.json \
      --cache <scratch>/locomo_cache --sample 500 --seed 0 --out <scratch>/locomo_run.json
  python scripts/bench_locomo.py ... --smoke        # 1 conversation, ~12 questions
"""
from __future__ import annotations

import argparse, json, random, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
from tenet import Tenet  # noqa: E402
from tenet.distill import distill  # noqa: E402
from bench_factcon import wilson_ci  # noqa: E402 — reuse the exact Wilson helper

CAT_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
ANSWER_MODEL = config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus")

# --------------------------------------------------------------------------
# Reader + judge (standard LoCoMo QA shape; single prompt for both arms so the
# only variable is retrieval quality).
# --------------------------------------------------------------------------
_ANS_SYS = ("You answer questions about a recorded conversation between two people, using "
            "ONLY the provided memory/context. Answer in a SHORT phrase (a few words). Give "
            "dates exactly as they appear in the conversation; give counts as a number. If "
            "the memory does not contain the answer, reply 'No information available'.")
_JUDGE_SYS = ("You compare a model's answer to the gold answer for a question about a "
              "conversation. Reply 'yes' if the model's answer is semantically correct (same "
              "meaning as the gold answer), else 'no'. Be lenient about phrasing, extra words, "
              "and equivalent date formats; the key fact must match. Reply exactly 'yes' or 'no'.")


def _chat(system: str, user: str, max_tokens: int) -> str:
    return config.chat([{"role": "system", "content": system},
                        {"role": "user", "content": user}],
                       qwen_default=ANSWER_MODEL, max_tokens=max_tokens)


def qa_answer(ctx: str, question: str) -> str:
    return _chat(_ANS_SYS, f"Memory:\n{ctx}\n\nQuestion: {question}\nShort answer:", 64)


def qa_judge(question: str, gold: str, pred: str) -> bool | None:
    out = _chat(_JUDGE_SYS, f"Question: {question}\nGold answer: {gold}\n"
                f"Model answer: {pred}\nIs the model answer correct?", 4).lower()
    return out.startswith("y") if out else None


# --------------------------------------------------------------------------
# Conversation flattening + ingestion (cache the fully-ingested store per conv).
# --------------------------------------------------------------------------
def _turn_text(t: dict) -> str:
    txt = t.get("text", "").strip()
    cap = t.get("blip_caption") or t.get("caption")
    return f"{txt} [shared image: {cap}]" if cap else txt


def session_order(conv: dict) -> list[str]:
    keys = [k for k in conv if re.fullmatch(r"session_\d+", k)]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def flatten(conv: dict):
    """-> (sess_pairs, turn_rows). sess_pairs=[(skey, date, session_text)] for distillation;
    turn_rows=[(dia_id, '[date] Speaker: text')] for RAG + raw storage. Dated turns are
    load-bearing — LoCoMo temporal answers are dates that live in the turn text."""
    sess_pairs, turn_rows = [], []
    for skey in session_order(conv):
        date = conv.get(f"{skey}_date_time", "")
        lines = []
        for t in conv[skey]:
            txt = _turn_text(t)
            if not txt:
                continue
            lines.append(f"{t['speaker']}: {txt}")
            turn_rows.append((t.get("dia_id", skey), f"[{date}] {t['speaker']}: {txt}"))
        body = "\n".join(lines)
        sess_pairs.append((skey, date, f"Session date: {date}\n{body}" if date else body))
    return sess_pairs, turn_rows


def build_conv_store(conv: dict, cache_dir: Path, cid: str, embedder):
    """Ingest one conversation into Tenet (distill facts + store dated raw turns). Cached:
    the store + turn embeddings persist across runs. Returns (Tenet, turn_vecs, turn_rows)."""
    sess_pairs, turn_rows = flatten(conv)
    texts = [txt for _, txt in turn_rows]
    npz = cache_dir / f"{cid}.npz"
    dbp = cache_dir / f"{cid}.db"
    if npz.exists():
        turn_vecs = np.load(npz)["v"]
    else:
        turn_vecs = np.array(embedder(texts))
        np.savez_compressed(npz, v=turn_vecs)
    if dbp.exists():
        return Tenet(dbp), turn_vecs, turn_rows

    m = Tenet(dbp)

    def _distill(pair):
        skey, _date, stext = pair
        try:
            return skey, distill(stext)
        except Exception:  # noqa: BLE001 — a dead distill call must not abort the conv
            return skey, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        distilled = list(ex.map(_distill, sess_pairs))
    facts = [(skey, f) for skey, fs in distilled for f in fs]
    if facts:
        fvecs = m.core.embed_batch([f.statement for _, f in facts])
        for (skey, f), fv in zip(facts, fvecs):
            m.core.store(f.statement, key=f.key, salience=f.salience, source=skey, _vec=fv)
    for (dia_id, text), tv in zip(turn_rows, turn_vecs):
        m.core.store(text, kind="raw", salience=0.35, source=dia_id, _vec=tv)
    # Integrity: a healthy conversation ingest must retain most turns (fail loud on a dead
    # distiller/embedder rather than silently answering "No information available").
    st = m.stats()
    stored = st["current"] + st["superseded"] + st["archived"]
    if len(texts) and stored < 0.5 * len(texts):
        raise RuntimeError(f"locomo ingest degraded: {stored} stored from {len(texts)} turns "
                           f"({cid}) — check distiller/embedder before trusting answers")
    return m, turn_vecs, turn_rows


# --------------------------------------------------------------------------
# Audit corrections: map dial481/locomo-audit errors.json onto questions.
# --------------------------------------------------------------------------
def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def load_audit(path: Path) -> dict[tuple[int, str], str]:
    """-> {(conv_idx, normalized_question): corrected_golden_answer}. conv_idx parsed from
    question_id 'locomo_<i>_qa<j>'; matched within-conv by exact question text (robust to any
    qa-index off-by-one)."""
    out = {}
    for e in json.load(open(path)):
        m = re.match(r"locomo_(\d+)_qa\d+", str(e.get("question_id", "")))
        if not m:
            continue
        out[(int(m.group(1)), _norm_q(e["question"]))] = str(e["golden_answer"])
    return out


# --------------------------------------------------------------------------
# Question selection: stratified-by-category subsample (seeded).
# --------------------------------------------------------------------------
def select_questions(data, sample: int, seed: int):
    """Gather all non-adversarial (cat 1-4) questions as (conv_idx, qa), then take a
    category-stratified subsample proportional to availability."""
    by_cat = defaultdict(list)
    for ci, conv in enumerate(data):
        for qa in conv["qa"]:
            c = qa.get("category")
            if c in CAT_NAME and "answer" in qa:
                by_cat[c].append((ci, qa))
    total = sum(len(v) for v in by_cat.values())
    rng = random.Random(seed)
    picked = []
    if sample <= 0 or sample >= total:
        for c in by_cat:
            picked += by_cat[c]
    else:
        for c, items in by_cat.items():
            rng.shuffle(items)
            n_c = max(1, round(sample * len(items) / total))
            picked += items[:n_c]
    rng.shuffle(picked)
    return picked, total


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--audit", default="")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--sample", type=int, default=500, help="stratified subsample size (<=0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--expand", type=int, default=10, help="tenet belief-anchored raw expansion")
    ap.add_argument("--arms", default="tenet,rag")
    ap.add_argument("--workers", type=int, default=8, help="reader/judge concurrency")
    ap.add_argument("--dump", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tenet-agg", action="store_true",
                    help="opt into recall()'s CAR-style read-time recency aggregation "
                         "(docs/COMPARISON.md follow-up #1) on the tenet arm; default off")
    args = ap.parse_args()
    if args.smoke:
        args.sample = 0  # all questions of the single conv we keep below
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    assert config.LLM_PROVIDER == "qwen", "run against the shipped qwen backbone"

    data = json.load(open(args.data))
    if args.smoke:
        data = data[:1]
    audit = load_audit(Path(args.audit)) if args.audit else {}
    cache_dir = Path(args.cache); cache_dir.mkdir(parents=True, exist_ok=True)
    dump_f = open(args.dump, "w") if args.dump else None
    t0 = time.time()

    host = Tenet(cache_dir / "emb_host.db")
    embedder = host.core.embed_batch

    picked, total = select_questions(data, 0 if args.smoke else args.sample, args.seed)
    if args.smoke:
        picked = picked[:12]
    print(f"selected {len(picked)} questions (of {total} non-adversarial) — ingesting "
          f"{len(set(ci for ci,_ in picked))} conversations ...", flush=True)

    # Ingest each needed conversation once (cached), collect per-conv recall material.
    stores: dict[int, tuple] = {}
    for ci in sorted(set(ci for ci, _ in picked)):
        cid = f"conv{ci}"
        # sessions live under the 'conversation' sub-dict; qa is top-level (see select_questions)
        stores[ci] = build_conv_store(data[ci]["conversation"], cache_dir, cid, embedder)
        print(f"  ingested conv {ci}", flush=True)

    # Build per-question reader jobs (tenet + rag contexts precomputed via recall).
    jobs = []           # (idx, arm)
    recs = []           # per-question record
    audit_hits = 0
    for ci, qa in picked:
        m, turn_vecs, turn_rows = stores[ci]
        q = qa["question"]; gold = str(qa["answer"]); cat = qa["category"]
        corr = audit.get((ci, _norm_q(q)))
        if corr is not None:
            audit_hits += 1
        qv = np.asarray(embedder([q])[0])
        ctx = {}
        if "rag" in arms:
            top = np.argsort(-(turn_vecs @ qv))[: args.k]
            ctx["rag"] = "\n".join(turn_rows[i][1] for i in sorted(top))
        if "tenet" in arms:
            hits = m.core.recall(q, k=args.k, expand=args.expand, agg_reader=args.tenet_agg)
            ctx["tenet"] = "\n".join(h.text for h in hits)
        idx = len(recs)
        recs.append({"ci": ci, "q": q, "cat": cat, "gold": gold,
                     "gold_corr": corr if corr is not None else gold,
                     "changed": corr is not None and _norm_q(corr) != _norm_q(gold),
                     "ctx": ctx, "pred": {}, "ok": {}, "ok_corr": {}, "err": {}})
        for a in arms:
            jobs.append((idx, a))
    print(f"audit corrections matched: {audit_hits}/{len(audit)} "
          f"(answer-changing: {sum(1 for r in recs if r['changed'])})", flush=True)

    # Reader + judge, concurrent. Judge against original key; re-judge only questions whose
    # corrected answer actually differs.
    def _run(job):
        idx, arm = job
        r = recs[idx]
        pred = qa_answer(r["ctx"][arm], r["q"])
        if not pred.strip():
            return idx, arm, pred, None, None
        j = qa_judge(r["q"], r["gold"], pred)
        jc = j
        if j is not None and r["changed"]:
            jc = qa_judge(r["q"], r["gold_corr"], pred)
        return idx, arm, pred, j, jc

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, arm, pred, j, jc in ex.map(_run, jobs):
            r = recs[idx]
            r["pred"][arm] = pred
            if j is None:
                r["err"][arm] = True
            else:
                r["ok"][arm] = j
                r["ok_corr"][arm] = jc if jc is not None else j
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} reader+judge calls done", flush=True)

    for _ci, (m, _tv, _tr) in stores.items():
        m.close()
    host.close()

    # ---- aggregate: per category x arm x key ----
    def agg(key_field: str):
        # returns {arm: {cat: (ok, n), 'all': (ok, n)}}
        out = {a: defaultdict(lambda: [0, 0]) for a in arms}
        for r in recs:
            for a in arms:
                if a in r["err"] or a not in r["ok"]:
                    continue
                v = r[key_field][a]
                out[a][r["cat"]][1] += 1; out[a][r["cat"]][0] += int(v)
                out[a]["all"][1] += 1; out[a]["all"][0] += int(v)
        return out

    orig = agg("ok"); corr = agg("ok_corr")
    n_excl = sum(1 for r in recs for a in arms if a in r["err"])

    def _fmt(ok, n):
        p = ok / n if n else 0.0
        lo, hi = wilson_ci(p, n)
        return f"{100*p:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] n={n:>3}"

    print(f"\n=== LoCoMo-10 QA accuracy (qwen-judged; NOT comparable to vendor "
          f"gpt-4o-mini-judged numbers) — reader {ANSWER_MODEL}, k={args.k} ===")
    print(f"excluded (API fail): {n_excl} arm-questions   wall={time.time()-t0:.0f}s\n")
    for label, table in (("ORIGINAL KEY", orig), ("AUDIT-CORRECTED KEY", corr)):
        print(f"--- {label} ---")
        cats = sorted(CAT_NAME) + ["all"]
        hdr = "  ".join(f"{a.upper():>26}" for a in arms)
        print(f"{'category':>14} | {hdr}")
        for c in cats:
            name = "OVERALL" if c == "all" else f"{c}:{CAT_NAME[c]}"
            cells = "  ".join(f"{_fmt(*table[a][c]):>26}" for a in arms)
            print(f"{name:>14} | {cells}")
        print()
    # delta original -> corrected (overall, per arm)
    print("--- key-correction delta (corrected - original, OVERALL) ---")
    for a in arms:
        o_ok, o_n = orig[a]["all"]; c_ok, c_n = corr[a]["all"]
        op = 100 * o_ok / o_n if o_n else 0; cp = 100 * c_ok / c_n if c_n else 0
        print(f"  {a:>8}: {op:.1f}% -> {cp:.1f}%  (delta {cp-op:+.1f}pp)")

    if dump_f:
        for r in recs:
            if any(a in r["ok"] and not r["ok"][a] for a in arms) or r["err"]:
                dump_f.write(json.dumps({k: r[k] for k in
                             ("ci", "q", "cat", "gold", "gold_corr", "changed", "pred",
                              "ok", "err")}) + "\n")
        dump_f.close()
    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": {"sample": len(picked), "total_nonadv": total, "seed": args.seed,
                       "k": args.k, "expand": args.expand, "arms": arms,
                       "reader": ANSWER_MODEL, "audit_matched": audit_hits,
                       "audit_answer_changing": sum(1 for r in recs if r["changed"])},
            "original": {a: {str(c): orig[a][c] for c in list(CAT_NAME) + ["all"]} for a in arms},
            "corrected": {a: {str(c): corr[a][c] for c in list(CAT_NAME) + ["all"]} for a in arms},
            "excluded": n_excl,
        }, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
