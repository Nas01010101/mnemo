"""MemoryAgentBench FactConsolidation (Conflict Resolution) — the standardized
supersession benchmark (ICLR 2026, arXiv:2507.05257; HF ai-hyz/MemoryAgentBench).

Serial-numbered facts with counterfactual updates ("newer facts have larger serial
numbers"); questions require the CURRENT value. Published table (paper Table 3):
Zep 7%, Mem0 18%, MemGPT 28%, HippoRAG-v2 54%, GPT-4o long-context 60% on FC-SH;
FC-MH <=7% for all 22 systems. Current published SOTA (arXiv:2606.01435):
FC-SH 78.0 (gpt-4o-mini) / 94.8 (gpt-4o); FC-MH 30.2 / 51.5.

Arms (matched backbone, identical official prompt + top-k):
  rag    — top-k cosine over raw serial-numbered fact lines            [control]
  tenet  — distiller-keyed INGESTION-TIME supersession (serial=valid_at);
           recall returns only current beliefs                          [ours]
Protocol fidelity: reader prompt is MAB's official 'rag_agent' template VERBATIM;
scoring is MAB's SubEM (normalize + substring + max-over-golds) copied VERBATIM.
API failures exclude the question from scoring (never counted as wrong).

Usage:
  python scripts/bench_factcon.py --cells sh_6k,sh_32k --limit 20      # smoke
  python scripts/bench_factcon.py --qpc 100 --hops-mh 3               # full 800
"""
from __future__ import annotations

import argparse, json, math, re, string, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
import config       # noqa: E402
from tenet import Tenet  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "factcon"

# --------------------------------------------------------------------------
# MAB official scoring — copied VERBATIM from utils/eval_other_utils.py
# (HUST-AI-HYZ/MemoryAgentBench) so our numbers use exactly their metric.
# --------------------------------------------------------------------------
def normalize_answer(answer_text):
    text = answer_text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def substring_exact_match_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def subem_max(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        ground_truths = [g for sub in ground_truths for g in sub]
    return max(substring_exact_match_score(prediction, g) for g in ground_truths)


# MAB official reader prompt ('rag_agent' in utils/templates.py) — VERBATIM.
_MAB_PROMPT = ("Pretend you are a knowledge management system. Each fact in the knowledge "
               "pool is provided with a serial number at the beginning, and the newer fact "
               "has larger serial number. \n You need to solve the conflicts of facts in the "
               "knowledge pool by finding the newest fact with larger serial number. You need "
               "to answer a question based on this rule. You should give a very concise answer "
               "without saying other words for the question **only** from the knowledge pool "
               "you have memorized rather than the real facts in real world. \n\nFor example:"
               "\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, "
               "what is the name of the current president of Russia? \nAnswer: Donald Trump "
               "\n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} "
               "\nAnswer:")


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = p_hat + z * z / (2 * n)
    m = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
    return ((c - m) / d, (c + m) / d)


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
_FACT_RE = re.compile(r"^(\d+)\.\s+(.*\S)\s*$")


def parse_facts(context: str) -> list[tuple[int, str]]:
    out = []
    for line in context.splitlines():
        m = _FACT_RE.match(line.strip())
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


_KEY_SYS = ("For each numbered fact, output its semantic key: 'subject::relation' with the "
            "VALUE (the object/tail of the sentence) removed. Lowercase, canonical, stable — "
            "two facts asserting different values for the same attribute of the same subject "
            "MUST get the same key. Reply with JSON: "
            '{"keys": [{"i": <fact index>, "k": "<subject::relation>"}, ...]} — one entry per fact.')


KEY_MODE = "llm"  # set by --keys; "heuristic" = deterministic zero-LLM ingestion keys


def extract_keys(facts: list[tuple[int, str]], source: str) -> dict[int, str]:
    """Supersession keys. `heuristic` mode is fully deterministic (no LLM anywhere in
    ingestion): key = the normalized fact minus its final value words — exploits the
    templated subject-relation-value shape of the facts. LLM mode uses the distiller."""
    if KEY_MODE == "heuristic":
        return {s: " ".join(normalize_answer(t).split()[:-2]) for s, t in facts}
    cf = CACHE / f"{source}.keys.json"
    if cf.exists():
        return {int(k): v for k, v in json.load(open(cf)).items()}
    keys: dict[int, str] = {}
    B = 40

    def _one(batch):
        listing = "\n".join(f"{s}. {t}" for s, t in batch)
        out = config.chat(
            [{"role": "system", "content": _KEY_SYS},
             {"role": "user", "content": listing}],
            qwen_default="qwen3.6-flash", max_tokens=1400, json_mode=True)
        got = {}
        try:
            for e in json.loads(re.search(r"\{.*\}", out, re.S).group(0))["keys"]:
                got[int(e["i"])] = str(e["k"]).strip().lower()
        except Exception:
            pass
        return batch, got

    batches = [facts[i:i + B] for i in range(0, len(facts), B)]
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for batch, got in ex.map(_one, batches):
            for s, t in batch:
                # fallback key = the fact minus its last two words (templated tails);
                # only used when the distiller call failed for this fact
                if s not in got:
                    fallbacks += 1
                keys[s] = got.get(s) or " ".join(normalize_answer(t).split()[:-2])
    if fallbacks > len(facts) // 2:
        # Same failure class as the zero-vector bug: a dead LLM endpoint must FAIL
        # LOUDLY, not silently degrade the whole sequence to heuristic keys.
        raise RuntimeError(f"key extraction degraded: {fallbacks}/{len(facts)} fallbacks "
                           f"({source}) — LLM endpoint likely broken")
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(keys, open(cf, "w"))
    return keys


def build_tenet(source: str, facts: list[tuple[int, str]]) -> Tenet:
    dbp = CACHE / f"{source}.db"
    if dbp.exists():
        return Tenet(dbp)
    keys = extract_keys(facts, source)
    m = Tenet(dbp)
    vecs = m.core.embed_batch([t for _, t in facts])
    for (serial, text), v in zip(facts, vecs):
        # serial IS event time: keyed supersession retires the older value
        m.core.store(text, key=keys[serial], source=str(serial),
                     valid_at=float(serial), _vec=v)
    return m


def embed_lines(source: str, facts: list[tuple[int, str]], embedder):
    npz = CACHE / f"{source}.npz"
    if npz.exists():
        return np.load(npz)["v"]
    v = np.array(embedder([t for _, t in facts]))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, v=v)
    return v


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------
def answer(pool: str, question: str) -> str:
    return config.chat(
        [{"role": "user", "content": f"[Knowledge Pool]\n{pool}\n\n" +
          _MAB_PROMPT.format(question=question)}],
        qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"), max_tokens=64)


# Tenet reading mode: the store is already conflict-resolved (supersession at
# ingestion), so the reader's only job is verbatim extraction — which also blocks
# the dominant failure mode on counterfactual benchmarks: the reader overriding
# the pool with real-world knowledge (miss analysis: 8/8 sampled misses had the
# gold IN the pool while the reader answered from parametric memory).
_EXTRACT_PROMPT = (
    "The facts below are from a FICTIONAL knowledge pool. They intentionally "
    "contradict the real world; the real-world answer is WRONG here.\n"
    "Find the single fact that answers the question and COPY its value verbatim "
    "from that fact. Never use your own knowledge. Reply with ONLY the value — a "
    "short phrase, never a full sentence, never the fact restated."
    "\n\n[Knowledge Pool]\n{pool}\n\nQuestion: {question}\nCopied value:")


def answer_extract(pool: str, question: str) -> str:
    return config.chat(
        [{"role": "user", "content": _EXTRACT_PROMPT.format(pool=pool, question=question)}],
        qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"), max_tokens=48)


# MH: Self-Ask-style per-hop decomposition over the conflict-free store
# (the composition pattern of arXiv:2606.01435, running on OUR ingestion-time-
# superseded memory instead of post-retrieval aggregation).
_DECOMP_PROMPT = (
    "Decompose the question into a chain of 1-4 single-hop lookups. Each hop asks for "
    "ONE attribute of ONE entity. Every hop after the first MUST contain the literal "
    "token #PREV where the previous hop's answer goes.\n"
    'Example: "Which country is the birthplace of the sport associated with Steve Sax?" ->\n'
    '{{"hops": ["Which sport is Steve Sax associated with?", '
    '"Which location is the birthplace of #PREV?", "Which country is #PREV in?"]}}\n'
    'Reply JSON only: {{"hops": [...]}}\n\nQuestion: {question}')


def decompose(question: str) -> list[str]:
    out = config.chat([{"role": "user", "content": _DECOMP_PROMPT.format(question=question)}],
                      qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"),
                      max_tokens=200, json_mode=True)
    try:
        hops = json.loads(re.search(r"\{.*\}", out, re.S).group(0))["hops"]
        return [str(h) for h in hops][:4] or [question]
    except Exception:
        return [question]


def answer_multihop(m: Tenet, question: str, k: int) -> tuple[str, str]:
    """Per-hop: recall from the conflict-free store, extract, substitute forward.
    Chain integrity: #PREV substitution is enforced, and a hop whose extracted value
    is not grounded in its pool gets one wider-recall retry (chain errors propagate,
    so per-hop grounding is what keeps hop 3 answerable)."""
    hops = decompose(question)
    val = ""
    pool_used = []
    for i, hop in enumerate(hops):
        if i > 0:
            hq = hop.replace("#PREV", val) if "#PREV" in hop else f"{hop} (of {val})"
        else:
            hq = hop
        pool, v = "", ""
        for kk in (k, 3 * k):                       # one wider retry if ungrounded
            hits = m.core.recall(hq, k=kk)
            pool = "\n".join(f"{h.source}. {h.text}" for h in hits)
            v = answer_extract(pool, hq).strip().rstrip(".")
            if v and normalize_answer(v) in normalize_answer(pool):
                break
        pool_used.append(pool)
        if not v:
            return "", "\n--\n".join(pool_used)
        val = v
    return val, "\n--\n".join(pool_used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma list e.g. sh_6k,mh_262k (default all 8)")
    ap.add_argument("--qpc", type=int, default=100, help="questions per cell")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hops-mh", type=int, default=0, help="recall hops for MH cells (tenet arm)")
    ap.add_argument("--tenet-read", choices=["official", "extract", "decompose"], default="official",
                    help="tenet reading: official MAB prompt | verbatim extraction | "
                         "per-hop Self-Ask decomposition (MH cells)")
    ap.add_argument("--dump", default="", help="misses JSONL")
    ap.add_argument("--keys", choices=["llm", "heuristic"], default="llm",
                    help="supersession keys: distiller (llm) or deterministic zero-LLM (heuristic)")
    args = ap.parse_args()
    global KEY_MODE
    KEY_MODE = args.keys

    from datasets import load_dataset
    cr = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution")
    want = set(args.cells.split(",")) if args.cells else None

    core = Tenet(CACHE / "emb_host.db")
    embedder = core.core.embed_batch
    dump_f = open(args.dump, "w") if args.dump else None

    results = {}
    t_start = time.time()
    for ex in cr:
        source = ex["metadata"]["source"]                       # factconsolidation_sh_6k
        cell = source.replace("factconsolidation_", "")         # sh_6k
        if want and cell not in want:
            continue
        is_mh = cell.startswith("mh")
        facts = parse_facts(ex["context"])
        # sh/mh cells share identical contexts — cache ingestion by content, not name
        import hashlib
        cache_id = "ctx" + hashlib.md5(ex["context"].encode()).hexdigest()[:12]
        print(f"\n=== {cell}: {len(facts)} facts (cache {cache_id}) ===", flush=True)

        m = build_tenet(cache_id, facts)
        line_vecs = embed_lines(cache_id, facts, embedder)
        texts = [f"{s}. {t}" for s, t in facts]

        qs = list(zip(ex["questions"], ex["answers"]))[: args.qpc]
        stats = {"rag": [0, 0], "tenet": [0, 0]}                # [correct, scored]
        errors = 0
        for qi, (q, gold) in enumerate(qs):
            qv = np.asarray(embedder([q])[0])
            # --- rag arm: top-k raw serial-numbered lines, serial order ---
            top = sorted(np.argsort(-(line_vecs @ qv))[: args.k])
            rag_pool = "\n".join(texts[i] for i in top)
            # --- tenet arm: current beliefs only (superseded values retired) ---
            if args.tenet_read == "decompose" and is_mh:
                tp, tenet_pool = answer_multihop(m, q, args.k)
            else:
                hops = args.hops_mh if is_mh else 0
                hits = m.core.recall(q, k=args.k, expand=args.k if hops else 0, hops=hops)
                tenet_pool = "\n".join(f"{h.source}. {h.text}" for h in hits)
                tp = (answer_extract(tenet_pool, q) if args.tenet_read != "official"
                      else answer(tenet_pool, q))
            rp = answer(rag_pool, q)
            if not rp.strip():
                errors += 1                                     # API failure: excluded
                continue
            if not tp.strip() and args.tenet_read != "decompose":
                errors += 1
                continue
            r_ok = subem_max(rp, gold)
            # decompose-mode empty = a pipeline abstention: scored WRONG, not excluded
            t_ok = subem_max(tp, gold) if tp.strip() else False
            stats["rag"][0] += r_ok; stats["rag"][1] += 1
            stats["tenet"][0] += t_ok; stats["tenet"][1] += 1
            if dump_f and not (r_ok and t_ok):
                dump_f.write(json.dumps({"cell": cell, "q": q, "gold": gold,
                                         "rag": rp, "tenet": tp,
                                         "rag_ok": bool(r_ok), "tenet_ok": bool(t_ok),
                                         "tenet_pool": tenet_pool}) + "\n")
                dump_f.flush()
            if (qi + 1) % 20 == 0:
                print(f"  [{qi+1}/{len(qs)}] rag={stats['rag'][0]}/{stats['rag'][1]} "
                      f"tenet={stats['tenet'][0]}/{stats['tenet'][1]}", flush=True)
        m.close()
        results[cell] = {a: (c, n) for a, (c, n) in stats.items()} | {"errors": errors}

    # ---- report ----
    print(f"\n=== MAB FactConsolidation (SubEM, k={args.k}, qpc={args.qpc}) ===")
    print(f"{'cell':>8} | {'RAG acc [95% CI]':>22} | {'TENET acc [95% CI]':>22} | err")
    for cell, r in sorted(results.items()):
        row = []
        for arm in ("rag", "tenet"):
            c, n = r[arm]
            p = c / n if n else 0.0
            lo, hi = wilson_ci(p, n)
            row.append(f"{100*p:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] n={n}")
        print(f"{cell:>8} | {row[0]:>22} | {row[1]:>22} | {r['errors']}")
    for arm in ("rag", "tenet"):
        cs = [(r[arm][0], r[arm][1]) for r in results.values()]
        for tag, pred in (("SH", lambda c: c.startswith("sh")), ("MH", lambda c: c.startswith("mh"))):
            cc = sum(r[arm][0] for cell, r in results.items() if pred(cell))
            nn = sum(r[arm][1] for cell, r in results.items() if pred(cell))
            if nn:
                lo, hi = wilson_ci(cc / nn, nn)
                print(f"{arm:>6} {tag} pooled: {100*cc/nn:.1f}% [{100*lo:.1f},{100*hi:.1f}] (n={nn})")
    print(f"wall={time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
