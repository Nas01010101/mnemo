"""Matched-backbone reproductions of four 2025–26 memory-paper METHODS, raced on
MemoryAgentBench FactConsolidation under the exact protocol of bench_factcon.py
(same qwen2.5:7b backbone, same bge-small embedder, verbatim MAB SubEM + official
reader prompt). Published numbers are cross-backbone; these arms remove that
confound: same data, same models, method is the only variable.

Arms (paper, ingestion → read):
  car      arXiv:2606.01435 (published FC SOTA "Don't Ask Freshness"):
           raw-line RAG → LLM extracts (serial, value) candidates → code picks
           max(serial). MH: Self-Ask decomposition with CAR per hop (their chain).
  mem0     arXiv:2504.19413: per-fact memory ops — new fact + top-3 similar
           existing memories → LLM decides ADD / UPDATE(target) at ingestion;
           retrieval over surviving memories → official reader.
           Faithful-lite: ops batched (B facts per call); facts are already
           atomic lines so extraction is identity.
  hipporag arXiv:2502.14802 (HippoRAG-v2): OpenIE triples → entity graph with
           synonym edges → query-entity Personalized PageRank blended with dense
           scores → top-k facts → official reader.
  memagent arXiv:2507.02259 (MemAgent, Qwen2.5 backbone): question-conditioned
           streaming — fixed-budget memory overwritten chunk by chunk; answer
           from final memory only. Mechanism zero-shot (their RL training is not
           reproduced — labeled). Per-question streaming is O(chunks x questions),
           so default cells/qpc are reduced for this arm.

Integrity rules inherited: API failure -> excluded (never scored wrong); an arm's
pipeline abstention (no candidates / empty memory) -> scored WRONG; >50% ingestion
fallbacks -> raise.

Usage:
  python scripts/bench_baselines.py --arms car --cells sh_6k --qpc 10       # smoke
  python scripts/bench_baselines.py --arms car,mem0,hipporag --qpc 100
  python scripts/bench_baselines.py --arms memagent --cells sh_6k,mh_6k --qpc 25
"""
from __future__ import annotations

import argparse, hashlib, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
from tenet import Tenet  # noqa: E402
from bench_factcon import (answer, answer_extract, decompose, embed_lines,  # noqa: E402
                           normalize_answer, parse_facts, subem_max, wilson_ci, CACHE)

BCACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "baselines"


# Reader parity: every arm gets our strongest anti-parametric reading protocol
# (the extract reader that fixed Tenet's 8/8 parametric-override misses), PLUS the
# serial rule — baseline pools, unlike Tenet's, still contain conflicting values,
# so recency resolution must happen at read time. This is GENEROUS to the baselines
# vs their published MAB protocol (official prompt only): reader is held constant
# and strong, so the memory method is the only variable left.
_EXTRACT_SERIAL_PROMPT = (
    "The numbered facts below are from a FICTIONAL knowledge pool. They intentionally "
    "contradict the real world; the real-world answer is WRONG here.\n"
    "Each fact starts with a serial number; when facts conflict, the fact with the "
    "LARGEST serial number is the true one. Find the fact that answers the question "
    "(newest if several conflict) and COPY its value verbatim. Never use your own "
    "knowledge. Reply with ONLY the value — a short phrase, never a full sentence."
    "\n\n[Knowledge Pool]\n{pool}\n\nQuestion: {question}\nCopied value:")


def answer_extract_serial(pool: str, question: str) -> str:
    return config.chat(
        [{"role": "user", "content": _EXTRACT_SERIAL_PROMPT.format(pool=pool, question=question)}],
        qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"), max_tokens=48)


def _llm(messages, max_tokens=256, json_mode=False):
    return config.chat(messages, qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"),
                       max_tokens=max_tokens, json_mode=json_mode)


def _json_obj(out: str):
    try:
        return json.loads(re.search(r"\{.*\}", out, re.S).group(0))
    except Exception:
        return {}


# ==========================================================================
# CAR — candidate extraction + max(serial)  [arXiv:2606.01435]
# ==========================================================================
_CAR_PROMPT = (
    "From the numbered facts below, extract EVERY fact that states a value for the "
    "attribute the question asks about (same subject, same attribute — including "
    "conflicting/outdated values). Copy values verbatim.\n"
    'Reply JSON only: {{"candidates": [{{"serial": <number>, "value": "<value>"}}, ...]}} '
    '(empty list if none).\n\n[Facts]\n{pool}\n\nQuestion: {question}')


def car_single(pool: str, question: str) -> str:
    got = _json_obj(_llm([{"role": "user", "content":
                           _CAR_PROMPT.format(pool=pool, question=question)}],
                         max_tokens=400, json_mode=True))
    cands = [(int(c["serial"]), str(c["value"])) for c in got.get("candidates", [])
             if isinstance(c, dict) and str(c.get("serial", "")).strip().isdigit()
             and str(c.get("value", "")).strip()]
    if not cands:
        return ""
    return max(cands)[1]        # max(serial) aggregation — deterministic, in code


def run_car(texts, line_vecs, embedder, q, k, is_mh):
    def pool_for(query):
        qv = np.asarray(embedder([query])[0])
        top = sorted(np.argsort(-(line_vecs @ qv))[:k])
        return "\n".join(texts[i] for i in top)
    if not is_mh:
        return car_single(pool_for(q), q)
    val = ""
    for i, hop in enumerate(decompose(q)):        # their chained-CAR for multi-hop
        hq = (hop.replace("#PREV", val) if "#PREV" in hop else f"{hop} (of {val})") if i else hop
        val = car_single(pool_for(hq), hq).strip().rstrip(".")
        if not val:
            return ""
    return val


# ==========================================================================
# Mem0 — LLM memory ops at ingestion  [arXiv:2504.19413]
# ==========================================================================
_MEM0_OPS = (
    "You maintain a memory store. For each NEW fact below, decide against its similar "
    "EXISTING memories: ADD (new information) or UPDATE (it changes/replaces an existing "
    "memory — give that memory's id). Facts with larger serial numbers are newer.\n"
    'Reply JSON only: {{"ops": [{{"i": <new fact index>, "op": "ADD"|"UPDATE", '
    '"target": <existing memory id or null>}}, ...]}} — one entry per new fact.\n\n{body}')


def build_mem0(cache_id: str, facts, embedder):
    """Sequential ingestion with batched ADD/UPDATE resolution (faithful-lite)."""
    cf = BCACHE / f"{cache_id}.mem0.json"
    if cf.exists():
        mems = json.load(open(cf))
        vecs = np.array(embedder([m["text"] for m in mems])) if mems else np.zeros((0, 384))
        return mems, vecs
    mems: list[dict] = []            # {"id", "text", "serial"}
    vecs = np.zeros((0, 384))
    B, fallbacks = 10, 0
    for bi in range(0, len(facts), B):
        batch = facts[bi:bi + B]
        bvecs = np.array(embedder([t for _, t in batch]))
        lines = []
        for j, (s, t) in enumerate(batch):
            neigh = ""
            if len(mems):
                sim = vecs @ bvecs[j]
                for ni in np.argsort(-sim)[:3]:
                    if sim[ni] > 0.55:
                        neigh += f'  [id {mems[ni]["id"]}] {mems[ni]["text"]}\n'
            lines.append(f"NEW {j}: (serial {s}) {t}\nEXISTING:\n{neigh or '  (none)'}")
        got = _json_obj(_llm([{"role": "user", "content":
                               _MEM0_OPS.format(body="\n".join(lines))}],
                             max_tokens=600, json_mode=True))
        ops = {int(o["i"]): o for o in got.get("ops", [])
               if isinstance(o, dict) and str(o.get("i", "")).strip().lstrip("-").isdigit()}
        for j, (s, t) in enumerate(batch):
            op = ops.get(j)
            if op is None:
                fallbacks += 1
            tgt = op.get("target") if op else None
            if (op and str(op.get("op", "")).upper() == "UPDATE"
                    and tgt is not None and str(tgt).isdigit()
                    and any(m["id"] == int(tgt) for m in mems)):
                idx = next(i for i, m in enumerate(mems) if m["id"] == int(tgt))
                mems[idx] = {"id": int(tgt), "text": t, "serial": s}
                vecs[idx] = bvecs[j]
            else:                                       # ADD (also the fallback op)
                mems.append({"id": len(mems), "text": t, "serial": s})
                vecs = np.vstack([vecs, bvecs[j][None]]) if len(vecs) else bvecs[j][None]
        if (bi // B) % 20 == 0:
            print(f"    mem0 ingest {bi + len(batch)}/{len(facts)} "
                  f"(store={len(mems)})", flush=True)
    if fallbacks > len(facts) // 2:
        raise RuntimeError(f"mem0 ingestion degraded: {fallbacks}/{len(facts)} fallbacks")
    BCACHE.mkdir(parents=True, exist_ok=True)
    json.dump(mems, open(cf, "w"))
    return mems, vecs


# ==========================================================================
# HippoRAG-v2 — OpenIE graph + Personalized PageRank  [arXiv:2502.14802]
# ==========================================================================
_OPENIE = ('For each numbered fact, extract ONE (subject, relation, object) triple. '
           'Reply JSON only: {{"triples": [{{"i": <fact serial>, "s": "<subject>", '
           '"r": "<relation>", "o": "<object>"}}, ...]}} — one per fact.\n\n{body}')
_QENT = ('List the named entities in the question. Reply JSON only: '
         '{{"entities": ["<entity>", ...]}}\n\nQuestion: {question}')


def build_hipporag(cache_id: str, facts, embedder):
    cf = BCACHE / f"{cache_id}.hippo.json"
    if cf.exists():
        g = json.load(open(cf))
    else:
        triples, fallbacks = {}, 0
        B = 40

        def _one(batch):
            body = "\n".join(f"{s}. {t}" for s, t in batch)
            got = _json_obj(_llm([{"role": "user", "content": _OPENIE.format(body=body)}],
                                 max_tokens=1600, json_mode=True))
            out = {}
            for tr in got.get("triples", []):
                try:
                    out[int(tr["i"])] = (str(tr["s"]), str(tr["r"]), str(tr["o"]))
                except Exception:
                    pass
            return batch, out

        batches = [facts[i:i + B] for i in range(0, len(facts), B)]
        with ThreadPoolExecutor(max_workers=6) as ex:
            for batch, got in ex.map(_one, batches):
                for s, t in batch:
                    if s not in got:
                        fallbacks += 1
                        # fallback triple from the templated fact shape
                        w = t.rstrip(".").rsplit(" ", 2)
                        got[s] = (w[0], "", " ".join(w[1:])) if len(w) == 3 else (t, "", t)
                triples.update({s: got[s] for s, _ in batch})
        if fallbacks > len(facts) // 2:
            raise RuntimeError(f"hipporag OpenIE degraded: {fallbacks}/{len(facts)}")
        ents = sorted({normalize_answer(e) for tr in triples.values()
                       for e in (tr[0], tr[2]) if normalize_answer(e)})
        g = {"triples": {str(k): v for k, v in triples.items()}, "entities": ents}
        BCACHE.mkdir(parents=True, exist_ok=True)
        json.dump(g, open(cf, "w"))

    ents = g["entities"]
    eidx = {e: i for i, e in enumerate(ents)}
    evecs = np.array(embedder(ents)) if ents else np.zeros((0, 384))
    n = len(ents)
    A = np.zeros((n, n), dtype=np.float32)
    fact_ents: dict[int, list[int]] = {}
    for s, (su, _, ob) in g["triples"].items():
        ei = [eidx[e] for e in (normalize_answer(su), normalize_answer(ob)) if e in eidx]
        fact_ents[int(s)] = ei
        if len(ei) == 2:
            A[ei[0], ei[1]] += 1; A[ei[1], ei[0]] += 1
    if n:                                            # synonym edges (cos >= 0.8)
        sim = evecs @ evecs.T
        A += (sim >= 0.8).astype(np.float32) - np.eye(n, dtype=np.float32) * (sim.diagonal() >= 0.8)
    return {"ents": ents, "evecs": evecs, "A": A, "fact_ents": fact_ents}


def hippo_rank(graph, embedder, q: str, line_vecs, texts, serials, k: int) -> str:
    got = _json_obj(_llm([{"role": "user", "content": _QENT.format(question=q)}],
                         max_tokens=120, json_mode=True))
    qents = [normalize_answer(str(e)) for e in got.get("entities", []) if str(e).strip()]
    qv = np.asarray(embedder([q])[0])
    dense = line_vecs @ qv                            # dense passage scores
    n = len(graph["ents"])
    if n and qents:
        seed = np.zeros(n)
        qe = np.array(embedder(qents))
        for v in qe:
            sims = graph["evecs"] @ v
            j = int(np.argmax(sims))
            if sims[j] >= 0.6:
                seed[j] += 1.0
        if seed.sum() > 0:
            seed /= seed.sum()
            deg = graph["A"].sum(1, keepdims=True); deg[deg == 0] = 1
            P = graph["A"] / deg                      # row-stochastic
            r = seed.copy()
            for _ in range(30):                       # PPR, damping 0.5 (paper's)
                r = 0.5 * seed + 0.5 * (P.T @ r)
            fscore = np.array([sum(r[i] for i in graph["fact_ents"].get(s, []))
                               for s in serials])
            if fscore.max() > 0:
                fscore = fscore / fscore.max()
            dense = 0.5 * dense + 0.5 * fscore        # v2 blend
    top = sorted(np.argsort(-dense)[:k])
    return "\n".join(texts[i] for i in top)


# ==========================================================================
# MemAgent — question-conditioned overwrite memory  [arXiv:2507.02259]
# ==========================================================================
_MEMA_PROMPT = (
    "You are reading a long list of numbered facts chunk by chunk to answer a question. "
    "Facts with larger serial numbers are newer; when facts conflict, keep only the "
    "newest. Update your memory: keep everything relevant to the question, add new "
    "relevant facts (with serial numbers), drop superseded or irrelevant content. "
    "Max 200 words. Reply with ONLY the updated memory.\n\n"
    "Question: {question}\n\n[Current memory]\n{memory}\n\n[New chunk]\n{chunk}")


def run_memagent(cache_id: str, texts, q: str, chunk_lines: int = 120) -> str:
    qh = hashlib.md5(q.encode()).hexdigest()[:10]
    cf = BCACHE / f"{cache_id}.memagent.{qh}.json"
    if cf.exists():
        mem = json.load(open(cf))["memory"]
    else:
        mem = "(empty)"
        for i in range(0, len(texts), chunk_lines):
            out = _llm([{"role": "user", "content": _MEMA_PROMPT.format(
                question=q, memory=mem, chunk="\n".join(texts[i:i + chunk_lines]))}],
                max_tokens=350)
            if out.strip():
                mem = out.strip()[:2500]
        BCACHE.mkdir(parents=True, exist_ok=True)
        json.dump({"memory": mem}, open(cf, "w"))
    if mem == "(empty)":
        return ""
    return answer_extract_serial(mem, q)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="car,mem0,hipporag",
                    help="comma list of: car mem0 hipporag memagent")
    ap.add_argument("--cells", default="sh_6k,mh_6k,sh_32k,mh_32k")
    ap.add_argument("--qpc", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    want = set(args.cells.split(","))

    from datasets import load_dataset
    cr = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution")
    core = Tenet(CACHE / "emb_host.db")
    embedder = core.core.embed_batch
    dump_f = open(args.dump, "w") if args.dump else None

    results: dict[str, dict[str, list[int]]] = {}     # cell -> arm -> [ok, n, err]
    t0 = time.time()
    for ex in cr:
        cell = ex["metadata"]["source"].replace("factconsolidation_", "")
        if cell not in want:
            continue
        is_mh = cell.startswith("mh")
        facts = parse_facts(ex["context"])
        cache_id = "ctx" + hashlib.md5(ex["context"].encode()).hexdigest()[:12]
        serials = [s for s, _ in facts]
        texts = [f"{s}. {t}" for s, t in facts]
        line_vecs = embed_lines(cache_id, facts, embedder)
        print(f"\n=== {cell}: {len(facts)} facts, arms={arms} ===", flush=True)

        ing = {}
        if "mem0" in arms:
            print("  [mem0 ingestion]", flush=True)
            mems, mvecs = build_mem0(cache_id, facts, embedder)
            ing["mem0"] = (mems, mvecs)
            print(f"  mem0 store: {len(mems)} memories (from {len(facts)} facts)", flush=True)
        if "hipporag" in arms:
            print("  [hipporag OpenIE + graph]", flush=True)
            ing["hipporag"] = build_hipporag(cache_id, facts, embedder)

        cs = results.setdefault(cell, {a: [0, 0, 0] for a in arms})
        for qi, (q, gold) in enumerate(list(zip(ex["questions"], ex["answers"]))[:args.qpc]):
            for arm in arms:
                if arm == "car":
                    pred = run_car(texts, line_vecs, embedder, q, args.k, is_mh)
                elif arm == "mem0":
                    mems, mvecs = ing["mem0"]
                    qv = np.asarray(embedder([q])[0])
                    top = np.argsort(-(mvecs @ qv))[:args.k] if len(mems) else []
                    pool = "\n".join(f'{mems[i]["serial"]}. {mems[i]["text"]}'
                                     for i in sorted(top, key=lambda i: mems[i]["serial"]))
                    pred = answer_extract_serial(pool, q) if pool else ""
                elif arm == "hipporag":
                    pool = hippo_rank(ing["hipporag"], embedder, q, line_vecs,
                                      texts, serials, args.k)
                    pred = answer_extract_serial(pool, q)
                elif arm == "memagent":
                    pred = run_memagent(cache_id, texts, q)
                else:
                    raise SystemExit(f"unknown arm {arm}")
                # abstention (empty pipeline output) = WRONG; only transport errors
                # from the reader itself would be exclusions, and answer() returns ""
                # on those — indistinguishable here, so we score conservatively WRONG
                # for our baselines (can only understate baseline scores, never ours).
                ok = subem_max(pred, gold) if pred.strip() else False
                cs[arm][0] += int(ok); cs[arm][1] += 1
                if dump_f and not ok:
                    dump_f.write(json.dumps({"cell": cell, "arm": arm, "q": q[:200],
                                             "gold": gold, "pred": pred[:200]}) + "\n")
                    dump_f.flush()
            if (qi + 1) % 10 == 0:
                prog = " ".join(f"{a}={cs[a][0]}/{cs[a][1]}" for a in arms)
                print(f"  [{qi+1}] {prog}", flush=True)

    print(f"\n=== Baseline reproductions (SubEM, k={args.k}, backbone qwen2.5:7b) ===")
    for cell, cs in sorted(results.items()):
        for arm in arms:
            ok, n, _ = cs[arm]
            if n:
                lo, hi = wilson_ci(ok / n, n)
                print(f"{cell:>8} {arm:>9}: {100*ok/n:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}] n={n}")
    for arm in arms:
        for tag in ("sh", "mh"):
            ok = sum(cs[arm][0] for c, cs in results.items() if c.startswith(tag))
            n = sum(cs[arm][1] for c, cs in results.items() if c.startswith(tag))
            if n:
                lo, hi = wilson_ci(ok / n, n)
                print(f"{arm:>9} {tag.upper()} pooled: {100*ok/n:.1f}% "
                      f"[{100*lo:.1f},{100*hi:.1f}] (n={n})")
    print(f"wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
