"""Supersession-firing benchmark — does keyed supersession actually fire on
natural-language preference updates?

Motivation (BENCHMARK.md §13, PersonaMem-v2): keyed supersession fired on only ~3.8% of
NL updates because the per-message distiller keys the SAME attribute differently across
turns ("user::milk" then "user::milk_preference") → no exact-skey collision → the stale
value stays current. This measures the fix (embedding-based key resolution, memory.py
`_resolve_key_supersede`, flag TENET_KEY_RESOLUTION) on a LABELED set with known ground
truth:

  POSITIVES — "switched from X to Y" chains (each update a SEPARATE distill call so keys
              vary naturally, reproducing the real failure). Ground truth: after the chain,
              exactly the LATEST value is current; earlier values are superseded.
  NEGATIVES — pairs of DISTINCT attributes of the same subject that must NOT collapse
              (pet=dog vs pet_name=Rex, milk vs coffee, car vs car_color, …).

Metrics per config:
  true-fire   = fraction of positive chains fully resolved (latest value current, no
                earlier value current) — excludes chains the distiller never stored the
                value for (ingestion failure ≠ supersession failure).
  false-fire  = fraction of negative pairs where a distinct fact was wrongly superseded.
  end-state   = fraction of positives whose CURRENT store contains the correct latest value.

Frugal: each message is distilled ONCE (cached), then store() is replayed into a fresh
in-memory store per config — so the tau sweep costs no extra LLM calls.

Usage:
  LLM_PROVIDER=qwen EMBED_PROVIDER=local python scripts/bench_supersession_firing.py \
      --taus 0.75,0.82,0.88,0.92 --cache <scratch>/superfire_cache
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tenet import config  # noqa: E402
from tenet.distill import distill  # noqa: E402
import tenet.memory as M  # noqa: E402
from tenet.memory import MemoryCore  # noqa: E402
from bench_factcon import wilson_ci  # noqa: E402


# --------------------------------------------------------------------------
# Labeled set — synthetic but phrased in varied natural language so the distiller
# keys each update independently (the real failure mode).
# --------------------------------------------------------------------------
_UPDATE_TMPL = [
    "I've switched to {v} now.", "These days I'm really into {v}.",
    "Update — I now go with {v}.", "I recently changed to {v}.",
    "Lately it's been {v} for me.", "I made the switch to {v}.",
]

# (attribute label, query, [values oldest→newest], value-phrase templates)
_POS_ATTRS = [
    ("milk", "what milk does the user drink", ["oat milk", "almond milk", "soy milk"], None),
    ("coffee", "the user's coffee order", ["black americano", "oat milk latte", "caramel macchiato"], None),
    ("gym", "which gym the user goes to", ["Gold's Gym", "Planet Fitness", "Equinox"], None),
    ("phone", "the user's phone", ["iPhone 14", "Pixel 8", "Galaxy S24"], None),
    ("car", "the car the user drives", ["Toyota Camry", "Honda Civic", "Tesla Model 3"], None),
    ("diet", "the user's diet", ["vegetarian", "pescatarian", "vegan"], None),
    ("hobby", "the user's main hobby", ["rock climbing", "oil painting", "salsa dancing"], None),
    ("laptop", "the user's laptop", ["MacBook Air", "ThinkPad X1", "Dell XPS 13"], None),
    ("streaming", "the user's streaming service", ["Netflix", "Hulu", "Disney Plus"], None),
    ("workout", "the user's workout routine", ["CrossFit", "yoga", "marathon training"], None),
    ("city", "which city the user lives in",
     ["Portland", "Austin", "Denver"], ["I moved to {v}.", "I just relocated to {v}.", "Now living in {v}."]),
    ("job", "the user's job title",
     ["junior analyst", "senior analyst", "team lead"],
     ["I got promoted to {v}.", "My new title is {v}.", "I'm now a {v}."]),
    ("email_client", "the user's email client", ["Outlook", "Gmail", "Proton Mail"], None),
    ("dog_food", "the brand of dog food the user buys", ["Blue Buffalo", "Purina One", "Hill's Science Diet"], None),
    ("commute", "how the user commutes to work", ["by bike", "by subway", "by carpool"],
     ["I now commute {v}.", "Switched my commute to {v}.", "These days I get to work {v}."]),
]

# NEGATIVES — distinct attributes of the same subject; must NOT collapse.
_NEG_PAIRS = [
    ("pet vs pet_name", ("The user has a golden retriever.", "user::pet", "golden retriever"),
     ("The user's dog is named Max.", "user::pet_name", "Max")),
    ("car vs car_color", ("The user drives a Toyota Camry.", "user::car", "Toyota Camry"),
     ("The user's car is blue.", "user::car_color", "blue")),
    ("milk vs coffee", ("The user drinks oat milk.", "user::milk", "oat milk"),
     ("The user's coffee order is a flat white.", "user::coffee", "flat white")),
    ("job vs employer", ("The user is a software engineer.", "user::job_title", "software engineer"),
     ("The user works at Acme Corp.", "user::employer", "Acme Corp")),
    ("residence vs hometown", ("The user lives in Portland.", "user::residence", "Portland"),
     ("The user grew up in Chicago.", "user::hometown", "Chicago")),
    ("spouse vs child", ("The user's wife is named Anna.", "user::spouse_name", "Anna"),
     ("The user's son is named Leo.", "user::child_name", "Leo")),
    ("phone vs laptop", ("The user uses an iPhone 14.", "user::phone", "iPhone 14"),
     ("The user's laptop is a MacBook Air.", "user::laptop", "MacBook Air")),
    ("gym vs trainer", ("The user goes to Gold's Gym.", "user::gym", "Gold's Gym"),
     ("The user's trainer is named Sarah.", "user::trainer_name", "Sarah")),
    ("diet vs allergy", ("The user is vegetarian.", "user::diet", "vegetarian"),
     ("The user is allergic to peanuts.", "user::allergy", "peanuts")),
    ("hobby vs job", ("The user's hobby is oil painting.", "user::hobby", "oil painting"),
     ("The user is a data scientist.", "user::job_title", "data scientist")),
    ("dog_breed vs dog_name", ("The user's dog is a beagle.", "user::dog_breed", "beagle"),
     ("The user's dog is called Biscuit.", "user::dog_name", "Biscuit")),
    ("city vs workplace_city", ("The user lives in Seattle.", "user::residence", "Seattle"),
     ("The user's office is in Bellevue.", "user::workplace_city", "Bellevue")),
]


def build_items(seed: int):
    import random
    rng = random.Random(seed)
    positives = []
    for label, query, values, tmpls in _POS_ATTRS:
        tmpls = tmpls or _UPDATE_TMPL
        msgs = []
        for i, v in enumerate(values):
            t = tmpls[i % len(tmpls)] if i < len(tmpls) else rng.choice(tmpls)
            msgs.append(t.format(v=v))
        positives.append({"label": label, "query": query, "values": values, "msgs": msgs})
    return positives, _NEG_PAIRS


# --------------------------------------------------------------------------
# Distillation cache (message text -> list of (statement, key, salience)).
# --------------------------------------------------------------------------
def distill_cached(text: str, cache: dict, cache_path: Path) -> list[tuple]:
    h = hashlib.sha256(text.encode()).hexdigest()[:16]
    if h in cache:
        return [tuple(x) for x in cache[h]]
    facts = [(f.statement, f.key, f.salience) for f in distill(text)]
    cache[h] = facts
    cache_path.write_text(json.dumps(cache))
    return facts


def norm(s: str) -> str:
    return s.lower().strip()


def run_config(positives, negatives, resolution: bool, tau: float):
    M._KEY_RESOLUTION = resolution
    M._TAU_KEY = tau
    core = MemoryCore(tempfile.mkdtemp() + "/f.db")

    # ---- positives: each has its own subject-namespace so chains don't cross-talk.
    resolved = end_ok = scorable = 0
    for pi, item in enumerate(positives):
        # replay distilled facts for each update message, in order
        for text in item["msgs"]:
            for stmt, key, sal in item["_facts"][text]:
                # namespace the subject per-item so two positive chains (both 'user')
                # don't collide in one shared store — we test WITHIN-chain supersession.
                core.store(stmt, key=f"p{pi}::{key.split('::',1)[-1]}" if "::" in key else key,
                           salience=sal)
        cur = [m for m in core.recall(item["query"], k=25) if m.kind != "raw"]
        cur_txt = " || ".join(norm(m.text) for m in cur)
        final_v = norm(item["values"][-1])
        earlier = [norm(v) for v in item["values"][:-1]]
        if final_v not in cur_txt:
            continue  # distiller never stored the final value → ingestion miss, not scored
        scorable += 1
        end_ok += 1
        if not any(e in cur_txt for e in earlier):
            resolved += 1

    # ---- negatives: store both facts (own subject namespace), both must stay current.
    false_fire = 0
    for ni, (label, a, b) in enumerate(negatives):
        (ta, ka, va), (tb, kb, vb) = a, b
        ka2 = f"n{ni}::{ka.split('::',1)[-1]}"
        kb2 = f"n{ni}::{kb.split('::',1)[-1]}"
        core.store(ta, key=ka2)
        core.store(tb, key=kb2)
        cur_txt = " || ".join(norm(m.text) for m in core.recall(label, k=25) if m.kind != "raw")
        # wrongly superseded if either value dropped out of current
        if norm(va) not in cur_txt or norm(vb) not in cur_txt:
            false_fire += 1
    core.close()
    return {"resolved": resolved, "scorable": scorable, "end_ok": end_ok,
            "n_pos": len(positives), "false_fire": false_fire, "n_neg": len(negatives)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.75,0.82,0.88,0.92")
    ap.add_argument("--text-floor", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    M._TEXT_FLOOR = args.text_floor
    taus = [float(x) for x in args.taus.split(",")]

    positives, negatives = build_items(args.seed)
    cache_path = Path(args.cache or (tempfile.mkdtemp() + "/distill.json"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    # distill every message once (cached)
    n_msgs = 0
    for item in positives:
        item["_facts"] = {}
        for text in item["msgs"]:
            item["_facts"][text] = distill_cached(text, cache, cache_path)
            n_msgs += 1
    for _label, a, b in negatives:
        pass  # negatives use hand-written statements/keys directly (no distill needed)
    print(f"labeled set: {len(positives)} positive chains ({n_msgs} update msgs), "
          f"{len(negatives)} negative pairs. text_floor={args.text_floor}\n", flush=True)

    configs = [("OFF (exact-key only)", False, 0.0)] + [(f"ON tau={t}", True, t) for t in taus]
    results = []
    print(f"{'config':>22} | {'true-fire':>22} | {'false-fire':>20} | {'end-state':>16}")
    for name, res, tau in configs:
        r = run_config(positives, negatives, res, tau)
        tf = r["resolved"] / r["scorable"] if r["scorable"] else 0.0
        ff = r["false_fire"] / r["n_neg"] if r["n_neg"] else 0.0
        es = r["end_ok"] / r["n_pos"] if r["n_pos"] else 0.0
        tlo, thi = wilson_ci(tf, r["scorable"])
        flo, fhi = wilson_ci(ff, r["n_neg"])
        results.append({"config": name, "tau": tau, "true_fire": tf, "false_fire": ff,
                        "end_state": es, **r})
        print(f"{name:>22} | {100*tf:5.1f}% [{100*tlo:4.1f},{100*thi:5.1f}] {r['resolved']:>2}/{r['scorable']:<2} "
              f"| {100*ff:5.1f}% [{100*flo:4.1f},{100*fhi:5.1f}] {r['false_fire']}/{r['n_neg']} "
              f"| {100*es:5.1f}% {r['end_ok']}/{r['n_pos']}")

    # pick: max true-fire with false-fire <= 2% (fallback: minimize false-fire then max true-fire)
    ok = [r for r in results if r["config"].startswith("ON") and r["false_fire"] <= 0.02]
    pick = (max(ok, key=lambda r: r["true_fire"]) if ok
            else max((r for r in results if r["config"].startswith("ON")),
                     key=lambda r: (r["true_fire"], -r["false_fire"])))
    base = next(r for r in results if not r["config"].startswith("ON"))
    print(f"\nbaseline (exact-key) true-fire: {100*base['true_fire']:.1f}%")
    print(f"RECOMMEND: {pick['config']}  true-fire={100*pick['true_fire']:.1f}%  "
          f"false-fire={100*pick['false_fire']:.1f}%  (gate: true-fire>=60% at false-fire<=2%)")
    gate = pick["true_fire"] >= 0.60 and pick["false_fire"] <= 0.02
    print(f"FIRING GATE: {'PASS' if gate else 'NOT MET'}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"results": results, "recommend": pick["config"], "recommend_tau": pick["tau"],
             "baseline_true_fire": base["true_fire"], "gate_pass": gate}, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
