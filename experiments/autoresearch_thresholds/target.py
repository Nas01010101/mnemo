"""Autoresearch HP-sweep target — tune Tenet's three supersession thresholds
against a deterministic, LLM-FREE churn-retrieval metric.

The driver (sweep.py) proposes {consistency_threshold, tau_key, text_floor} and
writes it to $AUTORESEARCH_CONFIG; this target builds a FIXED, seeded synthetic
knowledge-churn corpus, ingests it (embeddings only — bge-small local, NO LLM,
NO cloud call, NO ollama), runs recall(), and writes the scalar ERROR to
$AUTORESEARCH_RESULTS (lower is better). The number the driver reads comes from
the results file, never this script's stdout.

The JUDGE — the synthetic corpus generator, the fixed seed, and the scoring rule
below — is NOT the search surface: the config only sets three numeric thresholds.
There is no knob here to reward-hack; a "win" that required changing the corpus or
the scorer would be void.

Proxy honesty: this synthetic corpus models the distiller-key-drift + stale-raw-echo
regime that the real supersession thresholds exist for (a subject's attribute is
updated U times, each update keyed with a DIFFERENT synonym for the same attribute,
plus a verbatim raw echo of each old value). Any config that wins here MUST be
re-verified on the real benchmarks (bench_churn / bench_supersession_firing) before
being adopted as a default — it is a fast search proxy, not the final judge.
"""
import json
import os
import tempfile

# ── read driver-proposed config, set the thresholds via env BEFORE importing tenet ──
cfg_path = os.environ.get("AUTORESEARCH_CONFIG")
cfg = json.loads(open(cfg_path).read()) if cfg_path and os.path.exists(cfg_path) else {}
consistency_threshold = float(cfg.get("consistency_threshold", 0.70))
tau_key = float(cfg.get("tau_key", 0.78))
text_floor = float(cfg.get("text_floor", 0.66))

# key-resolution thresholds are read at import from these env vars (memory.py)
os.environ["TENET_KEY_RESOLUTION"] = "on"
os.environ["TENET_KEY_RESOLUTION_TAU"] = str(tau_key)
os.environ["TENET_KEY_RESOLUTION_TEXTFLOOR"] = str(text_floor)
os.environ["EMBED_PROVIDER"] = "local"          # bge-small, ~130MB, deterministic, RAM-safe
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.pop("OLLAMA_MODEL", None)             # belt-and-suspenders: never a local LLM
os.environ.pop("OLLAMA_BASE_URL", None)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tenet.memory import MemoryCore  # noqa: E402

# ── JUDGE v2: fixed, seeded synthetic churn corpus WITH negatives (do not edit) ──
# Run 1 (journal rows 1-16) used a positives-only metric — every candidate pair was
# a TRUE supersession, so the BO surrogate reward-hacked tau_key down to 0.66 (more
# aggressive fuzzy-matching can only help when there is nothing to falsely match).
# Judge v2 adds FALSE-SUPERSESSION NEGATIVES: each subject also owns a `work_city`
# fact — embedding-near the residence synonyms (city/location/home_city) and sharing
# the subject token in its text, so a too-low tau_key/text_floor WILL collapse it —
# that must still be CURRENT after the residence churn. The objective is now
# 1 − F1(currency-recall, negative-survival): driving tau_key down buys recall but
# bleeds precision, which is exactly the trade the real thresholds arbitrate.
SEED = 0
S = 15          # subjects
U = 4           # updates per attribute (churn depth)
# synonym sets for one attribute — the distiller-key-drift the thresholds must absorb
ATTR_SYNONYMS = ["residence", "home_city", "city", "location", "home_town"]
BASE_ATTR = "residence"
CITIES = ["Boston", "Seattle", "Denver", "Austin", "Chicago", "Portland",
          "Miami", "Dallas", "Phoenix", "Atlanta", "Newark", "Fresno",
          "Tucson", "Reno", "Tulsa", "Akron", "Provo", "Ogden", "Salem", "Erie"]
WORK_CITIES = ["Springfield", "Richmond", "Columbus", "Lansing", "Trenton",
               "Boise", "Helena", "Topeka", "Augusta", "Pierre", "Juneau",
               "Concord", "Dover", "Annapolis", "Olympia"]


def _rng(seed):
    import random
    r = random.Random(seed)
    return r


def build_and_score() -> float:
    r = _rng(SEED)
    dbdir = tempfile.mkdtemp(prefix="ar_tenet_")
    clock = [1000.0]
    core = MemoryCore(os.path.join(dbdir, "t.db"), now=lambda: clock[0])
    truth = {}   # subject -> current (last) value string
    for s in range(S):
        subj = f"subj{s}"
        # NEGATIVE anchor: a distinct same-subject attribute, stored BEFORE the churn,
        # that no residence update may supersede. Key + text deliberately adversarial:
        # `work_city` is embedding-near the residence synonyms, and the sentence shares
        # the subject token with every churn sentence.
        clock[0] += 100.0
        core.store(f"{subj} works in {WORK_CITIES[s]}", key=f"{subj}::work_city",
                   valid_at=clock[0])
        # U successive values for this subject's residence, each keyed with a DIFFERENT
        # synonym (distiller drift) + a verbatim raw echo of the value at that time.
        vals = r.sample(CITIES, U)
        for i, v in enumerate(vals):
            clock[0] += 100.0
            syn = ATTR_SYNONYMS[i % len(ATTR_SYNONYMS)]
            core.store(f"{subj} lives in {v}", key=f"{subj}::{syn}", valid_at=clock[0])
            core.store(f"{subj} said: I live in {v}", kind="raw",
                       source=subj, valid_at=clock[0], surprise_gate=None)
        truth[subj] = vals[-1]

    # RECALL component (positives): current city surfaced and ranked above any stale
    # city of the same subject (pure retrieval; LLM-free).
    correct = 0
    for s in range(S):
        subj = f"subj{s}"
        cur = truth[subj]
        mems = core.recall(f"where does {subj} live now",
                           k=5, consistency_threshold=consistency_threshold)
        ranked_cities = []
        for m in mems:
            if subj not in m.text:
                continue
            for c in CITIES:
                if c in m.text:
                    ranked_cities.append(c)
                    break
        if ranked_cities and ranked_cities[0] == cur:
            correct += 1
    rec = correct / S

    # PRECISION component (negatives): every work_city fact must still be CURRENT —
    # a residence update that superseded it is a false supersession.
    survived = 0
    for s in range(S):
        row = core.db.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE skey=? AND archived=0 "
            "AND expired_at IS NULL", (f"subj{s}::work_city",)).fetchone()
        if row["n"] == 1:
            survived += 1
    prec = survived / S

    core.close()
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    print(f"  components: recall={rec:.3f} negative-survival={prec:.3f} f1={f1:.3f}")
    return 1.0 - f1   # ERROR, lower is better


err = build_and_score()

results = os.environ.get("AUTORESEARCH_RESULTS")
if results:
    with open(results, "w") as f:
        json.dump({"metric": err}, f)
print(f"error={err:.4f}  (consistency={consistency_threshold} tau_key={tau_key} text_floor={text_floor})")
