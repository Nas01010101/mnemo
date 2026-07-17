"""Build the three fact-update event datasets for the neural world-model (dynamics).

Torch-free by design: this runs on the Mac (or the RTX box) and emits event streams
as JSONL + a value-vocabulary file. Embeddings (bge-small 384d) are computed LATER,
on the GPU box, by train_dynamics.py — this file never imports torch.

An EVENT is one observed value of a key at a time:
    {seq, key, kclass, t, dt_prev, value, vid}
  seq      : sequence/group id (a per-actor or per-context chain) — the split unit
  key      : full "subject::attribute" skey
  kclass   : attribute class (dynamics unit), e.g. "mood", "residence"
  t        : absolute event time (seconds, or serial for MAB); float
  dt_prev  : seconds since the PREVIOUS event on THIS key in this seq (-1 if first)
  value    : value text (the object/tail) — what the key was set to
  vid      : index into the global value vocab (for cheap negatives)
  tmean    : 1 if wall-clock time features are meaningful for this source else 0

Sequences are the leakage boundary: whole sequences go to train OR test, never split.

Usage:
  python scripts/wm_data.py --out data/wm --synthetic 50000
  python scripts/wm_data.py --out data/wm --mab --lme --synthetic 50000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

DAY = 86400.0
HOUR = 3600.0


# ===========================================================================
# 1. SYNTHETIC — planted structure the Gamma-exponential model provably cannot
#    capture (constant per-key hazard, memoryless, no covariates, no cascades).
# ===========================================================================
# Job ladder: next value walks UP one rung (structured next-value target).
_JOB_LADDER = [
    "intern", "junior engineer", "software engineer", "senior engineer",
    "staff engineer", "engineering manager", "director of engineering", "vp engineering",
]
# Geography graph: cities linked to neighbours; a move goes to a NEIGHBOUR (structured).
_CITY_GRAPH = {
    "seattle": ["portland", "vancouver", "san francisco"],
    "portland": ["seattle", "san francisco"],
    "vancouver": ["seattle", "portland"],
    "san francisco": ["portland", "los angeles", "seattle"],
    "los angeles": ["san francisco", "san diego", "phoenix"],
    "san diego": ["los angeles", "phoenix"],
    "phoenix": ["los angeles", "san diego", "denver"],
    "denver": ["phoenix", "austin", "chicago"],
    "austin": ["denver", "dallas", "houston"],
    "dallas": ["austin", "houston"],
    "houston": ["austin", "dallas"],
    "chicago": ["denver", "detroit", "new york"],
    "detroit": ["chicago", "new york"],
    "new york": ["chicago", "boston", "philadelphia"],
    "boston": ["new york", "philadelphia"],
    "philadelphia": ["new york", "boston"],
}
_CITIES = list(_CITY_GRAPH)
_MOODS = ["happy", "tired", "excited", "stressed", "calm", "focused", "anxious", "content"]
_ENERGY = ["high", "medium", "low", "drained", "peak"]
_TASKS = ["todo", "in progress", "blocked", "review", "done"]
_CHECKINS = ["home", "office", "gym", "cafe", "airport", "park", "restaurant"]


def gen_synthetic(n_events: int, seed: int = 0) -> list[dict]:
    """~n_events events across many actors, with four planted structures:
      (a) PERIODIC  keys (mood, energy): change ~daily on weekdays near a preferred
          hour  -> hazard is time-of-day/day-of-week modulated, NOT constant.
      (b) BURSTY    keys (task_status, checkin): changes cluster then go quiet
          -> hazard is decreasing-within-burst (Weibull k<1), NOT memoryless.
      (c) CASCADE   residence -> job within a few days (correlated cross-key), and
          job value walks the ladder up.
      (d) VALUE     structure: job = ladder step; residence = geography-graph neighbour.
    None of these are representable by per-key constant-hazard Gamma-exponential.
    """
    rng = random.Random(seed)
    events: list[dict] = []
    actor = 0
    # generate actor chains until we hit the event budget
    while len(events) < n_events:
        aid = f"a{actor}"
        actor += 1
        # each actor spans ~1 year starting at a random epoch (2 yrs window)
        t0 = 1_600_000_000.0 + rng.uniform(0, 2 * 365 * DAY)
        span = rng.uniform(120 * DAY, 400 * DAY)
        pref_hour = rng.randint(7, 21)          # this actor's preferred change hour
        ev: list[dict] = []

        def add(kclass, t, value):
            ev.append({"seq": aid, "key": f"user::{kclass}", "kclass": kclass,
                       "t": float(t), "value": str(value)})

        # (a) PERIODIC: mood + energy change ~daily on weekdays, near pref_hour.
        for kclass, vocab in (("mood", _MOODS), ("energy", _ENERGY)):
            t = t0
            last = None
            while t < t0 + span:
                # advance ~1 day, but snap to weekday + preferred hour (periodic hazard)
                t += DAY * rng.uniform(0.7, 1.4)
                dow = int((t / DAY) % 7)
                if dow >= 5:                     # weekends: usually no change (silence)
                    if rng.random() < 0.75:
                        continue
                # snap toward preferred hour
                day_start = (t // DAY) * DAY
                t_snap = day_start + pref_hour * HOUR + rng.uniform(-1.5, 1.5) * HOUR
                if t_snap <= (last or 0):
                    continue
                v = rng.choice([x for x in vocab if x != last])
                add(kclass, t_snap, v)
                last = t_snap

        # (b) BURSTY: task_status + checkin cluster in bursts (Weibull k<1 within burst).
        for kclass, vocab in (("task_status", _TASKS), ("checkin", _CHECKINS)):
            t = t0 + rng.uniform(0, 20 * DAY)
            last = None
            while t < t0 + span:
                # a burst: 3-8 rapid changes minutes-hours apart, then a long quiet gap
                burst = rng.randint(3, 8)
                for _ in range(burst):
                    t += rng.uniform(0.02, 0.5) * DAY   # rapid (decreasing hazard shape)
                    if t >= t0 + span:
                        break
                    v = rng.choice([x for x in vocab if x != last])
                    add(kclass, t, v)
                    last = v                        # bursty keys: value unstructured
                t += rng.uniform(10, 40) * DAY          # long quiet gap between bursts

        # (c)+(d) CASCADE residence->job, with structured values.
        city = rng.choice(_CITIES)
        rung = rng.randint(0, 2)
        add("residence", t0 + rng.uniform(0, 10 * DAY), city)
        add("job", t0 + rng.uniform(0, 10 * DAY), _JOB_LADDER[rung])
        t = t0 + rng.uniform(30 * DAY, 90 * DAY)
        while t < t0 + span:
            city = rng.choice(_CITY_GRAPH[city])                # move to a NEIGHBOUR
            add("residence", t, city)
            # job follows the move within 1-6 days (cascade), climbing the ladder
            if rng.random() < 0.8:
                rung = min(rung + 1, len(_JOB_LADDER) - 1)
                add("job", t + rng.uniform(1, 6) * DAY, _JOB_LADDER[rung])
            t += rng.uniform(60 * DAY, 150 * DAY)               # residence is slow

        events.extend(ev)
    events = events[:n_events]
    for e in events:
        e["tmean"] = 1
    return _finalize(events)


# ===========================================================================
# 2. MAB FactConsolidation — real serial-numbered counterfactual update chains.
#    Keys via the SAME deterministic heuristic bench_factcon.py uses (zero-LLM):
#    key = normalized fact minus its last two (templated value) words.
# ===========================================================================
def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


_FACT_RE = re.compile(r"^(\d+)\.\s+(.*\S)\s*$")


def gen_mab(limit_contexts: int | None = None) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        print(f"[mab] datasets unavailable: {e}", file=sys.stderr)
        return []
    try:
        cr = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution")
    except Exception as e:  # noqa: BLE001
        print(f"[mab] load failed: {e}", file=sys.stderr)
        return []
    events: list[dict] = []
    seen_ctx: set[str] = set()
    import hashlib
    n_ctx = 0
    for ex in cr:
        ctx = ex["context"]
        h = hashlib.md5(ctx.encode()).hexdigest()[:12]
        if h in seen_ctx:                                   # sh/mh share contexts
            continue
        seen_ctx.add(h)
        n_ctx += 1
        if limit_contexts and n_ctx > limit_contexts:
            break
        facts = []
        for line in ctx.splitlines():
            m = _FACT_RE.match(line.strip())
            if m:
                facts.append((int(m.group(1)), m.group(2)))
        for serial, text in facts:
            norm = _normalize(text)
            kclass = " ".join(norm.split()[:-2]) or "unk"    # heuristic key (value removed)
            value = " ".join(norm.split()[-2:])              # the templated tail
            events.append({"seq": f"mab_{h}", "key": f"{h}::{kclass}", "kclass": kclass,
                           "t": float(serial), "value": value, "tmean": 0})
    print(f"[mab] {n_ctx} contexts -> {len(events)} events", file=sys.stderr)
    return _finalize(events)


# ===========================================================================
# 3. LongMemEval_S — real knowledge-update chains across sessions.
#    Extract facts whose value changes across sessions (same subject::attribute).
#    LME_S has session timestamps -> wall-clock time features are meaningful.
# ===========================================================================
_LME_PATHS = [Path("data/lme/longmemeval_s.json")]
if os.environ.get("TENET_LME_DATA_PATH"):  # override/extra search location, no personal path baked in
    _LME_PATHS.append(Path(os.environ["TENET_LME_DATA_PATH"]))


def _parse_iso(s: str) -> float | None:
    import datetime
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s.strip(), fmt).timestamp()
        except (ValueError, TypeError):
            continue
    return None


def gen_lme() -> list[dict]:
    """Heuristic knowledge-update extraction: within each LME instance, mine repeated
    'my X is/was Y' style attribute mentions across sessions and chain them by session
    time. This is a NOISY real signal (regex distillation, not an LLM) — reported as
    such. The goal is real cross-session update chains, not perfect facts."""
    path = next((p for p in _LME_PATHS if p.exists()), None)
    if path is None:
        print("[lme] no longmemeval_s.json found", file=sys.stderr)
        return []
    data = json.loads(path.read_text())
    # attribute patterns: capture (attribute, value) from first-person statements
    pats = [
        re.compile(r"\bmy ([a-z][a-z ]{2,25}?) (?:is|was|are|will be) ([a-z0-9][a-z0-9 '\-]{1,30})", re.I),
        re.compile(r"\bi (?:now )?(?:live|moved to|work at|work as|am a|am an) ([a-z0-9][a-z0-9 '\-]{2,30})", re.I),
    ]
    events: list[dict] = []
    for inst in data:
        qid = inst.get("question_id", inst.get("id", str(id(inst))))
        sessions = inst.get("haystack_sessions") or inst.get("sessions") or []
        dates = inst.get("haystack_dates") or inst.get("session_dates") or []
        # per-instance: attribute -> list of (t, value)
        chains: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for si, sess in enumerate(sessions):
            ts = _parse_iso(dates[si]) if si < len(dates) and dates else None
            if ts is None:
                ts = float(si)                              # fall back to session order
            turns = sess if isinstance(sess, list) else sess.get("turns", [])
            for turn in turns:
                content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
                role = turn.get("role", "") if isinstance(turn, dict) else ""
                if role and role != "user":
                    continue
                for pat in pats:
                    for m in pat.finditer(content):
                        if len(m.groups()) == 2:
                            attr = _normalize(m.group(1)); val = _normalize(m.group(2))
                        else:
                            attr = "attribute"; val = _normalize(m.group(1))
                        if attr and val and len(val.split()) <= 4:
                            chains[attr].append((ts, val))
        # keep only attributes that actually CHANGE value (>=2 distinct values = updates)
        for attr, seq in chains.items():
            seq.sort(key=lambda x: x[0])
            # collapse consecutive identical values
            collapsed = []
            for t, v in seq:
                if not collapsed or collapsed[-1][1] != v:
                    collapsed.append((t, v))
            if len({v for _, v in collapsed}) < 2:
                continue
            for t, v in collapsed:
                events.append({"seq": f"lme_{qid}", "key": f"{qid}::{attr}",
                               "kclass": attr, "t": float(t), "value": v, "tmean": 1})
    print(f"[lme] {len(events)} update events from {path}", file=sys.stderr)
    return _finalize(events)


# ===========================================================================
# finalize: compute dt_prev per key, assign value vocab ids, sort.
# ===========================================================================
def _finalize(events: list[dict]) -> list[dict]:
    events.sort(key=lambda e: (e["seq"], e["key"], e["t"]))
    last_t: dict[tuple, float] = {}
    for e in events:
        kk = (e["seq"], e["key"])
        e["dt_prev"] = (e["t"] - last_t[kk]) if kk in last_t else -1.0
        last_t[kk] = e["t"]
    return events


def build_vocab(events: list[dict]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for e in events:
        v = e["value"]
        if v not in vocab:
            vocab[v] = len(vocab)
    for e in events:
        e["vid"] = vocab[e["value"]]
    return vocab


def summarize(events: list[dict], src: str) -> dict:
    keys = {e["key"] for e in events}
    seqs = {e["seq"] for e in events}
    kclasses = {e["kclass"] for e in events}
    # supersession events = non-first events on a key (an observed lifetime end)
    supers = sum(1 for e in events if e["dt_prev"] >= 0)
    return {"source": src, "events": len(events), "sequences": len(seqs),
            "keys": len(keys), "key_classes": len(kclasses), "supersessions": supers,
            "unique_values": len({e["value"] for e in events})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wm")
    ap.add_argument("--synthetic", type=int, default=0, help="n synthetic events (0=skip)")
    ap.add_argument("--mab", action="store_true")
    ap.add_argument("--lme", action="store_true")
    ap.add_argument("--mab-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[dict]] = {}
    if args.synthetic:
        sources["synthetic"] = gen_synthetic(args.synthetic, seed=args.seed)
    if args.mab:
        ev = gen_mab(args.mab_limit)
        if ev:
            sources["mab"] = ev
    if args.lme:
        ev = gen_lme()
        if ev:
            sources["lme"] = ev
    if not sources:
        print("nothing to build (pass --synthetic N / --mab / --lme)", file=sys.stderr)
        sys.exit(1)

    summaries = []
    for src, events in sources.items():
        build_vocab(events)
        path = out / f"{src}.jsonl"
        with path.open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        s = summarize(events, src)
        summaries.append(s)
        print(f"wrote {path}  {s}")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("\n=== DATA SUMMARY ===")
    print(f"{'source':>10} | {'events':>7} | {'seqs':>5} | {'keys':>5} | "
          f"{'classes':>7} | {'supers':>6} | {'uniqvals':>8}")
    for s in summaries:
        print(f"{s['source']:>10} | {s['events']:>7} | {s['sequences']:>5} | {s['keys']:>5} | "
              f"{s['key_classes']:>7} | {s['supersessions']:>6} | {s['unique_values']:>8}")


if __name__ == "__main__":
    main()
