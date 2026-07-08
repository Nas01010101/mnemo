"""LongMemEval-V2 memory-backend adapter for Tenet (zero-LLM read path).

Maps the official LME-V2 `Memory` interface onto Tenet's MemoryCore:

  insert(trajectory) -> chunk the trajectory into date/step/url-aware text slices
                        (goal + outcome header, then per-state thought/action and a
                        trimmed accessibility-tree excerpt), embed in batches, and
                        store each as a raw slice (kind="raw", surprise-gated so the
                        repetitive ServiceNow/WebArena DOM boilerplate is not stored
                        many times over).  ZERO LLM calls — embeddings only.

  query(q, img) -> core.recall(q, k, expand, hops, char_budget), returned as the
                   harness's list[{"type":"text","value":...}] context payload.
                   ZERO LLM calls, sub-second — this is the LAFS latency lever.

Wiring into the official harness (the clone is throwaway, not committed):
  1. Ensure both repos are importable:
       export PYTHONPATH="$LMEV2_REPO:$TENET_REPO/src:$TENET_REPO/scripts"
  2. Register the backend by importing this module. Either add
       `from lmev2_adapter import TenetMemory  # noqa`
     to the bottom of `memory_modules/memory.py`, or pass it via a sitecustomize.
  3. Memory config JSON:
       {"memory_type": "tenet", "memory_params": {"k": 12, "expand": 24, "hops": 2}}
     and run:  python evaluation/harness.py --memory-config-path tenet.json ...

Env for the local, no-paid-API path:
  EMBED_PROVIDER=local  LOCAL_EMBED_MODEL=BAAI/bge-small-en-v1.5

Standalone mechanism smoke test (no reader, no judge, no API cost):
  python scripts/lmev2_adapter.py --data-root ~/scratch/lmev2/data/longmemeval-v2 \
      --domain enterprise --n-trajectories 8 --n-questions 5
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_TENET_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_TENET_SRC) not in sys.path:
    sys.path.insert(0, str(_TENET_SRC))

from tenet.core import Tenet  # noqa: E402

# ----------------------------------------------------------------------------
# Trajectory -> text chunking (structure-aware, zero-LLM). The accessibility
# tree is the text-reachable observation; it is huge (~1.1M chars/trajectory)
# and mostly boilerplate, so we (a) keep the high-signal thought+action lines
# whole, and (b) chunk a *trimmed* per-state tree excerpt, letting Tenet's
# surprise-gate drop near-duplicate DOM slices at store time.
# ----------------------------------------------------------------------------

DEFAULTS = dict(
    k=12,               # context items returned per query
    expand=24,          # belief-anchored evidence expansion slots
    hops=2,             # associative recall rounds
    char_budget=12000,  # cap total returned context chars (well under 200k-tok reader cap)
    tree_chars=1200,    # per-state accessibility-tree excerpt cap (0 = drop trees)
    slice_chars=1400,   # target chunk size
    surprise_gate=0.97, # drop raw slices this cosine-similar to an existing one
    embed_batch=256,
)


def _url_path(url: str) -> str:
    """Compact a long WebArena/ServiceNow URL down to its path tail for the prefix."""
    if not url:
        return ""
    u = url.split("://", 1)[-1]
    return u[:90]


def trajectory_chunks(trajectory: dict[str, Any], *, tree_chars: int,
                      slice_chars: int) -> list[str]:
    """Emit step/url-prefixed text slices for one trajectory. Structure-aware:
    each state's thought+action stays attached to its step/url header (divorcing
    them is the #1 miss cause for procedure/dynamic questions)."""
    tid = trajectory.get("id", "?")
    goal = str(trajectory.get("goal") or "").strip()
    outcome = str(trajectory.get("outcome") or "").strip()
    env = str(trajectory.get("environment") or "").strip()
    header = f"[trajectory {tid} | env={env} | outcome={outcome}] GOAL: {goal}"

    out: list[str] = [header[: slice_chars * 2]]
    for st in trajectory.get("states", []) or []:
        if not isinstance(st, dict):
            continue
        step = st.get("step")
        url = _url_path(str(st.get("url") or ""))
        thought = str(st.get("thought") or "").strip()
        action = str(st.get("action") or "").strip()
        prefix = f"[{tid} step {step} | {url}]"
        # high-signal slice: thought + action, kept whole
        if thought or action:
            body = ""
            if action:
                body += f" ACTION: {action}"
            if thought:
                body += f" THOUGHT: {thought}"
            out.append((prefix + body)[: slice_chars * 3])
        # trimmed observation slice: the page state (accessibility tree)
        if tree_chars > 0:
            tree = str(st.get("accessibility_tree") or "").strip()
            if tree:
                out.append(f"{prefix} PAGE: {tree[:tree_chars]}")
    return [c for c in out if c.strip()]


# ----------------------------------------------------------------------------
# The registered backend. Import of memory_modules is deferred/optional so this
# file also runs standalone (mechanism smoke test) without the LME-V2 repo.
# ----------------------------------------------------------------------------

try:
    from memory_modules.memory import (  # type: ignore
        Memory, MemoryContextItem, register_memory, require,
    )

    @register_memory
    class TenetMemory(Memory):
        """Tenet as an LME-V2 memory backend. Zero-LLM insert and query."""

        memory_type = "tenet"

        def __init__(self, memory_params: dict[str, object]) -> None:
            super().__init__(memory_params)
            p = {**DEFAULTS, **{k: v for k, v in memory_params.items() if v is not None}}
            self.p = p
            workspace_dir = memory_params.get("workspace_dir")
            db_path = (Path(str(workspace_dir)) / "tenet.db") if workspace_dir else None
            self._store = Tenet(db_path)
            self._inserted: set[str] = set()

        def insert(self, trajectory: dict[str, object]) -> None:
            tid = str(trajectory.get("id"))
            if tid in self._inserted:
                return
            self._inserted.add(tid)
            chunks = trajectory_chunks(
                trajectory,
                tree_chars=int(self.p["tree_chars"]),
                slice_chars=int(self.p["slice_chars"]),
            )
            core = self._store.core
            B = int(self.p["embed_batch"])
            for i in range(0, len(chunks), B):
                batch = chunks[i:i + B]
                vecs = core.embed_batch(batch)
                for text, vec in zip(batch, vecs):
                    core.store(text, kind="raw", salience=0.4, source=tid,
                               surprise_gate=float(self.p["surprise_gate"]), _vec=vec)

        def query(self, query: str, query_image: str | None = None
                  ) -> list["MemoryContextItem"]:
            hits = self._store.recall(
                query,
                k=int(self.p["k"]),
                expand=int(self.p["expand"]),
                hops=int(self.p["hops"]),
                char_budget=int(self.p["char_budget"]),
            )
            items: list[MemoryContextItem] = []
            for h in hits:
                text = (h.text or "").strip()
                if text:
                    items.append({"type": "text", "value": text})
            if not items:  # harness requires non-empty text values; never return []
                items.append({"type": "text", "value": "(no relevant memory found)"})
            return items

        def _save_backend(self, output_dir: Path) -> None:
            return None

    _HAS_HARNESS = True
except ModuleNotFoundError:
    _HAS_HARNESS = False


# ----------------------------------------------------------------------------
# Standalone mechanism smoke test: build a store from a few real trajectories,
# time recall on real questions, and report whether the gold-answer string
# surfaces in retrieved context. No reader, no judge, no paid API.
# ----------------------------------------------------------------------------

def _smoke(data_root: Path, domain: str, n_traj: int, n_q: int) -> None:
    import json

    questions = [json.loads(l) for l in (data_root / "questions.jsonl").read_text().splitlines() if l.strip()]
    questions = [q for q in questions if q.get("domain") == domain]
    haystack = json.loads((data_root / "haystacks" / "lme_v2_small.json").read_text())
    # the domain's shared 100-trajectory haystack
    want_ids: list[str] = []
    for q in questions:
        want_ids = haystack.get(q["id"], [])
        if want_ids:
            break
    subset = set(want_ids[:n_traj])

    print(f"[smoke] domain={domain} loading {len(subset)} of {len(want_ids)} haystack trajectories...")
    trajs: dict[str, dict] = {}
    tpath = data_root / "trajectories.jsonl"
    with tpath.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            # cheap id prefilter before full json parse
            if not any(tid in line[:80] for tid in subset - set(trajs)):
                continue
            row = json.loads(line)
            if row.get("id") in subset:
                trajs[row["id"]] = row
                if len(trajs) == len(subset):
                    break
    print(f"[smoke] loaded {len(trajs)} trajectories")

    import tempfile
    ws = Path(tempfile.mkdtemp())
    store = Tenet(ws / "tenet.db")
    t0 = time.time()
    n_chunks = 0
    for tid, tr in trajs.items():
        chunks = trajectory_chunks(tr, tree_chars=DEFAULTS["tree_chars"], slice_chars=DEFAULTS["slice_chars"])
        core = store.core
        for i in range(0, len(chunks), DEFAULTS["embed_batch"]):
            b = chunks[i:i + DEFAULTS["embed_batch"]]
            vecs = core.embed_batch(b)
            for text, vec in zip(b, vecs):
                core.store(text, kind="raw", salience=0.4, source=tid,
                           surprise_gate=DEFAULTS["surprise_gate"], _vec=vec)
        n_chunks += len(chunks)
    insert_s = time.time() - t0
    stored = store.stats().get("total", "?") if hasattr(store, "stats") else "?"
    print(f"[smoke] ingest: {n_chunks} chunks -> {stored} stored in {insert_s:.1f}s "
          f"({insert_s / max(1, len(trajs)):.1f}s/trajectory)")

    # recall latency + answer-in-context proxy on a few questions
    lat = []
    hits_gold = 0
    for q in questions[:n_q]:
        t0 = time.time()
        ctx = store.recall(q["question"], k=DEFAULTS["k"], expand=DEFAULTS["expand"],
                           hops=DEFAULTS["hops"], char_budget=DEFAULTS["char_budget"])
        dt = time.time() - t0
        lat.append(dt)
        blob = "\n".join(h.text for h in ctx).lower()
        gold = q.get("answer")
        gold_s = (gold[0] if isinstance(gold, list) else str(gold)).lower().strip()
        found = bool(gold_s) and gold_s in blob
        hits_gold += found
        print(f"[smoke] q={q['id'][:10]} type={q['question_type'][:16]:16s} "
              f"recall={dt*1000:6.1f}ms ctx={len(blob)}ch gold_in_ctx={found}")
    if lat:
        print(f"[smoke] recall latency: mean={sum(lat)/len(lat)*1000:.1f}ms "
              f"median={sorted(lat)[len(lat)//2]*1000:.1f}ms  "
              f"gold-in-context {hits_gold}/{n_q} (partial-haystack proxy only)")
    store.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--domain", default="enterprise", choices=["web", "enterprise"])
    ap.add_argument("--n-trajectories", type=int, default=8)
    ap.add_argument("--n-questions", type=int, default=5)
    args = ap.parse_args()
    _smoke(args.data_root, args.domain, args.n_trajectories, args.n_questions)
