"""Deterministic tests for usage-scenario retrieval (usage_recall.py) — the OPTIONAL
ReMe-style (arXiv:2512.10696, Tongyi Lab) recall knob that RRF-fuses content-similarity
ranking with a second ranking over each fact's distiller-tagged usage-scenario
embedding ("when would this be useful to retrieve").

Part 1: MemoryCore.store()/recall() directly, synthetic vectors (no embedder network
call — same `_vec`/`rng.standard_normal` pattern as scripts/test_raw_recall.py), plus
the new `_scenario_vec` test-only override (mirrors `_vec`) so the scenario arm is
deterministic too.

Part 2: distill()'s extended JSON schema (the `scenario` field), with the LLM call
STUBBED (same `config.chat_client` monkeypatch pattern as scripts/test_retract.py —
no network).

Run: EMBED_PROVIDER=local python scripts/test_usage_recall.py
"""
import os
os.environ.setdefault("EMBED_PROVIDER", "local")  # before importing config

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
from tenet.distill import distill  # noqa: E402
from tenet.memory import MemoryCore  # noqa: E402
from tenet.usage_recall import rrf_merge  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ---- Part 1: MemoryCore, synthetic vectors, no embedder network call --------

def test_default_off_byte_identical():
    """usage_recall unset/False -> recall() ranking is untouched (no scenario
    embeddings are even fetched), byte-identical to before this knob existed."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "usage_off.db")
    rng = np.random.default_rng(1)
    qv = _unit(rng.standard_normal(384).astype(np.float32))
    off_topic_scen = _unit(rng.standard_normal(384).astype(np.float32))
    # Store WITH a scenario (usage_recall=True at write time) so scenario data
    # exists in the db — the read-time flag is what's under test here.
    core.store("fact A", key="k::a", _vec=_unit(rng.standard_normal(384).astype(np.float32)),
               scenario="irrelevant scenario", usage_recall=True, _scenario_vec=off_topic_scen)
    core.store("fact B", key="k::b", _vec=qv)

    default = core.recall("probe", k=5)
    explicit_off = core.recall("probe", k=5, usage_recall=False)
    check("usage_recall unset == usage_recall=False (same texts, same order)",
          [h.text for h in default] == [h.text for h in explicit_off])
    core.close()


def test_usage_recall_surfaces_scenario_match():
    """A fact with POOR content similarity but a usage-scenario embedding that
    closely matches the query should out-rank, via RRF, a fact with better content
    similarity but no scenario at all — the whole point of a second retrieval
    signal. Content-only ranking (usage_recall=False) must show the OPPOSITE order
    first, so this is a genuine re-rank, not a coincidence of the synthetic vectors."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "usage_on.db")
    rng = np.random.default_rng(2)
    qv = _unit(rng.standard_normal(384).astype(np.float32))

    # fact_content: high content similarity to qv, no scenario.
    content_vec = _unit(qv * 0.95 + rng.standard_normal(384).astype(np.float32) * 0.05)
    core.store("fact_content: paris weather is mild", key="k::content", _vec=content_vec)

    # fact_scenario: LOW content similarity to qv (near-orthogonal random vec), but
    # its usage-scenario embedding is a close match to qv.
    far_vec = _unit(rng.standard_normal(384).astype(np.float32))
    scen_vec = _unit(qv * 0.95 + rng.standard_normal(384).astype(np.float32) * 0.05)
    core.store("fact_scenario: user is allergic to shellfish", key="k::scenario", _vec=far_vec,
               scenario="when recommending restaurants or food", usage_recall=True,
               _scenario_vec=scen_vec)

    # A few filler facts between them in content rank, to make this a non-trivial pool.
    for i in range(3):
        v = _unit(rng.standard_normal(384).astype(np.float32) * 0.5 + qv * 0.3)
        core.store(f"filler {i}", key=f"k::filler{i}", _vec=v)

    baseline = core.recall("probe", k=5, usage_recall=False)
    b_texts = [h.text for h in baseline]
    check("content-only ranking puts fact_content ahead of fact_scenario (sanity check)",
          b_texts.index("fact_content: paris weather is mild")
          < b_texts.index("fact_scenario: user is allergic to shellfish"),
          f"order={b_texts}")

    fused = core.recall("probe", k=5, usage_recall=True)
    f_texts = [h.text for h in fused]
    check("usage_recall=True: fact_scenario now ranks ABOVE fact_content (RRF fusion)",
          f_texts.index("fact_scenario: user is allergic to shellfish")
          < f_texts.index("fact_content: paris weather is mild"),
          f"order={f_texts}")
    core.close()


def test_no_scenario_embeddings_stored_is_noop():
    """If nothing in the store has a usage-scenario embedding (the knob was never
    used at write time), usage_recall=True at read time changes nothing — there is
    no second ranking to fuse."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "usage_noop.db")
    rng = np.random.default_rng(3)
    qv = _unit(rng.standard_normal(384).astype(np.float32))
    for i in range(4):
        v = _unit(rng.standard_normal(384).astype(np.float32) * 0.5 + qv * 0.3)
        core.store(f"plain fact {i}", key=f"k::{i}", _vec=v)

    off = core.recall("probe", k=4, usage_recall=False)
    on = core.recall("probe", k=4, usage_recall=True)
    check("usage_recall=True with zero scenario embeddings in the store is a no-op",
          [h.text for h in off] == [h.text for h in on])
    core.close()


def test_store_time_flag_gates_the_embedding_write():
    """store(scenario=..., usage_recall=False) drops the scenario — no embedding
    call, nothing stored — exactly like a retract-tagged fact is stored plainly
    when TENET_RETRACT is off (docs/COMPARISON.md follow-up #3 precedent)."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "usage_gate.db")
    rng = np.random.default_rng(4)
    v = _unit(rng.standard_normal(384).astype(np.float32))
    mid = core.store("fact with an ignored scenario", key="k::x", _vec=v,
                      scenario="should be dropped", usage_recall=False)
    row = core.db.execute(
        "SELECT scenario_text, scenario_embedding FROM memories WHERE id=?", (mid,)
    ).fetchone()
    check("usage_recall=False at store time: scenario_text/embedding stay NULL",
          row["scenario_text"] is None and row["scenario_embedding"] is None)
    core.close()


def test_rrf_merge_pure_function():
    """rrf_merge() in isolation: a row present in BOTH rankings must score higher
    than the same row would with the content-only arm alone, and a row absent from
    scenario_sims contributes only its content-arm term (never crashes/KeyErrors)."""
    rows = [{"id": i} for i in range(3)]
    scored = [(0.9 - 0.1 * i, 0.9 - 0.1 * i, rows[i]) for i in range(3)]  # id 0 best content
    # ONLY id 2 (worst content rank) has a usage-scenario embedding at all — id 0/1
    # have none, so their combined score is content-arm only.
    merged = rrf_merge(scored, {2: 0.99})
    order = [row["id"] for _s, _r, row in merged]
    check("rrf_merge: id 2 (worst content, only one with a scenario match) moves to the front",
          order[0] == 2, f"order={order}")
    check("rrf_merge: empty scenario_sims returns the original list unchanged",
          rrf_merge(scored, {}) == scored)


# ---- Part 2: distill()'s extended JSON schema, LLM call STUBBED -------------

class _FakeResp:
    def __init__(self, content):
        msg = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": msg})()
        self.choices = [choice]


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)

    def create(self, **_kw):
        return _FakeResp(self.script.pop(0))


class _FakeClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(script)})()


def _stub_distill(*json_strings):
    config.chat_client = lambda: _FakeClient(list(json_strings))


def test_distill_parses_scenario_field():
    orig_client = config.chat_client
    try:
        _stub_distill(json.dumps({"facts": [
            {"statement": "user is allergic to shellfish", "key": "user::allergy",
             "salience": 0.9, "valid_at": None, "action": "remember",
             "scenario": "when recommending restaurants or food"},
        ]}))
        facts = distill("I should mention I'm allergic to shellfish")
        check("distill() parses one fact", len(facts) == 1, f"got {facts}")
        check("distill() carries the scenario field through",
              facts[0].scenario == "when recommending restaurants or food",
              f"scenario={facts[0].scenario!r}")
    finally:
        config.chat_client = orig_client


def test_distill_scenario_defaults_empty_when_omitted():
    """Older/weaker-model output with no `scenario` key at all must not crash —
    Fact.scenario defaults to ''."""
    orig_client = config.chat_client
    try:
        _stub_distill(json.dumps({"facts": [
            {"statement": "user lives in Denver", "key": "user::residence",
             "salience": 0.8, "valid_at": None},
        ]}))
        facts = distill("I live in Denver")
        check("distill() defaults scenario to '' when the model omits it",
              len(facts) == 1 and facts[0].scenario == "", f"got {facts}")
    finally:
        config.chat_client = orig_client


def main() -> int:
    test_default_off_byte_identical()
    test_usage_recall_surfaces_scenario_match()
    test_no_scenario_embeddings_stored_is_noop()
    test_store_time_flag_gates_the_embedding_write()
    test_rrf_merge_pure_function()
    test_distill_parses_scenario_field()
    test_distill_scenario_defaults_empty_when_omitted()
    if FAILS:
        print("\nFAILURES:", FAILS)
        return 1
    print("\nALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
