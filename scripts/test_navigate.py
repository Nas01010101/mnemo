"""Deterministic, LLM-free, torch-free test for tenet.navigate.navigate().

Runs with NO network and NO torch: a hashed bag-of-words embedder is injected onto
the MemoryCore instance (overriding embed_batch, which both store() and recall() use),
giving fully controllable cosine geometry. This validates the *mechanism* — adaptive
associative descent + saturation stop — not the end-to-end LLM benchmark (see
scripts/bench_factcon.py + the spec for the FC-MH number-moving run).

Two falsifiable claims:
  1. MULTI-HOP REACH: a bridge fact that shares NO tokens with the query (so it is
     invisible to broad top-k recall) IS surfaced by navigate(), because an
     associative hop re-conditions the cue on an in-pool fact that DOES share a token
     with the bridge. Falsifier: navigate returns the same set as broad recall.
  2. EARLY STOP: a simple query whose evidence is fully in the broad pool makes
     navigate stop at hop 2 with a "saturated" trace entry (no over-fetch).
     Falsifier: navigate keeps descending to max_hops on a saturated query.

Run:  python scripts/test_navigate.py   (exit 0 = pass)
"""
import hashlib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tenet.memory import MemoryCore          # noqa: E402
from tenet.navigate import navigate          # noqa: E402

_D = 64


def _fake_embed(texts):
    """Torch-free, network-free, process-stable hashed bag-of-words -> unit vectors.
    Shared tokens -> high cosine, so we can hand-build the recall geometry."""
    out = []
    for t in texts:
        v = np.zeros(_D, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _D
            v[h] += 1.0
        n = float(np.linalg.norm(v)) or 1.0
        out.append((v / n).astype(np.float32))
    return out


def _core():
    tmp = Path(tempfile.mkdtemp()) / "nav_test.db"
    c = MemoryCore(tmp)
    c.embed_batch = _fake_embed          # both store() and recall() route through this
    return c


def test_multihop_reach():
    c = _core()
    # Query shares tokens {sport, play, steve, sax} with the anchor fact only.
    query = "which sport does steve sax play"
    # Anchor: matches the query AND contains the bridge token "baseball".
    c.store("steve sax plays the sport baseball", key="steve_sax::sport")
    # Bridge: shares "baseball" with the anchor but NOTHING with the query.
    bridge_id = c.store("baseball originated in the country america",
                        key="baseball::origin")
    # Distractors: each shares a query token so they outrank the bridge in broad recall.
    c.store("maria plays the sport tennis", key="maria::sport")
    c.store("the sport of chess needs no play area", key="chess::info")
    c.store("steve enjoys play with his dog", key="steve::hobby")
    c.store("sax is a musical instrument you play", key="sax::music")

    broad = c.recall(query, k=5)
    broad_ids = {m.id for m in broad}
    assert bridge_id not in broad_ids, "bridge should be INVISIBLE to broad recall"

    nav, trace = navigate(c, query, k=5, max_hops=3, tau_gain=0.10)
    nav_ids = {m.id for m in nav}
    assert bridge_id in nav_ids, (
        f"navigate must surface the bridge via an associative hop; trace={trace}")
    assert any(t.get("adopted") and t["hop"] > 1 for t in trace), \
        f"a deeper hop must have been adopted; trace={trace}"
    print("PASS multihop_reach: bridge reached at hop >1")
    print("      trace:", trace)
    c.close()


def test_early_stop():
    c = _core()
    query = "what is the capital of france"
    c.store("the capital of france is paris", key="france::capital")
    # Unrelated facts: share no tokens with the query, so deeper hops find nothing
    # relevant to adopt -> navigate must stop early.
    for i, txt in enumerate([
        "the moon orbits the earth", "water boils at temperature",
        "guitars have six strings", "penguins live in cold regions",
    ]):
        c.store(txt, key=f"misc::{i}")

    nav, trace = navigate(c, query, k=5, max_hops=4, tau_gain=0.15)
    assert trace[-1].get("stop") == "saturated", f"expected early saturation; trace={trace}"
    assert trace[-1]["hop"] <= 3, f"should stop well before max_hops=4; trace={trace}"
    assert any(m.text == "the capital of france is paris" for m in nav), \
        "the answer fact must still be present"
    print("PASS early_stop: stopped at hop", trace[-1]["hop"], "-", trace[-1]["stop"])
    print("      trace:", trace)
    c.close()


if __name__ == "__main__":
    test_multihop_reach()
    test_early_stop()
    print("\nALL PASS")
