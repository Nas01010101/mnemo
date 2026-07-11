"""Deterministic tests for embedding-based key resolution (memory.py
`_resolve_key_supersede` + `_value_compatible`), the fix for the 3.8% NL-update
supersession-firing rate (BENCHMARK.md §13 / scripts/bench_supersession_firing.py).

Uses local embeddings (EMBED_PROVIDER=local, bge-small) so it runs offline and
deterministically — no LLM, no network. Run: python scripts/test_key_resolution.py
"""
import os
os.environ.setdefault("EMBED_PROVIDER", "local")  # before importing config

import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet import memory as M  # noqa: E402
from tenet.memory import MemoryCore, _value_compatible  # noqa: E402


def _fresh():
    return MemoryCore(Path(tempfile.mkdtemp()) / "kr.db")


def main() -> int:
    fails = []

    # --- unit: value-compatibility guard --------------------------------
    assert _value_compatible("user::milk_preference", "user::milk"), "milk synonym should be compatible"
    assert _value_compatible("user::residence", "user::current_residence"), "meta-qualifier compatible"
    assert not _value_compatible("user::pet_name", "user::pet"), "pet_name is a sub-attribute of pet"
    assert not _value_compatible("user::car_color", "user::car"), "car_color is a sub-attribute of car"
    assert not _value_compatible("user::milk", "bob::milk"), "different subjects never compatible"

    # --- ON: synonym-drift keys collapse (the core fix) ------------------
    # Use the SHIPPED defaults (tau=0.78, text_floor=0.66) so the test guards the
    # actual default-on behaviour, including the hardened floor.
    M._KEY_RESOLUTION, M._TAU_KEY, M._TEXT_FLOOR = True, 0.78, 0.66
    c = _fresh()
    c.store("The user drinks oat milk.", key="user::milk")
    c.store("The user has switched to almond milk.", key="user::milk_preference")
    cur = [m for m in c.recall("what milk does the user drink", k=10) if m.kind != "raw"]
    txt = " ".join(m.text.lower() for m in cur)
    if "almond" not in txt:
        fails.append("resolution: latest value (almond) not current")
    if "oat" in txt:
        fails.append("resolution: stale value (oat) not superseded across variant keys")
    if c.stats()["superseded"] != 1:
        fails.append(f"resolution: expected 1 superseded, got {c.stats()['superseded']}")
    c.close()

    # --- guard: distinct sub-attributes must NOT collapse ----------------
    c = _fresh()
    c.store("The user has a golden retriever.", key="user::pet")
    c.store("The user's dog is named Max.", key="user::pet_name")
    cur = " ".join(m.text.lower() for m in c.recall("user pet", k=10) if m.kind != "raw")
    if "golden retriever" not in cur or "max" not in cur:
        fails.append("guard: pet=dog and pet_name=Rex were wrongly collapsed")
    if c.stats()["superseded"] != 0:
        fails.append(f"guard: expected 0 superseded for distinct sub-attrs, got {c.stats()['superseded']}")
    c.close()

    # --- guard: semantically unrelated attributes must NOT collapse ------
    c = _fresh()
    c.store("The user is a data scientist.", key="user::job_title")
    c.store("The user drinks oat milk.", key="user::milk")
    if c.stats()["superseded"] != 0:
        fails.append("guard: unrelated attributes (job vs milk) collapsed")
    c.close()

    # --- guard: ADVERSARIAL shared-salient-word pairs must NOT collapse ---
    # (a shared word like "probe"/"location"/"pet"/"coffee" inflates BOTH the key
    # embedding and the text embedding; the fact-text floor at 0.66 is what stops these.)
    adversarial = [
        (("The e2e probe fact is: X wrote this.", "user::surface_probe"),
         ("temporal probe v1", "user::temporal_probe")),
        (("The user's work is located in Denver.", "user::work_location"),
         ("The user's workout happens at the downtown gym.", "user::workout_location")),
        (("The user has a dog.", "user::pet"),
         ("The user's pet peeve is loud chewing.", "user::pet_peeve")),
        (("The user goes to Blue Bottle coffee shop.", "user::coffee_shop"),
         ("The user prefers oat milk lattes.", "user::coffee_preference")),
    ]
    for (t1, k1), (t2, k2) in adversarial:
        c = _fresh()
        c.store(t1, key=k1, pinned=True)
        c.store(t2, key=k2)
        if c.stats()["superseded"] != 0:
            fails.append(f"adversarial: {k1} wrongly superseded by {k2} (shared-word false-fire)")
        c.close()

    # --- OFF: exact-key only, variant keys do NOT collapse ---------------
    M._KEY_RESOLUTION = False
    c = _fresh()
    c.store("The user drinks oat milk.", key="user::milk")
    c.store("The user has switched to almond milk.", key="user::milk_preference")
    if c.stats()["superseded"] != 0:
        fails.append("flag OFF: resolution fired when disabled")
    cur = " ".join(m.text.lower() for m in c.recall("milk", k=10) if m.kind != "raw")
    if "oat" not in cur or "almond" not in cur:
        fails.append("flag OFF: both variant-key facts should remain current")
    c.close()
    M._KEY_RESOLUTION = True  # restore default

    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("test_key_resolution: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
