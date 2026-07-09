"""Run the tenet unit tests + an LLM-free churn-integrity guard on the RTX box with a
LOCAL bge-small embedder (transformers CLS pooling — no paid API, no sentence-transformers
install). Verifies the memory.py TENET_DYNAMICS hook did NOT regress default behaviour.

  PYTHONPATH=src python scripts/run_tests_rtx.py --npz data/dynamics_neural.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def install_local_embedder(model_name="BAAI/bge-small-en-v1.5", device="cuda"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    cache = {}

    def embed_texts(texts):
        out, todo = [None] * len(texts), []
        for i, t in enumerate(texts):
            if t in cache:
                out[i] = cache[t]
            else:
                todo.append((i, t))
        if todo:
            with torch.no_grad():
                for j in range(0, len(todo), 128):
                    chunk = todo[j:j + 128]
                    enc = tok([t for _, t in chunk], padding=True, truncation=True,
                              max_length=256, return_tensors="pt").to(device)
                    h = mdl(**enc).last_hidden_state[:, 0]
                    h = torch.nn.functional.normalize(h, dim=-1).cpu().numpy().astype(np.float32)
                    for (i, t), v in zip(chunk, h):
                        cache[t] = v; out[i] = v
        return out

    import tenet.config as cfg
    cfg.embed_texts = embed_texts
    import tenet.memory as memory
    memory.config.embed_texts = embed_texts   # already-imported reference
    return embed_texts


def churn_guard(npz_path):
    """LLM-free churn regression: a key updated 12× must still recall its LATEST value
    FIRST — under BOTH the default closed-form dynamics AND the neural world model. This
    is the essence of bench_horizon (lesson 452ae5fd: confidence is annotation-only; a
    doubted current fact must NOT be rank-demoted behind stale distractors)."""
    import time
    import tempfile
    import tenet.memory as M
    DAY = 86400.0
    results = []
    for mode in ("default", "neural"):
        if mode == "neural":
            os.environ["TENET_DYNAMICS"] = "neural"
            os.environ["TENET_NEURAL_NPZ"] = str(npz_path)
        else:
            os.environ.pop("TENET_DYNAMICS", None)
        clock = {"t": 1_000_000_000.0}
        core = M.MemoryCore(Path(tempfile.mkdtemp()) / f"churn_{mode}.db", now=lambda: clock["t"])
        cities = ["Boston", "Seattle", "Austin", "Denver", "Chicago", "Miami",
                  "Portland", "Dallas", "Phoenix", "Newark", "Reno", "Tampa"]
        for c in cities:
            clock["t"] += 20 * DAY
            core.store(f"lives in {c}", key="user::residence")
        got = core.recall("where does the user live?", k=5)
        cur = [m for m in got if m.key == "user::residence" and m.is_current]
        top_is_latest = bool(got) and got[0].key == "user::residence" and "Tampa" in got[0].text
        one_current = len(cur) == 1 and "Tampa" in cur[0].text
        results.append((mode, top_is_latest, one_current,
                        got[0].text if got else None,
                        cur[0].confidence if cur else None))
        core.close()
    os.environ.pop("TENET_DYNAMICS", None)
    os.environ.pop("TENET_NEURAL_NPZ", None)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/dynamics_neural.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("=== installing local bge-small embedder (transformers CLS) ===", flush=True)
    install_local_embedder(device=args.device)

    fails = 0
    print("\n=== test_dynamics.py ===", flush=True)
    import test_dynamics
    try:
        test_dynamics.main()
    except SystemExit as e:
        if e.code:
            fails += 1

    print("\n=== test_memory.py ===", flush=True)
    import importlib
    import test_memory
    importlib.reload(test_memory)
    rc = test_memory.main()
    if rc:
        fails += 1

    print("\n=== churn-integrity guard (LLM-free; default + neural) ===", flush=True)
    for mode, top_latest, one_cur, top_text, conf in churn_guard(args.npz):
        ok = top_latest and one_cur
        print(f"  [{mode}] top-is-latest={top_latest} one-current={one_cur} "
              f"top={top_text!r} conf={conf}  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails += 1

    print(f"\n{'ALL GREEN' if fails == 0 else str(fails)+' FAILURES'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
