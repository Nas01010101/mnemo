"""Shared distiller harness — routes tenet.distill.distill() at any OpenAI-compatible
endpoint (Qwen cloud reference OR ollama on the RTX box) and scores the output.

We reuse the REAL production distiller (src/tenet/distill.py: _SYS prompt + Fact parsing
guards) by monkeypatching tenet.config.chat per candidate. That means every metric below
reflects the exact path the running memory system uses — not a re-implementation.

CPU-only: no torch. Safe to run on the Mac.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from openai import OpenAI  # noqa: E402
import tenet.config as tconfig  # noqa: E402
import tenet.distill as tdistill  # noqa: E402


# --------------------------------------------------------------------------
# Endpoint = one (base_url, model, api_key). Records the last raw completion so
# we can score JSON validity and the raw key=value pathology (pre-guard).
# --------------------------------------------------------------------------
@dataclass
class Endpoint:
    name: str
    base_url: str
    model: str
    api_key: str = "ollama"
    last_raw: str = ""
    _client: object = field(default=None, repr=False)

    def client(self):
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=120)
        return self._client

    def _chat(self, messages, *, qwen_default, max_tokens=512, temperature=0,
              json_mode=False, or_model=None):
        kw = dict(model=self.model, messages=messages,
                  max_tokens=max_tokens, temperature=temperature)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        try:
            r = self.client().chat.completions.create(**kw)
        except Exception:
            kw.pop("response_format", None)  # endpoint may reject json mode
            r = self.client().chat.completions.create(**kw)
        self.last_raw = r.choices[0].message.content or ""
        return self.last_raw

    def distill(self, text: str):
        """Run the REAL tenet distiller against this endpoint. Returns (facts, raw)."""
        tconfig.chat = self._chat            # monkeypatch: production distill uses this
        self.last_raw = ""
        facts = tdistill.distill(text)
        return facts, self.last_raw


def ollama_endpoint(model: str, host: str = None) -> Endpoint:
    host = host or os.environ.get("BOX_OLLAMA", "http://100.88.179.78:11434")
    return Endpoint(name=model, base_url=host.rstrip("/") + "/v1", model=model)


def qwen_endpoint(model: str = "qwen3.7-plus") -> Endpoint:
    tconfig._load_env()
    return Endpoint(name=f"qwen/{model}",
                    base_url=os.environ["QWEN_BASE_URL"],
                    model=model, api_key=os.environ["DASHSCOPE_API_KEY"])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def raw_json_valid(raw: str) -> bool:
    """Does the raw completion parse as the expected JSON envelope?"""
    try:
        data = json.loads(raw)
    except Exception:
        return False
    if isinstance(data, list):
        return True  # bare facts list — distill.py tolerates it
    return isinstance(data, dict) and "facts" in data


_KV_RE = re.compile(r"::[a-z0-9_ ]+\s*=", re.I)


def raw_keyvalue_pathology(raw: str) -> bool:
    """True if any statement in the raw output is the 'key::attr=value' failure mode."""
    try:
        data = json.loads(raw)
    except Exception:
        return False
    facts = data.get("facts", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    for f in facts:
        if isinstance(f, dict):
            st = f.get("statement", "")
            k = (f.get("key") or "")
            if "=" in st and (_KV_RE.search(st) or (k and st.lower().replace(" ", "").startswith(k.lower().replace(" ", "")))):
                return True
    return False


def stmt_match(a: str, b: str, thresh: float = 0.6) -> bool:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio() >= thresh


def prf(cand_stmts: list[str], ref_stmts: list[str], thresh: float = 0.6):
    """Greedy fuzzy statement matching → precision, recall, f1."""
    if not ref_stmts and not cand_stmts:
        return 1.0, 1.0, 1.0
    used = [False] * len(ref_stmts)
    matched = 0
    for c in cand_stmts:
        for i, r in enumerate(ref_stmts):
            if not used[i] and stmt_match(c, r, thresh):
                used[i] = True
                matched += 1
                break
    p = matched / len(cand_stmts) if cand_stmts else (1.0 if not ref_stmts else 0.0)
    r = matched / len(ref_stmts) if ref_stmts else (1.0 if not cand_stmts else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1
