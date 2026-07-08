"""Write-time distillation (Tenet Upgrade B).

Turns a raw conversation turn / note into atomic, self-contained facts with a
semantic key for supersession, a salience score for forgetting, and an optional
event time. This is the Mem0 "extract salient facts on write" idea — and it's
what makes bi-temporal supersession reliable (embedding similarity alone can't
tell a value-change from a restatement; a stable `subject::attribute` key can).

One cheap LLM call per ingested message (qwen3.6-flash by default).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import config

_MODEL = config.get("QWEN_DISTILL_MODEL", "qwen3.6-flash")

_SYS = """You extract durable, atomic facts from a message for an agent's long-term memory.
Return STRICT JSON: {"facts": [{"statement","key","salience","valid_at"}...]}.

Rules:
- statement: one self-contained fact. Resolve pronouns to names. No fluff.
  PRESERVE specific values VERBATIM — numbers, dates, times, durations, quantities,
  prices, proper nouns (e.g. keep "2 days", "March 3 at 14:20", "$50", "gate B12").
  Never generalize a specific away; those exact details are what gets asked about.
- key: a stable "subject::attribute" slug (lowercase, snake_case), e.g.
  "user::residence", "user::coffee_pref", "project_nimbus::ship_date". The SAME
  real-world attribute must always get the SAME key so later updates supersede it.
  CRITICAL: the account owner / first-person speaker ("I", "me", "my", and any name
  they give for themselves) is ALWAYS the subject `user` — never their proper name.
  So "I live in X", "I moved to Y", "My name is Z" all use subject `user`
  (keys user::residence, user::residence, user::name). This keeps updates on the
  same attribute colliding on one key so later values supersede earlier ones.
- salience: 0.0-1.0. Durable/identity/preference/commitment facts are high (0.7-1.0);
  transient small talk is low (0.0-0.3). Skip pure chit-chat entirely.
- valid_at: an ISO-8601 date/time if the fact states when it becomes true, else null.
- Extract nothing (empty list) if there is no durable fact worth remembering.
Return ONLY the JSON object."""


@dataclass
class Fact:
    statement: str
    key: str
    salience: float
    valid_at_iso: str | None


def distill(text: str, *, model: str = _MODEL, client=None) -> list[Fact]:
    raw = config.chat(
        [{"role": "system", "content": _SYS}, {"role": "user", "content": text}],
        qwen_default=model, max_tokens=800, temperature=0, json_mode=True,
    ) or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):  # weaker models sometimes return the bare facts list
        data = {"facts": data}  # instead of the {"facts": [...]} envelope — tolerate it
    elif not isinstance(data, dict):
        return []
    out: list[Fact] = []
    for f in data.get("facts", []):
        stmt = (f.get("statement") or "").strip()
        key = (f.get("key") or "").strip().lower() or None
        if not stmt or not key:
            continue
        try:
            sal = float(f.get("salience", 0.5))
        except (TypeError, ValueError):
            sal = 0.5
        out.append(Fact(stmt, key, max(0.0, min(1.0, sal)), f.get("valid_at") or None))
    return out
