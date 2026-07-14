#!/usr/bin/env python3
"""Verify README.zh-CN.md against README.md — deterministic, offline.

The translation was produced by an LLM (Claude Fable 5, Anthropic) and semantically
spot-checked by back-translation with Argos Translate (open-source NMT). This script
is the machine-checkable half: every number, model name, CLI command, and link target
in the English README must survive into the Chinese one. Numbers are where a hallucinated
translation does real damage, so they are compared as exact multisets.

Usage: python scripts/check_translation.py   (exit 0 = consistent)
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN = (ROOT / "README.md").read_text(encoding="utf-8")
ZH = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def numbers(text: str) -> Counter:
    """All numeric tokens (ints, decimals, percentages stripped of %)."""
    # strip code fences? no — numbers in code blocks must match too.
    # strip badge/shield URLs (encodings differ is impossible — they're identical), keep all.
    return Counter(re.findall(r"\d+(?:[.,]\d+)*", text))


# --- 1. numbers: zh must contain every number the English README contains -------
en_nums, zh_nums = numbers(EN), numbers(ZH)
missing = en_nums - zh_nums
# allow zh to have extras only from the translation-note footer (e.g. none expected)
check("every English number appears in the Chinese version (exact multiset ⊆)",
      not missing, f"missing: {dict(missing)}" if missing else f"{sum(en_nums.values())} numeric tokens matched")

# --- 2. load-bearing terms/models/commands present verbatim ---------------------
TERMS = [
    "qwen3.7-plus", "qwen3.6-flash", "text-embedding-v4", "gpt-4o", "gpt-4o-mini",
    "gpt-5.5", "Gemini-3.5-flash", "qwen2.5:7b", "tenet-distiller-1.5b-v2",
    "DASHSCOPE_API_KEY", "EMBED_PROVIDER=local", "LLM_PROVIDER=ollama",
    "pip install tenet-memory", "recall(as_of=", "get_all", "navigate",
    "bench_horizon", "bench_factcon", "SubEM", "recall@10", "MIT",
    "Mem0", "Zep", "Letta", "Graphiti", "Neo4j", "FalkorDB", "LangGraph", "MCP",
]
for t in TERMS:
    check(f"term preserved: {t}", t in ZH)

# --- 3. link targets: every relative link in EN exists in ZH --------------------
en_links = set(re.findall(r"\]\(((?:docs|paper|src|scripts|examples|LICENSE)[^)#]*)", EN))
zh_links = set(re.findall(r"\]\(((?:docs|paper|src|scripts|examples|LICENSE)[^)#]*)", ZH))
missing_links = en_links - zh_links
check("every relative link target survives", not missing_links,
      f"missing: {sorted(missing_links)}" if missing_links else f"{len(en_links)} targets")

# --- 4. link targets actually exist on disk (both files) ------------------------
dead = [l for l in sorted(zh_links) if not (ROOT / l).exists()]
check("no dead relative links in zh", not dead, f"dead: {dead}" if dead else "")

# --- 5. structure: same number of code fences and tables roughly ---------------
check("same count of code fences", EN.count("```") == ZH.count("```"),
      f"en={EN.count('```')} zh={ZH.count('```')}")

print()
if failures:
    print(f"TRANSLATION CHECK FAILED — {len(failures)} problem(s)")
    sys.exit(1)
print("TRANSLATION CHECK PASS ✅  (numbers, terms, links, structure consistent with README.md)")
