"""Central, fail-loud config loader for qwen-hackathon.

Reads the gitignored `.env`. Nothing else in the codebase should read os.environ
for secrets directly — import from here so a missing key fails once, clearly.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ENV = _ROOT / ".env"


def _load_env(path: Path = _ENV) -> None:
    """Minimal .env loader (no dependency). Real env vars win over the file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_env()


def require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val or val.startswith("sk-xxxx") or val.startswith("<"):
        raise RuntimeError(
            f"Missing/placeholder secret: {name}. Set it in {_ENV} (chmod 600)."
        )
    return val


def get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip() or default


# Convenience accessors
QWEN_BASE_URL = get("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = get("QWEN_MODEL", "qwen3.7-plus")
QWEN_EMBED_MODEL = get("QWEN_EMBED_MODEL", "text-embedding-v4")


def qwen_client():
    """Return an OpenAI SDK client pointed at Qwen Cloud."""
    from openai import OpenAI  # lazy import so config loads without the dep

    return OpenAI(api_key=require("DASHSCOPE_API_KEY"), base_url=QWEN_BASE_URL)
