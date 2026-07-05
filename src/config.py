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


# ---------------------------------------------------------------------------
# Provider abstraction — the SHIPPED product uses Qwen Cloud (defaults below).
# For off-Qwen validation (e.g. when the Qwen quota is exhausted) set in .env:
#   LLM_PROVIDER=openrouter   EMBED_PROVIDER=local
# Chat providers: qwen (default) | openrouter | agy (Gemini via ~/.local/bin/agy).
# Embed providers: qwen (default) | local (sentence-transformers, e.g. bge-small).
# ---------------------------------------------------------------------------
LLM_PROVIDER = get("LLM_PROVIDER", "qwen")
EMBED_PROVIDER = get("EMBED_PROVIDER", "qwen")
OPENROUTER_MODEL = get("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")
LOCAL_EMBED_MODEL = get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

_or_client = None
_local_embedder = None


def chat_client():
    """OpenAI-compatible chat client for the active provider (qwen | openrouter)."""
    global _or_client
    if LLM_PROVIDER == "openrouter":
        if _or_client is None:
            from openai import OpenAI
            _or_client = OpenAI(api_key=require("OPENROUTER_API_KEY"),
                                base_url=get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
        return _or_client
    return qwen_client()


def chat_model(qwen_default: str) -> str:
    """Model id for the active provider; OpenRouter uses one configured model."""
    return OPENROUTER_MODEL if LLM_PROVIDER == "openrouter" else qwen_default


def chat(messages, *, qwen_default: str, max_tokens: int = 512, temperature: float = 0,
         json_mode: bool = False) -> str:
    """Unified chat completion returning text. Routes qwen/openrouter (OpenAI SDK) or
    agy (Gemini CLI, off-Claude-plan)."""
    if LLM_PROVIDER == "agy":
        return _agy_chat(messages)
    import time as _t
    kw = {"model": chat_model(qwen_default), "messages": messages,
          "temperature": temperature, "max_tokens": max_tokens}
    if LLM_PROVIDER == "qwen":
        kw["extra_body"] = {"enable_thinking": False}
        if json_mode:  # Qwen supports it reliably; OpenRouter provider support varies,
            kw["response_format"] = {"type": "json_object"}  # so we rely on the prompt there
    client = chat_client()
    for attempt in range(5):
        try:
            r = client.chat.completions.create(**kw)
            if r.choices:
                return (r.choices[0].message.content or "").strip()
            kw.pop("response_format", None)  # empty -> drop json constraint, retry
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "response_format" in msg or "json_object" in msg:
                kw.pop("response_format", None)  # provider rejects json mode
            elif "429" in msg or "rate" in msg.lower() or "temporarily" in msg.lower():
                _t.sleep(2 * (attempt + 1))      # transient rate limit — back off
            elif attempt >= 3:
                return ""
            else:
                _t.sleep(1)
    return ""


def _agy_chat(messages) -> str:
    """Route a chat turn through agy (Gemini Pro sub, zero Claude tokens)."""
    import subprocess
    prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
    agy = os.path.expanduser("~/.local/bin/agy")
    out = subprocess.run([agy, "ask", prompt], capture_output=True, text=True, timeout=120)
    return out.stdout.strip()


def embed_texts(texts: list[str]):
    """Unit-normalised embeddings for the active provider (qwen | local)."""
    import numpy as np
    if EMBED_PROVIDER == "local":
        global _local_embedder
        if _local_embedder is None:
            from sentence_transformers import SentenceTransformer
            _local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
        vecs = _local_embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [np.asarray(v, dtype=np.float32) for v in vecs]
    # Qwen/DashScope — batched (cap 10/call), truncate long inputs, per-item fallback
    client = qwen_client()
    clipped = [t[:6000] for t in texts]
    out = []
    for i in range(0, len(clipped), 10):
        chunk = clipped[i:i + 10]
        try:
            data = client.embeddings.create(model=QWEN_EMBED_MODEL, input=chunk).data
            vecs = [d.embedding for d in data]
        except Exception:
            vecs = []
            for one in chunk:
                try:
                    vecs.append(client.embeddings.create(model=QWEN_EMBED_MODEL, input=one).data[0].embedding)
                except Exception:
                    vecs.append(None)
        for v in vecs:
            if v is None:
                out.append(np.zeros(384 if EMBED_PROVIDER == "local" else 1024, dtype=np.float32))
                continue
            a = np.asarray(v, dtype=np.float32)
            n = np.linalg.norm(a)
            out.append(a / n if n else a)
    return out
