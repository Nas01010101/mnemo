"""Tenet HTTP API — the deployable backend (runs on Alibaba Cloud).

This is the surface judges can `curl`, and the process shown running on Alibaba
Cloud for the mandatory proof-of-deployment. Same MemoryCore as the MCP server.

Also serves the belief-state demo UI (GET /) — a single static page that shows
supersession happening live: chat on the left, the belief state (current facts,
struck-through history, a time-travel slider) on the right.

Run locally:  uvicorn tenet.api:app --host 0.0.0.0 --port 8000  (needs the `api` extra)
"""
from __future__ import annotations

import secrets
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .agent import MemoryAgent

app = FastAPI(
    title="Tenet — Self-Managing Memory API",
    description="Persistent, self-forgetting, bi-temporal memory for LLM agents, powered by Qwen Cloud.",
    version="0.2.0",
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_SESSION_COOKIE = "tenet_sid"
# Demo (/reset) sessions get their own throwaway db file, so concurrent judges
# never see each other's beliefs; the unkeyed "default" session keeps using the
# normal data/tenet.db (backward compat for the CLI, curl, scripts/*).
_SESSION_DIR = Path(tempfile.gettempdir()) / "tenet-demo-sessions"
_SESSION_DIR.mkdir(parents=True, exist_ok=True)


class _Session:
    """One isolated memory store: a MemoryAgent (chat) over one Tenet (facts)."""

    def __init__(self, db_path: Path | None = None):
        self.agent = MemoryAgent(db_path) if db_path else MemoryAgent()
        self.tenet = self.agent.m


_sessions: dict[str, _Session] = {"default": _Session()}


def _resolve_session(
    response: Response,
    session: str | None = Query(None, description="explicit session id (testing/curl)"),
    tenet_sid: str | None = Cookie(None),
) -> _Session:
    sid = session or tenet_sid or "default"
    if sid not in _sessions:
        _sessions[sid] = _Session(_SESSION_DIR / f"{sid}.db")
    if sid != "default":
        response.set_cookie(_SESSION_COOKIE, sid, httponly=True, samesite="lax", max_age=86400)
    return _sessions[sid]


def _parse_as_of(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return float(raw)  # epoch seconds — what the time-travel slider sends
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        raise HTTPException(422, f"invalid as_of: {raw!r} (expected epoch seconds or ISO datetime)")


class ChatReq(BaseModel):
    message: str = Field(..., min_length=1)


class IngestReq(BaseModel):
    message: str = Field(..., min_length=1)
    pinned: bool = False


class StoreReq(BaseModel):
    text: str = Field(..., min_length=1)
    pinned: bool = False


class RecallReq(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    char_budget: int | None = Field(None, ge=1)


@app.get("/")
def index():
    """The belief-state demo page — single static file, no build step."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health(sess: _Session = Depends(_resolve_session)):
    return {
        "status": "ok",
        "provider": config.LLM_PROVIDER,
        "embed_provider": config.EMBED_PROVIDER,
        **sess.tenet.stats(),
    }


@app.get("/state")
def state(as_of: str | None = None, sess: _Session = Depends(_resolve_session)):
    """Belief state for the demo UI: facts (current + superseded history) grouped
    by key. Pass as_of=<epoch|iso> to time-travel — what was true at that instant."""
    ts = _parse_as_of(as_of)
    return {
        "beliefs": sess.tenet.core.list_beliefs(ts),
        "stats": sess.tenet.stats(),
        "provider": config.LLM_PROVIDER,
        "as_of": ts,
    }


@app.post("/reset")
def reset(response: Response, tenet_sid: str | None = Cookie(None)):
    """Demo-only: hand this caller a fresh, isolated session (own throwaway db)
    so each judge gets a clean slate, even running concurrently."""
    if tenet_sid and tenet_sid in _sessions and tenet_sid != "default":
        old = _sessions.pop(tenet_sid)
        old.tenet.close()
        (_SESSION_DIR / f"{tenet_sid}.db").unlink(missing_ok=True)
    sid = secrets.token_hex(8)
    sess = _Session(_SESSION_DIR / f"{sid}.db")
    _sessions[sid] = sess
    response.set_cookie(_SESSION_COOKIE, sid, httponly=True, samesite="lax", max_age=86400)
    return {"session": sid, **sess.tenet.stats()}


@app.post("/chat")
def chat(req: ChatReq, sess: _Session = Depends(_resolve_session)):
    """The Tenet Assistant: recall relevant memory → answer with Qwen → learn from the
    message (with supersession). A persistent, self-managing memory agent over HTTP."""
    out = sess.agent.respond(req.message)
    return {**out, "facts_added": out["learned"]}


@app.post("/ingest")
def ingest(req: IngestReq, sess: _Session = Depends(_resolve_session)):
    """Distill a raw message into atomic facts and store them (with supersession)."""
    ids = sess.tenet.ingest(req.message, pinned=req.pinned)
    return {"stored": len(ids), "ids": ids}


@app.post("/memories")
def store(req: StoreReq, sess: _Session = Depends(_resolve_session)):
    try:
        mem_id = sess.tenet.core.store(req.text, pinned=req.pinned)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"id": mem_id, "pinned": req.pinned}


@app.post("/recall")
def recall(req: RecallReq, sess: _Session = Depends(_resolve_session)):
    hits = sess.tenet.core.recall(req.query, k=req.k, char_budget=req.char_budget)
    return {
        "query": req.query,
        "results": [
            {"id": m.id, "text": m.text, "score": m.score, "pinned": m.pinned}
            for m in hits
        ],
    }


@app.post("/forget")
def forget(sess: _Session = Depends(_resolve_session)):
    n = sess.tenet.core.forget_sweep()
    return {"archived": n, **sess.tenet.core.stats()}
