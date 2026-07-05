"""Mnemo HTTP API — the deployable backend (runs on Alibaba Cloud).

This is the surface judges can `curl`, and the process shown running on Alibaba
Cloud for the mandatory proof-of-deployment. Same MemoryCore as the MCP server.

Run locally:  uvicorn api:app --host 0.0.0.0 --port 8000  (from src/)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memory import MemoryCore

app = FastAPI(
    title="Mnemo — Self-Managing Memory API",
    description="Persistent, self-forgetting memory for LLM agents, powered by Qwen Cloud.",
    version="0.1.0",
)
_core = MemoryCore()


class StoreReq(BaseModel):
    text: str = Field(..., min_length=1)
    pinned: bool = False


class RecallReq(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    char_budget: int | None = Field(None, ge=1)


@app.get("/health")
def health():
    return {"status": "ok", **_core.stats()}


@app.post("/memories")
def store(req: StoreReq):
    try:
        mem_id = _core.store(req.text, pinned=req.pinned)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"id": mem_id, "pinned": req.pinned}


@app.post("/recall")
def recall(req: RecallReq):
    hits = _core.recall(req.query, k=req.k, char_budget=req.char_budget)
    return {
        "query": req.query,
        "results": [
            {"id": m.id, "text": m.text, "score": m.score, "pinned": m.pinned}
            for m in hits
        ],
    }


@app.post("/forget")
def forget():
    n = _core.forget_sweep()
    return {"archived": n, **_core.stats()}
