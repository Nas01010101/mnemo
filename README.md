# Mnemo — a personal AI assistant whose memory stays true as your life changes

Built for the [Global AI Hackathon with Qwen Cloud](https://qwencloud-hackathon.devpost.com) —
**Track 1: MemoryAgent**.

**The agent** ([`src/agent.py`](src/agent.py)): a warm personal assistant, powered by Qwen
Cloud, that remembers you across sessions — and, unlike a chatbot bolted onto RAG, **stays
correct when your facts change**. Tell it you moved cities or changed jobs and it updates;
ask what you *used* to say and it time-travels; it forgets stale trivia on its own.

**The engine** ([`src/mnemo.py`](src/mnemo.py)): a bi-temporal, self-forgetting memory that
**stores** what matters, **supersedes** facts when they change, **forgets** what goes
stale, and **recalls** under a context budget — **no LLM in the read path** — exposed over
MCP + HTTP so any agent can use it.

```
you › Hi! I'm Wissem, I live in Montreal and work as a data analyst.
assistant › Hi Wissem! Montreal's a great city. How's the analyst work going?   [remembered 2 facts]
… three weeks later …
you › Big news — I moved to Toronto and got promoted to senior analyst!
you › Where do I live and what's my job now?
assistant › You live in Toronto and you're a senior analyst. Congrats on the promotion!
```
Run it: `python src/agent.py` · or the scripted story: `python scripts/demo_agent.py`

![architecture](docs/architecture.svg)

## The result that matters
RAG-style memory re-reads raw history every query and **drowns as facts change**. Mnemo
maintains a compact, self-consistent *world-model of the user*. Measured on LongMemEval_S
(honest protocol in [`docs/BENCHMARK.md`](docs/BENCHMARK.md)):

- **Best accuracy-per-token** of any approach — matches most of RAG's answer quality on
  **half the context**, and **99% less** than full-context (41.2 vs 27.4 vs 0.5 acc/1k-tok).
- **Long-horizon dominance** — as one fact is updated 2→12 times, **RAG collapses 100%→50%
  while Mnemo holds 100%.** Supersession keeps exactly one current value; RAG's top-k
  drowns in stale versions.

  ![long-horizon](docs/horizon.svg)
- **Retrieval recall on par with strong RAG** (95% = 95%).
- Honest: for one-shot factual retrieval a strong RAG matches us; multi-hop temporal
  synthesis is our weak spot. We report it.

## Why it's different
Most memory systems are append-and-retrieve. The three things Track 1 explicitly asks
for — efficient retrieval, *timely forgetting*, and *recall under limited context* — are
exactly what append-only systems don't do. Mnemo is built around them:

| Capability | How |
|---|---|
| **Bi-temporal supersession** | facts carry event time (`valid_at`/`invalid_at`) **and** transaction time (`created_at`/`expired_at`); a changed fact **retires** the old value instead of overwriting it |
| **Time-travel** | `recall(as_of=…)` answers "what did I believe last March" from retained history |
| **Timely forgetting** | salience-weighted recency decay; a background sweep archives stale, low-value memories (pinned facts never forgotten) |
| **Recall under a budget** | `recall(char_budget=N)` fills to a token budget — recall under a limited context window |
| **Write-time distillation** | raw messages → atomic facts with a stable `subject::attribute` key so updates reliably supersede |
| **Hybrid index** | distilled facts (consistency/temporal) **+** raw slices (verbatim detail), dual-pool recall |
| **MCP-native** | plug persistent memory into Claude Desktop / any MCP client (`learn` · `recall` · `forget_stale` · `stats`) |

The read path uses pure vector + decay scoring — **no LLM call** — so retrieval is fast
and the memory is frontier-correct on the accuracy/latency axis (LongMemEval-V2's direction).

## Architecture
Two surfaces (MCP server + FastAPI) over one `MemoryCore`, backed by Qwen Cloud
(Alibaba Cloud Model Studio) for distillation + embeddings, with optional Alibaba Cloud
OSS snapshots for durability. See [`docs/architecture.svg`](docs/architecture.svg) and
[`docs/DESIGN.md`](docs/DESIGN.md).

## Quickstart
```bash
cp .env.example .env && chmod 600 .env     # add your DASHSCOPE_API_KEY
pip install -r requirements.txt
python scripts/smoke_test.py               # verify Qwen Cloud connectivity

# use it
python -c "import sys; sys.path.insert(0,'src'); from mnemo import Mnemo; \
  m=Mnemo(); m.ingest('I moved to Toronto last week.'); \
  print([x.text for x in m.recall('where does the user live?')])"

# run the HTTP API
cd src && uvicorn api:app --host 0.0.0.0 --port 8000
# or the MCP server
python src/mcp_server.py
```
Claude Desktop MCP config:
```json
{ "mcpServers": { "mnemo": { "command": "python", "args": ["/ABS/PATH/src/mcp_server.py"] } } }
```

## Results (honest) — full numbers in [`docs/BENCHMARK.md`](docs/BENCHMARK.md)
LongMemEval_S, off-Qwen validation (local `bge-small` + `gpt-4o-mini`), n=20:
- **recall@10: 95% = RAG.**
- **QA accuracy-per-1k-tokens: Mnemo 41.2 > RAG 27.4 > full-context 0.5** (best efficiency;
  45% acc on 1,092 tokens vs RAG 60% on 2,193 vs full-context 65% on 123,773).
- **Long-horizon:** RAG 100%→50% as a fact is updated 2→12×; **Mnemo stays 100%.**
- Weakness we report: temporal-reasoning QA (compression loses detail).

We do **not** claim a leaderboard-topping accuracy number — a strong RAG wins one-shot
retrieval. Mnemo wins on efficiency, long-horizon robustness, and capabilities RAG lacks.

## Tests
```bash
python scripts/test_memory.py        # bi-temporal, supersession, time-travel, forgetting, budget
python scripts/test_mnemo_e2e.py     # raw messages → distill → supersede → clean recall
```

## Deploy on Alibaba Cloud
Only `DASHSCOPE_API_KEY` is needed to run. Qwen Cloud/DashScope *is* Alibaba Cloud Model
Studio, so the model + embedding calls satisfy the "uses Alibaba Cloud services/APIs"
proof; `src/alicloud_oss.py` (OSS snapshots) is an optional stronger proof. Full runbook:
[`docs/DEPLOY.md`](docs/DEPLOY.md).

## Layout
```
src/  agent.py  ← the assistant
      mnemo.py memory.py distill.py config.py  ← the memory engine
      mcp_server.py api.py alicloud_oss.py     ← surfaces + deploy
scripts/  demo_agent.py  ← the video walkthrough
          test_memory.py test_mnemo_e2e.py smoke_test.py
          lme_recall.py bench_knowledge_update.py bench_horizon.py  ← benchmarks
docs/  DESIGN.md SOTA.md BENCHMARK.md COMPARISON.md DEPLOY.md COMPETITION.md
       architecture.svg horizon.svg
```

## License
MIT — see [LICENSE](LICENSE).
