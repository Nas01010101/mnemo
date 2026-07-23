<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/banner-dark.svg">
  <img src="docs/brand/banner-light.svg" alt="Tenet, bi-temporal belief memory for agents: temporal correctness without a graph database" width="820">
</picture>

<h3>
  <a href="https://nas01010101.github.io/tenet/">🌐&nbsp;Website</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="paper/tenet.pdf">📄&nbsp;Paper&nbsp;(PDF)</a>
</h3>

[![tests](https://github.com/Nas01010101/tenet/actions/workflows/test.yml/badge.svg)](https://github.com/Nas01010101/tenet/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg?logo=python&logoColor=white)](#quickstart)
[![release](https://img.shields.io/badge/release-v0.1.0-8b7cf8.svg?logo=github&logoColor=white)](https://github.com/Nas01010101/tenet/releases/latest)
[![Qwen Cloud](https://img.shields.io/badge/built%20on-Qwen%20Cloud-6a5acd.svg)](https://qwencloud-hackathon.devpost.com)
[![MCP](https://img.shields.io/badge/MCP-native-000000.svg)](src/tenet/mcp_server.py)

*Agent memory that stays true as your life changes: a self-consistent, bi-temporal belief
state instead of retrieval over a log, with zero LLM calls on the read path.*

Most agent memory is retrieval over a growing log of turns, and that log silently breaks
the moment a fact changes. Tenet gives every fact two clocks, event time and transaction
time, so a new value **supersedes** the old one instead of sitting beside it: current vs.
superseded, always queryable, always human-readable.

**English** · [简体中文](README.zh-CN.md)

</div>

---

```bash
pip install git+https://github.com/Nas01010101/tenet.git
```
```python
from tenet import Tenet

mem = Tenet()
mem.ingest("I live in Boston")              # needs an LLM key (distills the raw message)
mem.ingest("I moved to Seattle")            # supersedes: Boston kept in history
mem.recall("where do I live?")              # → [Seattle]  (current beliefs, no LLM call)
mem.recall("where do I live?", as_of=t0)    # → [Boston]   (time-travel, no LLM call)
mem.navigate("where do I live and work?")   # → adaptive multi-hop recall, no LLM call
```

Everything except `ingest` is **LLM-free** (embeddings + closed-form math, low-milliseconds;
fully offline with `EMBED_PROVIDER=local`). Only `ingest` needs a model key, because turning
free-form text into atomic facts is the one judgment call that needs one.
The [60-second zero-key demo](#quickstart) shows exactly where that line sits.

## Tenet vs Zep · Mem0 · Letta

The 2026 agent-memory field splits by job: **Mem0** for per-user personalization, **Zep/Graphiti**
for facts that change over time, **Letta** for self-managing long-horizon agents. Tenet targets
Zep's job (*temporal correctness when facts change*) but removes its cost of entry.

| | **Tenet** | Zep / Graphiti | Mem0 | Letta |
|---|---|---|---|---|
| Facts that change over time | ✅ bi-temporal supersession | ✅ bi-temporal graph | ❌ create-ts only | agent-managed |
| **Infra to run it** | **`pip install`: sqlite + numpy** | graph DB (Neo4j / FalkorDB) | vector DB | agent server + Postgres |
| Read path cost | **no LLM call** | no LLM call | no LLM call | an LLM call per op |
| **Read what it knows?** | ✅ **plain belief state** (`get_all()`) | ❌ graph nodes | ❌ opaque vectors | ❌ state blocks |
| Drop-in API | ✅ **Mem0-compatible** (`add`/`search`/`get_all`/`delete`) | graph API | `add`/`search`/… | full runtime |
| Time-travel (`as_of`) | ✅ | ✅ | ❌ | ❌ |

**The one-liner:** *Zep's temporal correctness, Mem0's drop-in API, and a belief state you can
actually open and read, with zero infrastructure.* Full honest matrix + comparability caveats:
[`docs/COMPARISON.md`](docs/COMPARISON.md).

## What you'd build on it

The general primitive for **any state that must be *currently true* while its history stays
auditable**. Every pattern below runs on a surface that already ships in this repo:

| You're building | The failure Tenet removes | Shipped surface |
|---|---|---|
| An agent in Claude Desktop / an IDE / any MCP client | memory that silently goes stale as facts change | MCP server (`learn`/`recall`/`time_travel`/`doubts`) |
| A LangGraph / LlamaIndex / LangChain / Mem0-style app | framework memory that keeps *both* the old and new value | `BaseStore` · `BaseMemoryBlock` · `TenetMemory` adapters, Mem0-compatible API |
| A support / CRM / ops bot over account state | answering from a superseded plan, address, or entitlement | keyed supersession + the churn result (RAG collapses 100→50%, Tenet holds 100%) |
| A multi-agent system with shared state | concurrent writers trampling each other's facts | the [Majalis](https://github.com/Nas01010101/majalis) pattern: supersession as the write-arbitration rule |
| Anything audited or regulated | "what did we believe when we decided X?" is unanswerable | provenance + `recall(as_of=…)` + `tenet timeline`/`export` |

And it scales like a library, not a service: reads are **~9–12 ms flat from 1k to 100k facts**
on a laptop (one SQLite file), so "add memory" never becomes "operate a database cluster."

> **Reproducibility is the pitch.** Independent 2026 audits found the field's headline numbers
> don't survive reproduction (Mem0's 93.4% on LongMemEval reproduces at
> [73.8%](docs/COMPARISON.md#-frontier-reality-check--the-2026-reproduction-crisis-verified-2026-07-14);
> LoCoMo's answer key is 6.4% wrong). Tenet reports **every** number with a Wilson 95% CI, ships
> **five flags default-OFF because we measured them as no-benefit**, and **falsified its own churn
> claim in public** before fixing it. Built **100% on Qwen Cloud**. Every result reproduces from
> one command.

## Results at a glance

| benchmark | metric | Tenet | comparison | source |
|---|---|---:|---:|---|
| MemoryAgentBench FactConsolidation (arXiv:2507.05257), single-hop | SubEM, pooled 6K–262K | **97.0** [94.8, 98.3] | > published gpt-4o-tier 94.8 · mini-tier SOTA 78.0 · naive-RAG 47.8 | [`BENCHMARK.md` §6](docs/BENCHMARK.md#6-mab-factconsolidation--the-standardized-supersession-benchmark-scriptsbench_factconpy) |
| MAB FactConsolidation, multi-hop | SubEM, pooled 6K–262K | **45.8** [40.9, 50.6] | **1.5×** published mini SOTA 30.2 (CI excludes) · every published memory system ≤7 | [`BENCHMARK.md` §6](docs/BENCHMARK.md#6-mab-factconsolidation--the-standardized-supersession-benchmark-scriptsbench_factconpy) |
| MAB Accurate-Retrieval | avg. official metric | **59.3** (2nd of all published systems) | Mem0 32.6 · Zep 37.5 | [`BENCHMARK.md` §7](docs/BENCHMARK.md#7-mab-accurate-retrieval--the-second-mab-competency-scriptsbench_mab_arpy) |
| MAB Test-Time Learning (5 ICL cells, n=500) | official substr-EM, avg | **77.2** [73.3, 80.7] (local 7B reader, $0) | > BM25 75.4 · MemGPT 67.6 · Zep 62.8 · Mem0 32.4 | [`BENCHMARK.md` §16](docs/BENCHMARK.md#16-mab-test-time-learning--the-third-mab-competency-scriptsbench_mab_ttlpy) |
| Knowledge-churn horizon (fact updated 2→12×) | current-value accuracy | **100%** throughout | naive-RAG collapses 100%→50% | [`BENCHMARK.md` §3](docs/BENCHMARK.md#3-long-horizon-knowledge-churn--where-memory-structurally-wins-scriptsbench_horizonpy) |
| LongMemEval_S (n=100, `qwen3.7-plus` reader) | QA accuracy | **81.0%** | ≥ matched RAG 79.0% · 100% recall@10 · 98.5% less context than full | [`BENCHMARK.md` §1–2](docs/BENCHMARK.md#1-retrieval-recall--longmemeval_s-scriptslme_recallpy) |
| Head-to-head vs **ReMe** (Alibaba's memory framework), LME_S n=100 | QA acc, same reader/judge | **67.0%** [57.3, 75.4] | ReMe 34.0% · matched RAG 64.0% · McNemar p≈2×10⁻⁶ | [`reme_h2h_results.json`](docs/reme_h2h_results.json) |
| Local LoRA distiller (offline, zero-cloud) | key-consistency, decontaminated | **0.775** | cloud reference (`qwen3.7-plus`) 0.707 | [`BENCHMARK.md` §10](docs/BENCHMARK.md#10-local-distiller-zero-cloud-verdict) |

Honest weak spots (multi-session synthesis, multi-hop chaining) are reported, not hidden.
Full tables, protocol notes (including the 2026-07-19 ingestion-keyer fix our own miss-file
audit exposed), and every reproduction command: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Memory reads shouldn't cost an LLM call

Most memory systems architect the *read* path around an LLM in the loop: a rerank call, a
synthesis pass, an agent deciding what to fetch next. **Tenet's bet is the opposite:** the one
judgment call (distilling a message into keyed facts) happens once, at **write time**; every
read is pure vector similarity + closed-form math. Supersession itself is deterministic
bi-temporal bookkeeping; no model is in that loop either.

| system | read/retrieval latency | LLM in read path | infra to run |
|---|---:|:---:|---|
| **Tenet** | **~11 ms** (@100k facts, flat) | **no** | none: sqlite + numpy |
| Zep / Graphiti | ~150–300 ms (graph search) | no | graph DB (Neo4j / FalkorDB) |
| Mem0 | ~1.44 s p95 (base) | no | vector DB |
| Letta | model-dependent (an LLM call per op) | yes | agent server + Postgres |

<sub>Flat ~9–12 ms from 1k to 100k facts ([`docs/SCALE.md`](docs/SCALE.md)). Latency scopes
differ across systems; competitor figures are each project's own published retrieval latency.
The point isn't a race: temporal correctness here costs no graph database and no inference call.</sub>

<div align="center">

<img src="docs/brand/demo.gif" alt="Tenet assistant staying correct as facts change: supersession, time-travel, forgetting" width="740">

<sub>Real recorded session: facts change, the belief state supersedes them, time-travel recalls what was true before, and the read path never calls an LLM.</sub>

</div>

## The failure mode nobody benchmarks

<div align="center">

![knowledge churn](docs/horizon.svg)

**As one templated fact is updated 2→12 times, RAG-memory falls 100%→50%. Tenet holds 100%.**

<sub>The single-attribute churn primitive (`bench_horizon`), pre-registered to favor Tenet. Under harder
*paraphrased* churn ([ChurnBench §9](docs/BENCHMARK.md#9-churnbench--parametric-high-churn-stress-test-measured-2026-07-10)),
the honest picture: read-time fixes lift Tenet's half-life <2→32; it ties an idealized delete-arm but
beats the real `mem0ai` package. Falsification and fix reported in full.</sub>

</div>

## Why it's different

| | retrieval memory (RAG) | **Tenet** |
|---|---|---|
| abstraction | document index of turns | **bi-temporal belief state** |
| a changed fact | two similar passages | **superseded** (bi-temporal, history kept) |
| stale evidence | retrieved forever | **retired** (belief–evidence consistency) |
| write policy | store everything | **surprise-gated** (predictive coding) |
| forgetting | none (grows forever) | salience-decay sweep |
| fact drift | unmodeled | **staleness hints**: learned P(still-valid) per attribute, `tenet doubts` |
| queryable across time | no | **time-travel** (`recall(as_of=t)`) |
| multi-hop bridging | fixed-depth *k*, or none | **adaptive `navigate()`**: LLM-free, gated by relevance gain |
| read path | n/a | **no LLM call** |

## Quickstart

### 1. One command, no API key

```bash
git clone https://github.com/Nas01010101/tenet && cd tenet
pip install -e ".[local]"                   # bge-small embedder, CPU; offline after install
python examples/00_zero_key_demo.py         # supersession + time-travel + doubts, zero LLM calls
```
Walks the entire LLM-free read path end to end against a pre-formed fact ledger. The one thing
it can't show is `ingest()` turning conversation into facts; that's the one call that needs a
model (next step). First-ever install downloads ~1 GB of wheels; every run after is offline.

### 2. The full agent (needs an API key)

```bash
cp .env.example .env && chmod 600 .env      # add DASHSCOPE_API_KEY (Qwen Cloud)
pip install -e ".[all]"                     # base + api/mcp/oss/local/cli/langgraph extras
python scripts/smoke_test.py                # verify connectivity
uvicorn tenet.api:app --host 0.0.0.0 --port 8000  # HTTP API incl. POST /chat
python -m tenet.mcp_server                   # or the MCP server (learn/recall/navigate/forget/stats)
```
No key yet? `tenet recall` / `navigate` / `stats` / `doubts` / `timeline` / `export` work fully
offline with `EMBED_PROVIDER=local`. `tenet timeline --all` is the fastest way to *see* the
bi-temporal chain: current value highlighted, retired values dimmed. Default DB:
`data/tenet.db` (override with `TENET_DB_PATH`; falls back to `~/.tenet/tenet.db`).

**Works with:** any MCP client ([Claude Desktop](examples/03_mcp_client.md), IDEs) ·
[LangChain](examples/04_langchain_memory.py) · [LangGraph](examples/05_langgraph_store.py) ·
[LlamaIndex](examples/06_llamaindex_memory.py) · plain HTTP (`tenet.api:app`, `POST /chat`).
All examples: [`examples/`](examples/).

### LangGraph `BaseStore` adapter

Tenet drops in as a LangGraph [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/)
so a LangGraph agent's long-term memory gets bi-temporal supersession for free:

```bash
pip install "tenet-memory[langgraph] @ git+https://github.com/Nas01010101/tenet.git"
```
```python
from tenet.integrations.langgraph import TenetStore

store = TenetStore(db_path="data/agent.db")
store.put(("users", "alex"), "residence", {"city": "Montreal"})
store.put(("users", "alex"), "residence", {"city": "Toronto"})  # supersedes, not overwritten
store.get(("users", "alex"), "residence").value                # -> {"city": "Toronto"}
```

### 3. Fully local / air-gapped

Both write-path calls (`ingest()`'s fact-distillation and `embed_texts()`) can run against a
local model, so the whole loop works with **zero cloud calls**:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=tenet-distiller-1.5b-v2 EMBED_PROVIDER=local \
  tenet remember "I moved from Boston to Seattle"   # distilled + embedded 100% locally
```

`tenet-distiller-1.5b-v2` is our LoRA-tuned Qwen2.5-1.5B distiller: on a decontaminated
held-out eval it supersedes 6/6 clean-churn cases (untuned base: 0/6) at **0.775
key-consistency, beating the cloud reference's 0.707**. Trained on one RTX 3080; the full
reproducible pipeline lives in [`scripts/distiller_lora/`](scripts/distiller_lora/) and the
measurement caveats in [`BENCHMARK.md` §10](docs/BENCHMARK.md#10-local-distiller-zero-cloud-verdict).

## The agent

Tenet ships as a personal assistant ([`src/tenet/agent.py`](src/tenet/agent.py)) on Qwen Cloud:
```
you › Hi! I'm Alex, I live in Montreal and work as a data analyst.
assistant › Nice to meet you, Alex! How's the analyst work in Montreal?   [remembered 2 facts]
… weeks later …
you › I moved to Toronto and got promoted to senior analyst!
you › Where do I live and what's my job now?
assistant › You live in Toronto and you're a senior analyst. Congrats on the promotion!
```
```bash
python -m tenet.agent          # interactive assistant (or: tenet-agent)
python scripts/demo_agent.py   # the scripted story (video walkthrough)
```

## Architecture
![architecture](docs/architecture.svg)

Two layers over one bi-temporal store (beliefs + evidence), two surfaces (MCP + HTTP), powered
by Qwen Cloud. Component diagram + key equations: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
original scoping: [`docs/DESIGN.md`](docs/DESIGN.md) · positioning: [`docs/COMPARISON.md`](docs/COMPARISON.md).

## Reproduce the paper

Every benchmark is one CLI command: provider preset + config + git-sha logged to
`data/bench_runs.jsonl`.

```bash
tenet bench list                        # all benchmarks + which figure/§ each reproduces
tenet bench run <name> --dry-run ...     # print the exact command+env, run nothing
tenet bench run churn --provider ollama --principals 12 --k 6 --updates 2,4,6,8,10,12   # Fig.1
```
`--provider` presets: `ollama` (fully offline), `openrouter`, `local`, `qwen` (Qwen Cloud).
Full matrix: [`docs/BENCHMARK.md`](docs/BENCHMARK.md) · [`docs/HARNESS.md`](docs/HARNESS.md).

## Repository
```
paper/      tenet.md tenet.pdf                        the paper
src/tenet/  core.py memory.py distill.py navigate.py  the belief-state memory engine
            agent.py mcp_server.py api.py             the assistant + MCP/HTTP surfaces
            integrations/                             LangGraph + LlamaIndex adapters
examples/   00_zero_key_demo.py … 06_llamaindex_memory.py
scripts/    bench_*.py test_*.py demo_agent.py        benchmarks, tests, walkthrough
docs/       BENCHMARK.md COMPARISON.md ARCHITECTURE.md DESIGN.md DEPLOY.md
```

## Citation
```bibtex
@misc{tenet2026,
  title  = {Tenet: Agent Memory as a Self-Consistent Belief State},
  author = {Elghoudane, Anas},
  year   = {2026},
  note   = {Global AI Hackathon with Qwen Cloud, Track 1},
  url    = {https://github.com/Nas01010101/tenet}
}
```

## Origin

Tenet started as a [Global AI Hackathon with Qwen Cloud](https://qwencloud-hackathon.devpost.com)
(Track 1: MemoryAgent) entry. Hackathon materials: [`docs/hackathon/`](docs/hackathon/).
[Majalis](https://github.com/Nas01010101/majalis), our Track 3 agent society, runs its shared
belief board on Tenet's exact supersession design; the mechanism is load-bearing in a second
product.

## License

MIT. See [LICENSE](LICENSE).
