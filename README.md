<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/banner-dark.svg">
  <img src="docs/brand/banner-light.svg" alt="Tenet — bi-temporal belief memory for agents: temporal correctness without a graph database" width="820">
</picture>

<p>
  <a href="paper/tenet.pdf"><b>📄 Paper</b></a> ·
  <a href="paper/extended_abstract.md"><b>Extended abstract</b></a> ·
  <a href="docs/BENCHMARK.md"><b>Benchmarks</b></a> ·
  <a href="docs/COMPARISON.md"><b>vs Mem0 / Zep / Letta</b></a> ·
  <a href="src/tenet/mcp_server.py"><b>MCP server</b></a> ·
  <a href="scripts/demo_agent.py"><b>Demo</b></a>
</p>

[![tests](https://github.com/Nas01010101/tenet/actions/workflows/test.yml/badge.svg)](https://github.com/Nas01010101/tenet/actions/workflows/test.yml)
[![paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/tenet.pdf)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg?logo=python&logoColor=white)](#quickstart)
[![PyPI](https://img.shields.io/badge/pypi-coming%20soon-yellow.svg?logo=pypi&logoColor=white)](pyproject.toml)
[![Qwen Cloud](https://img.shields.io/badge/built%20on-Qwen%20Cloud-6a5acd.svg)](https://qwencloud-hackathon.devpost.com)
[![MCP](https://img.shields.io/badge/MCP-native-000000.svg)](src/tenet/mcp_server.py)
[![stars](https://img.shields.io/github/stars/Nas01010101/tenet?style=flat&color=8b7cf8)](https://github.com/Nas01010101/tenet/stargazers)

*Memory reads shouldn't cost an LLM call.*

**English** · [简体中文](README.zh-CN.md)

```bash
pip install tenet-memory   # not yet on PyPI — until it lands, install from source (below)
```
```python
from tenet import Tenet

mem = Tenet()
mem.ingest("I live in Boston")              # needs an LLM key (distills the raw message)
mem.ingest("I moved to Seattle")            # supersedes — Boston kept in history
mem.recall("where do I live?")              # → [Seattle]  (current beliefs, no LLM call)
mem.recall("where do I live?", as_of=t0)    # → [Boston]   (time-travel, no LLM call)
mem.navigate("where do I live and work?")   # → adaptive multi-hop recall, no LLM call
```
`recall` / `stats` / `doubts` / time-travel (`recall(as_of=...)`) / `navigate` are **LLM-free** —
embeddings + cosine + closed-form math only, low-milliseconds, and with `EMBED_PROVIDER=local`
none of them need an API key at all. `ingest` (and the chat agent) need a working
`DASHSCOPE_API_KEY` (or `LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY`) since turning free-form
text into atomic facts is the one judgment call that needs a model — see
[the 60-second zero-key demo](#quickstart) below for exactly where that line sits.

</div>

---

## Tenet vs Zep · Mem0 · Letta

The 2026 agent-memory field splits by job: **Mem0** for per-user personalization, **Zep/Graphiti**
for facts that change over time, **Letta** for self-managing long-horizon agents. Tenet targets
Zep's job — *temporal correctness when facts change* — but removes its cost of entry.

| | **Tenet** | Zep / Graphiti | Mem0 | Letta |
|---|---|---|---|---|
| Facts that change over time | ✅ bi-temporal supersession | ✅ bi-temporal graph | ❌ create-ts only | agent-managed |
| **Infra to run it** | **`pip install` — sqlite + numpy** | graph DB (Neo4j / FalkorDB) | vector DB | agent server + Postgres |
| Read path cost | **no LLM call** | no LLM call | no LLM call | an LLM call per op |
| **Read what it knows?** | ✅ **plain belief state** (`get_all()`) | ❌ graph nodes | ❌ opaque vectors | ❌ state blocks |
| Drop-in API | ✅ **Mem0-compatible** (`add`/`search`/`get_all`/`delete`) | graph API | `add`/`search`/… | full runtime |
| Time-travel (`as_of`) | ✅ | ✅ | ❌ | ❌ |

**The one-liner:** *Zep's temporal correctness, Mem0's drop-in API, and a belief state you can
actually open and read — with zero infrastructure.* Every other temporal system here needs a
database server running; Tenet is a library. And unlike vector or graph memory, what Tenet stores
is **human-readable** — `subject::attribute → value`, current vs. superseded — so you can audit
exactly what your agent believes.

```python
mem = Tenet()
mem.add("I moved to Seattle", user_id="alex")     # Mem0-style, drop-in
mem.search("where do I live?", user_id="alex")    # → [Seattle]  (no LLM call)
mem.get_all(user_id="alex")                        # → readable belief state, not opaque vectors
```

Full honest matrix + benchmark comparability caveats: [`docs/COMPARISON.md`](docs/COMPARISON.md).

> **Reproducibility is the pitch.** Independent 2026 audits found the field's headline
> numbers don't survive reproduction — Mem0 claims 93.4% on LongMemEval but reproduces at
> [73.8% hosted / 32.4% OSS](docs/COMPARISON.md#-frontier-reality-check--the-2026-reproduction-crisis-verified-2026-07-14);
> LoCoMo's answer key is 6.4% wrong. Tenet reports **every** number with a Wilson 95% CI,
> ships **four flags default-OFF because we measured them as no-benefit**, and **falsified
> its own churn claim in public** before fixing it. Built **100% on Qwen Cloud** (no OpenAI
> in the product path). Every result reproduces from one command.

## Results at a glance

| benchmark | metric | Tenet | comparison | source |
|---|---|---:|---:|---|
| MemoryAgentBench FactConsolidation (ICLR 2026), single-hop | SubEM, pooled 6K–262K | **86.5** [82.8, 89.5] | published mini-tier SOTA 78.0 · naive-RAG 47.8 | [`BENCHMARK.md` §6](docs/BENCHMARK.md#6-mab-factconsolidation--the-standardized-supersession-benchmark-scriptsbench_factconpy) |
| MAB Accurate-Retrieval | avg. official metric | **59.3** (2nd of all published systems) | Mem0 32.6 · Zep 37.5 | [`BENCHMARK.md` §7](docs/BENCHMARK.md#7-mab-accurate-retrieval--the-second-mab-competency-scriptsbench_mab_arpy) |
| Knowledge-churn horizon (fact updated 2→12×) | current-value accuracy | **100%** throughout | naive-RAG collapses 100%→50% | [`BENCHMARK.md` §3](docs/BENCHMARK.md#3-long-horizon-knowledge-churn--where-memory-structurally-wins-scriptsbench_horizonpy) |
| LongMemEval_S (n=100, `qwen3.7-plus` Qwen-Cloud reader) | QA accuracy | **81.0%** | ≥ matched RAG 79.0% · **100%** recall@10 · **98.5% less context** than full | [`BENCHMARK.md` §1–2](docs/BENCHMARK.md#1-retrieval-recall--longmemeval_s-scriptslme_recallpy) |
| Local LoRA distiller (offline, zero-cloud) | key-consistency, decontaminated | **0.775** | cloud reference (`qwen3.7-plus`) 0.707 | [`BENCHMARK.md` §10](docs/BENCHMARK.md#10-local-distiller-zero-cloud-verdict) |

Honest weak spots (multi-session synthesis, multi-hop chaining) are reported, not
hidden — full tables and reproduction commands: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Memory reads shouldn't cost an LLM call

Most agent-memory systems architect the *read* path around an LLM in the loop — a rerank
call, a synthesis pass, an agent deciding what to fetch next. **Tenet's bet is the opposite:**
`recall`, `doubts`, time-travel (`recall(as_of=...)`), and the adaptive multi-hop `navigate()` are pure vector
similarity + closed-form math, so they cost no API call and no inference latency — the thing
that *does* need judgment (turning a raw message into atomic, keyed facts) happens once, at
**write time** (`ingest`), not on every read. Supersession itself — the mechanism that keeps
answers correct as facts change — is deterministic bi-temporal bookkeeping; no model is in that
loop either.

That makes the read path *fast* — and it stays fast at scale:

| system | read/retrieval latency | LLM in read path | infra to run |
|---|---:|:---:|---|
| **Tenet** | **~11 ms** (@100k facts, flat) | **no** | none — sqlite + numpy |
| Zep / Graphiti | ~150–300 ms (graph search) | no | graph DB (Neo4j / FalkorDB) |
| Mem0 | ~1.44 s p95 (base) | no | vector DB |
| Letta | model-dependent (an LLM call per op) | yes | agent server + Postgres |

<sub>Tenet's read is embedding + cosine + closed-form decay over a resident matrix — **flat ~9–12 ms
from 1k to 100k facts** ([`docs/SCALE.md`](docs/SCALE.md)), ~100× faster than its own pre-resident-index
baseline. Latency *scopes* differ across systems (all exclude the downstream reader LLM); competitor
figures are each project's own published retrieval latency. The point isn't a benchmarked race — it's
that temporal correctness here costs **no graph database and no inference call.**</sub>

LLM-agent memory is almost always **retrieval over a log of past turns**. That's the wrong
abstraction for an agent modeling a *changing* world: as a fact is updated over a long
interaction — **knowledge churn** — stale versions crowd the retrieval budget and the agent
answers with an out-of-date value. **Tenet** reframes memory as a **self-consistent belief
state** — the current, supersession-aware set of facts about the user — and stays correct where
retrieval collapses.

<div align="center">

<img src="docs/brand/demo.gif" alt="Tenet assistant staying correct as facts change — supersession, time-travel, forgetting" width="740">

<sub>Real recorded session: facts change, the belief state supersedes them, time-travel recalls what was true before — and the read path never calls an LLM.</sub>

</div>

## The failure mode nobody benchmarks

<div align="center">

![knowledge churn](docs/horizon.svg)

**As one templated fact is updated 2→12 times, RAG-memory falls 100%→50%. Tenet holds 100%.**

<sub>The single-attribute churn primitive (`bench_horizon`), pre-registered to favor Tenet. Under harder
*paraphrased*, multi-attribute churn ([ChurnBench §9](docs/BENCHMARK.md#9-churnbench--parametric-high-churn-stress-test-measured-2026-07-10)),
the honest picture: read-time fixes lift Tenet's half-life <2→32 (U=32 ≈ 82–100% across runs); it ties an
idealized delete-arm but beats the real `mem0ai` package — falsification and fix reported in full.</sub>

</div>

## Why it's different

| | retrieval memory (RAG) | **Tenet** |
|---|---|---|
| abstraction | document index of turns | **bi-temporal belief state** |
| a changed fact | two similar passages | **superseded** (bi-temporal, history kept) |
| stale evidence | retrieved forever | **retired** (belief–evidence consistency) |
| write policy | store everything | **surprise-gated** (predictive coding) |
| forgetting | none (grows forever) | salience-decay sweep |
| fact drift | unmodeled | **staleness hints** — learned P(still-valid) per attribute, `tenet doubts` |
| queryable across time | no | **time-travel** (`recall(as_of=t)`) |
| multi-hop bridging | fixed-depth *k*, or none | **adaptive `navigate()`** — deepens hops only while new evidence clears a relevance-gain gate, LLM-free |
| read path | — | **no LLM call** |

Read the 2-page paper: **[`paper/tenet.md`](paper/tenet.md)**.

## Quickstart

### 1. 60 seconds, no API key

```bash
pip install tenet-memory[local]             # bge-small embedder, CPU — no network call at all
python examples/00_zero_key_demo.py         # supersession + time-travel + doubts, zero LLM calls
```
Walks the entire LLM-free read path end to end — recall, supersession, time-travel, and the
learned-dynamics `doubts` — against a pre-formed fact ledger. The one thing it *can't* show is
`ingest()` turning free-form conversation into those facts; that's the one call in Tenet that
needs a model (next step).

### 2. The full agent (needs an API key)

```bash
cp .env.example .env && chmod 600 .env      # add DASHSCOPE_API_KEY (Qwen Cloud)
pip install -e ".[all]"                     # base + api/mcp/oss/local/cli/langgraph extras
python scripts/smoke_test.py                # verify connectivity
uvicorn tenet.api:app --host 0.0.0.0 --port 8000  # HTTP API incl. POST /chat
python -m tenet.mcp_server                   # or the MCP server (learn/recall/navigate/forget/stats)
```
`pip install -e .` alone only pulls the base library (`openai`, `numpy`) — the API server and
MCP server need the `api`/`mcp` extras (bundled in `[all]` above), or install just what you need,
e.g. `pip install -e ".[api]"`. No key yet? `tenet recall` / `tenet navigate` / `tenet stats` /
`tenet doubts` work fully offline with `EMBED_PROVIDER=local` (installs `sentence-transformers`,
no network call at all); `tenet remember` / `tenet chat` / the MCP `learn` tool need a real
`DASHSCOPE_API_KEY` (or `LLM_PROVIDER=openrouter`) since they distill text with an LLM call —
without one you'll see a clear "memory write failed: ..." error rather than a silent no-op.

More in [`examples/`](examples/) — zero-key demo, quickstart, assistant loop, MCP client,
LangChain adapter, LangGraph `BaseStore` adapter.

**Works with:** any MCP client ([Claude Desktop](examples/03_mcp_client.md), IDEs, other
agents) · [LangChain](examples/04_langchain_memory.py) via a thin `TenetMemory` adapter ·
[LangGraph](examples/05_langgraph_store.py) via a `BaseStore` adapter (below) · plain HTTP
(`tenet.api:app`, `POST /chat`).

### LangGraph `BaseStore` adapter

Tenet drops in as a LangGraph [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/)
— the interface `StateGraph.compile(store=...)` expects — so a LangGraph agent's long-term memory
gets bi-temporal supersession for free: re-`put()`-ting a `(namespace, key)` retires the old
value to history instead of overwriting it, the same mechanism `Tenet.ingest()` uses.

```bash
pip install tenet-memory[langgraph]
```
```python
from tenet.integrations.langgraph import TenetStore

store = TenetStore(db_path="data/agent.db")
store.put(("users", "alex"), "residence", {"city": "Montreal"})
store.put(("users", "alex"), "residence", {"city": "Toronto"})  # supersedes, not overwritten
store.get(("users", "alex"), "residence").value                # -> {"city": "Toronto"}
```
Full example (put/get/search/delete/list_namespaces): [`examples/05_langgraph_store.py`](examples/05_langgraph_store.py).

### 3. Fully local / air-gapped

Every call in the write path — `ingest()`'s fact-distillation and `embed_texts()` — can run
against a local model, so the whole loop (learn → supersede → doubt → time-travel) works with
**zero cloud calls**, network off included:

```bash
# .env or shell env
LLM_PROVIDER=ollama
OLLAMA_MODEL=tenet-distiller-1.5b-v2   # our LoRA-tuned distiller (below), or any local model
EMBED_PROVIDER=ollama                  # or EMBED_PROVIDER=local for bge-small (no ollama needed)
OLLAMA_BASE_URL=http://localhost:11434/v1   # or a GPU box over Tailscale, e.g. http://100.x.x.x:11434/v1
```
```bash
tenet remember "I moved from Boston to Seattle"   # distilled + embedded 100% locally
tenet doubts                                       # learned-dynamics confidence, still zero-LLM
```

**What "tenet-distiller-1.5b-v2" is and what was measured:** a LoRA-tuned Qwen2.5-1.5B-Instruct
that replaces the cloud fact-distiller (`qwen3.7-plus`) for the one LLM-dependent step in the
write path — turning a message into `subject::attribute` JSON facts with keys stable enough for
bi-temporal supersession. On a **decontaminated** held-out eval (novel values + phrasings, zero
train overlap): the untuned 1.5B base model **cannot supersede at all** (0/6 clean-churn cases
superseded correctly), while the LoRA-tuned model reproduces the cloud reference's supersession
behavior fully offline — **6/6 clean-churn superseded, 0.0 fabrication rate, 0.775 key-consistency**
(the metric that actually drives supersession — same attribute must map to the same key across
paraphrases). That **beats the cloud reference's own key-consistency (0.707)**, because the
training labels force-canonicalize keys in a way ad hoc cloud prompting doesn't. Full tables:
[`docs/BENCHMARK.md` §10](docs/BENCHMARK.md#10-local-distiller-zero-cloud-verdict).

**Caveat, stated plainly:** these are deterministic point estimates on a small eval (n=26
messages / 8 churn groups), not confidence intervals — a probe result, not a production SLA.
Directionally strong enough to ship as an opt-in path; wider-N validation is future work.

The training pipeline (data generation, canonicalization, empty-target rebalancing, LoRA SFT)
lives in [`scripts/distiller_lora/`](scripts/distiller_lora/) and is fully reproducible —
everything above was trained and evaluated on a single RTX 3080 (16GB). The GGUFs currently
live on that box, served via ollama; to export and serve your own:

```bash
# on the GPU box, after training (train_lora.py) + merging (merge_and_export.py):
#   1. merge the LoRA adapter into the base model (merge_and_export.py does this, bf16 safetensors)
#   2. convert to GGUF with llama.cpp — ollama's native safetensors import mangles merged
#      Qwen2.5 bf16 weights (garbage output); GGUF is the path that actually works:
python llama.cpp/convert_hf_to_gguf.py <merged_dir> --outtype q8_0 \
    --outfile tenet-distiller-1.5b-v2.gguf
ollama create tenet-distiller-1.5b-v2 -f Modelfile   # Modelfile: FROM ./tenet-distiller-1.5b-v2.gguf
```

## Results

LongMemEval_S — honest, reproducible; full detail in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

**Absolute accuracy tracks reader strength, not the memory — because retrieval is already
saturated (recall@10 = 97.5–100%).** The right facts are in the context; a capable reader is what
turns them into a right answer. On **Qwen Cloud's own reader** (`qwen3.7-plus`, the shipped
product stack, fully cloud, n=100) Tenet reaches **81.0 %** — and frontier off-Qwen readers agree
(clean, un-batched, one call per item):

| reader | n | RAG | **Tenet** |
|---|---:|---:|---:|
| **`qwen3.7-plus`** (product · Qwen Cloud) | 100 | 79.0 | **81.0** |
| **gpt-5.5** (frontier) | 40 | 75.0 | **77.5** |
| **Gemini-3.5-flash** | 40 | 70.0 | **75.0** |
| gpt-4o (the weak-reader efficiency point, below) | 40 | 57.5 | 57.5 |

So with a capable reader Tenet reaches **~75–81 %, ≥ matched RAG at every reader**, at **100 %
recall@10** and **98.5 % less context** than full history — and on the Qwen reader it *wins* the
multi-session category (75.0 vs 54.2) and temporal reasoning (80.0 vs 73.3). The "57.5 %" you'll
see below is a *deliberately weak-reader efficiency operating point*, **not** Tenet's accuracy
ceiling; the reason our headline once looked lower than Mem0/Zep's 90 %+ was the eval
reader/embedder, not the memory design. The table further down traces the accuracy↔token
**frontier** under a fixed gpt-4o reader.

> **Note:** the shipped product runs **entirely on Qwen Cloud** (`text-embedding-v4`,
> `qwen3.6-flash`, `qwen3.7-plus`). `gpt-4o`/`gpt-4o-mini` appear below **only as frozen
> evaluation readers**, to match the exact protocol Mem0/Zep/MemoryAgentBench publish
> against — apples-to-apples with the published leaderboards.

Tenet is a **frontier, not a point** — one `expand` knob trades tokens for accuracy:

| | mode | recall@10 | QA acc | reader tokens | **acc / 1k tok** |
|---|---|---:|---:|---:|---:|
| full-context | — | — | 65% | ~124,000 | 0.5 |
| RAG | top-*k* turns | 95% | 57.5% | 2,101 | 27.4 |
| **Tenet** | efficiency | **97.5%** | 52.5% | **1,067** | **49.2** ← best/token |
| **Tenet** | parity | **97.5%** | **57.5%** | 2,083 | 27.6 |

- **Matches strong RAG on one-shot accuracy at equal-or-lower tokens** (57.5% = 57.5%, gpt-4o) —
  belief-anchored evidence expansion closed the gap belief-only compression left open. On a
  `gpt-4o-mini` reader the parity point edges ahead (60.0 vs 55.0).
- **Best accuracy-per-token** at the efficiency point (1.6× RAG at *half* its context) — and
  **reader-robust** across the `gpt-4o-mini` and `gpt-4o` readers we ran (≈1.6×).
- **Churn — reported honestly, no strawman.** On the single-attribute primitive (§3,
  `bench_horizon`) Tenet holds 100% vs RAG's 50% collapse. But that primitive is
  pre-registered to *structurally favor* Tenet — so we also run the harder multi-fact
  [ChurnBench](docs/BENCHMARK.md#9-churnbench--parametric-high-churn-stress-test-measured-2026-07-10)
  (§9), where the honest picture is: read-time fixes lift Tenet's churn half-life from <2 to
  **32** (~82% at U=32, §9.1), but an *idealized* delete-outright Mem0-style arm stays flat 100
  there — **Tenet does not beat it on raw churn accuracy, and we say so.** What Tenet wins is
  against the **real `mem0ai` package**, which (unlike that idealized arm) *accumulates* stale
  copies — in a live head-to-head it answers with a superseded value while Tenet keeps a clean,
  queryable belief history ([§A.2](docs/COMPARISON.md)). The durable edge over Mem0 isn't
  churn accuracy — it's **staying correct *and* keeping the history Mem0 deletes, at far less
  context.** (We ported delete-outright into Tenet as `TENET_CONSOLIDATE` and measured it a
  **no-benefit** — default-OFF, [§9.2](docs/BENCHMARK.md#92-write-time-consolidation-tenet_consolidate--a-measured-negative-measured-2026-07-14).)
- **Ablation:** the belief–evidence consistency rule alone lifts current-value accuracy 55%→100%.
- **Honest:** the one category still behind RAG is multi-session synthesis (42.9 vs 57.1, up
  from 28.6). We report it. *(Eval off-Qwen, one seed, reader noise ≈±5–7pp; shipped system uses Qwen Cloud.)*

### 🏆 Standardized: MemoryAgentBench FactConsolidation (ICLR 2026, all 800 questions)

Conflict resolution — the axis famous memory systems fail hardest (original table: **Zep 7%,
Mem0 18%, MemGPT 28%** single-hop; **≤7%** multi-hop for all 22 systems):

| pooled 6K–262K | naive-RAG | **Tenet** | published SOTA (mini / gpt-4o) |
|---|---:|---:|---:|
| single-hop | 47.8 | **86.5** [82.8, 89.5] | 78.0 / 94.8 |
| multi-hop | 4.5 | **30.2** [26.0, 34.9] | 30.2 / 51.5 |

**Above the published mini-tier single-hop SOTA and tied on multi-hop — with a local 7B
backbone and *zero-LLM* deterministic ingestion.** SubEM + official prompt verbatim; Wilson
CIs; no length collapse (SH ≥81% at every haystack size). Details: [`docs/BENCHMARK.md`](docs/BENCHMARK.md) §6.

**MAB Accurate-Retrieval** (~2,000 questions over 197K–534K-token contexts, official
per-benchmark metrics, matched gpt-4o-mini reader): AR average **59.3** — second only to
HippoRAG-v2 (65.1, which runs LLM OpenIE over every context token; Tenet ingests with
**embeddings only**), 20+ points above Mem0 (32.6) / Zep (37.5) / MemGPT, and **beats the
field on EventQA (70.7 vs 67.6, CI excludes)**. RULER MH is the honest loss (45 vs 66).
Details: [`docs/BENCHMARK.md`](docs/BENCHMARK.md) §7.

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

Two layers over one bi-temporal store (beliefs + evidence), two surfaces (MCP + HTTP),
powered by Qwen Cloud (Alibaba Cloud Model Studio). One-page component diagram + key
equations + the annotation-only invariant story: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Original scoping: [`docs/DESIGN.md`](docs/DESIGN.md); positioning vs Mem0/Zep/Letta/Mastra:
[`docs/COMPARISON.md`](docs/COMPARISON.md).

## Reproduce the paper
Every benchmark is one CLI command — provider preset + config + git-sha logged to
`data/bench_runs.jsonl`. `tenet bench run` dispatches to the `scripts/bench_*.py` that
produce the paper numbers (the source of truth); it never reimplements them.
```bash
tenet bench list                        # all benchmarks + which figure/§ each reproduces
tenet bench run <name> --dry-run ...     # print the exact command+env, run nothing
tenet bench results                     # table of past runs

python scripts/test_memory.py ; python scripts/test_tenet_e2e.py                  # capabilities
tenet bench run churn --provider ollama --principals 12 --k 6 --updates 2,4,6,8,10,12   # Fig.1 churn
tenet bench run lme-recall --provider openrouter --limit 40 --k 10 --seed 2 --qa            # efficiency point
tenet bench run lme-recall --provider openrouter --limit 40 --k 10 --seed 2 --qa --expand 20  # parity point
tenet bench run knowledge-update --provider ollama --principals 4                 # supersession ablation
```
`--provider` presets: `ollama` (fully offline: local embeddings + qwen2.5:7b reader),
`openrouter` (local embeddings + gpt-4o-mini reader), `local` (embeddings only),
`qwen` (Qwen Cloud). Full matrix + read-path perf analysis: [`docs/BENCHMARK.md`](docs/BENCHMARK.md), [`docs/HARNESS.md`](docs/HARNESS.md).

## Repository
```
paper/tenet.md tenet_full.pdf   the paper (2-page + full preprint)
src/tenet/  core.py memory.py distill.py config.py   the belief-state memory engine
            navigate.py                               adaptive LLM-free multi-hop recall
            agent.py                                  the assistant
            mcp_server.py api.py alicloud_oss.py      surfaces + Alibaba Cloud deploy
            integrations/langgraph.py                 LangGraph BaseStore adapter
examples/   00_zero_key_demo.py 01_quickstart.py 02_assistant.py 04_langchain_memory.py
            05_langgraph_store.py                     zero-key demo, quickstart, assistant loop,
                                                        LangChain + LangGraph adapters
scripts/    demo_agent.py    video walkthrough
            bench_horizon.py bench_factcon.py bench_mab_ar.py lme_recall.py   benchmarks
            test_memory.py test_dynamics.py test_agent_uncertainty.py test_errors.py
            test_langgraph_store.py test_navigate.py test_tenet_e2e.py smoke_test.py   tests
docs/ BENCHMARK.md COMPARISON.md DESIGN.md DEPLOY.md  architecture.svg horizon.svg
```

## Citation
```bibtex
@misc{tenet2026,
  title  = {Tenet: Agent Memory as a Self-Consistent Belief State},
  author = {Anas},
  year   = {2026},
  note   = {Global AI Hackathon with Qwen Cloud, Track 1},
  url    = {https://github.com/Nas01010101/tenet}
}
```

## Origin
Tenet started as a [Global AI Hackathon with Qwen Cloud](https://qwencloud-hackathon.devpost.com)
(Track 1: MemoryAgent) entry — hackathon materials live in [`docs/hackathon/`](docs/hackathon/).

## License
MIT — see [LICENSE](LICENSE).
