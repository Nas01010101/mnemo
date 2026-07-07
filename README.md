<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/banner-dark.svg">
  <img src="docs/brand/banner-light.svg" alt="Tenet — agent memory as a self-consistent world model" width="820">
</picture>

<p>
  <a href="paper/tenet.pdf"><b>📄 Paper</b></a> ·
  <a href="docs/BENCHMARK.md"><b>Benchmarks</b></a> ·
  <a href="docs/COMPARISON.md"><b>vs Mem0 / Zep / Letta</b></a> ·
  <a href="src/mcp_server.py"><b>MCP server</b></a> ·
  <a href="scripts/demo_agent.py"><b>Demo</b></a>
</p>

[![tests](https://github.com/Nas01010101/tenet/actions/workflows/test.yml/badge.svg)](https://github.com/Nas01010101/tenet/actions/workflows/test.yml)
[![paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/tenet.pdf)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg?logo=python&logoColor=white)](#quickstart)
[![Qwen Cloud](https://img.shields.io/badge/built%20on-Qwen%20Cloud-6a5acd.svg)](https://qwencloud-hackathon.devpost.com)
[![MCP](https://img.shields.io/badge/MCP-native-000000.svg)](src/mcp_server.py)
[![stars](https://img.shields.io/github/stars/Nas01010101/tenet?style=flat&color=8b7cf8)](https://github.com/Nas01010101/tenet/stargazers)

*A memory that stays true as the world changes.* Built for the
[Global AI Hackathon with Qwen Cloud](https://qwencloud-hackathon.devpost.com) — **Track 1: MemoryAgent**.

</div>

---

LLM-agent memory is almost always **retrieval over a log of past turns**. That's the wrong
abstraction for an agent modeling a *changing* world: as a fact is updated over a long
interaction — **knowledge churn** — stale versions crowd the retrieval budget and the agent
answers with an out-of-date value. **Tenet** reframes memory as a **self-consistent belief
state** — a compact *world model of the user* — and stays correct where retrieval collapses.

<div align="center">

<img src="docs/brand/demo.gif" alt="Tenet assistant staying correct as facts change — supersession, time-travel, forgetting" width="740">

<sub>Real recorded session: facts change, the belief state supersedes them, time-travel recalls what was true before — and the read path never calls an LLM.</sub>

</div>

## The failure mode nobody benchmarks

<div align="center">

![knowledge churn](docs/horizon.svg)

**As a fact is updated 2→12 times, RAG-memory falls 100%→50%. Tenet holds 100%.**

</div>

## Why it's different

| | retrieval memory (RAG) | **Tenet** |
|---|---|---|
| abstraction | document index of turns | **belief state (world model)** |
| a changed fact | two similar passages | **superseded** (bi-temporal, history kept) |
| stale evidence | retrieved forever | **retired** (belief–evidence consistency) |
| write policy | store everything | **surprise-gated** (predictive coding) |
| forgetting | none (grows forever) | salience-decay sweep |
| queryable across time | no | **time-travel** (`recall(as_of=t)`) |
| read path | — | **no LLM call** |

Read the 2-page paper: **[`paper/tenet.md`](paper/tenet.md)**.

## Results (LongMemEval_S, n=40, gpt-4o reader — honest, reproducible; detail in [`docs/BENCHMARK.md`](docs/BENCHMARK.md))

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
  **reader-robust** across `gpt-4o-mini` / `gpt-4o` / `claude-opus-4.8` (≈1.6–1.7×).
- **Churn-robust:** 100% at every update level while RAG collapses to 50% — the collapse holds
  under a gpt-4o reader, so it's *structural*, not reader weakness.
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

## The agent

Tenet ships as a personal assistant ([`src/agent.py`](src/agent.py)) on Qwen Cloud:
```
you › Hi! I'm Alex, I live in Montreal and work as a data analyst.
assistant › Nice to meet you, Alex! How's the analyst work in Montreal?   [remembered 2 facts]
… weeks later …
you › I moved to Toronto and got promoted to senior analyst!
you › Where do I live and what's my job now?
assistant › You live in Toronto and you're a senior analyst. Congrats on the promotion!
```
```bash
python src/agent.py            # interactive assistant
python scripts/demo_agent.py   # the scripted story (video walkthrough)
```

## Quickstart
```bash
cp .env.example .env && chmod 600 .env      # add DASHSCOPE_API_KEY (Qwen Cloud)
pip install -r requirements.txt
python scripts/smoke_test.py                # verify connectivity
uvicorn api:app --host 0.0.0.0 --port 8000  # (from src/) HTTP API incl. POST /chat
python src/mcp_server.py                     # or the MCP server (learn/recall/forget/stats)
```

## Reproduce the paper
```bash
python scripts/test_memory.py ; python scripts/test_tenet_e2e.py               # capabilities
python scripts/bench_horizon.py --principals 12 --k 6 --updates 2,4,6,8,10,12  # Fig. 1 (churn)
python scripts/lme_recall.py --limit 40 --k 10 --qa --seed 2                    # efficiency point
python scripts/lme_recall.py --limit 40 --k 10 --qa --seed 2 --expand 20        # parity point (budget-capped)
python scripts/bench_knowledge_update.py --principals 4                         # ablation + efficiency
# off-Qwen: prefix with  LLM_PROVIDER=openrouter EMBED_PROVIDER=local OPENROUTER_MODEL=openai/gpt-4o-mini
```

## Architecture
![architecture](docs/architecture.svg)

Two layers over one bi-temporal store (beliefs + evidence), two surfaces (MCP + HTTP),
powered by Qwen Cloud (Alibaba Cloud Model Studio). Details: [`docs/DESIGN.md`](docs/DESIGN.md),
positioning vs Mem0/Zep/Letta/Mastra: [`docs/COMPARISON.md`](docs/COMPARISON.md).

## Repository
```
paper/tenet.md            the paper
src/  agent.py            the assistant
      tenet.py memory.py distill.py config.py   the belief-state memory engine
      mcp_server.py api.py alicloud_oss.py       surfaces + Alibaba Cloud deploy
scripts/ demo_agent.py    video walkthrough
         bench_horizon.py bench_knowledge_update.py lme_recall.py   benchmarks
         test_memory.py test_tenet_e2e.py smoke_test.py            tests
docs/ BENCHMARK.md COMPARISON.md DESIGN.md DEPLOY.md SOTA.md  architecture.svg horizon.svg
```

## Citation
```bibtex
@misc{tenet2026,
  title  = {Tenet: Agent Memory as a Self-Consistent World Model},
  author = {Anas},
  year   = {2026},
  note   = {Global AI Hackathon with Qwen Cloud, Track 1},
  url    = {https://github.com/Nas01010101/tenet}
}
```

## License
MIT — see [LICENSE](LICENSE).
