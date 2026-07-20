# Tenet — 3-minute demo video script

Built from what actually wins Devpost AI hackathons (judges watch the video first, back-to-back,
and score requirements + storytelling): **lead with a problem the judge feels, show the product
working within ~90s, make the invisible mechanism visible, address every rubric axis in one line,
and never let the demo stall.** Screen + voice recording, uploaded **public** to YouTube, marked
**"Not for Kids"**. Keep it under 3:00 — judges aren't required to watch past it.

Rubric this maps to: Innovation 30 / Technical Depth 30 / Impact 25 / Presentation 15 (ties → Innovation).

---

## The script (shot by shot)

**0:00–0:18 — Hook: a problem the judge feels (no "hi, we're…")**
> "Give an LLM agent 'memory' today and it's really just RAG over a transcript — append and
> retrieve. That breaks the moment a fact *changes*: you move cities, change jobs — and the agent
> still has both the old and new answer sitting in its retrieval pool, and sometimes serves the
> stale one. Tenet fixes that."

**0:18–1:05 — The wow, live and visible: bi-temporal supersession you can watch**
Screen: the belief-ledger web demo (already deployed on **Alibaba Cloud Function Compute**).
Pre-warm it (one `/health` hit) before recording so the first write is instant.
> "This is Tenet's belief state, live. I tell it: *I live in Boston, I work at Acme.*"
Type it into the demo chat → two belief cards appear.
> "Now: *I just moved to Seattle.*"
The `user::residence` card **strikes through Boston and shows Seattle** — on screen.
> "It didn't keep both. The old value is *superseded* — retired to history, not deleted — so
> 'where do I live?' answers Seattle, only. And I can scrub the time-travel slider back to before
> the move and it correctly says Boston. Nothing here called an LLM — the read path is pure
> vectors plus closed-form decay math."

**1:05–1:35 — The stress test RAG structurally can't pass**
Screen: the "churn" panel / `docs/horizon.svg`, or run `tenet bench run horizon`.
> "Here's why this matters at scale. Update one fact over and over. Naive RAG collapses from 100%
> to 50% correct once the stale versions crowd the top-k. Tenet holds **100%** — because it's
> resolved at write time, not re-derived at every read."

**1:35–2:10 — How it works + the technical edge (one architecture view)**
Screen: `docs/architecture.svg`.
> "Built end-to-end on Qwen Cloud: `qwen3.6-flash` distills each message into an atomic
> `subject::attribute` fact, `text-embedding-v4` embeds, `qwen3.7-plus` reads. Every fact carries
> event time and transaction time — that's the bi-temporal model Zep and Engram also use — but
> Tenet is the first *embedded, deterministic* version: no graph database, no LLM in the
> supersession path, no LLM on the read path. `pip install`, SQLite + numpy, reads ~11 ms flat
> from 1k to 100k facts."

**2:10–2:40 — Proof + reach (Impact)**
Screen: README "Results at a glance" / `docs/BENCHMARK.md` §6.
> "On the standardized MemoryAgentBench, single-hop **97.0%** — above the published gpt-4o-tier
> result — and multi-hop **45.8%**, 1.5× the published SOTA, on a *local 7B* reader at $0. It's
> MCP-native, drops into LangGraph / LlamaIndex / Mem0, and it's already the belief board inside
> our Track-3 agent society. One memory engine, real products."

**2:40–3:00 — Honest close**
> "It's not a better one-shot retriever — strong RAG ties it on raw recall, and we report that.
> Its edge is staying *correct* when facts change, offline if you need it, with a belief state you
> can actually open and read. That's Tenet. Thanks for watching."

---

## Exact commands per beat

Clean store per take + local-embedder cache (only if a beat runs locally instead of the live box):
```bash
export TENET_DB_PATH=/tmp/tenet_demo.db
export HF_HOME=~/.cache/huggingface TRANSFORMERS_CACHE=~/.cache/huggingface
```

| beat | time | on screen | how |
|---|---|---|---|
| Hook | 0:00–0:18 | title card / talking head | voiceover only |
| Supersession wow | 0:18–1:05 | belief-ledger web demo (FC box, or `uvicorn tenet.api:app --port 8000` + `src/tenet/static/index.html`) | pre-warm `/health`, then type the 3 messages; strike-through + time-travel slider are the money shot |
| Churn stress | 1:05–1:35 | `docs/horizon.svg` or `tenet bench run horizon` | static slide, or run the command with output pre-captured |
| Architecture | 1:35–2:10 | `docs/architecture.svg` | static |
| Results | 2:10–2:40 | README results table / `docs/BENCHMARK.md` §6 | static scroll |
| Close | 2:40–3:00 | logo / repo URL card | voiceover |

Zero-key alternative for the wow (if you'd rather show a terminal than the web box):
`pip install -e ".[local]" && python examples/00_zero_key_demo.py` (from a clone; pre-fetch the
bge-small model once so the take is airplane-mode-safe) or `tenet timeline --all` after two
`tenet remember` calls — it renders the superseded chain (○ Boston [superseded] → ● Seattle).

## Never-stall checklist (do before the real take)
- [ ] **Pre-warm the FC box** (`curl .../health` once) — cold start can add ~8s to the first write.
- [ ] Point `TENET_DB_PATH` at a fresh scratch file so each take starts from an empty ledger.
- [ ] Have the three messages copied to clipboard — don't type live.
- [ ] Pre-fetch the local embedder model if using the zero-key terminal beat.
- [ ] Dry-run the whole flow once end-to-end and **time it** — cut to fit under 3:00, trim the churn
      beat first if long.
- [ ] YouTube: **public** + **"Not for Kids"**; grab the link as soon as upload starts.

## One-line-per-rubric-axis (say these, or ensure the visuals land them)
- **Innovation (30):** first embedded, deterministic bi-temporal memory — supersession by stable-key
  collision, zero LLM in the write-conflict or read path.
- **Technical Depth (30):** one core shared by CLI / MCP / HTTP / adapters; ~11 ms flat reads;
  learned drift model; fail-loud provider layer; every number a Wilson CI.
- **Impact (25):** 97.0 SH above gpt-4o tier at $0 local; five shipped deployment patterns; already
  load-bearing in a second product.
- **Presentation (15):** the belief state is visible and readable — supersession happens *on screen*.

## AliCloud proof (for the separate/optional deploy shot + the form field)
Live backend on Alibaba Cloud Function Compute: `https://tenet-demo-wrenarokun.ap-southeast-1.fcapp.run`.
Code-file proof: `src/tenet/config.py` (`dashscope-intl.aliyuncs.com`) + `src/tenet/alicloud_oss.py`.
