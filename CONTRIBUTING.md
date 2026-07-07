# Contributing to Tenet

This is a solo-maintainer research repo (built for the Qwen Cloud hackathon,
Track 1: MemoryAgent), not a corporate OSS project. No CLA, no formal RFC
process. If something's broken or missing, open an issue or send a PR.

## Dev setup

```bash
git clone https://github.com/Nas01010101/tenet.git
cd tenet
pip install -e .
cp .env.example .env && chmod 600 .env
```

You don't need a Qwen Cloud API key to develop. Set this in `.env` (or export
it) to run everything against a local embedder instead:

```
EMBED_PROVIDER=local        # sentence-transformers (bge-small), no API key
LLM_PROVIDER=openrouter     # only needed if you're exercising ingest()/distill
```

`EMBED_PROVIDER=local` requires `pip install sentence-transformers` (see
`requirements.txt`). Chat/distillation (`Tenet.ingest`, the assistant) still
needs a working LLM provider — `DASHSCOPE_API_KEY` for Qwen Cloud, or
`LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY`. `MemoryCore` itself
(store/recall) never calls an LLM, so most of the core is testable offline.

## Running the tests

Two test scripts, both plain `python`, no test framework:

```bash
python scripts/test_memory.py      # deterministic, no API key needed (EMBED_PROVIDER=local)
python scripts/test_tenet_e2e.py   # exercises ingest()/distill — needs a working LLM provider
```

`scripts/test_memory.py` is what CI runs (`.github/workflows/test.yml`) — it
drives `MemoryCore` directly (store/recall/supersession/time-travel/forgetting)
with a controllable fake clock, so it's fast and deterministic.
`scripts/test_tenet_e2e.py` goes through the full `Tenet.ingest()` path
(raw message → LLM distillation → store), so it needs real API access and
isn't run in CI. Run it locally before touching `src/distill.py` or
`src/tenet.py`'s ingest path.

Both exit nonzero on failure and print a `FAILURES:` list — read that first.

## Benchmarks

If your change touches recall ranking, supersession, or forgetting, re-run
the relevant benchmark and update `docs/BENCHMARK.md` with the new numbers —
don't just claim it's better. Benchmark scripts live in `scripts/` (`bench_*.py`,
`lme_recall.py`, `lme_bench.py`). Read `docs/BENCHMARK.md` first for the
protocol (reader model, seed, what's compared against what) so a new number is
apples-to-apples with what's already published.

## PR conventions

- Keep diffs surgical — match the file you're editing, don't reformat unrelated
  code.
- If you fix a bug, add a test that fails before your fix and passes after.
- If you touch `docs/BENCHMARK.md` numbers, say which script produced them and
  with what seed/config, in the PR description.
- No new hard dependency without a reason in the PR description — `EMBED_PROVIDER=local`
  and the OpenRouter fallback exist so this stays runnable without any single
  vendor's API key; don't quietly break that.
- Comments should explain *why*, not *what* — the code already says what.

## Where to discuss

Bugs, design questions, feature ideas: [GitHub issues](https://github.com/Nas01010101/tenet/issues)
or [Discussions](https://github.com/Nas01010101/tenet/discussions) on the repo.
There's no chat/Discord for this project.
