## What this does

<!-- One or two sentences. Link an issue if there is one. -->

## Checklist

- [ ] `python scripts/test_memory.py` passes (`EMBED_PROVIDER=local`, no API key needed)
- [ ] `python scripts/test_tenet_e2e.py` passes, if this touches `ingest`/`distill`
      (needs a working LLM provider — `DASHSCOPE_API_KEY` or `LLM_PROVIDER=openrouter`)
- [ ] If this changes recall ranking, supersession, or forgetting: benchmark numbers in
      `docs/BENCHMARK.md` are either unchanged or updated with the new numbers + how
      they were produced (script, seed, reader model)
- [ ] No new hard dependency without a reason below (or it's justified in the PR body)
- [ ] Diff is surgical — matches surrounding style, no unrelated reformatting

## Notes

<!-- Anything a reviewer should know: tradeoffs, things you considered and rejected, etc. -->
