---
name: Bug report
about: Something in Tenet is broken
title: ""
labels: bug
---

**What happened**
A short description of the bug.

**Repro**
Minimal steps or a snippet (e.g. a `Tenet.ingest`/`recall` call) that
reproduces it. If it needs specific data (e.g. two facts with the same key),
include it.

**Expected vs actual**
What you expected `recall`/`ingest`/etc. to return, and what you got instead.

**Environment**
- Tenet version / commit:
- Python version:
- Provider config: `LLM_PROVIDER=` / `EMBED_PROVIDER=` (no need to share keys)

**Anything else**
Logs, stack trace, or a link to `docs/BENCHMARK.md`/`docs/SOTA.md` if this is
a regression against a documented number.
