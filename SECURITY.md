# Security Policy

Tenet is a research project (single maintainer), not a hardened production
service — treat it accordingly, but real vulnerabilities are still taken
seriously and fixed.

## Reporting a vulnerability

**Do not open a public issue for a security report.** Use GitHub's private
reporting instead:

[github.com/Nas01010101/tenet/security/advisories/new](https://github.com/Nas01010101/tenet/security/advisories/new)

Include what you found, how to reproduce it, and its impact if you can. I'll
acknowledge reports within a few days and fix confirmed issues as soon as
practical — this is a solo-maintained repo, so there's no formal SLA, but
security reports get priority over everything else in the queue.

## Scope

Roughly: the memory core (`src/memory.py`, `src/tenet.py`), the distillation
path (`src/distill.py`), the HTTP API (`src/api.py`), and the MCP server
(`src/mcp_server.py`). Things like "the demo assistant's REPL doesn't sanitize
input" are lower priority than anything touching the store itself,
credentials, or the provider clients in `src/config.py`.

## API key hygiene

Tenet talks to Qwen Cloud (DashScope), optionally OpenRouter, and optionally
Alibaba Cloud OSS. All secrets are read from `.env` (see `.env.example`) via
`src/config.py` — nothing else in the codebase should read credentials
directly from `os.environ`.

- `.env` is gitignored and should be `chmod 600`. Never commit it.
- If you accidentally commit a real key, rotate it immediately (in the
  DashScope/OpenRouter/Alibaba console) — assume a key that ever touched git
  history is burned, even after a force-push.
- `config.require()` fails loudly on a missing or placeholder
  (`sk-xxxx...`/`<...>`) secret rather than silently proceeding — don't work
  around that by hardcoding a key inline.
- Use least-privilege credentials where the provider supports it (e.g. a
  fine-grained GitHub PAT, an Alibaba Cloud RAM user scoped to ECS/FC + OSS
  rather than the root AccessKey).
