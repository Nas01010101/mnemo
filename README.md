# qwen-hackathon

Entry for the **Global AI Hackathon Series with Qwen Cloud** (Alibaba Cloud / Devpost).
- Prize pool: $70K+ · 5 tracks · **Deadline: Jul 9, 2026, 2:00pm PT**
- Hub: https://qwencloud-hackathon.devpost.com

## Setup

```bash
cp .env.example .env && chmod 600 .env   # already done; fill in real keys
# get DASHSCOPE_API_KEY from https://www.qwencloud.com
pip install openai            # or: uv pip install openai
python scripts/smoke_test.py  # verifies key + endpoint (expects "pong")
```

## Secrets

All keys live in `.env` (gitignored, chmod 600). Code reads them **only** via
`src/config.py` — never `os.environ` directly. `require()` fails loud on a
missing/placeholder key. Template + docs of every key: `.env.example`.

## Submission checklist (from the rules)

- [ ] Public repo + detectable LICENSE (visible in GitHub About)
- [ ] Backend running on Alibaba Cloud + proof-of-deploy recording + link to the code file that calls AliCloud APIs
- [ ] Architecture diagram (Qwen Cloud ↔ backend ↔ DB ↔ frontend)
- [ ] ~3-min public demo video (YouTube/Vimeo/Facebook)
- [ ] Text description + track identified
- [ ] (optional) blog/social post → Blog Post Prize

## Track

_TBD — see `docs/` once chosen._

## API quick reference

| Thing | Value |
|---|---|
| Base URL (OpenAI-compat) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| Responses/Conversations URL | `https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1` |
| Default model | `qwen3.7-plus` (1M ctx, tools, structured output) |
| Docs index (llms.txt) | https://docs.qwencloud.com/llms.txt |
