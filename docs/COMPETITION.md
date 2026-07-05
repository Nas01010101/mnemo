# Competition Reference — Global AI Hackathon Series with Qwen Cloud

> Canonical, verified from the official rules + overview pages on 2026-07-05.
> Source of truth: https://qwencloud-hackathon.devpost.com  (and `/rules`)
> Sponsor: **Alibaba Cloud** (Singapore) · Administrator: **Devpost** · 6,311 participants

## Dates (Pacific Time)
| Milestone | When |
|---|---|
| Submission Period | May 26, 2026 8:00am – **Jul 9, 2026 2:00pm PT** (hard deadline) |
| Judging Period | Jul 10 – Jul 31, 2026 |
| Winners announced | ~Aug 7, 2026 |

## Prizes ($70K+ total; $45K cash)
- **Each of 5 tracks:** $7,000 cash + $3,000 cloud credits + blog feature + swag + Ambassador invite (1 winner/track)
- **Blog Post Award:** $500 cash + $500 credits × 10 winners
- **Top 10 Honorable Mention:** $500 cash + $500 credits × 10 winners
- Standout teams: invite to the **AI Catalyst program**

## Tracks (pick ≥1)
1. **MemoryAgent** — persistent memory across multi-turn/cross-session; efficient store+retrieval, timely forgetting, recall of critical memories under limited context.
2. **AI Showrunner** — video-gen (Wan / HappyHorse) short-drama pipeline: script→storyboard→video→edit; narrative + multimodal orchestration under a token budget. *Highest token allowance.*
3. **Agent Society** — multi-agent collaboration/negotiation; task decomposition, role assignment, conflict resolution, **measurable efficiency gain vs single-agent baseline**.
4. **Autopilot Agent** — end-to-end business workflow automation; ambiguous inputs, external tool use, **human-in-the-loop checkpoints**. Production-ready, not toy.
5. **EdgeAgent** — Qwen-powered physical devices; edge-cloud orchestration under bandwidth/latency limits, privacy-aware, graceful offline degradation. (Sponsor may demand physical hardware access.)

## Judging — VERIFIED official breakdown
**Stage 1 (pass/fail):** reasonably fits the track theme AND reasonably applies the required Qwen APIs/SDKs.
**Stage 2 (weighted):**
| Criterion | Weight | What they reward |
|---|---|---|
| **Innovation & AI Creativity** | **30%** | Sophisticated Qwen Cloud API use (custom skills, **MCP integrations**); algorithmic/engineering innovation, novel components, perf optimization |
| **Technical Depth & Engineering** | **30%** | Architecture quality (modularity, scalability, error handling); clean code + non-trivial logic; advanced patterns |
| **Problem Value & Impact** | **25%** | Real-world pain point solved; productization / OSS-community potential |
| **Presentation & Documentation** | **15%** | Clear technical demo (key logic visualized); clear docs incl. architecture docs |
> Note: the main overview page mislabels the first two criteria's descriptions; the `/rules` page above is authoritative. Judging may use expert panels, peer review, AND automated AI analysis. Ties broken by highest score in the first listed criterion (Innovation), then next, etc.

## Mandatory submission checklist
- [ ] **Public open-source repo** with a **detectable LICENSE** visible in the GitHub *About* section — all source, assets, run instructions.
- [ ] **Proof of Alibaba Cloud deployment** — BOTH: (a) a short recording *separate from the demo* showing the backend running on Alibaba Cloud, AND (b) a **link to a code file in the repo** that demonstrably uses Alibaba Cloud services/APIs. *(Backend must actually run on Alibaba Cloud — not just call the model API.)*
- [ ] **Architecture diagram** — Qwen Cloud ↔ backend ↔ database ↔ frontend.
- [ ] **Demo video** <3 min, **public** on YouTube / Vimeo / (rules say Youku; overview says Facebook — **use YouTube**, common to both). Must show the project functioning. No unlicensed music/trademarks.
- [ ] **Text description** of features + functionality.
- [ ] **Track identified** on the submission form.
- [ ] **Testing access** — project free + accessible for judges until judging ends; if private, include login creds in testing instructions.
- [ ] **English** (or English translation of video, description, testing instructions).
- [ ] *(Optional)* public **blog/social post** on the build journey, link included → Blog Post Prize eligible.

## Rules nuances that matter for US
- **New & Existing:** a pre-existing project is allowed ONLY if **significantly updated after May 26, 2026**, and you must **explain the update**. → Our harness/memory system is pre-existing; a submission must be a *new build on top* that we can document as new work.
- **IP:** must be your original, solely-owned work; open-source components OK **only if you build on/enhance** them and comply with their licenses.
- **Multiple submissions** allowed if each is substantially different.
- **Restricted jurisdictions:** excluded wherever qwencloud.com registration is unsupported or under sanctions; must be age of majority.
- **No preferential support** from Sponsor/Administrator.

## Access / credits (verified)
- Sign up: https://www.qwencloud.com → check free quota at https://home.qwencloud.com/benefits
- Not free-trial eligible? Request **$40 Qwen Cloud voucher** via the hackathon coupon form. **You pay any usage above $40.** ($3,000 credits figure is a *prize*, not starting credit.)
- Discord + support: `global.hackathon@alibaba-inc.com`; docs index: https://docs.qwencloud.com/llms.txt

## Non-automatable submission steps (need browser / accounts)
Devpost has **no public submission API** — registration + the "Enter a Submission" form are web-only. Demo video needs a screen recording + YouTube upload. Proof-of-deploy needs a screen recording. These require your Devpost/YouTube accounts and browser automation or manual steps.
