# Measurement methodology & what each number can / can't support

An adversarial self-audit of *how* every headline number in [`BENCHMARK.md`](BENCHMARK.md)
is computed, and the exact claim each measurement licenses. Scope: methodology only — no
measured value was changed by this audit. **Headline finding: no number is computed
*wrong*** (the McNemar test, Wilson intervals, and SubEM scorer are all correct
implementations). The genuine defects are in *interpretation wording* and one *harness
contamination risk*; both are fixed as caveat wording in BENCHMARK.md and listed below.

## How we measure (one paragraph per instrument)

- **Accuracy + Wilson 95% CI.** Per-cell accuracy is `correct/scored` (API failures
  excluded from the denominator, never scored wrong). CIs are the Wilson score interval
  (`bench_factcon.wilson_ci`) — verified against the closed form; correct for proportions,
  and correctly *wide* at small n.
- **Paired comparison (McNemar).** Arms answer the *same* items (paired). Significance is
  the **exact two-sided binomial sign test on discordant pairs** (`bench_persona.mcnemar`,
  `bench_locomo`) — the exact test, so no continuity correction is needed or used. `b` =
  ours-right/theirs-wrong, `c` = the reverse, `p = min(1, 2·P(X ≤ min(b,c) | n_d, ½))`.
- **Deterministic MC (§13 PersonaMem).** 4-way multiple-choice; we parse the letter and
  exact-match. **No LLM judge** → no judge-comparability caveat. Options are shuffled with a
  per-item deterministic seed, *identical across arms* (paired).
- **SubEM (§6 FC, §9 ChurnBench).** `normalize_answer` (lowercase, strip punctuation +
  articles, collapse whitespace) then substring match — matches the MAB-official metric;
  the MAB reader prompt is copied verbatim. 13 hand-checked cases pass
  (`test_churnbench.py`), including the stale-vs-current non-collision integrity case.
- **LLM judge (§12 LoCoMo, §7 MAB-AR).** A judge model grades semantic correctness vs gold.
  §7 uses a **cross-family** judge (gpt-4o judging gpt-4o-mini answers); §12 uses
  **qwen3.7-plus judging qwen3.7-plus** (same-family — see Defect 4).
- **Reader token cost.** `tokens ≈ chars/4` (`lme_recall.py`) — an approximation, see
  Defect 6.

## Per-construct verdict

| # | construct | verdict | severity |
|---|---|---|---|
| 1 | Blind/no-memory control (§13) | **valid mechanism, wording overstates** | low |
| 2 | McNemar paired test | **test correct; "dead tie" interpretation overstates** | **medium** |
| 3 | Wilson CIs | valid | low (tiny-n prose only) |
| 4 | Same-family LLM judge (§12) | valid concern, **empirically small (100% cross-judge agreement measured)** | low |
| 5 | SubEM / deterministic scorers | valid — matches official metric | none |
| 6 | Token count `chars/4` | valid; **bias is conservative against our own claim** | low |
| 7 | Seeded, paired sampling | valid | none |
| 8 | Batched-10-per-CLI reader (§2 reader-generality) | **contaminable harness under the headline result** | **HIGH** |

### Defect 1 — "blind = no memory" overstates (low)
The §13 blind arm gets the static profile one-liner + question + options, **zero retrieved
turns** (`bench_persona.py:214`), and conversation ingestion drops the persona card
(`flatten_history`), so no answer text leaks through a second path. The control is *sound*:
retrieval arms beat blind by **+16pp overall (CI-separated)**, which licenses exactly one
claim — **retrieved memory is load-bearing** (adds ≥16pp). It does **not** license "the
Tenet≈RAG tie is real"; that is a *separate* claim resting entirely on Defect 2's test.
34.4% is a **profile + option-plausibility floor, not a 25% random floor** — "no memory"
in the table header is imprecise. *Fix: relabel "no retrieved memory (profile only)"; state
the +16pp as the load-bearing evidence and keep it separate from the tie claim.*

### Defect 2 — "dead tie" is an underpowered null, not proven equivalence (medium)
The test and the p-value are correct (`p=0.92`). But p=0.92 with the observed 0.4pp gap
(Tenet 50.5 vs RAG 50.9, 245 vs 247 of 485) implies **~100 discordant pairs**, and a
sign test on 100 discordant pairs has **80% power only for a ~6pp accuracy gap** (computed:
gap=6pp → power 0.80; gap=4pp → 0.44). So the study **cannot distinguish "equivalent" from
"a true difference up to ~6pp"** — "dead tie" claims equivalence the design can't establish.
*Fix: "statistically indistinguishable at n=485 (MDE ≈ 6pp at 80% power); a true difference
below ~6pp is not excluded."* (§12 LoCoMo `p=0.031`, 125 discordant, **is** a real detected
difference — well-powered, no change needed.)

### Defect 4 — same-family judge, quantified (low)
§12 grades qwen answers with a qwen judge. We measured judge-family sensitivity directly:
30 real (question, gold, prediction) triples judged by **qwen3.7-plus vs Gemini-3.5-Flash**
with the identical LoCoMo judge prompt → **30/30 = 100% agreement** (both 80% yes-rate).
So judge choice barely moves verdicts on short factual answers; the caveat is correct to
state but **empirically small**. *Limitation:* measured on Sonnet-reader predictions (no
qwen-reader answer set is committed), so it bounds judge *variance* but does not isolate
same-family *leniency direction* — a qwen-reader re-judge is the clean future test.

### Defect 6 — `chars/4` bias runs the safe direction (low)
Real tokenizer (tiktoken cl100k) on representative context: distilled Tenet text **4.15
chars/tok**, raw RAG turns **3.82 chars/tok**. `chars/4` therefore *over*-counts Tenet
tokens (+4%) and *under*-counts RAG (−4%) — the true efficiency ratio is **~47% of RAG's
tokens, better than the reported 51%**. The per-token frontier claim is **conservative,
not inflated**. *Fix: one-line caveat; the claim stands.*

### Defect 8 — batched-10-per-CLI reader is the biggest weakness (HIGH)
The §2 reader-generality matrix (Sonnet-5 / gpt-5.5 / Gemini-3.5) — source of the headline
"**6/6 directional replication across reader families**" and the "multi-session weakness
**flips** under strong readers" claim — runs readers **10 questions per CLI invocation**
with prompt-level "isolation instructions." A single context window holding 10 items means
a reader **can attend to another item's retrieved context while answering** — the mitigation
is a soft instruction, not true isolation. This contamination sits under the strongest-looking
result in the document. (§12 LoCoMo and §13 PersonaMem are **not** affected — they issue
one independent `config.chat` call per item.) *Fix: caveat hard in §2; a headline
cross-reader claim should be re-run with one API call per item before it is load-bearing.*

## What each headline number licenses

- **§6 FC 97.0% SH / 45.8% MH pooled (2026-07-19 fixed-keys run)** — solid: deterministic
  SubEM, official prompt, Wilson CIs at n≥100, weak-backbone control, RAG arm reproduced
  exactly across runs. Supports "SH above the published gpt-4o-tier pooled; MH 1.5× the
  published SOTA." Pre-fix run (86.5/30.0) preserved in the artifact.
- **§9 ChurnBench / §9.1 fix** — solid: deterministic, seeded, hand-checked scorer. Supports
  the *falsified-then-partially-fixed* half-life story as stated (half-life 8, still < Mem0 32).
- **§13 PersonaMem tie** — supports "retrieved memory adds ≥16pp" and "no *detectable*
  Tenet-vs-RAG difference (MDE ≈ 6pp)"; does **not** support "proven equivalent" nor a
  supersession win (keyed supersession fires on only 3.8% of natural updates — the real
  reason, correctly diagnosed).
- **§2 reader-generality** — supports a *directional* (≥, not CI-separated) pattern across
  readers; **downgrade until re-run un-batched** (Defect 8).

## Single biggest weakness
**Defect 8** — the §2 batched-10-per-CLI reader harness under the headline reader-generality
result. It is the one place where a load-bearing claim rests on a contaminable measurement;
everything else is either sound or errs conservatively.
