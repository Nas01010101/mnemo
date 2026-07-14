# Autoresearch program — Tenet supersession thresholds

## Objective
Find the setting of Tenet's three supersession-governing thresholds that MINIMIZES
knowledge-churn retrieval error — i.e. maximizes how reliably `recall()` surfaces the
*current* value of a churned attribute and suppresses stale ones — WITHOUT any LLM call.

## Metric (ungameable, runner-owned)
`error = 1 - (subjects whose recall ranks the CURRENT city first) / S`, computed by
`target.py` on a FIXED seeded synthetic corpus and written to `$AUTORESEARCH_RESULTS`.
Lower is better. The corpus generator, seed, and scorer are the JUDGE and are out of bounds.

## Editable surface
ONLY the three thresholds, supplied by the driver via `$AUTORESEARCH_CONFIG`:
- `consistency_threshold` ∈ [0.55, 0.85]   (read-time belief–evidence consistency; default 0.70)
- `tau_key`               ∈ [0.65, 0.88]   (embedding key-resolution attribute-cosine; default 0.78)
- `text_floor`            ∈ [0.50, 0.78]   (fact-text cosine floor for key resolution; default 0.66)

`target.py` itself (corpus, scoring) is NOT a search knob — it only translates a config to a metric.

## Boundaries (hard rails — learned the expensive way)
- **Embeddings only.** `EMBED_PROVIDER=local` (bge-small, ~130 MB). **NO LLM reader, NO Qwen
  Cloud call, NO ollama / local large model.** A 14B local model on this 16 GB Mac is forbidden.
- Time-box per eval via the harness; fail-closed (error/no-metric → reverted, never kept).
- Aborts on `~/.claude/STOP`; stop when `mcp__budget_guard__check` != ok.

## Proxy caveat
This is a fast SEARCH PROXY for the distiller-key-drift + stale-raw-echo regime, not the final
judge. Any winning config MUST be re-verified on the real benchmarks (`scripts/bench_churn.py`,
`scripts/bench_supersession_firing.py`) and must not regress the deterministic suites before it
is adopted as a default. Current shipped defaults (0.70 / 0.78 / 0.66) are the incumbent baseline.

## Run 1 result (2026-07-12) — a proxy-infidelity NEGATIVE, do NOT adopt
16-eval sweep (6 random + 10 BO, bge-small, LLM-free): default (0.70/0.78/0.66) → error 0.667;
BO converged to error 0.0 at ~{consistency 0.81, tau_key 0.66, text_floor 0.70}. **This is a
reward-hacked proxy win, NOT a real improvement.** The proxy contains only synonym-drift POSITIVES
(keys that *should* collapse) and NO adversarial distinct-attribute NEGATIVES (keys that must NOT
collapse — e.g. `pet` vs `pet_name`). So it rewards ever-more-aggressive key-resolution (lower
`tau_key`), which is precisely the direction that re-introduces the false-supersession bug that
`tau_key=0.78` + `text_floor=0.66` were tuned to prevent on the *labeled* firing set (which DOES
have negatives, `scripts/bench_supersession_firing.py`). **Defaults left unchanged.**
**Fix before Run 2:** add distinct-attribute negatives to the corpus and make the metric a
FIRE-precision/recall F-score (penalize false collapses), so the proxy can't win by over-firing.
Then re-sweep, then verify on the real firing benchmark before adopting anything.

## Run 2 result (2026-07-13) — judge v2 with negatives: NO threshold wins; a real
## false-supersession class discovered at the shipped default

Judge v2 (`target.py`, journal `journal.jsonl`; Run 1 rows archived to `journal_run1.jsonl`)
adds one FALSE-SUPERSESSION NEGATIVE per subject: a `subjN::work_city` fact — embedding-near
the residence synonyms and sharing the subject token — that must remain CURRENT after the
residence churn. Objective: `error = 1 − F1(currency-recall, negative-survival)`.

16-eval sweep (6 random + 10 BO) + manual component probes:

| config (ct / tau / floor) | recall | negative-survival | error |
|---|---:|---:|---:|
| **0.70 / 0.78 / 0.66 (shipped default)** | 0.333 | **0.000** | 1.000 |
| 0.55 / 0.65 / 0.50 (Run 1's "winner") | 1.000 | 0.000 | 1.000 |
| 0.70 / 0.84 / 0.66 | 0.333 | 1.000 | 0.500 |
| 0.70 / 0.88 / 0.78 | 0.333 | 1.000 | 0.500 |

Three findings, in order of importance:
1. **Run 1's reward-hack is confirmed quantitatively**: its tau=0.66 "winner" has perfect
   recall and ZERO negative survival — the hack traded silent false supersessions for recall,
   invisible to a positives-only metric.
2. **The shipped default (tau 0.78) has a real, previously-unmeasured false-supersession
   class**: semantically-adjacent distinct attributes (`work_city` vs `city`/`home_city`).
   The 2026-07-10 firing-set negatives were shared-WORD probes; embedding-NEAR distinct
   attributes were not covered, and they fire falsely at 0.78.
3. **No threshold in the box separates the classes** (best F1 = 0.5 across the entire sweep;
   flat plateau). The slug-cosine of `work_city`↔`city` (must NOT collapse) overlaps the
   synonym pairs `home_city`↔`city` (MUST collapse). A token-structure rule can't separate
   them either — `work_city` vs `city` and `home_city` vs `city` are structurally identical
   (strict token superset + one non-sub-attr extra token); only the semantics of the extra
   token ("work" vs "home") differs. Threshold tuning is structurally insufficient here.

**Decision: defaults unchanged** (raising tau to 0.84 would fix the negatives on this proxy
but forfeits the synonym-drift recall that key-resolution exists to provide — and the
PersonaMem +2.9pp win was measured at 0.78). The false-fire class is real but narrow
(same-subject attributes that are embedding-adjacent AND text-adjacent); candidate real fix
is a conditional stricter text-floor when the key-token sets are in a strict-superset
relation with a non-meta extra token — needs the labeled firing benchmark (fresh distillation;
RTX) + PersonaMem regression gates before touching `_value_compatible`. Filed as follow-up.

## Run
```
AUTORESEARCH_MAXIMIZE=0 python ~/code/claude-config/scripts/autoresearch/sweep.py experiments/autoresearch_thresholds
```
