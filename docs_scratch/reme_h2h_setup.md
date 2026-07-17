# ReMe head-to-head — setup, findings, and projections

Companion to `scripts/bench_reme_h2h.py` (harness/orchestration + cost/stats),
`scripts/reme_h2h_driver.py` (the ReMe subprocess driver, split out to keep
the main file under the repo's 500-line cap), `scripts/reme_h2h_config.yaml`
(ReMe config), and `scripts/test_reme_h2h.py` (CI suite 17). Written under the
HARD CONSTRAINT of this task: zero live LLM/embedding API calls. Everything
below is either (a) verified by reading the installed `reme-ai` package's
source, or (b) computed from the LongMemEval dataset with no network calls —
never a live measurement, and each is labeled accordingly.

## 1. The venv

```
/private/tmp/claude-501/-Users-wissem-Projects-qwen-hackathon/cab15454-108f-4416-b742-2b0e18763a0b/scratchpad/reme-venv
```

Created from `~/.pyenv/versions/3.11.9/bin/python` (repo's `pyproject.toml` pins
`requires-python>=3.10`; `reme-ai` needs `>=3.11` — isolated venv, never
installed into the repo's own `.venv`, mirroring the existing
`scratchpad/mem0-venv` pattern for real `mem0ai`).

```
python -m venv reme-venv
reme-venv/bin/pip install 'reme-ai[core]'
```

**This path is session-scratchpad, not repo-scratchpad — it is ephemeral** (tied
to this Claude Code session's temp dir, not `qwen-hackathon/scratchpad/`) and
will not survive past this session. Before the live run, recreate it with the
same two commands at a durable path (e.g. `scratchpad/reme-venv` inside the
repo, gitignored, matching `mem0-venv`'s convention) — `pip install 'reme-ai[core]'`
resolves to the same `reme-ai==0.4.1.1` either way (verified below), so nothing
else in this doc changes.

Verified (no network beyond the `pip install` itself, which the task's
constraint scopes to "LLM/embedding API calls", not package installation):

```
$ reme-venv/bin/pip show reme-ai
Name: reme-ai
Version: 0.4.1.1
Home-page: https://github.com/agentscope-ai/ReMe

$ reme-venv/bin/python -c "import reme; print(reme.__version__)"
0.4.1.1

$ reme-venv/bin/python -c "from reme.reme import main; print(main)"
<function main at 0x...>       # CLI entry point importable

$ ls reme-venv/bin/reme
reme-venv/bin/reme              # console_script entry point present
```

Import + CLI presence: **confirmed.**

## 2. Correction to the design doc's assumption

The design doc's Part 1 assumed ReMe's "own recommended default" distillation
model was `qwen-flash`, based on a default arg in an older `eval_longmemeval_reme_retrieve.py`
script. **That script doesn't exist in `reme-ai==0.4.1.1`** — the package has
been restructured; the equivalent LongMemEval eval scaffolding now lives at
`reme/steps/benchmark/lme/*.py` + `reme/config/jinli_lme.yaml` (ReMe's own
bundled eval config). Reading that config directly: its `auto_memory` extraction
step (the "plus" LLM tier) is hardcoded to **`qwen3.7-plus`**, not a flash model
at all — there is no flash-tier distillation in ReMe's own shipped LongMemEval
config.

`scripts/reme_h2h_config.yaml` therefore pins ReMe's distiller to `qwen-flash`
as **our own deliberate choice** (documented in the config file's header), not
a verified ReMe default — matching Tenet's `qwen3.6-flash` distiller for a fair
flash-vs-flash cost comparison, per the design doc's explicit intent ("Alibaba's
two cheapest models... deliberately not picking a weaker extractor for ReMe").
Confidence: 95% this is the right call given the design doc's stated goal;
0% confidence it reflects ReMe's own recommendation (verified it does not).

ReMe's own general-purpose default config (`reme/config/default.yaml`, distinct
from its LME-specific one) confirms the OTHER design-doc claim correctly: BM25
retrieval is the default (`components.file_store.default.embedding_store` is
commented out; `keyword_index.default.backend: bm25` is always on). Our config
mirrors that with an env-toggle (`LME_EMBEDDING_STORE`, empty = BM25-only by
default).

## 3. Config validation (offline, no network)

`scripts/reme_h2h_config.yaml` is adapted from `reme-ai==0.4.1.1`'s own
`jinli_lme.yaml` (see the file's header comment for the exact diff). Validated
through ReMe's own config resolver — pure local YAML deep-merge + env-var
expansion, zero network:

```
$ reme-venv/bin/python -c "
from reme.config import resolve_app_config
c = resolve_app_config(config='scripts/reme_h2h_config.yaml', log_config=False)
print('as_llm.default.model  =', c['components']['as_llm']['default']['model'])
print('as_llm.plus.model     =', c['components']['as_llm']['plus']['model'])
print('as_llm.default.base_url =', c['components']['as_llm']['default']['credential']['base_url'])
print('as_embedding.model    =', c['components']['as_embedding']['default']['model'])
print('file_store.embedding_store =', repr(c['components']['file_store']['default']['embedding_store']))
"
as_llm.default.model  = qwen3.7-plus
as_llm.plus.model     = qwen-flash
as_llm.default.base_url = https://dashscope-intl.aliyuncs.com/compatible-mode/v1
as_embedding.model    = text-embedding-v4
file_store.embedding_store = ''      # empty -> BM25-only, ReMe's own default posture

$ LME_EMBEDDING_STORE=default reme-venv/bin/python -c "... same as above ..."
file_store.embedding_store = 'default'   # vector-retrieval stretch toggle confirmed
```

Reader/judge model (`qwen3.7-plus`), base URL (DashScope intl — same constant
as `src/tenet/config.py`'s `QWEN_BASE_URL`), embedder model (`text-embedding-v4`,
matching Tenet/RAG), and the BM25-default retrieval posture all resolve
correctly. **Config loads clean, zero live traffic.**

## 4. Does `reme start` make an LLM call before any job runs?

**Finding: no — 80% confidence, verified by reading the source, not by
executing it** (executing it would risk the zero-spend constraint if
credentials were even superficially valid, so this is a static-analysis
answer, not a measurement).

Trace of `reme start`'s startup path:

- `reme/utils/service_utils.py::precheck_start()` — only probes whether the
  configured TCP `host:port` is already bound (a local socket check via
  `find_reme()`), before deciding whether to proceed. No LLM/embedding call.
- `reme/config/config_parser.py::resolve_app_config()` — pure local YAML
  load + deep-merge + `${VAR}` expansion from `os.environ`. No network.
- `reme/application.py::Application.__init__()` calls `_init_components()` /
  `_init_jobs()`, which construct component objects (`_instantiate()` ->
  `backend_cls(**params)`) but never call `.start()` at this stage.
- `reme/components/as_llm/__init__.py::BaseAsLLM._start()` — the ONLY place an
  LLM client object gets built — constructs
  `credential = self.credential_cls(**kwargs.pop("credential", {}))` then
  `self.model = model_cls(credential=credential, parameters=parameters, **kwargs)`.
  This is object construction (an AgentScope `ChatModelBase` subclass), not a
  network call — no `.create()`/`.chat()` invocation happens here.
- `_start()` on `as_llm`/`as_embedding` components only runs when
  `Application._start()` (via `Application.start()`, invoked by the service
  layer when a job actually runs) walks the component list — i.e. even THIS
  step, which the CLI service (`service.backend: cli`, what we use) always
  hits before running its one configured job, is credential-object
  construction only.

So: **starting the server (or running the CLI one-shot service) is
network-free**; the first real LLM/embedding call happens only when a JOB is
actually invoked that calls the model (`auto_memory`, `bm25_search` does not —
it's pure BM25 lexical search against the local file store, no LLM/embedding
call at all; `search`/`vector_search` DO call the embedder if
`embedding_store` is enabled). This means our primary BM25-only config can run
`reme start job=bm25_search ...` against an ALREADY-INGESTED workspace with
zero DashScope traffic — only `job=auto_memory` (ingestion) touches the LLM.

**Not verified:** whether `precheck_start`'s local socket probe or the CLI
service's job-dispatch path has any other side effect I didn't trace (e.g. a
telemetry ping) — I did not grep exhaustively for outbound `httpx`/`requests`
calls beyond the `as_llm`/`as_embedding`/service-precheck paths above. If this
matters before the live run, a 30-second check: run
`reme start job=version config=scripts/reme_h2h_config.yaml` (the `version`
job — no LLM/embedding steps at all) with network access blocked
(e.g. `unshare -n` or a firewall rule) and confirm it still succeeds.

## 5. Token/cost projections — n=25/50/100 (NO API calls)

Computed by `scripts/bench_reme_h2h.py --project 25,50,100` straight from
`data/lme/longmemeval_s.json` (chars/4 heuristic, per design doc §2.3).
**Grounded inputs:** full-haystack char count per question (exact, read from
the dataset). **Measured-but-projected inputs:** the RAG/Tenet reader-context
sizes use the ACTUAL ratios from the real n=100 run in
`docs/lme_qwen_n100_result.txt` (rag_ctx/full_ctx = 7,689/505,619 = 1.521%;
tenet_ctx/full_ctx = 7,643/505,619 = 1.512%). **Unmeasured assumptions,
flagged:** ReMe's reader-context size (assumed = the RAG ratio — both are
top-k retrieval over a comparable pool, but ReMe's own context truly is
unmeasured), all distiller/reader OUTPUT token counts (reader ≈600 chars/~150
tok chain-of-note answer; distiller output ≈12% of its input chars — both
unmeasured without executing).

```
=== Token/cost projection — NO API calls, computed from the dataset ===
(source: longmemeval_s.json, 470 instances total)

--- n=25 (full haystack ≈12,707,169 chars, ≈3,176,792 tok total) ---
  [blind] subtotal=$0.008
    qwen3.7-plus         in=     8,823 tok  out=   3,762 tok  $0.008
  [rag] subtotal=$0.179
    qwen3.7-plus         in=    57,133 tok  out=   3,762 tok  $0.020
    text-embedding-v4    in= 3,176,792 tok  out=       0 tok  $0.159
  [reme] subtotal=$0.331
    qwen-flash           in= 3,176,792 tok  out= 381,215 tok  $0.311
    qwen3.7-plus         in=    57,133 tok  out=   3,762 tok  $0.020
  [tenet] subtotal=$0.490
    qwen3.6-flash        in= 3,176,792 tok  out= 381,215 tok  $0.311
    qwen3.7-plus         in=    56,844 tok  out=   3,762 tok  $0.020
    text-embedding-v4    in= 3,176,792 tok  out=       0 tok  $0.159
  TOTAL                $1.008

--- n=50 (full haystack ≈25,342,174 chars, ≈6,335,544 tok total) ---
  [blind] subtotal=$0.016
  [rag] subtotal=$0.357
  [reme] subtotal=$0.661
  [tenet] subtotal=$0.977
  TOTAL                $2.010

--- n=100 (full haystack ≈50,691,811 chars, ≈12,672,953 tok total) ---
  [blind] subtotal=$0.032
    qwen3.7-plus         in=    35,828 tok  out=  15,050 tok  $0.032
  [rag] subtotal=$0.713
    qwen3.7-plus         in=   228,546 tok  out=  15,050 tok  $0.080
    text-embedding-v4    in=12,672,953 tok  out=       0 tok  $0.634
  [reme] subtotal=$1.322
    qwen-flash           in=12,672,953 tok  out=1,520,754 tok  $1.242
    qwen3.7-plus         in=   228,546 tok  out=  15,050 tok  $0.080
  [tenet] subtotal=$1.955
    qwen3.6-flash        in=12,672,953 tok  out=1,520,754 tok  $1.242
    qwen3.7-plus         in=   227,393 tok  out=  15,050 tok  $0.079
    text-embedding-v4    in=12,672,953 tok  out=       0 tok  $0.634
  TOTAL                $4.022
```
(n=50's per-model rows omitted here for space — same shape as n=25/n=100;
`bench_reme_h2h.py --project 50` reproduces the full breakdown.)

Reading (per-ARM, the deliverable's own framing — not just per-model, so you
can see which arm is actually expensive): **tenet is the priciest arm**
(distiller + embeddings both), **reme is second** (distiller only, no
embeddings — BM25-default retrieval), **rag is cheap** (embeddings only, no
distiller), **blind is nearly free** (no context, no distiller, no
embeddings). `qwen-flash` = ReMe's distiller (auto_memory, every session
once). `qwen3.6-flash` = Tenet's distiller (identical order of magnitude by
construction — both distill the full haystack once). `qwen3.7-plus` = the
SHARED reader+judge, charged to whichever arm made the call. `text-embedding-v4`
= RAG's + Tenet's embedding pass; **ReMe's BM25-default retrieval issues zero
embedding calls** — a genuine, real cost/infra difference this table already
surfaces, not a modeling artifact.

Total: **≈$4.02 projected for n=100** (design doc's own pre-execution estimate
was $6-10 — same order of magnitude; the difference is mostly the design doc's
estimate not yet having the real n=100 rag/tenet context-size ratio to anchor
on, since it hadn't been computed at the time the design doc's projection was
written). n=500 (stretch) scales ≈linearly to **≈$20** by this projection.

Recommended run size given this budget and the Jul 20 deadline pressure noted
in the design doc §2.3: **n=100, all 4 arms** comfortably fits inside a $10
`--budget-cap` with margin; n=500 is affordable in dollars (≈$20) but the wall-
clock risk flagged in the design doc (concurrency against ReMe's own CLI
one-shot-process-per-job pattern, ~2 subprocess spawns/question for ingest+
retrieve, untested at scale) is the real constraint, not cost.

## 6. The go-command (once DashScope quota is confirmed)

```bash
cd /Users/wissem/Projects/qwen-hackathon
set -a; . ./.env; set +a

# 1. Recreate the venv at a durable path (session scratchpad won't survive):
~/.pyenv/versions/3.11.9/bin/python -m venv scratchpad/reme-venv
scratchpad/reme-venv/bin/pip install 'reme-ai[core]'

# 2. Smoke-test on n=2 with a tight cap first (design doc §2.3: "pilot at n=5-10
#    before committing the full run" — the ReMe distiller call-multiplier per
#    session is unverified without executing):
python scripts/bench_reme_h2h.py --n 2 --arms blind,rag,reme,tenet \
    --reme-venv scratchpad/reme-venv --budget-cap 1 \
    --out docs_scratch/reme_h2h_pilot.jsonl

# 3. Full MUST-have run (n=100, all 4 arms, ~$4 projected, resumable):
python scripts/bench_reme_h2h.py --n 100 --arms blind,rag,reme,tenet \
    --reme-venv scratchpad/reme-venv --budget-cap 10 \
    --out docs_scratch/reme_h2h_n100.jsonl

# Stretch (n=500, ~$20 projected, wall-clock risk flagged above):
python scripts/bench_reme_h2h.py --n 500 --arms blind,rag,reme,tenet \
    --reme-venv scratchpad/reme-venv --budget-cap 25 \
    --out docs_scratch/reme_h2h_n500.jsonl
```

No calls were made against this go-command in this task — HARD CONSTRAINT.
