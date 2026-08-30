# Benchmark

A LongMemEval-style benchmark for the two memory agents. Entry point:
[`run_benchmark.ipynb`](run_benchmark.ipynb).

## What it measures, and how it differs from LongMemEval

LongMemEval ([ICLR 2025](https://arxiv.org/abs/2410.10813)) evaluates a retriever
over a frozen chat log. Here the episodes are replayed **through the real manager
agent**, one session at a time, so what is scored is the whole pipeline —
extraction, routing, supersession, consolidation — and not a retriever over
pre-loaded facts.

That extra stage is why the error decomposition is three-way rather than the
paper's two-way. A wrong answer means something different depending on where it
broke, and the three places live in three different prompts:

| consolidation | retrieval | answer | points at |
| --- | --- | --- | --- |
| ✗ | any | ✗ | the consolidation prompt — the fact never reached memory |
| ✓ | ✗ | ✗ | indexing and search — it is on file and was not found |
| ✓ | ✓ | ✗ | the reading strategy — found and then misused |
| ✓ | ✓ | ✓ | working as intended |

## Categories

Six transfer from LongMemEval and are driven by its published datasets:

| LongMemEval | here |
| --- | --- |
| single-session-user / -assistant | `single-hop` |
| single-session-preference | `preference` |
| multi-session | `multi-hop` |
| knowledge-update | `knowledge-update` |
| temporal-reasoning | `temporal-reasoning` |
| `*_abs` question ids | `abstention` |

Three do not exist in LongMemEval, because it has no agent that writes its own
memory:

- **`supersede-integrity`** — the knowledge-update cases re-judged under a
  stricter prompt. LongMemEval accepts an answer carrying the old value next to
  the new one; that tolerance hides exactly the failure a curated memory has, so
  here the retired value must not be presented as current. Labelling it as
  history is fine. No extra ingestion: it is a second judgement on cases already
  run.
- **`procedural-retrieval`** and **`non-repetition`** — from the generated
  operational corpus (below).
- **`consolidation-quality`** is not a category at all: it is the difference
  between two runs over the same cases, one with consolidation and one without.

## Scales

| scale | corpus | sessions / case | what it tests |
| --- | --- | --- | --- |
| `small` | `longmemeval_oracle.json` | ~2 | the pipeline cold: one shard, nothing consolidated |
| `large` | `longmemeval_s_cleaned.json` | ~48 | the loaded pipeline, with distractors |

`longmemeval_m_cleaned.json` (~480 sessions, ~5 MB of history per question) is
readable by name but is not wired to a scale — it is a different order of spend.

Timestamps are left exactly as published. Stretching them over six months would
give the monthly sharding more to do, but the questions quote absolute dates
inside the message text, so a rescaled timeline would contradict its own
evidence. The generated corpus is where a long timeline is built on purpose.

## Getting the data

The LongMemEval datasets are not in this repository — they are gigabytes and
`.gitignore` keeps them out. Download them from the
[LongMemEval release](https://github.com/xiaowu0162/LongMemEval) and put them in:

```
benchmark/longmemeval/data/longmemeval_oracle.json
benchmark/longmemeval/data/longmemeval_s_cleaned.json
benchmark/longmemeval/data/longmemeval_m_cleaned.json
```

## The operational corpus

LongMemEval never asks an assistant to follow a procedure, and has no notion of a
mistake corrected once that must not come back — so those two categories need
data of their own. `dma_bench/generation/ontology.py` is the replacement for the
paper's Table 5: the axes of an operational domain (clients, projects, errors and
their corrections, procedures, feedback) that scenarios are sampled from.

Sessions are written as working narrative rather than statements of fact —
"the renewal call confirmed they'd moved up to Enterprise" instead of
"FACT: Acme = Enterprise". The paper does this to stop a retriever scoring by
lexical match; here it matters more, because a session that states the fact
outright leaves the manager agent nothing to extract and never tests
`memory_consolidate` at all.

Generate a corpus once and keep it — regenerating per run makes every number
incomparable with the last:

```bash
uv run --group benchmark python -m dma_bench.generation.generator \
    --config small --out benchmark/data/operational_small.json \
    --model gpt-4o --base-url https://api.openai.com/v1 --api-key "$OPENAI_API_KEY"
```

`--config large` spreads its sessions over about eight months, which is where
shard routing, accumulated supersessions and repeated consolidation passes are
actually exercised — the one thing the published datasets cannot offer.

`--config` fixes every dimension at once, though, and size is usually the one
you want to move on its own — a two-case trial of the `large` timeline has no
configuration of its own. `--cases-per-category` overrides just that number and
leaves the sessions per case and the timeline where the configuration put them,
so a trial run and the full one differ only in how many cases they pay for:

```bash
uv run --group benchmark python -m dma_bench.generation.generator \
    --config large --cases-per-category 2 --out benchmark/data/trial.json
```

`generate_corpus(..., cases_per_category=2)` is the same override from Python,
for the notebook.

Generation shows one progress bar per category, counting **model calls** rather
than cases: `large` is a few dozen cases but over a thousand calls, and a bar
that moves twelve times in an hour says nothing useful. Finished bars stay on
screen, so a run ends with one line per category and what it cost. Pass
`--quiet` for non-interactive runs.

## Running

```bash
uv sync --group benchmark
uv run --group benchmark jupyter lab benchmark/run_benchmark.ipynb
```

Configure the first cell, then run top to bottom. Output:

```
<EXPERIMENT_ROOT>/<question_id>/memory/       the tree that case built
<EXPERIMENT_ROOT>/<question_id>/result.json   that case's ingestion, trace and three verdicts
<EXPERIMENT_ROOT>/result.json                 config + every case + the summary
```

A case that already has a `result.json` is skipped, so an interrupted run resumes
instead of paying twice. Cases run concurrently and share nothing — separate
memory trees, separate agents, a context-local write clock.

## How the clock works, and why

`MemoryStore.write` stamps entries with `datetime.now()` unless a caller passes
`when=`, and the agent tools do not expose that argument. Replayed as-is, a
conversation from 2023 would land in the current month's shard with today's date,
which destroys every temporal question.

Freezing the process clock would fix the stamps and break everything else — the
same `datetime.now` backs HTTP timeouts inside the model client. So
`dma_bench/clock.py` patches `datetime` only in the three memory modules that
read the wall clock, behind a `ContextVar`. A context variable rather than a
thread-local because LangGraph runs tool calls on worker threads: LangChain
copies the caller's context into those executors, and a thread-local set around
`agent.invoke` would simply not be visible where the write happens.

No change to `src/` is involved, and with no override set the patched class is
the real clock.

## Layout

```
dma_bench/
├── schema.py        Case / RunConfig / CaseResult — everything that reaches disk
├── categories.py    the taxonomy and the LongMemEval mapping
├── clock.py         the simulated write clock
├── datasets/        longmemeval adapter + operational corpus loader
├── generation/      the operational ontology and its generator
├── agents.py        building the manager and search agents
├── ingest.py        replaying a history, session by session
├── answer.py        asking the question, capturing the trace
├── judges/          consolidation, retrieval and QA graders
├── runner.py        orchestration, persistence, resume
├── metrics.py       aggregation and the three-stage decomposition
└── report.py        tables and charts
```

Tests live in `tests/benchmark/` and run with the rest of the suite
(`uv run pytest`). They cover the pure logic — the adapter, the clock, the
metrics, the judges' scoring rules and one end-to-end case — with scripted
models, so no test spends anything on an API.
