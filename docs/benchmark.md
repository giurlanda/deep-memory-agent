# Benchmark

The repository ships a benchmark for the two memory agents under
[`benchmark/`](https://github.com/giurlanda/deep-memory-agent/tree/main/benchmark).
It is not part of the published package: it is a research harness kept next to
the code it measures.

## What it does differently

The methodology comes from
[LongMemEval](https://arxiv.org/abs/2410.10813) (ICLR 2025) — a taxonomy of
question types, one LLM judge prompt per type, retrieval scored separately from
generation, and failures decomposed by stage.

What changes is the system under test. LongMemEval evaluates a retriever over a
frozen chat log. Here the episodes are replayed **through the real manager
agent**, one session at a time, so `memory_write`, supersession and
`memory_consolidate` are all part of what is being measured — not a retriever
over facts that were loaded in advance.

## Three stages, not two

That extra stage is why the error decomposition has eight cells rather than the
paper's four. A wrong answer means something different depending on where it
broke, and the three places live in three different prompts:

| consolidation | retrieval | answer | points at |
| --- | --- | --- | --- |
| ✗ | any | ✗ | the consolidation prompt — the fact never reached memory |
| ✓ | ✗ | ✗ | indexing and search — it is on file and was not found |
| ✓ | ✓ | ✗ | the reading strategy — found and then misused |
| ✓ | ✓ | ✓ | working as intended |

Cells where the answer is right despite an earlier stage failing are the ones to
be suspicious of: they usually mean the model knew the answer without needing
memory at all.

## Categories

Six transfer directly from LongMemEval and run on its published datasets:
`single-hop`, `preference`, `multi-hop`, `knowledge-update`,
`temporal-reasoning` and `abstention`.

Three have no counterpart there, because LongMemEval has no agent that writes its
own memory:

- **`supersede-integrity`** re-judges the knowledge-update cases under a stricter
  prompt. LongMemEval accepts an answer that carries the old value next to the
  new one; that tolerance hides exactly the failure a curated memory has, so here
  a retired value presented as current is wrong. Labelling it as history is fine.
- **`procedural-retrieval`** and **`non-repetition`** come from a generated
  corpus over an operational-domain ontology, which replaces the paper's
  personal-attribute ontology.
- **`consolidation-quality`** is not a category: it is the difference between two
  runs over the same cases, one with consolidation and one without.

## Running it

```bash
uv sync --group benchmark
uv run --group benchmark jupyter lab benchmark/run_benchmark.ipynb
```

The LongMemEval datasets are not in the repository — download them from the
[LongMemEval release](https://github.com/xiaowu0162/LongMemEval) into
`benchmark/longmemeval/data/`.

Each case gets its own memory tree and its own result file, and a run resumes
where it stopped:

```
<experiment_root>/<question_id>/memory/       the tree that case built
<experiment_root>/<question_id>/result.json   its ingestion, trace and three verdicts
<experiment_root>/result.json                 config + every case + the summary
```

The full write-up, including how historical timestamps are simulated without
freezing the process clock, is in
[`benchmark/README.md`](https://github.com/giurlanda/deep-memory-agent/blob/main/benchmark/README.md).
