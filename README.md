# deep-memory-agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/deep-memory-agent)](https://pypi.org/project/deep-memory-agent/)
[![Version](https://img.shields.io/github/v/tag/giurlanda/deep-memory-agent?sort=semver&label=version)](https://github.com/giurlanda/deep-memory-agent/tags)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://giurlanda.github.io/deep-memory-agent/)

Factory for building LangChain deepagents equipped with episodic, semantic, and procedural memory.

Memory is a small wiki of plain markdown files — readable, diffable, versionable
with git — that the agent maintains itself: it records what happened, supersedes
what is no longer true, and consolidates recurring patterns into durable
knowledge.

The tree lives at the virtual path `/memory/`, served by a
[deepagents](https://github.com/langchain-ai/deepagents) backend. No tool in
this package touches the host filesystem directly, so the same agent runs
unchanged against a directory on disk, ephemeral thread state, or a remote store.

## Installation

```bash
pip install deep-memory-agent

# with semantic (embedding) search over memory:
pip install "deep-memory-agent[semantic]"
```

## Quickstart

Two factories, one tree. The manager writes; the search agent cannot.

```python
from deep_memory_agent import create_memory_manager_agent, create_memory_search_agent

manager = create_memory_manager_agent("claude-sonnet-5", memory_dir="./memory")
manager.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Remember that I manage Python projects with uv.",
            }
        ]
    }
)

recall = create_memory_search_agent("claude-sonnet-5", memory_dir="./memory")
recall.invoke(
    {"messages": [{"role": "user", "content": "Which package manager do I use?"}]}
)
```

Each factory takes **either** `memory_dir` — the default on-disk wiring — or a
ready-made `backend`. Exactly one of the two is required; passing both raises.

## The memory tree

```
/memory/
├── index.md                      router for the whole tree
├── preferences.md                how the user wants the agent to behave
├── episodic_memory/              what happened
│   ├── index.md
│   ├── events/YYYY-MM.md         sharded by month
│   ├── feedbacks/YYYY-MM.md
│   └── errors/YYYY-MM.md
├── semantic_memory/              what is true
│   ├── index.md
│   ├── facts.md
│   └── rules.md
└── procedural_memory/            how things are done
    ├── index.md
    └── <slug>.md                 one file per procedure
```

Every entry carries YAML frontmatter, so provenance travels with the content and
a newer statement can explicitly retire an older one:

```markdown
---
id: mem_2026-08-24_9f3a1c
created: 2026-08-24T10:15:00+00:00
type: semantic
category: facts
source: user_message
confidence: high
tags: [pricing, acme]
supersedes: mem_2026-06-01_4b2e77
summary: ACME moved to the Enterprise plan
---

ACME switched from the Team plan to Enterprise on 2026-08-24.
```

Three rules keep the tree from degenerating into an append-only log:

- **Indexes are routers, never content** — one line per file, so the agent can
  decide what to load without loading everything.
- **Semantic memory is superseded, not appended** — a changed fact produces a new
  entry that retires the old one, which stays on disk as history.
- **Episodic memory is sharded by month** — no single file outgrows the context
  window.

## Consolidation

Episodic memory on its own is a log. `consolidate_memory` reads recent episodes
and promotes what has hardened into facts, rules or procedures. It is a plain
function, so it can be scheduled from ordinary code; the manager agent also
exposes it as the `memory_consolidate` tool.

```python
from deep_memory_agent import build_memory_backend, consolidate_memory

backend = build_memory_backend("./memory", for_deep_agent=False)
result = consolidate_memory(backend, "claude-sonnet-5")
```

`for_deep_agent=False` is what makes the backend usable outside an agent: the
default wiring parks non-memory paths in LangGraph thread state, which is only
reachable from inside a graph execution.

Episodes are never deleted: consolidation only adds durable knowledge and
supersedes what it contradicts.

See [examples/](examples/) for runnable scripts, or the
[docs](https://giurlanda.github.io/deep-memory-agent/).

## Semantic search

`memory_search` is a substring test, so it misses any question phrased
differently from the entry that answers it — a rule stored as *"always apply the
Enterprise discount to customers with more than 3 years of contract"* is
invisible to *"what discounts exist for long-standing customers?"*. Hand either
factory an embedding model and a vector store and the agents gain an index that
matches meaning instead of wording:

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from deep_memory_agent import create_memory_manager_agent

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

manager = create_memory_manager_agent(
    "claude-sonnet-5",
    memory_dir="./memory",
    embeddings=embeddings,
    vector_store=InMemoryVectorStore(embeddings),
)
```

The manager gets `semantic_ingest` and `semantic_search`; the recall agent gets
only the search, since withholding the ingest tool is the only way to keep it
read-only over a store the filesystem permissions cannot guard. Refresh the index
from ordinary code with `ingest_semantic_index`:

```python
from deep_memory_agent import ingest_semantic_index

report = ingest_semantic_index(embeddings, vector_store, memory_dir="./memory")
print(report.summary())
```

The index is **derived data**: the markdown files stay the source of truth, and
dropping the index costs the ability to search by meaning until the next ingest,
never a fact. No vector store is pinned — bring your own; the only requirement is
that it upserts on a repeated id. See
[Semantic search](https://giurlanda.github.io/deep-memory-agent/semantic-search/)
for chunking, filters, the manifest and the cost of the explicit-ingest choice.

## Benchmark

[`benchmark/`](benchmark/) holds a LongMemEval-style benchmark for the two
agents. Unlike LongMemEval, which scores a retriever over a frozen chat log, it
replays episodes **through the real manager agent** one session at a time, so
extraction, supersession and consolidation are all part of what is measured —
and failures are decomposed across those three stages rather than two.

```bash
uv sync --group benchmark
uv run --group benchmark jupyter lab benchmark/run_benchmark.ipynb
```

## Scope of this version

Memory is file-based, and the files are always the source of truth. Retrieval
comes in two flavours: lexical by default (case-insensitive substring over
summary, body and tags), and semantic when an embedding model and a vector store
are supplied — the latter an index derived from the files, never a second copy of
them. BM25 is still addable the same way, without changing the format.

Out of scope: automatic decay, and locking between concurrent writers — keeping a
single writer over the tree is the manager agent's job. The semantic index is
also refreshed explicitly rather than on every write, so it can trail the files
between ingests.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run ruff format .
```

## License

MIT — see [LICENSE](LICENSE).
