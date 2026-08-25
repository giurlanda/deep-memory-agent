# deep-memory-agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/giurlanda/deep-memory-agent?sort=semver&label=version)](https://github.com/giurlanda/deep-memory-agent/tags)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

Factory for building LangChain deepagents equipped with episodic, semantic, and procedural memory.

Memory is a small wiki of plain markdown files — readable, diffable, versionable
with git — that the agent maintains itself: it records what happened, supersedes
what is no longer true, and consolidates recurring patterns into durable
knowledge.

The tree lives at the virtual path `/memory/`, served by a
[deepagents](https://github.com/langchain-ai/deepagents) backend. No tool in
this package touches the host filesystem directly, so the same agent runs
unchanged against a directory on disk, ephemeral thread state, or a remote store.

📖 **Documentation:** <https://giurlanda.github.io/deep-memory-agent/>

## Installation

```bash
pip install deep-memory-agent
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

## Scope of this version

Memory is file-based and retrieval is lexical (case-insensitive substring over
summary, body and tags). The structured frontmatter is what makes a stronger
index — BM25, embeddings — addable later without changing the files themselves.
Automatic decay, locking between concurrent writers and vector retrieval are out
of scope; keeping a single writer over the tree is the manager agent's job.

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
