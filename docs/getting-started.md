# Getting started

## Installation

```bash
pip install deep-memory-agent
```

Requires Python 3.12 or newer, and credentials for whichever model provider you
use.

## Recording and recalling

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
result = recall.invoke(
    {"messages": [{"role": "user", "content": "Which package manager do I use?"}]}
)
print(result["messages"][-1].content)
```

After the first call, `./memory` holds a scaffolded tree of markdown files you
can open, diff and commit like any other source.

## Choosing where memory lives

Each factory takes **either** `memory_dir` or `backend`, never both, and one of
them is required — a default would mean writing files somewhere the caller never
named.

```python
# On disk: /memory/ maps to ./memory, everything else stays in thread state.
create_memory_manager_agent(model, memory_dir="./memory")

# Or bring your own backend, e.g. to put the tree in a LangGraph store.
create_memory_manager_agent(model, backend=my_backend)
```

[`build_memory_backend`][deep_memory_agent.build_memory_backend] is what the
`memory_dir` form builds: a `CompositeBackend` routing `/memory/` to a
`FilesystemBackend` and leaving everything else on an ephemeral `StateBackend`.

That ephemeral default only works inside a deep agent: `StateBackend` reads and
writes through LangGraph, and raises outside a graph execution. Building the
backend yourself for use outside one — consolidation from a cron job, a store
you drive directly — needs `for_deep_agent=False`, which serves non-memory paths
from an empty scratch directory instead:

```python
backend = build_memory_backend("./memory", for_deep_agent=False)
```

## Why the search agent cannot write

Withholding the write tools is not enough on its own — the built-in `write_file`,
`edit_file` and `delete` tools would still reach `/memory/`. The search agent is
therefore also given
[`READ_ONLY_MEMORY_PERMISSIONS`][deep_memory_agent.READ_ONLY_MEMORY_PERMISSIONS],
a deny rule enforced by the filesystem middleware, so recall stays read-only even
if you add tools of your own.

## Consolidating from code

Consolidation is a plain function as well as a tool, so a nightly job can run it
without going through a conversation:

```python
from datetime import UTC, datetime, timedelta

from deep_memory_agent import build_memory_backend, consolidate_memory

result = consolidate_memory(
    build_memory_backend("./memory", for_deep_agent=False),
    "claude-sonnet-5",
    since=datetime.now(tz=UTC) - timedelta(days=7),
)
print(result.rationale)
```

Writing nothing is a normal outcome: it means no episode had hardened into
durable knowledge yet.

## Using the store directly

[`MemoryStore`][deep_memory_agent.MemoryStore] is the layer the tools sit on. It
is useful for seeding memory, or for inspecting it in tests, without a model:

```python
from deep_memory_agent import MemoryCategory, MemoryStore, build_memory_backend

store = MemoryStore(build_memory_backend("./memory", for_deep_agent=False))
store.ensure_tree()
store.write(
    MemoryCategory.FACTS,
    "ACME is on the Enterprise plan.",
    summary="ACME plan",
    tags=("acme", "pricing"),
)
```
