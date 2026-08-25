"""Factory for LangChain deepagents with episodic, semantic and procedural memory.

The package builds two agents over a wiki-like memory tree of plain markdown
files: [`create_memory_search_agent`][deep_memory_agent.create_memory_search_agent]
answers from memory without being able to change it, and
[`create_memory_manager_agent`][deep_memory_agent.create_memory_manager_agent]
is the single writer that records, supersedes and consolidates.

```python
from deep_memory_agent import create_memory_manager_agent

agent = create_memory_manager_agent("claude-sonnet-5", memory_dir="./memory")
agent.invoke({"messages": [{"role": "user", "content": "Remember I prefer uv."}]})
```

Memory lives under the virtual path `/memory/`, served by a deepagents backend.
No tool in this package touches the host filesystem directly, so the same agent
runs unchanged against a directory on disk, ephemeral thread state, or a remote
store.
"""

from deep_memory_agent.agents import (
    READ_ONLY_MEMORY_PERMISSIONS,
    create_memory_manager_agent,
    create_memory_search_agent,
)
from deep_memory_agent.backends import build_memory_backend, resolve_backend
from deep_memory_agent.consolidation import (
    ConsolidatedItem,
    ConsolidationProposal,
    ConsolidationResult,
    consolidate_memory,
)
from deep_memory_agent.entry import Confidence, MemoryEntry
from deep_memory_agent.layout import MEMORY_ROOT, MemoryCategory, MemoryKind
from deep_memory_agent.scaffold import PROCEDURE_TEMPLATE, ensure_memory_tree
from deep_memory_agent.store import MemoryHit, MemoryStore
from deep_memory_agent.tools import build_recall_tools, build_write_tools

__version__ = "0.1.2"

__all__ = [
    "MEMORY_ROOT",
    "PROCEDURE_TEMPLATE",
    "READ_ONLY_MEMORY_PERMISSIONS",
    "Confidence",
    "ConsolidatedItem",
    "ConsolidationProposal",
    "ConsolidationResult",
    "MemoryCategory",
    "MemoryEntry",
    "MemoryHit",
    "MemoryKind",
    "MemoryStore",
    "__version__",
    "build_memory_backend",
    "build_recall_tools",
    "build_write_tools",
    "consolidate_memory",
    "create_memory_manager_agent",
    "create_memory_search_agent",
    "ensure_memory_tree",
    "resolve_backend",
]
