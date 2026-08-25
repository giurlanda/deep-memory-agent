"""Consolidate episodic memory from plain code, without a conversation.

Consolidation reads recent episodes and promotes what has hardened into facts,
rules or procedures. It is a plain function so it can be scheduled — a nightly
job, a cron entry — instead of having to go through the manager agent.

Running outside an agent is what `for_deep_agent=False` is for: the default
wiring keeps non-memory paths in LangGraph thread state, which only exists
inside a graph execution.

Run with: uv run python examples/consolidate_memory.py
"""

from datetime import UTC, datetime, timedelta

from deep_memory_agent import build_memory_backend, consolidate_memory

backend = build_memory_backend("./memory", for_deep_agent=False)
result = consolidate_memory(
    backend,
    "claude-sonnet-5",
    since=datetime.now(tz=UTC) - timedelta(days=7),
)

print(f"Read {result.episodes_considered} episodes: {result.rationale}")
for hit in result.entries:
    print(f"  {hit.entry.entry_id} -> {hit.path} ({hit.entry.summary})")
