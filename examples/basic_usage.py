"""Record something in memory, then recall it with the read-only agent.

Run with: uv run python examples/basic_usage.py

Needs credentials for the model you pass; set the relevant provider key first,
e.g. `export ANTHROPIC_API_KEY=...`.
"""

from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from deep_memory_agent import create_memory_manager_agent, create_memory_search_agent

MODEL = "claude-sonnet-5"
MEMORY_DIR = Path("./memory")


def say(agent: CompiledStateGraph, message: str) -> str:
    """Send one message to an agent and return its final reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].content


manager = create_memory_manager_agent(MODEL, memory_dir=MEMORY_DIR)
print(say(manager, "Remember that I manage Python projects with uv, never with pip."))

# A separate agent, on the same tree, that cannot change what it reads.
recall = create_memory_search_agent(MODEL, memory_dir=MEMORY_DIR)
print(say(recall, "Which package manager do I use?"))

print(f"\nMemory written to {MEMORY_DIR.resolve()}:")
for path in sorted(MEMORY_DIR.rglob("*.md")):
    print(f"  {path.relative_to(MEMORY_DIR)}")
