"""Construction of the two agents under test.

The answering side is the shipped factory, unchanged — that is the point of the
benchmark. It is built twice when the run compares arms, once with the semantic
pair and once without, so what separates the two numbers is entirely the
factory's doing. The writing side is the shipped factory too, *except* in the
`none` arm of the consolidation ablation, where the agent must not be able to
consolidate at all.

That exception needs its own assembly because `create_memory_manager_agent`
resolves `consolidation_model or model`, so there is no argument that takes
`memory_consolidate` away. Telling the agent in the prompt not to consolidate
would leave the arm at the mercy of whether it complies; the ablation only means
something if the tool is absent. Everything used here is public API, and the
prompt is the shipped one, so the agent is otherwise identical to the factory's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents import create_deep_agent

from deep_memory_agent import (
    MemoryStore,
    build_memory_backend,
    build_recall_tools,
    build_write_tools,
    create_memory_manager_agent,
    create_memory_search_agent,
)
from deep_memory_agent.prompts import MANAGER_AGENT_PROMPT

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore
    from langgraph.graph.state import CompiledStateGraph

__all__ = ["build_manager_agent", "build_search_agent", "open_store"]


def build_manager_agent(
    model: BaseChatModel,
    memory_dir: Path,
    *,
    allow_consolidation: bool,
    middleware: Sequence[AgentMiddleware] = (),
) -> CompiledStateGraph:
    """Build the agent that replays sessions into memory.

    Args:
        model: The model under test.
        memory_dir: Directory holding this case's memory tree.
        allow_consolidation: Whether `memory_consolidate` is available. `False`
            builds the agent without it, so the cold arm cannot consolidate.
        middleware: Extra middleware, e.g. the invocation budget. It bounds the
            harness, not the agent's behaviour, so both arms get the same.

    Returns:
        The compiled manager agent.
    """
    if allow_consolidation:
        return create_memory_manager_agent(
            model,
            memory_dir=memory_dir,
            consolidation_model=model,
            middleware=list(middleware),
        )

    backend = build_memory_backend(memory_dir)
    store = MemoryStore(backend)
    store.ensure_tree()
    return create_deep_agent(
        model,
        [
            *build_recall_tools(store),
            *build_write_tools(store, consolidation_model=None),
        ],
        system_prompt=MANAGER_AGENT_PROMPT,
        backend=backend,
        middleware=list(middleware),
        name="memory_manager_agent",
    )


def build_search_agent(
    model: BaseChatModel,
    memory_dir: Path,
    *,
    embeddings: Embeddings | None = None,
    vector_store: VectorStore | None = None,
    search_k: int = 5,
    middleware: Sequence[AgentMiddleware] = (),
) -> CompiledStateGraph:
    """Build the read-only agent that answers the question.

    Args:
        model: The model under test.
        memory_dir: Directory holding this case's memory tree.
        embeddings: Embedding model for the semantic arm. Given together with
            `vector_store`, the agent also gets `semantic_search` and the prompt
            section that comes with it. Both `None` builds the lexical arm — the
            agent exactly as the package ships it by default.
        vector_store: The case's own index. Must be the one `index_case` wrote.
        search_k: Entries a semantic search returns by default.
        middleware: Extra middleware, e.g. the invocation budget. Both arms
            get the same, so it bounds the run without separating them.

    Returns:
        The compiled search agent. Nothing else about it is shaped here: the
        arms differ in the tools the shipped factory hands out, not in a prompt
        or a tool the benchmark wrote for itself.
    """
    return create_memory_search_agent(
        model,
        memory_dir=memory_dir,
        embeddings=embeddings,
        vector_store=vector_store,
        search_k=search_k,
        middleware=list(middleware),
    )


def open_store(memory_dir: Path) -> MemoryStore:
    """Open a case's memory tree from plain code.

    Args:
        memory_dir: Directory holding the tree.

    Returns:
        A store bound to a backend that works outside a graph execution, which
        is what inspection and scheduled consolidation both need.
    """
    return MemoryStore(build_memory_backend(memory_dir, for_deep_agent=False))
