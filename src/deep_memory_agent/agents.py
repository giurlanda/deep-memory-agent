"""Factories for the two memory agents.

The package deliberately ships two agents instead of one. Recall and curation
pull in opposite directions: recall wants a narrow, read-only surface it cannot
corrupt, while curation needs write access and a prompt about supersession and
consolidation. Splitting them also keeps a single writer over the tree, which is
what makes a file-based memory safe to share between agents.

Both factories take **either** `memory_dir` — the default on-disk wiring — or a
ready-made `backend`, never both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents import FilesystemPermission, create_deep_agent

from deep_memory_agent.backends import resolve_backend
from deep_memory_agent.layout import MEMORY_ROOT
from deep_memory_agent.prompts import (
    MANAGER_AGENT_PROMPT,
    SEARCH_AGENT_PROMPT,
    SEMANTIC_MANAGER_BLOCK,
    SEMANTIC_READER_BLOCK,
)
from deep_memory_agent.semantic_tools import SemanticTools, create_semantic_tools
from deep_memory_agent.store import MemoryStore
from deep_memory_agent.tools import build_recall_tools, build_write_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langchain_core.vectorstores import VectorStore
    from langgraph.graph.state import CompiledStateGraph

    from deep_memory_agent.semantic_index import SemanticConfig

__all__ = [
    "READ_ONLY_MEMORY_PERMISSIONS",
    "create_memory_manager_agent",
    "create_memory_search_agent",
]

READ_ONLY_MEMORY_PERMISSIONS = [
    FilesystemPermission(
        operations=["write"],
        paths=[f"{MEMORY_ROOT}**", MEMORY_ROOT.rstrip("/")],
        mode="deny",
    )
]
"""Backend-level guard that makes the memory tree read-only.

Withholding the write tools is not enough on its own: the built-in `write_file`,
`edit_file` and `delete` tools would still reach `/memory/`. This rule denies
them at the filesystem middleware, so recall stays read-only even if a caller
adds tools of their own.
"""


def _build_semantic_tools(
    embeddings: Embeddings | None,
    vector_store: VectorStore | None,
    store: MemoryStore,
    *,
    search_k: int,
    semantic_config: SemanticConfig | None,
) -> SemanticTools | None:
    """Build the semantic tools, or nothing when semantic search is off.

    Args:
        embeddings: Caller-supplied embedding model, or `None`.
        vector_store: Caller-supplied vector store, or `None`.
        store: Store over the memory tree the index will cover.
        search_k: Default number of entries a search returns.
        semantic_config: Index configuration, or `None` for the defaults.

    Returns:
        The tools, or `None` when neither `embeddings` nor `vector_store` was
        given.

    Raises:
        ValueError: If exactly one of `embeddings` and `vector_store` is given.
            Half a configuration is a mistake, not a request to run without an
            index, and silently ignoring it would leave the caller believing
            memory is searchable by meaning when it is not.
    """
    if embeddings is None and vector_store is None:
        return None
    if embeddings is None or vector_store is None:
        msg = (
            "semantic search needs both `embeddings` and `vector_store`; got "
            f"embeddings={embeddings is not None}, "
            f"vector_store={vector_store is not None}"
        )
        raise ValueError(msg)
    return create_semantic_tools(
        embeddings,
        vector_store,
        store,
        search_k=search_k,
        config=semantic_config,
    )


def create_memory_search_agent(
    model: str | BaseChatModel,
    *,
    memory_dir: str | Path | None = None,
    backend: BackendProtocol | None = None,
    system_prompt: str | None = None,
    tools: Sequence[BaseTool] = (),
    embeddings: Embeddings | None = None,
    vector_store: VectorStore | None = None,
    search_k: int = 5,
    semantic_config: SemanticConfig | None = None,
    name: str = "memory_search_agent",
    **kwargs: Any,
) -> CompiledStateGraph:
    """Build a read-only agent that answers from memory.

    The agent gets `memory_index`, `memory_search`, `memory_get` and
    `memory_read`, and is denied every write on `/memory/` at the backend level.
    It is the safe agent to expose to callers who should be able to consult
    memory but never change it.

    Args:
        model: Chat model, or an identifier resolvable by deepagents.
        memory_dir: Host directory holding the memory tree. Mutually exclusive
            with `backend`; exactly one of the two is required.
        backend: A ready-made backend serving `/memory/`. Mutually exclusive
            with `memory_dir`.
        system_prompt: Overrides the built-in recall prompt. Pass one only if
            you also restate the layout rules the built-in prompt carries — a
            prompt of your own also replaces the semantic section, which is not
            appended to it.
        tools: Extra tools to expose alongside the recall tools.
        embeddings: Embedding model enabling semantic search. Given together
            with `vector_store`, the agent gains `semantic_search` — and only
            that: the ingest tool is deliberately withheld, since it is the only
            way to keep this agent read-only over an index the filesystem
            permissions cannot guard. Refresh the index out of band with
            [`ingest_semantic_index`][deep_memory_agent.semantic_tools.ingest_semantic_index],
            or from the manager agent. Requires the optional `semantic` extra.
        vector_store: Store holding the index to search. Must be the one the
            ingest wrote to.
        search_k: Default number of entries a semantic search returns.
        semantic_config: Index configuration. Only its search-side fields matter
            here, since this agent never ingests.
        name: Name of the compiled graph.
        **kwargs: Forwarded to `deepagents.create_deep_agent`, e.g.
            `checkpointer` or `interrupt_on`.

    Returns:
        The compiled agent.

    Raises:
        ValueError: If both `memory_dir` and `backend` are given, or neither, or
            if exactly one of `embeddings` and `vector_store` is given.
    """
    resolved = resolve_backend(memory_dir, backend)
    store = MemoryStore(resolved)
    store.ensure_tree()

    semantic = _build_semantic_tools(
        embeddings,
        vector_store,
        store,
        search_k=search_k,
        semantic_config=semantic_config,
    )
    agent_tools: list[BaseTool] = [*build_recall_tools(store)]
    prompt = system_prompt or SEARCH_AGENT_PROMPT
    if semantic is not None:
        agent_tools.append(semantic.search_tool)
        if system_prompt is None:
            prompt += SEMANTIC_READER_BLOCK

    return create_deep_agent(
        model,
        [*agent_tools, *tools],
        system_prompt=prompt,
        backend=resolved,
        permissions=[*READ_ONLY_MEMORY_PERMISSIONS, *kwargs.pop("permissions", [])],
        name=name,
        **kwargs,
    )


def create_memory_manager_agent(
    model: str | BaseChatModel,
    *,
    memory_dir: str | Path | None = None,
    backend: BackendProtocol | None = None,
    system_prompt: str | None = None,
    tools: Sequence[BaseTool] = (),
    consolidation_model: str | BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    vector_store: VectorStore | None = None,
    search_k: int = 5,
    semantic_config: SemanticConfig | None = None,
    name: str = "memory_manager_agent",
    **kwargs: Any,
) -> CompiledStateGraph:
    """Build the agent that curates memory.

    It gets the recall tools plus `memory_write`, `memory_update` and
    `memory_consolidate`, and is meant to be the **single writer** of the tree:
    concurrent writers on plain markdown files lose data, and nothing here
    locks.

    Args:
        model: Chat model, or an identifier resolvable by deepagents.
        memory_dir: Host directory holding the memory tree. Mutually exclusive
            with `backend`; exactly one of the two is required.
        backend: A ready-made backend serving `/memory/`. Mutually exclusive
            with `memory_dir`.
        system_prompt: Overrides the built-in curation prompt. Pass one only if
            you also restate the layout rules the built-in prompt carries — a
            prompt of your own also replaces the semantic section, which is not
            appended to it.
        tools: Extra tools to expose alongside the memory tools.
        consolidation_model: Model used by `memory_consolidate`. Defaults to
            `model`.
        embeddings: Embedding model enabling semantic search. Given together
            with `vector_store`, the agent gains `semantic_ingest` and
            `semantic_search`, plus a prompt section on keeping the index
            current. It gets both because it is the single writer of the tree,
            so it is the only agent that can keep a derived index in step with
            it. Requires the optional `semantic` extra.
        vector_store: Store the chunks are written to and searched in. It must
            upsert on a repeated id, so a re-ingest updates a chunk rather than
            duplicating it.
        search_k: Default number of entries a semantic search returns.
        semantic_config: Index configuration — chunking, the manifest location,
            batch sizes, an optional server-side filter builder. Defaults to
            [`SemanticConfig`][deep_memory_agent.semantic_index.SemanticConfig].
        name: Name of the compiled graph.
        **kwargs: Forwarded to `deepagents.create_deep_agent`, e.g.
            `checkpointer` or `interrupt_on`.

    Returns:
        The compiled agent.

    Raises:
        ValueError: If both `memory_dir` and `backend` are given, or neither, or
            if exactly one of `embeddings` and `vector_store` is given.
    """
    resolved = resolve_backend(memory_dir, backend)
    store = MemoryStore(resolved)
    store.ensure_tree()

    memory_tools = [
        *build_recall_tools(store),
        *build_write_tools(store, consolidation_model=consolidation_model or model),
    ]
    semantic = _build_semantic_tools(
        embeddings,
        vector_store,
        store,
        search_k=search_k,
        semantic_config=semantic_config,
    )
    prompt = system_prompt or MANAGER_AGENT_PROMPT
    if semantic is not None:
        memory_tools.extend(semantic.as_list())
        if system_prompt is None:
            prompt += SEMANTIC_MANAGER_BLOCK

    return create_deep_agent(
        model,
        [*memory_tools, *tools],
        system_prompt=prompt,
        backend=resolved,
        name=name,
        **kwargs,
    )
