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
from deep_memory_agent.prompts import MANAGER_AGENT_PROMPT, SEARCH_AGENT_PROMPT
from deep_memory_agent.store import MemoryStore
from deep_memory_agent.tools import build_recall_tools, build_write_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

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


def create_memory_search_agent(
    model: str | BaseChatModel,
    *,
    memory_dir: str | Path | None = None,
    backend: BackendProtocol | None = None,
    system_prompt: str | None = None,
    tools: Sequence[BaseTool] = (),
    name: str = "memory_search_agent",
    **kwargs: Any,
) -> CompiledStateGraph:
    """Build a read-only agent that answers from memory.

    The agent gets `memory_index`, `memory_search` and `memory_read`, and is
    denied every write on `/memory/` at the backend level. It is the safe agent
    to expose to callers who should be able to consult memory but never change
    it.

    Args:
        model: Chat model, or an identifier resolvable by deepagents.
        memory_dir: Host directory holding the memory tree. Mutually exclusive
            with `backend`; exactly one of the two is required.
        backend: A ready-made backend serving `/memory/`. Mutually exclusive
            with `memory_dir`.
        system_prompt: Overrides the built-in recall prompt. Pass one only if
            you also restate the layout rules the built-in prompt carries.
        tools: Extra tools to expose alongside the recall tools.
        name: Name of the compiled graph.
        **kwargs: Forwarded to `deepagents.create_deep_agent`, e.g.
            `checkpointer` or `interrupt_on`.

    Returns:
        The compiled agent.

    Raises:
        ValueError: If both `memory_dir` and `backend` are given, or neither.
    """
    resolved = resolve_backend(memory_dir, backend)
    store = MemoryStore(resolved)
    store.ensure_tree()

    return create_deep_agent(
        model,
        [*build_recall_tools(store), *tools],
        system_prompt=system_prompt or SEARCH_AGENT_PROMPT,
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
            you also restate the layout rules the built-in prompt carries.
        tools: Extra tools to expose alongside the memory tools.
        consolidation_model: Model used by `memory_consolidate`. Defaults to
            `model`.
        name: Name of the compiled graph.
        **kwargs: Forwarded to `deepagents.create_deep_agent`, e.g.
            `checkpointer` or `interrupt_on`.

    Returns:
        The compiled agent.

    Raises:
        ValueError: If both `memory_dir` and `backend` are given, or neither.
    """
    resolved = resolve_backend(memory_dir, backend)
    store = MemoryStore(resolved)
    store.ensure_tree()

    memory_tools = [
        *build_recall_tools(store),
        *build_write_tools(store, consolidation_model=consolidation_model or model),
    ]
    return create_deep_agent(
        model,
        [*memory_tools, *tools],
        system_prompt=system_prompt or MANAGER_AGENT_PROMPT,
        backend=resolved,
        name=name,
        **kwargs,
    )
