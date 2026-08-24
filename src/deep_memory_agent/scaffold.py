"""Creation of the memory tree.

Scaffolding runs through the backend like everything else, so pointing an agent
at a `StateBackend`, a `StoreBackend` or a directory on disk all behave the
same. It is idempotent: existing files are never overwritten, which is what
makes it safe to call on every agent construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deep_memory_agent.layout import (
    EPISODIC_DIR,
    PREFERENCES_PATH,
    PROCEDURAL_DIR,
    ROOT_INDEX_PATH,
    SEMANTIC_DIR,
    MemoryKind,
    index_path,
)

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

__all__ = ["PROCEDURE_TEMPLATE", "ensure_memory_tree"]

PROCEDURE_TEMPLATE = """# {title}

## When to use

{when_to_use}

## Preconditions

{preconditions}

## Steps

{steps}

## Tools required

{tools}

## Known failures

{known_failures}
"""
"""Fixed section layout every procedure file follows.

`Known failures` is where episodic memory feeds back into procedural memory: a
mistake that keeps showing up in `errors/` belongs here, so the agent reads the
fix at the moment it is about to repeat the mistake.
"""

_ROOT_INDEX = """# Memory index

Router for the whole memory tree. Load a file before relying on its content;
this index only says what exists and roughly what is in it.

| File | Description | Tags | Updated |
| --- | --- | --- | --- |
| episodic_memory/index.md | What happened, sharded by month | episodic | |
| semantic_memory/index.md | What is true: facts and rules | semantic | |
| procedural_memory/index.md | How things are done, one file each | procedural | |
| preferences.md | How the user wants the agent to behave | preferences | |
"""

_EPISODIC_INDEX = """# Episodic memory index

Sessions, feedback and mistakes, each entry tied to the moment it happened.
Files are sharded by month (`events/2026-08.md`) so no single file outgrows the
context window. Episodes are raw history: stable conclusions drawn from them
belong in semantic or procedural memory, put there by consolidation.

| File | Description | Tags | Updated |
| --- | --- | --- | --- |
"""

_SEMANTIC_INDEX = """# Semantic memory index

Statements believed to be true right now, detached from any single episode. A
fact that changes is *superseded*, not appended to: the new entry carries
`supersedes`, the old one gets `superseded_by`, and only the new one is active.

| File | Description | Tags | Updated |
| --- | --- | --- | --- |
| facts.md | Facts about the user, their work and their world | facts | |
| rules.md | Constraints and policies that govern behaviour | rules | |
"""

_PROCEDURAL_INDEX = """# Procedural memory index

One file per operating procedure, each with the same sections: when to use,
preconditions, steps, tools required, known failures.

| File | Description | Tags | Updated |
| --- | --- | --- | --- |
"""

_PREFERENCES = """# Preferences

How the user wants the agent to behave. Treated as semantic memory: a changed
preference supersedes the previous one rather than sitting next to it.
"""

_FACTS = """# Facts

Statements believed to be true right now.
"""

_RULES = """# Rules

Constraints and policies that govern how the agent acts.
"""

_SEED_FILES: dict[str, str] = {
    ROOT_INDEX_PATH: _ROOT_INDEX,
    PREFERENCES_PATH: _PREFERENCES,
    index_path(MemoryKind.EPISODIC): _EPISODIC_INDEX,
    index_path(MemoryKind.SEMANTIC): _SEMANTIC_INDEX,
    index_path(MemoryKind.PROCEDURAL): _PROCEDURAL_INDEX,
    f"{SEMANTIC_DIR}facts.md": _FACTS,
    f"{SEMANTIC_DIR}rules.md": _RULES,
}

MEMORY_DIRECTORIES = (EPISODIC_DIR, SEMANTIC_DIR, PROCEDURAL_DIR)
"""Directories the tree is made of, for documentation and tests."""


def ensure_memory_tree(backend: BackendProtocol) -> list[str]:
    """Create any missing file of the memory tree.

    Episodic shards are deliberately not pre-created: they are named after the
    month they cover and appear the first time something is written to them.

    Args:
        backend: Backend serving the memory tree.

    Returns:
        The paths that were created, in a stable order. Empty when the tree was
        already complete.
    """
    created: list[str] = []
    for path, content in _SEED_FILES.items():
        if _exists(backend, path):
            continue
        backend.write(path, content)
        created.append(path)
    return created


def _exists(backend: BackendProtocol, path: str) -> bool:
    """Return whether a file can be read through the backend."""
    result = backend.read(path, limit=1)
    return not result.error
