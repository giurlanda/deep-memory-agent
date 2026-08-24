"""Layout of the virtual memory filesystem.

Every path in this module is a *virtual* path inside the deepagents backend,
rooted at [`MEMORY_ROOT`][deep_memory_agent.layout.MEMORY_ROOT]. Nothing here
touches the host filesystem: the mapping from `/memory/` to real storage is the
backend's job (see [`deep_memory_agent.backends`][]).

The tree deliberately departs from a flat `events.md` / `facts.md` layout in two
ways:

- Episodic categories are **sharded by month** so a single file never grows past
  what fits in a context window.
- Every directory owns an `index.md` that is a *router* — one line per file — and
  never holds memory content itself.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

__all__ = [
    "CATEGORY_KIND",
    "EPISODIC_DIR",
    "KIND_DIRECTORIES",
    "MEMORY_ROOT",
    "PREFERENCES_PATH",
    "PROCEDURAL_DIR",
    "ROOT_INDEX_PATH",
    "SEMANTIC_DIR",
    "MemoryCategory",
    "MemoryKind",
    "category_directory",
    "entry_path",
    "index_path",
    "shard_label",
    "slugify",
]

MEMORY_ROOT = "/memory/"
"""Virtual mount point of the memory tree."""

ROOT_INDEX_PATH = "/memory/index.md"
"""Top-level router pointing at the three memory kinds."""

PREFERENCES_PATH = "/memory/preferences.md"
"""Stable user preferences, kept at the root because they cut across kinds."""

EPISODIC_DIR = "/memory/episodic_memory/"
SEMANTIC_DIR = "/memory/semantic_memory/"
PROCEDURAL_DIR = "/memory/procedural_memory/"


class MemoryKind(StrEnum):
    """The three memory kinds of the CoALA-style taxonomy."""

    EPISODIC = "episodic"
    """What happened: events, feedback and errors, tied to a moment in time."""

    SEMANTIC = "semantic"
    """What is true: facts, rules and preferences, detached from any episode."""

    PROCEDURAL = "procedural"
    """How things are done: repeatable operating procedures."""


class MemoryCategory(StrEnum):
    """A concrete file family inside a memory kind."""

    EVENTS = "events"
    """Episodic: things that happened during a session."""

    FEEDBACKS = "feedbacks"
    """Episodic: corrections and judgements received from a user."""

    ERRORS = "errors"
    """Episodic: mistakes made, so they are not repeated."""

    FACTS = "facts"
    """Semantic: statements believed to be true right now."""

    RULES = "rules"
    """Semantic: constraints and policies that govern behaviour."""

    PREFERENCES = "preferences"
    """Semantic: how the user wants the agent to behave."""

    PROCEDURE = "procedure"
    """Procedural: one file per operating procedure."""


CATEGORY_KIND: dict[MemoryCategory, MemoryKind] = {
    MemoryCategory.EVENTS: MemoryKind.EPISODIC,
    MemoryCategory.FEEDBACKS: MemoryKind.EPISODIC,
    MemoryCategory.ERRORS: MemoryKind.EPISODIC,
    MemoryCategory.FACTS: MemoryKind.SEMANTIC,
    MemoryCategory.RULES: MemoryKind.SEMANTIC,
    MemoryCategory.PREFERENCES: MemoryKind.SEMANTIC,
    MemoryCategory.PROCEDURE: MemoryKind.PROCEDURAL,
}
"""Which kind each category belongs to."""

KIND_DIRECTORIES: dict[MemoryKind, str] = {
    MemoryKind.EPISODIC: EPISODIC_DIR,
    MemoryKind.SEMANTIC: SEMANTIC_DIR,
    MemoryKind.PROCEDURAL: PROCEDURAL_DIR,
}
"""Directory that holds each kind, including its `index.md`."""

_SHARDED_CATEGORIES = frozenset(
    {MemoryCategory.EVENTS, MemoryCategory.FEEDBACKS, MemoryCategory.ERRORS}
)

_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")


def index_path(kind: MemoryKind | None = None) -> str:
    """Return the path of a router index.

    Args:
        kind: Memory kind whose index is wanted. `None` returns the root index.

    Returns:
        Virtual path of the `index.md` for that kind.
    """
    if kind is None:
        return ROOT_INDEX_PATH
    return f"{KIND_DIRECTORIES[kind]}index.md"


def category_directory(category: MemoryCategory) -> str:
    """Return the directory holding a category's files.

    Args:
        category: Category to locate.

    Returns:
        Virtual directory path, with a trailing slash.
    """
    if category is MemoryCategory.PREFERENCES:
        return MEMORY_ROOT
    kind = CATEGORY_KIND[category]
    if category in _SHARDED_CATEGORIES:
        return f"{KIND_DIRECTORIES[kind]}{category.value}/"
    return KIND_DIRECTORIES[kind]


def shard_label(when: datetime | None = None) -> str:
    """Return the monthly shard label (`YYYY-MM`) an entry belongs to.

    Args:
        when: Instant to label. Defaults to now, in UTC.

    Returns:
        The shard label, e.g. `"2026-08"`.
    """
    moment = when or datetime.now(tz=UTC)
    return moment.strftime("%Y-%m")


def slugify(text: str) -> str:
    """Turn free text into a filename-safe slug.

    Args:
        text: Text to slugify, typically a procedure title.

    Returns:
        A lowercase, hyphen-separated slug.

    Raises:
        ValueError: If `text` contains no slug-able character.
    """
    slug = _SLUG_SEPARATORS.sub("-", text.strip().lower()).strip("-")
    if not slug:
        msg = f"cannot build a slug from {text!r}"
        raise ValueError(msg)
    return slug


def entry_path(
    category: MemoryCategory,
    *,
    when: datetime | None = None,
    title: str | None = None,
) -> str:
    """Return the file an entry of this category must be written to.

    Args:
        category: Category of the entry.
        when: Instant the entry refers to; selects the monthly shard for
            episodic categories. Defaults to now, in UTC.
        title: Procedure title, required for
            [`MemoryCategory.PROCEDURE`][deep_memory_agent.layout.MemoryCategory]
            and ignored otherwise.

    Returns:
        Virtual path of the target file.

    Raises:
        ValueError: If a procedure is requested without a title.
    """
    directory = category_directory(category)
    if category is MemoryCategory.PROCEDURE:
        if not title:
            msg = "a procedure entry requires a title"
            raise ValueError(msg)
        return f"{directory}{slugify(title)}.md"
    if category in _SHARDED_CATEGORIES:
        return f"{directory}{shard_label(when)}.md"
    return f"{directory}{category.value}.md"
