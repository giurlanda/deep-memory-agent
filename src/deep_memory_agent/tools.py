"""Agent-facing memory tools.

Each tool is a thin wrapper over
[`MemoryStore`][deep_memory_agent.store.MemoryStore], bound to a backend by
closure. That binding is what enforces the core constraint of this package:
tools reach memory **only** through the deepagents backend, never through the
host filesystem, so the same agent works unchanged against a directory on disk,
thread state, or a remote store.

Tools are split in two sets on purpose. The recall set is read-only and is all
the search agent gets; the write set is what makes the manager agent the single
writer of the tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

from deep_memory_agent.consolidation import consolidate_memory
from deep_memory_agent.entry import Confidence
from deep_memory_agent.index import read_index_rows
from deep_memory_agent.layout import MemoryCategory, MemoryKind, index_path

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from deep_memory_agent.store import MemoryHit, MemoryStore

__all__ = ["build_recall_tools", "build_write_tools"]

_MAX_SEARCH_LIMIT = 50
_EXCERPT_CHARS = 800


def build_recall_tools(store: MemoryStore) -> list[BaseTool]:
    """Build the read-only memory tools.

    Args:
        store: Store bound to the backend serving `/memory/`.

    Returns:
        The `memory_index`, `memory_search` and `memory_read` tools.
    """

    @tool
    def memory_index(kind: str | None = None) -> str:
        """List what memory holds, without loading any of it.

        Read this before searching: indexes are routers that name every file
        with a one-line description, its tags and when it last changed.

        Args:
            kind: One of `episodic`, `semantic`, `procedural`, or omitted for
                the top-level index of the whole tree.

        Returns:
            The router index as a markdown table.
        """
        try:
            target = index_path(MemoryKind(kind)) if kind else index_path()
        except ValueError:
            return f"Error: unknown memory kind {kind!r}."
        rows = read_index_rows(store.backend, target)
        if not rows:
            return f"{target} lists no files yet."
        lines = [f"{target}:"]
        lines += [
            f"- {row.path} — {row.description or 'no description'} "
            f"[tags: {', '.join(row.tags) or '-'}] [updated: {row.updated or '-'}]"
            for row in sorted(rows.values(), key=lambda item: item.path)
        ]
        return "\n".join(lines)

    @tool
    def memory_search(
        query: str = "",
        kind: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        *,
        include_superseded: bool = False,
    ) -> str:
        """Search memory entries by text and tags, most recent first.

        Matching is a case-insensitive substring test over each entry's summary,
        body and tags, so prefer short distinctive terms over full sentences,
        and search more than once with different wordings rather than relying on
        one phrasing to hit.

        Args:
            query: Text to look for. Leave empty to browse by kind or tag.
            kind: Restrict to `episodic`, `semantic` or `procedural`.
            category: Restrict to `events`, `feedbacks`, `errors`, `facts`,
                `rules`, `preferences` or `procedure`.
            tags: Only return entries carrying all of these tags.
            limit: Maximum number of entries to return.
            include_superseded: Also return entries that a newer entry replaced.
                Use only when the history itself is what matters.

        Returns:
            The matching entries with their id, file, date, confidence and body.
        """
        try:
            hits = store.search(
                query,
                kind=MemoryKind(kind) if kind else None,
                category=MemoryCategory(category) if category else None,
                tags=tuple(tags or ()),
                include_superseded=include_superseded,
                limit=max(1, min(limit, _MAX_SEARCH_LIMIT)),
            )
        except ValueError as exc:
            return f"Error: {exc}"
        if not hits:
            return "No memory entry matched. Memory may simply not hold this yet."
        return "\n\n".join(_format_hit(hit) for hit in hits)

    @tool
    def memory_read(path: str, offset: int = 0, limit: int = 400) -> str:
        """Read a memory file in full.

        Use this after `memory_index` or `memory_search` has pointed at a file
        and you need its whole content rather than matching entries.

        Args:
            path: Absolute path under `/memory/`, as printed by the other tools.
            offset: First line to read, 0-indexed.
            limit: Maximum number of lines to read.

        Returns:
            The file content, or an error message if it cannot be read.
        """
        try:
            return store.read_file(path, offset=offset, limit=limit)
        except ValueError as exc:
            return f"Error: {exc}"

    return [memory_index, memory_search, memory_read]


def build_write_tools(
    store: MemoryStore,
    *,
    consolidation_model: str | BaseChatModel | None = None,
) -> list[BaseTool]:
    """Build the memory-writing tools.

    Args:
        store: Store bound to the backend serving `/memory/`.
        consolidation_model: Model used by `memory_consolidate`. When `None`,
            the tool is not built, since consolidation needs a model to judge
            what has hardened into durable knowledge.

    Returns:
        The `memory_write` and `memory_update` tools, plus `memory_consolidate`
        when a model was given.
    """

    @tool
    def memory_write(
        category: str,
        content: str,
        summary: str,
        tags: list[str] | None = None,
        source: str = "agent",
        confidence: str = "medium",
        title: str | None = None,
    ) -> str:
        """Record a new memory entry.

        Search first: if an entry already covers this, use `memory_update` so
        the old statement is retired instead of coexisting with the new one.
        Write one idea per entry — small entries can be superseded precisely.

        Args:
            category: `events`, `feedbacks` or `errors` for something that
                happened; `facts`, `rules` or `preferences` for something that
                is true; `procedure` for something repeatable.
            content: Markdown body. For a procedure, use the sections: When to
                use, Preconditions, Steps, Tools required, Known failures.
            summary: One-line description. This lands in the router index and is
                what makes the entry findable, so make it specific.
            tags: Short labels for later retrieval, e.g. `["pricing", "acme"]`.
            source: Where this came from: `user_message`, `user_feedback`,
                `tool_result`, or `agent` for your own inference.
            confidence: `low`, `medium` or `high`. Use `high` only for something
                the user stated directly or you verified.
            title: Procedure title. Required when category is `procedure`; it
                becomes the filename.

        Returns:
            The id and path of the stored entry, or an error message.
        """
        try:
            hit = store.write(
                MemoryCategory(category),
                content,
                summary=summary,
                tags=tuple(tags or ()),
                source=source,
                confidence=Confidence(confidence),
                title=title,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return f"Stored {hit.entry.entry_id} in {hit.path}."

    @tool
    def memory_update(
        entry_id: str,
        content: str,
        summary: str = "",
        tags: list[str] | None = None,
        source: str = "agent",
        confidence: str = "medium",
    ) -> str:
        """Replace an existing entry with a corrected one.

        The new entry records what is true now and points at the old one via
        `supersedes`; the old one is marked `superseded_by` and drops out of
        normal search results, but stays on disk as history.

        Args:
            entry_id: Id of the entry being replaced, as printed by
                `memory_search`.
            content: Markdown body of the corrected entry.
            summary: One-line description of the corrected entry. Defaults to
                the summary of the entry being replaced.
            tags: Labels for the corrected entry. Defaults to the old entry's.
            source: Where the correction came from.
            confidence: `low`, `medium` or `high` for the corrected entry.

        Returns:
            The id and path of the new entry, or an error message.
        """
        previous = store.get(entry_id)
        if previous is None:
            return f"Error: no memory entry with id {entry_id!r}."
        try:
            hit = store.write(
                previous.entry.category,
                content,
                summary=summary or previous.entry.summary,
                tags=tuple(tags) if tags is not None else previous.entry.tags,
                source=source,
                confidence=Confidence(confidence),
                supersedes=entry_id,
                title=previous.path.rsplit("/", 1)[-1].removesuffix(".md"),
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return f"Stored {hit.entry.entry_id} in {hit.path}, superseding {entry_id}."

    tools: list[BaseTool] = [memory_write, memory_update]
    if consolidation_model is None:
        return tools

    @tool
    def memory_consolidate(limit: int = 100) -> str:
        """Promote stable patterns from episodic memory into durable knowledge.

        Reads recent episodes and writes what has hardened as facts, rules or
        procedures. Episodes are never deleted — consolidation only adds durable
        knowledge and supersedes what it contradicts. Writing nothing is a
        normal outcome.

        Args:
            limit: Maximum number of recent episodes to read.

        Returns:
            What was written, or a note that nothing had hardened yet.
        """
        result = consolidate_memory(store.backend, consolidation_model, limit=limit)
        if not result.entries:
            reason = result.rationale or "nothing had hardened into durable knowledge"
            return f"Consolidated {result.episodes_considered} episodes: {reason}."
        written = "; ".join(
            f"{hit.entry.entry_id} -> {hit.path}" for hit in result.entries
        )
        return (
            f"Consolidated {result.episodes_considered} episodes into "
            f"{len(result.entries)} entries: {written}."
        )

    tools.append(memory_consolidate)
    return tools


def _format_hit(hit: MemoryHit) -> str:
    """Render one search hit for the model."""
    entry = hit.entry
    body = entry.body.strip()
    excerpt = body[:_EXCERPT_CHARS] + ("…" if len(body) > _EXCERPT_CHARS else "")
    header = (
        f"[{entry.entry_id}] {hit.path} | {entry.category.value} | "
        f"{entry.created:%Y-%m-%d} | confidence: {entry.confidence.value} | "
        f"source: {entry.source} | tags: {', '.join(entry.tags) or '-'}"
    )
    if not entry.is_active:
        header += f" | SUPERSEDED BY {entry.superseded_by}"
    summary = f"\n{entry.summary}" if entry.summary else ""
    return f"{header}{summary}\n{excerpt}"
