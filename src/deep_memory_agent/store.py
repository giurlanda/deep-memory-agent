"""Read and write operations over the memory tree.

[`MemoryStore`][deep_memory_agent.store.MemoryStore] is the single place that
knows how memory is laid out on the virtual filesystem: which file an entry
belongs in, how frontmatter is appended, how a superseded entry is retired and
how the router indexes are kept in step. The agent tools in
[`deep_memory_agent.tools`][] are thin wrappers over it, which keeps the file
format testable without spinning up a model.

Every operation goes through the deepagents backend. Nothing in this module
opens a host path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from deep_memory_agent.entry import (
    Confidence,
    MemoryEntry,
    new_entry_id,
    render_document,
    render_entry,
    split_document,
)
from deep_memory_agent.index import update_index
from deep_memory_agent.layout import (
    KIND_DIRECTORIES,
    MEMORY_ROOT,
    MemoryCategory,
    MemoryKind,
    entry_path,
)
from deep_memory_agent.scaffold import ensure_memory_tree

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

__all__ = ["MemoryHit", "MemoryStore"]

_DEFAULT_SEARCH_LIMIT = 20
_READ_LIMIT = 100_000


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """An entry found by a search, with the file it lives in.

    Attributes:
        path: Absolute virtual path of the file holding the entry.
        entry: The entry itself.
    """

    path: str
    entry: MemoryEntry


class MemoryStore:
    """File-backed store for episodic, semantic and procedural memory.

    Args:
        backend: Backend serving the `/memory/` tree.
    """

    def __init__(self, backend: BackendProtocol) -> None:
        """Bind the store to a backend."""
        self._backend = backend

    @property
    def backend(self) -> BackendProtocol:
        """The backend every operation goes through."""
        return self._backend

    def ensure_tree(self) -> list[str]:
        """Create any missing file of the memory tree.

        Returns:
            The paths that were created.
        """
        return ensure_memory_tree(self._backend)

    def read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a memory file.

        Args:
            path: Absolute virtual path, which must sit under `/memory/`.
            offset: First line to read, 0-indexed.
            limit: Maximum number of lines to read.

        Returns:
            The file's content.

        Raises:
            ValueError: If the path escapes the memory tree or cannot be read.
        """
        self._check_path(path)
        result = self._backend.read(path, offset=offset, limit=limit)
        if result.error or not result.file_data:
            msg = result.error or f"{path} is empty or unreadable"
            raise ValueError(msg)
        return str(result.file_data.get("content", ""))

    def write(
        self,
        category: MemoryCategory,
        body: str,
        *,
        summary: str = "",
        tags: tuple[str, ...] = (),
        source: str = "agent",
        confidence: Confidence = Confidence.MEDIUM,
        supersedes: str | None = None,
        title: str | None = None,
        when: datetime | None = None,
    ) -> MemoryHit:
        """Append an entry to the file its category maps to.

        The target file is created with a heading if it does not exist yet, the
        router index of the owning kind is updated, and — when `supersedes` is
        given — the replaced entry is retired in place.

        Args:
            category: Which file family the entry belongs to.
            body: Markdown content of the entry.
            summary: One-line description, used in the router index.
            tags: Labels used to narrow later searches.
            source: Where the information came from.
            confidence: How much the entry is trusted.
            supersedes: Identifier of an entry this one replaces.
            title: Procedure title; required for procedural entries.
            when: Instant the entry refers to. Defaults to now, in UTC.

        Returns:
            The stored entry and the file it landed in.

        Raises:
            ValueError: If the entry cannot be rendered or the write fails.
        """
        moment = when or datetime.now(tz=UTC)
        entry = MemoryEntry(
            entry_id=new_entry_id(moment),
            created=moment,
            category=category,
            body=body,
            summary=summary,
            source=source,
            confidence=confidence,
            tags=tags,
            supersedes=supersedes,
        )
        path = entry_path(category, when=moment, title=title)
        rendered = render_entry(entry)

        existing = self._read_or_empty(path)
        content = f"{existing.rstrip()}\n\n{rendered}" if existing.strip() else rendered
        self._write(path, content)

        if supersedes:
            self._retire(supersedes, entry.entry_id)

        update_index(
            self._backend,
            category=category,
            file_path=path,
            description=summary,
            tags=tags,
            when=moment,
        )
        return MemoryHit(path=path, entry=entry)

    def get(self, entry_id: str) -> MemoryHit | None:
        """Find an entry by identifier.

        Args:
            entry_id: Identifier to look for.

        Returns:
            The entry and its file, or `None` if no entry carries that id.
        """
        for path in self._memory_files():
            for entry in split_document(self._read_or_empty(path))[1]:
                if entry.entry_id == entry_id:
                    return MemoryHit(path=path, entry=entry)
        return None

    def search(
        self,
        query: str = "",
        *,
        kind: MemoryKind | None = None,
        category: MemoryCategory | None = None,
        tags: tuple[str, ...] = (),
        include_superseded: bool = False,
        limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> list[MemoryHit]:
        """Search entries lexically, newest first.

        Matching is a case-insensitive substring test over summary, body and
        tags. It is deliberately simple: the frontmatter is what makes a
        stronger index — BM25, embeddings — addable later without changing the
        files themselves.

        Args:
            query: Text to look for. Empty matches everything.
            kind: Restrict to one memory kind.
            category: Restrict to one category.
            tags: Only return entries carrying all of these tags.
            include_superseded: Whether to return entries a newer one replaced.
            limit: Maximum number of hits.

        Returns:
            Matching entries, most recent first.
        """
        needle = query.strip().lower()
        wanted_tags = {tag.lower() for tag in tags}
        hits: list[MemoryHit] = []

        for path in self._memory_files(kind):
            for entry in split_document(self._read_or_empty(path))[1]:
                if not include_superseded and not entry.is_active:
                    continue
                if category is not None and entry.category is not category:
                    continue
                entry_tags = {tag.lower() for tag in entry.tags}
                if not wanted_tags <= entry_tags:
                    continue
                if needle and needle not in self._haystack(entry):
                    continue
                hits.append(MemoryHit(path=path, entry=entry))

        hits.sort(key=lambda hit: hit.entry.created, reverse=True)
        return hits[:limit]

    def recent_episodes(
        self,
        *,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MemoryHit]:
        """Return recent episodic entries, oldest first.

        Args:
            since: Only return entries created at or after this instant.
            limit: Maximum number of entries.

        Returns:
            Episodic entries in chronological order.
        """
        hits = self.search(kind=MemoryKind.EPISODIC, limit=10_000)
        if since is not None:
            hits = [hit for hit in hits if hit.entry.created >= since]
        return sorted(hits, key=lambda hit: hit.entry.created)[:limit]

    def _retire(self, entry_id: str, replaced_by: str) -> None:
        """Mark an entry as superseded, rewriting its file in place."""
        hit = self.get(entry_id)
        if hit is None:
            return
        preamble, entries = split_document(self._read_or_empty(hit.path))
        updated = [
            entry.replaced_by(replaced_by) if entry.entry_id == entry_id else entry
            for entry in entries
        ]
        self._write(hit.path, render_document(preamble, updated))

    def _memory_files(self, kind: MemoryKind | None = None) -> list[str]:
        """List the markdown files of the tree, skipping router indexes."""
        root = KIND_DIRECTORIES[kind] if kind is not None else MEMORY_ROOT
        result = self._backend.glob(f"{root}**/*.md")
        if result.error:
            return []
        paths = [str(match["path"]) for match in result.matches or []]
        return sorted(path for path in paths if not path.endswith("/index.md"))

    def _read_or_empty(self, path: str) -> str:
        """Read a file, returning `""` when it does not exist."""
        result = self._backend.read(path, limit=_READ_LIMIT)
        if result.error or not result.file_data:
            return ""
        return str(result.file_data.get("content", ""))

    def _write(self, path: str, content: str) -> None:
        """Write a file through the backend, surfacing backend errors."""
        result = self._backend.write(path, content)
        if result.error:
            raise ValueError(result.error)

    @staticmethod
    def _haystack(entry: MemoryEntry) -> str:
        """Return the lowercased text a query is matched against."""
        return " ".join((entry.summary, entry.body, *entry.tags)).lower()

    @staticmethod
    def _check_path(path: str) -> None:
        """Reject paths that would step outside the memory tree."""
        if not path.startswith(MEMORY_ROOT) or ".." in path:
            msg = f"{path!r} is outside the memory tree ({MEMORY_ROOT})"
            raise ValueError(msg)
