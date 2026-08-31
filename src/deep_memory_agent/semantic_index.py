"""The semantic index over the memory tree: ingestion, incremental updates, search.

The markdown files stay the single source of truth. This index is *derived*
data: it can be deleted and rebuilt from `/memory/` at any time, and nothing in
the tree depends on it — the worst a lost index costs is the ability to search by
meaning until the next ingest, never a fact.

Two halves:

- **Ingestion.** Entries are enumerated with
  [`MemoryStore.search`][deep_memory_agent.store.MemoryStore.search], which
  already walks the tree and hands back every entry with its file, category and
  frontmatter — so nothing here re-implements the scan. Each entry is hashed,
  chunked and written to the vector store under deterministic ids. A manifest —
  a small JSON file kept next to the tree, outside the `**/*.md` the store
  globs — records each entry's digest and the ids its chunks went in under,
  which is what lets a second ingest skip what has not changed and remove the
  chunks of entries that are gone.
- **Search.** Dense retrieval over the chunks, with the frontmatter available as
  filters. They run server-side when a `filter_builder` is configured, and
  client-side over a widened candidate set otherwise.

Ingestion is explicit: nothing here is wired into `MemoryStore.write` the way
the router indexes are. The manager calls `semantic_ingest` after writing, or a
job outside the agent calls
[`ingest_semantic_index`][deep_memory_agent.semantic_tools.ingest_semantic_index].
The cost of that choice is a window in which the index trails the files — see
the prompt blocks in [`deep_memory_agent.prompts`][], which say so to both
agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from deep_memory_agent.entry import Confidence
from deep_memory_agent.layout import CATEGORY_KIND, MemoryCategory, MemoryKind
from deep_memory_agent.semantic_chunking import (
    ChunkingConfig,
    chunk_entry,
    entry_hash,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    from deep_memory_agent.store import MemoryHit, MemoryStore

__all__ = [
    "MANIFEST_PATH",
    "IngestReport",
    "SemanticConfig",
    "SemanticIndex",
]

MANIFEST_PATH = "/memory/.semantic-manifest.json"
"""Where the ingest manifest lives.

Inside the memory tree, so it travels with it and is versioned by the same git
repository — but not a `.md` file, so
[`MemoryStore`][deep_memory_agent.store.MemoryStore] never globs it into a
lexical search.
"""

_MANIFEST_VERSION = 1
_FULL_READ_LIMIT = 1_000_000
_ENTRIES_IN_SUMMARY = 5

_CONFIDENCE_ORDER: dict[str, int] = {
    Confidence.LOW.value: 0,
    Confidence.MEDIUM.value: 1,
    Confidence.HIGH.value: 2,
}


@dataclass(frozen=True)
class SemanticConfig:
    """Everything about the index that the model does not get to choose.

    Attributes:
        chunking: How an entry is cut into chunks.
        manifest_path: Where the ingest manifest is kept, on the virtual
            filesystem.
        scan_limit: Ceiling on the entries one ingest enumerates. When it trips,
            pruning is skipped for that call: a partial scan cannot tell a
            deleted entry from one it never reached.
        batch_size: Documents per `add_documents` call.
        over_fetch: Multiplier on `k` when filters are applied client-side, so
            filtering does not empty the result set.
        snippet_chars: Longest snippet per hit in the text handed to the model.
            The full chunk always travels in the tool's artifact.
        filter_builder: Converts the active filters into the store's own filter
            object, for server-side filtering. Left `None`, filters are applied
            client-side over `over_fetch * k` candidates — correct either way,
            but the client-side path gets wasteful on a large store.
    """

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    manifest_path: str = MANIFEST_PATH
    scan_limit: int = 10_000
    batch_size: int = 64
    over_fetch: int = 4
    snippet_chars: int = 500
    filter_builder: Callable[[dict[str, Any]], Any] | None = None


@dataclass
class IngestReport:
    """What one ingest did.

    Counts are in entries; `chunks` and `deleted_chunks` are in chunks, since an
    entry with a long body produces several.

    Attributes:
        added: Entries the index had never seen.
        updated: Entries whose content or frontmatter changed — most often
            because a newer entry superseded them.
        deleted: Entries that were indexed and are no longer in the tree.
        unchanged: Entries skipped because nothing about them moved.
        chunks: Chunks written to the store.
        deleted_chunks: Chunks removed from it, superseded by an update or
            belonging to an entry that is gone.
        entry_ids: Identifiers of the entries that were written.
        errors: Per-entry failures, as `"<entry_id>: <reason>"`. One entry that
            cannot be chunked does not abort the run.
        truncated: Whether `scan_limit` cut the enumeration short. When it did,
            nothing is pruned.
    """

    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    chunks: int = 0
    deleted_chunks: int = 0
    entry_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def changed(self) -> bool:
        """Whether this run altered the index at all."""
        return bool(self.added or self.updated or self.deleted)

    def summary(self) -> str:
        """Render the report as the one-paragraph text a model reads.

        A run that changed nothing says so in as many words rather than
        reporting three zeros: "already aligned" is the answer to "is the index
        current?", and it is the common outcome of an ingest called after every
        write.
        """
        if not self.changed:
            return (
                f"No update needed: {self.unchanged} entries already match the index."
            )
        parts = [
            (
                f"Indexed {self.chunks} chunk(s): {self.added} new entry(ies), "
                f"{self.updated} updated, {self.deleted} removed, "
                f"{self.unchanged} already current."
            )
        ]
        if self.entry_ids:
            shown = ", ".join(self.entry_ids[:_ENTRIES_IN_SUMMARY])
            more = (
                f" and {len(self.entry_ids) - _ENTRIES_IN_SUMMARY} more"
                if len(self.entry_ids) > _ENTRIES_IN_SUMMARY
                else ""
            )
            parts.append(f"Entries: {shown}{more}.")
        if self.deleted_chunks:
            parts.append(f"{self.deleted_chunks} outdated chunk(s) were removed.")
        if self.truncated:
            parts.append(
                "The per-call entry limit was reached, so nothing was pruned; "
                "run the ingest again."
            )
        if self.errors:
            shown = "; ".join(self.errors[:_ENTRIES_IN_SUMMARY])
            parts.append(f"{len(self.errors)} entry(ies) failed: {shown}")
        return " ".join(parts)


class SemanticIndex:
    """Ingestion and search over one memory tree and one vector store.

    The store, the embedding model, the vector store and the configuration are
    all fixed at construction. What a caller — or a model through a tool —
    chooses per call is only which slice of memory to ingest and what to look
    for.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        vector_store: VectorStore,
        store: MemoryStore,
        *,
        search_k: int = 5,
        config: SemanticConfig | None = None,
    ) -> None:
        """Bind an index to a memory tree and a store.

        Args:
            embeddings: Embedding model. Used directly only against a
                dense-only vector store, where embedding the query here saves a
                round trip.
            vector_store: Where the chunks live. It must upsert on a repeated
                id — `add_documents(docs, ids=[...])` overwriting rather than
                duplicating — which is what makes a re-ingest idempotent.
            store: Store over the memory tree. Its `search` is what enumerates
                the entries, and its backend is where the manifest is kept.
            search_k: Default number of results a search returns.
            config: Index configuration. Defaults to
                [`SemanticConfig`][deep_memory_agent.semantic_index.SemanticConfig].
        """
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._store = store
        self._search_k = search_k
        self._config = config or SemanticConfig()

    @property
    def config(self) -> SemanticConfig:
        """The configuration this index was built with."""
        return self._config

    # ------------------------------------------------------------- manifest --
    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        """Return the recorded state of every indexed entry.

        A missing, unreadable or malformed manifest is not an error: it means
        the next ingest is a full one, which is always correct, only slower.
        """
        result = self._store.backend.read(
            self._config.manifest_path, limit=_FULL_READ_LIMIT
        )
        if result.error or not result.file_data:
            return {}
        try:
            payload = json.loads(str(result.file_data.get("content", "")))
        except json.JSONDecodeError:
            return {}
        entries = payload.get("entries") if isinstance(payload, dict) else None
        return entries if isinstance(entries, dict) else {}

    def _write_manifest(self, entries: dict[str, dict[str, Any]]) -> str | None:
        """Persist the manifest through the backend.

        Returns:
            An error message when the manifest could not be written, `None` on
            success. A failure only costs the *next* ingest its incremental fast
            path, so it is reported rather than raised.
        """
        payload = json.dumps(
            {
                "version": _MANIFEST_VERSION,
                "updated_at": datetime.now(UTC).isoformat(),
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        )
        result = self._store.backend.write(self._config.manifest_path, payload)
        return f"manifest not written: {result.error}" if result.error else None

    # ------------------------------------------------------------ ingestion --
    def ingest(
        self,
        *,
        kind: MemoryKind | None = None,
        category: MemoryCategory | None = None,
        only_modified: bool = True,
    ) -> IngestReport:
        """Index the memory tree, or the slice of it `kind`/`category` name.

        The operation is idempotent by construction: running it again with
        nothing changed in the tree leaves the vector store exactly as it was
        and reports that nothing needed doing.

        Args:
            kind: Restrict the ingest to one memory kind. Entries outside it are
                left alone, manifest rows included — a scoped ingest never
                prunes what it did not look at.
            category: Restrict the ingest to one category, same caveat.
            only_modified: Skip entries whose digest matches the manifest. Pass
                `False` to force a rebuild — after a change to the chunking
                parameters, for instance, which the manifest cannot notice.

        Returns:
            What the run did, as an
            [`IngestReport`][deep_memory_agent.semantic_index.IngestReport].
        """
        hits = self._store.search(
            include_superseded=True,
            kind=kind,
            category=category,
            limit=self._config.scan_limit,
        )
        report = IngestReport(truncated=len(hits) >= self._config.scan_limit)

        recorded = self._read_manifest()
        entries: dict[str, dict[str, Any]] = {}
        documents: list[Document] = []
        ids: list[str] = []
        stale: list[str] = []

        for hit in hits:
            stale.extend(
                self._ingest_one(
                    hit,
                    recorded=recorded,
                    entries=entries,
                    documents=documents,
                    ids=ids,
                    report=report,
                    only_modified=only_modified,
                )
            )

        if not report.truncated:
            pruned, removed = self._prune(
                recorded, entries, kind=kind, category=category
            )
            stale.extend(pruned)
            report.deleted = removed
        else:
            entries.update(
                {key: value for key, value in recorded.items() if key not in entries}
            )

        self._add(documents, ids)
        report.deleted_chunks = self._delete(stale, report)

        error = self._write_manifest(entries)
        if error:
            report.errors.append(error)
        return report

    def _ingest_one(
        self,
        hit: MemoryHit,
        *,
        recorded: dict[str, dict[str, Any]],
        entries: dict[str, dict[str, Any]],
        documents: list[Document],
        ids: list[str],
        report: IngestReport,
        only_modified: bool,
    ) -> list[str]:
        """Fold one entry into the run, returning the chunk ids it made stale."""
        entry = hit.entry
        previous = recorded.get(entry.entry_id)
        digest = entry_hash(entry)

        # The path is compared alongside the digest because it is metadata the
        # hash does not cover: an entry moved between files by hand keeps its
        # content but must be re-indexed to stop pointing at the old file.
        if (
            only_modified
            and previous is not None
            and previous.get("hash") == digest
            and previous.get("path") == hit.path
        ):
            report.unchanged += 1
            entries[entry.entry_id] = previous
            return []

        try:
            chunks = chunk_entry(entry, hit.path, config=self._config.chunking)
        except (ImportError, ValueError) as exc:
            report.errors.append(f"{entry.entry_id}: {exc}")
            if previous is not None:
                entries[entry.entry_id] = previous
            return []

        chunk_ids = [str(chunk.id) for chunk in chunks]
        documents.extend(chunks)
        ids.extend(chunk_ids)
        report.chunks += len(chunks)
        report.entry_ids.append(entry.entry_id)
        entries[entry.entry_id] = {
            "hash": digest,
            "path": hit.path,
            "category": entry.category.value,
            "chunk_ids": chunk_ids,
            "ingested_at": datetime.now(UTC).isoformat(),
        }

        if previous is None:
            report.added += 1
            return []
        report.updated += 1
        fresh = set(chunk_ids)
        return [old for old in previous.get("chunk_ids", []) if old not in fresh]

    def _prune(
        self,
        recorded: dict[str, dict[str, Any]],
        entries: dict[str, dict[str, Any]],
        *,
        kind: MemoryKind | None,
        category: MemoryCategory | None,
    ) -> tuple[list[str], int]:
        """Collect the chunks of entries that were indexed and are now gone.

        Nothing in this package deletes an entry — `MemoryStore` has no `delete`
        and `_retire` rewrites an entry rather than removing it — so this path
        exists for changes made outside the agent's API: a shard file deleted or
        moved by hand, a block of entries edited out, a `git checkout` of
        `/memory/` to an earlier state. Without it the index would keep
        answering with entries the files no longer hold, which is the one way a
        derived index can outlive its source and lie about it.

        Args:
            recorded: The manifest as it was before this run.
            entries: The manifest being built, mutated here to carry forward
                every entry this run's scope excluded.
            kind: The kind this run was scoped to, if any.
            category: The category this run was scoped to, if any.

        Returns:
            The chunk ids to delete, and how many entries they came from.
        """
        stale: list[str] = []
        removed = 0
        for entry_id, record in recorded.items():
            if entry_id in entries:
                continue
            if not _in_scope(record.get("category"), kind=kind, category=category):
                entries[entry_id] = record
                continue
            stale.extend(record.get("chunk_ids", []))
            removed += 1
        return stale, removed

    def _add(self, documents: list[Document], ids: list[str]) -> None:
        """Write the new chunks to the store, in batches."""
        size = self._config.batch_size
        for start in range(0, len(documents), size):
            self._vector_store.add_documents(
                documents[start : start + size], ids=ids[start : start + size]
            )

    def _delete(self, stale: list[str], report: IngestReport) -> int:
        """Remove superseded chunks, tolerating a store that cannot.

        Args:
            stale: Ids to remove.
            report: Report to note a refusal on.

        Returns:
            How many chunks were removed.
        """
        if not stale:
            return 0
        try:
            self._vector_store.delete(ids=stale)
        except (NotImplementedError, TypeError) as exc:
            report.errors.append(
                f"{len(stale)} outdated chunk(s) could not be deleted: {exc}"
            )
            return 0
        return len(stale)

    # --------------------------------------------------------------- search --
    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        kind: MemoryKind | str | None = None,
        category: MemoryCategory | str | None = None,
        tags: Sequence[str] = (),
        source: str | None = None,
        min_confidence: Confidence | str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Find the entries closest in meaning to a query.

        Args:
            query: What to look for, in natural language. Unlike
                `MemoryStore.search` this does not need to share words with the
                entry it should find.
            k: How many results to return. Defaults to the index's `search_k`.
            kind: Restrict to one memory kind.
            category: Restrict to one category.
            tags: Only return entries carrying all of these tags.
            source: Only return entries recorded with this source.
            min_confidence: Only return entries at or above this confidence.
            created_after: ISO timestamp; only entries created at or after it.
            created_before: ISO timestamp; only entries created at or before it.
            include_superseded: Also return entries a newer one replaced. The
                default matches `MemoryStore.search` and `memory_search`: an
                agent searching by meaning does not expect a retired fact to
                surface unless it asked for history.

        Returns:
            One dict per hit — rank, score, text, entry id, path, and the whole
            frontmatter — ordered best first. Hits are *chunks*, so an entry
            with a long body can occupy more than one of the `k` slots;
            `entry_id` and `chunk_index` are what a caller regroups them by.
        """
        limit = k or self._search_k
        filters = {
            "kind": _enum_value(kind),
            "category": _enum_value(category),
            "tags": [tag for tag in tags if tag],
            "source": source,
            "min_confidence": _enum_value(min_confidence),
            "created_after": created_after,
            "created_before": created_before,
            "include_superseded": include_superseded,
        }
        active: dict[str, Any] = {
            key: value
            for key, value in filters.items()
            if value and key != "include_superseded"
        }
        if not include_superseded:
            # Not a falsy filter to drop: leaving retired entries out is the
            # default, so it is the one constraint that must still be applied
            # when the caller asked for nothing else.
            active["is_active"] = True

        kwargs: dict[str, Any] = {"k": limit}
        if active and self._config.filter_builder is not None:
            kwargs["filter"] = self._config.filter_builder(active)
        elif active:
            kwargs["k"] = limit * self._config.over_fetch

        client_side = bool(active) and self._config.filter_builder is None
        results: list[dict[str, Any]] = []
        for document, score in self._retrieve(query, kwargs):
            if client_side and not _matches(document.metadata, filters):
                continue
            results.append(_render_hit(document, score, rank=len(results) + 1))
            if len(results) >= limit:
                break
        return results

    def _retrieve(
        self, query: str, kwargs: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        """Query the store, keeping the lexical half of a hybrid store alive.

        A hybrid or sparse store searched by vector would quietly answer with
        its dense half alone, disabling its keyword side without saying so. Only
        a dense-only store is queried by vector; everything else is handed the
        query as text.
        """
        mode = getattr(self._vector_store, "retrieval_mode", None)
        dense_only = (
            mode is None or str(getattr(mode, "value", mode)).lower() == "dense"
        )
        by_vector = hasattr(
            self._vector_store, "similarity_search_with_score_by_vector"
        )
        if dense_only and by_vector:
            vector = self._embeddings.embed_query(query)
            return list(
                self._vector_store.similarity_search_with_score_by_vector(
                    vector, **kwargs
                )
            )
        return list(self._vector_store.similarity_search_with_score(query, **kwargs))


def _enum_value(value: object) -> str | None:
    """Return the plain string behind an enum member, or `None` for a blank."""
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _in_scope(
    recorded_category: object,
    *,
    kind: MemoryKind | None,
    category: MemoryCategory | None,
) -> bool:
    """Whether an ingest scoped this way would have enumerated this category.

    A manifest row whose category no longer parses — the enum changed, the file
    was hand-edited — is only prunable by an unscoped run, which is the one that
    provably looked everywhere.

    Args:
        recorded_category: The `category` field of a manifest row.
        kind: The kind the ingest was scoped to, if any.
        category: The category the ingest was scoped to, if any.

    Returns:
        Whether the row was in this run's scope, and so may be pruned.
    """
    try:
        recorded = MemoryCategory(str(recorded_category))
    except ValueError:
        return kind is None and category is None
    if category is not None and recorded is not category:
        return False
    return not (kind is not None and CATEGORY_KIND[recorded] is not kind)


def _render_hit(
    document: Document, score: float | None, *, rank: int
) -> dict[str, Any]:
    """Turn one retrieved chunk into the dict a caller and a tool both read."""
    metadata = document.metadata
    return {
        "rank": rank,
        "score": float(score) if score is not None else None,
        "text": document.page_content,
        "entry_id": metadata.get("entry_id", ""),
        "path": metadata.get("path", ""),
        "kind": metadata.get("kind", ""),
        "category": metadata.get("category", ""),
        "summary": metadata.get("summary", ""),
        "tags": metadata.get("tags", []),
        "confidence": metadata.get("confidence", ""),
        "source": metadata.get("source", ""),
        "created": metadata.get("created", ""),
        "is_active": metadata.get("is_active", True),
        "chunk_index": metadata.get("chunk_index", 0),
        "metadata": metadata,
    }


def _matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Whether one chunk's metadata satisfies every active filter."""
    if not filters.get("include_superseded") and not metadata.get("is_active", True):
        return False
    for key in ("kind", "category", "source"):
        wanted = filters.get(key)
        if wanted and metadata.get(key) != wanted:
            return False
    wanted_tags = {tag.lower() for tag in filters.get("tags") or ()}
    if wanted_tags and not wanted_tags <= {
        str(tag).lower() for tag in metadata.get("tags", ())
    }:
        return False
    return _matches_confidence(metadata, filters) and _matches_dates(metadata, filters)


def _matches_confidence(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Whether the chunk reaches the requested confidence floor."""
    floor = filters.get("min_confidence")
    if not floor:
        return True
    have = _CONFIDENCE_ORDER.get(str(metadata.get("confidence", "")), -1)
    return have >= _CONFIDENCE_ORDER.get(str(floor), 0)


def _matches_dates(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Whether the chunk's `created` falls inside the requested window."""
    created = _as_datetime(metadata.get("created"))
    if created is None:
        return not (filters.get("created_after") or filters.get("created_before"))
    after = _as_datetime(filters.get("created_after"))
    before = _as_datetime(filters.get("created_before"))
    if after is not None and created < after:
        return False
    return not (before is not None and created > before)


def _as_datetime(value: object) -> datetime | None:
    """Parse an ISO timestamp, returning `None` when it is absent or unusable."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
