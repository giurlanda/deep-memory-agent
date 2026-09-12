"""Cutting memory entries into the chunks the semantic index embeds.

Pure functions over entries: nothing here touches a backend, a vector store or
an embedding model, so what gets embedded — and under which id — can be
inspected and tested on its own.

A memory entry is not a wiki page. The manager prompt asks for *one idea per
entry*, so a body is usually a single paragraph and one chunk is the whole
entry. Procedures are the exception: their body follows a fixed set of sections
(`## When to use`, `## Preconditions`, `## Steps`, …) and comfortably outgrows a
single useful chunk. Hence the rule below — one chunk per entry, split only when
the body passes `chunk_size`.

Chunk ids are always derived with `uuid5` from `entry_id::chunk_index`, never
the raw `entry_id`. Two reasons: a body can always end up split in two, at which
point one id is not enough anyway, and several stores (Qdrant among them) accept
only UUID-shaped ids, which `mem_2026-08-24_9f3a1c` is not. Deriving them the
same way every time is also what makes a re-ingest *update* a chunk instead of
duplicating it.

Splitting a long body needs the optional `semantic` extra::

    pip install "deep-memory-agent[semantic]"

The import happens inside
[`chunk_entry`][deep_memory_agent.semantic_chunking.chunk_entry], so importing
this module — and indexing entries that fit in one chunk — stays free.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from deep_memory_agent.entry import MemoryEntry

__all__ = [
    "ChunkingConfig",
    "chunk_entry",
    "chunk_id",
    "entry_hash",
]

_ID_NAMESPACE = uuid.UUID("9b1d2a3c-6f47-5e8a-b0c1-2d3e4f5a6b7c")
"""Namespace the chunk ids are derived in.

Fixed forever: changing it would orphan every chunk already written to a store
rather than update it, leaving the old vectors behind with no way to reach them.
"""

_MISSING_DEPENDENCY_HINT = (
    "splitting a long entry body requires the optional `semantic` extra: "
    'install it with `pip install "deep-memory-agent[semantic]"` '
    '(or `uv add "deep-memory-agent[semantic]"`).'
)

_HASH_CHARS = 16
_FIELD_SEPARATOR = "\x00"


@dataclass(frozen=True)
class ChunkingConfig:
    """How an entry is cut into chunks.

    Fixed when the index is built, never exposed to the model: chunk sizes are a
    property of the index, and an agent that could change them per call would
    produce an index whose parts do not compare.

    Attributes:
        chunk_size: Body length, in characters, past which the body is split.
            Defaults to the excerpt size `memory_search` already uses, for
            internal consistency — not to a measurement of real procedure
            bodies, which is what should eventually calibrate it.
        chunk_overlap: Characters repeated between adjacent chunks, so a
            sentence cut in two is still retrievable from either side.
        prepend_summary: Put the entry's one-line summary at the top of every
            chunk's text. Costs a few tokens and buys a chunk that still says
            what it is about once it is out of its entry.
    """

    chunk_size: int = 800
    chunk_overlap: int = 100
    prepend_summary: bool = True


def chunk_id(entry_id: str, chunk_index: int) -> str:
    """Derive the stable vector-store id of one chunk.

    Args:
        entry_id: Identifier of the entry the chunk came from.
        chunk_index: Position of the chunk inside that entry, 0-indexed.

    Returns:
        A UUID string, the same one for the same `(entry_id, chunk_index)` pair
        on every ingest.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, f"{entry_id}::{chunk_index}"))


def entry_hash(entry: MemoryEntry) -> str:
    """Return the digest an entry is tracked by in the ingest manifest.

    Everything that ends up embedded or stored as metadata is folded in, so a
    retirement — `_retire` writing `superseded_by` — reads as a change and gets
    re-indexed, while a re-ingest of an untouched entry does not.

    Args:
        entry: Entry to digest.

    Returns:
        A short hex digest.
    """
    material = _FIELD_SEPARATOR.join(
        (
            entry.category.value,
            entry.summary,
            entry.body,
            ",".join(entry.tags),
            entry.confidence.value,
            entry.source,
            entry.supersedes or "",
            entry.superseded_by or "",
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_HASH_CHARS]


def chunk_entry(
    entry: MemoryEntry,
    path: str,
    *,
    config: ChunkingConfig | None = None,
) -> list[Document]:
    """Cut one entry into the chunks the index stores, ids and metadata included.

    Args:
        entry: The entry to index.
        path: Absolute virtual path of the file holding it, as
            [`MemoryStore.search`][deep_memory_agent.store.MemoryStore.search]
            reports it. This is what a search hit points at, so it must be the
            path the agent's own file tools would use.
        config: Chunking parameters. Defaults to
            [`ChunkingConfig`][deep_memory_agent.semantic_chunking.ChunkingConfig].

    Returns:
        One `Document` per chunk, in body order, each carrying its derived id
        and the entry's whole frontmatter as metadata.

    Raises:
        ImportError: If the body needs splitting and the optional `semantic`
            extra is not installed.
    """
    config = config or ChunkingConfig()
    body = entry.body.strip()
    pieces = [body] if len(body) <= config.chunk_size else _split_body(body, config)

    documents: list[Document] = []
    for index, piece in enumerate(pieces):
        content = piece
        if config.prepend_summary and entry.summary:
            content = f"{entry.summary}\n\n{piece}".strip()
        documents.append(
            Document(
                id=chunk_id(entry.entry_id, index),
                page_content=content,
                metadata=_metadata(entry, path, index, len(pieces)),
            )
        )
    return documents


def _split_body(body: str, config: ChunkingConfig) -> list[str]:
    """Split a body that outgrew one chunk, or return it whole if it cannot be.

    Raises:
        ImportError: If the optional `semantic` extra is not installed.
    """
    # Imported here, not at module scope: the extra is opt-in, and importing
    # this module must stay free for anyone who never builds an index.
    try:
        from langchain_text_splitters import (  # noqa: PLC0415
            RecursiveCharacterTextSplitter,
        )
    except ImportError as exc:
        raise ImportError(_MISSING_DEPENDENCY_HINT) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_text(body) or [body]


def _metadata(
    entry: MemoryEntry,
    path: str,
    chunk_index: int,
    chunk_count: int,
) -> dict[str, Any]:
    """Build one chunk's metadata: the entry's frontmatter, plus where it sits.

    The whole frontmatter travels with every chunk on purpose — it is what
    `semantic_search` filters on, so a filter never has to go back to the file
    to decide whether a hit qualifies.
    """
    return {
        "entry_id": entry.entry_id,
        "path": path,
        "kind": entry.kind.value,
        "category": entry.category.value,
        "summary": entry.summary,
        "source": entry.source,
        "confidence": entry.confidence.value,
        "tags": list(entry.tags),
        "created": entry.created.isoformat(),
        "supersedes": entry.supersedes or "",
        "superseded_by": entry.superseded_by or "",
        "is_active": entry.is_active,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
    }
