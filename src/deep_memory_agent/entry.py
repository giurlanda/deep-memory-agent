"""Memory entries and their YAML frontmatter.

A memory file is an append-only markdown document made of *entries*. Each entry
carries a YAML frontmatter block so that provenance (when, from where, how sure)
travels with the content, and so a newer statement can explicitly supersede an
older one instead of silently coexisting with it.

```markdown
---
id: mem_2026-08-24_9f3a1c
created: 2026-08-24T10:15:00+00:00
type: semantic
category: facts
source: user_message
confidence: high
tags: [pricing, acme]
supersedes: mem_2026-06-01_4b2e77
summary: ACME moved to the Enterprise plan
---

ACME switched from the Team plan to Enterprise on 2026-08-24.
```

The format stays plain markdown on purpose: readable by a human, diffable by
git, and cheap to index later with BM25 or embeddings without changing the
source of truth.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum

import yaml

from deep_memory_agent.layout import CATEGORY_KIND, MemoryCategory, MemoryKind

__all__ = [
    "Confidence",
    "MemoryEntry",
    "new_entry_id",
    "parse_entries",
    "render_document",
    "render_entry",
    "split_document",
]

_FRONTMATTER_FENCE = "---"
_YAML_RESERVED_PREFIXES = ("[", "{", "*", "&", "!", "%", "@", "`", ">", "|", "'", '"')


class Confidence(StrEnum):
    """How much the agent trusts an entry."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def new_entry_id(when: datetime | None = None) -> str:
    """Mint a unique, sortable-by-day entry identifier.

    Args:
        when: Instant the entry was created. Defaults to now, in UTC.

    Returns:
        An identifier such as `"mem_2026-08-24_9f3a1c"`.
    """
    moment = when or datetime.now(tz=UTC)
    return f"mem_{moment:%Y-%m-%d}_{secrets.token_hex(3)}"


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single unit of memory, with its provenance.

    Attributes:
        entry_id: Unique identifier, used by `supersedes`/`superseded_by`.
        created: Creation instant, always timezone-aware.
        category: Which file family the entry belongs to.
        body: Markdown content of the entry.
        summary: One-line description, used to build router indexes.
        source: Where the information came from, e.g. `"user_message"`.
        confidence: How much the entry is trusted.
        tags: Free-form labels used to narrow searches.
        supersedes: Identifier of the entry this one replaces, if any.
        superseded_by: Identifier of the entry that replaced this one, if any.
    """

    entry_id: str
    created: datetime
    category: MemoryCategory
    body: str
    summary: str = ""
    source: str = "agent"
    confidence: Confidence = Confidence.MEDIUM
    tags: tuple[str, ...] = field(default_factory=tuple)
    supersedes: str | None = None
    superseded_by: str | None = None

    @property
    def kind(self) -> MemoryKind:
        """Memory kind this entry's category belongs to."""
        return CATEGORY_KIND[self.category]

    @property
    def is_active(self) -> bool:
        """Whether the entry still holds, i.e. nothing has superseded it."""
        return self.superseded_by is None

    def replaced_by(self, entry_id: str) -> MemoryEntry:
        """Return a copy of this entry marked as superseded.

        Args:
            entry_id: Identifier of the entry that replaces this one.

        Returns:
            A new entry; the original is left untouched.
        """
        return replace(self, superseded_by=entry_id)


def render_entry(entry: MemoryEntry) -> str:
    """Render an entry as a frontmatter block followed by its body.

    Args:
        entry: Entry to render.

    Returns:
        Markdown text ending with a trailing newline.

    Raises:
        ValueError: If the body contains a line that is exactly `---`, which
            would be indistinguishable from the start of the next entry.
    """
    body = entry.body.strip()
    if any(line.strip() == _FRONTMATTER_FENCE for line in body.splitlines()):
        msg = f"entry {entry.entry_id} body must not contain a bare '---' line"
        raise ValueError(msg)

    lines = [
        _FRONTMATTER_FENCE,
        f"id: {entry.entry_id}",
        f"created: {entry.created.isoformat()}",
        f"type: {entry.kind.value}",
        f"category: {entry.category.value}",
        f"source: {entry.source}",
        f"confidence: {entry.confidence.value}",
        f"tags: [{', '.join(entry.tags)}]",
    ]
    if entry.supersedes:
        lines.append(f"supersedes: {entry.supersedes}")
    if entry.superseded_by:
        lines.append(f"superseded_by: {entry.superseded_by}")
    if entry.summary:
        lines.append(f"summary: {_scalar(entry.summary)}")
    lines += [_FRONTMATTER_FENCE, "", body, ""]
    return "\n".join(lines)


def parse_entries(text: str) -> list[MemoryEntry]:
    """Parse every well-formed entry out of a memory file.

    Args:
        text: Full content of a memory file.

    Returns:
        The entries found, in file order.
    """
    return split_document(text)[1]


def split_document(text: str) -> tuple[str, list[MemoryEntry]]:
    """Split a memory file into its heading preamble and its entries.

    Blocks whose frontmatter is unreadable — hand-edited files happen — are
    skipped rather than raising, so a single malformed entry never blinds the
    agent to the rest of the file. Skipped blocks are also dropped from the
    round-trip, so callers that rewrite a file should expect to lose them.

    Args:
        text: Full content of a memory file.

    Returns:
        A `(preamble, entries)` pair. The preamble is everything above the first
        entry, kept verbatim.
    """
    entries: list[MemoryEntry] = []
    lines = text.splitlines()
    total = len(lines)
    cursor = 0
    preamble_end = total

    while cursor < total:
        if lines[cursor].strip() != _FRONTMATTER_FENCE:
            cursor += 1
            continue

        closing = _find_fence(lines, cursor + 1)
        if closing is None:
            break

        preamble_end = min(preamble_end, cursor)
        body_end = _find_next_entry(lines, closing + 1)
        entry = _build_entry(
            "\n".join(lines[cursor + 1 : closing]),
            "\n".join(lines[closing + 1 : body_end]).strip(),
        )
        if entry is not None:
            entries.append(entry)
        cursor = body_end

    return "\n".join(lines[:preamble_end]).rstrip(), entries


def render_document(preamble: str, entries: list[MemoryEntry]) -> str:
    """Render a whole memory file from its preamble and entries.

    Args:
        preamble: Heading text to keep above the entries.
        entries: Entries to render, in the order they should appear.

    Returns:
        The full file content, ending with a trailing newline.
    """
    blocks = [render_entry(entry).rstrip() for entry in entries]
    head = preamble.rstrip()
    parts = [part for part in (head, *blocks) if part]
    return "\n\n".join(parts) + "\n"


def _find_fence(lines: list[str], start: int) -> int | None:
    """Return the index of the next closing fence, or `None` if unterminated."""
    for index in range(start, len(lines)):
        if lines[index].strip() == _FRONTMATTER_FENCE:
            return index
    return None


def _find_next_entry(lines: list[str], start: int) -> int:
    """Return the index where the next entry begins, or the end of the file.

    An entry always starts on a fence line preceded by a blank line, which is
    how `render_entry` lays entries out.
    """
    for index in range(start, len(lines)):
        preceded_by_blank = index > 0 and not lines[index - 1].strip()
        if lines[index].strip() == _FRONTMATTER_FENCE and preceded_by_blank:
            return index
    return len(lines)


def _build_entry(frontmatter: str, body: str) -> MemoryEntry | None:
    """Build an entry from a frontmatter block, or `None` if it is unusable."""
    try:
        meta = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None

    entry_id = meta.get("id")
    category = _as_category(meta.get("category"))
    created = _as_datetime(meta.get("created"))
    if not entry_id or category is None or created is None:
        return None

    return MemoryEntry(
        entry_id=str(entry_id),
        created=created,
        category=category,
        body=body,
        summary=str(meta.get("summary") or ""),
        source=str(meta.get("source") or "agent"),
        confidence=_as_confidence(meta.get("confidence")),
        tags=_as_tags(meta.get("tags")),
        supersedes=_optional_str(meta.get("supersedes")),
        superseded_by=_optional_str(meta.get("superseded_by")),
    )


def _as_category(value: object) -> MemoryCategory | None:
    try:
        return MemoryCategory(str(value))
    except ValueError:
        return None


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _as_confidence(value: object) -> Confidence:
    try:
        return Confidence(str(value))
    except ValueError:
        return Confidence.MEDIUM


def _as_tags(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(tag.strip() for tag in value.strip("[]").split(",") if tag.strip())
    if isinstance(value, list):
        return tuple(str(tag).strip() for tag in value if str(tag).strip())
    return ()


def _optional_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _scalar(text: str) -> str:
    """Quote a summary when YAML would otherwise misread it."""
    collapsed = " ".join(text.split())
    if collapsed.startswith(_YAML_RESERVED_PREFIXES) or ": " in collapsed:
        escaped = collapsed.replace('"', '\\"')
        return f'"{escaped}"'
    return collapsed
