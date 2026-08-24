"""Router indexes.

Every memory directory owns an `index.md` whose only job is to let an agent
decide *what to load* without loading everything: one row per file, holding a
one-line description, the tags seen in it and the date it last changed — never
the memory content itself.

The index is a markdown table so it stays readable and diffable:

```markdown
| File | Description | Tags | Updated |
| --- | --- | --- | --- |
| events/2026-08.md | Session events for August 2026 | acme, pricing | 2026-08-24 |
```
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from deep_memory_agent.layout import (
    CATEGORY_KIND,
    KIND_DIRECTORIES,
    MEMORY_ROOT,
    ROOT_INDEX_PATH,
    MemoryCategory,
    index_path,
)

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

__all__ = [
    "IndexRow",
    "index_target",
    "parse_index",
    "read_index_rows",
    "render_index_table",
    "update_index",
]

_TABLE_HEADER = "| File | Description | Tags | Updated |"
_TABLE_SEPARATOR = "| --- | --- | --- | --- |"
_MAX_ROW_TAGS = 12
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


@dataclass(frozen=True, slots=True)
class IndexRow:
    """One line of a router index.

    Attributes:
        path: File path, relative to the directory holding the index.
        description: One-line summary of what the file contains.
        tags: Labels seen across the file's entries.
        updated: Date of the last write, as `YYYY-MM-DD`.
    """

    path: str
    description: str = ""
    tags: tuple[str, ...] = ()
    updated: str = ""

    def to_row(self) -> str:
        """Render the row as a markdown table line."""
        return (
            f"| {self.path} | {_cell(self.description)} "
            f"| {_cell(', '.join(self.tags))} | {self.updated} |"
        )


def index_target(category: MemoryCategory, file_path: str) -> tuple[str, str]:
    """Return which index tracks a file, and under which relative path.

    Args:
        category: Category of the entry that was written.
        file_path: Absolute virtual path of the file that was written.

    Returns:
        A `(index_path, relative_path)` pair.
    """
    if category is MemoryCategory.PREFERENCES:
        return ROOT_INDEX_PATH, file_path.removeprefix(MEMORY_ROOT)
    kind = CATEGORY_KIND[category]
    return index_path(kind), file_path.removeprefix(KIND_DIRECTORIES[kind])


def parse_index(text: str) -> tuple[str, dict[str, IndexRow]]:
    """Split index content into its preamble and its rows.

    Args:
        text: Full content of an `index.md`.

    Returns:
        A `(preamble, rows)` pair, where `rows` is keyed by relative path.
        The preamble is everything above the table, kept verbatim so a
        hand-written explanation survives updates.
    """
    preamble_lines: list[str] = []
    rows: dict[str, IndexRow] = {}
    in_table = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            in_table = True
            row = _parse_row(stripped)
            if row is not None:
                rows[row.path] = row
            continue
        if not in_table:
            preamble_lines.append(line)

    return "\n".join(preamble_lines).rstrip(), rows


def render_index_table(rows: dict[str, IndexRow]) -> str:
    """Render index rows as a markdown table, sorted by path.

    Args:
        rows: Rows keyed by relative path.

    Returns:
        The table, without a trailing newline.
    """
    lines = [_TABLE_HEADER, _TABLE_SEPARATOR]
    lines += [rows[path].to_row() for path in sorted(rows)]
    return "\n".join(lines)


def read_index_rows(backend: BackendProtocol, path: str) -> dict[str, IndexRow]:
    """Read the rows of an index through the backend.

    Args:
        backend: Backend serving the memory tree.
        path: Absolute virtual path of the `index.md`.

    Returns:
        Rows keyed by relative path; empty if the index does not exist yet.
    """
    _, rows = parse_index(_read(backend, path))
    return rows


def update_index(
    backend: BackendProtocol,
    *,
    category: MemoryCategory,
    file_path: str,
    description: str = "",
    tags: tuple[str, ...] = (),
    when: datetime | None = None,
) -> str:
    """Record a file in its router index, creating or merging its row.

    Tags accumulate across writes so the index keeps working as a lookup table;
    the description of the most recent write wins, since it is the freshest
    statement about what the file now holds.

    Args:
        backend: Backend serving the memory tree.
        category: Category of the entry that was written.
        file_path: Absolute virtual path of the file that was written.
        description: One-line summary of the entry just written.
        tags: Tags of the entry just written.
        when: Instant of the write. Defaults to now, in UTC.

    Returns:
        The absolute virtual path of the index that was updated.
    """
    target, relative = index_target(category, file_path)
    preamble, rows = parse_index(_read(backend, target))

    existing = rows.get(relative)
    merged_tags = tuple(dict.fromkeys((*(existing.tags if existing else ()), *tags)))
    rows[relative] = IndexRow(
        path=relative,
        description=description or (existing.description if existing else ""),
        tags=merged_tags[:_MAX_ROW_TAGS],
        updated=(when or datetime.now(tz=UTC)).strftime("%Y-%m-%d"),
    )

    table = render_index_table(rows)
    backend.write(target, f"{preamble}\n\n{table}\n" if preamble else f"{table}\n")
    return target


def _read(backend: BackendProtocol, path: str) -> str:
    """Return a file's content through the backend, or `""` if unreadable."""
    result = backend.read(path, limit=100_000)
    if result.error or not result.file_data:
        return ""
    return str(result.file_data.get("content", ""))


def _parse_row(line: str) -> IndexRow | None:
    """Parse one table line, skipping the header and separator rows."""
    cells = [
        cell.replace("\\|", "|").strip() for cell in _UNESCAPED_PIPE.split(line.strip())
    ]
    if cells and not cells[0]:
        cells = cells[1:]
    if cells and not cells[-1]:
        cells = cells[:-1]
    expected_cells = 4
    if len(cells) != expected_cells:
        return None
    path, description, tags, updated = cells
    if not path or path == "File" or set(path) <= {"-", " ", ":"}:
        return None
    return IndexRow(
        path=path,
        description=description,
        tags=tuple(tag.strip() for tag in tags.split(",") if tag.strip()),
        updated=updated,
    )


def _cell(text: str) -> str:
    """Flatten text so it cannot break out of a markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")
