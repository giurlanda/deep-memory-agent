from datetime import UTC, datetime

import pytest

from deep_memory_agent.entry import Confidence, MemoryEntry
from deep_memory_agent.layout import MemoryCategory
from deep_memory_agent.semantic_chunking import (
    ChunkingConfig,
    chunk_entry,
    chunk_id,
    entry_hash,
)

PATH = "/memory/semantic_memory/facts.md"


def build_entry(body="ACME moved to Enterprise.", **overrides):
    fields = {
        "entry_id": "mem_2026-08-24_9f3a1c",
        "created": datetime(2026, 8, 24, 10, 15, tzinfo=UTC),
        "category": MemoryCategory.FACTS,
        "body": body,
        "summary": "ACME moved to the Enterprise plan",
        "source": "user_message",
        "confidence": Confidence.HIGH,
        "tags": ("pricing", "acme"),
    }
    return MemoryEntry(**{**fields, **overrides})


def test_short_body_makes_one_chunk():
    chunks = chunk_entry(build_entry(), PATH)

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_count"] == 1


def test_chunk_ids_are_uuids_derived_from_entry_and_position():
    import uuid

    first = chunk_id("mem_2026-08-24_9f3a1c", 0)

    # A UUID, not the raw entry id: several stores accept nothing else.
    assert uuid.UUID(first)
    assert first == chunk_id("mem_2026-08-24_9f3a1c", 0)
    assert first != chunk_id("mem_2026-08-24_9f3a1c", 1)
    assert first != chunk_id("mem_2026-08-24_other", 0)


def test_chunk_carries_its_derived_id():
    (chunk,) = chunk_entry(build_entry(), PATH)

    assert chunk.id == chunk_id("mem_2026-08-24_9f3a1c", 0)


def test_metadata_carries_the_whole_frontmatter():
    (chunk,) = chunk_entry(build_entry(), PATH)

    assert chunk.metadata == {
        "entry_id": "mem_2026-08-24_9f3a1c",
        "path": PATH,
        "kind": "semantic",
        "category": "facts",
        "summary": "ACME moved to the Enterprise plan",
        "source": "user_message",
        "confidence": "high",
        "tags": ["pricing", "acme"],
        "created": "2026-08-24T10:15:00+00:00",
        "supersedes": "",
        "superseded_by": "",
        "is_active": True,
        "chunk_index": 0,
        "chunk_count": 1,
    }


def test_superseded_entry_is_marked_inactive_in_metadata():
    entry = build_entry(superseded_by="mem_2026-09-01_aaaaaa")

    (chunk,) = chunk_entry(entry, PATH)

    assert chunk.metadata["is_active"] is False
    assert chunk.metadata["superseded_by"] == "mem_2026-09-01_aaaaaa"


def test_summary_is_prepended_so_a_chunk_says_what_it_is_about():
    (chunk,) = chunk_entry(build_entry(), PATH)

    assert chunk.page_content.startswith("ACME moved to the Enterprise plan")
    assert "ACME moved to Enterprise." in chunk.page_content


def test_summary_can_be_left_out():
    config = ChunkingConfig(prepend_summary=False)

    (chunk,) = chunk_entry(build_entry(), PATH, config=config)

    assert chunk.page_content == "ACME moved to Enterprise."


def test_long_body_is_split_across_numbered_chunks():
    body = "\n\n".join(f"## Step {i}\n" + ("word " * 100) for i in range(6))

    chunks = chunk_entry(build_entry(body=body), PATH)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert {chunk.metadata["chunk_count"] for chunk in chunks} == {len(chunks)}
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(
        chunk.metadata["entry_id"] == "mem_2026-08-24_9f3a1c" for chunk in chunks
    )


def test_split_respects_the_configured_chunk_size():
    body = "word " * 400
    config = ChunkingConfig(chunk_size=200, chunk_overlap=20)

    chunks = chunk_entry(build_entry(body=body), PATH, config=config)

    assert len(chunks) > 1


def test_empty_body_still_produces_one_chunk():
    chunks = chunk_entry(build_entry(body=""), PATH)

    assert len(chunks) == 1
    assert chunks[0].page_content == "ACME moved to the Enterprise plan"


@pytest.mark.parametrize(
    "overrides",
    [
        {"body": "something else"},
        {"summary": "another summary"},
        {"tags": ("pricing",)},
        {"confidence": Confidence.LOW},
        {"source": "agent"},
        {"supersedes": "mem_2026-01-01_000000"},
        {"superseded_by": "mem_2026-12-01_000000"},
        {"category": MemoryCategory.RULES},
    ],
)
def test_hash_changes_when_any_indexed_field_changes(overrides):
    assert entry_hash(build_entry()) != entry_hash(build_entry(**overrides))


def test_hash_is_stable_for_an_untouched_entry():
    assert entry_hash(build_entry()) == entry_hash(build_entry())


def test_retiring_an_entry_changes_its_hash():
    entry = build_entry()

    retired = entry.replaced_by("mem_2026-09-01_aaaaaa")

    # This is the one mutation the store performs on an existing entry, so it
    # has to read as "changed" or a retired entry stays active in the index.
    assert entry_hash(entry) != entry_hash(retired)
