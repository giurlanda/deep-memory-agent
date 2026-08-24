from datetime import UTC, datetime

import pytest

from deep_memory_agent.entry import (
    Confidence,
    MemoryEntry,
    new_entry_id,
    parse_entries,
    render_document,
    render_entry,
    split_document,
)
from deep_memory_agent.layout import MemoryCategory, MemoryKind

CREATED = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


def make_entry(**overrides):
    defaults = {
        "entry_id": "mem_2026-08-24_aaaaaa",
        "created": CREATED,
        "category": MemoryCategory.FACTS,
        "body": "ACME is on the Enterprise plan.",
        "summary": "ACME plan",
        "tags": ("acme", "pricing"),
    }
    return MemoryEntry(**{**defaults, **overrides})


def test_entry_kind_follows_its_category():
    assert make_entry().kind is MemoryKind.SEMANTIC
    assert make_entry(category=MemoryCategory.ERRORS).kind is MemoryKind.EPISODIC


def test_render_parse_round_trip_preserves_provenance():
    entry = make_entry(
        source="user_message",
        confidence=Confidence.HIGH,
        supersedes="mem_2026-06-01_bbbbbb",
    )
    (parsed,) = parse_entries(render_entry(entry))

    assert parsed.entry_id == entry.entry_id
    assert parsed.created == entry.created
    assert parsed.category is entry.category
    assert parsed.body == entry.body
    assert parsed.summary == entry.summary
    assert parsed.tags == entry.tags
    assert parsed.source == "user_message"
    assert parsed.confidence is Confidence.HIGH
    assert parsed.supersedes == "mem_2026-06-01_bbbbbb"


def test_multiple_entries_are_parsed_in_file_order():
    second = make_entry(entry_id="mem_2026-08-25_cccccc", body="Second\n\nparagraph.")
    text = render_document("# Facts", [make_entry(), second])

    parsed = parse_entries(text)

    assert [entry.entry_id for entry in parsed] == [
        "mem_2026-08-24_aaaaaa",
        "mem_2026-08-25_cccccc",
    ]
    assert parsed[1].body == "Second\n\nparagraph."


def test_split_document_keeps_the_preamble():
    document = render_document("# Facts\n\nIntro.", [make_entry()])

    preamble, entries = split_document(document)

    assert preamble == "# Facts\n\nIntro."
    assert len(entries) == 1


def test_summary_with_a_colon_survives_the_round_trip():
    (parsed,) = parse_entries(render_entry(make_entry(summary="ACME: moved plan")))

    assert parsed.summary == "ACME: moved plan"


def test_body_with_a_bare_fence_is_rejected():
    with pytest.raises(ValueError, match="bare '---' line"):
        render_entry(make_entry(body="before\n---\nafter"))


def test_malformed_entries_are_skipped_not_raised():
    text = "---\nnot: an entry\n---\n\nbody\n\n" + render_entry(make_entry())

    parsed = parse_entries(text)

    assert [entry.entry_id for entry in parsed] == ["mem_2026-08-24_aaaaaa"]


def test_superseded_entries_are_inactive():
    entry = make_entry()
    assert entry.is_active

    retired = entry.replaced_by("mem_2026-09-01_dddddd")

    assert not retired.is_active
    assert retired.superseded_by == "mem_2026-09-01_dddddd"
    assert entry.is_active, "the original entry must not be mutated"


def test_entry_ids_are_unique_and_dated():
    first, second = new_entry_id(CREATED), new_entry_id(CREATED)

    assert first.startswith("mem_2026-08-24_")
    assert first != second
