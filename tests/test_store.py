from datetime import UTC, datetime

import pytest

from deep_memory_agent.entry import Confidence
from deep_memory_agent.layout import MemoryCategory, MemoryKind
from deep_memory_agent.store import MemoryStore

AUGUST = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
SEPTEMBER = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)


def test_write_appends_to_the_category_file(store):
    hit = store.write(
        MemoryCategory.EVENTS,
        "The user asked about ACME pricing.",
        summary="Pricing question",
        tags=("acme",),
        when=AUGUST,
    )

    assert hit.path == "/memory/episodic_memory/events/2026-08.md"
    assert "ACME pricing" in store.read_file(hit.path)


def test_write_lands_on_the_host_as_markdown(store, memory_dir):
    store.write(MemoryCategory.FACTS, "ACME is on Enterprise.", when=AUGUST)

    assert (
        "ACME is on Enterprise."
        in (memory_dir / "semantic_memory" / "facts.md").read_text()
    )


def test_write_shards_episodes_by_month(store):
    august = store.write(MemoryCategory.EVENTS, "August", when=AUGUST)
    september = store.write(MemoryCategory.EVENTS, "September", when=SEPTEMBER)

    assert august.path != september.path
    assert "September" not in store.read_file(august.path)


def test_write_updates_the_router_index(store):
    store.write(
        MemoryCategory.FACTS,
        "ACME is on Enterprise.",
        summary="ACME plan",
        tags=("acme",),
        when=AUGUST,
    )

    index = store.read_file("/memory/semantic_memory/index.md")
    assert "| facts.md | ACME plan | facts, acme | 2026-08-24 |" in index


def test_write_keeps_earlier_entries(store):
    store.write(MemoryCategory.FACTS, "First fact.", when=AUGUST)
    store.write(MemoryCategory.FACTS, "Second fact.", when=SEPTEMBER)

    content = store.read_file("/memory/semantic_memory/facts.md")
    assert "First fact." in content
    assert "Second fact." in content


def test_supersession_retires_the_previous_entry(store):
    old = store.write(MemoryCategory.FACTS, "ACME is on Team.", when=AUGUST)

    new = store.write(
        MemoryCategory.FACTS,
        "ACME is on Enterprise.",
        supersedes=old.entry.entry_id,
        when=SEPTEMBER,
    )

    retired = store.get(old.entry.entry_id)
    assert retired.entry.superseded_by == new.entry.entry_id
    assert not retired.entry.is_active
    assert store.get(new.entry.entry_id).entry.supersedes == old.entry.entry_id


def test_superseded_entries_drop_out_of_search(store):
    old = store.write(MemoryCategory.FACTS, "ACME is on Team.", when=AUGUST)
    store.write(
        MemoryCategory.FACTS,
        "ACME is on Enterprise.",
        supersedes=old.entry.entry_id,
        when=SEPTEMBER,
    )

    active = store.search("ACME")
    history = store.search("ACME", include_superseded=True)

    assert [hit.entry.body for hit in active] == ["ACME is on Enterprise."]
    assert len(history) == 2


def test_search_is_case_insensitive_and_newest_first(store):
    store.write(MemoryCategory.FACTS, "ACME uses uv.", when=AUGUST)
    store.write(MemoryCategory.FACTS, "ACME uses ruff.", when=SEPTEMBER)

    hits = store.search("acme")

    assert [hit.entry.body for hit in hits] == ["ACME uses ruff.", "ACME uses uv."]


def test_search_filters_by_kind_category_and_tags(store):
    store.write(MemoryCategory.EVENTS, "An episode.", tags=("acme",), when=AUGUST)
    store.write(MemoryCategory.FACTS, "A fact.", tags=("acme", "pricing"), when=AUGUST)

    assert len(store.search(kind=MemoryKind.EPISODIC)) == 1
    assert len(store.search(category=MemoryCategory.FACTS)) == 1
    assert len(store.search(tags=("pricing",))) == 1
    assert len(store.search(tags=("acme", "missing"))) == 0


def test_search_skips_router_indexes(store):
    assert store.search("router") == []


def test_search_honours_its_limit(store):
    for number in range(5):
        store.write(MemoryCategory.FACTS, f"Fact {number}.", when=AUGUST)

    assert len(store.search("Fact", limit=2)) == 2


def test_recent_episodes_are_chronological_and_episodic_only(store):
    store.write(MemoryCategory.FACTS, "A fact.", when=AUGUST)
    store.write(MemoryCategory.EVENTS, "Later.", when=SEPTEMBER)
    store.write(MemoryCategory.ERRORS, "Earlier.", when=AUGUST)

    episodes = store.recent_episodes()

    assert [hit.entry.body for hit in episodes] == ["Earlier.", "Later."]


def test_recent_episodes_can_start_from_an_instant(store):
    store.write(MemoryCategory.EVENTS, "Old.", when=AUGUST)
    store.write(MemoryCategory.EVENTS, "New.", when=SEPTEMBER)

    episodes = store.recent_episodes(since=SEPTEMBER)

    assert [hit.entry.body for hit in episodes] == ["New."]


def test_get_returns_none_for_an_unknown_id(store):
    assert store.get("mem_1970-01-01_000000") is None


def test_procedures_get_their_own_file(store):
    hit = store.write(
        MemoryCategory.PROCEDURE,
        "## Steps\n\n1. Ask.",
        summary="How to answer pricing",
        title="Answer pricing questions",
        when=AUGUST,
    )

    assert hit.path == "/memory/procedural_memory/answer-pricing-questions.md"


def test_confidence_travels_with_the_entry(store):
    hit = store.write(MemoryCategory.FACTS, "Verified.", confidence=Confidence.HIGH)

    assert store.get(hit.entry.entry_id).entry.confidence is Confidence.HIGH


def test_read_file_refuses_paths_outside_the_memory_tree(store):
    with pytest.raises(ValueError, match="outside the memory tree"):
        store.read_file("/etc/passwd")


def test_read_file_refuses_traversal(store):
    with pytest.raises(ValueError, match="outside the memory tree"):
        store.read_file("/memory/../etc/passwd")


def test_store_scaffolds_through_the_backend_only(backend, memory_dir):
    MemoryStore(backend).ensure_tree()

    assert (memory_dir / "index.md").exists()
