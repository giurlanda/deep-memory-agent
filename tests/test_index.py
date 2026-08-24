from datetime import UTC, datetime

import pytest

from deep_memory_agent.index import (
    IndexRow,
    index_target,
    parse_index,
    read_index_rows,
    render_index_table,
    update_index,
)
from deep_memory_agent.layout import (
    ROOT_INDEX_PATH,
    MemoryCategory,
    MemoryKind,
    index_path,
)

WHEN = datetime(2026, 8, 24, tzinfo=UTC)


def test_index_target_routes_to_the_owning_kind():
    target, relative = index_target(
        MemoryCategory.EVENTS, "/memory/episodic_memory/events/2026-08.md"
    )

    assert target == index_path(MemoryKind.EPISODIC)
    assert relative == "events/2026-08.md"


def test_preferences_are_tracked_by_the_root_index():
    target, relative = index_target(
        MemoryCategory.PREFERENCES, "/memory/preferences.md"
    )

    assert target == ROOT_INDEX_PATH
    assert relative == "preferences.md"


@pytest.mark.usefixtures("store")
def test_update_index_adds_a_row(backend):
    update_index(
        backend,
        category=MemoryCategory.EVENTS,
        file_path="/memory/episodic_memory/events/2026-08.md",
        description="August sessions",
        tags=("acme",),
        when=WHEN,
    )

    rows = read_index_rows(backend, index_path(MemoryKind.EPISODIC))
    assert rows["events/2026-08.md"] == IndexRow(
        path="events/2026-08.md",
        description="August sessions",
        tags=("acme",),
        updated="2026-08-24",
    )


@pytest.mark.usefixtures("store")
def test_update_index_merges_tags_and_refreshes_the_description(backend):
    for description, tags in (("first", ("acme",)), ("second", ("pricing",))):
        update_index(
            backend,
            category=MemoryCategory.EVENTS,
            file_path="/memory/episodic_memory/events/2026-08.md",
            description=description,
            tags=tags,
            when=WHEN,
        )

    row = read_index_rows(backend, index_path(MemoryKind.EPISODIC))["events/2026-08.md"]
    assert row.description == "second"
    assert row.tags == ("acme", "pricing")


@pytest.mark.usefixtures("store")
def test_update_index_keeps_the_hand_written_preamble(backend):
    update_index(
        backend,
        category=MemoryCategory.FACTS,
        file_path="/memory/semantic_memory/facts.md",
        description="Facts",
        when=WHEN,
    )

    content = backend.read(index_path(MemoryKind.SEMANTIC)).file_data["content"]
    assert content.startswith("# Semantic memory index")


@pytest.mark.usefixtures("store")
def test_index_never_holds_entry_content(backend):
    update_index(
        backend,
        category=MemoryCategory.FACTS,
        file_path="/memory/semantic_memory/facts.md",
        description="ACME plan",
        when=WHEN,
    )

    _, rows = parse_index(
        backend.read(index_path(MemoryKind.SEMANTIC)).file_data["content"]
    )
    assert all(len(row.description) < 200 for row in rows.values())


def test_pipes_in_a_description_cannot_break_the_table():
    table = render_index_table(
        {
            "facts.md": IndexRow(
                path="facts.md", description="a | b", updated="2026-08-24"
            )
        }
    )

    _, rows = parse_index(table)
    assert list(rows) == ["facts.md"]


def test_parse_index_ignores_header_and_separator_rows():
    _, rows = parse_index(
        "| File | Description | Tags | Updated |\n| --- | --- | --- | --- |\n"
    )

    assert rows == {}
