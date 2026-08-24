from datetime import UTC, datetime

import pytest

from deep_memory_agent.layout import (
    EPISODIC_DIR,
    PREFERENCES_PATH,
    ROOT_INDEX_PATH,
    MemoryCategory,
    MemoryKind,
    category_directory,
    entry_path,
    index_path,
    shard_label,
    slugify,
)

AUGUST = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


def test_index_path_defaults_to_root():
    assert index_path() == ROOT_INDEX_PATH
    assert index_path(MemoryKind.EPISODIC) == f"{EPISODIC_DIR}index.md"


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (MemoryCategory.EVENTS, "/memory/episodic_memory/events/2026-08.md"),
        (MemoryCategory.FEEDBACKS, "/memory/episodic_memory/feedbacks/2026-08.md"),
        (MemoryCategory.ERRORS, "/memory/episodic_memory/errors/2026-08.md"),
        (MemoryCategory.FACTS, "/memory/semantic_memory/facts.md"),
        (MemoryCategory.RULES, "/memory/semantic_memory/rules.md"),
        (MemoryCategory.PREFERENCES, PREFERENCES_PATH),
    ],
)
def test_entry_path_routes_by_category(category, expected):
    assert entry_path(category, when=AUGUST) == expected


def test_episodic_entries_are_sharded_by_month():
    september = datetime(2026, 9, 1, tzinfo=UTC)
    assert entry_path(MemoryCategory.EVENTS, when=AUGUST) != entry_path(
        MemoryCategory.EVENTS, when=september
    )
    assert shard_label(september) == "2026-09"


def test_procedures_are_one_file_per_title():
    path = entry_path(MemoryCategory.PROCEDURE, title="Answer pricing questions!")
    assert path == "/memory/procedural_memory/answer-pricing-questions.md"


def test_procedure_without_title_is_rejected():
    with pytest.raises(ValueError, match="requires a title"):
        entry_path(MemoryCategory.PROCEDURE)


def test_unsluggable_title_is_rejected():
    with pytest.raises(ValueError, match="cannot build a slug"):
        slugify("!!!")


def test_preferences_live_at_the_root():
    assert category_directory(MemoryCategory.PREFERENCES) == "/memory/"
