from datetime import UTC, datetime

from deep_memory_agent.consolidation import ConsolidatedItem, consolidate_memory
from deep_memory_agent.layout import MemoryCategory

AUGUST = datetime(2026, 8, 24, tzinfo=UTC)


def test_consolidation_promotes_an_episode_to_a_fact(
    backend, store, consolidation_model
):
    store.write(MemoryCategory.EVENTS, "The user said they always use uv.", when=AUGUST)
    model = consolidation_model(
        ConsolidatedItem(
            category="facts",
            summary="Uses uv",
            content="The user manages Python projects with uv.",
            tags=["tooling"],
            confidence="high",
        )
    )

    result = consolidate_memory(backend, model)

    assert result.episodes_considered == 1
    assert len(result.entries) == 1
    (hit,) = store.search("uv", category=MemoryCategory.FACTS)
    assert hit.entry.source == "consolidation"
    assert hit.entry.confidence.value == "high"


def test_consolidation_leaves_episodes_untouched(backend, store, consolidation_model):
    episode = store.write(MemoryCategory.EVENTS, "An episode.", when=AUGUST)
    model = consolidation_model(
        ConsolidatedItem(category="facts", summary="A fact", content="A fact.")
    )

    consolidate_memory(backend, model)

    assert store.get(episode.entry.entry_id).entry.is_active


def test_consolidation_can_supersede_an_existing_fact(
    backend, store, consolidation_model
):
    store.write(MemoryCategory.EVENTS, "The user moved to Enterprise.", when=AUGUST)
    old = store.write(MemoryCategory.FACTS, "ACME is on Team.", when=AUGUST)
    model = consolidation_model(
        ConsolidatedItem(
            category="facts",
            summary="ACME plan",
            content="ACME is on Enterprise.",
            supersedes=old.entry.entry_id,
        )
    )

    consolidate_memory(backend, model)

    assert not store.get(old.entry.entry_id).entry.is_active


def test_a_hallucinated_supersedes_id_is_dropped(backend, store, consolidation_model):
    store.write(MemoryCategory.EVENTS, "An episode.", when=AUGUST)
    model = consolidation_model(
        ConsolidatedItem(
            category="facts",
            summary="A fact",
            content="A fact.",
            supersedes="mem_1970-01-01_000000",
        )
    )

    result = consolidate_memory(backend, model)

    assert result.entries[0].entry.supersedes is None


def test_a_procedure_without_a_title_is_skipped(backend, store, consolidation_model):
    store.write(MemoryCategory.EVENTS, "An episode.", when=AUGUST)
    model = consolidation_model(
        ConsolidatedItem(category="procedure", summary="How to", content="## Steps")
    )

    assert consolidate_memory(backend, model).entries == []


def test_consolidation_writes_a_procedure_with_its_title(
    backend, store, consolidation_model
):
    store.write(MemoryCategory.EVENTS, "An episode.", when=AUGUST)
    model = consolidation_model(
        ConsolidatedItem(
            category="procedure",
            summary="How to answer pricing",
            content="## Steps\n\n1. Ask.",
            title="Answer pricing questions",
        )
    )

    (hit,) = consolidate_memory(backend, model).entries

    assert hit.path == "/memory/procedural_memory/answer-pricing-questions.md"


def test_consolidation_without_episodes_does_nothing(backend, consolidation_model):
    result = consolidate_memory(backend, consolidation_model())

    assert result.entries == []
    assert result.rationale == "no episodes to consolidate"


def test_an_empty_proposal_is_a_valid_outcome(backend, store, consolidation_model):
    store.write(MemoryCategory.EVENTS, "An episode.", when=AUGUST)

    result = consolidate_memory(
        backend, consolidation_model(rationale="nothing stable yet")
    )

    assert result.entries == []
    assert result.rationale == "nothing stable yet"


def test_consolidation_only_reads_episodes_since_the_given_instant(
    backend, store, consolidation_model
):
    store.write(MemoryCategory.EVENTS, "Old.", when=AUGUST)
    store.write(MemoryCategory.EVENTS, "New.", when=datetime(2026, 9, 2, tzinfo=UTC))

    result = consolidate_memory(
        backend, consolidation_model(), since=datetime(2026, 9, 1, tzinfo=UTC)
    )

    assert result.episodes_considered == 1
