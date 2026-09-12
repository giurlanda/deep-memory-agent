import json
from datetime import UTC, datetime

import pytest

from deep_memory_agent.entry import Confidence
from deep_memory_agent.layout import MemoryCategory, MemoryKind
from deep_memory_agent.semantic_chunking import ChunkingConfig
from deep_memory_agent.semantic_index import (
    MANIFEST_PATH,
    IngestReport,
    SemanticConfig,
    SemanticIndex,
)

JANUARY = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
JUNE = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


def write_fact(store, body="ACME is on the Enterprise plan.", **kwargs):
    return store.write(MemoryCategory.FACTS, body, summary="ACME plan", **kwargs)


def manifest(memory_dir):
    return json.loads((memory_dir / ".semantic-manifest.json").read_text())


# --------------------------------------------------------------- ingestion --


def test_first_ingest_indexes_every_entry(store, semantic_index, vector_store):
    write_fact(store)
    store.write(MemoryCategory.EVENTS, "Kickoff call.", summary="Kickoff")

    report = semantic_index.ingest()

    assert report.added == 2
    assert report.updated == report.deleted == report.unchanged == 0
    assert report.chunks == 2
    assert len(vector_store.store) == 2


def test_second_ingest_changes_nothing_and_says_so(store, semantic_index):
    write_fact(store)
    semantic_index.ingest()

    report = semantic_index.ingest()

    assert not report.changed
    assert report.unchanged == 1
    assert report.summary() == "No update needed: 1 entries already match the index."


def test_ingest_is_idempotent_over_the_store(store, semantic_index, vector_store):
    write_fact(store)
    semantic_index.ingest()
    before = dict(vector_store.store)

    semantic_index.ingest()

    assert vector_store.store.keys() == before.keys()


def test_a_new_entry_is_added_without_touching_the_others(store, semantic_index):
    write_fact(store)
    semantic_index.ingest()

    store.write(MemoryCategory.RULES, "Discount policy.", summary="Discounts")
    report = semantic_index.ingest()

    assert (report.added, report.updated, report.unchanged) == (1, 0, 1)


def test_supersession_re_indexes_the_retired_entry(store, semantic_index):
    first = write_fact(store)
    semantic_index.ingest()

    write_fact(store, body="ACME moved to Team.", supersedes=first.entry.entry_id)
    report = semantic_index.ingest()

    # The new entry is added; the old one changed because `_retire` wrote
    # `superseded_by` on it, which is the mutation that has to be noticed.
    assert (report.added, report.updated) == (1, 1)
    hits = semantic_index.search("ACME", k=10, include_superseded=True)
    retired = [hit for hit in hits if hit["entry_id"] == first.entry.entry_id]
    assert retired
    assert retired[0]["is_active"] is False


def test_a_retired_entry_stops_surfacing_by_default(store, semantic_index):
    first = write_fact(store)
    write_fact(store, body="ACME moved to Team.", supersedes=first.entry.entry_id)
    semantic_index.ingest()

    hits = semantic_index.search("ACME", k=10)

    assert first.entry.entry_id not in {hit["entry_id"] for hit in hits}


def test_only_modified_false_rebuilds_everything(store, semantic_index):
    write_fact(store)
    semantic_index.ingest()

    report = semantic_index.ingest(only_modified=False)

    assert report.updated == 1
    assert report.unchanged == 0


def test_a_long_body_occupies_several_chunks_under_one_manifest_row(
    store, semantic_index, memory_dir, vector_store
):
    store.write(
        MemoryCategory.PROCEDURE,
        "\n\n".join(f"## Step {i}\n" + ("word " * 100) for i in range(6)),
        summary="Rollback a half-migrated deploy",
        title="Rollback deploy",
    )

    report = semantic_index.ingest()

    assert report.added == 1
    assert report.chunks > 1
    (row,) = manifest(memory_dir)["entries"].values()
    assert len(row["chunk_ids"]) == report.chunks
    assert len(vector_store.store) == report.chunks


# ---------------------------------------------------------------- manifest --


def test_manifest_lands_outside_what_the_store_globs(store, semantic_index, memory_dir):
    write_fact(store)

    semantic_index.ingest()

    assert (memory_dir / ".semantic-manifest.json").exists()
    assert MANIFEST_PATH not in {hit.path for hit in store.search(limit=100)}


def test_manifest_records_hash_path_category_and_chunk_ids(
    store, semantic_index, memory_dir
):
    hit = write_fact(store)

    semantic_index.ingest()

    row = manifest(memory_dir)["entries"][hit.entry.entry_id]
    assert row["path"] == hit.path
    assert row["category"] == "facts"
    assert row["hash"]
    assert len(row["chunk_ids"]) == 1


def test_a_malformed_manifest_only_costs_a_full_re_ingest(
    store, semantic_index, memory_dir
):
    write_fact(store)
    semantic_index.ingest()
    (memory_dir / ".semantic-manifest.json").write_text("{ not json")

    report = semantic_index.ingest()

    assert report.added == 1
    assert not report.errors


def test_a_missing_manifest_is_treated_as_a_first_ingest(
    store, semantic_index, memory_dir
):
    write_fact(store)
    semantic_index.ingest()
    (memory_dir / ".semantic-manifest.json").unlink()

    report = semantic_index.ingest()

    assert report.added == 1


# ------------------------------------------- entries removed out of band ----


def test_removing_a_whole_shard_file_removes_its_chunks(
    store, semantic_index, memory_dir, vector_store
):
    kept = write_fact(store)
    store.write(MemoryCategory.EVENTS, "Kickoff call.", summary="Kickoff", when=JUNE)
    semantic_index.ingest()

    (memory_dir / "episodic_memory" / "events" / "2026-06.md").unlink()
    report = semantic_index.ingest()

    assert report.deleted == 1
    assert report.deleted_chunks == 1
    assert len(vector_store.store) == 1
    assert manifest(memory_dir)["entries"].keys() == {kept.entry.entry_id}


def test_removing_one_entry_from_a_shared_file_removes_only_its_chunks(
    store, semantic_index, memory_dir, vector_store
):
    first = write_fact(store, body="ACME is on Enterprise.")
    second = write_fact(store, body="ACME renews in March.")
    semantic_index.ingest()
    assert len(vector_store.store) == 2

    facts = memory_dir / "semantic_memory" / "facts.md"
    text = facts.read_text()
    kept, _, _ = text.partition(f"---\nid: {second.entry.entry_id}")
    facts.write_text(kept.rstrip() + "\n")
    report = semantic_index.ingest()

    assert report.deleted == 1
    assert manifest(memory_dir)["entries"].keys() == {first.entry.entry_id}
    assert len(vector_store.store) == 1


def test_a_rollback_of_the_tree_is_reflected_in_the_index(
    store, semantic_index, memory_dir, vector_store
):
    first = write_fact(store, body="ACME is on Enterprise.")
    facts = memory_dir / "semantic_memory" / "facts.md"
    snapshot = facts.read_text()
    later = write_fact(store, body="ACME renews in March.")
    semantic_index.ingest()

    facts.write_text(snapshot)  # git checkout of /memory/ to an earlier state
    report = semantic_index.ingest()

    assert report.deleted == 1
    assert report.unchanged == 1
    ids = {hit["entry_id"] for hit in semantic_index.search("ACME", k=10)}
    assert ids == {first.entry.entry_id}
    assert later.entry.entry_id not in ids
    assert len(vector_store.store) == 1


def test_a_store_that_cannot_delete_reports_it_instead_of_raising(
    store, semantic_index, memory_dir, vector_store
):
    write_fact(store)
    store.write(MemoryCategory.EVENTS, "Kickoff.", summary="Kickoff", when=JUNE)
    semantic_index.ingest()
    (memory_dir / "episodic_memory" / "events" / "2026-06.md").unlink()

    def refuse(**_kwargs):
        raise NotImplementedError

    vector_store.delete = refuse
    report = semantic_index.ingest()

    assert report.deleted_chunks == 0
    assert "could not be deleted" in report.errors[0]


# ------------------------------------------------------- scoped ingestion ----


def test_a_scoped_ingest_only_indexes_its_slice(store, semantic_index):
    write_fact(store)
    store.write(MemoryCategory.EVENTS, "Kickoff.", summary="Kickoff")

    report = semantic_index.ingest(kind=MemoryKind.SEMANTIC)

    assert report.added == 1


def test_a_scoped_ingest_never_prunes_what_it_did_not_look_at(
    store, semantic_index, memory_dir, vector_store
):
    write_fact(store)
    store.write(MemoryCategory.EVENTS, "Kickoff.", summary="Kickoff", when=JUNE)
    semantic_index.ingest()
    (memory_dir / "episodic_memory" / "events" / "2026-06.md").unlink()

    report = semantic_index.ingest(kind=MemoryKind.SEMANTIC)

    assert report.deleted == 0
    assert len(manifest(memory_dir)["entries"]) == 2
    assert len(vector_store.store) == 2


def test_a_category_scoped_ingest_leaves_other_categories_alone(
    store, semantic_index, memory_dir
):
    write_fact(store)
    store.write(MemoryCategory.RULES, "Discount policy.", summary="Discounts")
    semantic_index.ingest()
    (memory_dir / "semantic_memory" / "facts.md").unlink()

    report = semantic_index.ingest(category=MemoryCategory.RULES)

    assert report.deleted == 0
    assert len(manifest(memory_dir)["entries"]) == 2


def test_a_truncated_scan_prunes_nothing(store, embeddings, vector_store):
    index = SemanticIndex(
        embeddings, vector_store, store, config=SemanticConfig(scan_limit=1)
    )
    write_fact(store)
    store.write(MemoryCategory.RULES, "Discount policy.", summary="Discounts")

    report = index.ingest()

    assert report.truncated is True
    assert report.deleted == 0
    assert "run the ingest again" in report.summary()


# ------------------------------------------------------------------ search --


def test_search_returns_the_entry_behind_a_chunk(store, semantic_index):
    hit = write_fact(store)
    semantic_index.ingest()

    (result,) = semantic_index.search("ACME", k=1)

    assert result["entry_id"] == hit.entry.entry_id
    assert result["path"] == hit.path
    assert result["category"] == "facts"
    assert result["rank"] == 1


def test_search_over_an_empty_index_returns_nothing(semantic_index):
    assert semantic_index.search("anything") == []


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"kind": MemoryKind.SEMANTIC}, {"facts", "rules"}),
        ({"category": MemoryCategory.RULES}, {"rules"}),
        ({"tags": ("pricing",)}, {"facts"}),
        ({"source": "user_message"}, {"facts"}),
        ({"min_confidence": Confidence.HIGH}, {"facts"}),
    ],
)
def test_search_filters_on_the_frontmatter(store, semantic_index, filters, expected):
    store.write(
        MemoryCategory.FACTS,
        "ACME is on Enterprise.",
        summary="ACME plan",
        tags=("pricing",),
        source="user_message",
        confidence=Confidence.HIGH,
    )
    store.write(MemoryCategory.RULES, "Discount policy.", summary="Discounts")
    store.write(MemoryCategory.EVENTS, "Kickoff.", summary="Kickoff")
    semantic_index.ingest()

    hits = semantic_index.search("anything", k=10, **filters)

    assert {hit["category"] for hit in hits} == expected


def test_search_filters_on_the_creation_window(store, semantic_index):
    old = write_fact(store, body="Old fact.", when=JANUARY)
    new = write_fact(store, body="New fact.", when=JUNE)
    semantic_index.ingest()

    after = semantic_index.search("fact", k=10, created_after="2026-03-01")
    before = semantic_index.search("fact", k=10, created_before="2026-03-01")

    assert {hit["entry_id"] for hit in after} == {new.entry.entry_id}
    assert {hit["entry_id"] for hit in before} == {old.entry.entry_id}


def test_tags_filter_requires_every_tag(store, semantic_index):
    store.write(MemoryCategory.FACTS, "One tag.", summary="One", tags=("pricing",))
    both = store.write(
        MemoryCategory.FACTS, "Two tags.", summary="Two", tags=("pricing", "acme")
    )
    semantic_index.ingest()

    hits = semantic_index.search("tag", k=10, tags=("pricing", "acme"))

    assert {hit["entry_id"] for hit in hits} == {both.entry.entry_id}


def test_k_caps_the_number_of_hits(store, semantic_index):
    for i in range(5):
        write_fact(store, body=f"Fact {i}.")
    semantic_index.ingest()

    assert len(semantic_index.search("fact", k=2)) == 2


def test_a_filter_builder_pushes_the_filters_to_the_store(
    store, embeddings, vector_store
):
    seen = {}

    def build(active):
        seen.update(active)
        return lambda document: document.metadata["category"] == active.get("category")

    index = SemanticIndex(
        embeddings,
        vector_store,
        store,
        config=SemanticConfig(filter_builder=build),
    )
    write_fact(store)
    store.write(MemoryCategory.RULES, "Discount policy.", summary="Discounts")
    index.ingest()

    hits = index.search("anything", k=10, category=MemoryCategory.RULES)

    assert seen["category"] == "rules"
    assert seen["is_active"] is True
    assert {hit["category"] for hit in hits} == {"rules"}


def test_search_uses_the_configured_chunking(store, embeddings, vector_store):
    index = SemanticIndex(
        embeddings,
        vector_store,
        store,
        config=SemanticConfig(chunking=ChunkingConfig(chunk_size=100)),
    )
    write_fact(store, body="word " * 200)

    report = index.ingest()

    assert report.chunks > 1


# ------------------------------------------------------------------ report --


def test_report_summary_names_the_entries_it_wrote():
    report = IngestReport(added=1, chunks=2, entry_ids=["mem_2026-08-24_9f3a1c"])

    text = report.summary()

    assert "1 new entry" in text
    assert "mem_2026-08-24_9f3a1c" in text


def test_report_summary_reports_errors():
    report = IngestReport(added=1, chunks=1, errors=["mem_x: boom"])

    assert "1 entry(ies) failed: mem_x: boom" in report.summary()
