import pytest

from deep_memory_agent.layout import MemoryCategory, MemoryKind
from deep_memory_agent.semantic_index import SemanticConfig
from deep_memory_agent.semantic_tools import (
    SEMANTIC_INGEST_TOOL_NAME,
    SEMANTIC_SEARCH_TOOL_NAME,
    create_semantic_tools,
    ingest_semantic_index,
)


@pytest.fixture
def semantic(embeddings, vector_store, store):
    return create_semantic_tools(embeddings, vector_store, store)


def call(tool, **args):
    """Invoke a tool the way an agent does, so the artifact comes back too."""
    message = tool.invoke(
        {"type": "tool_call", "name": tool.name, "args": args, "id": "call-1"}
    )
    return message.content, message.artifact


def write_fact(store, body="ACME is on the Enterprise plan.", **kwargs):
    return store.write(MemoryCategory.FACTS, body, summary="ACME plan", **kwargs)


# ------------------------------------------------------------- the factory --


def test_it_builds_the_two_tools_under_the_expected_names(semantic):
    assert semantic.ingest_tool.name == SEMANTIC_INGEST_TOOL_NAME
    assert semantic.search_tool.name == SEMANTIC_SEARCH_TOOL_NAME
    assert semantic.as_list() == [semantic.ingest_tool, semantic.search_tool]


def test_it_exposes_the_index_and_its_functions(semantic):
    assert semantic.index is not None
    assert semantic.ingest == semantic.index.ingest
    assert semantic.search == semantic.index.search


@pytest.mark.parametrize("search_k", [0, -1, 26])
def test_it_rejects_an_out_of_range_search_k(embeddings, vector_store, store, search_k):
    with pytest.raises(ValueError, match="search_k"):
        create_semantic_tools(embeddings, vector_store, store, search_k=search_k)


def test_search_k_reaches_the_schema_the_model_sees(embeddings, vector_store, store):
    semantic = create_semantic_tools(embeddings, vector_store, store, search_k=3)

    field = semantic.search_tool.args_schema.model_fields["k"]

    # The default has to live in the schema: a model that omits `k` would
    # otherwise silently get the pydantic default instead of this index's.
    assert field.default == 3


# ------------------------------------------------------------- ingest tool --


def test_the_ingest_tool_reports_what_it_did(store, semantic):
    write_fact(store)

    content, artifact = call(semantic.ingest_tool)

    assert "1 new entry" in content
    assert artifact["added"] == 1
    assert artifact["chunks"] == 1


def test_the_ingest_tool_says_when_nothing_changed(store, semantic):
    write_fact(store)
    call(semantic.ingest_tool)

    content, artifact = call(semantic.ingest_tool)

    assert content == "No update needed: 1 entries already match the index."
    assert artifact["unchanged"] == 1


def test_the_ingest_tool_accepts_a_scope(store, semantic):
    write_fact(store)
    store.write(MemoryCategory.EVENTS, "Kickoff.", summary="Kickoff")

    _, artifact = call(semantic.ingest_tool, kind="semantic")

    assert artifact["added"] == 1


def test_the_ingest_tool_rejects_an_unknown_scope(semantic):
    content, artifact = call(semantic.ingest_tool, kind="nonsense")

    assert content.startswith("Error:")
    assert artifact == {}


# ------------------------------------------------------------- search tool --


def test_the_search_tool_renders_the_hits_and_carries_them_as_an_artifact(
    store, semantic
):
    hit = write_fact(store)
    call(semantic.ingest_tool)

    content, artifact = call(semantic.search_tool, query="ACME")

    assert hit.entry.entry_id in content
    assert hit.path in content
    assert artifact[0]["entry_id"] == hit.entry.entry_id


def test_the_search_tool_says_so_when_nothing_matches(semantic):
    content, artifact = call(semantic.search_tool, query="anything")

    assert "No entry matched" in content
    assert "memory_search" in content
    assert artifact == []


def test_the_search_tool_marks_a_superseded_hit(store, semantic):
    first = write_fact(store)
    write_fact(store, body="ACME moved to Team.", supersedes=first.entry.entry_id)
    call(semantic.ingest_tool)

    content, _ = call(semantic.search_tool, query="ACME", include_superseded=True)

    assert "SUPERSEDED" in content


def test_the_search_tool_hides_superseded_entries_by_default(store, semantic):
    first = write_fact(store)
    write_fact(store, body="ACME moved to Team.", supersedes=first.entry.entry_id)
    call(semantic.ingest_tool)

    content, _ = call(semantic.search_tool, query="ACME")

    assert first.entry.entry_id not in content


def test_the_search_tool_trims_long_excerpts(embeddings, vector_store, store):
    semantic = create_semantic_tools(
        embeddings, vector_store, store, config=SemanticConfig(snippet_chars=50)
    )
    write_fact(store, body="word " * 100)
    semantic.index.ingest()

    content, artifact = call(semantic.search_tool, query="word")

    assert "[…]" in content
    assert len(artifact[0]["text"]) > 50


def test_the_search_tool_filters_by_category(store, semantic):
    write_fact(store)
    rule = store.write(MemoryCategory.RULES, "Discount policy.", summary="Discounts")
    call(semantic.ingest_tool)

    _, artifact = call(semantic.search_tool, query="anything", category="rules")

    assert {hit["entry_id"] for hit in artifact} == {rule.entry.entry_id}


# --------------------------------------------------- ingest without an agent --


def test_ingest_semantic_index_runs_off_a_memory_dir(
    embeddings, vector_store, memory_dir, store
):
    hit = write_fact(store)

    report = ingest_semantic_index(embeddings, vector_store, memory_dir=memory_dir)

    assert report.added == 1
    assert (memory_dir / ".semantic-manifest.json").exists()
    assert hit.entry.entry_id in report.entry_ids


def test_ingest_semantic_index_runs_off_a_backend(
    embeddings, vector_store, backend, store
):
    write_fact(store)

    report = ingest_semantic_index(embeddings, vector_store, backend=backend)

    assert report.added == 1


def test_ingest_semantic_index_accepts_a_scope(
    embeddings, vector_store, backend, store
):
    write_fact(store)
    store.write(MemoryCategory.EVENTS, "Kickoff.", summary="Kickoff")

    report = ingest_semantic_index(
        embeddings, vector_store, backend=backend, kind=MemoryKind.SEMANTIC
    )

    assert report.added == 1


def test_ingest_semantic_index_rejects_both_memory_dir_and_backend(
    embeddings, vector_store, memory_dir, backend
):
    with pytest.raises(ValueError, match="not both"):
        ingest_semantic_index(
            embeddings, vector_store, memory_dir=memory_dir, backend=backend
        )


def test_ingest_semantic_index_requires_one_of_them(embeddings, vector_store):
    with pytest.raises(ValueError, match="one of memory_dir or backend"):
        ingest_semantic_index(embeddings, vector_store)


def test_ingest_semantic_index_scaffolds_an_empty_tree(
    embeddings, vector_store, tmp_path
):
    fresh = tmp_path / "fresh"

    report = ingest_semantic_index(embeddings, vector_store, memory_dir=fresh)

    assert report.added == 0
    assert (fresh / "index.md").exists()
