from datetime import UTC, datetime

from deep_memory_agent.layout import MemoryCategory
from deep_memory_agent.tools import build_recall_tools, build_write_tools

AUGUST = datetime(2026, 8, 24, tzinfo=UTC)


def tools_by_name(tools):
    return {tool.name: tool for tool in tools}


def test_recall_tools_are_read_only(store):
    names = set(tools_by_name(build_recall_tools(store)))

    assert names == {"memory_index", "memory_search", "memory_read"}


def test_write_tools_need_a_model_for_consolidation(store):
    without = set(tools_by_name(build_write_tools(store)))
    with_model = set(
        tools_by_name(build_write_tools(store, consolidation_model="fake"))
    )

    assert without == {"memory_write", "memory_update"}
    assert with_model == {"memory_write", "memory_update", "memory_consolidate"}


def test_memory_write_stores_an_entry(store):
    write = tools_by_name(build_write_tools(store))["memory_write"]

    result = write.invoke(
        {
            "category": "facts",
            "content": "ACME is on Enterprise.",
            "summary": "ACME plan",
            "tags": ["acme"],
            "confidence": "high",
        }
    )

    assert "Stored mem_" in result
    (hit,) = store.search("ACME")
    assert hit.entry.summary == "ACME plan"
    assert hit.entry.confidence.value == "high"


def test_memory_write_rejects_an_unknown_category(store):
    write = tools_by_name(build_write_tools(store))["memory_write"]

    result = write.invoke({"category": "nonsense", "content": "x", "summary": "x"})

    assert result.startswith("Error:")


def test_memory_write_rejects_a_procedure_without_a_title(store):
    write = tools_by_name(build_write_tools(store))["memory_write"]

    result = write.invoke({"category": "procedure", "content": "x", "summary": "x"})

    assert "requires a title" in result


def test_memory_update_supersedes_the_previous_entry(store):
    tools = tools_by_name(build_write_tools(store))
    old = store.write(
        MemoryCategory.FACTS, "ACME is on Team.", summary="ACME plan", when=AUGUST
    )

    result = tools["memory_update"].invoke(
        {"entry_id": old.entry.entry_id, "content": "ACME is on Enterprise."}
    )

    assert f"superseding {old.entry.entry_id}" in result
    assert [hit.entry.body for hit in store.search("ACME")] == [
        "ACME is on Enterprise."
    ]


def test_memory_update_inherits_summary_and_tags(store):
    tools = tools_by_name(build_write_tools(store))
    old = store.write(
        MemoryCategory.FACTS, "Old.", summary="ACME plan", tags=("acme",), when=AUGUST
    )

    tools["memory_update"].invoke({"entry_id": old.entry.entry_id, "content": "New."})

    (hit,) = store.search("New.")
    assert hit.entry.summary == "ACME plan"
    assert hit.entry.tags == ("acme",)


def test_memory_update_reports_an_unknown_id(store):
    tools = tools_by_name(build_write_tools(store))

    result = tools["memory_update"].invoke(
        {"entry_id": "mem_1970-01-01_000000", "content": "x"}
    )

    assert result.startswith("Error:")


def test_memory_search_reports_ids_paths_and_provenance(store):
    store.write(
        MemoryCategory.FACTS,
        "ACME is on Enterprise.",
        summary="ACME plan",
        tags=("acme",),
    )
    search = tools_by_name(build_recall_tools(store))["memory_search"]

    result = search.invoke({"query": "acme"})

    assert "/memory/semantic_memory/facts.md" in result
    assert "confidence: medium" in result
    assert "ACME is on Enterprise." in result


def test_memory_search_says_so_when_nothing_matches(store):
    search = tools_by_name(build_recall_tools(store))["memory_search"]

    assert "No memory entry matched" in search.invoke({"query": "nothing"})


def test_memory_search_hides_superseded_entries_by_default(store):
    old = store.write(MemoryCategory.FACTS, "ACME is on Team.", when=AUGUST)
    store.write(
        MemoryCategory.FACTS, "ACME is on Enterprise.", supersedes=old.entry.entry_id
    )
    search = tools_by_name(build_recall_tools(store))["memory_search"]

    default = search.invoke({"query": "ACME"})
    history = search.invoke({"query": "ACME", "include_superseded": True})

    assert "ACME is on Team." not in default
    assert "SUPERSEDED BY" in history


def test_memory_index_lists_files_not_content(store):
    store.write(MemoryCategory.FACTS, "A very distinctive body.", summary="ACME plan")
    index = tools_by_name(build_recall_tools(store))["memory_index"]

    result = index.invoke({"kind": "semantic"})

    assert "facts.md" in result
    assert "A very distinctive body." not in result


def test_memory_index_rejects_an_unknown_kind(store):
    index = tools_by_name(build_recall_tools(store))["memory_index"]

    assert index.invoke({"kind": "nonsense"}).startswith("Error:")


def test_memory_read_refuses_paths_outside_the_tree(store):
    read = tools_by_name(build_recall_tools(store))["memory_read"]

    assert read.invoke({"path": "/etc/passwd"}).startswith("Error:")
