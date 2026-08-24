from deep_memory_agent.layout import MemoryKind, index_path
from deep_memory_agent.scaffold import ensure_memory_tree

EXPECTED_FILES = {
    "/memory/index.md",
    "/memory/preferences.md",
    "/memory/episodic_memory/index.md",
    "/memory/semantic_memory/index.md",
    "/memory/procedural_memory/index.md",
    "/memory/semantic_memory/facts.md",
    "/memory/semantic_memory/rules.md",
}


def test_ensure_memory_tree_creates_the_whole_tree(backend, memory_dir):
    created = ensure_memory_tree(backend)

    assert set(created) == EXPECTED_FILES
    assert (memory_dir / "semantic_memory" / "facts.md").exists()


def test_ensure_memory_tree_is_idempotent(backend):
    ensure_memory_tree(backend)

    assert ensure_memory_tree(backend) == []


def test_ensure_memory_tree_never_overwrites(backend):
    ensure_memory_tree(backend)
    backend.write("/memory/semantic_memory/facts.md", "# Facts\n\nhand written\n")

    ensure_memory_tree(backend)

    content = backend.read("/memory/semantic_memory/facts.md").file_data["content"]
    assert "hand written" in content


def test_episodic_shards_are_not_pre_created(backend, memory_dir):
    ensure_memory_tree(backend)

    assert not (memory_dir / "episodic_memory" / "events").exists()


def test_kind_indexes_are_seeded(backend):
    ensure_memory_tree(backend)

    for kind in MemoryKind:
        result = backend.read(index_path(kind))
        assert result.error is None
