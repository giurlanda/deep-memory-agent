import pytest
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from deep_memory_agent.backends import build_memory_backend, resolve_backend
from deep_memory_agent.consolidation import consolidate_memory
from deep_memory_agent.layout import MEMORY_ROOT
from deep_memory_agent.store import MemoryStore


def test_build_memory_backend_routes_memory_to_disk(tmp_path):
    backend = build_memory_backend(tmp_path / "memory")

    backend.write(f"{MEMORY_ROOT}facts.md", "hello")

    assert isinstance(backend, CompositeBackend)
    assert (tmp_path / "memory" / "facts.md").read_text() == "hello"


def test_build_memory_backend_creates_a_missing_directory(tmp_path):
    target = tmp_path / "nested" / "memory"

    build_memory_backend(target)

    assert target.is_dir()


def test_build_memory_backend_defaults_to_thread_state(tmp_path):
    backend = build_memory_backend(tmp_path / "memory")

    assert isinstance(backend.default, StateBackend)


def test_build_memory_backend_standalone_defaults_to_a_scratch_directory(tmp_path):
    backend = build_memory_backend(tmp_path / "memory", for_deep_agent=False)

    assert isinstance(backend.default, FilesystemBackend)
    assert not any(backend.default.cwd.iterdir())


def test_build_memory_backend_standalone_shares_one_scratch_directory(tmp_path):
    first = build_memory_backend(tmp_path / "one", for_deep_agent=False)
    second = build_memory_backend(tmp_path / "two", for_deep_agent=False)

    assert first.default.cwd == second.default.cwd


def test_standalone_backend_still_routes_memory_to_disk(tmp_path):
    backend = build_memory_backend(tmp_path / "memory", for_deep_agent=False)

    backend.write(f"{MEMORY_ROOT}facts.md", "hello")

    assert (tmp_path / "memory" / "facts.md").read_text() == "hello"


def test_standalone_backend_globs_without_a_graph_context(tmp_path):
    """An unscoped glob reaches the default backend, so it must not need LangGraph."""
    store = MemoryStore(build_memory_backend(tmp_path / "memory", for_deep_agent=False))
    store.ensure_tree()

    assert store.search() == []


def test_consolidate_memory_runs_on_a_standalone_backend(tmp_path, consolidation_model):
    backend = build_memory_backend(tmp_path / "memory", for_deep_agent=False)

    result = consolidate_memory(backend, consolidation_model())

    assert result.rationale == "no episodes to consolidate"


def test_consolidate_memory_rejects_a_deep_agent_backend(tmp_path, consolidation_model):
    backend = build_memory_backend(tmp_path / "memory")

    with pytest.raises(RuntimeError, match="LangGraph"):
        consolidate_memory(backend, consolidation_model())


def test_resolve_backend_builds_from_memory_dir(tmp_path):
    assert isinstance(resolve_backend(memory_dir=tmp_path), CompositeBackend)


def test_resolve_backend_passes_an_explicit_backend_through(backend):
    assert resolve_backend(backend=backend) is backend


def test_resolve_backend_rejects_both_arguments(tmp_path, backend):
    with pytest.raises(ValueError, match="not both"):
        resolve_backend(memory_dir=tmp_path, backend=backend)


def test_resolve_backend_rejects_neither_argument():
    with pytest.raises(ValueError, match="one of memory_dir or backend"):
        resolve_backend()
