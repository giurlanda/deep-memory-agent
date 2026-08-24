import pytest
from deepagents.backends import CompositeBackend

from deep_memory_agent.backends import build_memory_backend, resolve_backend
from deep_memory_agent.layout import MEMORY_ROOT


def test_build_memory_backend_routes_memory_to_disk(tmp_path):
    backend = build_memory_backend(tmp_path / "memory")

    backend.write(f"{MEMORY_ROOT}facts.md", "hello")

    assert isinstance(backend, CompositeBackend)
    assert (tmp_path / "memory" / "facts.md").read_text() == "hello"


def test_build_memory_backend_creates_a_missing_directory(tmp_path):
    target = tmp_path / "nested" / "memory"

    build_memory_backend(target)

    assert target.is_dir()


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
