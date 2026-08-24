"""Backend wiring for the memory filesystem.

The agents in this package never touch the host filesystem: every read and
write goes through a deepagents `BackendProtocol`, which is what maps the
virtual `/memory/` tree onto real storage.

The default wiring is a `CompositeBackend` that routes `/memory/` to a
`FilesystemBackend` rooted at `memory_dir`, and leaves everything else on an
ephemeral `StateBackend`. Memory therefore lands on disk as plain markdown
files — inspectable, diffable, versionable with git — while the agent's scratch
files stay in thread state and disappear with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from deep_memory_agent.layout import MEMORY_ROOT

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

__all__ = ["build_memory_backend", "resolve_backend"]

_MAX_MEMORY_FILE_MB = 10


def build_memory_backend(memory_dir: str | Path) -> CompositeBackend:
    """Build the default backend for a memory directory.

    The directory is created if it does not exist. Paths under `/memory/` are
    served from it; anything else stays in ephemeral thread state.

    Args:
        memory_dir: Host directory that stores the memory tree.

    Returns:
        A composite backend with `/memory/` routed to disk.
    """
    root = Path(memory_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return CompositeBackend(
        default=StateBackend(),
        routes={
            MEMORY_ROOT: FilesystemBackend(
                root_dir=root,
                virtual_mode=True,
                max_file_size_mb=_MAX_MEMORY_FILE_MB,
            )
        },
    )


def resolve_backend(
    memory_dir: str | Path | None = None,
    backend: BackendProtocol | None = None,
) -> BackendProtocol:
    """Resolve the backend an agent factory should use.

    Exactly one of `memory_dir` and `backend` must be given: `memory_dir` for
    the default on-disk wiring, `backend` to plug in your own storage. Passing
    both would leave it ambiguous which one actually serves `/memory/`, so it is
    rejected rather than silently resolved.

    Args:
        memory_dir: Host directory to build the default backend from.
        backend: A ready-made backend serving the `/memory/` tree.

    Returns:
        The backend to hand to the agent.

    Raises:
        ValueError: If both arguments are given, or neither is.
    """
    if memory_dir is not None and backend is not None:
        msg = "pass either memory_dir or backend, not both"
        raise ValueError(msg)
    if memory_dir is None and backend is None:
        msg = "one of memory_dir or backend is required"
        raise ValueError(msg)
    if backend is not None:
        return backend
    return build_memory_backend(memory_dir)  # type: ignore[arg-type]
