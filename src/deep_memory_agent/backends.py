"""Backend wiring for the memory filesystem.

The agents in this package never touch the host filesystem: every read and
write goes through a deepagents `BackendProtocol`, which is what maps the
virtual `/memory/` tree onto real storage.

The default wiring is a `CompositeBackend` that routes `/memory/` to a
`FilesystemBackend` rooted at `memory_dir`, and leaves everything else on an
ephemeral `StateBackend`. Memory therefore lands on disk as plain markdown
files — inspectable, diffable, versionable with git — while the agent's scratch
files stay in thread state and disappear with it.

That default only holds inside a deep agent. `StateBackend` reads and writes
through LangGraph's config keys and raises outside a graph execution, and
`CompositeBackend` fans unscoped `glob`/`grep` calls out to *every* backend —
so a standalone caller such as `consolidate_memory` hits the default even
though all of its paths live under `/memory/`. Pass `for_deep_agent=False` to
swap the default for an empty scratch directory instead.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from deep_memory_agent.layout import MEMORY_ROOT

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

__all__ = ["build_memory_backend", "resolve_backend"]

_MAX_MEMORY_FILE_MB = 10


@lru_cache(maxsize=1)
def _scratch_root() -> Path:
    """Return a process-wide empty directory standing in for thread state.

    Nothing is meant to land here: it exists so that the composite backend has
    a default it can query without a LangGraph context. One directory per
    process is enough, and it is removed when the process exits.
    """
    root = Path(tempfile.mkdtemp(prefix="deep-memory-agent-scratch-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


def build_memory_backend(
    memory_dir: str | Path,
    *,
    for_deep_agent: bool = True,
) -> CompositeBackend:
    """Build the default backend for a memory directory.

    The directory is created if it does not exist. Paths under `/memory/` are
    served from it; where anything else goes depends on `for_deep_agent`.

    Args:
        memory_dir: Host directory that stores the memory tree.
        for_deep_agent: Whether the backend will be driven by a deep agent.
            When `True`, non-memory paths stay in ephemeral thread state, which
            requires a LangGraph execution context. Set it to `False` for
            standalone use — `consolidate_memory`, `MemoryStore`, a cron job —
            and they are served from an empty scratch directory instead.

    Returns:
        A composite backend with `/memory/` routed to disk.
    """
    root = Path(memory_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    default: BackendProtocol = (
        StateBackend()
        if for_deep_agent
        else FilesystemBackend(root_dir=_scratch_root(), virtual_mode=True)
    )
    return CompositeBackend(
        default=default,
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
