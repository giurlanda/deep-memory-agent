"""Where model-call records go, and which stage of a case they belong to.

`dma_bench.llm` records every call it makes; this module decides where those
records land. A case opens a `record_calls` block and names each stage with
`call_stage`, and any call made inside — however deep in an agent loop — is
appended to that case's log with that stage attached.

Both are context variables, for the same reason the write clock is
(`dma_bench.clock`): LangGraph runs nodes and tools on worker threads, LangChain
copies the caller's context into those executors, and cases running side by side
each get a context of their own. A thread-local would be invisible where the
call happens; a global would mix the cases up.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dma_bench.schema import CallRecord

__all__ = ["CallLog", "call_stage", "record_call", "record_calls"]


class CallLog:
    """The calls made inside one `record_calls` block, safe to append from threads."""

    def __init__(self) -> None:
        """Start an empty log."""
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []

    def append(self, record: CallRecord) -> None:
        """Add one call."""
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> list[CallRecord]:
        """The calls recorded so far, in the order they finished."""
        with self._lock:
            return list(self._records)


_log: ContextVar[CallLog | None] = ContextVar("dma_bench_call_log", default=None)
_stage: ContextVar[str] = ContextVar("dma_bench_call_stage", default="")


@contextmanager
def record_calls() -> Iterator[CallLog]:
    """Collect every call made inside the block.

    Yields:
        The log the calls are appended to.
    """
    log = CallLog()
    token = _log.set(log)
    try:
        yield log
    finally:
        _log.reset(token)


@contextmanager
def call_stage(name: str) -> Iterator[None]:
    """Attribute the calls made inside the block to `name`.

    Args:
        name: The stage, e.g. `ingestion` or `judge:lexical`. Nested blocks
            override the enclosing one for their duration.

    Yields:
        Nothing.
    """
    token = _stage.set(name)
    try:
        yield
    finally:
        _stage.reset(token)


def record_call(record: CallRecord) -> None:
    """Add a call to the enclosing `record_calls` block, if there is one.

    Args:
        record: The call. Its stage is filled in from the enclosing
            `call_stage` when it has none of its own.
    """
    log = _log.get()
    if log is None:
        return
    if not record.stage:
        record = record.model_copy(update={"stage": _stage.get()})
    log.append(record)
