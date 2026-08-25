"""A write clock the benchmark can move back in time.

`MemoryStore.write` stamps every entry with `datetime.now(tz=UTC)` unless a
caller passes `when=`, and the agent tools do not expose that argument. Replayed
through the real manager agent, a conversation from 2023 would therefore land in
the current month's shard with today's `created` date, which destroys every
temporal question and collapses the sharding the tree is built around.

Freezing the process clock fixes the stamps and breaks everything else: the same
`datetime.now` backs HTTP timeouts and retry budgets inside the model client, and
a signed-request provider would reject calls dated two years ago. So this module
patches `datetime` **only in the three memory modules that read the wall clock**,
with a subclass whose `now()` consults an override.

The override is a `ContextVar`, not a thread-local, because it has to survive two
opposite requirements at once. LangGraph runs tool calls on worker threads, so a
thread-local set around `agent.invoke` would simply not be visible where the
write actually happens — LangChain copies the caller's context into those
executors, so a context variable is. And cases running side by side each get
their own context, so one case's simulated date never bleeds into another's.

`deep_memory_agent.entry` is deliberately left alone: it runs
`isinstance(value, datetime)` when parsing frontmatter, and a patched subclass
there would make every real `datetime` fail that check.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING

from deep_memory_agent import index as _index_module
from deep_memory_agent import layout as _layout_module
from deep_memory_agent import store as _store_module

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["SimulatedDatetime", "install", "simulated_now", "uninstall"]

_moment: ContextVar[datetime | None] = ContextVar("dma_bench_moment", default=None)
_PATCHED_MODULES = (_store_module, _index_module, _layout_module)
_originals: dict[str, type[datetime]] = {}


class SimulatedDatetime(datetime):
    """A `datetime` whose `now()` honours a context-local override."""

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        """Return the simulated instant for this context, or the real one.

        Args:
            tz: Timezone to express the result in, as on `datetime.now`.

        Returns:
            The overridden instant when one is set in the calling context,
            otherwise whatever the real clock says.
        """
        moment = _moment.get()
        if moment is None:
            return datetime.now(tz)
        return moment.astimezone(tz) if tz is not None else moment


def install() -> None:
    """Route the memory modules' `datetime.now` through the override.

    Idempotent, and safe to leave installed: with no override set the patched
    class defers to the real clock.
    """
    for module in _PATCHED_MODULES:
        _originals.setdefault(module.__name__, module.datetime)
        module.datetime = SimulatedDatetime  # type: ignore[misc]


def uninstall() -> None:
    """Put the real `datetime` back in the memory modules."""
    for module in _PATCHED_MODULES:
        original = _originals.pop(module.__name__, None)
        if original is not None:
            module.datetime = original  # type: ignore[misc]


@contextmanager
def simulated_now(moment: datetime | None) -> Iterator[None]:
    """Make memory writes inside the block believe it is `moment`.

    Args:
        moment: Instant to pretend it is, timezone-aware. `None` disables the
            override for the block, which is how a nested call falls back to the
            real clock.

    Yields:
        Nothing; the override is in force for the duration of the block, and for
        anything LangChain runs on a worker thread on its behalf.
    """
    install()
    token = _moment.set(moment)
    try:
        yield
    finally:
        _moment.reset(token)
