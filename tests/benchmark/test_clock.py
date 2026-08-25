import asyncio
import threading
from datetime import UTC, datetime

from deep_memory_agent import store as store_module
from deep_memory_agent.layout import MemoryCategory
from dma_bench import clock


def test_writes_land_in_the_shard_of_the_simulated_month(store):
    with clock.simulated_now(datetime(2023, 4, 10, 12, 0, tzinfo=UTC)):
        hit = store.write(MemoryCategory.EVENTS, "the migration cutover ran")

    assert hit.path == "/memory/episodic_memory/events/2023-04.md"
    assert hit.entry.created.year == 2023


def test_the_index_row_carries_the_simulated_date(store, memory_dir):
    with clock.simulated_now(datetime(2023, 4, 10, 12, 0, tzinfo=UTC)):
        store.write(MemoryCategory.EVENTS, "body", summary="a cutover")

    index = (memory_dir / "episodic_memory" / "index.md").read_text()
    assert "2023-04-10" in index


def test_the_real_clock_is_restored_outside_the_block():
    with clock.simulated_now(datetime(2023, 4, 10, tzinfo=UTC)):
        pass

    assert store_module.datetime.now(tz=UTC).year >= 2024


def test_the_override_does_not_leak_between_threads():
    seen: dict[str, int] = {}

    def record(name: str, year: int) -> None:
        with clock.simulated_now(datetime(year, 1, 5, tzinfo=UTC)):
            barrier.wait(timeout=5)
            seen[name] = store_module.datetime.now(tz=UTC).year

    barrier = threading.Barrier(2)
    threads = [
        threading.Thread(target=record, args=("a", 2021)),
        threading.Thread(target=record, args=("b", 2023)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen == {"a": 2021, "b": 2023}


def test_the_override_reaches_langchain_worker_threads(store):
    # The whole design rests on this: LangGraph runs tool calls off the calling
    # thread, so a thread-local override would never be seen where the write
    # actually happens.
    from langchain_core.runnables.config import run_in_executor

    from deep_memory_agent.layout import MemoryCategory

    async def write() -> str:
        return await run_in_executor(
            None, store.write, MemoryCategory.EVENTS, "written off-thread"
        )

    with clock.simulated_now(datetime(2023, 4, 10, tzinfo=UTC)):
        hit = asyncio.run(write())

    assert hit.path == "/memory/episodic_memory/events/2023-04.md"
