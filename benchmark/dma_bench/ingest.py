"""Replaying a history into memory, one session at a time.

This is the benchmark's counterpart to LongMemEval's evidence-session
construction, and the reason the two measure different things. LongMemEval
builds a chat log and hands it to a retriever; here each session is handed to the
real manager agent, which decides on its own what is worth recording, which
category it belongs in, and whether it retires something already on file. What is
being scored is therefore the whole pipeline — extraction, routing, supersession,
consolidation — not a retriever over pre-loaded facts.

The session is presented whole, as a structured transcript rather than as a
statement of fact, so that the extraction step is actually exercised. Writes are
stamped with the session's own date via `dma_bench.clock`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

from dma_bench.clock import simulated_now
from dma_bench.schema import ConsolidationRecord, IngestionRecord

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from dma_bench.schema import Case, ConsolidationMode, Session

__all__ = ["INGESTION_PROMPT", "consolidate", "ingest_case", "render_session"]

INGESTION_PROMPT = """\
The conversation below already happened, on {date}. Read it as a record of a
past working session and update your memory from it.

Record what is worth remembering and nothing else: a session with nothing
durable in it is a normal outcome. Search before you write, and when the
session changes something you already hold, retire the old entry with
`memory_update` rather than adding a second, conflicting one.

Write as if it were {date}: that is when this happened.

{transcript}
"""

_WRITE_TOOLS = frozenset({"memory_write", "memory_update"})
_CONSOLIDATE_TOOL = "memory_consolidate"


def render_session(session: Session) -> str:
    """Render a session as a transcript the agent can read.

    Args:
        session: The session to render.

    Returns:
        The transcript, one labelled block per turn.
    """
    lines = [f"## Session {session.session_id} — {session.date:%Y-%m-%d %H:%M}"]
    lines += [f"[{turn.role}] {turn.content}" for turn in session.turns]
    return "\n\n".join(lines)


def consolidate(
    memory_dir: Path,
    model: BaseChatModel,
    *,
    after_session: int,
    moment: datetime | None = None,
) -> ConsolidationRecord:
    """Run one consolidation pass over a case's memory.

    Driven from here rather than left to the agent so that both arms of the
    ablation differ only in what the harness did, and by exactly how much.

    Args:
        memory_dir: Directory holding the memory tree.
        model: Model used to judge what has hardened.
        after_session: Index of the session this ran after; `-1` for the final
            pass.
        moment: Instant to stamp the durable entries with.

    Returns:
        What the pass read and wrote.
    """
    from deep_memory_agent import build_memory_backend, consolidate_memory

    try:
        with simulated_now(moment):
            result = consolidate_memory(
                build_memory_backend(memory_dir, for_deep_agent=False), model
            )
    except Exception as exc:
        return ConsolidationRecord(after_session=after_session, error=repr(exc))
    return ConsolidationRecord(
        after_session=after_session,
        episodes_considered=result.episodes_considered,
        entries_written=len(result.entries),
        rationale=result.rationale,
    )


def ingest_case(
    agent: CompiledStateGraph,
    case: Case,
    *,
    memory_dir: Path,
    model: BaseChatModel,
    consolidation_mode: ConsolidationMode,
    consolidate_every_n: int = 10,
    recursion_limit: int = 60,
) -> IngestionRecord:
    """Replay a case's whole history into its memory tree.

    Args:
        agent: The manager agent to drive.
        case: The case whose sessions are replayed.
        memory_dir: Directory holding the memory tree.
        model: Model used by the scheduled consolidation passes.
        consolidation_mode: When consolidation runs.
        consolidate_every_n: Sessions between periodic passes.
        recursion_limit: Cap on agent steps per session.

    Returns:
        What the replay did, including any session that failed. A failed session
        is skipped rather than aborting the case: losing one episode is a
        measurable degradation, losing the case is a hole in the results.
    """
    started = time.monotonic()
    record = IngestionRecord()
    periodic = "periodic" in consolidation_mode

    for index, session in enumerate(case.sessions):
        prompt = INGESTION_PROMPT.format(
            date=f"{session.date:%Y-%m-%d}", transcript=render_session(session)
        )
        try:
            with simulated_now(session.date):
                state = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"recursion_limit": recursion_limit},
                )
        except Exception as exc:
            record.failed_sessions.append(f"{session.session_id}: {exc!r}")
            continue

        record.sessions += 1
        writes, consolidations = _count_tool_calls(state.get("messages", []))
        record.write_calls += writes
        record.unsolicited_consolidations += consolidations

        if periodic and (index + 1) % consolidate_every_n == 0:
            record.consolidations.append(
                consolidate(memory_dir, model, after_session=index, moment=session.date)
            )

    if "final" in consolidation_mode:
        record.consolidations.append(
            consolidate(memory_dir, model, after_session=-1, moment=case.question_date)
        )

    record.duration_s = round(time.monotonic() - started, 2)
    return record


def _count_tool_calls(messages: list) -> tuple[int, int]:
    """Return how many write and consolidate calls a session produced."""
    writes = consolidations = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or ():
            name = call.get("name", "")
            writes += name in _WRITE_TOOLS
            consolidations += name == _CONSOLIDATE_TOOL
    return writes, consolidations
