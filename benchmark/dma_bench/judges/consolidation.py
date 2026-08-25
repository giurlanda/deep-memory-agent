"""Scoring the stage LongMemEval does not have.

Before anything can be retrieved it has to have been written, and in this
architecture writing is a model's judgement call: the manager agent decides what
in a session was worth keeping, and consolidation decides what has hardened into
a durable fact. Both can quietly drop the one detail a question turns on. Without
a check at this stage, that failure is indistinguishable from a retrieval failure
— and the two live in different prompts.

So the judge is given the gold messages and the memory tree as it stands after
ingestion, and asked one question: is the information needed to answer actually
in there, and is it right? For an abstention case the correct state is the
opposite — memory must *not* assert something it was never told, and an entry
that invented it is a consolidation failure even if the agent later abstains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from dma_bench.judges.base import JudgeError, ask_judge, render_block
from dma_bench.schema import MemorySnapshot, StageVerdict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from deep_memory_agent import MemoryStore
    from dma_bench.schema import Case

__all__ = [
    "CONSOLIDATION_PROMPT",
    "judge_consolidation",
    "render_memory",
    "snapshot_memory",
]

CONSOLIDATION_PROMPT = """\
You are auditing what an agent wrote into its own memory after being shown a
series of past conversations.

You will be given the source messages that carried the information, and the
memory entries the agent produced from them. Judge only what is on file — not
whether the agent could find it later.

Answer `correct: true` if the information needed to answer the question is
present in the memory entries and is factually right. Wording may differ freely;
what matters is that the substance is there and not distorted. Where the
information changed over time, the entry holding the current value must be the
active one.

Answer `correct: false` if the information is absent, garbled, or if a stale
value is still on file as active.

When the source messages are empty, the question is one the agent was never told
the answer to. In that case answer `correct: true` when memory holds no entry
asserting an answer, and `correct: false` when some entry invented one.
"""

_MAX_ENTRIES = 250
_BODY_CHARS = 400


class _Audit(BaseModel):
    """Structured form of the consolidation audit."""

    correct: bool = Field(description="Whether memory holds the right information.")
    reasoning: str = Field(default="", description="Short justification.")


def snapshot_memory(store: MemoryStore) -> MemorySnapshot:
    """Summarise the state of a memory tree after ingestion.

    Args:
        store: Store bound to the case's memory tree.

    Returns:
        Counts, files and size. Cheap enough to record for every case, and it is
        what makes "the tree grew but the answers got worse" visible.
    """
    active = store.search(limit=100_000)
    everything = store.search(limit=100_000, include_superseded=True)
    counts: dict[str, int] = {}
    for hit in active:
        key = hit.entry.category.value
        counts[key] = counts.get(key, 0) + 1
    return MemorySnapshot(
        entries_by_category=dict(sorted(counts.items())),
        active_entries=len(active),
        superseded_entries=len(everything) - len(active),
        files=sorted({hit.path for hit in everything}),
        total_chars=sum(len(hit.entry.body) for hit in everything),
    )


def render_memory(store: MemoryStore, *, limit: int = _MAX_ENTRIES) -> str:
    """Render the active memory entries for the judge.

    Args:
        store: Store bound to the case's memory tree.
        limit: Maximum entries to include, newest first.

    Returns:
        One block per entry, with its id, category, date and body excerpt.
    """
    blocks = []
    for hit in store.search(limit=limit):
        entry = hit.entry
        body = entry.body.strip()
        excerpt = body[:_BODY_CHARS] + ("…" if len(body) > _BODY_CHARS else "")
        blocks.append(
            f"[{entry.entry_id}] {entry.category.value} | {entry.created:%Y-%m-%d} "
            f"| confidence: {entry.confidence.value}\n"
            f"summary: {entry.summary or '-'}\n{excerpt}"
        )
    return "\n\n".join(blocks)


def judge_consolidation(
    model: BaseChatModel, case: Case, store: MemoryStore
) -> StageVerdict:
    """Judge whether memory holds what the question needs.

    Args:
        model: The grading model.
        case: The case being audited.
        store: Store bound to the case's memory tree.

    Returns:
        The verdict on stage one.
    """
    gold = "\n\n".join(f"({turn.role}) {turn.content}" for turn in case.gold_turns)
    payload = (
        render_block("Question", case.question)
        + render_block("Reference answer", case.answer)
        + render_block("Source messages that carried the information", gold, limit=8000)
        + render_block("Memory entries on file", render_memory(store), limit=24000)
    )
    try:
        audit = ask_judge(model, _Audit, CONSOLIDATION_PROMPT, payload)
    except JudgeError as exc:
        return StageVerdict(error=str(exc))
    return StageVerdict(correct=audit.correct, reasoning=audit.reasoning)
