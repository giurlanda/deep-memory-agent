"""Scoring retrieval separately from the answer.

LongMemEval computes Recall@k over one retriever's ranked list. There is no such
list here: the agent chooses how many times to search, with which wordings, and
which files to open, so the retrieved set is whatever its trace shows it actually
brought into view. A judge reads that trace against the dataset's gold turns —
the messages flagged `has_answer` — and reports which of them were surfaced.

Recall is over messages, not sessions, because that is the granularity the gold
annotation has. The strict criterion of the paper is extended by one clause: for
a current-state question, retrieval is correct only if the active fact is
surfaced **and** the superseded one is not presented as still valid. Surfacing it
marked as history is fine, and is in fact what a working bi-temporal graph should
do.

Abstention cases have no gold turn at all. Scoring them zero would be a lie about
a system that correctly found nothing, so they come back `applicable=False` and
are left out of the averages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from dma_bench.categories import STRICT_RETRIEVAL_CATEGORIES
from dma_bench.judges.base import JudgeError, ask_judge, render_block
from dma_bench.schema import RetrievalVerdict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from dma_bench.schema import Case, TraceMessage

__all__ = ["RETRIEVAL_PROMPT", "judge_retrieval", "render_trace"]

RETRIEVAL_PROMPT = """\
You are auditing how an agent searched its own memory.

You will be given the agent's search trace — its reasoning turns and the results
its memory tools returned — and a numbered list of source messages that carry
the information needed to answer the question. The source messages were written
into memory earlier, so the agent will not see them word for word: it will see
whatever it chose to record about them. Count a source message as surfaced when
the trace clearly brings its information into view, in any wording.

Report:
- `retrieved_ids`: the ids of the source messages the trace surfaced. Be strict:
  a search that returned nothing, or returned something merely on the same
  topic, has not surfaced the message.
- `superseded_present`: whether the trace surfaced the superseded fact **as if
  it were still valid**. Surfacing it while marking it as history, replaced, or
  no longer current is not a leak — answer false in that case, and false also
  when no superseded fact was given.
- `reasoning`: one or two sentences on what the trace did and did not find.
"""


class _RetrievalAudit(BaseModel):
    """Structured form of the retrieval audit."""

    retrieved_ids: list[str] = Field(
        default_factory=list, description="Ids of the source messages surfaced."
    )
    superseded_present: bool = Field(
        default=False,
        description="Whether a retired fact was surfaced as if still valid.",
    )
    reasoning: str = Field(default="", description="Short justification.")


def render_trace(trace: list[TraceMessage]) -> str:
    """Render an agent's search trace for the judge.

    Args:
        trace: The AI and tool steps of the answering run.

    Returns:
        The trace as labelled text, tool calls included with their arguments.
    """
    blocks: list[str] = []
    for step in trace:
        if step.kind == "ai":
            calls = ", ".join(
                f"{call['name']}({call.get('args', {})})" for call in step.tool_calls
            )
            body = step.content.strip()
            if calls:
                body = f"{body}\n-> calls: {calls}" if body else f"-> calls: {calls}"
            if body:
                blocks.append(f"[agent] {body}")
        else:
            blocks.append(f"[{step.name or 'tool'} result] {step.content.strip()}")
    return "\n\n".join(blocks)


def judge_retrieval(
    model: BaseChatModel,
    case: Case,
    trace: list[TraceMessage],
    *,
    recall_threshold: float = 1.0,
) -> RetrievalVerdict:
    """Score what the agent's search actually surfaced.

    Args:
        model: The grading model.
        case: The case being scored.
        trace: The answering run's AI and tool steps.
        recall_threshold: Fraction of gold turns that has to be surfaced for
            retrieval to count as correct.

    Returns:
        The recall, which turns were surfaced, and whether the strict criterion
        holds. Cases with no gold turn come back `applicable=False`.
    """
    gold = case.gold_turns
    if not gold:
        return RetrievalVerdict(
            applicable=False,
            reasoning="the case has no answer-bearing message to retrieve",
        )

    numbered = "\n\n".join(
        f"[{turn.turn_id}] ({turn.role}) {turn.content}" for turn in gold
    )
    payload = (
        render_block("Question", case.question)
        + render_block("Source messages that carry the answer", numbered, limit=8000)
        + render_block(
            "Superseded fact (must not be surfaced as current)",
            "\n\n".join(case.superseded_evidence),
        )
        + render_block("Agent search trace", render_trace(trace), limit=20000)
    )

    try:
        audit = ask_judge(model, _RetrievalAudit, RETRIEVAL_PROMPT, payload)
    except JudgeError as exc:
        return RetrievalVerdict(
            gold_turn_ids=[turn.turn_id for turn in gold], error=str(exc)
        )

    gold_ids = {turn.turn_id for turn in gold}
    retrieved = sorted(gold_ids & set(audit.retrieved_ids))
    recall = len(retrieved) / len(gold_ids)
    leaked = bool(audit.superseded_present and case.superseded_evidence)
    strict = case.category in STRICT_RETRIEVAL_CATEGORIES
    return RetrievalVerdict(
        correct=recall >= recall_threshold and not (strict and leaked),
        reasoning=audit.reasoning,
        recall=recall,
        gold_turn_ids=sorted(gold_ids),
        retrieved_turn_ids=retrieved,
        superseded_leaked=leaked,
    )
