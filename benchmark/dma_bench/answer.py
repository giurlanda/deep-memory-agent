"""Asking the question and capturing how the answer was reached.

Two things come out of this step, and they are graded by different judges. The
final reply is what the QA judge reads. The trace — every AI turn and every tool
result on the way there — is what the retrieval judge reads, because in this
architecture there is no single retriever call to score: the agent decides for
itself how many times to search, with which wordings, and which files to open.

The Chain-of-Note variant of the prompt is the paper's Figure 13, which reports
up to ten points of headroom even when retrieval is already perfect. It is a
reading strategy layered on top of the shipped recall prompt, not a replacement
for it — the agent's own system prompt is untouched.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, ToolMessage

from dma_bench.clock import simulated_now
from dma_bench.schema import AnswerRecord, TraceMessage

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from dma_bench.schema import Case

__all__ = ["ANSWER_PROMPT", "ANSWER_PROMPT_CON", "answer_case", "extract_trace"]

ANSWER_PROMPT = """\
Today is {date}.

Answer the question below using only what you can find in your memory.

Question: {question}
Answer:"""

ANSWER_PROMPT_CON = """\
Today is {date}.

Answer the question below using only what you can find in your memory. Answer
step by step: first search memory and extract every piece of relevant
information you find, then reason over what you extracted to reach the answer.

Question: {question}
Answer (step by step):"""


def extract_trace(messages: list) -> list[TraceMessage]:
    """Turn an agent's message history into the trace the judge reads.

    Args:
        messages: The `messages` list an agent invocation returned.

    Returns:
        One entry per AI turn and per tool result, in order. The human prompt is
        left out: it is the question, which the judge is given separately.
    """
    trace: list[TraceMessage] = []
    for message in messages:
        if isinstance(message, AIMessage):
            trace.append(
                TraceMessage(
                    kind="ai",
                    content=_as_text(message.content),
                    tool_calls=[
                        {"name": call.get("name", ""), "args": call.get("args", {})}
                        for call in message.tool_calls or ()
                    ],
                )
            )
        elif isinstance(message, ToolMessage):
            trace.append(
                TraceMessage(
                    kind="tool",
                    name=str(message.name or ""),
                    content=_as_text(message.content),
                )
            )
    return trace


def answer_case(
    agent: CompiledStateGraph,
    case: Case,
    *,
    chain_of_note: bool = True,
    recursion_limit: int = 60,
) -> AnswerRecord:
    """Ask the case's question and record both the reply and the trace.

    The clock is moved to the question date so that "today" means what it meant
    when the question was posed — a temporal question answered from a 2026
    vantage point over 2023 episodes is a different question.

    Args:
        agent: The read-only search agent.
        case: The case being answered.
        chain_of_note: Whether to use the extract-then-reason prompt.
        recursion_limit: Cap on agent steps.

    Returns:
        The reply, the trace, and how long it took. A failure is recorded rather
        than raised, so one unanswerable case does not end the run.
    """
    template = ANSWER_PROMPT_CON if chain_of_note else ANSWER_PROMPT
    prompt = template.format(
        date=f"{case.question_date:%Y-%m-%d}", question=case.question
    )
    started = time.monotonic()
    try:
        with simulated_now(case.question_date):
            state = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"recursion_limit": recursion_limit},
            )
    except Exception as exc:
        return AnswerRecord(
            error=repr(exc), duration_s=round(time.monotonic() - started, 2)
        )

    messages = state.get("messages", [])
    trace = extract_trace(messages)
    return AnswerRecord(
        answer=_as_text(messages[-1].content) if messages else "",
        trace=trace,
        tool_calls=sum(len(step.tool_calls) for step in trace),
        duration_s=round(time.monotonic() - started, 2),
    )


def _as_text(content: object) -> str:
    """Flatten message content, which may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "\n".join(part for part in parts if part)
    return str(content)
