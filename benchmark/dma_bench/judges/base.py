"""Shared plumbing for the judges.

One helper, because all three judges do the same thing: send a system prompt and
a payload, ask for a typed answer, and turn a model failure into a recorded
`error` rather than an exception that takes the run down. A judge that crashes
mid-run would otherwise cost every case already graded behind it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = ["JudgeError", "ask_judge", "render_block"]

_ATTEMPTS = 2


class JudgeError(RuntimeError):
    """Raised when a judge could not produce a verdict."""


def ask_judge[SchemaT: BaseModel](
    model: BaseChatModel,
    schema: type[SchemaT],
    system: str,
    payload: str,
) -> SchemaT:
    """Ask a judge for a typed verdict.

    Args:
        model: The grading model.
        schema: Pydantic model the answer must fit.
        system: The judging instructions.
        payload: The material to judge.

    Returns:
        The verdict.

    Raises:
        JudgeError: If the model failed on every attempt.
    """
    grader = model.with_structured_output(schema)
    last: Exception | None = None
    for _ in range(_ATTEMPTS):
        try:
            verdict = grader.invoke(
                [SystemMessage(content=system), HumanMessage(content=payload)]
            )
        except Exception as exc:
            last = exc
            continue
        if isinstance(verdict, schema):
            return verdict
        return schema.model_validate(verdict)
    msg = f"judge failed after {_ATTEMPTS} attempts: {last!r}"
    raise JudgeError(msg)


def render_block(title: str, body: str, *, limit: int = 4000) -> str:
    """Render one labelled section of a judge payload.

    Args:
        title: Section heading.
        body: Section content.
        limit: Maximum characters kept; the middle is what gets dropped, since
            the head and the tail of a trace both carry signal.

    Returns:
        The section, ready to concatenate.
    """
    text = body.strip() or "(none)"
    if len(text) > limit:
        head, tail = text[: limit // 2], text[-limit // 2 :]
        text = f"{head}\n\n[… {len(text) - limit} characters elided …]\n\n{tail}"
    return f"## {title}\n\n{text}\n"
