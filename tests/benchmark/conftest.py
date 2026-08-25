"""Fixtures for the benchmark tests.

The benchmark drives real agents and three different judges, so its fake model
has to do two things the library's `ScriptedChatModel` does not: play a tool-
calling script *and* return a different structured answer depending on which
judge asked. Keying the structured answers by schema name is what lets one model
stand in for all three graders in a single end-to-end run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from dma_bench.schema import Case, Session, Turn

FAILURE = "no service"


def _raise(_):
    raise RuntimeError(FAILURE)


class FakeModel(BaseChatModel):
    """Replays a script of AI messages and answers judges by schema name."""

    responses: list[AIMessage] = Field(default_factory=list)
    structured: dict[str, Any] = Field(default_factory=dict)
    default_reply: str = "done"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        message = (
            self.responses.pop(0)
            if self.responses
            else AIMessage(content=self.default_reply)
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
        payload = self.structured.get(schema.__name__, {})
        return RunnableLambda(lambda _: schema.model_validate(payload))


@pytest.fixture
def fake_model():
    """Factory for a model that replays a script and grades on demand."""

    def build(
        *responses: AIMessage,
        structured: dict[str, Any] | None = None,
        default_reply: str = "done",
    ) -> FakeModel:
        return FakeModel(
            responses=list(responses),
            structured=structured or {},
            default_reply=default_reply,
        )

    return build


class FailingModel(BaseChatModel):
    """A model whose structured-output call always fails."""

    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        raise RuntimeError(FAILURE)

    def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
        return RunnableLambda(_raise)


@pytest.fixture
def failing_model():
    """A model that cannot produce a verdict, to exercise judge error paths."""
    return FailingModel()


@pytest.fixture
def turn():
    """Factory for a single turn."""

    def build(session_id: str, index: int, content: str, *, gold: bool = False) -> Turn:
        return Turn(
            turn_id=f"{session_id}#{index}",
            role="user" if index % 2 == 0 else "assistant",
            content=content,
            has_answer=gold,
        )

    return build


@pytest.fixture
def case(turn):
    """A two-session case whose second session carries the answer."""
    first = Session(
        session_id="s1",
        date=datetime(2023, 3, 2, 10, 0, tzinfo=UTC),
        turns=[turn("s1", 0, "we kicked off the migration")],
    )
    second = Session(
        session_id="s2",
        date=datetime(2023, 5, 20, 15, 0, tzinfo=UTC),
        turns=[
            turn("s2", 0, "the renewal moved Acme up to Enterprise", gold=True),
            turn("s2", 1, "noted"),
        ],
        is_evidence=True,
    )
    return Case(
        question_id="case-1",
        category="single-hop",
        question="Which plan is Acme on?",
        answer="Enterprise",
        question_date=datetime(2023, 6, 1, 9, 0, tzinfo=UTC),
        sessions=[first, second],
    )
