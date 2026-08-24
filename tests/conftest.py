"""Shared pytest fixtures for deep-memory-agent."""

from __future__ import annotations

from typing import Any

import pytest
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from deep_memory_agent.consolidation import ConsolidatedItem, ConsolidationProposal
from deep_memory_agent.layout import MEMORY_ROOT
from deep_memory_agent.store import MemoryStore


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays a fixed script, with tool-calling support.

    Enough of the interface is implemented to drive a compiled deep agent and
    to stand in for the model that consolidation asks for structured output.
    """

    responses: list[AIMessage] = Field(default_factory=list)
    structured_response: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        message = self.responses.pop(0) if self.responses else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
        return RunnableLambda(lambda _: self.structured_response)


@pytest.fixture
def memory_dir(tmp_path):
    """Host directory backing the `/memory/` tree."""
    path = tmp_path / "memory"
    path.mkdir()
    return path


@pytest.fixture
def backend(tmp_path, memory_dir):
    """Backend routing `/memory/` to disk, everything else to a scratch dir.

    The scratch route stands in for the `StateBackend` used at runtime, which
    cannot be exercised outside a LangGraph execution.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return CompositeBackend(
        default=FilesystemBackend(root_dir=scratch),
        routes={MEMORY_ROOT: FilesystemBackend(root_dir=memory_dir)},
    )


@pytest.fixture
def store(backend):
    """A store on a freshly scaffolded memory tree."""
    memory_store = MemoryStore(backend)
    memory_store.ensure_tree()
    return memory_store


@pytest.fixture
def scripted_model():
    """Factory for a chat model that replays scripted messages."""

    def build(*responses: AIMessage) -> ScriptedChatModel:
        return ScriptedChatModel(responses=list(responses))

    return build


@pytest.fixture
def consolidation_model():
    """Factory for a model that returns a fixed consolidation proposal."""

    def build(*items: ConsolidatedItem, rationale: str = "") -> ScriptedChatModel:
        return ScriptedChatModel(
            structured_response=ConsolidationProposal(
                items=list(items), rationale=rationale
            )
        )

    return build
