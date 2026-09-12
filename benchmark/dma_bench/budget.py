"""A budget on one agent invocation: wall-clock seconds and model calls.

`dma_bench.llm` bounds a single model call. That is not enough on its own: one
invocation — a session replayed into memory, or a question answered — is a loop
of calls, and the step cap (`recursion_limit`) counts graph steps rather than
time, so an agent that keeps searching can spend twenty minutes of perfectly
healthy calls on one session. This middleware stops the loop instead.

It stops it the way `ModelCallLimitMiddleware` does, by jumping to the end with
an AI message rather than raising, so whatever the agent already wrote stays
written and the trace up to that point stays readable. The message carries a
marker in its metadata, so the harness can record *why* the invocation ended
without matching on its wording.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    hook_config,
)
from langchain_core.messages import AIMessage
from langgraph.channels.untracked_value import UntrackedValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langgraph.runtime import Runtime

__all__ = ["BUDGET_KEY", "InvocationBudget", "StopReason", "budget_stop"]

BUDGET_KEY = "dma_bench_budget"
"""Key in `response_metadata` marking the message the budget ended a run with."""

StopReason = Literal["time", "calls"]

_STOP_MESSAGES: dict[StopReason, str] = {
    "time": "[stopped: this invocation ran out of its time budget]",
    "calls": "[stopped: this invocation ran out of its model-call budget]",
}


class BudgetState(AgentState):
    """Agent state extended with what the budget needs to track.

    Untracked and private: the values belong to one invocation, are never
    checkpointed, and never appear in the agent's input or output.
    """

    budget_deadline: NotRequired[
        Annotated[float | None, UntrackedValue, PrivateStateAttr]
    ]
    budget_model_calls: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class InvocationBudget(AgentMiddleware):
    """End an invocation that outlives its time or model-call budget.

    The check runs before every model call, so an invocation overruns its
    deadline by at most one call — and `dma_bench.llm` bounds that.

    Args:
        seconds: Wall-clock budget for one invocation, measured from its start.
            `None` leaves time unbounded.
        model_calls: Model calls one invocation may make. `None` leaves them
            unbounded.

    Raises:
        ValueError: If both budgets are `None` — a budget that bounds nothing
            is a misconfiguration, not a request for no limit.
    """

    state_schema = BudgetState

    def __init__(
        self, *, seconds: float | None = None, model_calls: int | None = None
    ) -> None:
        """Initialise the budget; see the class docstring for the arguments."""
        super().__init__()
        if seconds is None and model_calls is None:
            msg = "InvocationBudget needs at least one of seconds or model_calls"
            raise ValueError(msg)
        self.seconds = seconds
        self.model_calls = model_calls

    def before_agent(
        self,
        state: BudgetState,  # noqa: ARG002 - the hook's signature, not used here
        runtime: Runtime,  # noqa: ARG002 - the hook's signature, not used here
    ) -> dict[str, Any]:
        """Start the clock and the call counter for this invocation."""
        deadline = None if self.seconds is None else time.monotonic() + self.seconds
        return {"budget_deadline": deadline, "budget_model_calls": 0}

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: BudgetState,
        runtime: Runtime,  # noqa: ARG002 - the hook's signature, not used here
    ) -> dict[str, Any] | None:
        """End the invocation if either budget is spent."""
        reason = self._spent(state)
        if reason is None:
            return None
        message = AIMessage(
            content=_STOP_MESSAGES[reason], response_metadata={BUDGET_KEY: reason}
        )
        return {"jump_to": "end", "messages": [message]}

    def after_model(
        self,
        state: BudgetState,
        runtime: Runtime,  # noqa: ARG002 - the hook's signature, not used here
    ) -> dict[str, Any]:
        """Count the call that just finished."""
        return {"budget_model_calls": state.get("budget_model_calls", 0) + 1}

    def _spent(self, state: BudgetState) -> StopReason | None:
        """Return which budget is spent, if either is."""
        deadline = state.get("budget_deadline")
        if deadline is not None and time.monotonic() >= deadline:
            return "time"
        calls = state.get("budget_model_calls", 0)
        if self.model_calls is not None and calls >= self.model_calls:
            return "calls"
        return None


def budget_stop(messages: Sequence[Any]) -> StopReason | None:
    """Return why an invocation was stopped by its budget, if it was.

    Args:
        messages: The `messages` list the invocation returned.

    Returns:
        `"time"` or `"calls"` when the budget ended the run, otherwise `None`.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            reason = message.response_metadata.get(BUDGET_KEY)
            if reason is not None:
                return reason
    return None
