import pytest
from langchain_core.messages import AIMessage

from dma_bench.agents import build_search_agent
from dma_bench.answer import answer_case
from dma_bench.budget import InvocationBudget


def searching(times):
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "memory_search", "args": {"query": "Acme"}, "id": f"call-{i}"}
            ],
        )
        for i in range(times)
    ]


def test_an_agent_that_keeps_searching_is_stopped_at_its_call_budget(
    case, fake_model, tmp_path
):
    agent = build_search_agent(
        fake_model(*searching(5)),
        tmp_path / "memory",
        middleware=[InvocationBudget(model_calls=2)],
    )

    record = answer_case(agent, case)

    assert record.error is None
    assert record.stopped_by == "calls"
    assert record.tool_calls == 2


def test_a_spent_time_budget_stops_before_the_next_model_call(
    case, fake_model, tmp_path
):
    model = fake_model(*searching(5))
    agent = build_search_agent(
        model, tmp_path / "memory", middleware=[InvocationBudget(seconds=0)]
    )

    record = answer_case(agent, case)

    assert record.stopped_by == "time"
    assert record.tool_calls == 0
    assert len(model.responses) == 5


def test_an_invocation_within_its_budget_is_left_alone(case, fake_model, tmp_path):
    agent = build_search_agent(
        fake_model(default_reply="Enterprise"),
        tmp_path / "memory",
        middleware=[InvocationBudget(seconds=60, model_calls=5)],
    )

    record = answer_case(agent, case)

    assert record.stopped_by is None
    assert record.answer == "Enterprise"


def test_a_budget_that_bounds_nothing_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        InvocationBudget()
