import pytest
from langchain_core.messages import AIMessage

from deep_memory_agent.layout import MemoryCategory
from dma_bench.categories import BenchCategory
from dma_bench.judges.base import JudgeError, ask_judge, render_block
from dma_bench.judges.consolidation import (
    judge_consolidation,
    render_memory,
    snapshot_memory,
)
from dma_bench.judges.qa import QA_PROMPTS, judge_answer, judge_supersede
from dma_bench.judges.retrieval import judge_retrieval, render_trace
from dma_bench.schema import TraceMessage


def verdict(*, correct=True, reasoning="because"):
    # The QA judge and the consolidation judge take the same shape under
    # different private names, so one helper can stand in for both.
    payload = {"correct": correct, "reasoning": reasoning}
    return {"_Verdict": payload, "_Audit": payload}


def audit(ids=(), *, superseded=False):
    return {
        "_RetrievalAudit": {
            "retrieved_ids": list(ids),
            "superseded_present": superseded,
            "reasoning": "checked",
        }
    }


def test_every_category_has_a_grading_prompt():
    assert set(QA_PROMPTS) == set(BenchCategory)


def test_a_long_block_is_elided_in_the_middle_not_the_tail():
    rendered = render_block("Trace", "A" + "B" * 500 + "Z", limit=100)

    assert rendered.startswith("## Trace")
    assert "elided" in rendered
    assert rendered.rstrip().endswith("Z")


def test_a_judge_that_keeps_failing_raises_rather_than_returning_a_pass(
    failing_model, fake_model
):
    with pytest.raises(JudgeError):
        ask_judge(failing_model, type(fake_model()), "system", "payload")


def test_a_qa_judge_failure_is_recorded_not_raised(failing_model, case):
    result = judge_answer(failing_model, case, "Enterprise")

    assert result.correct is False
    assert result.error is not None


def test_the_qa_judge_is_tagged_with_the_category_prompt(fake_model, case):
    result = judge_answer(fake_model(structured=verdict()), case, "Enterprise")

    assert result.correct is True
    assert result.judge == "single-hop"


def test_the_strict_prompt_is_applied_by_the_supersede_judge(fake_model, case):
    case.superseded_evidence = ["they were on Team"]

    result = judge_supersede(
        fake_model(structured=verdict(correct=False)), case, "Team"
    )

    assert result.correct is False
    assert result.judge == "supersede-integrity"


def test_recall_is_the_fraction_of_gold_turns_the_trace_surfaced(fake_model, case):
    case.sessions[0].turns[0].has_answer = True

    result = judge_retrieval(
        fake_model(structured=audit(ids=["s2#0"])), case, [], recall_threshold=1.0
    )

    assert result.recall == 0.5
    assert result.retrieved_turn_ids == ["s2#0"]
    assert result.correct is False


def test_a_turn_the_judge_invented_does_not_count_towards_recall(fake_model, case):
    result = judge_retrieval(
        fake_model(structured=audit(ids=["s2#0", "made-up#9"])), case, []
    )

    assert result.retrieved_turn_ids == ["s2#0"]
    assert result.recall == 1.0


def test_a_case_with_no_gold_turn_is_not_scored_zero(fake_model, case):
    for session in case.sessions:
        for turn in session.turns:
            turn.has_answer = False

    result = judge_retrieval(fake_model(structured=audit()), case, [])

    assert result.applicable is False
    assert result.recall is None


def test_a_leaked_superseded_fact_fails_strict_retrieval(fake_model, case):
    case.category = BenchCategory.KNOWLEDGE_UPDATE
    case.superseded_evidence = ["they were on Team"]

    result = judge_retrieval(
        fake_model(structured=audit(ids=["s2#0"], superseded=True)), case, []
    )

    assert result.recall == 1.0
    assert result.superseded_leaked is True
    assert result.correct is False


def test_a_leak_outside_the_strict_categories_does_not_fail_retrieval(fake_model, case):
    case.superseded_evidence = ["they were on Team"]

    result = judge_retrieval(
        fake_model(structured=audit(ids=["s2#0"], superseded=True)), case, []
    )

    assert result.superseded_leaked is True
    assert result.correct is True


def test_a_lower_threshold_accepts_partial_recall(fake_model, case):
    case.sessions[0].turns[0].has_answer = True

    result = judge_retrieval(
        fake_model(structured=audit(ids=["s2#0"])), case, [], recall_threshold=0.5
    )

    assert result.correct is True


def test_the_trace_renders_tool_calls_and_their_results():
    rendered = render_trace(
        [
            TraceMessage(
                kind="ai",
                content="let me look",
                tool_calls=[{"name": "memory_search", "args": {"query": "acme"}}],
            ),
            TraceMessage(kind="tool", name="memory_search", content="found one entry"),
        ]
    )

    assert "memory_search" in rendered
    assert "acme" in rendered
    assert "found one entry" in rendered


def test_the_snapshot_counts_active_and_superseded_entries(store):
    first = store.write(MemoryCategory.FACTS, "Acme is on Team", summary="plan")
    store.write(
        MemoryCategory.FACTS,
        "Acme is on Enterprise",
        summary="plan",
        supersedes=first.entry.entry_id,
    )

    snapshot = snapshot_memory(store)

    assert snapshot.active_entries == 1
    assert snapshot.superseded_entries == 1
    assert snapshot.entries_by_category == {"facts": 1}


def test_only_active_entries_are_shown_to_the_consolidation_judge(store):
    first = store.write(MemoryCategory.FACTS, "Acme is on Team", summary="old plan")
    store.write(
        MemoryCategory.FACTS,
        "Acme is on Enterprise",
        summary="current plan",
        supersedes=first.entry.entry_id,
    )

    rendered = render_memory(store)

    assert "Enterprise" in rendered
    assert "Acme is on Team" not in rendered


def test_the_consolidation_judge_reads_the_tree_not_the_transcript(
    fake_model, case, store
):
    store.write(MemoryCategory.FACTS, "Acme is on Enterprise", summary="plan")

    result = judge_consolidation(fake_model(structured=verdict()), case, store)

    assert result.correct is True
    assert result.applicable is True


def test_the_scripted_model_is_still_usable_as_an_agent_model(fake_model):
    model = fake_model(AIMessage(content="hello"))

    assert model.bind_tools([]) is model
