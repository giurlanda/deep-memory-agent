import json
import threading

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from dma_bench.answer import (
    ANSWER_PROMPT,
    ANSWER_PROMPT_CON,
    answer_case,
    extract_trace,
)
from dma_bench.calls import record_call
from dma_bench.categories import BenchCategory
from dma_bench.ingest import INGESTION_PROMPT, render_session
from dma_bench.runner import (
    Runtime,
    case_directory,
    estimate_run,
    iter_pending,
    load_case_results,
    load_experiment,
    run_case,
    run_experiment,
)
from dma_bench.schema import CallRecord, CaseResult, RunConfig

FAILURE = "no service"


def config(tmp_path, **overrides):
    return RunConfig(experiment_root=str(tmp_path / "run"), **overrides)


def write_call(summary="Acme moved to Enterprise"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "memory_write",
                "args": {
                    "category": "facts",
                    "content": "Acme is on the Enterprise plan.",
                    "summary": summary,
                },
                "id": "call-1",
            }
        ],
    )


def runtime(fake_model, *, agent_script=(), structured=None):
    return Runtime(
        agent_model=lambda: fake_model(*agent_script, default_reply="Enterprise"),
        judge_model=lambda: fake_model(structured=structured or {}),
    )


def graded(*, qa=True, consolidation=True, retrieved=("s2#0",)):
    return {
        "_Verdict": {"correct": qa, "reasoning": "graded"},
        "_Audit": {"correct": consolidation, "reasoning": "graded"},
        "_RetrievalAudit": {
            "retrieved_ids": list(retrieved),
            "superseded_present": False,
            "reasoning": "graded",
        },
    }


def test_a_session_is_rendered_as_a_labelled_transcript(case):
    rendered = render_session(case.sessions[1])

    assert "## Session s2" in rendered
    assert "[user] the renewal moved Acme up to Enterprise" in rendered
    assert "[assistant] noted" in rendered


def test_the_ingestion_prompt_states_when_the_session_happened(case):
    prompt = INGESTION_PROMPT.format(
        date="2023-05-20", transcript=render_session(case.sessions[1])
    )

    assert "2023-05-20" in prompt
    assert "already happened" in prompt


def test_chain_of_note_asks_for_extraction_before_reasoning():
    assert "step by step" in ANSWER_PROMPT_CON
    assert "step by step" not in ANSWER_PROMPT


def test_the_trace_keeps_ai_and_tool_steps_and_drops_the_question():
    trace = extract_trace(
        [
            {"role": "user", "content": "the question"},
            AIMessage(
                content="searching",
                tool_calls=[{"name": "memory_search", "args": {}, "id": "1"}],
            ),
            ToolMessage(content="one hit", name="memory_search", tool_call_id="1"),
            AIMessage(content="Enterprise"),
        ]
    )

    assert [step.kind for step in trace] == ["ai", "tool", "ai"]
    assert trace[0].tool_calls[0]["name"] == "memory_search"


def test_an_answering_failure_is_recorded_rather_than_raised(case, failing_model):
    record = answer_case(failing_model, case)

    assert record.answer == ""
    assert record.error is not None


def test_the_estimate_counts_sessions_and_invocations(case, tmp_path):
    estimate = estimate_run([case], config(tmp_path))

    assert estimate["cases"] == 1
    assert estimate["sessions"] == 2
    assert estimate["ingestion_invocations"] == 2
    assert estimate["consolidation_invocations"] == 0
    assert estimate["estimated_model_calls_low"] > 0


def test_the_estimate_accounts_for_scheduled_consolidation(case, tmp_path):
    estimate = estimate_run(
        [case],
        config(tmp_path, consolidation_mode="periodic+final", consolidate_every_n=1),
    )

    assert estimate["consolidation_invocations"] == 3


def test_a_case_runs_end_to_end_and_writes_its_result(case, fake_model, tmp_path):
    settings = config(tmp_path)
    result = run_case(
        case,
        settings,
        runtime(
            fake_model,
            agent_script=[write_call(), AIMessage(content="stored")],
            structured=graded(),
        ),
    )

    assert result.error is None
    assert result.ingestion.sessions == 2
    assert result.qa.correct is True
    assert result.retrieval.recall == 1.0
    assert result.consolidation.correct is True

    written = json.loads((case_directory(settings, case) / "result.json").read_text())
    assert written["question_id"] == "case-1"
    assert written["retrieval"]["recall"] == 1.0


def test_the_memory_tree_lands_under_the_case_directory(case, fake_model, tmp_path):
    settings = config(tmp_path)
    run_case(case, settings, runtime(fake_model, structured=graded()))

    memory = case_directory(settings, case) / "memory"
    assert (memory / "index.md").exists()
    assert (memory / "semantic_memory" / "facts.md").exists()


def test_writes_are_stamped_with_the_session_date_not_today(case, fake_model, tmp_path):
    settings = config(tmp_path)
    run_case(
        case,
        settings,
        runtime(
            fake_model,
            agent_script=[write_call(), AIMessage(content="stored")],
            structured=graded(),
        ),
    )

    facts = case_directory(settings, case) / "memory" / "semantic_memory" / "facts.md"
    assert "created: 2023-03-02" in facts.read_text()


def test_the_cold_arm_has_no_consolidate_tool(fake_model, tmp_path):
    from dma_bench.agents import build_manager_agent

    agent = build_manager_agent(
        fake_model(), tmp_path / "memory", allow_consolidation=False
    )

    names = set(agent.nodes["tools"].bound._tools_by_name)
    assert "memory_write" in names
    assert "memory_consolidate" not in names


def test_the_consolidating_arm_keeps_the_tool(fake_model, tmp_path):
    from dma_bench.agents import build_manager_agent

    agent = build_manager_agent(
        fake_model(), tmp_path / "memory", allow_consolidation=True
    )

    assert "memory_consolidate" in set(agent.nodes["tools"].bound._tools_by_name)


def test_a_knowledge_update_case_is_judged_twice(case, fake_model, tmp_path):
    case.category = BenchCategory.KNOWLEDGE_UPDATE
    case.superseded_evidence = ["they were on Team"]

    result = run_case(case, config(tmp_path), runtime(fake_model, structured=graded()))

    assert result.qa.judge == "knowledge-update"
    assert result.supersede_integrity is not None
    assert result.supersede_integrity.judge == "supersede-integrity"


def test_a_case_without_superseded_evidence_is_judged_once(case, fake_model, tmp_path):
    result = run_case(case, config(tmp_path), runtime(fake_model, structured=graded()))

    assert result.supersede_integrity is None


def test_a_finished_case_is_reused_instead_of_rerun(case, fake_model, tmp_path):
    settings = config(tmp_path)
    first = run_case(case, settings, runtime(fake_model, structured=graded()))

    second = run_case(case, settings, runtime(fake_model, structured=graded(qa=False)))

    assert second.qa.correct == first.qa.correct is True


def test_resume_can_be_turned_off(case, fake_model, tmp_path):
    settings = config(tmp_path)
    run_case(case, settings, runtime(fake_model, structured=graded()))

    rerun = run_case(
        case,
        settings.model_copy(update={"resume": False}),
        runtime(fake_model, structured=graded(qa=False)),
    )

    assert rerun.qa.correct is False


def test_pending_cases_are_the_ones_without_a_result(case, fake_model, tmp_path):
    settings = config(tmp_path)
    assert iter_pending([case], settings) == [case]

    run_case(case, settings, runtime(fake_model, structured=graded()))

    assert iter_pending([case], settings) == []


def test_a_truncated_result_is_redone_rather_than_trusted(case, fake_model, tmp_path):
    settings = config(tmp_path)
    (case_directory(settings, case) / "result.json").write_text("{ truncated")

    result = run_case(case, settings, runtime(fake_model, structured=graded()))

    assert result.error is None
    assert result.question_id == "case-1"


def test_an_experiment_writes_a_summary_that_reloads(case, fake_model, tmp_path):
    settings = config(tmp_path, max_workers=1)
    seen = []

    experiment = run_experiment(
        [case],
        settings,
        runtime(fake_model, structured=graded()),
        on_result=seen.append,
    )

    assert len(seen) == 1
    assert experiment.summary["cases"] == 1
    assert experiment.summary["qa_accuracy"] == 1.0

    reloaded = load_experiment(settings.experiment_root)
    assert reloaded.summary["qa_accuracy"] == 1.0
    assert [
        result.question_id for result in load_case_results(settings.experiment_root)
    ] == ["case-1"]


def test_a_case_that_blows_up_is_recorded_not_propagated(case, tmp_path):
    def explode():
        raise RuntimeError(FAILURE)

    result = run_case(
        case, config(tmp_path), Runtime(agent_model=explode, judge_model=explode)
    )

    assert result.error is not None
    assert FAILURE in result.error


def semantic_runtime(
    fake_model, embeddings, *, agent_script=(), structured=None, store=None
):
    from langchain_core.vectorstores import InMemoryVectorStore

    base = runtime(fake_model, agent_script=agent_script, structured=structured)
    stores = {}

    def build_store(model, case):
        if store is not None:
            return store(model, case)
        return stores.setdefault(case.question_id, InMemoryVectorStore(model))

    return Runtime(
        agent_model=base.agent_model,
        judge_model=base.judge_model,
        embeddings=lambda: embeddings,
        vector_store=build_store,
    )


def test_a_semantic_run_answers_the_same_case_through_both_arms(
    case, fake_model, embeddings, tmp_path
):
    settings = config(tmp_path, semantic_search=True)

    result = run_case(
        case,
        settings,
        semantic_runtime(
            fake_model,
            embeddings,
            agent_script=[write_call(), AIMessage(content="stored")],
            structured=graded(),
        ),
    )

    assert result.error is None
    assert result.answer.attempted
    assert result.semantic is not None
    assert result.semantic.answer.attempted
    assert result.semantic.index.error is None
    assert result.semantic.index.chunks >= 1
    assert result.qa.correct is result.semantic.qa.correct is True


def test_the_semantic_agent_is_the_one_holding_semantic_search(
    fake_model, embeddings, tmp_path
):
    from langchain_core.vectorstores import InMemoryVectorStore

    from dma_bench.agents import build_search_agent

    lexical = build_search_agent(fake_model(), tmp_path / "memory")
    semantic = build_search_agent(
        fake_model(),
        tmp_path / "memory",
        embeddings=embeddings,
        vector_store=InMemoryVectorStore(embeddings),
    )

    assert "semantic_search" not in set(lexical.nodes["tools"].bound._tools_by_name)
    assert "semantic_search" in set(semantic.nodes["tools"].bound._tools_by_name)
    assert "semantic_ingest" not in set(semantic.nodes["tools"].bound._tools_by_name)


def test_a_finished_case_pays_only_for_the_arm_it_is_missing(
    case, fake_model, embeddings, tmp_path
):
    settings = config(tmp_path)
    first = run_case(
        case,
        settings,
        runtime(
            fake_model,
            agent_script=[write_call(), AIMessage(content="stored")],
            structured=graded(),
        ),
    )
    facts = case_directory(settings, case) / "memory" / "semantic_memory" / "facts.md"
    written_once = facts.read_text()

    second = run_case(
        case,
        settings.model_copy(update={"semantic_search": True}),
        semantic_runtime(
            fake_model,
            embeddings,
            agent_script=[write_call("a second, redundant write")],
            structured=graded(),
        ),
    )

    # The history was not replayed: the tree is byte for byte what it was, and
    # the lexical arm is the one the first run graded.
    assert facts.read_text() == written_once
    assert second.ingestion == first.ingestion
    assert second.answer.answer == first.answer.answer
    assert second.semantic is not None
    assert second.semantic.answer.attempted


def test_a_case_missing_its_semantic_arm_is_still_pending(
    case, fake_model, embeddings, tmp_path
):
    settings = config(tmp_path)
    run_case(case, settings, runtime(fake_model, structured=graded()))
    assert iter_pending([case], settings) == []

    semantic = settings.model_copy(update={"semantic_search": True})
    assert iter_pending([case], semantic) == [case]

    run_case(
        case, semantic, semantic_runtime(fake_model, embeddings, structured=graded())
    )
    assert iter_pending([case], semantic) == []


def test_a_store_that_will_not_open_costs_the_semantic_arm_only(
    case, fake_model, embeddings, tmp_path
):
    def explode(_model, _case):
        raise RuntimeError(FAILURE)

    result = run_case(
        case,
        config(tmp_path, semantic_search=True),
        semantic_runtime(fake_model, embeddings, structured=graded(), store=explode),
    )

    assert result.error is None
    assert result.answer.attempted
    assert result.semantic is not None
    assert FAILURE in (result.semantic.index.error or "")
    assert not result.semantic.answer.attempted


def test_a_run_asking_for_an_index_it_cannot_build_is_refused(
    case, fake_model, tmp_path
):
    with pytest.raises(ValueError, match="semantic_search"):
        run_experiment(
            [case],
            config(tmp_path, semantic_search=True),
            runtime(fake_model, structured=graded()),
        )


def test_the_estimate_doubles_the_answering_side_for_a_second_arm(case, tmp_path):
    lexical = estimate_run([case], config(tmp_path))
    both = estimate_run([case], config(tmp_path, semantic_search=True))

    assert both["answer_invocations"] == 2 * lexical["answer_invocations"]
    assert both["judge_invocations"] == 2 * lexical["judge_invocations"]
    assert both["ingestion_invocations"] == lexical["ingestion_invocations"]
    assert both["semantic_index_passes"] == 1


def edit_result(settings, case, change):
    path = case_directory(settings, case) / "result.json"
    written = json.loads(path.read_text())
    change(written)
    path.write_text(json.dumps(written))


@pytest.mark.parametrize("stage", ["answer", "retrieval", "qa"])
def test_an_arm_that_failed_is_run_again_on_resume(case, fake_model, tmp_path, stage):
    settings = config(tmp_path)
    run_case(case, settings, runtime(fake_model, structured=graded()))
    edit_result(
        settings,
        case,
        lambda written: written[stage].update(error="OpenAIConnectionError()"),
    )
    assert iter_pending([case], settings) == [case]

    rerun = run_case(case, settings, runtime(fake_model, structured=graded(qa=False)))

    assert getattr(rerun, stage).error is None
    assert rerun.qa.correct is False
    assert iter_pending([case], settings) == []


def test_a_replay_that_lost_a_session_is_redone_from_an_empty_tree(
    case, fake_model, tmp_path
):
    settings = config(tmp_path)
    run_case(case, settings, runtime(fake_model, structured=graded()))

    def lose_a_session(written):
        written["ingestion"]["sessions"] = 1
        written["ingestion"]["failed_sessions"] = ["s1: OpenAIConnectionError()"]

    edit_result(settings, case, lose_a_session)
    stale = case_directory(settings, case) / "memory" / "stale.md"
    stale.write_text("left behind by the attempt that failed")
    assert iter_pending([case], settings) == [case]

    rerun = run_case(case, settings, runtime(fake_model, structured=graded()))

    assert rerun.ingestion.sessions == 2
    assert rerun.ingestion.failed_sessions == []
    assert not stale.exists()


def test_a_spent_invocation_budget_stops_each_invocation_and_is_recorded(
    case, fake_model, tmp_path
):
    result = run_case(
        case,
        config(tmp_path, invocation_timeout_s=0),
        runtime(fake_model, structured=graded()),
    )

    assert result.error is None
    assert result.ingestion.sessions == 2
    assert result.ingestion.budget_stops == ["s1: time", "s2: time"]
    assert result.answer.stopped_by == "time"


def test_model_calls_are_recorded_under_the_stage_that_made_them(
    case, fake_model, tmp_path
):
    class RecordingModel(type(fake_model())):
        """A scripted model that records each call, as `dma_bench.llm` does."""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            record_call(CallRecord(model="recording"))
            return super()._generate(messages, stop, run_manager, **kwargs)

    result = run_case(
        case,
        config(tmp_path),
        Runtime(
            agent_model=lambda: RecordingModel(default_reply="Enterprise"),
            judge_model=lambda: fake_model(structured=graded()),
        ),
    )

    stages = [call.stage for call in result.llm_calls]
    assert stages.count("ingestion") == 2
    assert stages.count("answer:lexical") == 1


def test_cases_are_reported_as_they_finish_and_kept_in_input_order(
    case, monkeypatch, tmp_path
):
    from dma_bench import runner as runner_module

    fast_reported = threading.Event()

    def fake_run_case(case, config, runtime):  # noqa: ARG001
        if case.question_id == "slow":
            fast_reported.wait(timeout=5)
        return CaseResult(question_id=case.question_id, category=case.category)

    def report(result):
        seen.append(result.question_id)
        if result.question_id == "fast":
            fast_reported.set()

    monkeypatch.setattr(runner_module, "run_case", fake_run_case)
    seen = []
    slow = case.model_copy(update={"question_id": "slow"})
    fast = case.model_copy(update={"question_id": "fast"})

    experiment = run_experiment(
        [slow, fast],
        config(tmp_path, max_workers=2),
        Runtime(agent_model=lambda: None, judge_model=lambda: None),
        on_result=report,
    )

    assert seen == ["fast", "slow"]
    assert [result.question_id for result in experiment.results] == ["slow", "fast"]
