from dma_bench.categories import BenchCategory
from dma_bench.metrics import CELL_DIAGNOSIS, cell_label, decompose, summarise
from dma_bench.schema import (
    CallRecord,
    CaseResult,
    IngestionRecord,
    MemorySnapshot,
    QaVerdict,
    RetrievalVerdict,
    SemanticArm,
    StageVerdict,
)


def result(
    question_id="c1",
    category=BenchCategory.SINGLE_HOP,
    *,
    consolidation=True,
    retrieval=True,
    answer=True,
    recall=1.0,
    applicable=True,
    **overrides,
):
    return CaseResult(
        question_id=question_id,
        category=category,
        consolidation=StageVerdict(correct=consolidation),
        retrieval=RetrievalVerdict(
            correct=retrieval, recall=recall, applicable=applicable
        ),
        qa=QaVerdict(correct=answer),
        **overrides,
    )


def calls(*spec, stage="ingestion"):
    return [
        CallRecord(stage=stage, seconds=seconds, attempts=attempts, outcome=outcome)
        for seconds, attempts, outcome in spec
    ]


def test_the_cost_shows_the_tail_of_the_model_calls_and_how_they_ended():
    recorded = calls(
        (1.0, 1, "ok"),
        (2.0, 2, "ok"),
        (30.0, 1, "truncated"),
        (120.0, 3, "timeout"),
        (5.0, 1, "error"),
    )

    cost = summarise([result(llm_calls=recorded)])["cost"]

    assert cost["llm_calls"] == 5
    assert cost["llm_seconds"] == 158.0
    assert cost["llm_p95_call_s"] == cost["llm_max_call_s"] == 120.0
    assert cost["llm_retried_calls"] == 2
    assert cost["llm_truncated_calls"] == 1
    assert cost["llm_timed_out_calls"] == 1
    assert cost["llm_failed_calls"] == 1


def test_a_run_without_recorded_calls_reports_no_call_cost():
    assert "llm_calls" not in summarise([result()])["cost"]


def test_the_semantic_summary_leaves_out_the_lexical_arms_calls():
    recorded = [
        CallRecord(stage=stage, seconds=1.0)
        for stage in ("ingestion", "answer:lexical", "judge:lexical", "answer:semantic")
    ]

    summary = summarise([result(llm_calls=recorded, semantic=SemanticArm())])

    assert summary["cost"]["llm_calls"] == 4
    assert summary["semantic"]["cost"]["llm_calls"] == 2


def test_every_combination_has_a_diagnosis():
    labels = {
        cell_label(consolidation=c, retrieval=r, answer=g)
        for c in (True, False)
        for r in (True, False)
        for g in (True, False)
    }

    assert labels == set(CELL_DIAGNOSIS)


def test_the_decomposition_counts_each_case_once():
    results = [
        result("a"),
        result("b", consolidation=False, retrieval=False, answer=False),
        result("c", retrieval=False, answer=False),
    ]

    decomposition = decompose(results)

    assert decomposition["scored"] == 3
    assert decomposition["cells"]["C✓ R✓ G✓"]["count"] == 1
    assert decomposition["cells"]["C✗ R✗ G✗"]["count"] == 1
    assert decomposition["cells"]["C✓ R✗ G✗"]["count"] == 1


def test_cases_whose_retrieval_cannot_be_scored_are_excluded_not_failed():
    results = [
        result("a"),
        result("b", applicable=False, recall=None, retrieval=False),
    ]

    decomposition = decompose(results)

    assert decomposition["scored"] == 1
    assert decomposition["excluded"] == 1
    assert sum(cell["count"] for cell in decomposition["cells"].values()) == 1


def test_a_case_that_errored_is_excluded_from_the_decomposition():
    decomposition = decompose([result("a"), result("b", error="boom")])

    assert decomposition["scored"] == 1
    assert decomposition["excluded"] == 1


def test_recall_is_averaged_only_over_cases_that_have_gold():
    summary = summarise(
        [
            result("a", recall=1.0),
            result("b", recall=0.5),
            result("c", applicable=False, recall=None),
        ]
    )

    assert summary["retrieval_recall"] == 0.75
    assert summary["retrieval_correct_rate"] == 1.0


def test_categories_are_reported_separately():
    summary = summarise(
        [
            result("a", BenchCategory.SINGLE_HOP, answer=True),
            result("b", BenchCategory.MULTI_HOP, answer=False),
        ]
    )

    assert summary["by_category"]["single-hop"]["qa_accuracy"] == 1.0
    assert summary["by_category"]["multi-hop"]["qa_accuracy"] == 0.0
    assert summary["qa_accuracy"] == 0.5


def test_supersede_integrity_is_reported_as_its_own_row():
    lenient = result("a", BenchCategory.KNOWLEDGE_UPDATE, answer=True)
    lenient.supersede_integrity = QaVerdict(correct=False)
    lenient.retrieval.superseded_leaked = True

    summary = summarise([lenient])

    assert summary["by_category"]["knowledge-update"]["qa_accuracy"] == 1.0
    strict = summary["by_category"]["supersede-integrity"]
    assert strict["qa_accuracy"] == 0.0
    assert strict["superseded_leak_rate"] == 1.0


def test_failed_cases_are_counted_but_kept_out_of_the_averages():
    summary = summarise([result("a", answer=True), result("b", error="boom")])

    assert summary["cases"] == 2
    assert summary["failed_cases"] == 1
    assert summary["qa_accuracy"] == 1.0


def test_the_cost_block_sums_what_the_run_spent():
    first = result("a")
    first.ingestion = IngestionRecord(sessions=3, write_calls=5, duration_s=1.5)
    first.memory_snapshot = MemorySnapshot(active_entries=4, superseded_entries=1)
    second = result("b")
    second.ingestion = IngestionRecord(sessions=2, write_calls=1, duration_s=0.5)

    cost = summarise([first, second])["cost"]

    assert cost["sessions_ingested"] == 5
    assert cost["write_calls"] == 6
    assert cost["active_entries"] == 4
    assert cost["ingestion_seconds"] == 2.0


def test_an_empty_run_summarises_to_nothing_rather_than_crashing():
    summary = summarise([])

    assert summary["cases"] == 0
    assert summary["qa_accuracy"] is None
    assert summary["decomposition"]["scored"] == 0


def test_the_semantic_arm_is_summarised_beside_the_lexical_one():
    from dma_bench.metrics import semantic_view
    from dma_bench.schema import AnswerRecord, SemanticArm, SemanticIndexRecord

    results = [
        result(
            "c1",
            answer=False,
            semantic=SemanticArm(
                index=SemanticIndexRecord(added=3, chunks=4),
                answer=AnswerRecord(answer="Enterprise", tool_calls=2),
                retrieval=RetrievalVerdict(correct=True, recall=1.0),
                qa=QaVerdict(correct=True),
            ),
        )
    ]

    summary = summarise(results)

    assert summary["qa_accuracy"] == 0.0
    assert summary["semantic"]["qa_accuracy"] == 1.0
    assert summary["semantic"]["consolidation_correct_rate"] == 1.0
    assert summary["cost"]["indexed_chunks"] == 4
    assert "semantic" not in summary["semantic"]
    assert [view.qa.correct for view in semantic_view(results)] == [True]


def test_a_case_without_a_semantic_arm_is_left_out_of_its_summary():
    from dma_bench.metrics import semantic_view

    summary = summarise([result("c1"), result("c2")])

    assert semantic_view([result("c1")]) == []
    assert "semantic" not in summary
    assert "indexed_chunks" not in summary["cost"]
