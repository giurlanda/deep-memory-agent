import json
from datetime import UTC, datetime

import pytest

from dma_bench.categories import BenchCategory
from dma_bench.datasets.longmemeval import (
    dataset_path,
    iter_cases,
    load_cases,
    parse_timestamp,
    to_case,
)


def record(question_id="q1", question_type="single-session-user", **overrides):
    base = {
        "question_id": question_id,
        "question_type": question_type,
        "question": "Which plan?",
        "answer": "Enterprise",
        "question_date": "2023/06/01 (Thu) 09:00",
        "answer_session_ids": ["b"],
        "haystack_session_ids": ["a", "b"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/03/02 (Thu) 11:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "unrelated"}],
            [{"role": "user", "content": "moved to Enterprise", "has_answer": True}],
        ],
    }
    return base | overrides


def test_parses_a_timestamp_without_relying_on_the_weekday():
    assert parse_timestamp("2023/04/10 (Mon) 23:07") == datetime(
        2023, 4, 10, 23, 7, tzinfo=UTC
    )


def test_rejects_text_that_holds_no_timestamp():
    with pytest.raises(ValueError, match="cannot read a timestamp"):
        parse_timestamp("last Tuesday")


def test_sessions_come_back_in_chronological_order():
    case = to_case(record())

    assert [session.session_id for session in case.sessions] == ["b", "a"]
    assert case.sessions[0].date < case.sessions[1].date


def test_evidence_sessions_and_gold_turns_are_marked():
    case = to_case(record())

    assert [session.is_evidence for session in case.sessions] == [True, False]
    assert [turn.turn_id for turn in case.gold_turns] == ["b#0"]


def test_abstention_is_read_off_the_question_id_not_the_type():
    case = to_case(record(question_id="q1_abs", question_type="multi-session"))

    assert case.category is BenchCategory.ABSTENTION
    assert case.is_abstention


def test_knowledge_update_keeps_the_earlier_evidence_as_superseded():
    case = to_case(
        record(
            question_type="knowledge-update",
            answer_session_ids=["old", "new"],
            haystack_session_ids=["old", "new"],
            haystack_dates=["2023/03/02 (Thu) 11:00", "2023/05/20 (Sat) 02:21"],
            haystack_sessions=[
                [{"role": "user", "content": "we are on Team", "has_answer": True}],
                [
                    {
                        "role": "user",
                        "content": "moved to Enterprise",
                        "has_answer": True,
                    }
                ],
            ],
        )
    )

    # The latest evidence holds the current value; everything before it is what
    # must no longer be presented as true.
    assert case.superseded_evidence == ["we are on Team"]


def test_other_categories_carry_no_superseded_evidence():
    assert to_case(record()).superseded_evidence == []


def test_a_single_evidence_session_cannot_supersede_anything():
    case = to_case(record(question_type="knowledge-update"))

    assert case.superseded_evidence == []


def test_records_stream_out_of_a_json_array(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record("q1"), record("q2"), record("q3")]))

    assert [case.question_id for case in iter_cases(path)] == ["q1", "q2", "q3"]


def test_streaming_survives_a_record_that_spans_several_reads(tmp_path, monkeypatch):
    from dma_bench.datasets import longmemeval

    monkeypatch.setattr(longmemeval, "_CHUNK_CHARS", 16)
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record("q1"), record("q2")], indent=2))

    assert [case.question_id for case in iter_cases(path)] == ["q1", "q2"]


def test_sampling_is_balanced_and_reproducible(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [record(f"u{index}") for index in range(5)]
            + [record(f"m{index}", "multi-session") for index in range(5)]
        )
    )

    first = load_cases(path, n_per_category=2, seed=7)
    second = load_cases(path, n_per_category=2, seed=7)

    assert len(first) == 4
    assert {case.category for case in first} == {
        BenchCategory.SINGLE_HOP,
        BenchCategory.MULTI_HOP,
    }
    assert [case.question_id for case in first] == [case.question_id for case in second]


def test_sampling_can_be_restricted_to_one_category(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record("u1"), record("m1", "multi-session")]))

    cases = load_cases(path, categories=[BenchCategory.MULTI_HOP])

    assert [case.question_id for case in cases] == ["m1"]


def test_scales_resolve_to_the_published_files(tmp_path):
    assert dataset_path(tmp_path, "small").name == "longmemeval_oracle.json"
    assert dataset_path(tmp_path, "large").name == "longmemeval_s_cleaned.json"
    assert dataset_path(tmp_path, "m").name == "longmemeval_m_cleaned.json"


def test_an_unknown_variant_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown LongMemEval variant"):
        dataset_path(tmp_path, "xl")
