import json
from datetime import UTC, datetime

import pytest

from dma_bench.categories import BenchCategory
from dma_bench.datasets import DATASETS, load_dataset
from dma_bench.datasets.operational import corpus_path, load_cases
from dma_bench.generation.generator import CORPUS_SHAPES
from dma_bench.generation.ontology import ERRORS, PROCEDURES


def case_record(question_id, category=BenchCategory.PROCEDURAL_RETRIEVAL):
    return {
        "question_id": question_id,
        "category": category.value,
        "question": "the deploy just failed mid-migration, what now?",
        "answer": "roll back from the snapshot",
        "question_date": datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
        "sessions": [
            {
                "session_id": f"{question_id}-s0",
                "date": datetime(2023, 12, 1, tzinfo=UTC).isoformat(),
                "turns": [
                    {
                        "turn_id": f"{question_id}-s0#0",
                        "role": "user",
                        "content": "we restored from the snapshot instead",
                        "has_answer": True,
                    }
                ],
                "is_evidence": True,
            }
        ],
        "source": "operational",
        "expected_procedure": "restore the pre-migration snapshot",
    }


def test_every_procedure_carries_a_trigger_and_its_steps():
    assert PROCEDURES
    for procedure in PROCEDURES:
        assert procedure["title"]
        assert procedure["trigger"]
        assert procedure["steps"]


def test_every_error_carries_its_correction():
    assert ERRORS
    for error in ERRORS:
        assert error["mistake"]
        assert error["consequence"]
        assert error["correction"]


def test_the_large_shape_spans_a_timeline_longmemeval_cannot_offer():
    # The published datasets pack their sessions into about ten days, so monthly
    # sharding never has more than a shard or two to route between. This is the
    # configuration that actually exercises it.
    assert CORPUS_SHAPES["large"].span_days >= 180
    assert CORPUS_SHAPES["large"].distractor_sessions > 20
    assert CORPUS_SHAPES["small"].span_days <= 31


def test_a_generated_corpus_round_trips(tmp_path):
    path = tmp_path / "operational_small.json"
    path.write_text(json.dumps([case_record("p-00"), case_record("p-01")]))

    cases = load_cases(path)

    assert [case.question_id for case in cases] == ["p-00", "p-01"]
    assert cases[0].category is BenchCategory.PROCEDURAL_RETRIEVAL
    assert cases[0].expected_procedure
    assert [turn.turn_id for turn in cases[0].gold_turns] == ["p-00-s0#0"]


def test_sampling_a_corpus_is_reproducible(tmp_path):
    path = tmp_path / "operational_small.json"
    path.write_text(json.dumps([case_record(f"p-{index:02d}") for index in range(6)]))

    first = load_cases(path, n_per_category=2, seed=3)
    second = load_cases(path, n_per_category=2, seed=3)

    assert len(first) == 2
    assert [case.question_id for case in first] == [case.question_id for case in second]


def test_a_missing_corpus_says_how_to_generate_it(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"generation\.generator"):
        load_cases(tmp_path / "operational_small.json")


def test_scales_map_to_corpus_files(tmp_path):
    assert corpus_path(tmp_path, "small").name == "operational_small.json"
    with pytest.raises(ValueError, match="unknown operational scale"):
        corpus_path(tmp_path, "xl")


def test_the_unified_loader_dispatches_to_the_right_provider(tmp_path):
    (tmp_path / "operational_small.json").write_text(json.dumps([case_record("p-00")]))

    cases = load_dataset("operational", tmp_path, scale="small")

    assert [case.question_id for case in cases] == ["p-00"]


def test_the_unified_loader_rejects_an_unknown_provider(tmp_path):
    assert "longmemeval" in DATASETS
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("nowhere", tmp_path)
