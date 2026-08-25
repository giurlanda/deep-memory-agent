from contextlib import contextmanager

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from dma_bench.categories import BenchCategory
from dma_bench.generation import generator
from dma_bench.generation.generator import (
    CORPUS_SHAPES,
    CorpusShape,
    calls_per_category,
    generate_corpus,
    write_corpus,
)

SHAPE = CorpusShape(
    cases_per_category=2, evidence_sessions=1, distractor_sessions=1, span_days=10
)


class CountingModel(BaseChatModel):
    """Records every structured-output call and can be told to fail some."""

    calls: list[str] = Field(default_factory=list)
    fail: str = ""

    @property
    def _llm_type(self) -> str:
        return "counting"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        return ChatResult(generations=[ChatGeneration(message=None)])

    def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
        def run(_):
            self.calls.append(schema.__name__)
            if schema.__name__ == self.fail:
                msg = "no service"
                raise RuntimeError(msg)
            if schema.__name__ == "_GeneratedSession":
                return schema(turns=["something happened", "noted"])
            return schema(question="what now?", answer="follow the steps")

        return RunnableLambda(run)


@contextmanager
def recorder(log):
    """Replace the bar with something that records how it was driven."""

    @contextmanager
    def fake(label, total, *, enabled):
        log.append(("bar", label, total, enabled))
        yield lambda detail: log.append(("step", detail))

    yield fake


def test_the_call_count_covers_every_session_and_every_question():
    # Two cases, one evidence plus one distractor session each, one question each.
    assert calls_per_category(SHAPE) == 6


def test_the_large_shape_is_the_expensive_one():
    assert calls_per_category(CORPUS_SHAPES["large"]) > calls_per_category(
        CORPUS_SHAPES["small"]
    )


def test_a_corpus_is_generated_with_the_shape_it_was_asked_for():
    model = CountingModel()

    cases = generate_corpus(
        model,
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
    )

    assert len(cases) == 2
    assert {case.category for case in cases} == {BenchCategory.PROCEDURAL_RETRIEVAL}
    assert all(len(case.sessions) == 2 for case in cases)
    assert all(case.expected_procedure for case in cases)
    assert all(case.gold_turns for case in cases)


def test_a_non_repetition_case_carries_its_error_and_correction():
    cases = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.NON_REPETITION],
        progress=False,
    )

    assert all(case.past_error and case.past_correction for case in cases)
    assert all(case.expected_procedure is None for case in cases)


def test_only_evidence_turns_are_marked_as_gold():
    cases = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
    )

    gold_sessions = {turn.turn_id.split("#")[0] for turn in cases[0].gold_turns}
    evidence = {s.session_id for s in cases[0].sessions if s.is_evidence}
    assert gold_sessions == evidence


def test_sessions_come_out_in_chronological_order():
    cases = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.NON_REPETITION],
        progress=False,
    )

    dates = [session.date for session in cases[0].sessions]
    assert dates == sorted(dates)


def test_progress_is_off_by_request_and_never_touches_tqdm(monkeypatch):
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(
            CountingModel(),
            SHAPE,
            categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
            progress=False,
        )

    assert [entry for entry in log if entry[0] == "bar"] == [
        ("bar", "procedural-retrieval", 6, False)
    ]


def test_there_is_one_bar_per_category(monkeypatch):
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(CountingModel(), SHAPE, progress=True)

    bars = [entry for entry in log if entry[0] == "bar"]
    assert [entry[1] for entry in bars] == ["procedural-retrieval", "non-repetition"]
    assert all(entry[2] == calls_per_category(SHAPE) for entry in bars)
    assert all(entry[3] is True for entry in bars)


def test_the_bar_reaches_its_total(monkeypatch):
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(
            CountingModel(),
            SHAPE,
            categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
            progress=True,
        )

    steps = [entry for entry in log if entry[0] == "step"]
    assert len(steps) == calls_per_category(SHAPE)


def test_the_bar_still_reaches_its_total_when_every_session_fails(monkeypatch):
    # A case with no evidence is abandoned before the question is ever asked, so
    # that skipped call has to be accounted for or the bar hangs short.
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        cases = generate_corpus(
            CountingModel(fail="_GeneratedSession"),
            SHAPE,
            categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
            progress=True,
        )

    assert cases == []
    assert len([entry for entry in log if entry[0] == "step"]) == calls_per_category(
        SHAPE
    )
    assert [entry[1] for entry in log if entry[0] == "step"][-1].endswith("abandoned")


def test_a_case_whose_question_fails_is_dropped():
    cases = generate_corpus(
        CountingModel(fail="_GeneratedQuestion"),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
    )

    assert cases == []


def test_a_generated_corpus_survives_a_round_trip(tmp_path):
    from dma_bench.datasets.operational import load_cases

    cases = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
    )
    path = write_corpus(cases, tmp_path / "operational_small.json")

    reloaded = load_cases(path)

    assert [case.question_id for case in reloaded] == [
        case.question_id for case in cases
    ]
    assert reloaded[0].expected_procedure == cases[0].expected_procedure


def test_the_cli_can_turn_the_bar_off():
    parser = generator._build_parser()

    assert parser.parse_args(["--out", "x.json"]).quiet is False
    assert parser.parse_args(["--out", "x.json", "--quiet"]).quiet is True
