from contextlib import contextmanager

import pytest
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
    load_corpus,
    session_budget,
    write_corpus,
)

SHAPE = CorpusShape(
    cases_per_category=2, evidence_sessions=1, distractor_sessions=1, span_days=10
)

GOOD_TURNS = [
    ("user", "can you look at the Northfield job that ran overnight?"),
    ("assistant", "pulling it up now, it tripped around two in the morning"),
]


class CountingModel(BaseChatModel):
    """Records every structured-output call and can be told to misbehave.

    `fail` names a schema whose calls raise. `turns` is what a session comes back
    as. `rejections` is how many verdicts come back unusable before one passes,
    which is how the retry budget is exercised without a real validator.
    """

    calls: list[str] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)
    fail: str = ""
    turns: list[tuple[str, str]] = Field(default_factory=lambda: list(GOOD_TURNS))
    rejections: int = 0
    reject_matching: str = ""

    @property
    def _llm_type(self) -> str:
        return "counting"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        return ChatResult(generations=[ChatGeneration(message=None)])

    def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
        def run(prompt):
            self.calls.append(schema.__name__)
            self.prompts.append(str(prompt))
            if schema.__name__ == self.fail:
                msg = "no service"
                raise RuntimeError(msg)
            if schema.__name__ == "GeneratedSession":
                return schema(
                    turns=[{"role": role, "content": text} for role, text in self.turns]
                )
            if schema.__name__ == "SessionVerdict":
                if self.rejections > 0:
                    self.rejections -= 1
                    return schema(usable=False, reason="the payload never came through")
                if self.reject_matching and self.reject_matching in str(prompt):
                    return schema(usable=False, reason="nothing of the payload landed")
                return schema(usable=True)
            return schema(question="what now?", answer="follow the steps")

        return RunnableLambda(run)


@contextmanager
def recorder(log):
    """Replace the bar with something that records how it was driven."""

    @contextmanager
    def fake(label, total, *, enabled):
        log.append(("bar", label, total, enabled))
        yield lambda detail, steps=1: log.append(("step", detail, steps))

    yield fake


def stepped(log):
    """Return how far the bar was actually moved."""
    return sum(entry[2] for entry in log if entry[0] == "step")


def test_the_call_count_covers_every_session_and_every_question():
    # Two cases, one evidence plus one distractor session each, one question each.
    assert calls_per_category(SHAPE) == 6


def test_the_large_shape_is_the_expensive_one():
    assert calls_per_category(CORPUS_SHAPES["large"]) > calls_per_category(
        CORPUS_SHAPES["small"]
    )


def test_the_medium_shape_sits_between_the_other_two():
    # The point of `medium` is the middle of the cost range, not a third corner
    # of it: a long timeline like `large`'s, at a spend closer to `small`'s.
    small, medium, large = (
        CORPUS_SHAPES[name] for name in ("small", "medium", "large")
    )

    assert (
        calls_per_category(small)
        < calls_per_category(medium)
        < calls_per_category(large)
    )
    assert small.span_days < medium.span_days < large.span_days


def test_the_case_count_can_be_overridden_without_moving_anything_else():
    cases = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=3,
        progress=False,
    )

    assert len(cases) == 3
    # Only the count moves: SHAPE's one evidence and one distractor session per
    # case survive the override.
    assert all(len(case.sessions) == 2 for case in cases)


def test_the_overridden_count_leaves_the_shape_it_was_handed_untouched():
    generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=1,
        progress=False,
    )

    assert SHAPE.cases_per_category == 2


def test_a_case_count_of_zero_or_less_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        generate_corpus(CountingModel(), SHAPE, cases_per_category=0, progress=False)


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
            validate=False,
            max_session_retries=0,
            progress=False,
        )

    assert [entry for entry in log if entry[0] == "bar"] == [
        ("bar", "procedural-retrieval", 6, False)
    ]


def test_there_is_one_bar_per_category(monkeypatch):
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(
            CountingModel(),
            SHAPE,
            validate=False,
            max_session_retries=0,
            progress=True,
        )

    bars = [entry for entry in log if entry[0] == "bar"]
    assert [entry[1] for entry in bars] == ["procedural-retrieval", "non-repetition"]
    assert all(entry[2] == calls_per_category(SHAPE) for entry in bars)
    assert all(entry[3] is True for entry in bars)


def test_the_bar_is_sized_from_the_overridden_count(monkeypatch):
    # The bar counts model calls, so an override that halves the cases has to
    # halve the total too, or the bar never reaches its end.
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(
            CountingModel(),
            SHAPE,
            categories=[BenchCategory.NON_REPETITION],
            cases_per_category=1,
            validate=False,
            max_session_retries=0,
            progress=True,
        )

    assert [entry for entry in log if entry[0] == "bar"] == [
        ("bar", "non-repetition", 3, True)
    ]
    assert stepped(log) == 3


def test_the_bar_reaches_its_total(monkeypatch):
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(
            CountingModel(),
            SHAPE,
            categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
            validate=False,
            max_session_retries=0,
            progress=True,
        )

    assert stepped(log) == calls_per_category(SHAPE)


def test_the_bar_still_reaches_its_total_when_every_session_fails(monkeypatch):
    # A case with no evidence is abandoned before the question is ever asked, so
    # that skipped call has to be accounted for or the bar hangs short.
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        cases = generate_corpus(
            CountingModel(fail="GeneratedSession"),
            SHAPE,
            categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
            validate=False,
            max_session_retries=0,
            progress=True,
        )

    assert cases == []
    assert stepped(log) == calls_per_category(SHAPE)
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


def test_the_cli_takes_a_case_count_and_defaults_to_the_configured_one():
    parser = generator._build_parser()

    assert parser.parse_args(["--out", "x.json"]).cases_per_category is None
    args = parser.parse_args(["--out", "x.json", "--cases-per-category", "3"])
    assert args.cases_per_category == 3


def test_the_cli_refuses_a_case_count_of_zero_or_less():
    parser = generator._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--out", "x.json", "--cases-per-category", "0"])


def test_every_finished_case_is_on_disk_before_the_next_one_starts(
    tmp_path, monkeypatch
):
    # The point of the checkpoint: a run that dies after the first case still
    # has the first case, so the file grows one case at a time rather than once
    # at the end.
    out = tmp_path / "corpus.json"
    written = []
    original = generator.write_corpus

    def spy(cases, path):
        written.append(len(cases))
        return original(cases, path)

    monkeypatch.setattr(generator, "write_corpus", spy)
    generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
        out=out,
    )

    assert written == [1, 2]
    assert len(load_corpus(out)) == 2


def test_a_corpus_is_only_topped_up_to_the_count_it_was_asked_for(tmp_path):
    out = tmp_path / "corpus.json"
    generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=1,
        progress=False,
        out=out,
    )

    model = CountingModel()
    cases = generate_corpus(
        model,
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=3,
        progress=False,
        out=out,
    )

    assert len(cases) == 3
    # Two cases were missing, so only two were paid for.
    assert model.calls.count("_GeneratedQuestion") == 2
    assert [case.question_id for case in load_corpus(out)] == [
        "procedural-retrieval-00",
        "procedural-retrieval-01",
        "procedural-retrieval-02",
    ]


def test_a_corpus_that_is_already_full_costs_nothing_to_rerun(tmp_path):
    out = tmp_path / "corpus.json"
    generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
        out=out,
    )
    before = out.read_text()

    model = CountingModel()
    cases = generate_corpus(
        model,
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
        out=out,
    )

    assert model.calls == []
    assert len(cases) == 2
    assert out.read_text() == before


def test_a_resumed_case_draws_what_it_would_have_drawn_in_one_run(tmp_path):
    # Entity sampling is keyed by the case index, so growing a corpus does not
    # hand the new cases the clients and procedures the first ones already had.
    one_run = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=2,
        progress=False,
    )

    out = tmp_path / "corpus.json"
    generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=1,
        progress=False,
        out=out,
    )
    resumed = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=2,
        progress=False,
        out=out,
    )

    assert [case.expected_procedure for case in resumed] == [
        case.expected_procedure for case in one_run
    ]


def test_ids_continue_past_a_gap_left_by_an_abandoned_case(tmp_path):
    # A case the model abandoned mid-run is not on disk, so the corpus has a
    # hole in its numbering. The next run must not hand its new case an id that
    # a later, surviving case already took.
    cases = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=3,
        progress=False,
    )
    out = write_corpus([cases[0], cases[2]], tmp_path / "corpus.json")

    grown = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=3,
        progress=False,
        out=out,
    )

    assert [case.question_id for case in grown] == [
        "procedural-retrieval-00",
        "procedural-retrieval-02",
        "procedural-retrieval-03",
    ]


def test_the_bar_is_sized_from_what_is_left_to_generate(tmp_path, monkeypatch):
    out = tmp_path / "corpus.json"
    generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.NON_REPETITION],
        cases_per_category=1,
        progress=False,
        out=out,
    )

    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(
            CountingModel(),
            SHAPE,
            categories=[BenchCategory.NON_REPETITION],
            validate=False,
            max_session_retries=0,
            progress=True,
            out=out,
        )

    # One case left of the two, so one case worth of calls.
    assert [entry for entry in log if entry[0] == "bar"] == [
        ("bar", "non-repetition", 3, True)
    ]


def test_a_full_category_gets_no_bar_at_all(monkeypatch, tmp_path):
    out = tmp_path / "corpus.json"
    generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
        out=out,
    )

    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(CountingModel(), SHAPE, progress=True, out=out)

    assert [entry[1] for entry in log if entry[0] == "bar"] == ["non-repetition"]


def test_a_corpus_survives_being_written_and_read_back(tmp_path):
    cases = generate_corpus(
        CountingModel(),
        SHAPE,
        categories=[BenchCategory.NON_REPETITION],
        progress=False,
    )
    path = write_corpus(cases, tmp_path / "nested" / "corpus.json")

    assert load_corpus(path) == cases
    assert not list(path.parent.glob("*.tmp"))


def test_a_file_that_is_not_a_corpus_is_refused_rather_than_overwritten(tmp_path):
    out = tmp_path / "corpus.json"
    out.write_text('{"cases": []}')

    with pytest.raises(ValueError, match="not a corpus"):
        generate_corpus(CountingModel(), SHAPE, progress=False, out=out)

    assert out.read_text() == '{"cases": []}'


def test_the_cli_explains_that_an_existing_out_is_resumed():
    parser = generator._build_parser()

    action = next(a for a in parser._actions if a.dest == "out")
    assert "resumed" in (action.help or "")


ONLY_EVIDENCE = CorpusShape(
    cases_per_category=1, evidence_sessions=1, distractor_sessions=0, span_days=10
)


FOUR_TURNS = [
    ("user", "can you look at the overnight job?"),
    ("assistant", "pulling it up, it tripped at two"),
    ("user", "right. how bad is the gap on the rollups?"),
    ("assistant", "about forty thousand rows, staged for her to check"),
]

OFFSET_TURNS = [
    ("user", "can you look at the overnight job?"),
    ("assistant", "pulling it up, it tripped at two"),
    ("assistant", "scratch database is up, diffing the tables now"),
    ("user", "good, that is what she needs. how bad is the gap?"),
]


def test_the_role_of_a_turn_is_the_one_the_model_declared():
    model = CountingModel(turns=FOUR_TURNS)

    cases = generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        validate=False,
        max_session_retries=0,
        progress=False,
    )

    turns = cases[0].sessions[0].turns
    assert [(turn.role, turn.content) for turn in turns] == FOUR_TURNS


def test_a_conversation_that_does_not_alternate_never_enters_the_corpus():
    # The bug this replaces: roles came from the turn's position, so a model that
    # answered twice in a row had everything after it silently relabelled. Now
    # the disagreement is caught instead of papered over.
    model = CountingModel(turns=OFFSET_TURNS)

    cases = generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        validate=False,
        progress=False,
    )

    assert cases == []


def test_a_conversation_that_opens_with_the_assistant_never_enters_the_corpus():
    model = CountingModel(turns=[("assistant", "pulling it up, it tripped at two")])

    cases = generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.NON_REPETITION],
        validate=False,
        progress=False,
    )

    assert cases == []


def test_gold_turns_are_the_user_turns_of_an_evidence_session():
    # `has_answer` used to be derived from the same parity as the role, so a
    # slipped session handed the retrieval judge assistant turns as gold.
    model = CountingModel(turns=FOUR_TURNS)

    cases = generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        validate=False,
        max_session_retries=0,
        progress=False,
    )

    gold = cases[0].gold_turns
    assert [turn.role for turn in gold] == ["user", "user"]
    assert [turn.turn_id.split("#")[-1] for turn in gold] == ["0", "2"]


def test_a_speaker_label_never_reaches_the_corpus():
    model = CountingModel(
        turns=[
            ("user", "USER: can you look at the overnight job?"),
            ("assistant", "Assistant: pulling it up, it tripped at two"),
        ]
    )

    cases = generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.NON_REPETITION],
        validate=False,
        max_session_retries=0,
        progress=False,
    )

    contents = [turn.content for turn in cases[0].sessions[0].turns]
    assert contents == [
        "can you look at the overnight job?",
        "pulling it up, it tripped at two",
    ]


def test_a_session_is_validated_by_the_model_that_wrote_it_by_default():
    model = CountingModel()

    generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
    )

    assert model.calls.count("SessionVerdict") == 1


def test_a_validator_of_its_own_takes_the_judging_off_the_writer():
    model = CountingModel()
    judge = CountingModel()

    generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        validator=judge,
        progress=False,
    )

    assert "SessionVerdict" not in model.calls
    assert judge.calls == ["SessionVerdict"]


def test_validation_can_be_turned_off():
    model = CountingModel()

    generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        validate=False,
        progress=False,
    )

    assert "SessionVerdict" not in model.calls


def test_a_rejected_session_is_written_again_with_the_reason_it_was_rejected():
    model = CountingModel(rejections=1)

    cases = generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
    )

    assert model.calls.count("GeneratedSession") == 2
    assert len(cases) == 1
    retry = model.prompts[2]
    assert "rejected: the payload never came through" in retry


def test_a_session_is_only_rewritten_as_many_times_as_the_budget_allows():
    model = CountingModel(rejections=99)

    generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        max_session_retries=3,
        progress=False,
    )

    assert model.calls.count("GeneratedSession") == 4


def test_a_distractor_that_never_holds_up_is_dropped_and_the_case_survives():
    # Distractor payloads are the only ones that talk about routine work, so
    # this rejects those and leaves the evidence alone.
    model = CountingModel(reject_matching="Routine work on this account")

    cases = generate_corpus(
        model,
        SHAPE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        cases_per_category=1,
        progress=False,
    )

    assert len(cases) == 1
    assert [session.is_evidence for session in cases[0].sessions] == [True]


def test_an_evidence_session_that_never_holds_up_abandons_the_case():
    # The question is written from the evidence, so a case that lost it would
    # ask about material the corpus no longer carries.
    model = CountingModel(rejections=99)

    cases = generate_corpus(
        model,
        ONLY_EVIDENCE,
        categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
        progress=False,
    )

    assert cases == []
    assert "_GeneratedQuestion" not in model.calls


def test_the_bar_still_reaches_its_total_when_sessions_are_retried(monkeypatch):
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(
            CountingModel(rejections=1),
            SHAPE,
            categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
            progress=True,
        )

    budget = session_budget(validate=True, max_retries=2)
    assert [entry[2] for entry in log if entry[0] == "bar"] == [
        calls_per_category(SHAPE, budget=budget)
    ]
    assert stepped(log) == calls_per_category(SHAPE, budget=budget)


def test_the_bar_still_reaches_its_total_when_a_case_is_abandoned(monkeypatch):
    log = []
    with recorder(log) as fake:
        monkeypatch.setattr(generator, "_progress", fake)
        generate_corpus(
            CountingModel(rejections=99),
            SHAPE,
            categories=[BenchCategory.PROCEDURAL_RETRIEVAL],
            progress=True,
        )

    budget = session_budget(validate=True, max_retries=2)
    assert stepped(log) == calls_per_category(SHAPE, budget=budget)


def test_a_validated_run_costs_more_than_an_unvalidated_one():
    assert calls_per_category(
        SHAPE, budget=session_budget(validate=True, max_retries=2)
    ) > calls_per_category(SHAPE, budget=session_budget())


def test_the_cli_validates_by_default_and_can_be_told_not_to():
    parser = generator._build_parser()

    assert parser.parse_args(["--out", "x.json"]).validate is True
    assert parser.parse_args(["--out", "x.json", "--no-validate"]).validate is False


def test_the_cli_takes_a_retry_budget_and_defaults_it_to_two():
    parser = generator._build_parser()

    assert parser.parse_args(["--out", "x.json"]).max_session_retries == 2
    assert (
        parser.parse_args(
            ["--out", "x.json", "--max-session-retries", "0"]
        ).max_session_retries
        == 0
    )


def test_the_cli_refuses_a_negative_retry_budget():
    with pytest.raises(SystemExit):
        generator._build_parser().parse_args(
            ["--out", "x.json", "--max-session-retries", "-1"]
        )


def test_the_validator_falls_back_to_the_provider_of_the_writing_model():
    args = generator._build_parser().parse_args(
        [
            "--out",
            "x.json",
            "--model",
            "qwen",
            "--base-url",
            "http://local",
            "--api-key",
            "secret",
        ]
    )
    built = {}

    def factory(**kwargs):
        built.update(kwargs)
        return "model"

    generator._build_validator(args, factory)

    assert built["model"] == "qwen"
    assert built["base_url"] == "http://local"
    assert built["api_key"] == "secret"
    assert built["temperature"] == 0


def test_the_validator_can_be_pointed_at_a_provider_of_its_own():
    args = generator._build_parser().parse_args(
        [
            "--out",
            "x.json",
            "--model",
            "qwen",
            "--base-url",
            "http://local",
            "--api-key",
            "secret",
            "--validator-model",
            "gpt-4o",
            "--validator-base-url",
            "https://api.openai.com/v1",
            "--validator-api-key",
            "other",
        ]
    )
    built = {}

    def factory(**kwargs):
        built.update(kwargs)
        return "model"

    generator._build_validator(args, factory)

    assert built["model"] == "gpt-4o"
    assert built["base_url"] == "https://api.openai.com/v1"
    assert built["api_key"] == "other"
