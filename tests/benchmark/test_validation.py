import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from dma_bench.generation.validation import (
    GeneratedTurn,
    strip_speaker_label,
    structural_problem,
    validate_session,
)

GOOD = [
    GeneratedTurn(role="user", content="can you look at the overnight job?"),
    GeneratedTurn(role="assistant", content="pulling it up, it tripped at two"),
]


class Judge(BaseChatModel):
    """Answers with a fixed verdict, or raises."""

    usable: bool = True
    reason: str = ""
    broken: bool = False
    calls: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "judge"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        return ChatResult(generations=[ChatGeneration(message=None)])

    def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
        def run(prompt):
            self.calls.append(str(prompt))
            if self.broken:
                msg = "no service"
                raise RuntimeError(msg)
            return schema(usable=self.usable, reason=self.reason)

        return RunnableLambda(run)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("USER: morning, any news?", "morning, any news?"),
        ("Assistant: it tripped at two", "it tripped at two"),
        ("Me: leave it queued for now", "leave it queued for now"),
        ("U: nope, that's it", "nope, that's it"),
        ("Sablefin: draft it but keep it short", "draft it but keep it short"),
        # A colon that belongs to the sentence is not a speaker label.
        ("Note: the window moved", "the window moved"),
        ("what about 03:30 instead?", "what about 03:30 instead?"),
    ],
)
def test_a_speaker_label_is_taken_off_the_front_of_a_turn(text, expected):
    assert strip_speaker_label(text) == expected


def test_only_the_leading_label_goes():
    # The rest of the sentence keeps whatever colons it had.
    assert strip_speaker_label("User: tell them: it moved") == "tell them: it moved"


def test_a_well_formed_conversation_has_no_structural_problem():
    assert structural_problem(GOOD) is None


def test_a_conversation_with_no_turns_is_refused():
    assert structural_problem([]) == "the conversation came back empty"


def test_a_conversation_that_opens_with_the_assistant_is_refused():
    turns = [GeneratedTurn(role="assistant", content="pulling it up right now")]

    assert structural_problem(turns) == "the conversation has to start with the user"


def test_two_turns_from_the_same_speaker_are_refused():
    # This is the shape that used to slip through and offset every role after it.
    turns = [
        *GOOD,
        GeneratedTurn(role="assistant", content="scratch database is up now"),
    ]

    problem = structural_problem(turns)
    assert problem is not None
    assert "turn 2" in problem
    assert "alternate" in problem


def test_a_turn_holding_both_speakers_is_refused():
    turns = [
        GeneratedTurn(
            role="user",
            content="quick note on the cleanup\n\nassistant: nothing else pending",
        )
    ]

    assert structural_problem(turns) == "turn 0 holds both speakers instead of one"


def test_a_turn_that_is_the_response_schema_is_refused():
    turns = [GeneratedTurn(role="user", content='type: user\ncontent: "kick it off"')]

    problem = structural_problem(turns)
    assert problem == "turn 0 has the response schema written out as text"


@pytest.mark.parametrize("text", ["user", "assistant", "...", "  ", ""])
def test_a_placeholder_turn_is_refused(text):
    turns = [GeneratedTurn(role="user", content=text)]

    assert structural_problem(turns) == "turn 0 is empty or a placeholder"


def test_a_session_the_validator_accepts_comes_back_clean():
    assert validate_session(Judge(), GOOD, "the payload") is None


def test_the_reason_a_session_was_rejected_comes_back():
    judge = Judge(usable=False, reason="both sides are the user")

    assert validate_session(judge, GOOD, "the payload") == "both sides are the user"


def test_a_rejection_without_a_reason_still_says_something():
    judge = Judge(usable=False)

    assert validate_session(judge, GOOD, "the payload") == "the validator rejected it"


def test_the_validator_is_shown_the_payload_and_the_conversation():
    judge = Judge()
    validate_session(judge, GOOD, "they moved up to Enterprise")

    prompt = judge.calls[0]
    assert "they moved up to Enterprise" in prompt
    assert "[user] can you look at the overnight job?" in prompt


def test_a_validator_that_cannot_be_reached_lets_the_session_through():
    # An unreachable judge is not a verdict. Treating it as one would burn every
    # session's retry budget and abandon the run a case at a time.
    assert validate_session(Judge(broken=True), GOOD, "the payload") is None
