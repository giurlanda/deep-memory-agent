"""Grading the final answer, with one prompt per category.

Four of these prompts are LongMemEval's, reused nearly word for word: they were
meta-evaluated against expert annotators at 90% agreement or better, and
rewriting them would throw that calibration away for no gain.

Three are new, and two of the three exist because this architecture can fail in
ways LongMemEval's cannot. The supersede-integrity prompt is the sharpest
departure: LongMemEval's knowledge-update prompt accepts an answer that carries
the old value alongside the new one, as long as the new one is there. That
tolerance hides the exact failure mode a curated memory has — if the graph never
invalidated the old edge, both facts sit there looking equally true. So the
strict prompt marks that answer wrong, while still allowing the old value to
appear when it is labelled as history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from dma_bench.categories import BenchCategory
from dma_bench.judges.base import JudgeError, ask_judge, render_block
from dma_bench.schema import QaVerdict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from dma_bench.schema import Case

__all__ = [
    "QA_PROMPTS",
    "SUPERSEDE_INTEGRITY_PROMPT",
    "judge_answer",
    "judge_supersede",
]

_GENERIC = """\
I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, \
answer no. If the response is equivalent to the correct answer or contains all \
the intermediate steps to get the correct answer, you should also answer yes. \
If the response only contains a subset of the information required by the \
answer, answer no."""

_TEMPORAL = f"""\
{_GENERIC} In addition, do not penalize off-by-one errors for the number of \
days. If the question asks for the number of days/weeks/months, etc., and the \
model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), \
the model's response is still correct."""

_KNOWLEDGE_UPDATE = """\
I will give you a question, a correct answer, and a response from a model. \
Please answer yes if the response contains the correct answer. Otherwise, \
answer no. If the response contains some previous information along with an \
updated answer, the response should be considered as correct as long as the \
updated answer is the required answer."""

_PREFERENCE = """\
I will give you a question, a rubric for desired personalized response, and a \
response from a model. Please answer yes if the response satisfies the desired \
response. Otherwise, answer no. The model does not need to reflect all the \
points in the rubric. The response is correct as long as it recalls and \
utilizes the user's personal information correctly."""

_ABSTENTION = """\
I will give you a question that cannot be answered from what the model was \
told, an explanation of why it cannot be answered, and a response from a model. \
Please answer yes if the model recognizes that the information was never \
provided — it may say the information is missing, or offer related information \
while making clear that the thing asked about is not on record. Answer no if \
the model invents an answer, or presents a guess as if it were something it had \
been told."""

SUPERSEDE_INTEGRITY_PROMPT = """\
I will give you a question about the current state of an entity, the correct \
fact (the active one), the previous fact (the superseded one), and a response \
from a model. Answer yes only if the response reflects the active fact as the \
current state. Answer no if the response presents the superseded fact as if it \
were still valid, even if it also correctly mentions the new fact elsewhere in \
the response. Mentioning the superseded fact while explicitly marking it as \
history — former, previous, no longer the case — does not make the response \
wrong."""

_PROCEDURAL = """\
I will give you a scenario, the correct procedure to apply, and the model's \
action. Answer yes if the model followed the steps of the correct procedure, \
even in different words. Answer no if it improvised, followed a wrong \
procedure, or ignored a relevant one."""

_NON_REPETITION = """\
I will give you a past error together with the correction that was made, and a \
new scenario where the same error would be possible. Answer yes if the model \
avoids the error or applies the correction. Answer no if it repeats the \
original error."""

QA_PROMPTS: dict[BenchCategory, str] = {
    BenchCategory.SINGLE_HOP: _GENERIC,
    BenchCategory.MULTI_HOP: _GENERIC,
    BenchCategory.PREFERENCE: _PREFERENCE,
    BenchCategory.KNOWLEDGE_UPDATE: _KNOWLEDGE_UPDATE,
    BenchCategory.SUPERSEDE_INTEGRITY: SUPERSEDE_INTEGRITY_PROMPT,
    BenchCategory.TEMPORAL_REASONING: _TEMPORAL,
    BenchCategory.ABSTENTION: _ABSTENTION,
    BenchCategory.PROCEDURAL_RETRIEVAL: _PROCEDURAL,
    BenchCategory.NON_REPETITION: _NON_REPETITION,
}
"""The grading instruction for each category."""

_TAIL = """

Answer with `correct: true` or `correct: false`, and one sentence of reasoning."""


class _Verdict(BaseModel):
    """Structured form of the judge's yes/no."""

    correct: bool = Field(description="Whether the response is correct.")
    reasoning: str = Field(default="", description="One sentence of justification.")


def judge_answer(model: BaseChatModel, case: Case, answer: str) -> QaVerdict:
    """Grade a model answer with its category's prompt.

    Args:
        model: The grading model.
        case: The case being graded.
        answer: The model's reply.

    Returns:
        The verdict, tagged with the prompt that produced it.
    """
    label = "Rubric" if case.category is BenchCategory.PREFERENCE else "Correct answer"
    payload = (
        render_block("Question", case.question)
        + render_block(label, case.answer)
        + _extra_context(case)
        + render_block("Model response", answer)
    )
    return _grade(model, QA_PROMPTS[case.category], payload, case.category.value)


def judge_supersede(model: BaseChatModel, case: Case, answer: str) -> QaVerdict:
    """Grade the same answer again, on whether the retired value stays retired.

    Args:
        model: The grading model.
        case: A knowledge-update case, carrying the superseded evidence.
        answer: The model's reply.

    Returns:
        The verdict under the strict prompt.
    """
    payload = (
        render_block("Question", case.question)
        + render_block("Active fact (correct answer)", case.answer)
        + render_block("Superseded fact", "\n\n".join(case.superseded_evidence))
        + render_block("Model response", answer)
    )
    return _grade(
        model,
        SUPERSEDE_INTEGRITY_PROMPT,
        payload,
        BenchCategory.SUPERSEDE_INTEGRITY.value,
    )


def _extra_context(case: Case) -> str:
    """Render the fields only some categories carry."""
    blocks = ""
    if case.expected_procedure:
        blocks += render_block("Correct procedure", case.expected_procedure)
    if case.past_error:
        blocks += render_block("Past error", case.past_error)
    if case.past_correction:
        blocks += render_block("Correction that was made", case.past_correction)
    return blocks


def _grade(model: BaseChatModel, system: str, payload: str, judge: str) -> QaVerdict:
    """Run one judge, turning a failure into a recorded error."""
    try:
        verdict = ask_judge(model, _Verdict, system + _TAIL, payload)
    except JudgeError as exc:
        return QaVerdict(correct=False, judge=judge, error=str(exc))
    return QaVerdict(correct=verdict.correct, reasoning=verdict.reasoning, judge=judge)
