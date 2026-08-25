r"""Building the operational corpus from the ontology.

The method is LongMemEval's, with one change that matters. They generate
evidence by self-chat between two models and instruct the speaker to mention the
fact *incidentally* rather than state it, so that a system cannot score by
lexical pattern-matching. That instruction is kept here and is arguably more
important, because the step it protects is different: if a session says "FACT:
Acme is on Enterprise", the manager agent has nothing to extract and
`memory_consolidate` is never tested at all. So sessions are written as working
narrative — "the renewal call confirmed they'd moved up to Enterprise after all"
— and the extraction has to do real work.

Timelines are ours to choose here, which is the other reason this corpus exists.
LongMemEval's sessions are packed into about ten days, so monthly sharding never
has more than a shard or two to route between. The `large` configuration spreads
its sessions across roughly eight months, which is where shard routing,
accumulated supersessions and repeated consolidation passes actually get
exercised.

Run it once and keep the output:

```bash
uv run --group benchmark python -m dma_bench.generation.generator \\
    --config small --out benchmark/data/operational_small.json
```
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from dma_bench.categories import BenchCategory
from dma_bench.generation.ontology import (
    CLIENTS,
    ERRORS,
    FEEDBACK_THEMES,
    PLANS,
    PROCEDURES,
    PROJECT_EVENTS,
    STACKS,
)
from dma_bench.schema import Case, Session, Turn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from langchain_core.language_models import BaseChatModel

__all__ = [
    "CORPUS_SHAPES",
    "CorpusShape",
    "calls_per_category",
    "generate_corpus",
    "write_corpus",
]


class CorpusShape(BaseModel):
    """How big a generated corpus is and how far it spreads.

    Attributes:
        cases_per_category: How many cases to generate for each category.
        evidence_sessions: Sessions that carry the answer.
        distractor_sessions: Sessions about other accounts and projects.
        span_days: How far apart the first and last session sit. This is the
            knob LongMemEval cannot offer, and the reason a long-timeline
            configuration exists here at all.
    """

    cases_per_category: int = 6
    evidence_sessions: int = 2
    distractor_sessions: int = 4
    span_days: int = 30


CORPUS_SHAPES: dict[str, CorpusShape] = {
    "small": CorpusShape(
        cases_per_category=6,
        evidence_sessions=2,
        distractor_sessions=4,
        span_days=30,
    ),
    "large": CorpusShape(
        cases_per_category=12,
        evidence_sessions=3,
        distractor_sessions=45,
        span_days=240,
    ),
}
"""The two fixed shapes, mirroring the `small` and `large` LongMemEval scales."""

_SESSION_PROMPT = """\
Write one working conversation between a user and their AI assistant, as it
would actually have happened on {date}.

Context to work from:
{context}

Rules:
- {turns} turns in total, alternating user and assistant, starting with the user.
- The information listed under "must come through" has to be present, but
  mentioned **in passing, as part of doing the work** — never announced as a
  fact. Someone reading the conversation should be able to infer it; nobody
  should be able to point at a sentence that states it like a database row.
- Everything else should read as ordinary working chatter: scheduling, small
  decisions, half-finished threads.
- No dates in the text unless they are part of the information that must come
  through. Never mention that this is an example or a test.

Must come through:
{payload}
"""

_QUESTION_PROMPT = """\
Write the question that tests whether an assistant remembers what came out of
the sessions below, and the reference answer.

The question is asked on {date}, after all of them. It must be answerable only
from what the sessions carry — not from general knowledge — and it must not
quote their wording.

{intent}

Sessions:
{sessions}
"""

_INTENTS: dict[BenchCategory, str] = {
    BenchCategory.PROCEDURAL_RETRIEVAL: (
        "Write the question as a live situation in which the procedure's trigger "
        "condition has just been met, asking what to do now. Do not name the "
        "procedure. The reference answer is the sequence of steps that should be "
        "followed."
    ),
    BenchCategory.NON_REPETITION: (
        "Write the question as a new situation in which the same mistake would be "
        "the natural thing to do again, asking how to proceed. Do not mention "
        "that a mistake was made before. The reference answer states the "
        "correction that must be applied and the outcome it avoids."
    ),
}


class _GeneratedSession(BaseModel):
    """A conversation as the generating model returns it."""

    turns: list[str] = Field(
        default_factory=list,
        description="Turns in order, alternating, starting with the user.",
    )


class _GeneratedQuestion(BaseModel):
    """A question and its reference answer."""

    question: str = Field(description="The question, asked after every session.")
    answer: str = Field(description="The reference answer.")


def generate_corpus(
    model: BaseChatModel,
    shape: CorpusShape,
    *,
    categories: list[BenchCategory] | None = None,
    seed: int = 0,
    end_date: datetime | None = None,
    progress: bool = True,
) -> list[Case]:
    """Generate a corpus of operational cases.

    Args:
        model: Model used to write the sessions and the questions.
        shape: How many cases, how many sessions, how long a timeline.
        categories: Categories to generate. Defaults to the two that LongMemEval
            cannot supply.
        seed: Makes the sampling of entities reproducible. The model's own
            output is not deterministic, which is why the corpus is written to
            disk once and reused rather than regenerated per run.
        end_date: The day the questions are asked. Defaults to today.
        progress: Show one `tqdm` bar per category. Each counts **model
            calls**, not cases: a `large` corpus is a few dozen cases but well
            over a thousand calls, and a bar that moves twelve times in an hour
            tells you nothing. Finished bars stay on screen, so a run ends with
            one line per category and what it cost. Set it to `False` to keep the
            run silent, which also avoids importing `tqdm` at all.

    Returns:
        The generated cases.
    """
    wanted = categories or [
        BenchCategory.PROCEDURAL_RETRIEVAL,
        BenchCategory.NON_REPETITION,
    ]
    rng = random.Random(seed)
    asked_on = end_date or datetime.now(tz=UTC)
    cases: list[Case] = []

    for category in wanted:
        with _progress(
            category.value, calls_per_category(shape), enabled=progress
        ) as advance:
            for index in range(shape.cases_per_category):
                case = _generate_case(
                    model,
                    category,
                    shape,
                    rng,
                    asked_on,
                    f"{category.value}-{index:02d}",
                    advance=advance,
                )
                if case is not None:
                    cases.append(case)
    return cases


def calls_per_category(shape: CorpusShape) -> int:
    """Return how many model calls one category of this shape will make.

    One call per session, plus one per case for the question. A failed call is
    still a call, so this is the exact total rather than an upper bound — which
    is what lets every bar actually reach the end.

    Args:
        shape: The corpus shape.

    Returns:
        The number of model calls.
    """
    per_case = shape.evidence_sessions + shape.distractor_sessions + 1
    return shape.cases_per_category * per_case


@contextmanager
def _progress(
    label: str, total: int, *, enabled: bool
) -> Iterator[Callable[[str], None]]:
    """Yield a callable that advances one category's progress bar.

    One bar per category, left on screen when it finishes, so a run ends with a
    readable line per category instead of a single bar that hides which half of
    the corpus was slow.

    The bar goes to stdout while the failure notices go to stderr, so a session
    that fails mid-run prints cleanly instead of being overwritten by the next
    redraw. `tqdm` is imported here rather than at module scope: it lives in the
    `benchmark` dependency group, which CI does not install, and the tests that
    import this module have to keep working without it.
    """
    if not enabled:
        yield lambda _label: None
        return

    from tqdm.auto import tqdm

    with tqdm(total=total, unit="call", desc=label, file=sys.stdout) as bar:

        def advance(detail: str) -> None:
            bar.set_postfix_str(detail, refresh=False)
            bar.update(1)

        yield advance


def write_corpus(cases: list[Case], path: Path) -> Path:
    """Write a corpus to disk.

    Args:
        cases: The generated cases.
        path: Destination file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [case.model_dump(mode="json") for case in cases]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def _generate_case(
    model: BaseChatModel,
    category: BenchCategory,
    shape: CorpusShape,
    rng: random.Random,
    asked_on: datetime,
    case_id: str,
    *,
    advance: Callable[[str], None],
) -> Case | None:
    """Generate one case, or `None` when the model failed to produce one.

    `advance` is called once per model call on every path, including the ones
    that give up, so the bar always reaches its total.
    """
    client = rng.choice(CLIENTS)
    subject = (
        rng.choice(PROCEDURES)
        if category is BenchCategory.PROCEDURAL_RETRIEVAL
        else rng.choice(ERRORS)
    )
    dates = _timeline(shape, asked_on, rng)
    evidence_dates = sorted(
        rng.sample(dates, k=min(shape.evidence_sessions, len(dates)))
    )

    sessions: list[Session] = []
    for index, date in enumerate(dates):
        is_evidence = date in evidence_dates
        context, payload = (
            _evidence_material(category, client, subject, index)
            if is_evidence
            else _distractor_material(rng)
        )
        turns = _write_session(model, date, context, payload, rng)
        advance(f"{case_id} s{index:02d}")
        if turns is None:
            continue
        sessions.append(
            Session(
                session_id=f"{case_id}-s{index:02d}",
                date=date,
                turns=[
                    Turn(
                        turn_id=f"{case_id}-s{index:02d}#{position}",
                        role="user" if position % 2 == 0 else "assistant",
                        content=text,
                        has_answer=is_evidence and position % 2 == 0,
                    )
                    for position, text in enumerate(turns)
                ],
                is_evidence=is_evidence,
            )
        )

    evidence = [session for session in sessions if session.is_evidence]
    if not evidence:
        advance(f"{case_id} abandoned")
        return None

    question = _write_question(model, category, evidence, asked_on)
    advance(f"{case_id} question")
    if question is None:
        return None

    return Case(
        question_id=case_id,
        category=category,
        question=question.question,
        answer=question.answer,
        question_date=asked_on,
        sessions=sessions,
        source="operational",
        expected_procedure=(
            f"{subject['title']}: {subject['steps']}"
            if category is BenchCategory.PROCEDURAL_RETRIEVAL
            else None
        ),
        past_error=(
            f"{subject['mistake']} — {subject['consequence']}"
            if category is BenchCategory.NON_REPETITION
            else None
        ),
        past_correction=(
            subject["correction"] if category is BenchCategory.NON_REPETITION else None
        ),
    )


def _timeline(
    shape: CorpusShape, asked_on: datetime, rng: random.Random
) -> list[datetime]:
    """Spread the sessions over the configured span, oldest first."""
    total = shape.evidence_sessions + shape.distractor_sessions
    start = asked_on - timedelta(days=shape.span_days)
    offsets = sorted(rng.uniform(0, shape.span_days - 1) for _ in range(total))
    return [
        start + timedelta(days=offset, hours=rng.uniform(8, 19)) for offset in offsets
    ]


def _evidence_material(
    category: BenchCategory,
    client: dict[str, str],
    subject: dict[str, str],
    index: int,
) -> tuple[str, str]:
    """Return the context and the payload of an evidence session."""
    context = (
        f"The account is {client['name']}, in {client['sector']}. "
        f"They run {STACKS[index % len(STACKS)]}. "
        f"The session happens around {PROJECT_EVENTS[index % len(PROJECT_EVENTS)]}."
    )
    if category is BenchCategory.PROCEDURAL_RETRIEVAL:
        payload = (
            f"The team hits the situation where {subject['trigger']}, and works "
            f"through it in this order: {subject['steps']}. The order is what "
            f"matters and it has to be recoverable from the conversation."
        )
    else:
        payload = (
            f"Someone {subject['mistake']}. The result: {subject['consequence']}. "
            f"By the end of the conversation the fix is agreed: "
            f"{subject['correction']}."
        )
    return context, payload


def _distractor_material(rng: random.Random) -> tuple[str, str]:
    """Return the context and payload of a session that carries no answer."""
    client = rng.choice(CLIENTS)
    context = (
        f"The account is {client['name']}, in {client['sector']}. "
        f"They run {rng.choice(STACKS)}. "
        f"The session happens around {rng.choice(PROJECT_EVENTS)}."
    )
    payload = (
        f"Routine work on this account: a plan sitting at {rng.choice(PLANS)}, "
        f"scheduling, and the fact that the client {rng.choice(FEEDBACK_THEMES)}. "
        f"Nothing about deploy rollbacks, restores, onboarding, escalation, "
        f"release notes, security questionnaires, or any past mistake."
    )
    return context, payload


def _write_session(
    model: BaseChatModel,
    date: datetime,
    context: str,
    payload: str,
    rng: random.Random,
) -> list[str] | None:
    """Ask the model for one conversation, or `None` if it failed."""
    prompt = _SESSION_PROMPT.format(
        date=f"{date:%Y-%m-%d}",
        context=context,
        payload=payload,
        turns=rng.choice((6, 8, 10)),
    )
    try:
        generated = model.with_structured_output(_GeneratedSession).invoke(prompt)
    except Exception as exc:
        print(f"  session generation failed: {exc!r}", file=sys.stderr)
        return None
    turns = getattr(generated, "turns", None) or []
    return [turn for turn in turns if turn.strip()] or None


def _write_question(
    model: BaseChatModel,
    category: BenchCategory,
    evidence: list[Session],
    asked_on: datetime,
) -> _GeneratedQuestion | None:
    """Ask the model for the question and its reference answer."""
    rendered = "\n\n".join(
        f"### {session.date:%Y-%m-%d}\n"
        + "\n".join(f"[{turn.role}] {turn.content}" for turn in session.turns)
        for session in evidence
    )
    prompt = _QUESTION_PROMPT.format(
        date=f"{asked_on:%Y-%m-%d}", intent=_INTENTS[category], sessions=rendered
    )
    try:
        return model.with_structured_output(_GeneratedQuestion).invoke(prompt)
    except Exception as exc:
        print(f"  question generation failed: {exc!r}", file=sys.stderr)
        return None


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CORPUS_SHAPES), default="small")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the progress bar, for non-interactive runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate a corpus from the command line.

    Args:
        argv: Arguments to parse. Defaults to `sys.argv`.

    Returns:
        Process exit code.
    """
    from langchain_openai import ChatOpenAI

    args = _build_parser().parse_args(argv)
    model = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.7,
        timeout=240,
        max_retries=2,
    )
    shape = CORPUS_SHAPES[args.config]
    print(f"generating {args.config} corpus with {args.model}…")
    cases = generate_corpus(model, shape, seed=args.seed, progress=not args.quiet)
    path = write_corpus(cases, args.out)
    sessions = sum(len(case.sessions) for case in cases)
    print(f"wrote {len(cases)} cases / {sessions} sessions to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
