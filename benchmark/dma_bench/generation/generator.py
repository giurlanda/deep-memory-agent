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
exercised. The `medium` configuration keeps six months of that timeline for
under half the sessions per case: the cheapest shape that still has several
shards to route between.

Run it once and keep the output:

```bash
uv run --group benchmark python -m dma_bench.generation.generator \\
    --config small --out benchmark/data/operational_small.json
```

`--config` fixes every dimension of the corpus at once, which is the wrong knob
when only the size is in question — a five-case smoke run of the `large` shape
has no configuration of its own. `--cases-per-category` overrides that one
number and leaves the timeline and the session counts where the configuration
put them:

```bash
uv run --group benchmark python -m dma_bench.generation.generator \\
    --config large --cases-per-category 2 --out benchmark/data/trial.json
```

Every session is checked before it is kept, because the corpus is generated once
and reused by every run afterwards: a session that is wrong is wrong for the life
of the file. The structural pass in `validation` runs first and costs nothing,
then a second model judges what only reading the conversation reveals — see that
module for what each looks for. A rejected session is rewritten with the reason
it was rejected, up to `--max-session-retries` times; one that never holds up is
dropped when it was a distractor and abandons the case when it was evidence.

The validator defaults to the model that wrote the session, on the same provider,
and each of its flags falls back to the primary one:

```bash
uv run --group benchmark python -m dma_bench.generation.generator \
    --config small --out benchmark/data/operational_small.json \
    --model qwen2.5:3b --base-url http://localhost:11434/v1 \
    --validator-model gpt-4o --validator-base-url https://api.openai.com/v1
```

`--no-validate` keeps whatever the model returns, at the calls the run used to
cost.

Generation is slow and paid for by the call, so no case is ever generated
twice: each one is written to `--out` as soon as it is finished, and a run
pointed at a file that already exists resumes it — the cases already there are
kept, and only what is missing to reach the count is generated. An interrupted
run is restarted with the same command, and a corpus already on disk is grown
by asking for more cases:

```bash
uv run --group benchmark python -m dma_bench.generation.generator \\
    --config large --cases-per-category 12 --out benchmark/data/trial.json
```

Delete the file to start over.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

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
from dma_bench.generation.validation import (
    GeneratedSession,
    GeneratedTurn,
    strip_speaker_label,
    structural_problem,
    validate_session,
)
from dma_bench.schema import Case, Session, Turn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.language_models import BaseChatModel

__all__ = [
    "CORPUS_SHAPES",
    "CorpusShape",
    "calls_per_category",
    "generate_corpus",
    "load_corpus",
    "session_budget",
    "write_corpus",
]


class CorpusShape(BaseModel):
    """How big a generated corpus is and how far it spreads.

    Attributes:
        cases_per_category: How many cases to generate for each category. This
            is the one dimension `generate_corpus` will override, through its
            `cases_per_category` argument.
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
    "medium": CorpusShape(
        cases_per_category=10,
        evidence_sessions=2,
        distractor_sessions=20,
        span_days=180,
    ),
    "large": CorpusShape(
        cases_per_category=12,
        evidence_sessions=3,
        distractor_sessions=45,
        span_days=240,
    ),
}
"""The three fixed shapes.

`small` and `large` mirror the LongMemEval scales. `medium` sits between them:
six months of timeline, so monthly sharding still has several shards to route
between, at less than half of `large`'s sessions per case — which is where the
cost is.
"""

_SESSION_PROMPT = """\
Write one working conversation between a user and their AI assistant, as it
would actually have happened on {date}.

Context to work from:
{context}

Rules:
- {turns} turns in total, alternating user and assistant, starting with the user.
  Every turn carries its own role; never write the speaker's name into the text.
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

_RETRY_NOTE = """\
An earlier attempt at this conversation was rejected: {problem}
Write it again, without that problem.
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


class _GeneratedQuestion(BaseModel):
    """A question and its reference answer."""

    question: str = Field(description="The question, asked after every session.")
    answer: str = Field(description="The reference answer.")


def generate_corpus(
    model: BaseChatModel,
    shape: CorpusShape,
    *,
    categories: list[BenchCategory] | None = None,
    cases_per_category: int | None = None,
    validate: bool = True,
    validator: BaseChatModel | None = None,
    max_session_retries: int = 2,
    seed: int = 0,
    end_date: datetime | None = None,
    progress: bool = True,
    out: Path | None = None,
) -> list[Case]:
    """Generate a corpus of operational cases.

    Args:
        model: Model used to write the sessions and the questions.
        shape: How many cases, how many sessions, how long a timeline.
        categories: Categories to generate. Defaults to the two that LongMemEval
            cannot supply.
        cases_per_category: How many cases to generate per category, overriding
            the count `shape` carries. `None` keeps the configured value.
            Nothing else moves — sessions per case and the timeline span stay as
            the configuration set them — so a trial run and a full one differ
            only in how many cases they pay for.
        validate: Judge every generated session before keeping it —
            structurally, and on whether the material it had to carry actually
            came through. On by default: the corpus is generated once and reused
            for every run afterwards, so a session that is wrong is wrong for the
            life of the file. Turning it off restores the older behaviour of
            keeping whatever the model returned, at half the calls.
        validator: The model that does the judging. Defaults to `model` — the
            question is not whether a stronger model would have written the
            session better, but whether this one did what it was asked. Ignored
            when `validate` is false.
        max_session_retries: How many times a rejected session is rewritten, with
            the reason it was rejected fed back into the prompt. A session that
            never holds up is dropped when it was a distractor and abandons the
            case when it was evidence.
        seed: Makes the sampling of entities reproducible. Each case draws
            from its own stream, keyed by the seed together with the category
            and the index of the case, so a case gets the same client and the
            same procedure whether it was generated in one run or in a resumed
            one. The model's own output is not deterministic, which is why the
            corpus is written to disk once and reused rather than regenerated
            per run.
        end_date: The day the questions are asked. Defaults to today.
        progress: Show one `tqdm` bar per category. Each counts **model
            calls**, not cases: a `large` corpus is a few dozen cases but well
            over a thousand calls, and a bar that moves twelve times in an hour
            tells you nothing. Finished bars stay on screen, so a run ends with
            one line per category and what it cost. Set it to `False` to keep the
            run silent, which also avoids importing `tqdm` at all.
        out: Where to keep the corpus as it is generated. Every finished case
            is written out immediately, so a run that dies — or is killed
            because it was going to cost too much — leaves everything it had
            paid for behind. A file that already exists is resumed rather than
            overwritten: its cases are kept, counted per category, and only the
            ones missing to reach `cases_per_category` are generated, which is
            also how a corpus is grown after the fact. `None` keeps everything
            in memory and writes nothing.

    Returns:
        The corpus: the cases read back from `out`, if any, followed by the
        ones this run generated.

    Raises:
        ValueError: If `cases_per_category` is given and is not positive, or if
            `out` exists and is not a corpus.
    """
    shape = _override_cases(shape, cases_per_category)
    judge = (validator or model) if validate else None
    wanted = categories or [
        BenchCategory.PROCEDURAL_RETRIEVAL,
        BenchCategory.NON_REPETITION,
    ]
    asked_on = end_date or datetime.now(tz=UTC)
    cases = load_corpus(out) if out is not None and out.exists() else []

    for category in wanted:
        done = sum(1 for case in cases if case.category == category)
        missing = shape.cases_per_category - done
        if missing < 1:
            continue
        first = _next_index(cases, category)
        budget = session_budget(
            validate=judge is not None, max_retries=max_session_retries
        )
        with _progress(
            category.value,
            calls_per_category(shape, cases=missing, budget=budget),
            enabled=progress,
        ) as advance:
            for index in range(first, first + missing):
                case = _generate_case(
                    model,
                    category,
                    shape,
                    random.Random(f"{seed}:{category.value}:{index}"),
                    asked_on,
                    f"{category.value}-{index:02d}",
                    validator=judge,
                    max_session_retries=max_session_retries,
                    advance=advance,
                )
                if case is None:
                    continue
                cases.append(case)
                if out is not None:
                    write_corpus(cases, out)
    return cases


def session_budget(*, validate: bool = False, max_retries: int = 0) -> int:
    """Return the most model calls one session can cost.

    A session is written once and, when it is validated, judged once; a rejected
    one is written and judged again, up to the retry budget. The worst case is
    what the bars are sized from, so a run that has to retry never overruns its
    total.

    Args:
        validate: Whether each session is judged by a second model.
        max_retries: How many times a rejected session is rewritten.

    Returns:
        The number of model calls.
    """
    return (1 + max_retries) * (2 if validate else 1)


def calls_per_category(
    shape: CorpusShape, *, cases: int | None = None, budget: int = 1
) -> int:
    """Return how many model calls one category of this shape can make.

    One call per session — or, once validation and retries are in play, up to
    `budget` of them — plus one per case for the question. A failed call is still
    a call, and a session that settles under budget has the rest of its budget
    stepped through in one go, so this stays the exact total the bar reaches
    rather than a ceiling it stops short of.

    Args:
        shape: The corpus shape.
        cases: How many cases the count is for. Defaults to a full category;
            a resumed run passes the number it still has to generate, so the
            bar is sized for the work left rather than for the whole corpus.
        budget: The worst-case calls one session can cost, from
            `session_budget`. Defaults to one, the unvalidated single attempt.

    Returns:
        The number of model calls.
    """
    per_case = (shape.evidence_sessions + shape.distractor_sessions) * budget + 1
    return (shape.cases_per_category if cases is None else cases) * per_case


def _override_cases(shape: CorpusShape, cases_per_category: int | None) -> CorpusShape:
    """Return `shape` with its case count replaced, when one was asked for.

    The override is resolved once, at the top of `generate_corpus`, so that the
    count the cases are generated from and the count the progress bars are
    sized from can never disagree.
    """
    if cases_per_category is None:
        return shape
    if cases_per_category < 1:
        msg = f"cases_per_category must be positive, got {cases_per_category}"
        raise ValueError(msg)
    return shape.model_copy(update={"cases_per_category": cases_per_category})


class _Advance(Protocol):
    """Moves one category's progress bar along."""

    def __call__(self, detail: str, steps: int = 1) -> None:
        """Advance the bar by `steps`, labelling it with `detail`."""


@contextmanager
def _progress(label: str, total: int, *, enabled: bool) -> Iterator[_Advance]:
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
        yield lambda _detail, _steps=1: None
        return

    from tqdm.auto import tqdm

    with tqdm(total=total, unit="call", desc=label, file=sys.stdout) as bar:

        def advance(detail: str, steps: int = 1) -> None:
            bar.set_postfix_str(detail, refresh=False)
            bar.update(steps)

        yield advance


def write_corpus(cases: list[Case], path: Path) -> Path:
    """Write a corpus to disk.

    The file is written beside itself and renamed into place, because this is
    called after every case during a long run: a process killed mid-write must
    leave the corpus it already had, not half a JSON document that the next run
    would refuse to resume.

    Args:
        cases: The generated cases.
        path: Destination file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [case.model_dump(mode="json") for case in cases]
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return path


def load_corpus(path: Path) -> list[Case]:
    """Read a corpus back from disk.

    A corpus that cannot be read raises rather than being treated as empty:
    the alternative is a run that quietly overwrites hours of generation.

    Args:
        path: The corpus JSON.

    Returns:
        The cases it holds.

    Raises:
        ValueError: If the file is not a list of cases.
    """
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        # TRY004 asks for a TypeError here, but nobody passed a bad argument —
        # the file on disk is malformed, which is a ValueError like any other
        # corpus that fails to validate below.
        msg = f"{path} is not a corpus: expected a list of cases"
        raise ValueError(msg)  # noqa: TRY004
    return [Case.model_validate(record) for record in records]


def _next_index(cases: list[Case], category: BenchCategory) -> int:
    """Return the first case index of `category` that is free to use.

    Numbering continues from the highest id in the corpus rather than from how
    many cases are in it: a case abandoned by a failed model call is never
    written, so the ids on disk have holes in them, and counting would hand a
    resumed run an id one of the surviving cases already took.
    """
    prefix = f"{category.value}-"
    used = [
        int(suffix)
        for case in cases
        if case.question_id.startswith(prefix)
        and (suffix := case.question_id[len(prefix) :]).isdigit()
    ]
    return max(used) + 1 if used else 0


def _generate_case(
    model: BaseChatModel,
    category: BenchCategory,
    shape: CorpusShape,
    rng: random.Random,
    asked_on: datetime,
    case_id: str,
    *,
    validator: BaseChatModel | None,
    max_session_retries: int,
    advance: _Advance,
) -> Case | None:
    """Generate one case, or `None` when the model failed to produce one.

    `advance` is stepped through the whole per-session budget on every path,
    including the ones that give up, so the bar always reaches its total.

    A distractor session that never held up is dropped: it was noise, and one
    fewer changes nothing. An evidence session that never held up abandons the
    case. The question is written from the evidence, so a case built on the
    evidence that survived asks about material the corpus may no longer carry —
    which is worse than not having the case at all.
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

    budget = session_budget(
        validate=validator is not None, max_retries=max_session_retries
    )
    sessions: list[Session] = []
    for index, date in enumerate(dates):
        is_evidence = date in evidence_dates
        context, payload = (
            _evidence_material(category, client, subject, index)
            if is_evidence
            else _distractor_material(rng)
        )
        session_id = f"{case_id}-s{index:02d}"
        turns = _write_session(
            model,
            validator,
            date,
            context,
            payload,
            rng,
            max_retries=max_session_retries,
            advance=advance,
            detail=session_id,
        )
        if turns is None:
            if is_evidence:
                advance(f"{case_id} abandoned", _remaining(shape, index, budget))
                return None
            continue
        sessions.append(
            Session(
                session_id=session_id,
                date=date,
                turns=[
                    Turn(
                        turn_id=f"{session_id}#{position}",
                        role=turn.role,
                        content=turn.content,
                        has_answer=is_evidence and turn.role == "user",
                    )
                    for position, turn in enumerate(turns)
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


def _remaining(shape: CorpusShape, index: int, budget: int) -> int:
    """Return the calls a case abandoned after session `index` will never make.

    The sessions after this one, at their full budget, plus the question. Stepped
    through in one go so the bar still lands on the total it was sized from.
    """
    left = shape.evidence_sessions + shape.distractor_sessions - index - 1
    return left * budget + 1


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
    validator: BaseChatModel | None,
    date: datetime,
    context: str,
    payload: str,
    rng: random.Random,
    *,
    max_retries: int,
    advance: _Advance,
    detail: str,
) -> list[GeneratedTurn] | None:
    """Write one conversation and check it, or return `None` if it never held up.

    Each attempt is written, stripped of any speaker labels that leaked into the
    text, checked structurally, and — when a validator was given — judged by it.
    The reason an attempt was rejected goes into the next prompt, so a retry is
    told what to fix instead of resampling blind.

    `advance` is stepped once per model call, then once more at the end for
    whatever the session did not spend, so the bar lands on the budget it was
    sized from however many attempts this took.

    Args:
        model: The model that writes the conversation.
        validator: The model that judges it, or `None` to skip validation.
        date: The day the session happens on.
        context: The account and the situation.
        payload: What the conversation has to carry.
        rng: Draws the turn count.
        max_retries: How many times a rejected session is rewritten.
        advance: Moves the progress bar.
        detail: What to label the bar with.

    Returns:
        The turns, or `None` when every attempt was rejected or failed.
    """
    budget = session_budget(validate=validator is not None, max_retries=max_retries)
    prompt = _SESSION_PROMPT.format(
        date=f"{date:%Y-%m-%d}",
        context=context,
        payload=payload,
        turns=rng.choice((6, 8, 10)),
    )
    spent = 0
    problem: str | None = None
    turns: list[GeneratedTurn] | None = None

    for _ in range(1 + max_retries):
        attempt = prompt
        if problem is not None:
            attempt = f"{prompt}\n{_RETRY_NOTE.format(problem=problem)}"
        try:
            generated = model.with_structured_output(GeneratedSession).invoke(attempt)
        except Exception as exc:
            print(f"  session generation failed: {exc!r}", file=sys.stderr)
            spent += 1
            advance(detail)
            break
        spent += 1
        advance(detail)

        turns = [
            GeneratedTurn(role=turn.role, content=strip_speaker_label(turn.content))
            for turn in getattr(generated, "turns", None) or []
        ]
        problem = structural_problem(turns)
        if problem is None and validator is not None:
            problem = validate_session(validator, turns, payload)
            spent += 1
            advance(detail)
        if problem is None:
            break
        print(f"  {detail} rejected: {problem}", file=sys.stderr)
        turns = None

    advance(detail, budget - spent)
    return turns


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


def _positive_int(text: str) -> int:
    """Parse a count from the command line, rejecting zero and negatives."""
    value = int(text)
    if value < 1:
        msg = f"expected a positive integer, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _non_negative_int(text: str) -> int:
    """Parse a retry budget from the command line, rejecting negatives.

    Zero is meaningful here and positive is not the floor: it asks for one
    attempt per session, validated but never rewritten.
    """
    value = int(text)
    if value < 0:
        msg = f"expected a non-negative integer, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CORPUS_SHAPES), default="small")
    parser.add_argument(
        "--cases-per-category",
        type=_positive_int,
        default=None,
        help="how many cases per category, overriding what --config fixes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to keep the corpus; an existing file is resumed, not overwritten",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="keep every session the model returns, without judging it",
    )
    parser.add_argument(
        "--max-session-retries",
        type=_non_negative_int,
        default=2,
        help="how many times a rejected session is rewritten before it is dropped",
    )
    parser.add_argument(
        "--validator-model",
        default=None,
        help="model that judges each session; defaults to --model",
    )
    parser.add_argument(
        "--validator-base-url",
        default=None,
        help="provider for the validator; defaults to --base-url",
    )
    parser.add_argument(
        "--validator-api-key",
        default=None,
        help="api key for the validator; defaults to --api-key",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the progress bar, for non-interactive runs",
    )
    return parser


def _build_validator(
    args: argparse.Namespace, factory: type[BaseChatModel]
) -> BaseChatModel:
    """Build the validating model from the arguments, falling back to the writer's.

    Each of the three validator arguments defaults to the corresponding one of
    the primary model, so judging with the same model on the same endpoint — the
    common case — needs no flags at all, and judging with a stronger model behind
    a different endpoint needs only the flags that actually differ.

    Args:
        args: The parsed command line.
        factory: The chat-model class to build with.

    Returns:
        The validating model. Temperature is zero: this is a judgement, and the
        same conversation should not be accepted on one run and rejected on the
        next.
    """
    return factory(
        model=args.validator_model or args.model,
        base_url=args.validator_base_url or args.base_url,
        api_key=args.validator_api_key or args.api_key,
        temperature=0,
        timeout=240,
        max_retries=2,
    )


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
    validator = _build_validator(args, ChatOpenAI) if args.validate else None
    shape = CORPUS_SHAPES[args.config]
    per_category = args.cases_per_category or shape.cases_per_category
    if args.out.exists():
        print(f"resuming {args.out}: {len(load_corpus(args.out))} cases already there")
    print(
        f"generating {args.config} corpus with {args.model}, "
        f"{per_category} cases per category…"
    )
    cases = generate_corpus(
        model,
        shape,
        cases_per_category=args.cases_per_category,
        validate=args.validate,
        validator=validator,
        max_session_retries=args.max_session_retries,
        seed=args.seed,
        progress=not args.quiet,
        out=args.out,
    )
    path = write_corpus(cases, args.out)
    sessions = sum(len(case.sessions) for case in cases)
    print(f"wrote {len(cases)} cases / {sessions} sessions to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
