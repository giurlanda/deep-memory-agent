"""What a generated session has to look like before it enters the corpus.

The generator used to trust the model's turn list on two counts at once: that it
alternated, and that it said what the prompt asked it to say. Both assumptions
fail. A model that emits two assistant messages in a row, collapses both speakers
into one string, or answers with a placeholder produces a session that looks
structurally fine and is silently mislabelled from that point on — and, because
`has_answer` was derived from the same alternation, hands the retrieval judge a
gold set pointing at the wrong turns.

So the shape of a session lives here, together with the two things that decide
whether one is usable:

- `structural_problem` — the deterministic pass. Alternation, speaker labels that
  leaked into the text, empty or placeholder turns, two speakers in one string.
  It costs nothing, so it runs first and most rejections never reach the model.
- `validate_session` — the model pass, for what only reading the conversation
  reveals: a session written entirely from the user's point of view passes every
  structural check, and an evidence session that never lets its payload through
  is worthless to the judges while looking perfectly well-formed.

Both return the reason a session was rejected, or `None` when it is fine. The
reason is not just for the log: the generator feeds it back into the retry, so a
second attempt is told what was wrong with the first.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "GeneratedSession",
    "GeneratedTurn",
    "SessionVerdict",
    "strip_speaker_label",
    "structural_problem",
    "validate_session",
]

_SPEAKER_LABEL = re.compile(r"^\s*([A-Za-z_][A-Za-z_ ]{0,19}):[ \t]+")
"""A speaker label at the head of a turn.

Deliberately wider than `user:` and `assistant:`. Small models sign their turns
with whatever they feel like — `Me:`, `AI:`, `U:`/`A:`, `You:`, `Client:`, and
the account name itself — and every one of those ends up rendered twice once
`ingest` prefixes the turn with its own role.
"""

_SCHEMA_LEAK = re.compile(r"^\s*type:\s*(?:user|assistant)\b", re.IGNORECASE)
"""The response schema written out as prose, e.g. `type: user\\ncontent: "…"`."""

_EMBEDDED_SPEAKER = re.compile(r"\n\s*(?:user|assistant)\s*:", re.IGNORECASE)
"""A second speaker starting mid-turn: both sides collapsed into one string."""

_MIN_CONTENT = 12
"""Shortest believable turn, in characters.

Set to catch the degenerate answers — a turn whose entire content is `user`,
`assistant` or `...` — rather than to police terseness. Real closing turns
("Perfect, that's everything. Thanks.") sit comfortably above it.
"""

_VALIDATOR_PROMPT = """\
Below is a conversation between a user and their AI assistant, generated for a
benchmark corpus. Judge whether it is usable, on two counts.

Structure:
- It reads as a genuine two-party exchange, not as one person narrating both
  sides and not as a monologue.
- Every turn is attributed to the speaker who plausibly said it.
- No turn is empty, truncated, a placeholder, or a stub like "..." .
- No turn carries a speaker label ("User:", "Assistant:", "Me:") in its text,
  and no turn contains two speakers at once.

Coverage — the conversation had to carry this:
{payload}

It has to be *inferable* from what is said, as something that came up while the
work was being done. It must not be announced as a fact, and a reader must not
be able to point at one sentence that states it outright. A conversation that
never lets it through is not usable; neither is one that states it flatly.

Reject on the first real problem and say which turn it is in, in one sentence,
so the conversation can be rewritten. Do not reject for style, length, or for
leaving threads unfinished — working chatter is supposed to look like that.

Conversation:
{conversation}
"""


class GeneratedTurn(BaseModel):
    """One turn, as the generating model returns it.

    The role is the model's own answer rather than something derived from the
    turn's position, which is the whole point: position is a guess that goes
    wrong silently, and took `has_answer` with it.

    Attributes:
        role: Who spoke.
        content: What they said, with no speaker label.
    """

    role: Literal["user", "assistant"]
    content: str = Field(description="The message text, with no speaker label.")


class GeneratedSession(BaseModel):
    """A conversation as the generating model returns it.

    Attributes:
        turns: The turns in order, starting with the user.
    """

    turns: list[GeneratedTurn] = Field(
        default_factory=list,
        description="Turns in order, starting with the user and alternating.",
    )


class SessionVerdict(BaseModel):
    """What the validating model decided about one session.

    Attributes:
        usable: Whether the session can go into the corpus.
        reason: Why it was rejected. Empty when it was not.
    """

    usable: bool = Field(description="True if the conversation is usable as it is.")
    reason: str = Field(
        default="",
        description="One sentence naming the problem. Empty when usable.",
    )


def strip_speaker_label(text: str) -> str:
    """Return `text` without a speaker label at its head.

    Args:
        text: A turn's content, as the model wrote it.

    Returns:
        The content with one leading label removed, trimmed.
    """
    return _SPEAKER_LABEL.sub("", text, count=1).strip()


def structural_problem(turns: list[GeneratedTurn]) -> str | None:
    """Return why `turns` are unusable, or `None` when they are fine.

    Runs before the model is asked anything, because most of what goes wrong is
    visible without reading the conversation, and a rejection here costs nothing.

    Args:
        turns: The turns the model returned, already stripped of their labels.

    Returns:
        A sentence naming the first problem found, or `None`.
    """
    if not turns:
        return "the conversation came back empty"
    if turns[0].role != "user":
        return "the conversation has to start with the user"
    for position, turn in enumerate(turns):
        expected = "user" if position % 2 == 0 else "assistant"
        if turn.role != expected:
            return (
                f"turn {position} is attributed to the {turn.role} where the "
                f"conversation had reached the {expected}; the turns have to "
                f"alternate"
            )
        if faulty := _turn_problem(turn.content):
            return f"turn {position} {faulty}"
    return None


def _turn_problem(content: str) -> str | None:
    """Return what is wrong with one turn's text, or `None`."""
    if _SCHEMA_LEAK.match(content):
        return "has the response schema written out as text"
    if _EMBEDDED_SPEAKER.search(content):
        return "holds both speakers instead of one"
    if len(content.strip()) < _MIN_CONTENT:
        return "is empty or a placeholder"
    return None


def validate_session(
    model: BaseChatModel,
    turns: list[GeneratedTurn],
    payload: str,
) -> str | None:
    """Ask a second model whether `turns` are usable.

    A model that cannot be reached is not a verdict, so the session is let
    through rather than rejected: an unreachable validator would otherwise burn
    the whole retry budget of every session and abandon the run one case at a
    time.

    Args:
        model: The validating model. Defaults, at the call site, to the one that
            wrote the session.
        turns: The turns to judge, already past `structural_problem`.
        payload: What the session had to carry, in the same words the writing
            prompt used.

    Returns:
        The reason the session was rejected, or `None` when it passed.
    """
    prompt = _VALIDATOR_PROMPT.format(
        payload=payload,
        conversation="\n".join(f"[{turn.role}] {turn.content}" for turn in turns),
    )
    try:
        verdict = model.with_structured_output(SessionVerdict).invoke(prompt)
    except Exception as exc:
        print(f"  session validation failed: {exc!r}", file=sys.stderr)
        return None
    if getattr(verdict, "usable", True):
        return None
    return getattr(verdict, "reason", "") or "the validator rejected it"
