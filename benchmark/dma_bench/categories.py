"""Question categories and how LongMemEval's map onto this architecture.

LongMemEval names its categories after *where the evidence sits* in a chat log
— one session, many sessions. This package names them after *what the memory
system has to do*: one hop through semantic memory, several hops across linked
entries, a point-in-time query, a supersession that has to hold. The mapping is
one-to-one for six of them, which is why the datasets transfer at all.

Three categories have no LongMemEval counterpart, because LongMemEval has no
agent that writes its own memory: `procedural-retrieval`, `non-repetition`, and
`supersede-integrity` — the last being a stricter second reading of the
knowledge-update cases rather than a separate set of questions.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "GENERATED_CATEGORIES",
    "LONGMEMEVAL_CATEGORY",
    "STRICT_RETRIEVAL_CATEGORIES",
    "BenchCategory",
]


class BenchCategory(StrEnum):
    """What a question asks the memory system to do."""

    SINGLE_HOP = "single-hop"
    """One fact, reachable from a single semantic entry."""

    PREFERENCE = "preference"
    """A stated working preference, recalled and acted on."""

    MULTI_HOP = "multi-hop"
    """An answer that has to be aggregated across linked entries."""

    KNOWLEDGE_UPDATE = "knowledge-update"
    """A fact that changed; the current value is what is asked for."""

    SUPERSEDE_INTEGRITY = "supersede-integrity"
    """The same cases as `KNOWLEDGE_UPDATE`, judged on whether the retired
    value stays retired. Scored separately because a memory tree can hold the
    new fact and still leak the old one as if both were true."""

    TEMPORAL_REASONING = "temporal-reasoning"
    """A point-in-time query: what held when, or how long between two events."""

    ABSTENTION = "abstention"
    """Something never recorded. The correct answer is to say so."""

    PROCEDURAL_RETRIEVAL = "procedural-retrieval"
    """A trigger condition is met and the matching procedure has to be applied."""

    NON_REPETITION = "non-repetition"
    """A past mistake, corrected once, must not come back in a similar setting."""


LONGMEMEVAL_CATEGORY: dict[str, BenchCategory] = {
    "single-session-user": BenchCategory.SINGLE_HOP,
    "single-session-assistant": BenchCategory.SINGLE_HOP,
    "single-session-preference": BenchCategory.PREFERENCE,
    "multi-session": BenchCategory.MULTI_HOP,
    "knowledge-update": BenchCategory.KNOWLEDGE_UPDATE,
    "temporal-reasoning": BenchCategory.TEMPORAL_REASONING,
}
"""LongMemEval's `question_type` to ours.

Abstention is not in this table: LongMemEval marks it on the `question_id`
(`..._abs`) rather than the type, and it can decorate any of them.
"""

GENERATED_CATEGORIES = frozenset(
    {BenchCategory.PROCEDURAL_RETRIEVAL, BenchCategory.NON_REPETITION}
)
"""Categories that only the generated operational corpus can supply."""

STRICT_RETRIEVAL_CATEGORIES = frozenset(
    {BenchCategory.KNOWLEDGE_UPDATE, BenchCategory.SUPERSEDE_INTEGRITY}
)
"""Categories where retrieval must also get the *validity* right.

LongMemEval's strict criterion asks that both the old and the new value be in
the retrieved pool. Ours adds the half that only matters once an agent curates
its own memory: the retired value must not come back labelled as current.
"""
