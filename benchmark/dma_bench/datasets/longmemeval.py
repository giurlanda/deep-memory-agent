"""Adapter from the published LongMemEval datasets to `Case`.

Three files ship with the paper and all three have the same shape — what
changes is how much hay surrounds the needle:

| file | sessions per question | span | role here |
| --- | --- | --- | --- |
| `longmemeval_oracle.json` | ~2 | 1 month | the `small` scale: the pipeline cold |
| `longmemeval_s_cleaned.json` | ~48 | ~10 days | the `large` scale |
| `longmemeval_m_cleaned.json` | ~480 | ~10 days | 5 MB of history per question |

Timestamps are left exactly as they are. Stretching them over six months would
give the sharding more to do, but the questions quote absolute dates inside the
message text ("my first service on March 15th"), so a rescaled timeline would
contradict its own evidence. The generated operational corpus is where a long
timeline is built on purpose instead.

The `m` file is 2.7 GB, so records are streamed one at a time rather than loaded
as a list; sampling is a reservoir per category, which keeps only the cases that
are actually going to be run.
"""

from __future__ import annotations

import json
import random
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dma_bench.categories import LONGMEMEVAL_CATEGORY, BenchCategory
from dma_bench.schema import Case, Session, Turn

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

__all__ = [
    "DATASET_FILES",
    "SCALE_DATASETS",
    "dataset_path",
    "iter_cases",
    "load_cases",
    "parse_timestamp",
    "to_case",
]

DATASET_FILES: dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
"""Which file backs each LongMemEval variant."""

SCALE_DATASETS: dict[str, str] = {"small": "oracle", "large": "s"}
"""The two fixed scales.

`small` is the evidence sessions alone: a couple of sessions, one shard, nothing
consolidated yet — the cold pipeline. `large` adds the distractors. `m` stays
available by name but is not wired to a scale: at ~480 sessions per question it
is a different order of spend.
"""

_TIMESTAMP = re.compile(r"(\d{4})/(\d{2})/(\d{2})[^\d]*(\d{2}):(\d{2})")
_CHUNK_CHARS = 1 << 22


def dataset_path(root: Path, variant: str) -> Path:
    """Return the file backing a LongMemEval variant.

    Args:
        root: Directory holding the downloaded datasets.
        variant: One of `oracle`, `s`, `m`, or a scale name.

    Returns:
        Path to the JSON file.

    Raises:
        ValueError: If the variant is unknown.
    """
    resolved = SCALE_DATASETS.get(variant, variant)
    if resolved not in DATASET_FILES:
        msg = f"unknown LongMemEval variant {variant!r}"
        raise ValueError(msg)
    return root / DATASET_FILES[resolved]


def parse_timestamp(text: str) -> datetime:
    """Parse a LongMemEval timestamp such as `"2023/04/10 (Mon) 23:07"`.

    The weekday in the middle is decorative and locale-dependent, so it is
    matched and thrown away rather than parsed with `%a`.

    Args:
        text: The timestamp as it appears in the dataset.

    Returns:
        A timezone-aware datetime in UTC.

    Raises:
        ValueError: If no timestamp can be read out of the text.
    """
    match = _TIMESTAMP.search(text)
    if match is None:
        msg = f"cannot read a timestamp out of {text!r}"
        raise ValueError(msg)
    year, month, day, hour, minute = (int(part) for part in match.groups())
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def to_case(record: dict) -> Case:
    """Convert one LongMemEval record into a `Case`.

    Sessions are sorted chronologically, which is the order they will be
    replayed in. Abstention is read off the `question_id` suffix, since
    LongMemEval marks it there rather than on `question_type`, and it can
    decorate any type.

    Args:
        record: A raw dataset record.

    Returns:
        The converted case.
    """
    question_id = str(record["question_id"])
    is_abstention = question_id.endswith("_abs")
    answer_sessions = set(record.get("answer_session_ids") or ())

    sessions: list[Session] = []
    for session_id, turns, date in zip(
        record["haystack_session_ids"],
        record["haystack_sessions"],
        record["haystack_dates"],
        strict=True,
    ):
        parsed = [
            Turn(
                turn_id=f"{session_id}#{index}",
                role=turn.get("role", "user"),
                content=turn.get("content", ""),
                has_answer=bool(turn.get("has_answer")),
            )
            for index, turn in enumerate(turns)
        ]
        sessions.append(
            Session(
                session_id=str(session_id),
                date=parse_timestamp(date),
                turns=parsed,
                is_evidence=str(session_id) in answer_sessions,
            )
        )
    sessions.sort(key=lambda session: session.date)

    category = (
        BenchCategory.ABSTENTION
        if is_abstention
        else LONGMEMEVAL_CATEGORY[record["question_type"]]
    )
    return Case(
        question_id=question_id,
        category=category,
        question=record["question"],
        answer=str(record.get("answer", "")),
        question_date=parse_timestamp(record["question_date"]),
        sessions=sessions,
        source="longmemeval",
        is_abstention=is_abstention,
        superseded_evidence=_superseded_evidence(sessions, category),
    )


def iter_cases(path: Path) -> Iterator[Case]:
    """Stream every case of a dataset file.

    Args:
        path: The dataset JSON.

    Yields:
        One case per record, in file order.
    """
    for record in _iter_records(path):
        yield to_case(record)


def load_cases(
    path: Path,
    *,
    categories: Iterable[BenchCategory] | None = None,
    n_per_category: int | None = None,
    seed: int = 0,
) -> list[Case]:
    """Load a reproducible sample of a dataset, balanced by category.

    Sampling is a reservoir per category, so a fixed `seed` gives the same cases
    every time without the whole file ever being held in memory at once.

    Args:
        path: The dataset JSON.
        categories: Categories to keep. `None` keeps every category the file
            can supply.
        n_per_category: How many cases to keep per category. `None` keeps all
            of them, which for `s` and `m` means a very large result.
        seed: Makes the sample reproducible.

    Returns:
        The sampled cases, ordered by category then question id.
    """
    wanted = set(categories) if categories is not None else None
    rng = random.Random(seed)
    reservoirs: dict[BenchCategory, list[Case]] = {}
    seen: dict[BenchCategory, int] = {}

    for case in iter_cases(path):
        if wanted is not None and case.category not in wanted:
            continue
        bucket = reservoirs.setdefault(case.category, [])
        count = seen[case.category] = seen.get(case.category, 0) + 1
        if n_per_category is None or len(bucket) < n_per_category:
            bucket.append(case)
            continue
        # Reservoir sampling: the nth candidate replaces a held one with
        # probability k/n, which keeps every case equally likely.
        index = rng.randrange(count)
        if index < n_per_category:
            bucket[index] = case

    sampled = [case for bucket in reservoirs.values() for case in bucket]
    sampled.sort(key=lambda case: (case.category.value, case.question_id))
    return sampled


def _superseded_evidence(sessions: list[Session], category: BenchCategory) -> list[str]:
    """Return the answer-bearing turns that the newest evidence retires.

    A knowledge-update case carries the same fact more than once: the earlier
    evidence session holds the value that changed, the later one holds the
    current value. Everything but the latest evidence session is therefore what
    must no longer be presented as true — which is exactly what the
    supersede-integrity judge needs, and it comes straight out of the dataset
    without an extraction step to distrust.
    """
    if category is not BenchCategory.KNOWLEDGE_UPDATE:
        return []
    evidence = [session for session in sessions if session.gold_turns]
    if len(evidence) < 2:
        return []
    return [turn.content for session in evidence[:-1] for turn in session.gold_turns]


def _iter_records(path: Path) -> Iterator[dict]:
    """Stream the objects of a JSON array without loading the whole file.

    The `m` variant is 2.7 GB; `json.load` on it needs many gigabytes of RAM to
    hand back records that are then mostly discarded by sampling. This walks the
    array with `raw_decode`, keeping at most one record plus a read buffer.
    """
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        while True:
            head = buffer[position:].lstrip()
            position = len(buffer) - len(head)
            if not head:
                chunk = handle.read(_CHUNK_CHARS)
                if not chunk:
                    return
                buffer = head + chunk
                position = 0
                continue
            if head[0] in "[,":
                position += 1
                continue
            if head[0] == "]":
                return
            try:
                record, end = decoder.raw_decode(buffer, position)
            except ValueError:
                chunk = handle.read(_CHUNK_CHARS)
                if not chunk:
                    return
                buffer += chunk
                continue
            yield record
            buffer = buffer[end:]
            position = 0
