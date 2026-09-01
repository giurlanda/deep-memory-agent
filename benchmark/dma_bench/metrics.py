"""Aggregating results, and locating where a failure came from.

The headline number is per-category QA accuracy, as in the paper. The number
worth acting on is the three-stage decomposition.

LongMemEval decomposes failures across two stages because it has two, giving the
four quadrants of its Figure 14. A system that curates its own memory has three,
so this decomposes across eight cells — and each cell points at a different file
to go and fix:

| consolidation | retrieval | answer | what to look at |
| --- | --- | --- | --- |
| ✗ | any | ✗ | the consolidation prompt: the fact never made it into memory |
| ✓ | ✗ | ✗ | indexing and search: it is on file and was not found |
| ✓ | ✓ | ✗ | the reading strategy: found and then misused |
| ✓ | ✓ | ✓ | working as intended |

The cells where the answer is right despite an earlier stage failing are the ones
to be suspicious of: they usually mean the model knew the answer without needing
memory, which is exactly the shortcut a memory benchmark has to detect.

`supersede-integrity` is folded in as its own category here even though it has no
cases of its own: it is the knowledge-update cases re-read under the strict
prompt, and it belongs next to them in the table.

A run that compared arms gets the same summary twice. The semantic arm is not a
separate metric bolted on the side: it is the same eight cells and the same
per-category table computed over the second answer, nested under `semantic`, so
the two are read with the identical machinery and the difference between them is
the whole point.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from dma_bench.categories import BenchCategory

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dma_bench.schema import CaseResult

__all__ = ["CELL_DIAGNOSIS", "cell_label", "decompose", "semantic_view", "summarise"]

CELL_DIAGNOSIS: dict[str, str] = {
    "C✓ R✓ G✓": "working as intended",
    "C✓ R✓ G✗": "reading strategy: the fact was in view and the answer misused it",
    "C✓ R✗ G✓": "answered without retrieving: check for parametric knowledge",
    "C✓ R✗ G✗": "indexing and search: the fact is on file but was not found",
    "C✗ R✓ G✓": "judge disagreement: memory looked wrong but the answer held",
    "C✗ R✓ G✗": "consolidation: what was written does not support the answer",
    "C✗ R✗ G✓": "answered without memory: check for parametric knowledge",
    "C✗ R✗ G✗": "consolidation: the fact never made it into memory",
}
"""What each cell of the three-way decomposition points at."""


def cell_label(*, consolidation: bool, retrieval: bool, answer: bool) -> str:
    """Return the label of a decomposition cell.

    Args:
        consolidation: Whether memory held the right information.
        retrieval: Whether the search surfaced it.
        answer: Whether the final answer was correct.

    Returns:
        A label such as `"C✓ R✗ G✗"`.
    """
    mark = "✓✗"
    return f"C{mark[not consolidation]} R{mark[not retrieval]} G{mark[not answer]}"


def decompose(results: Sequence[CaseResult]) -> dict:
    """Break failures down by which stage produced them.

    Cases whose retrieval cannot be scored — abstention, which has no gold
    message to find — are counted separately rather than being forced into a
    cell they do not belong in.

    Args:
        results: The case results.

    Returns:
        Per-cell counts with their diagnosis, plus what was excluded.
    """
    counts: Counter[str] = Counter()
    excluded = 0
    for result in results:
        if result.error or not result.retrieval.applicable:
            excluded += 1
            continue
        counts[
            cell_label(
                consolidation=result.consolidation.correct,
                retrieval=result.retrieval.correct,
                answer=result.qa.correct,
            )
        ] += 1

    total = sum(counts.values())
    return {
        "scored": total,
        "excluded": excluded,
        "cells": {
            label: {
                "count": counts.get(label, 0),
                "share": round(counts.get(label, 0) / total, 4) if total else 0.0,
                "diagnosis": diagnosis,
            }
            for label, diagnosis in CELL_DIAGNOSIS.items()
        },
    }


def semantic_view(results: Sequence[CaseResult]) -> list[CaseResult]:
    """Re-read the cases through their semantic arm.

    The arm answered the same question out of the same memory, so everything
    upstream of the answer — ingestion, the tree, the stage-one verdict — is
    shared and copied across unchanged. Promoting the arm's reply and verdicts
    into the fields the rest of this module reads is what lets one summariser,
    one decomposition and one set of charts serve both arms.

    Args:
        results: The case results.

    Returns:
        One result per case that has a semantic arm, in the same order. Cases
        without one are dropped rather than falling back to the lexical answer,
        which would quietly credit the index with numbers it never earned.
    """
    view = []
    for result in results:
        arm = result.semantic
        if arm is None:
            continue
        view.append(
            result.model_copy(
                update={
                    "answer": arm.answer,
                    "retrieval": arm.retrieval,
                    "qa": arm.qa,
                    "supersede_integrity": arm.supersede_integrity,
                }
            )
        )
    return view


def summarise(results: Sequence[CaseResult], *, arms: bool = True) -> dict:
    """Aggregate a run into the numbers the report shows.

    Args:
        results: The case results.
        arms: Whether to nest the semantic arm's own summary under `semantic`.
            `False` is how that nested summary is built, and stops the recursion.

    Returns:
        Overall metrics, a per-category breakdown, the three-stage
        decomposition, and the ingestion cost — plus, when the run compared
        arms, the same shape again under `semantic`.
    """
    scored = [result for result in results if not result.error]
    buckets: dict[str, list[CaseResult]] = {}
    for result in scored:
        buckets.setdefault(result.category.value, []).append(result)

    supersede = [result for result in scored if result.supersede_integrity is not None]
    by_category = {name: _category_row(rows) for name, rows in sorted(buckets.items())}
    if supersede:
        by_category[BenchCategory.SUPERSEDE_INTEGRITY.value] = _supersede_row(supersede)

    summary: dict = {
        "cases": len(results),
        "failed_cases": len(results) - len(scored),
        "qa_accuracy": _mean(result.qa.correct for result in scored),
        "retrieval_correct_rate": _mean(
            result.retrieval.correct for result in scored if result.retrieval.applicable
        ),
        "retrieval_recall": _mean(
            result.retrieval.recall
            for result in scored
            if result.retrieval.recall is not None
        ),
        "consolidation_correct_rate": _mean(
            result.consolidation.correct for result in scored
        ),
        "by_category": by_category,
        "decomposition": decompose(scored),
        "cost": _cost(scored),
    }
    semantic = semantic_view(results) if arms else []
    if semantic:
        summary["semantic"] = summarise(semantic, arms=False)
    return summary


def _category_row(results: list[CaseResult]) -> dict:
    """Aggregate one category."""
    return {
        "n": len(results),
        "qa_accuracy": _mean(result.qa.correct for result in results),
        "retrieval_correct_rate": _mean(
            result.retrieval.correct
            for result in results
            if result.retrieval.applicable
        ),
        "retrieval_recall": _mean(
            result.retrieval.recall
            for result in results
            if result.retrieval.recall is not None
        ),
        "consolidation_correct_rate": _mean(
            result.consolidation.correct for result in results
        ),
    }


def _supersede_row(results: list[CaseResult]) -> dict:
    """Aggregate the strict second reading of the knowledge-update cases."""
    row = {
        "n": len(results),
        "qa_accuracy": _mean(
            result.supersede_integrity.correct  # type: ignore[union-attr]
            for result in results
        ),
        "retrieval_correct_rate": _mean(
            result.retrieval.correct
            for result in results
            if result.retrieval.applicable
        ),
        "retrieval_recall": _mean(
            result.retrieval.recall
            for result in results
            if result.retrieval.recall is not None
        ),
        "consolidation_correct_rate": _mean(
            result.consolidation.correct for result in results
        ),
    }
    row["superseded_leak_rate"] = _mean(
        result.retrieval.superseded_leaked for result in results
    )
    return row


def _cost(results: list[CaseResult]) -> dict:
    """Summarise what the run spent and what it left behind.

    The index keys appear only for a run that built one, so a lexical run's cost
    table is unchanged and a semantic one does not report seven zeros.
    """
    indexed = [result.semantic.index for result in results if result.semantic]
    cost = {
        "sessions_ingested": sum(result.ingestion.sessions for result in results),
        "failed_sessions": sum(
            len(result.ingestion.failed_sessions) for result in results
        ),
        "write_calls": sum(result.ingestion.write_calls for result in results),
        "unsolicited_consolidations": sum(
            result.ingestion.unsolicited_consolidations for result in results
        ),
        "consolidation_passes": sum(
            len(result.ingestion.consolidations) for result in results
        ),
        "active_entries": sum(
            result.memory_snapshot.active_entries for result in results
        ),
        "superseded_entries": sum(
            result.memory_snapshot.superseded_entries for result in results
        ),
        "ingestion_seconds": round(
            sum(result.ingestion.duration_s for result in results), 1
        ),
        "answer_seconds": round(sum(result.answer.duration_s for result in results), 1),
    }
    if indexed:
        cost["indexed_entries"] = sum(
            record.added + record.updated for record in indexed
        )
        cost["indexed_chunks"] = sum(record.chunks for record in indexed)
        cost["failed_indexes"] = sum(record.error is not None for record in indexed)
        cost["indexing_seconds"] = round(
            sum(record.duration_s for record in indexed), 1
        )
    return cost


def _mean(values: Iterable[float | bool]) -> float | None:
    """Return the mean of an iterable, or `None` when it is empty."""
    collected = [float(value) for value in values]
    if not collected:
        return None
    return round(sum(collected) / len(collected), 4)
