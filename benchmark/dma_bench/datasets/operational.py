"""Loader for the generated operational corpus.

The corpus is written once by `dma_bench.generation.generator` and kept on disk,
so that a comparison run months later — after the consolidation prompt has been
changed, say — measures the same material. Regenerating per run would make every
number incomparable with the last.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING

from dma_bench.schema import Case

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dma_bench.categories import BenchCategory

__all__ = ["CORPUS_FILES", "corpus_path", "load_cases"]

CORPUS_FILES: dict[str, str] = {
    "small": "operational_small.json",
    "large": "operational_large.json",
}
"""Which generated file backs each scale."""


def corpus_path(root: Path, scale: str) -> Path:
    """Return the corpus file for a scale.

    Args:
        root: Directory holding the generated corpora.
        scale: `small` or `large`.

    Returns:
        Path to the JSON file.

    Raises:
        ValueError: If the scale is unknown.
    """
    if scale not in CORPUS_FILES:
        msg = f"unknown operational scale {scale!r}"
        raise ValueError(msg)
    return root / CORPUS_FILES[scale]


def load_cases(
    path: Path,
    *,
    categories: Iterable[BenchCategory] | None = None,
    n_per_category: int | None = None,
    seed: int = 0,
) -> list[Case]:
    """Load a reproducible sample of a generated corpus.

    Args:
        path: The corpus JSON.
        categories: Categories to keep. `None` keeps everything.
        n_per_category: How many cases to keep per category.
        seed: Makes the sample reproducible.

    Returns:
        The sampled cases, ordered by category then question id.

    Raises:
        FileNotFoundError: If the corpus has not been generated yet.
    """
    if not path.exists():
        msg = (
            f"{path} does not exist — generate it first with "
            f"`python -m dma_bench.generation.generator --config <scale> --out {path}`"
        )
        raise FileNotFoundError(msg)

    wanted = set(categories) if categories is not None else None
    rng = random.Random(seed)
    buckets: dict[str, list[Case]] = {}
    for record in json.loads(path.read_text()):
        case = Case.model_validate(record)
        if wanted is not None and case.category not in wanted:
            continue
        buckets.setdefault(case.category.value, []).append(case)

    sampled: list[Case] = []
    for name in sorted(buckets):
        bucket = sorted(buckets[name], key=lambda case: case.question_id)
        if n_per_category is not None and len(bucket) > n_per_category:
            bucket = sorted(
                rng.sample(bucket, n_per_category), key=lambda case: case.question_id
            )
        sampled.extend(bucket)
    return sampled
