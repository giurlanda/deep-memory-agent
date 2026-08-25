"""Corpora the benchmark can be run against.

Two providers, one `Case` schema. `longmemeval` adapts the published datasets,
which is where the six transferred categories come from; `operational` loads the
generated corpus that supplies the two categories LongMemEval has no data for —
following a procedure, and not repeating a corrected mistake.

`load_dataset` is the one entry point the notebook needs: it picks the provider,
resolves the file for the requested scale, and samples the same cases every time
for a given seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dma_bench.datasets import longmemeval, operational

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dma_bench.categories import BenchCategory
    from dma_bench.schema import Case

__all__ = ["DATASETS", "load_dataset", "longmemeval", "operational"]

DATASETS = ("longmemeval", "operational")
"""The providers `load_dataset` understands."""


def load_dataset(
    dataset: str,
    root: Path,
    *,
    scale: str = "small",
    categories: Iterable[BenchCategory] | None = None,
    n_per_category: int | None = None,
    seed: int = 0,
) -> list[Case]:
    """Load a reproducible sample from one of the corpora.

    Args:
        dataset: `longmemeval` or `operational`.
        root: Directory holding that corpus' files.
        scale: `small` or `large`; `longmemeval` also accepts a variant name
            (`oracle`, `s`, `m`) directly.
        categories: Categories to keep. `None` keeps whatever the corpus has.
        n_per_category: How many cases to keep per category.
        seed: Makes the sample reproducible.

    Returns:
        The sampled cases.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if dataset == "longmemeval":
        return longmemeval.load_cases(
            longmemeval.dataset_path(root, scale),
            categories=categories,
            n_per_category=n_per_category,
            seed=seed,
        )
    if dataset == "operational":
        return operational.load_cases(
            operational.corpus_path(root, scale),
            categories=categories,
            n_per_category=n_per_category,
            seed=seed,
        )
    msg = f"unknown dataset {dataset!r}; expected one of {DATASETS}"
    raise ValueError(msg)
