"""Tables and charts for a finished run.

Everything here reads a summary dict, never the agents, so a report can be
rebuilt from `result.json` long after the run — which is the point of writing the
results out in full.

The charts are deliberately few. Per-category accuracy is what gets compared
across runs; the decomposition is what says which prompt to go and change; the
ablation chart is the one that answers whether consolidation earned its cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from dma_bench.schema import CaseResult

__all__ = [
    "category_table",
    "cost_table",
    "decomposition_table",
    "plot_ablation",
    "plot_accuracy_by_category",
    "plot_decomposition",
    "plot_recall_vs_accuracy",
]

_METRICS = (
    ("qa_accuracy", "QA accuracy"),
    ("retrieval_correct_rate", "Retrieval correct"),
    ("retrieval_recall", "Retrieval recall"),
    ("consolidation_correct_rate", "Consolidation correct"),
)


def category_table(summary: dict) -> pd.DataFrame:
    """Build the per-category metrics table.

    Args:
        summary: A run summary, as produced by `dma_bench.metrics.summarise`.

    Returns:
        One row per category, plus an `ALL` row.
    """
    import pandas as pd

    rows = []
    for name, values in summary.get("by_category", {}).items():
        row = {"category": name, "n": values.get("n", 0)}
        row.update({label: values.get(key) for key, label in _METRICS})
        if "superseded_leak_rate" in values:
            row["Superseded leak"] = values["superseded_leak_rate"]
        rows.append(row)

    overall = {"category": "ALL", "n": summary.get("cases", 0)}
    overall.update({label: summary.get(key) for key, label in _METRICS})
    rows.append(overall)
    return pd.DataFrame(rows).set_index("category")


def decomposition_table(summary: dict) -> pd.DataFrame:
    """Build the three-stage decomposition table.

    Args:
        summary: A run summary.

    Returns:
        One row per cell, with its count, share and diagnosis, busiest first.
    """
    import pandas as pd

    cells = summary.get("decomposition", {}).get("cells", {})
    frame = pd.DataFrame(
        [
            {
                "cell": label,
                "count": values["count"],
                "share": values["share"],
                "points at": values["diagnosis"],
            }
            for label, values in cells.items()
        ]
    )
    return frame.sort_values("count", ascending=False).set_index("cell")


def cost_table(summary: dict) -> pd.DataFrame:
    """Build the run-cost table.

    Args:
        summary: A run summary.

    Returns:
        A single-column frame of what the run spent.
    """
    import pandas as pd

    return pd.DataFrame.from_dict(
        summary.get("cost", {}), orient="index", columns=["value"]
    )


def plot_accuracy_by_category(summary: dict, *, title: str = "") -> Figure:
    """Plot QA accuracy, retrieval and consolidation side by side per category.

    Args:
        summary: A run summary.
        title: Optional chart title.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    categories = list(summary.get("by_category", {}))
    series = {
        label: [summary["by_category"][name].get(key) or 0.0 for name in categories]
        for key, label in _METRICS[:3]
    }

    positions = np.arange(len(categories))
    width = 0.26
    figure, axes = plt.subplots(figsize=(1.6 * max(len(categories), 4) + 2, 4.5))
    for offset, (label, values) in enumerate(series.items()):
        axes.bar(positions + (offset - 1) * width, values, width, label=label)

    axes.set_xticks(positions)
    axes.set_xticklabels(categories, rotation=30, ha="right")
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("score")
    axes.set_title(title or "Per-category results")
    axes.legend(loc="lower right", fontsize=8)
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_decomposition(summary: dict, *, title: str = "") -> Figure:
    """Plot the eight cells of the three-stage decomposition.

    Cells where the answer was right are drawn in one colour and cells where it
    was wrong in another, so the chart reads as "how much is broken, and where"
    rather than as eight unrelated bars.

    Args:
        summary: A run summary.
        title: Optional chart title.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt

    cells = summary.get("decomposition", {}).get("cells", {})
    labels = list(cells)
    counts = [cells[label]["count"] for label in labels]
    colours = ["#2a9d8f" if label.endswith("G✓") else "#e76f51" for label in labels]

    figure, axes = plt.subplots(figsize=(9, 4.5))
    bars = axes.barh(labels, counts, color=colours)
    for bar, label in zip(bars, labels, strict=True):
        if cells[label]["count"]:
            axes.text(
                bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                cells[label]["diagnosis"],
                va="center",
                fontsize=7.5,
                color="#444",
            )
    axes.invert_yaxis()
    axes.set_xlabel("cases")
    axes.set_xlim(0, max([*counts, 1]) * 1.9)
    axes.set_title(title or "Consolidation × retrieval × answer")
    axes.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_ablation(summaries: dict[str, dict], *, title: str = "") -> Figure:
    """Compare runs that differ only in when consolidation ran.

    Args:
        summaries: Run summaries keyed by the label to show, typically the
            consolidation mode.
        title: Optional chart title.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    categories = sorted(
        {
            name
            for summary in summaries.values()
            for name in summary.get("by_category", {})
        }
    )
    positions = np.arange(len(categories))
    width = 0.8 / max(len(summaries), 1)

    figure, axes = plt.subplots(figsize=(1.6 * max(len(categories), 4) + 2, 4.5))
    for offset, (label, summary) in enumerate(summaries.items()):
        values = [
            (summary.get("by_category", {}).get(name) or {}).get("qa_accuracy") or 0.0
            for name in categories
        ]
        axes.bar(positions + offset * width, values, width, label=label)

    axes.set_xticks(positions + width * (len(summaries) - 1) / 2)
    axes.set_xticklabels(categories, rotation=30, ha="right")
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("QA accuracy")
    axes.set_title(title or "Consolidation ablation")
    axes.legend(loc="lower right", fontsize=8)
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_recall_vs_accuracy(results: list[CaseResult], *, title: str = "") -> Figure:
    """Plot answer correctness against retrieval recall, case by case.

    The paper's finding is that correct retrieval is necessary about 90% of the
    time. Points in the top-left — right answer, no recall — are where this run
    is being answered from the model's own knowledge rather than from memory,
    and they are worth reading individually.

    Args:
        results: The case results.
        title: Optional chart title.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)
    scored = [
        result
        for result in results
        if not result.error and result.retrieval.recall is not None
    ]
    figure, axes = plt.subplots(figsize=(7, 4.5))
    outcomes = (
        (True, "#2a9d8f", "answer correct"),
        (False, "#e76f51", "answer wrong"),
    )
    for correct, colour, label in outcomes:
        points = [result for result in scored if result.qa.correct is correct]
        if not points:
            continue
        axes.scatter(
            [result.retrieval.recall for result in points],
            # Jitter, or every case lands on one of two lines and overlaps.
            rng.normal(1.0 if correct else 0.0, 0.04, len(points)),
            s=32,
            alpha=0.7,
            color=colour,
            label=label,
        )
    axes.set_xlabel("retrieval recall")
    axes.set_yticks([0, 1])
    axes.set_yticklabels(["wrong", "correct"])
    axes.set_xlim(-0.05, 1.05)
    axes.set_title(title or "Retrieval recall against answer correctness")
    axes.legend(loc="center left", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    return figure
