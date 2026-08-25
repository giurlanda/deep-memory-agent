"""Running cases and persisting what they produced.

One case is one isolated world: its own memory tree under
`<root>/<question_id>/memory/`, its own agents, its own simulated clock. Nothing
is shared, which is what lets cases run concurrently — and what makes a single
case reproducible on its own, without replaying the whole experiment.

Two properties matter more than speed here. A case that blows up is recorded and
stepped over, because a run that dies at case forty has cost forty cases' worth
of tokens for nothing. And a case that already has a `result.json` is skipped, so
an interrupted run resumes instead of starting again.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dma_bench.agents import build_manager_agent, build_search_agent, open_store
from dma_bench.answer import answer_case
from dma_bench.categories import BenchCategory
from dma_bench.ingest import ingest_case
from dma_bench.judges.consolidation import judge_consolidation, snapshot_memory
from dma_bench.judges.qa import judge_answer, judge_supersede
from dma_bench.judges.retrieval import judge_retrieval
from dma_bench.schema import CaseResult, ExperimentResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from langchain_core.language_models import BaseChatModel

    from dma_bench.schema import Case, RunConfig

__all__ = [
    "Runtime",
    "case_directory",
    "estimate_run",
    "iter_pending",
    "load_case_results",
    "load_experiment",
    "run_case",
    "run_experiment",
]


@dataclass(frozen=True, slots=True)
class Runtime:
    """How to build the models a run needs.

    Factories rather than instances: each case builds its own clients, so a
    stateful or non-thread-safe client cannot leak state between cases running
    side by side.

    Attributes:
        agent_model: Builds the model under test.
        judge_model: Builds the grading model.
    """

    agent_model: Callable[[], BaseChatModel]
    judge_model: Callable[[], BaseChatModel]


def case_directory(config: RunConfig, case: Case) -> Path:
    """Return the directory a case owns.

    Args:
        config: The run configuration.
        case: The case.

    Returns:
        `<experiment_root>/<question_id>/`, created if missing.
    """
    path = Path(config.experiment_root).expanduser() / case.question_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def estimate_run(cases: Sequence[Case], config: RunConfig) -> dict:
    """Estimate what a run will cost before it is paid for.

    Ingestion dominates: one agent invocation per session, each of which is
    several model calls once tool use is counted. Answering and grading are a
    handful of calls per case. The multipliers are deliberately rough — the
    number worth looking at is the order of magnitude.

    Args:
        cases: The cases about to be run.
        config: The run configuration.

    Returns:
        Session, turn, character and invocation counts.
    """
    sessions = sum(len(case.sessions) for case in cases)
    periodic = (
        sessions // max(config.consolidate_every_n, 1)
        if "periodic" in config.consolidation_mode
        else 0
    )
    final = len(cases) if "final" in config.consolidation_mode else 0
    judges = sum(
        3 + (case.category is BenchCategory.KNOWLEDGE_UPDATE) for case in cases
    )
    return {
        "cases": len(cases),
        "sessions": sessions,
        "turns": sum(case.turn_count for case in cases),
        "history_chars": sum(case.char_count for case in cases),
        "ingestion_invocations": sessions,
        "consolidation_invocations": periodic + final,
        "answer_invocations": len(cases),
        "judge_invocations": judges,
        "estimated_model_calls_low": sessions * 2
        + periodic
        + final
        + len(cases) * 2
        + judges,
        "estimated_model_calls_high": sessions * 5
        + periodic
        + final
        + len(cases) * 6
        + judges,
    }


def run_case(case: Case, config: RunConfig, runtime: Runtime) -> CaseResult:
    """Run one case end to end and write its result.

    Args:
        case: The case to run.
        config: The run configuration.
        runtime: How to build the models.

    Returns:
        The case result, also written to `<case_dir>/result.json`.
    """
    directory = case_directory(config, case)
    result_path = directory / "result.json"
    if config.resume and result_path.exists():
        try:
            return CaseResult.model_validate_json(result_path.read_text())
        except (OSError, ValueError):
            pass  # a truncated result from a killed run is worth redoing

    memory_dir = directory / "memory"
    result = CaseResult(
        question_id=case.question_id,
        category=case.category,
        source=case.source,
        question=case.question,
        gold_answer=case.answer,
        memory_dir=str(memory_dir),
        consolidation_mode=config.consolidation_mode,
    )

    try:
        agent_model = runtime.agent_model()
        judge_model = runtime.judge_model()

        manager = build_manager_agent(
            agent_model,
            memory_dir,
            allow_consolidation=config.consolidation_mode != "none",
        )
        result.ingestion = ingest_case(
            manager,
            case,
            memory_dir=memory_dir,
            model=agent_model,
            consolidation_mode=config.consolidation_mode,
            consolidate_every_n=config.consolidate_every_n,
            recursion_limit=config.recursion_limit,
        )

        store = open_store(memory_dir)
        result.memory_snapshot = snapshot_memory(store)
        result.consolidation = judge_consolidation(judge_model, case, store)

        result.answer = answer_case(
            build_search_agent(agent_model, memory_dir),
            case,
            chain_of_note=config.chain_of_note,
            recursion_limit=config.recursion_limit,
        )
        result.retrieval = judge_retrieval(
            judge_model,
            case,
            result.answer.trace,
            recall_threshold=config.recall_threshold,
        )
        result.qa = judge_answer(judge_model, case, result.answer.answer)
        if case.category is BenchCategory.KNOWLEDGE_UPDATE and case.superseded_evidence:
            result.supersede_integrity = judge_supersede(
                judge_model, case, result.answer.answer
            )
    except Exception as exc:
        result.error = repr(exc)

    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
    )
    return result


def run_experiment(
    cases: Sequence[Case],
    config: RunConfig,
    runtime: Runtime,
    *,
    on_result: Callable[[CaseResult], None] | None = None,
) -> ExperimentResult:
    """Run every case and write the experiment result.

    Args:
        cases: The cases to run.
        config: The run configuration.
        runtime: How to build the models.
        on_result: Called as each case finishes, for progress reporting.

    Returns:
        The experiment result, also written to `<experiment_root>/result.json`.
    """
    from dma_bench.metrics import summarise

    root = Path(config.experiment_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    config = config.model_copy(update={"started_at": datetime.now(tz=UTC)})

    results: list[CaseResult] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(config.max_workers, 1)) as pool:
        for result in pool.map(lambda case: run_case(case, config, runtime), cases):
            results.append(result)
            if on_result is not None:
                on_result(result)

    experiment = ExperimentResult(
        config=config,
        results=results,
        summary=summarise(results),
        finished_at=datetime.now(tz=UTC),
    )
    experiment.summary["duration_s"] = round(time.monotonic() - started, 1)
    (root / "result.json").write_text(
        json.dumps(experiment.model_dump(mode="json"), indent=2, ensure_ascii=False)
    )
    return experiment


def load_experiment(root: Path | str) -> ExperimentResult:
    """Reload a finished experiment from disk.

    Args:
        root: The experiment root directory.

    Returns:
        The experiment result as it was written.
    """
    path = Path(root).expanduser() / "result.json"
    return ExperimentResult.model_validate_json(path.read_text())


def load_case_results(root: Path | str) -> list[CaseResult]:
    """Reload every per-case result under an experiment root.

    Useful when a run was interrupted before `result.json` was written: the
    per-case files are complete on their own.

    Args:
        root: The experiment root directory.

    Returns:
        The case results found, sorted by question id.
    """
    results: list[CaseResult] = []
    for path in sorted(Path(root).expanduser().glob("*/result.json")):
        try:
            results.append(CaseResult.model_validate_json(path.read_text()))
        except (OSError, ValueError):
            continue
    return results


def iter_pending(cases: Iterable[Case], config: RunConfig) -> list[Case]:
    """Return the cases a resumed run still has to do.

    Args:
        cases: The full case list.
        config: The run configuration.

    Returns:
        Cases with no result on disk, or all of them when `resume` is off.
    """
    root = Path(config.experiment_root).expanduser()
    if not config.resume:
        return list(cases)
    return [
        case for case in cases if not (root / case.question_id / "result.json").exists()
    ]
