"""Running cases and persisting what they produced.

One case is one isolated world: its own memory tree under
`<root>/<question_id>/memory/`, its own agents, its own semantic index, its own
simulated clock. Nothing is shared, which is what lets cases run concurrently —
and what makes a single case reproducible on its own, without replaying the
whole experiment.

Two properties matter more than speed here. A case that blows up is recorded and
stepped over, because a run that dies at case forty has cost forty cases' worth
of tokens for nothing. And nothing already on disk is paid for twice: resume is
about ingestion, which is where the tokens go — a case whose history has been
replayed keeps its memory tree, and only the answering arms it is still missing
are run. Turning `semantic_search` on over a finished experiment therefore costs
one index and one question per case, not the whole replay again.

A case runs its question twice when `semantic_search` is on: once through the
shipped search agent, once through the same agent holding `semantic_search` as
well, over the same tree. The lexical arm is not optional — a semantic number
with nothing beside it says nothing about whether the index helped.
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
from dma_bench.schema import (
    AnswerArm,
    CaseResult,
    ExperimentResult,
    SemanticArm,
    SemanticIndexRecord,
)
from dma_bench.semantic import index_case

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore
    from langgraph.graph.state import CompiledStateGraph

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
    """How to build the models and stores a run needs.

    Factories rather than instances: each case builds its own clients, so a
    stateful or non-thread-safe client cannot leak state between cases running
    side by side. `vector_store` is a factory for the same reason and one more —
    it takes the case, because two cases sharing an index would let one answer
    out of the other's memory, which is the one way this benchmark can silently
    stop measuring anything.

    Attributes:
        agent_model: Builds the model under test.
        judge_model: Builds the grading model.
        embeddings: Builds the embedding model for the semantic arm. `None`
            leaves the run lexical whatever `RunConfig.semantic_search` says.
        vector_store: Builds a case's own index, given the embeddings and the
            case.
    """

    agent_model: Callable[[], BaseChatModel]
    judge_model: Callable[[], BaseChatModel]
    embeddings: Callable[[], Embeddings] | None = None
    vector_store: Callable[[Embeddings, Case], VectorStore] | None = None

    @property
    def can_index(self) -> bool:
        """Whether this runtime can build a semantic index at all."""
        return self.embeddings is not None and self.vector_store is not None


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

    A run comparing arms asks and grades every question twice, so answering and
    judging double while ingestion — the expensive half — does not. The
    embedding calls the index costs are counted separately: they are a different
    model, usually a far cheaper one, and adding them to a chat-call total would
    misstate both.

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
    arms = 2 if config.semantic_search else 1
    judges = arms * sum(
        3 + (case.category is BenchCategory.KNOWLEDGE_UPDATE) for case in cases
    )
    answers = arms * len(cases)
    return {
        "cases": len(cases),
        "sessions": sessions,
        "turns": sum(case.turn_count for case in cases),
        "history_chars": sum(case.char_count for case in cases),
        "ingestion_invocations": sessions,
        "consolidation_invocations": periodic + final,
        "answer_invocations": answers,
        "judge_invocations": judges,
        "semantic_index_passes": len(cases) if config.semantic_search else 0,
        "estimated_model_calls_low": sessions * 2
        + periodic
        + final
        + answers * 2
        + judges,
        "estimated_model_calls_high": sessions * 5
        + periodic
        + final
        + answers * 6
        + judges,
    }


def _load_previous(result_path: Path) -> CaseResult | None:
    """Return the result already on disk, or `None` when there is none to trust.

    Args:
        result_path: Where the case writes its result.

    Returns:
        The parsed result, or `None` — a truncated file from a killed run is
        worth redoing rather than reading.
    """
    if not result_path.exists():
        return None
    try:
        return CaseResult.model_validate_json(result_path.read_text())
    except (OSError, ValueError):
        return None


def _ingestion_done(previous: CaseResult, case: Case, memory_dir: Path) -> bool:
    """Whether a previous result carries a finished replay of this history.

    Every session has to have been attempted — a run killed halfway leaves a
    record of the sessions it managed, and resuming from that would answer out
    of half a memory. A session that raised counts as attempted: `ingest_case`
    steps over it deliberately, and re-running the case would not bring it back.

    Args:
        previous: The result found on disk.
        case: The case it belongs to.
        memory_dir: Where the tree should be.

    Returns:
        Whether the replay can be reused instead of paid for again.
    """
    attempted = previous.ingestion.sessions + len(previous.ingestion.failed_sessions)
    return attempted == len(case.sessions) and memory_dir.exists()


def _judge_arm(
    agent: CompiledStateGraph,
    case: Case,
    config: RunConfig,
    judge_model: BaseChatModel,
) -> AnswerArm:
    """Ask the question through one agent and grade what came back.

    Args:
        agent: The search agent this arm is measuring.
        case: The case being answered.
        config: The run configuration.
        judge_model: The grading model.

    Returns:
        The reply, the retrieval verdict and the QA verdict — plus the strict
        second reading on a knowledge-update case.
    """
    answer = answer_case(
        agent,
        case,
        chain_of_note=config.chain_of_note,
        recursion_limit=config.recursion_limit,
    )
    arm = AnswerArm(
        answer=answer,
        retrieval=judge_retrieval(
            judge_model, case, answer.trace, recall_threshold=config.recall_threshold
        ),
        qa=judge_answer(judge_model, case, answer.answer),
    )
    if case.category is BenchCategory.KNOWLEDGE_UPDATE and case.superseded_evidence:
        arm.supersede_integrity = judge_supersede(judge_model, case, answer.answer)
    return arm


def _run_semantic_arm(
    case: Case,
    config: RunConfig,
    runtime: Runtime,
    memory_dir: Path,
    agent_model: BaseChatModel,
    judge_model: BaseChatModel,
) -> SemanticArm:
    """Index the finished tree, then answer the question through it.

    Args:
        case: The case being answered.
        config: The run configuration.
        runtime: Supplies the embeddings and this case's own store.
        memory_dir: The tree to index, already complete.
        agent_model: The model under test.
        judge_model: The grading model.

    Returns:
        The arm. When the index could not be built the arm carries the failure
        and nothing else: an agent searched against an empty index would score
        the harness rather than the index. A store that will not open — a vector
        database that is down, say — is recorded the same way rather than
        raised, so it costs this arm and not the lexical one beside it.
    """
    try:
        embeddings = runtime.embeddings()  # type: ignore[misc]
        vector_store = runtime.vector_store(embeddings, case)  # type: ignore[misc]
    except Exception as exc:
        return SemanticArm(index=SemanticIndexRecord(error=repr(exc)))

    index = index_case(memory_dir, embeddings, vector_store)
    if index.error is not None:
        return SemanticArm(index=index)

    agent = build_search_agent(
        agent_model,
        memory_dir,
        embeddings=embeddings,
        vector_store=vector_store,
        search_k=config.semantic_search_k,
    )
    arm = _judge_arm(agent, case, config, judge_model)
    return SemanticArm(index=index, **arm.model_dump())


def _restate(
    previous: CaseResult | None, case: Case, config: RunConfig, memory_dir: Path
) -> CaseResult:
    """Return the result this run will write, carrying over what is reusable.

    The case's own description is restated from the case rather than trusted
    from disk, so a corpus regenerated under the same ids cannot leave a result
    quoting a question that is no longer asked.

    Args:
        previous: The reusable result on disk, or `None` to start clean.
        case: The case being run.
        config: The run configuration.
        memory_dir: Where this case's tree lives.

    Returns:
        A result to fill in.
    """
    result = (
        previous.model_copy(deep=True)
        if previous is not None
        else CaseResult(question_id=case.question_id, category=case.category)
    )
    result.source = case.source
    result.question = case.question
    result.gold_answer = case.answer
    result.memory_dir = str(memory_dir)
    result.consolidation_mode = config.consolidation_mode
    return result


def run_case(case: Case, config: RunConfig, runtime: Runtime) -> CaseResult:
    """Run one case end to end and write its result.

    What actually runs depends on what is already on disk. A case with a
    finished replay keeps its memory tree, its snapshot and its consolidation
    verdict, and pays only for the answering arms it is missing — which is what
    makes turning `semantic_search` on over a finished experiment affordable.

    Args:
        case: The case to run.
        config: The run configuration.
        runtime: How to build the models and this case's index.

    Returns:
        The case result, also written to `<case_dir>/result.json`.
    """
    directory = case_directory(config, case)
    result_path = directory / "result.json"
    memory_dir = directory / "memory"

    previous = _load_previous(result_path) if config.resume else None
    reuse = previous is not None and _ingestion_done(previous, case, memory_dir)
    result = _restate(previous if reuse else None, case, config, memory_dir)

    # A run that died left whatever it had reached, and stage one is the piece a
    # resumed case cannot tell apart from a default, so an errored case re-takes
    # it. That is one judge call over a tree already on disk, not a replay.
    stage_pending = not reuse or result.error is not None
    lexical_pending = not result.answer.attempted
    semantic_pending = (
        config.semantic_search
        and runtime.can_index
        and (result.semantic is None or not result.semantic.answer.attempted)
    )
    if not (stage_pending or lexical_pending or semantic_pending):
        return result

    result.error = None
    try:
        agent_model = runtime.agent_model()
        judge_model = runtime.judge_model()

        if not reuse:
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

        if stage_pending:
            store = open_store(memory_dir)
            result.memory_snapshot = snapshot_memory(store)
            result.consolidation = judge_consolidation(judge_model, case, store)

        if lexical_pending:
            arm = _judge_arm(
                build_search_agent(agent_model, memory_dir),
                case,
                config,
                judge_model,
            )
            result.answer = arm.answer
            result.retrieval = arm.retrieval
            result.qa = arm.qa
            result.supersede_integrity = arm.supersede_integrity

        if semantic_pending:
            result.semantic = _run_semantic_arm(
                case, config, runtime, memory_dir, agent_model, judge_model
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
        runtime: How to build the models and the per-case indexes.
        on_result: Called as each case finishes, for progress reporting.

    Returns:
        The experiment result, also written to `<experiment_root>/result.json`.

    Raises:
        ValueError: If the run asks for semantic search and the runtime cannot
            build an index. Half a configuration is a mistake, not a request to
            run lexical-only, and every case would silently record one arm where
            the report expects two.
    """
    from dma_bench.metrics import summarise

    if config.semantic_search and not runtime.can_index:
        msg = (
            "config.semantic_search is on but the runtime has no index to "
            f"build; got embeddings={runtime.embeddings is not None}, "
            f"vector_store={runtime.vector_store is not None}"
        )
        raise ValueError(msg)

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

    A case counts as done only under the run it is about to take part in: one
    finished before `semantic_search` was turned on is missing an arm, and so is
    pending — for the price of an index and one question, not its replay.

    Args:
        cases: The full case list.
        config: The run configuration.

    Returns:
        Cases with work left, or all of them when `resume` is off.
    """
    root = Path(config.experiment_root).expanduser()
    if not config.resume:
        return list(cases)
    return [case for case in cases if not _case_done(case, config, root)]


def _case_done(case: Case, config: RunConfig, root: Path) -> bool:
    """Whether a case owes this run nothing.

    Args:
        case: The case.
        config: The run configuration.
        root: The experiment root.

    Returns:
        Whether every stage this run asks for is already on disk and unbroken.
    """
    directory = root / case.question_id
    previous = _load_previous(directory / "result.json")
    if previous is None or previous.error is not None:
        return False
    if not _ingestion_done(previous, case, directory / "memory"):
        return False
    if not previous.answer.attempted:
        return False
    return not config.semantic_search or (
        previous.semantic is not None and previous.semantic.answer.attempted
    )
