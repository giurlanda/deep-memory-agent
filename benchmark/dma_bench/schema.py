"""Data model shared by the datasets, the runner and the report.

Everything a run produces is a pydantic model so that `<root>/<question_id>/
result.json` and `<root>/result.json` are exact, reloadable serialisations of
what happened — a report is always rebuildable from disk without re-running the
agents, which is what makes a benchmark worth keeping.

`Case` is deliberately a superset of a LongMemEval record. The extra fields
(`superseded_evidence`, `expected_procedure`, `past_error`) are what the
categories LongMemEval does not have need in order to be graded, and they stay
empty for the cases that come from its datasets.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from dma_bench.categories import BenchCategory

__all__ = [
    "AnswerArm",
    "AnswerRecord",
    "Case",
    "CaseResult",
    "ConsolidationRecord",
    "ExperimentResult",
    "IngestionRecord",
    "MemorySnapshot",
    "QaVerdict",
    "RetrievalVerdict",
    "RunConfig",
    "SemanticArm",
    "SemanticIndexRecord",
    "Session",
    "StageVerdict",
    "TraceMessage",
    "Turn",
]

ConsolidationMode = Literal["none", "periodic", "final", "periodic+final"]
"""When consolidation runs during ingestion.

`none` is the cold-pipeline arm: the manager agent is built without
`memory_consolidate` at all, so the arm cannot be polluted by the agent
deciding to consolidate on its own.
"""


class Turn(BaseModel):
    """One message of a session.

    Attributes:
        turn_id: Stable identifier, `"<session_id>#<index>"`. It is the unit the
            retrieval judge counts recall over.
        role: Who spoke.
        content: What was said.
        has_answer: Whether this message carries information the question needs.
            This is LongMemEval's gold annotation and the benchmark's ground
            truth for retrieval.
    """

    turn_id: str
    role: Literal["user", "assistant"]
    content: str
    has_answer: bool = False


class Session(BaseModel):
    """One past conversation, replayed into memory as a single episode.

    Attributes:
        session_id: Identifier, unique within a case.
        date: When the conversation happened, timezone-aware. It is the instant
            the memory writes are stamped with.
        turns: The messages, in order.
        is_evidence: Whether the session carries any answer-bearing turn.
    """

    session_id: str
    date: datetime
    turns: list[Turn]
    is_evidence: bool = False

    @property
    def gold_turns(self) -> list[Turn]:
        """The answer-bearing turns of this session."""
        return [turn for turn in self.turns if turn.has_answer]


class Case(BaseModel):
    """One benchmark question with the history it has to be answered from.

    Attributes:
        question_id: Identifier; also the directory the case's memory and result
            are written under.
        category: What the question asks the memory system to do.
        question: The question put to the search agent.
        answer: The reference answer, or the explanation of why the question is
            unanswerable when `is_abstention`.
        question_date: The instant the question is asked, after every session.
        sessions: The history, in chronological order.
        source: Which dataset the case came from.
        is_abstention: Whether the correct behaviour is to decline.
        superseded_evidence: The answer-bearing turns of the *earlier* evidence
            sessions, verbatim. For a knowledge-update case these hold the value
            that must no longer be presented as current.
        expected_procedure: The procedure the scenario should trigger.
        past_error: A mistake made earlier, for non-repetition cases.
        past_correction: How that mistake was corrected.
    """

    question_id: str
    category: BenchCategory
    question: str
    answer: str
    question_date: datetime
    sessions: list[Session]
    source: str = "longmemeval"
    is_abstention: bool = False
    superseded_evidence: list[str] = Field(default_factory=list)
    expected_procedure: str | None = None
    past_error: str | None = None
    past_correction: str | None = None

    @property
    def gold_turns(self) -> list[Turn]:
        """Every answer-bearing turn of the case, in chronological order."""
        return [turn for session in self.sessions for turn in session.gold_turns]

    @property
    def turn_count(self) -> int:
        """Total number of messages that will be replayed."""
        return sum(len(session.turns) for session in self.sessions)

    @property
    def char_count(self) -> int:
        """Total size of the history, used for the dry-run cost estimate."""
        return sum(
            len(turn.content) for session in self.sessions for turn in session.turns
        )


class RunConfig(BaseModel):
    """Everything that defines a run, saved next to its results.

    Attributes:
        experiment_root: Directory holding `<question_id>/` and `result.json`.
        dataset: Which corpus the cases come from.
        scale: `small` for the cold pipeline, `large` for the loaded one.
        n_per_category: How many cases to sample per category.
        consolidation_mode: When consolidation runs.
        consolidate_every_n: Sessions between periodic consolidations.
        recall_threshold: Fraction of gold turns that has to be surfaced for
            retrieval to count as correct. `1.0` is LongMemEval's strict
            reading and the default.
        chain_of_note: Whether the answering prompt asks for the extract-then-
            reason form. Worth up to ten points in the paper, even at perfect
            retrieval.
        agent_model: Identifier of the model under test, for the record.
        judge_model: Identifier of the grading model, for the record.
        max_workers: Cases graded concurrently. Cases share nothing — separate
            memory trees, separate agents, a context-local write clock.
        recursion_limit: Cap on agent steps per invocation.
        seed: Makes the per-category sample reproducible.
        resume: Reuse what a case already has on disk instead of paying for it
            again. Ingestion is what it protects: a case whose history has been
            replayed keeps its memory tree, and only the answering arms it is
            still missing are run.
        semantic_search: Whether to run the question a second time through an
            agent that also has `semantic_search`. The lexical arm runs either
            way — the point of the second arm is the comparison, and a run with
            only the semantic number in it says nothing.
        semantic_search_k: Entries a semantic search returns by default.
        embedding_model: Identifier of the embedding model, for the record.
        started_at: When the run began.
    """

    experiment_root: str
    dataset: str = "longmemeval"
    scale: Literal["small", "medium", "large"] = "small"
    n_per_category: int = 3
    consolidation_mode: ConsolidationMode = "none"
    consolidate_every_n: int = 10
    recall_threshold: float = 1.0
    chain_of_note: bool = True
    agent_model: str = ""
    judge_model: str = ""
    max_workers: int = 4
    recursion_limit: int = 60
    seed: int = 0
    resume: bool = True
    semantic_search: bool = False
    semantic_search_k: int = 5
    embedding_model: str = ""
    started_at: datetime | None = None


class StageVerdict(BaseModel):
    """A yes/no judgement with the reasoning that produced it.

    Attributes:
        correct: The judgement.
        reasoning: Why, in the judge's words. Kept so a surprising score can be
            audited instead of trusted.
        applicable: `False` when the stage cannot be scored for this case — an
            abstention case has no gold turn to retrieve. Such cases are left
            out of the averages rather than counted as failures.
        error: Set when the judge itself failed.
    """

    correct: bool = False
    reasoning: str = ""
    applicable: bool = True
    error: str | None = None


class QaVerdict(StageVerdict):
    """A QA judgement, tagged with the prompt that produced it.

    Attributes:
        judge: Which per-category prompt was used.
    """

    judge: str = ""


class RetrievalVerdict(StageVerdict):
    """What the search actually surfaced, against what it should have.

    Attributes:
        recall: Fraction of gold turns surfaced, or `None` when there are none.
        gold_turn_ids: The turns that carry the answer.
        retrieved_turn_ids: The ones the judge found in the agent's search
            trace.
        superseded_leaked: Whether a retired value was surfaced as if it were
            still current. Mentioning it labelled as history does not count.
    """

    recall: float | None = None
    gold_turn_ids: list[str] = Field(default_factory=list)
    retrieved_turn_ids: list[str] = Field(default_factory=list)
    superseded_leaked: bool = False


class ConsolidationRecord(BaseModel):
    """One consolidation pass over the memory tree.

    Attributes:
        after_session: Index of the session it ran after, `-1` for the final
            pass.
        episodes_considered: How many episodes were read.
        entries_written: How many durable entries were produced.
        rationale: The model's explanation for its choices.
        error: Set when the pass failed.
    """

    after_session: int
    episodes_considered: int = 0
    entries_written: int = 0
    rationale: str = ""
    error: str | None = None


class IngestionRecord(BaseModel):
    """What happened while the history was replayed into memory.

    Attributes:
        sessions: How many sessions were ingested.
        failed_sessions: Session ids that raised and were skipped.
        write_calls: Number of `memory_write` / `memory_update` calls made.
        unsolicited_consolidations: Times the agent reached for
            `memory_consolidate` on its own. Non-zero here means the
            consolidation ablation is contaminated for this case.
        consolidations: The passes the harness drove.
        duration_s: Wall-clock seconds.
    """

    sessions: int = 0
    failed_sessions: list[str] = Field(default_factory=list)
    write_calls: int = 0
    unsolicited_consolidations: int = 0
    consolidations: list[ConsolidationRecord] = Field(default_factory=list)
    duration_s: float = 0.0


class MemorySnapshot(BaseModel):
    """The state of the memory tree once ingestion is done.

    Attributes:
        entries_by_category: Active entry count per category.
        active_entries: Total active entries.
        superseded_entries: Entries a newer one replaced.
        files: Paths of the memory files that exist.
        total_chars: Size of the tree, in characters.
    """

    entries_by_category: dict[str, int] = Field(default_factory=dict)
    active_entries: int = 0
    superseded_entries: int = 0
    files: list[str] = Field(default_factory=list)
    total_chars: int = 0


class TraceMessage(BaseModel):
    """One step of the search agent's reasoning, as the judge sees it.

    Attributes:
        kind: `ai` for the model's own turns, `tool` for what came back.
        name: Tool name, when the step is a tool result.
        content: The text of the step.
        tool_calls: Tools the model asked for, with their arguments.
    """

    kind: Literal["ai", "tool"]
    name: str = ""
    content: str = ""
    tool_calls: list[dict] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    """The search agent's reply and how it got there.

    Attributes:
        answer: The final reply, which is what the QA judge grades.
        trace: The AI and tool messages, which is what the retrieval judge
            reads.
        tool_calls: Number of tool calls made.
        duration_s: Wall-clock seconds.
        error: Set when answering failed.
    """

    answer: str = ""
    trace: list[TraceMessage] = Field(default_factory=list)
    tool_calls: int = 0
    duration_s: float = 0.0
    error: str | None = None

    @property
    def attempted(self) -> bool:
        """Whether this record is the outcome of a real answering attempt.

        A record straight out of `default_factory` is empty and has no error,
        which is indistinguishable from a run that produced nothing — except
        that a run always leaves a trace behind, or an error, or both. That is
        what a resumed case checks to decide whether the arm still owes work,
        and it reads results written before this field existed.
        """
        return bool(self.error or self.trace or self.answer)


class AnswerArm(BaseModel):
    """One pass at the question, with the verdicts over it.

    An arm is a way of reading memory, not a different memory: both arms answer
    the same case out of the same tree, built once. Only the tools the answering
    agent holds differ, which is what makes the two numbers comparable.

    Attributes:
        answer: The reply and its trace.
        retrieval: Did the search surface the answer-bearing turns?
        qa: Is the final answer correct?
        supersede_integrity: The stricter second reading, on knowledge-update
            cases only.
    """

    answer: AnswerRecord = Field(default_factory=AnswerRecord)
    retrieval: RetrievalVerdict = Field(default_factory=RetrievalVerdict)
    qa: QaVerdict = Field(default_factory=QaVerdict)
    supersede_integrity: QaVerdict | None = None


class SemanticIndexRecord(BaseModel):
    """One pass of the embedding index over a case's memory tree.

    Built once per case, after the last session has been ingested and before the
    question is asked. Indexing as memory is written would measure something
    else: an index that trails the files by a session is the shipped behaviour,
    but it would make retrieval depend on when the last write landed rather than
    on what the tree holds.

    Mirrors `deep_memory_agent.IngestReport`, so what the index cost and what it
    covered survive into `result.json` alongside everything else.

    Attributes:
        added: Entries the index had never seen.
        updated: Entries whose content or frontmatter changed.
        deleted: Entries that were indexed and are no longer in the tree.
        unchanged: Entries skipped because nothing about them moved.
        chunks: Chunks written to the vector store.
        deleted_chunks: Chunks removed from it.
        entry_errors: Per-entry failures the ingest survived.
        duration_s: Wall-clock seconds.
        error: Set when indexing failed outright, in which case the semantic arm
            is not run: scoring a search agent against an empty index would
            measure the harness, not the agent.
    """

    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    chunks: int = 0
    deleted_chunks: int = 0
    entry_errors: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None


class SemanticArm(AnswerArm):
    """The answering arm that also holds `semantic_search`.

    Attributes:
        index: What building the embedding index cost and covered.
    """

    index: SemanticIndexRecord = Field(default_factory=SemanticIndexRecord)


class CaseResult(BaseModel):
    """Everything one case produced, written to `<root>/<question_id>/result.json`.

    Attributes:
        question_id: The case.
        category: Its category.
        source: Which corpus it came from.
        question: The question asked.
        gold_answer: The reference answer.
        memory_dir: Where this case's memory tree lives.
        consolidation_mode: Which arm this result belongs to.
        ingestion: What the replay did.
        memory_snapshot: The tree it left behind.
        answer: The reply and its trace.
        consolidation: Stage one — is the fact in memory at all, and right?
        retrieval: Stage two — did the search surface it?
        qa: Stage three — is the final answer correct?
        supersede_integrity: The stricter second reading, on knowledge-update
            cases only.
        semantic: The second arm — the same question asked again of an agent
            that also has `semantic_search`, over the same memory. `None` when
            the run did not ask for it, which is what tells a resumed run the
            arm is still owed rather than merely empty.
        error: Set when the case failed outright.
    """

    question_id: str
    category: BenchCategory
    source: str = "longmemeval"
    question: str = ""
    gold_answer: str = ""
    memory_dir: str = ""
    consolidation_mode: ConsolidationMode = "none"
    ingestion: IngestionRecord = Field(default_factory=IngestionRecord)
    memory_snapshot: MemorySnapshot = Field(default_factory=MemorySnapshot)
    answer: AnswerRecord = Field(default_factory=AnswerRecord)
    consolidation: StageVerdict = Field(default_factory=StageVerdict)
    retrieval: RetrievalVerdict = Field(default_factory=RetrievalVerdict)
    qa: QaVerdict = Field(default_factory=QaVerdict)
    supersede_integrity: QaVerdict | None = None
    semantic: SemanticArm | None = None
    error: str | None = None


class ExperimentResult(BaseModel):
    """A whole run, written to `<root>/result.json`.

    Attributes:
        config: What was run.
        results: Every case result, in completion order.
        summary: The aggregated metrics, as built by `dma_bench.metrics`. A run
            that compared arms carries the semantic arm's own summary, of the
            same shape, under its `semantic` key.
        finished_at: When the run ended.
    """

    config: RunConfig
    results: list[CaseResult] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
    finished_at: datetime | None = None
